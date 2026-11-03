import asyncio
import os
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from services.api_errors import register_exception_handlers
from services.authentication import authenticated_subject
from services.cards.store import (
    CardRecord,
    CardStoreUnavailable,
    get_owned_card,
    list_user_cards,
    set_card_status,
)
from services.sandbox import require_sandbox_mode

app = FastAPI(title="Cards Service", version="0.1.0")
register_exception_handlers(app)


class CardResponse(BaseModel):
    card_id: str
    user_id: str
    name: str
    last_4: str
    status: Literal["active", "frozen"]


class CardFreezeRequest(BaseModel):
    otp_code: str | None = Field(default=None, pattern=r"^\d{6}$")


class CardFreezeResponse(BaseModel):
    card_id: str
    status: str
    card_name: str
    last_4: str
    timestamp: datetime
    reference_id: str
    message: str


def _card_response(card: CardRecord) -> CardResponse:
    return CardResponse(
        card_id=card.card_id,
        user_id=card.user_id,
        name=card.card_name,
        last_4=card.last_4,
        status=card.status,
    )


async def _owned_card_or_404(user_id: str, card_id: str) -> CardRecord:
    try:
        card = await asyncio.to_thread(get_owned_card, user_id, card_id)
    except CardStoreUnavailable:
        raise HTTPException(status_code=503, detail="Card storage is unavailable") from None
    if card is None:
        raise HTTPException(status_code=404, detail="Card not found")
    return card


async def _set_owned_card_status(
    user_id: str,
    card_id: str,
    status: Literal["active", "frozen"],
) -> CardRecord:
    try:
        card = await asyncio.to_thread(set_card_status, user_id, card_id, status)
    except CardStoreUnavailable:
        raise HTTPException(status_code=503, detail="Card storage is unavailable") from None
    if card is None:
        raise HTTPException(status_code=404, detail="Card not found")
    return card


def _freeze_response(card: CardRecord) -> CardFreezeResponse:
    timestamp = datetime.now(timezone.utc)
    reference_id = str(uuid4())
    return CardFreezeResponse(
        card_id=card.card_id,
        status=card.status,
        card_name=card.card_name,
        last_4=card.last_4,
        timestamp=timestamp,
        reference_id=reference_id,
        message=(
            f"{card.card_name} ending in {card.last_4} was frozen at "
            f"{timestamp.isoformat()} (Reference ID: {reference_id})."
        ),
    )


async def _verify_freeze_otp(
    user_id: str, otp_code: str, authorization: str | None
) -> None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                os.getenv(
                    "MFA_VERIFICATION_URL",
                    "http://localhost:8003/verify-otp",
                ),
                headers={"Authorization": authorization or ""},
                json={
                    "account_id": user_id,
                    "otp_code": otp_code,
                    "purpose": "card_freeze",
                },
            )
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="MFA service is unavailable") from None

    try:
        payload = response.json()
    except ValueError:
        raise HTTPException(
            status_code=502, detail="Invalid response from MFA service"
        ) from None
    if response.status_code >= 500:
        raise HTTPException(status_code=503, detail="MFA service is unavailable")
    if not response.is_success or not (
        isinstance(payload, dict) and payload.get("status") == "verified"
    ):
        raise HTTPException(status_code=401, detail="MFA verification failed")


async def freeze_card_for_fraud(user_id: str, card_id: str) -> CardFreezeResponse:
    """Freeze an owned card immediately when called by the trusted fraud workflow."""
    require_sandbox_mode()
    card = await _set_owned_card_status(user_id, card_id, "frozen")
    return _freeze_response(card)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"service": "cards", "status": "ok"}


@app.get(
    "/sandbox/cards",
    response_model=list[CardResponse],
    tags=["sandbox"],
    dependencies=[Depends(require_sandbox_mode)],
)
async def list_sandbox_cards(
    authenticated_user_id: str = Depends(authenticated_subject),
) -> list[CardResponse]:
    try:
        cards = await asyncio.to_thread(list_user_cards, authenticated_user_id)
    except CardStoreUnavailable:
        raise HTTPException(status_code=503, detail="Card storage is unavailable") from None
    return [_card_response(card) for card in cards]


@app.post(
    "/sandbox/cards/{card_id}/freeze",
    response_model=CardFreezeResponse,
    tags=["sandbox"],
    dependencies=[Depends(require_sandbox_mode)],
)
async def freeze_sandbox_card(
    card_id: str,
    request: CardFreezeRequest,
    authorization: str | None = Header(default=None),
    authenticated_user_id: str = Depends(authenticated_subject),
) -> CardFreezeResponse:
    await _owned_card_or_404(authenticated_user_id, card_id)
    if not request.otp_code:
        raise HTTPException(status_code=401, detail="MFA verification is required")
    await _verify_freeze_otp(
        authenticated_user_id,
        request.otp_code,
        authorization,
    )
    card = await _set_owned_card_status(authenticated_user_id, card_id, "frozen")
    return _freeze_response(card)


@app.post(
    "/sandbox/cards/{card_id}/reset",
    response_model=CardResponse,
    tags=["sandbox"],
    dependencies=[Depends(require_sandbox_mode)],
)
async def reset_sandbox_card(
    card_id: str,
    authenticated_user_id: str = Depends(authenticated_subject),
) -> CardResponse:
    card = await _set_owned_card_status(authenticated_user_id, card_id, "active")
    return _card_response(card)
