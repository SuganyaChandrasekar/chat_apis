import asyncio
import os
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from services.api_errors import register_exception_handlers
from services.authentication import (
    AuthorizationDenied,
    InvalidAccessToken,
    JwtConfigurationError,
    bearer_token,
    exchange_access_token,
    require_scope,
)
from services.authorization import (
    InvalidScaAttestation,
    card_freeze_action_id,
    validate_sca_attestation,
)
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


class FraudFreezeRequest(BaseModel):
    card_ids: list[str] | None = Field(default=None, max_length=20)


class FraudFreezeResponse(BaseModel):
    frozen_card_ids: list[str]
    newly_frozen_card_ids: list[str]
    already_frozen_card_ids: list[str]


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
    user_id: str,
    card_id: str,
    otp_code: str,
    authorization: str | None,
) -> None:
    action_id = card_freeze_action_id(subject=user_id, card_id=card_id)
    try:
        token = exchange_access_token(
            bearer_token(authorization),
            scopes=("cards:freeze",),
            consent_state="granted",
            consent_action=action_id,
            sca_state="required",
        )
    except JwtConfigurationError:
        raise HTTPException(
            status_code=503, detail="Authentication is not configured"
        ) from None
    except InvalidAccessToken:
        raise HTTPException(
            status_code=401,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except AuthorizationDenied:
        raise HTTPException(
            status_code=403, detail="Card freeze permission is not granted"
        ) from None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                os.getenv(
                    "MFA_VERIFICATION_URL",
                    "http://localhost:8003/verify-otp",
                ),
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "account_id": user_id,
                    "otp_code": otp_code,
                    "purpose": "card_freeze",
                    "transaction_id": card_id,
                    "action_id": action_id,
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
    if not response.is_success or not isinstance(payload, dict) or payload.get("status") != "verified":
        raise HTTPException(
            status_code=401,
            detail=(
                "Strong authentication failed. The card was not frozen; retry "
                "verification or cancel."
            ),
        )
    sca_token = payload.get("sca_token")
    if not isinstance(sca_token, str):
        raise HTTPException(status_code=401, detail="MFA verification failed")
    try:
        validate_sca_attestation(
            sca_token,
            subject=user_id,
            action_id=action_id,
            transaction_id=card_id,
            purpose="card_freeze",
        )
    except InvalidScaAttestation:
        raise HTTPException(
            status_code=401,
            detail=(
                "Strong authentication failed. The card was not frozen; retry "
                "verification or cancel."
            ),
        ) from None


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
    authenticated_user_id: str = Depends(require_scope("cards:read")),
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
    authenticated_user_id: str = Depends(require_scope("cards:freeze")),
) -> CardFreezeResponse:
    await _owned_card_or_404(authenticated_user_id, card_id)
    if not request.otp_code:
        raise HTTPException(
            status_code=401,
            detail=(
                "Strong authentication is required. The card was not frozen; "
                "retry verification or cancel."
            ),
        )
    await _verify_freeze_otp(
        authenticated_user_id,
        card_id,
        request.otp_code,
        authorization,
    )
    card = await _set_owned_card_status(authenticated_user_id, card_id, "frozen")
    return _freeze_response(card)


@app.post(
    "/sandbox/fraud/freeze",
    response_model=FraudFreezeResponse,
    tags=["sandbox", "fraud"],
    dependencies=[Depends(require_sandbox_mode)],
)
async def freeze_fraud_cards(
    request: FraudFreezeRequest,
    authenticated_user_id: str = Depends(require_scope("fraud:respond")),
) -> FraudFreezeResponse:
    if request.card_ids is not None and any(
        not card_id or len(card_id) > 128 for card_id in request.card_ids
    ):
        raise HTTPException(status_code=422, detail="Invalid card identifier")

    try:
        cards = await asyncio.to_thread(list_user_cards, authenticated_user_id)
    except CardStoreUnavailable:
        raise HTTPException(status_code=503, detail="Card storage is unavailable") from None

    owned_cards = {card.card_id: card for card in cards}
    if request.card_ids is None:
        target_ids = [card.card_id for card in cards]
    else:
        target_ids = list(dict.fromkeys(request.card_ids))
        if not target_ids or any(card_id not in owned_cards for card_id in target_ids):
            raise HTTPException(status_code=404, detail="Card not found")

    frozen_card_ids = []
    newly_frozen_card_ids = []
    already_frozen_card_ids = []
    for card_id in target_ids:
        if owned_cards[card_id].status == "active":
            await freeze_card_for_fraud(authenticated_user_id, card_id)
            newly_frozen_card_ids.append(card_id)
        else:
            already_frozen_card_ids.append(card_id)
        frozen_card_ids.append(card_id)
    return FraudFreezeResponse(
        frozen_card_ids=frozen_card_ids,
        newly_frozen_card_ids=newly_frozen_card_ids,
        already_frozen_card_ids=already_frozen_card_ids,
    )


@app.post(
    "/sandbox/cards/{card_id}/reset",
    response_model=CardResponse,
    tags=["sandbox"],
    dependencies=[Depends(require_sandbox_mode)],
)
async def reset_sandbox_card(
    card_id: str,
    authenticated_user_id: str = Depends(require_scope("cards:freeze")),
) -> CardResponse:
    card = await _set_owned_card_status(authenticated_user_id, card_id, "active")
    return _card_response(card)
