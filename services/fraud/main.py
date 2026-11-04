import asyncio
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from services.api_errors import register_exception_handlers
from services.authentication import require_scope
from services.cards.store import CardStoreUnavailable, get_owned_card
from services.fraud.store import (
    FraudCaseIdempotencyConflict,
    FraudCaseRecord,
    FraudCaseStoreUnavailable,
    create_fraud_case,
    get_idempotent_fraud_case,
)
from services.orchestrator.pii import PiiMaskingUnavailable, anonymize_message
from services.sandbox import SANDBOX_CARD_OWNER_BY_ACCOUNT_ID, require_sandbox_mode

app = FastAPI(title="Fraud Case Service", version="0.1.0")
register_exception_handlers(app)


class FraudCaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=4000)
    frozen_card_ids: list[str] = Field(default_factory=list, max_length=20)


class FraudCaseResponse(BaseModel):
    case_id: str
    user_id: str
    status: str
    reason: str
    frozen_card_ids: list[str]
    idempotency_key: str | None
    created_at: datetime


def _case_response(record: FraudCaseRecord) -> FraudCaseResponse:
    return FraudCaseResponse(
        case_id=record.case_id,
        user_id=record.user_id,
        status=record.status,
        reason=record.reason,
        frozen_card_ids=record.frozen_card_ids,
        idempotency_key=record.idempotency_key,
        created_at=record.created_at,
    )


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"service": "fraud", "status": "ok"}


@app.post(
    "/fraud/cases",
    response_model=FraudCaseResponse,
    status_code=201,
    tags=["fraud"],
    dependencies=[Depends(require_sandbox_mode)],
)
async def open_fraud_case(
    request: FraudCaseRequest,
    user_id: str = Depends(require_scope("fraud:report")),
    idempotency_key: Annotated[
        str | None, Header(alias="Idempotency-Key", min_length=1, max_length=128)
    ] = None,
) -> FraudCaseResponse:
    try:
        reason = anonymize_message(request.reason.strip())
    except PiiMaskingUnavailable:
        raise HTTPException(
            status_code=503, detail="Fraud report processing is unavailable"
        ) from None
    if not reason:
        raise HTTPException(status_code=422, detail="A fraud report reason is required")
    if any(not card_id or len(card_id) > 128 for card_id in request.frozen_card_ids):
        raise HTTPException(status_code=422, detail="Invalid frozen card identifier")

    frozen_card_ids = list(dict.fromkeys(request.frozen_card_ids))
    if idempotency_key:
        try:
            existing = await asyncio.to_thread(
                get_idempotent_fraud_case,
                user_id=user_id,
                idempotency_key=idempotency_key,
            )
        except FraudCaseStoreUnavailable:
            raise HTTPException(
                status_code=503, detail="Fraud case storage is unavailable"
            ) from None
        if existing is not None:
            if (
                existing.reason != reason
                or existing.frozen_card_ids != frozen_card_ids
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Idempotency key was already used for another report",
                )
            return _case_response(existing)

    if frozen_card_ids:
        card_owner_id = SANDBOX_CARD_OWNER_BY_ACCOUNT_ID.get(user_id)
        if card_owner_id is None:
            raise HTTPException(status_code=422, detail="Invalid frozen card identifier")
        try:
            for card_id in frozen_card_ids:
                card = await asyncio.to_thread(get_owned_card, card_owner_id, card_id)
                if card is None or card.status != "frozen":
                    raise HTTPException(
                        status_code=422, detail="Invalid frozen card identifier"
                    )
        except CardStoreUnavailable:
            raise HTTPException(
                status_code=503, detail="Card storage is unavailable"
            ) from None

    try:
        record = create_fraud_case(
            user_id=user_id,
            reason=reason,
            frozen_card_ids=frozen_card_ids,
            idempotency_key=idempotency_key,
        )
    except FraudCaseIdempotencyConflict:
        raise HTTPException(
            status_code=409, detail="Idempotency key was already used for another report"
        ) from None
    except FraudCaseStoreUnavailable:
        raise HTTPException(status_code=503, detail="Fraud case storage is unavailable") from None
    return _case_response(record)
