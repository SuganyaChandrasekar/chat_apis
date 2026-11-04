import asyncio
import os
from decimal import Decimal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from services.api_errors import register_exception_handlers
from services.authentication import (
    AuthorizationDenied,
    InvalidAccessToken,
    JwtConfigurationError,
    authenticated_subject,
    authorize_access_token,
    bearer_token,
    create_access_token,
    require_scope,
)
from services.authorization import (
    HIGH_VALUE_TRANSFER_THRESHOLD,
    InvalidScaAttestation,
    authorize_transfer_context,
    transfer_action_id,
    validate_sca_attestation,
)
from services.sandbox import (
    SANDBOX_CURRENCY,
    SANDBOX_CUSTOMER_ACCOUNTS,
    SANDBOX_INITIAL_BALANCES,
    SandboxTransferRequest,
    SandboxTransferResponse,
    require_sandbox_mode,
    sandbox_account_status,
)
from services.transfer.audit import (
    AuditStoreUnavailable,
    IdempotencyConflict,
    append_transfer_status,
    begin_transfer,
    latest_successful_transfer,
)

app = FastAPI(title="Transfer Service", version="0.1.0")
register_exception_handlers(app)
balances = SANDBOX_INITIAL_BALANCES.copy()
transfer_lock = asyncio.Lock()


class LatestTransferResponse(BaseModel):
    source_account_id: str
    destination_account_id: str


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"service": "transfer", "status": "ok"}


@app.get(
    "/transfers/latest",
    response_model=LatestTransferResponse,
    tags=["sandbox"],
    dependencies=[Depends(require_sandbox_mode)],
)
async def get_latest_transfer(
    authenticated_account_id: str = Depends(require_scope("transfer:read")),
) -> LatestTransferResponse:
    try:
        transfer = await asyncio.to_thread(
            latest_successful_transfer,
            authenticated_account_id,
        )
    except AuditStoreUnavailable:
        raise HTTPException(
            status_code=503, detail="Transfer audit is unavailable"
        ) from None
    if transfer is None:
        raise HTTPException(status_code=404, detail="No completed transfer found")

    return LatestTransferResponse(
        source_account_id=transfer.source_account_id,
        destination_account_id=transfer.destination_account_id,
    )


@app.post(
    "/transfer",
    response_model=SandboxTransferResponse,
    tags=["sandbox"],
    dependencies=[Depends(require_sandbox_mode)],
)
async def create_sandbox_transfer(
    transfer: SandboxTransferRequest,
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=128),
    authorization: str | None = Header(default=None),
    authenticated_account_id: str = Depends(authenticated_subject),
) -> SandboxTransferResponse:
    if not transfer.confirmed:
        raise HTTPException(status_code=409, detail="Transfer confirmation is required")
    try:
        access_token = bearer_token(authorization)
        context = authorize_access_token(
            access_token,
            required_scope="transfer:execute",
        )
        if context.subject != authenticated_account_id:
            raise AuthorizationDenied
        action_id = transfer_action_id(
            account_id=authenticated_account_id,
            source_account_id=transfer.source_account_id,
            destination_account_id=transfer.destination_account_id,
            amount=transfer.amount,
            currency=transfer.currency,
            idempotency_key=idempotency_key,
        )
        if context.consent_state == "granted":
            service_token = access_token
        elif context.consent_state == "not_granted":
            service_token = create_access_token(
                authenticated_account_id,
                ttl_seconds=60,
                scopes=("transfer:execute",),
                consent_state="granted",
                consent_action=action_id,
                sca_state=(
                    "required"
                    if transfer.amount >= HIGH_VALUE_TRANSFER_THRESHOLD
                    else "not_required"
                ),
                token_kind="delegated",
            )
        else:
            raise AuthorizationDenied
        authorize_transfer_context(
            service_token,
            account_id=authenticated_account_id,
            source_account_id=transfer.source_account_id,
            destination_account_id=transfer.destination_account_id,
            amount=transfer.amount,
            currency=transfer.currency,
            idempotency_key=idempotency_key,
        )
        authorization = f"Bearer {service_token}"
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
            status_code=403,
            detail="Transfer permission or transaction consent is not granted",
        ) from None

    authorized_accounts = SANDBOX_CUSTOMER_ACCOUNTS.get(authenticated_account_id, ())
    if (
        transfer.source_account_id not in authorized_accounts
        or transfer.destination_account_id not in authorized_accounts
    ):
        raise HTTPException(status_code=403, detail="Account access is not authorized")
    if transfer.source_account_id == transfer.destination_account_id:
        raise HTTPException(
            status_code=422, detail="Source and destination accounts must differ"
        )
    if transfer.currency != SANDBOX_CURRENCY:
        raise HTTPException(status_code=422, detail="Unsupported sandbox currency")

    try:
        start = await asyncio.to_thread(
            begin_transfer,
            account_id=authenticated_account_id,
            idempotency_key=idempotency_key,
            source_account_id=transfer.source_account_id,
            destination_account_id=transfer.destination_account_id,
            amount=transfer.amount,
            currency=transfer.currency,
            consent_state="GRANTED",
            sca_state=(
                "REQUIRED"
                if transfer.amount >= HIGH_VALUE_TRANSFER_THRESHOLD
                else "NOT_REQUIRED"
            ),
        )
    except IdempotencyConflict:
        raise HTTPException(status_code=409, detail="Idempotency key already used") from None
    except AuditStoreUnavailable:
        raise HTTPException(status_code=503, detail="Transfer audit is unavailable") from None

    if start.replayed:
        return SandboxTransferResponse(
            transaction_id=start.transaction_id,
            status="SUCCESS",
            source_account_id=transfer.source_account_id,
            destination_account_id=transfer.destination_account_id,
            amount=transfer.amount,
            currency=transfer.currency,
        )

    mfa_required = transfer.amount >= HIGH_VALUE_TRANSFER_THRESHOLD
    verified = not mfa_required
    sca_state = "NOT_REQUIRED" if not mfa_required else "REQUIRED"
    source_balance = balances.get(transfer.source_account_id)
    destination_balance = balances.get(transfer.destination_account_id)
    if source_balance is None or destination_balance is None:
        await _record_transfer_status(
            start.transaction_id,
            authenticated_account_id,
            idempotency_key,
            transfer,
            "FAILED",
            sca_state=sca_state,
        )
        raise HTTPException(status_code=404, detail="Sandbox account not found")
    if sandbox_account_status(transfer.source_account_id) != "ACTIVE":
        await _record_transfer_status(
            start.transaction_id,
            authenticated_account_id,
            idempotency_key,
            transfer,
            "FAILED",
            sca_state=sca_state,
        )
        raise HTTPException(
            status_code=409,
            detail="ACCOUNT_FROZEN: This source account is frozen and cannot send transfers.",
        )
    if source_balance < transfer.amount:
        await _record_transfer_status(
            start.transaction_id, authenticated_account_id, idempotency_key, transfer, "FAILED"
        )
        raise HTTPException(
            status_code=409,
            detail=f"INSUFFICIENT_FUNDS: Available balance is {SANDBOX_CURRENCY} {source_balance:.2f}.",
        )

    if mfa_required and not transfer.otp_code:
        await _record_transfer_status(
            start.transaction_id,
            authenticated_account_id,
            idempotency_key,
            transfer,
            "FAILED",
            sca_state="ABANDONED",
        )
        raise HTTPException(
            status_code=401,
            detail=(
                "Strong customer authentication is required. No transfer was made; "
                "retry in your bank's secure verification flow or cancel."
            ),
        )
    if mfa_required:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                verification = await client.post(
                    os.getenv("MFA_VERIFICATION_URL", "http://localhost:8003/verify-otp"),
                    headers={"Authorization": authorization or ""},
                    json={
                        "account_id": authenticated_account_id,
                        "otp_code": transfer.otp_code,
                        "transaction_amount": str(transfer.amount),
                        "transaction_id": start.transaction_id,
                        "source_account_id": transfer.source_account_id,
                        "destination_account_id": transfer.destination_account_id,
                        "currency": transfer.currency,
                        "idempotency_key": idempotency_key,
                    },
                )
                verification_payload = verification.json()
                verified = verification.is_success and (
                    isinstance(verification_payload, dict)
                    and verification_payload.get("status") == "verified"
                )
                if verified:
                    sca_token = verification_payload.get("sca_token")
                    if not isinstance(sca_token, str):
                        verified = False
                    else:
                        validate_sca_attestation(
                            sca_token,
                            subject=authenticated_account_id,
                            action_id=action_id,
                            transaction_id=start.transaction_id,
                            amount=transfer.amount,
                            currency=transfer.currency,
                        )
                sca_state = "VERIFIED" if verified else "FAILED"
        except httpx.RequestError:
            await _record_transfer_status(
                start.transaction_id,
                authenticated_account_id,
                idempotency_key,
                transfer,
                "FAILED",
                sca_state="UNAVAILABLE",
            )
            raise HTTPException(status_code=503, detail="MFA service is unavailable") from None
        except ValueError:
            verified = False
            sca_state = "FAILED"
        except InvalidScaAttestation:
            verified = False
            sca_state = "FAILED"

    if not verified:
        await _record_transfer_status(
            start.transaction_id,
            authenticated_account_id,
            idempotency_key,
            transfer,
            "FAILED",
            sca_state=sca_state,
        )
        raise HTTPException(
            status_code=401,
            detail=(
                "Strong customer authentication failed. No transfer was made; "
                "retry verification or cancel."
            ),
        )

    async with transfer_lock:
        source_balance = balances.get(transfer.source_account_id)
        destination_balance = balances.get(transfer.destination_account_id)
        if source_balance is None or destination_balance is None:
            await _record_transfer_status(
                start.transaction_id, authenticated_account_id, idempotency_key, transfer, "FAILED"
            )
            raise HTTPException(status_code=404, detail="Sandbox account not found")
        if source_balance < transfer.amount:
            await _record_transfer_status(
                start.transaction_id, authenticated_account_id, idempotency_key, transfer, "FAILED"
            )
            raise HTTPException(
                status_code=409,
                detail=f"INSUFFICIENT_FUNDS: Available balance is {SANDBOX_CURRENCY} {source_balance:.2f}.",
            )

        balances[transfer.source_account_id] -= transfer.amount
        balances[transfer.destination_account_id] += transfer.amount
        try:
            await _record_transfer_status(
                start.transaction_id,
                authenticated_account_id,
                idempotency_key,
                transfer,
                "SUCCESS",
                sca_state=sca_state,
            )
        except HTTPException:
            balances[transfer.source_account_id] += transfer.amount
            balances[transfer.destination_account_id] -= transfer.amount
            raise
    return SandboxTransferResponse(
        transaction_id=start.transaction_id,
        status="SUCCESS",
        source_account_id=transfer.source_account_id,
        destination_account_id=transfer.destination_account_id,
        amount=transfer.amount,
        currency=transfer.currency,
    )


async def _record_transfer_status(
    transaction_id: str,
    account_id: str,
    idempotency_key: str,
    transfer: SandboxTransferRequest,
    status: str,
    *,
    sca_state: str | None = None,
) -> None:
    try:
        await asyncio.to_thread(
            append_transfer_status,
            transaction_id=transaction_id,
            account_id=account_id,
            idempotency_key=idempotency_key,
            source_account_id=transfer.source_account_id,
            destination_account_id=transfer.destination_account_id,
            amount=transfer.amount,
            currency=transfer.currency,
            status=status,
            consent_state="GRANTED",
            sca_state=(
                sca_state
                if sca_state is not None
                else (
                    "NOT_REQUIRED"
                    if transfer.amount < HIGH_VALUE_TRANSFER_THRESHOLD
                    else "REQUIRED"
                )
            ),
        )
    except AuditStoreUnavailable:
        raise HTTPException(status_code=503, detail="Transfer audit is unavailable") from None