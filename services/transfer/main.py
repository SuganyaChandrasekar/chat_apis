import asyncio
import os
from decimal import Decimal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException

from services.api_errors import register_exception_handlers
from services.authentication import authenticated_subject
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
)

app = FastAPI(title="Transfer Service", version="0.1.0")
register_exception_handlers(app)
balances = SANDBOX_INITIAL_BALANCES.copy()
transfer_lock = asyncio.Lock()
HIGH_VALUE_TRANSFER_THRESHOLD = Decimal(
    os.getenv("HIGH_VALUE_TRANSFER_THRESHOLD", "1000.00")
)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"service": "transfer", "status": "ok"}


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

    source_balance = balances.get(transfer.source_account_id)
    destination_balance = balances.get(transfer.destination_account_id)
    if source_balance is None or destination_balance is None:
        await _record_transfer_status(
            start.transaction_id, authenticated_account_id, idempotency_key, transfer, "FAILED"
        )
        raise HTTPException(status_code=404, detail="Sandbox account not found")
    if sandbox_account_status(transfer.source_account_id) != "ACTIVE":
        await _record_transfer_status(
            start.transaction_id, authenticated_account_id, idempotency_key, transfer, "FAILED"
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

    mfa_required = transfer.amount >= HIGH_VALUE_TRANSFER_THRESHOLD
    verified = not mfa_required
    if mfa_required and not transfer.otp_code:
        await _record_transfer_status(
            start.transaction_id, authenticated_account_id, idempotency_key, transfer, "FAILED"
        )
        raise HTTPException(status_code=401, detail="MFA verification is required")
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
                    },
                )
                verification_payload = verification.json()
                verified = verification.is_success and (
                    isinstance(verification_payload, dict)
                    and verification_payload.get("status") == "verified"
                )
        except httpx.RequestError:
            await _record_transfer_status(
                start.transaction_id, authenticated_account_id, idempotency_key, transfer, "FAILED"
            )
            raise HTTPException(status_code=503, detail="MFA service is unavailable") from None
        except ValueError:
            verified = False

    if not verified:
        await _record_transfer_status(
            start.transaction_id, authenticated_account_id, idempotency_key, transfer, "FAILED"
        )
        raise HTTPException(status_code=401, detail="MFA verification failed")

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
        )
    except AuditStoreUnavailable:
        raise HTTPException(status_code=503, detail="Transfer audit is unavailable") from None