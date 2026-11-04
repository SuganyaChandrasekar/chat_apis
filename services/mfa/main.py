import ipaddress
from decimal import Decimal
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from services.authentication import (
    AuthorizationDenied,
    InvalidAccessToken,
    JwtConfigurationError,
    bearer_token,
)
from services.authorization import (
    authorize_card_freeze_context,
    authorize_transfer_context,
    create_sca_attestation,
)
from services.sandbox import require_sandbox_mode

DEMO_OTP = "123456"

app = FastAPI(title="Sandbox MFA Verifier", version="0.1.0")


class OtpVerificationRequest(BaseModel):
    account_id: str = Field(min_length=1, max_length=256)
    otp_code: str = Field(pattern=r"^\d{6}$")
    purpose: Literal["transfer", "card_freeze"] = "transfer"
    transaction_amount: Decimal | None = Field(
        default=None, gt=0, max_digits=12, decimal_places=2
    )
    transaction_id: str | None = Field(default=None, min_length=1, max_length=36)
    action_id: str | None = Field(default=None, min_length=1, max_length=128)
    source_account_id: str | None = Field(default=None, min_length=1, max_length=256)
    destination_account_id: str | None = Field(
        default=None, min_length=1, max_length=256
    )
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def require_transfer_context(self) -> "OtpVerificationRequest":
        if self.purpose == "transfer" and (
            self.transaction_amount is None
            or self.transaction_id is None
            or self.source_account_id is None
            or self.destination_account_id is None
            or self.idempotency_key is None
        ):
            raise ValueError("Complete transfer context is required for verification")
        return self


def require_loopback_client(request: Request) -> None:
    client = request.client
    try:
        is_loopback = client is not None and ipaddress.ip_address(client.host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise HTTPException(status_code=403, detail="Local access only")


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"service": "mfa", "status": "ok"}


@app.post(
    "/verify-otp",
    tags=["sandbox"],
    dependencies=[Depends(require_sandbox_mode), Depends(require_loopback_client)],
)
async def verify_otp(
    verification: OtpVerificationRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    try:
        token = bearer_token(authorization)
        if verification.purpose == "transfer":
            action_id, context = authorize_transfer_context(
                token,
                account_id=verification.account_id,
                source_account_id=verification.source_account_id or "",
                destination_account_id=verification.destination_account_id or "",
                amount=verification.transaction_amount,
                currency=verification.currency,
                idempotency_key=verification.idempotency_key or "",
            )
        else:
            if not verification.transaction_id:
                raise AuthorizationDenied
            action_id, context = authorize_card_freeze_context(
                token,
                subject=verification.account_id,
                card_id=verification.transaction_id,
            )
            if verification.action_id != action_id:
                raise AuthorizationDenied
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
        raise HTTPException(status_code=403, detail="Required permission is not granted") from None

    if verification.otp_code != DEMO_OTP:
        raise HTTPException(status_code=401, detail="MFA verification failed")
    if verification.purpose == "transfer":
        return {
            "status": "verified",
            "sca_token": create_sca_attestation(
                subject=context.subject,
                action_id=action_id,
                transaction_id=verification.transaction_id or "",
                purpose=verification.purpose,
                amount=verification.transaction_amount,
                currency=(
                    verification.currency
                    if verification.purpose == "transfer"
                    else None
                ),
            ),
        }
    return {
        "status": "verified",
        "sca_token": create_sca_attestation(
            subject=context.subject,
            action_id=action_id,
            transaction_id=verification.transaction_id or "",
            purpose=verification.purpose,
        ),
    }