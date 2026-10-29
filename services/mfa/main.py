import ipaddress
from decimal import Decimal

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from services.sandbox import require_sandbox_mode

DEMO_OTP = "123456"

app = FastAPI(title="Sandbox MFA Verifier", version="0.1.0")


class OtpVerificationRequest(BaseModel):
    account_id: str = Field(min_length=1, max_length=256)
    otp_code: str = Field(pattern=r"^\d{6}$")
    transaction_amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2)


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
async def verify_otp(verification: OtpVerificationRequest) -> dict[str, str]:
    if verification.otp_code != DEMO_OTP:
        raise HTTPException(status_code=401, detail="MFA verification failed")
    return {"status": "verified"}