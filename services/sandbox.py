import os
from decimal import Decimal
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field


class SandboxBalance(BaseModel):
    account_id: str
    balance: Decimal
    currency: str


class SandboxAccount(BaseModel):
    account_id: str
    label: str
    status: Literal["ACTIVE", "FROZEN"] = "ACTIVE"


class SandboxTransferRequest(BaseModel):
    source_account_id: str
    destination_account_id: str
    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    currency: str = "USD"
    confirmed: bool = False
    otp_code: str | None = Field(default=None, pattern=r"^\d{6}$")


class SandboxTransferResponse(BaseModel):
    transaction_id: str
    status: str
    source_account_id: str
    destination_account_id: str
    amount: Decimal
    currency: str


SANDBOX_INITIAL_BALANCES = {
    "sandbox-checking-001": Decimal("1500.00"),
    "sandbox-savings-001": Decimal("3200.00"),
}
SANDBOX_CUSTOMER_ACCOUNTS = {
    "sandbox-checking-001": tuple(SANDBOX_INITIAL_BALANCES),
}
SANDBOX_ACCOUNT_LABELS = {
    "sandbox-checking-001": "Checking account",
    "sandbox-savings-001": "Savings account",
}
SANDBOX_CURRENCY = "USD"


def sandbox_account_status(account_id: str) -> str:
    frozen_ids = {
        value.strip()
        for value in os.getenv("SANDBOX_FROZEN_ACCOUNT_IDS", "").split(",")
        if value.strip()
    }
    return "FROZEN" if account_id in frozen_ids else "ACTIVE"


def require_sandbox_mode() -> None:
    if os.getenv("SANDBOX_MODE", "false").lower() != "true":
        raise HTTPException(status_code=404, detail="Not found")