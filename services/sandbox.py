import os
from datetime import date, timedelta
from decimal import Decimal
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field


class SandboxBalance(BaseModel):
    account_id: str
    balance: Decimal
    currency: str


class SandboxTransaction(BaseModel):
    transaction_id: str
    date: date
    amount: Decimal
    currency: str
    description: str
    merchant: str


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
_SANDBOX_TRANSACTION_DATE = date.today()
SANDBOX_TRANSACTIONS: dict[str, tuple[SandboxTransaction, ...]] = {
    "sandbox-checking-001": (
        SandboxTransaction(
            transaction_id="checking-tx-001",
            date=_SANDBOX_TRANSACTION_DATE - timedelta(days=1),
            amount=Decimal("-5.75"),
            currency=SANDBOX_CURRENCY,
            description="Coffee",
            merchant="Northstar Coffee",
        ),
        SandboxTransaction(
            transaction_id="checking-tx-002",
            date=_SANDBOX_TRANSACTION_DATE - timedelta(days=2),
            amount=Decimal("-42.18"),
            currency=SANDBOX_CURRENCY,
            description="Grocery purchase",
            merchant="Market Street Grocer",
        ),
        SandboxTransaction(
            transaction_id="checking-tx-003",
            date=_SANDBOX_TRANSACTION_DATE - timedelta(days=4),
            amount=Decimal("-38.50"),
            currency=SANDBOX_CURRENCY,
            description="Fuel",
            merchant="City Fuel",
        ),
        SandboxTransaction(
            transaction_id="checking-tx-004",
            date=_SANDBOX_TRANSACTION_DATE - timedelta(days=8),
            amount=Decimal("-12.99"),
            currency=SANDBOX_CURRENCY,
            description="Music subscription",
            merchant="SoundStream",
        ),
        SandboxTransaction(
            transaction_id="checking-tx-005",
            date=_SANDBOX_TRANSACTION_DATE - timedelta(days=15),
            amount=Decimal("1250.00"),
            currency=SANDBOX_CURRENCY,
            description="Payroll deposit",
            merchant="Acme Payroll",
        ),
    ),
    "sandbox-savings-001": (
        SandboxTransaction(
            transaction_id="savings-tx-001",
            date=_SANDBOX_TRANSACTION_DATE - timedelta(days=3),
            amount=Decimal("25.00"),
            currency=SANDBOX_CURRENCY,
            description="Monthly interest",
            merchant="Bank Interest",
        ),
        SandboxTransaction(
            transaction_id="savings-tx-002",
            date=_SANDBOX_TRANSACTION_DATE - timedelta(days=10),
            amount=Decimal("-100.00"),
            currency=SANDBOX_CURRENCY,
            description="Transfer to checking",
            merchant="Internal Transfer",
        ),
    ),
}


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