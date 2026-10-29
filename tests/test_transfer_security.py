import asyncio
from decimal import Decimal

import pytest
from fastapi import HTTPException

from services.sandbox import SANDBOX_INITIAL_BALANCES, SandboxTransferRequest
from services.transfer import main as transfer_service
from services.transfer.audit import TransferStart


def _transfer(source_account_id: str, amount: str, otp_code: str | None = None):
    return SandboxTransferRequest(
        source_account_id=source_account_id,
        destination_account_id="sandbox-checking-001",
        amount=amount,
        confirmed=True,
        otp_code=otp_code,
    )


def test_frozen_source_is_blocked_and_audited(monkeypatch) -> None:
    monkeypatch.setattr(transfer_service, "balances", SANDBOX_INITIAL_BALANCES.copy())
    monkeypatch.setattr(
        transfer_service,
        "begin_transfer",
        lambda **kwargs: TransferStart("frozen-transfer", "PENDING_VALIDATION", False),
    )
    monkeypatch.setattr(
        transfer_service,
        "sandbox_account_status",
        lambda account_id: "FROZEN" if account_id == "sandbox-savings-001" else "ACTIVE",
    )
    audit_statuses = []
    monkeypatch.setattr(
        transfer_service,
        "append_transfer_status",
        lambda **kwargs: audit_statuses.append(kwargs["status"]),
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            transfer_service.create_sandbox_transfer(
                _transfer("sandbox-savings-001", "10.00"),
                "frozen-key",
                "Bearer validated-token",
                "sandbox-checking-001",
            )
        )

    assert error.value.status_code == 409
    assert error.value.detail.startswith("ACCOUNT_FROZEN:")
    assert audit_statuses == ["FAILED"]
    assert transfer_service.balances["sandbox-savings-001"] == Decimal("3200.00")


def test_high_value_transfer_requires_otp_and_is_audited(monkeypatch) -> None:
    monkeypatch.setattr(transfer_service, "balances", SANDBOX_INITIAL_BALANCES.copy())
    monkeypatch.setattr(
        transfer_service,
        "begin_transfer",
        lambda **kwargs: TransferStart("high-value-transfer", "PENDING_VALIDATION", False),
    )
    audit_statuses = []
    monkeypatch.setattr(
        transfer_service,
        "append_transfer_status",
        lambda **kwargs: audit_statuses.append(kwargs["status"]),
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            transfer_service.create_sandbox_transfer(
                    _transfer("sandbox-savings-001", "1000.00"),
                "high-value-key",
                "Bearer validated-token",
                "sandbox-checking-001",
            )
        )

    assert error.value.status_code == 401
    assert error.value.detail == "MFA verification is required"
    assert audit_statuses == ["FAILED"]
    assert transfer_service.balances["sandbox-savings-001"] == Decimal("3200.00")


def test_insufficient_balance_is_rejected_before_mfa_verification(monkeypatch) -> None:
    monkeypatch.setattr(transfer_service, "balances", SANDBOX_INITIAL_BALANCES.copy())
    monkeypatch.setattr(
        transfer_service,
        "begin_transfer",
        lambda **kwargs: TransferStart("insufficient-funds-transfer", "PENDING_VALIDATION", False),
    )
    audit_statuses = []
    monkeypatch.setattr(
        transfer_service,
        "append_transfer_status",
        lambda **kwargs: audit_statuses.append(kwargs["status"]),
    )

    class UnexpectedMfaClient:
        def __init__(self, *args, **kwargs):
            pytest.fail("MFA verifier must not be called when funds are insufficient")

    monkeypatch.setattr(transfer_service.httpx, "AsyncClient", UnexpectedMfaClient)
    transfer = SandboxTransferRequest(
        source_account_id="sandbox-checking-001",
        destination_account_id="sandbox-savings-001",
        amount="1600.00",
        confirmed=True,
        otp_code="123456",
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            transfer_service.create_sandbox_transfer(
                transfer,
                "insufficient-funds-key",
                "Bearer validated-token",
                "sandbox-checking-001",
            )
        )

    assert error.value.status_code == 409
    assert error.value.detail.startswith("INSUFFICIENT_FUNDS:")
    assert audit_statuses == ["FAILED"]
    assert transfer_service.balances == SANDBOX_INITIAL_BALANCES
