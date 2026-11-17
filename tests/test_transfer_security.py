import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from services.authentication import (
    bearer_token,
    create_access_token,
    validate_access_context,
)
from services.authorization import create_sca_attestation, transfer_action_id
from services.sandbox import SANDBOX_INITIAL_BALANCES, SandboxTransferRequest
from services.sandbox_balance_store import (
    SandboxBalanceBase,
    SandboxBalanceStoreUnavailable,
    execute_sandbox_transfer,
    get_sandbox_balance,
)
from services.transfer import main as transfer_service
from services.transfer import audit as transfer_audit
from services.transfer.audit import (
    LatestSuccessfulTransfer,
    TransferAuditEvent,
    TransferStart,
)


def _transfer(source_account_id: str, amount: str, otp_code: str | None = None):
    return SandboxTransferRequest(
        source_account_id=source_account_id,
        destination_account_id="sandbox-checking-001",
        amount=amount,
        confirmed=True,
        otp_code=otp_code,
    )


@pytest.fixture(autouse=True)
def transfer_authorization(monkeypatch, tmp_path):
    monkeypatch.setenv("JWT_SECRET", "transfer-security-test-secret-32-bytes")
    monkeypatch.setenv("SANDBOX_MODE", "true")
    engine = create_engine(f"sqlite:///{tmp_path / 'transfer.db'}")
    SandboxBalanceBase.metadata.create_all(engine)
    TransferAuditEvent.__table__.create(engine)
    monkeypatch.setattr(transfer_audit, "_engine", lambda: engine)
    monkeypatch.setattr(
        "services.sandbox_balance_store._engine", lambda: engine
    )
    token = create_access_token(
        "sandbox-checking-001",
        scopes=("transfer:execute",),
    )
    monkeypatch.setattr(transfer_service, "bearer_token", lambda _header: token)


def _authorization(monkeypatch) -> str:
    monkeypatch.setenv("JWT_SECRET", "transfer-security-test-secret-32-bytes")
    return f"Bearer {create_access_token('sandbox-checking-001', scopes=('transfer:execute',))}"


def test_frozen_source_is_blocked_and_audited(monkeypatch) -> None:
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
        lambda **kwargs: audit_statuses.append(kwargs),
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
    assert audit_statuses[0]["status"] == "FAILED"
    assert audit_statuses[0]["sca_state"] == "NOT_REQUIRED"
    assert get_sandbox_balance("sandbox-savings-001").balance == Decimal("3200.00")


def test_high_value_transfer_requires_otp_and_is_audited(monkeypatch) -> None:
    monkeypatch.setattr(
        transfer_service,
        "begin_transfer",
        lambda **kwargs: TransferStart("high-value-transfer", "PENDING_VALIDATION", False),
    )
    audit_statuses = []
    monkeypatch.setattr(
        transfer_service,
        "append_transfer_status",
        lambda **kwargs: audit_statuses.append(kwargs),
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
    assert "retry in your bank's secure verification flow or cancel" in error.value.detail
    assert audit_statuses[0]["status"] == "FAILED"
    assert audit_statuses[0]["sca_state"] == "ABANDONED"
    assert get_sandbox_balance("sandbox-savings-001").balance == Decimal("3200.00")


def test_insufficient_balance_is_rejected_before_mfa_verification(monkeypatch) -> None:
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
    assert {
        account_id: get_sandbox_balance(account_id).balance
        for account_id in SANDBOX_INITIAL_BALANCES
    } == SANDBOX_INITIAL_BALANCES


def test_high_value_transfer_executes_only_with_matching_sca_attestation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        transfer_service,
        "begin_transfer",
        lambda **kwargs: TransferStart("sca-transaction", "PENDING_VALIDATION", False),
    )
    audit_events = []
    monkeypatch.setattr(
        transfer_service,
        "append_transfer_status",
        lambda **kwargs: audit_events.append(kwargs),
    )

    class VerifiedMfaClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def post(self, url, headers, json):
            assert url.endswith("/verify-otp")
            context = validate_access_context(
                bearer_token(headers["Authorization"])
            )
            assert context.subject == "sandbox-checking-001"
            assert context.consent_state == "granted"
            assert context.sca_state == "required"
            action_id = transfer_action_id(
                account_id=json["account_id"],
                source_account_id=json["source_account_id"],
                destination_account_id=json["destination_account_id"],
                amount=Decimal(json["transaction_amount"]),
                currency=json["currency"],
                idempotency_key=json["idempotency_key"],
            )
            return httpx.Response(
                200,
                json={
                    "status": "verified",
                    "sca_token": create_sca_attestation(
                        subject=context.subject,
                        action_id=action_id,
                        transaction_id=json["transaction_id"],
                        amount=Decimal(json["transaction_amount"]),
                        currency=json["currency"],
                    ),
                },
            )

    monkeypatch.setattr(transfer_service.httpx, "AsyncClient", VerifiedMfaClient)
    result = asyncio.run(
        transfer_service.create_sandbox_transfer(
            SandboxTransferRequest(
                source_account_id="sandbox-checking-001",
                destination_account_id="sandbox-savings-001",
                amount="1000.00",
                confirmed=True,
                otp_code="123456",
            ),
            "sca-success-key",
            _authorization(monkeypatch),
            "sandbox-checking-001",
        )
    )

    assert result.status == "SUCCESS"
    with Session(transfer_audit._engine()) as session:
        success_event = session.query(TransferAuditEvent).filter_by(
            transaction_id="sca-transaction",
            status="SUCCESS",
        ).one()
    assert success_event.consent_state == "GRANTED"
    assert success_event.sca_state == "VERIFIED"
    assert get_sandbox_balance("sandbox-checking-001").balance == Decimal("500.00")
    assert get_sandbox_balance("sandbox-savings-001").balance == Decimal("4200.00")


def test_balance_updates_roll_back_when_success_event_cannot_be_recorded() -> None:
    get_sandbox_balance("sandbox-checking-001")
    engine = transfer_audit._engine()
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TRIGGER reject_success_audit BEFORE INSERT "
            "ON transfer_audit_events WHEN NEW.status = 'SUCCESS' "
            "BEGIN SELECT RAISE(ABORT, 'audit insert rejected'); END"
        )

    with pytest.raises(SandboxBalanceStoreUnavailable):
        execute_sandbox_transfer(
            transaction_id="rollback-transaction",
            account_id="sandbox-checking-001",
            idempotency_key="rollback-key",
            source_account_id="sandbox-checking-001",
            destination_account_id="sandbox-savings-001",
            amount=Decimal("10.00"),
            currency="USD",
            sca_state="NOT_REQUIRED",
        )

    assert get_sandbox_balance("sandbox-checking-001").balance == Decimal("1500.00")
    assert get_sandbox_balance("sandbox-savings-001").balance == Decimal("3200.00")


def test_latest_transfer_returns_account_ids_for_authenticated_customer(monkeypatch):
    monkeypatch.setattr(
        transfer_service,
        "latest_successful_transfer",
        lambda account_id: LatestSuccessfulTransfer(
            source_account_id="sandbox-checking-001",
            destination_account_id="sandbox-savings-001",
        ),
    )

    result = asyncio.run(
        transfer_service.get_latest_transfer("sandbox-checking-001")
    )

    assert result.model_dump() == {
        "source_account_id": "sandbox-checking-001",
        "destination_account_id": "sandbox-savings-001",
    }


def test_latest_transfer_returns_not_found_when_no_success_exists(monkeypatch):
    monkeypatch.setattr(
        transfer_service, "latest_successful_transfer", lambda account_id: None
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            transfer_service.get_latest_transfer("sandbox-checking-001")
        )

    assert error.value.status_code == 404
    assert error.value.detail == "No completed transfer found"


def test_latest_successful_transfer_uses_latest_final_status(monkeypatch):
    engine = create_engine("sqlite://")
    TransferAuditEvent.__table__.create(engine)
    monkeypatch.setattr(transfer_audit, "_engine", lambda: engine)
    now = datetime.now(timezone.utc)

    with Session(engine) as session:
        for event_id, transaction_id, idempotency_key, status, age_minutes, source, destination in [
            (
                "event-old-pending",
                "transaction-old",
                "key-old",
                "PENDING_VALIDATION",
                10,
                "checking",
                "savings",
            ),
            (
                "event-old-success",
                "transaction-old",
                "key-old",
                "SUCCESS",
                9,
                "checking",
                "savings",
            ),
            (
                "event-new-pending",
                "transaction-new",
                "key-new",
                "PENDING_VALIDATION",
                2,
                "savings",
                "checking",
            ),
            (
                "event-new-failed",
                "transaction-new",
                "key-new",
                "FAILED",
                1,
                "savings",
                "checking",
            ),
        ]:
            session.add(
                TransferAuditEvent(
                    event_id=event_id,
                    transaction_id=transaction_id,
                    idempotency_key=idempotency_key,
                    account_id="customer-1",
                    source_account_id=source,
                    destination_account_id=destination,
                    amount=Decimal("10.00"),
                    currency="USD",
                    status=status,
                    created_at=now - timedelta(minutes=age_minutes),
                )
            )
        session.commit()

    result = transfer_audit.latest_successful_transfer("customer-1")

    assert result == LatestSuccessfulTransfer(
        source_account_id="checking",
        destination_account_id="savings",
    )
