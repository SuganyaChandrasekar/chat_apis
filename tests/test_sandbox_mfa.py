from fastapi.testclient import TestClient

from services.authentication import create_access_token
from services.mfa.main import app
from services.authorization import card_freeze_action_id, transfer_action_id

SECRET = "sandbox-mfa-test-secret-with-32-bytes"


def _transfer_context_token(
    *,
    account_id: str,
    source_account_id: str,
    destination_account_id: str,
    amount: str,
    idempotency_key: str,
) -> str:
    from decimal import Decimal

    action_id = transfer_action_id(
        account_id=account_id,
        source_account_id=source_account_id,
        destination_account_id=destination_account_id,
        amount=Decimal(amount),
        currency="USD",
        idempotency_key=idempotency_key,
    )
    return create_access_token(
        account_id,
        scopes=("transfer:execute",),
        consent_state="granted",
        consent_action=action_id,
        sca_state="required",
        token_kind="delegated",
    )


def test_sandbox_mfa_accepts_demo_otp(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)
    token = _transfer_context_token(
        account_id="sandbox-checking-001",
        source_account_id="sandbox-checking-001",
        destination_account_id="sandbox-savings-001",
        amount="1000.00",
        idempotency_key="mfa-test",
    )
    client = TestClient(app, client=("127.0.0.1", 50000))

    response = client.post(
        "/verify-otp",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "account_id": "sandbox-checking-001",
            "otp_code": "123456",
            "transaction_amount": "1000.00",
            "transaction_id": "transaction-1",
            "source_account_id": "sandbox-checking-001",
            "destination_account_id": "sandbox-savings-001",
            "idempotency_key": "mfa-test",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "verified"
    assert response.json()["sca_token"]


def test_sandbox_mfa_accepts_card_freeze_purpose_without_transfer_amount(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)
    card_action = card_freeze_action_id(
        subject="usr_sandbox_01", card_id="card-1"
    )
    token = create_access_token(
        "usr_sandbox_01",
        scopes=("cards:freeze",),
        consent_state="granted",
        consent_action=card_action,
        sca_state="required",
        token_kind="delegated",
    )
    client = TestClient(app, client=("127.0.0.1", 50000))

    response = client.post(
        "/verify-otp",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "account_id": "usr_sandbox_01",
            "otp_code": "123456",
            "purpose": "card_freeze",
            "transaction_id": "card-1",
            "action_id": card_action,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "verified"
    assert response.json()["sca_token"]


def test_sandbox_mfa_rejects_invalid_otp(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", SECRET)
    token = _transfer_context_token(
        account_id="sandbox-checking-001",
        source_account_id="sandbox-checking-001",
        destination_account_id="sandbox-savings-001",
        amount="1000.00",
        idempotency_key="mfa-invalid",
    )
    client = TestClient(app, client=("127.0.0.1", 50000))

    response = client.post(
        "/verify-otp",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "account_id": "sandbox-checking-001",
            "otp_code": "654321",
            "transaction_amount": "1000.00",
            "transaction_id": "transaction-1",
            "source_account_id": "sandbox-checking-001",
            "destination_account_id": "sandbox-savings-001",
            "idempotency_key": "mfa-invalid",
        },
    )

    assert response.status_code == 401


def test_sandbox_mfa_is_hidden_when_sandbox_mode_is_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "false")
    client = TestClient(app, client=("127.0.0.1", 50000))

    response = client.post(
        "/verify-otp",
        json={
            "account_id": "sandbox-checking-001",
            "otp_code": "123456",
            "transaction_amount": "1000.00",
            "transaction_id": "transaction-1",
            "source_account_id": "sandbox-checking-001",
            "destination_account_id": "sandbox-savings-001",
            "idempotency_key": "mfa-disabled",
        },
    )

    assert response.status_code == 404


def test_sandbox_mfa_rejects_non_loopback_clients(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    client = TestClient(app, client=("192.0.2.1", 50000))

    response = client.post(
        "/verify-otp",
        json={
            "account_id": "sandbox-checking-001",
            "otp_code": "123456",
            "transaction_amount": "1000.00",
            "transaction_id": "transaction-1",
            "source_account_id": "sandbox-checking-001",
            "destination_account_id": "sandbox-savings-001",
            "idempotency_key": "mfa-non-loopback",
        },
    )

    assert response.status_code == 403