from fastapi.testclient import TestClient

from services.mfa.main import app


def test_sandbox_mfa_accepts_demo_otp(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    client = TestClient(app, client=("127.0.0.1", 50000))

    response = client.post(
        "/verify-otp",
        json={
            "account_id": "sandbox-checking-001",
            "otp_code": "123456",
            "transaction_amount": "1000.00",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "verified"}


def test_sandbox_mfa_rejects_invalid_otp(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    client = TestClient(app, client=("127.0.0.1", 50000))

    response = client.post(
        "/verify-otp",
        json={
            "account_id": "sandbox-checking-001",
            "otp_code": "654321",
            "transaction_amount": "1000.00",
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
        },
    )

    assert response.status_code == 403