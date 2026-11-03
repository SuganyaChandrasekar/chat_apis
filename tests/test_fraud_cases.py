from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from services.authentication import create_access_token
from services.fraud import main as fraud_main
from services.fraud import store as fraud_store
from services.fraud.store import FraudBase, FraudCase
from services.cards.store import CardRecord
from services.orchestrator.pii import PiiMaskingUnavailable


def test_fraud_case_creation_is_authenticated_and_idempotent(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    engine = create_engine(f"sqlite:///{tmp_path / 'fraud.db'}")
    FraudBase.metadata.create_all(engine)
    monkeypatch.setattr(fraud_store, "_engine", lambda: engine)
    monkeypatch.setattr(fraud_main, "anonymize_message", lambda value: value)

    def get_frozen_owned_card(user_id, card_id):
        assert user_id == "usr_sandbox_01"
        assert card_id == "card_debit_4321"
        return CardRecord(
            card_id=card_id,
            user_id=user_id,
            card_name="Albion Debit Card",
            last_4="4321",
            status="frozen",
        )

    monkeypatch.setattr(fraud_main, "get_owned_card", get_frozen_owned_card)
    token = create_access_token("sandbox-checking-001")
    client = TestClient(fraud_main.app)
    request = {
        "reason": "Someone stole money from me",
        "frozen_card_ids": ["card_debit_4321"],
    }

    first = client.post(
        "/fraud/cases",
        headers={
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": "report-1",
        },
        json=request,
    )
    monkeypatch.setattr(fraud_main, "get_owned_card", lambda user_id, card_id: None)
    retry = client.post(
        "/fraud/cases",
        headers={
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": "report-1",
        },
        json=request,
    )

    assert first.status_code == retry.status_code == 201
    assert first.json()["case_id"].startswith("FRD-")
    assert retry.json()["case_id"] == first.json()["case_id"]
    assert first.json()["user_id"] == "sandbox-checking-001"
    assert first.json()["status"] == "OPEN"
    assert first.json()["frozen_card_ids"] == ["card_debit_4321"]
    assert first.json()["idempotency_key"] == "report-1"
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(FraudCase)) == 1
    engine.dispose()


def test_fraud_case_rejects_reuse_of_key_with_different_report(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    engine = create_engine(f"sqlite:///{tmp_path / 'fraud-conflict.db'}")
    FraudBase.metadata.create_all(engine)
    monkeypatch.setattr(fraud_store, "_engine", lambda: engine)
    monkeypatch.setattr(fraud_main, "anonymize_message", lambda value: value)
    token = create_access_token("sandbox-checking-001")
    client = TestClient(fraud_main.app)

    first = client.post(
        "/fraud/cases",
        headers={
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": "report-1",
        },
        json={"reason": "I was scammed", "frozen_card_ids": []},
    )
    conflict = client.post(
        "/fraud/cases",
        headers={
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": "report-1",
        },
        json={"reason": "Different report", "frozen_card_ids": []},
    )

    assert first.status_code == 201
    assert conflict.status_code == 409
    engine.dispose()


def test_fraud_case_service_is_sandbox_only(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "false")
    response = TestClient(fraud_main.app).post(
        "/fraud/cases",
        headers={"Authorization": "Bearer invalid"},
        json={"reason": "I was scammed"},
    )

    assert response.status_code == 404


def test_fraud_case_rejects_card_ids_not_frozen_for_authenticated_owner(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    token = create_access_token("sandbox-checking-001")
    monkeypatch.setattr(fraud_main, "anonymize_message", lambda value: value)
    monkeypatch.setattr(fraud_main, "get_owned_card", lambda user_id, card_id: None)

    response = TestClient(fraud_main.app).post(
        "/fraud/cases",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "reason": "I was scammed",
            "frozen_card_ids": ["another-customer-card"],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid frozen card identifier"


def test_fraud_case_fails_closed_when_pii_masking_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    token = create_access_token("sandbox-checking-001")

    def fail_masking(value):
        raise PiiMaskingUnavailable

    monkeypatch.setattr(fraud_main, "anonymize_message", fail_masking)
    response = TestClient(fraud_main.app).post(
        "/fraud/cases",
        headers={"Authorization": f"Bearer {token}"},
        json={"reason": "Someone stole money from me"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Fraud report processing is unavailable"
