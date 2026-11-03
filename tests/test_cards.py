import asyncio

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from services.authentication import create_access_token
from services.cards import store as cards_store
from services.cards import main as cards_main
from services.cards.store import Card, CardBase


@pytest.fixture(autouse=True)
def seed_sandbox_cards(monkeypatch, tmp_path):
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    monkeypatch.setenv("SANDBOX_MODE", "true")
    engine = create_engine(f"sqlite:///{tmp_path / 'cards.db'}")
    CardBase.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                Card(
                    card_id="card_debit_4321",
                    user_id="usr_sandbox_01",
                    card_name="Albion Debit Card",
                    last_4="4321",
                    status="active",
                ),
                Card(
                    card_id="card_credit_8899",
                    user_id="usr_sandbox_01",
                    card_name="Albion Credit Card",
                    last_4="8899",
                    status="active",
                ),
            ]
        )
        session.commit()
    monkeypatch.setattr(cards_store, "_engine", lambda: engine)
    yield
    engine.dispose()


def _authorization(user_id: str = "usr_sandbox_01") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {create_access_token(user_id)}",
    }


def _card_status(card_id: str) -> str:
    with Session(cards_store._engine()) as session:
        return session.scalar(
            select(Card.status).where(Card.card_id == card_id)
        )


def test_card_list_is_limited_to_authenticated_owner(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    client = TestClient(cards_main.app)

    response = client.get("/sandbox/cards", headers=_authorization())
    other_user_response = client.get(
        "/sandbox/cards",
        headers=_authorization("usr_other"),
    )

    assert response.status_code == 200
    assert {card["card_id"] for card in response.json()} == {
        "card_debit_4321",
        "card_credit_8899",
    }
    assert other_user_response.json() == []


def test_card_freeze_requires_mfa_before_status_change(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    client = TestClient(cards_main.app)

    response = client.post(
        "/sandbox/cards/card_debit_4321/freeze",
        headers=_authorization(),
        json={},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "MFA verification is required"
    assert _card_status("card_debit_4321") == "active"


def test_card_freeze_rejects_card_not_owned_by_authenticated_user(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    client = TestClient(cards_main.app)

    response = client.post(
        "/sandbox/cards/card_debit_4321/freeze",
        headers=_authorization("usr_other"),
        json={"otp_code": "123456"},
    )

    assert response.status_code == 404
    assert _card_status("card_debit_4321") == "active"


def test_card_freeze_rejects_unverified_otp(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")

    class FakeMfaClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def post(self, url, headers, json):
            return httpx.Response(
                401,
                request=httpx.Request("POST", url),
                json={"detail": "MFA verification failed"},
            )

    monkeypatch.setattr(cards_main.httpx, "AsyncClient", FakeMfaClient)
    client = TestClient(cards_main.app)

    response = client.post(
        "/sandbox/cards/card_debit_4321/freeze",
        headers=_authorization(),
        json={"otp_code": "654321"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "MFA verification failed"
    assert _card_status("card_debit_4321") == "active"


def test_card_freeze_verifies_otp_persists_status_and_returns_confirmation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")

    class FakeMfaClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def post(self, url, headers, json):
            assert url == "http://localhost:8003/verify-otp"
            assert headers == _authorization()
            assert json == {
                "account_id": "usr_sandbox_01",
                "otp_code": "123456",
                "purpose": "card_freeze",
            }
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={"status": "verified"},
            )

    monkeypatch.setattr(cards_main.httpx, "AsyncClient", FakeMfaClient)
    client = TestClient(cards_main.app)

    response = client.post(
        "/sandbox/cards/card_debit_4321/freeze",
        headers=_authorization(),
        json={"otp_code": "123456"},
    )
    cards_response = client.get("/sandbox/cards", headers=_authorization())
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "frozen"
    assert payload["card_name"] == "Albion Debit Card"
    assert payload["last_4"] == "4321"
    assert payload["timestamp"]
    assert payload["reference_id"]
    assert payload["reference_id"] in payload["message"]
    assert _card_status("card_debit_4321") == "frozen"
    assert next(
        card["status"]
        for card in cards_response.json()
        if card["card_id"] == "card_debit_4321"
    ) == "frozen"


def test_fraud_workflow_freezes_owned_card_without_user_mfa() -> None:
    response = asyncio.run(
        cards_main.freeze_card_for_fraud("usr_sandbox_01", "card_credit_8899")
    )

    assert response.status == "frozen"
    assert response.card_name == "Albion Credit Card"
    assert response.last_4 == "8899"
    assert _card_status("card_credit_8899") == "frozen"


def test_fraud_workflow_bypass_is_unavailable_outside_sandbox(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "false")

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            cards_main.freeze_card_for_fraud("usr_sandbox_01", "card_credit_8899")
        )

    assert error.value.status_code == 404
    assert _card_status("card_credit_8899") == "active"


def test_sandbox_reset_restores_owned_card_to_active() -> None:
    cards_store.set_card_status("usr_sandbox_01", "card_debit_4321", "frozen")
    client = TestClient(cards_main.app)

    response = client.post(
        "/sandbox/cards/card_debit_4321/reset",
        headers=_authorization(),
    )
    cards_response = client.get("/sandbox/cards", headers=_authorization())

    assert response.status_code == 200
    assert response.json()["status"] == "active"
    assert next(
        card["status"]
        for card in cards_response.json()
        if card["card_id"] == "card_debit_4321"
    ) == "active"


def test_sandbox_reset_rejects_card_not_owned_by_authenticated_user() -> None:
    cards_store.set_card_status("usr_sandbox_01", "card_debit_4321", "frozen")
    client = TestClient(cards_main.app)

    response = client.post(
        "/sandbox/cards/card_debit_4321/reset",
        headers=_authorization("usr_other"),
    )

    assert response.status_code == 404
    assert _card_status("card_debit_4321") == "frozen"
