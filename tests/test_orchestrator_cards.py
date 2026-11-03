import httpx
from fastapi.testclient import TestClient

from services.authentication import create_access_token, validate_access_token
from services.orchestrator import main as orchestrator_main


def _headers(account_id: str = "sandbox-checking-001") -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(account_id)}"}


def test_orchestrator_lists_cards_for_mapped_sandbox_user(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    monkeypatch.setenv("CARDS_SERVICE_URL", "http://cards.test")

    class FakeCardsClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def get(self, url, headers):
            assert url == "http://cards.test/sandbox/cards"
            assert validate_access_token(headers["Authorization"].removeprefix("Bearer ")) == (
                "usr_sandbox_01"
            )
            return httpx.Response(
                200,
                request=httpx.Request("GET", url),
                json=[{"card_id": "card_debit_4321", "status": "active"}],
            )

    monkeypatch.setattr(orchestrator_main.httpx, "AsyncClient", FakeCardsClient)

    response = TestClient(orchestrator_main.app).get(
        "/sandbox/cards",
        headers=_headers(),
    )

    assert response.status_code == 200
    assert response.json() == [{"card_id": "card_debit_4321", "status": "active"}]


def test_orchestrator_freeze_forwards_otp_to_cards_service(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    monkeypatch.setenv("CARDS_SERVICE_URL", "http://cards.test")

    class FakeCardsClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def post(self, url, headers, json):
            assert url == "http://cards.test/sandbox/cards/card_debit_4321/freeze"
            assert validate_access_token(headers["Authorization"].removeprefix("Bearer ")) == (
                "usr_sandbox_01"
            )
            assert json == {"otp_code": "123456"}
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={"card_id": "card_debit_4321", "status": "frozen"},
            )

    monkeypatch.setattr(orchestrator_main.httpx, "AsyncClient", FakeCardsClient)

    response = TestClient(orchestrator_main.app).post(
        "/sandbox/cards/card_debit_4321/freeze",
        headers=_headers(),
        json={"otp_code": "123456"},
    )

    assert response.status_code == 200
    assert response.json() == {"card_id": "card_debit_4321", "status": "frozen"}


def test_orchestrator_card_routes_reject_unmapped_subject(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")

    response = TestClient(orchestrator_main.app).get(
        "/sandbox/cards",
        headers=_headers("unmapped-account"),
    )

    assert response.status_code == 403
