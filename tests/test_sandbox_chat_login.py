import os
from pathlib import Path
from uuid import uuid4

import jwt
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from services import load_local_environment
from services.orchestrator import main as orchestrator_main
from services.orchestrator.memory import ConversationNotFound


def test_load_local_environment_reads_repo_dotenv(monkeypatch, tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "SANDBOX_MODE=true\n"
        "LOCAL_TEST_AUTH_ENABLED=true\n"
        "JWT_SECRET=test-secret-key-with-at-least-32-bytes\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.delenv("LOCAL_TEST_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)

    load_local_environment(tmp_path)

    assert os.environ["SANDBOX_MODE"] == "true"
    assert os.environ["LOCAL_TEST_AUTH_ENABLED"] == "true"
    assert os.environ["JWT_SECRET"] == "test-secret-key-with-at-least-32-bytes"


def test_sandbox_test_login_is_disabled_without_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)
    monkeypatch.delenv("LOCAL_TEST_AUTH_ENABLED", raising=False)

    response = TestClient(orchestrator_main.app).post("/sandbox/test-login")

    assert response.status_code == 404


def test_sandbox_test_login_issues_http_only_cookie_for_loopback(monkeypatch) -> None:
    secret = "test-secret-key-with-at-least-32-bytes"
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("LOCAL_TEST_AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", secret)
    client = TestClient(orchestrator_main.app, client=("127.0.0.1", 50000))

    response = client.post("/sandbox/test-login")

    assert response.status_code == 200
    assert response.json() == {
        "account_id": "sandbox-checking-001",
        "expires_in": 1800,
    }
    assert "httponly" in response.headers["set-cookie"].lower()
    token = response.cookies[orchestrator_main.TEST_SESSION_COOKIE]
    claims = jwt.decode(
        token,
        secret,
        algorithms=["HS256"],
        issuer="banking-auth",
        audience="banking-api",
    )
    assert claims["sub"] == "sandbox-checking-001"


def test_sandbox_test_login_rejects_non_loopback_clients(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("LOCAL_TEST_AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    client = TestClient(orchestrator_main.app, client=("192.0.2.10", 50000))

    response = client.post("/sandbox/test-login")

    assert response.status_code == 404


def test_sandbox_test_login_rejects_unconfigured_browser_origin(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("LOCAL_TEST_AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173")
    client = TestClient(orchestrator_main.app, client=("127.0.0.1", 50000))

    response = client.post(
        "/sandbox/test-login", headers={"origin": "http://untrusted.local"}
    )

    assert response.status_code == 404


def test_websocket_accepts_local_sandbox_cookie(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("LOCAL_TEST_AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173")
    observed_history = []
    conversations = {}
    thread_owners = {}

    class FakeConversationStore:
        async def resolve_thread(self, owner_id, requested_thread_id):
            if requested_thread_id is None:
                thread_id = uuid4()
                thread_owners[thread_id] = owner_id
                conversations[thread_id] = []
                return thread_id
            if thread_owners.get(requested_thread_id) != owner_id:
                raise ConversationNotFound
            return requested_thread_id

        async def load(self, thread_id):
            return list(conversations[thread_id])

        async def append(self, thread_id, messages):
            conversations[thread_id].extend(messages)

    fake_store = FakeConversationStore()
    monkeypatch.setattr(
        orchestrator_main,
        "_conversation_store",
        lambda app_state: fake_store,
    )

    async def fake_stream_response(
        token: str,
        account_id: str,
        content: str,
        cache=None,
        account_choice=None,
        guardrails=None,
        request_path="/chat",
        prior_messages=None,
        turn_messages=None,
    ):
        assert account_id == "sandbox-checking-001"
        assert account_choice is None
        observed_history.append(
            [message.content for message in prior_messages]
        )
        turn_messages.append(HumanMessage(content=content))
        yield {"type": "content", "content": "Your test "}
        yield {"type": "content", "content": "balance is available."}
        yield {"type": "completion"}

    monkeypatch.setattr(orchestrator_main, "_stream_response", fake_stream_response)
    client = TestClient(orchestrator_main.app, client=("127.0.0.1", 50000))
    login = client.post("/sandbox/test-login")
    assert login.status_code == 200

    with client.websocket_connect(
        "/chat", headers={"origin": "http://localhost:5173"}
    ) as websocket:
        websocket.send_json(
            {"type": "message", "content": "What is my balance?"}
        )
        assert websocket.receive_json() == {
            "type": "content",
            "content": "Your test ",
        }
        assert websocket.receive_json() == {
            "type": "content",
            "content": "balance is available.",
        }
        completion = websocket.receive_json()
        assert completion["type"] == "completion"
        thread_id = completion["thread_id"]
        websocket.send_json(
            {"type": "message", "content": "What was the earlier question?"}
        )
        assert websocket.receive_json()["type"] == "content"
        assert websocket.receive_json()["type"] == "content"
        assert websocket.receive_json()["thread_id"] == thread_id

    with client.websocket_connect(
        "/chat", headers={"origin": "http://localhost:5173"}
    ) as websocket:
        websocket.send_json(
            {
                "type": "message",
                "content": "Continue our conversation.",
                "thread_id": thread_id,
            }
        )
        assert websocket.receive_json()["type"] == "content"
        assert websocket.receive_json()["type"] == "content"
        assert websocket.receive_json()["thread_id"] == thread_id

    assert observed_history == [
        [],
        [
            "What is my balance?",
            "Your test balance is available.",
        ],
        [
            "What is my balance?",
            "Your test balance is available.",
            "What was the earlier question?",
            "Your test balance is available.",
        ],
    ]