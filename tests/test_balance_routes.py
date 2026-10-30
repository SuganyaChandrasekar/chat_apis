import asyncio
from decimal import Decimal
from time import time
from types import SimpleNamespace

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient
from uuid import uuid4
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from services import sandbox
from services.balance.contracts import AuthenticatedCustomer as BalanceCustomer
from services.balance.main import (
    app as balance_app,
    get_authenticated_customer as get_balance_customer,
    get_balance_provider,
)
from services.orchestrator.contracts import AuthenticatedCustomer as OrchestratorCustomer
from services.orchestrator import main as orchestrator_main
from services.orchestrator import guardrails as orchestrator_guardrails
from services.orchestrator import agent as orchestrator_agent
from services.orchestrator.main import (
    app as orchestrator_app,
    get_authenticated_customer as get_orchestrator_customer,
    get_balance_service_client,
)
from services.transfer import main as transfer_service
from services.transfer.audit import TransferStart


async def _allow_advice_boundary(*args, **kwargs):
    return {
        "is_blocked": False,
        "reason": "No regulated advice request.",
        "refusal_message": orchestrator_guardrails.ADVICE_BOUNDARY_REFUSAL_MESSAGE,
    }


@pytest.fixture(autouse=True)
def classify_other_intents_by_default(monkeypatch):
    async def classify_intent(rails, content, context):
        return orchestrator_main.ChatIntentClassification(intent="other")

    monkeypatch.setattr(
        orchestrator_main, "_classify_chat_intent", classify_intent
    )


def _intent_classifier(intent, **entities):
    async def classify_intent(rails, content, context):
        return orchestrator_main.ChatIntentClassification(
            intent=intent,
            **entities,
        )

    return classify_intent


def test_balance_service_fails_closed_without_authentication() -> None:
    response = TestClient(balance_app).get("/balance")

    assert response.status_code == 503
    assert response.json() == {"detail": "Customer authentication is not configured"}


def test_balance_service_passes_through_test_provider_payload() -> None:
    customer = BalanceCustomer(subject="verified-subject")
    mock_payload = {"test_only": True, "balance": "NOT_A_REAL_BALANCE"}

    class Provider:
        async def get_balance(self, requested_customer: BalanceCustomer) -> dict[str, str]:
            assert requested_customer is customer
            return mock_payload

    balance_app.dependency_overrides[get_balance_customer] = lambda: customer
    balance_app.dependency_overrides[get_balance_provider] = Provider
    try:
        response = TestClient(balance_app).get("/balance")
    finally:
        balance_app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == mock_payload


def test_balance_service_fails_closed_without_data_provider() -> None:
    customer = BalanceCustomer(subject="verified-subject")
    balance_app.dependency_overrides[get_balance_customer] = lambda: customer
    try:
        response = TestClient(balance_app).get("/balance")
    finally:
        balance_app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "Balance retrieval is not configured"}


def test_orchestrator_fails_closed_without_authentication() -> None:
    response = TestClient(orchestrator_app).get("/balance")

    assert response.status_code == 503
    assert response.json() == {"detail": "Customer authentication is not configured"}


def test_orchestrator_routes_verified_identity_to_balance_client() -> None:
    customer = OrchestratorCustomer(subject="verified-subject")

    class Client:
        async def get_balance(
            self, requested_customer: OrchestratorCustomer
        ) -> dict[str, str]:
            assert requested_customer is customer
            return {"result": "balance-service-value"}

    orchestrator_app.dependency_overrides[get_orchestrator_customer] = lambda: customer
    orchestrator_app.dependency_overrides[get_balance_service_client] = Client
    try:
        response = TestClient(orchestrator_app).get("/balance")
    finally:
        orchestrator_app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"result": "balance-service-value"}


def test_orchestrator_fails_closed_without_balance_service_client() -> None:
    customer = OrchestratorCustomer(subject="verified-subject")
    orchestrator_app.dependency_overrides[get_orchestrator_customer] = lambda: customer
    try:
        response = TestClient(orchestrator_app).get("/balance")
    finally:
        orchestrator_app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Balance service integration is not configured"
    }


def test_sandbox_endpoints_are_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SANDBOX_MODE", raising=False)

    assert TestClient(balance_app).get(
        "/sandbox/balances/sandbox-checking-001"
    ).status_code == 404
    assert TestClient(transfer_service.app).post(
        "/transfer",
        json={
            "source_account_id": "sandbox-checking-001",
            "destination_account_id": "sandbox-savings-001",
            "amount": "10.00",
        },
    ).status_code == 404


def test_sandbox_balance_returns_fixture_account(monkeypatch) -> None:
    secret = "test-secret-key-with-at-least-32-bytes"
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", secret)
    token = jwt.encode(
        {
            "sub": "sandbox-checking-001",
            "iat": int(time()),
            "exp": time() + 300,
            "iss": "banking-auth",
            "aud": "banking-api",
            "typ": "access",
        },
        secret,
        algorithm="HS256",
    )

    response = TestClient(balance_app).get(
        "/sandbox/balances/sandbox-checking-001",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "account_id": "sandbox-checking-001",
        "balance": "1500.00",
        "currency": "USD",
    }


def test_sandbox_balance_rejects_account_not_authorized_for_customer(monkeypatch) -> None:
    secret = "test-secret-key-with-at-least-32-bytes"
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", secret)
    token = jwt.encode(
        {
            "sub": "sandbox-checking-001",
            "iat": int(time()),
            "exp": time() + 300,
            "iss": "banking-auth",
            "aud": "banking-api",
            "typ": "access",
        },
        secret,
        algorithm="HS256",
    )

    response = TestClient(balance_app).get(
        "/sandbox/balances/sandbox-unauthorized-001",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Account access is not authorized"}


def test_sandbox_customer_can_list_and_check_multiple_accounts(monkeypatch) -> None:
    secret = "test-secret-key-with-at-least-32-bytes"
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", secret)
    monkeypatch.setenv("SANDBOX_FROZEN_ACCOUNT_IDS", "sandbox-savings-001")
    token = jwt.encode(
        {
            "sub": "sandbox-checking-001",
            "iat": int(time()),
            "exp": time() + 300,
            "iss": "banking-auth",
            "aud": "banking-api",
            "typ": "access",
        },
        secret,
        algorithm="HS256",
    )
    client = TestClient(balance_app)
    headers = {"Authorization": f"Bearer {token}"}

    accounts_response = client.get("/sandbox/accounts", headers=headers)
    savings_response = client.get(
        "/sandbox/balances/sandbox-savings-001", headers=headers
    )

    assert accounts_response.status_code == 200
    assert accounts_response.json() == [
        {
            "account_id": "sandbox-checking-001",
            "label": "Checking account",
            "status": "ACTIVE",
        },
        {
            "account_id": "sandbox-savings-001",
            "label": "Savings account",
            "status": "FROZEN",
        },
    ]
    assert savings_response.status_code == 200
    assert savings_response.json() == {
        "account_id": "sandbox-savings-001",
        "balance": "3200.00",
        "currency": "USD",
    }


def test_sandbox_account_list_rejects_unknown_customer(monkeypatch) -> None:
    secret = "test-secret-key-with-at-least-32-bytes"
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", secret)
    token = jwt.encode(
        {
            "sub": "unknown-customer",
            "iat": int(time()),
            "exp": time() + 300,
            "iss": "banking-auth",
            "aud": "banking-api",
            "typ": "access",
        },
        secret,
        algorithm="HS256",
    )

    response = TestClient(balance_app).get(
        "/sandbox/accounts", headers={"Authorization": f"Bearer {token}"}
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Account access is not authorized"}


def test_transfer_rejects_missing_bearer_token(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")

    response = TestClient(transfer_service.app).post(
        "/transfer",
        headers={"Idempotency-Key": "transfer-no-auth"},
        json={
            "source_account_id": "sandbox-checking-001",
            "destination_account_id": "sandbox-savings-001",
            "amount": "5.00",
            "confirmed": True,
            "otp_code": "123456",
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid bearer token"}


def test_transfer_rejects_unverified_otp_without_mutating_balance(monkeypatch) -> None:
    secret = "test-secret-key-with-at-least-32-bytes"
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", secret)
    monkeypatch.setattr(
        transfer_service, "balances", sandbox.SANDBOX_INITIAL_BALANCES.copy()
    )
    monkeypatch.setattr(
        transfer_service,
        "begin_transfer",
        lambda **kwargs: TransferStart("failed-otp-transaction", "PENDING_VALIDATION", False),
    )
    audit_events = []
    monkeypatch.setattr(
        transfer_service,
        "append_transfer_status",
        lambda **kwargs: audit_events.append(kwargs["status"]),
    )

    class FakeAsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def post(self, url, headers, json):
            return httpx.Response(401, json={"status": "invalid"})

    monkeypatch.setattr(transfer_service.httpx, "AsyncClient", FakeAsyncClient)
    token = jwt.encode(
        {
            "sub": "sandbox-checking-001",
            "iat": int(time()),
            "exp": time() + 300,
            "iss": "banking-auth",
            "aud": "banking-api",
            "typ": "access",
        },
        secret,
        algorithm="HS256",
    )
    response = TestClient(transfer_service.app).post(
        "/transfer",
        headers={
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": "transfer-invalid-otp",
        },
        json={
            "source_account_id": "sandbox-checking-001",
            "destination_account_id": "sandbox-savings-001",
            "amount": "1000.00",
            "confirmed": True,
            "otp_code": "654321",
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "MFA verification failed"}
    assert audit_events == ["FAILED"]
    assert transfer_service.balances["sandbox-checking-001"] == Decimal("1500.00")


def test_sandbox_transfer_returns_unique_ids_and_checks_funds(monkeypatch) -> None:
    secret = "test-secret-key-with-at-least-32-bytes"
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", secret)
    monkeypatch.setattr(
        transfer_service, "balances", sandbox.SANDBOX_INITIAL_BALANCES.copy()
    )
    transaction_ids = iter(["transaction-one", "transaction-two", "transaction-three"])
    audit_events = []

    def begin_transfer(**kwargs):
        return TransferStart(next(transaction_ids), "PENDING_VALIDATION", False)

    def append_transfer_status(**kwargs):
        audit_events.append(kwargs["status"])

    class FakeAsyncClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def post(self, url, headers, json):
            assert url == "http://localhost:8003/verify-otp"
            assert headers == {"Authorization": f"Bearer {token}"}
            assert json == {
                "account_id": "sandbox-checking-001",
                "otp_code": "123456",
                "transaction_amount": "10.00",
            }
            return httpx.Response(
                200,
                json={"status": "verified", "message": "OTP verified successfully"},
            )

    monkeypatch.setattr(transfer_service, "begin_transfer", begin_transfer)
    monkeypatch.setattr(transfer_service, "append_transfer_status", append_transfer_status)
    monkeypatch.setattr(transfer_service.httpx, "AsyncClient", FakeAsyncClient)
    token = jwt.encode(
        {
            "sub": "sandbox-checking-001",
            "iat": int(time()),
            "exp": time() + 300,
            "iss": "banking-auth",
            "aud": "banking-api",
            "typ": "access",
        },
        secret,
        algorithm="HS256",
    )
    client = TestClient(transfer_service.app)
    payload = {
        "source_account_id": "sandbox-checking-001",
        "destination_account_id": "sandbox-savings-001",
        "amount": "10.00",
        "currency": "USD",
        "confirmed": True,
    }
    headers = {"Authorization": f"Bearer {token}"}

    first = client.post(
        "/transfer", json=payload, headers={**headers, "Idempotency-Key": "transfer-1"}
    )
    second = client.post(
        "/transfer", json=payload, headers={**headers, "Idempotency-Key": "transfer-2"}
    )
    insufficient = client.post(
        "/transfer",
        json={**payload, "amount": "2000.00"},
        headers={**headers, "Idempotency-Key": "transfer-3"},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["transaction_id"] != second.json()["transaction_id"]
    assert first.json()["status"] == "SUCCESS"
    assert insufficient.status_code == 409
    assert insufficient.json() == {
        "detail": "INSUFFICIENT_FUNDS: Available balance is USD 1480.00."
    }
    assert audit_events == ["SUCCESS", "SUCCESS", "FAILED"]
    assert transfer_service.balances["sandbox-checking-001"] == Decimal("1480.00")


def test_chat_sse_requires_a_valid_bearer_token(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")

    response = TestClient(orchestrator_app).post(
        "/chat/stream",
        json={"type": "message", "content": "Show my balance"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid bearer token"}


def test_orchestrator_forwards_transfer_with_jwt_and_idempotency(monkeypatch) -> None:
    secret = "test-secret-key-with-at-least-32-bytes"
    monkeypatch.setenv("JWT_SECRET", secret)
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("TRANSFER_SERVICE_URL", "http://transfer.test")
    token = jwt.encode(
        {
            "sub": "sandbox-checking-001",
            "iat": int(time()),
            "exp": time() + 300,
            "iss": "banking-auth",
            "aud": "banking-api",
            "typ": "access",
        },
        secret,
        algorithm="HS256",
    )

    class FakeAsyncClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def post(self, url, headers, json):
            assert url == "http://transfer.test/transfer"
            assert headers == {
                "Authorization": f"Bearer {token}",
                "Idempotency-Key": "request-123",
            }
            assert json["source_account_id"] == "sandbox-savings-001"
            assert json["confirmed"] is True
            return httpx.Response(200, json={"transaction_id": "transaction-123"})

    monkeypatch.setattr(orchestrator_main.httpx, "AsyncClient", FakeAsyncClient)
    client = TestClient(orchestrator_app)
    client.cookies.set(orchestrator_main.TEST_SESSION_COOKIE, token)
    response = client.post(
        "/transfer",
        headers={
            "Idempotency-Key": "request-123",
        },
        json={
            "source_account_id": "sandbox-savings-001",
            "destination_account_id": "sandbox-checking-001",
            "amount": "25.00",
            "currency": "USD",
            "confirmed": True,
            "otp_code": "123456",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"transaction_id": "transaction-123"}


def test_app_lifespan_initializes_guardrails_before_requests(monkeypatch) -> None:
    initialized = False
    guardrails = object()

    def initialize_guardrails():
        nonlocal initialized
        initialized = True
        return guardrails

    monkeypatch.setattr(
        orchestrator_main, "initialize_guardrails", initialize_guardrails
    )

    async def run_lifespan() -> None:
        async with orchestrator_main.lifespan(orchestrator_app):
            assert initialized
            assert orchestrator_app.state.guardrails is guardrails

    asyncio.run(run_lifespan())


def test_chat_sse_masks_message_and_streams_graph_events(monkeypatch) -> None:
    monkeypatch.setattr(
        orchestrator_main, "check_advice_boundary", _allow_advice_boundary
    )
    secret = "test-secret-key-with-at-least-32-bytes"
    monkeypatch.setenv("JWT_SECRET", secret)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    token = jwt.encode(
        {
            "sub": "trusted-account",
            "iat": int(time()),
            "exp": time() + 300,
            "iss": "banking-auth",
            "aud": "banking-api",
            "typ": "access",
        },
        secret,
        algorithm="HS256",
    )
    def fake_anonymize_message(content):
        return content.replace("555-0100", "<PHONE_NUMBER>")

    async def fake_check_input(
        rails, content, user_id, request_path="/chat", context=()
    ):
        assert "<PHONE_NUMBER>" in content
        return content

    async def fake_check_output(
        rails, content, user_id, request_path="/chat", context=()
    ):
        return content

    monkeypatch.setattr(
        "services.orchestrator.memory.anonymize_message", fake_anonymize_message
    )
    monkeypatch.setattr(orchestrator_main, "check_input", fake_check_input)
    monkeypatch.setattr(orchestrator_main, "check_output", fake_check_output)
    monkeypatch.setattr(
        orchestrator_main,
        "build_agent_graph",
        lambda access_token, account_id: {"token": access_token},
    )

    async def fake_stream_agent_events(graph, state):
        assert graph["token"] == token
        assert state["account_id"] == "trusted-account"
        assert state["messages"][-1].content == "Call me at <PHONE_NUMBER>"
        yield {"type": "content", "content": "Your balance is available."}
        yield {"type": "completion"}

    monkeypatch.setattr(
        orchestrator_main, "stream_agent_events", fake_stream_agent_events
    )
    class FakeConversationStore:
        async def resolve_thread(self, owner_id, requested_thread_id):
            assert owner_id == "trusted-account"
            return requested_thread_id or uuid4()

        async def load(self, thread_id):
            return []

        async def append(self, thread_id, messages):
            self.persisted = messages

    monkeypatch.setattr(
        orchestrator_main,
        "_conversation_store",
        lambda app_state: FakeConversationStore(),
    )
    orchestrator_app.state.faq_cache = None
    response = TestClient(orchestrator_app).post(
        "/chat/stream",
        headers={"Authorization": f"Bearer {token}"},
        json={"type": "message", "content": "Call me at 555-0100"},
    )

    assert response.status_code == 200
    assert '"type": "content"' in response.text
    assert '"content": "Your balance is available."' in response.text
    assert '"type": "completion"' in response.text
    assert '"thread_id":' in response.text


def test_chat_sse_rejects_foreign_thread_before_loading_checkpoint(monkeypatch) -> None:
    secret = "test-secret-key-with-at-least-32-bytes"
    monkeypatch.setenv("JWT_SECRET", secret)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    token = jwt.encode(
        {
            "sub": "trusted-account",
            "iat": int(time()),
            "exp": time() + 300,
            "iss": "banking-auth",
            "aud": "banking-api",
            "typ": "access",
        },
        secret,
        algorithm="HS256",
    )

    class ForeignThreadStore:
        async def resolve_thread(self, owner_id, requested_thread_id):
            raise orchestrator_main.ConversationNotFound

        async def load(self, thread_id):
            raise AssertionError("foreign checkpoint must not be loaded")

    monkeypatch.setattr(
        orchestrator_main,
        "_conversation_store",
        lambda app_state: ForeignThreadStore(),
    )
    response = TestClient(orchestrator_app).post(
        "/chat/stream",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "type": "message",
            "content": "Continue",
            "thread_id": str(uuid4()),
        },
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation is unavailable"}


def test_chat_balance_prompts_for_account_then_returns_selected_balance(monkeypatch) -> None:
    monkeypatch.setattr(
        orchestrator_main, "check_advice_boundary", _allow_advice_boundary
    )
    async def classify_balance(rails, content, context):
        hint = "Savings" if "saving" in content.casefold() else None
        return orchestrator_main.ChatIntentClassification(
            intent="balance_check",
            account_hint=hint,
        )

    monkeypatch.setattr(
        orchestrator_main, "_classify_chat_intent", classify_balance
    )
    token = "validated-test-token"
    monkeypatch.setenv("BALANCE_SERVICE_URL", "http://balance.test")

    async def passthrough_check(
        rails, content, user_id, request_path="/chat", context=()
    ):
        return content

    monkeypatch.setattr(orchestrator_main, "check_input", passthrough_check)
    monkeypatch.setattr(orchestrator_main, "check_output", passthrough_check)
    requested_urls = []

    class FakeAsyncClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def get(self, url, headers):
            assert headers == {"Authorization": f"Bearer {token}"}
            requested_urls.append(url)
            request = httpx.Request("GET", url)
            if url == "http://balance.test/sandbox/accounts":
                return httpx.Response(
                    200,
                    request=request,
                    json=[
                        {"account_id": "checking-001", "label": "Checking account"},
                        {"account_id": "savings-001", "label": "Savings account"},
                    ],
                )
            assert url == "http://balance.test/sandbox/balances/savings-001"
            return httpx.Response(
                200,
                request=request,
                json={
                    "kind": "balance",
                    "account_id": "savings-001",
                    "balance": "3200.00",
                    "currency": "USD",
                },
            )

    monkeypatch.setattr(orchestrator_main.httpx, "AsyncClient", FakeAsyncClient)

    async def collect(content, account_choice=None, prior_messages=None):
        return [
            event
            async for event in orchestrator_main._stream_response(
                token,
                "sandbox-checking-001",
                content,
                account_choice=account_choice,
                prior_messages=prior_messages,
            )
        ]

    prompt_events = asyncio.run(collect("What is my balance?"))
    assert prompt_events == [
        {
            "type": "account_options",
            "content": "Which account would you like to check?",
            "accounts": [
                {"choice": 0, "label": "Checking account"},
                {"choice": 1, "label": "Savings account"},
            ],
        },
        {"type": "completion"},
    ]
    assert requested_urls == ["http://balance.test/sandbox/accounts"]

    savings_events = asyncio.run(collect("What is my savings account balance?"))
    assert savings_events == [
        {
            "type": "financial",
            "payload": {
                "kind": "balance",
                "account_id": "savings-001",
                "balance": 3200.0,
                "currency": "USD",
                "account_label": "Savings account",
            },
        },
        {"type": "completion"},
    ]
    assert requested_urls[-2:] == [
        "http://balance.test/sandbox/accounts",
        "http://balance.test/sandbox/balances/savings-001",
    ]

    selected_by_name_events = asyncio.run(
        collect(
            "savings",
            prior_messages=[
                AIMessage(
                    content=(
                        "Which account would you like to check? Available accounts: "
                        "Checking account, Savings account."
                    )
                )
            ],
        )
    )
    assert selected_by_name_events == savings_events
    assert requested_urls[-2:] == [
        "http://balance.test/sandbox/accounts",
        "http://balance.test/sandbox/balances/savings-001",
    ]

    saving_events = asyncio.run(collect("What is my saving account balance?"))
    assert saving_events == savings_events
    assert requested_urls[-2:] == [
        "http://balance.test/sandbox/accounts",
        "http://balance.test/sandbox/balances/savings-001",
    ]

    savings_balance_events = asyncio.run(collect("savings balance"))
    assert savings_balance_events == savings_events
    assert requested_urls[-2:] == [
        "http://balance.test/sandbox/accounts",
        "http://balance.test/sandbox/balances/savings-001",
    ]

    selected_events = asyncio.run(collect("Check my savings balance", 1))
    assert selected_events == [
        {
            "type": "financial",
            "payload": {
                "kind": "balance",
                "account_id": "savings-001",
                "balance": 3200.0,
                "currency": "USD",
                "account_label": "Savings account",
            },
        },
        {"type": "completion"},
    ]
    assert requested_urls[-2:] == [
        "http://balance.test/sandbox/accounts",
        "http://balance.test/sandbox/balances/savings-001",
    ]


def test_chat_balance_uses_only_account_without_prompt(monkeypatch) -> None:
    monkeypatch.setattr(
        orchestrator_main, "check_advice_boundary", _allow_advice_boundary
    )
    monkeypatch.setattr(
        orchestrator_main,
        "_classify_chat_intent",
        _intent_classifier("balance_check"),
    )
    token = "validated-test-token"
    monkeypatch.setenv("BALANCE_SERVICE_URL", "http://balance.test")

    async def passthrough_check(
        rails, content, user_id, request_path="/chat", context=()
    ):
        return content

    monkeypatch.setattr(orchestrator_main, "check_input", passthrough_check)
    monkeypatch.setattr(orchestrator_main, "check_output", passthrough_check)

    class FakeAsyncClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def get(self, url, headers):
            request = httpx.Request("GET", url)
            if url.endswith("/sandbox/accounts"):
                return httpx.Response(
                    200,
                    request=request,
                    json=[{"account_id": "checking-001", "label": "Checking account"}],
                )
            assert url.endswith("/sandbox/balances/checking-001")
            return httpx.Response(
                200,
                request=request,
                json={
                    "kind": "balance",
                    "account_id": "checking-001",
                    "balance": "1500.00",
                    "currency": "USD",
                },
            )

    monkeypatch.setattr(orchestrator_main.httpx, "AsyncClient", FakeAsyncClient)

    async def collect():
        return [
            event
            async for event in orchestrator_main._stream_response(
                token, "customer-001", "What is my balance?"
            )
        ]

    events = asyncio.run(collect())

    assert events == [
        {
            "type": "financial",
            "payload": {
                "kind": "balance",
                "account_id": "checking-001",
                "balance": 1500.0,
                "currency": "USD",
                "account_label": "Checking account",
            },
        },
        {"type": "completion"},
    ]


def test_contextual_sufficiency_followup_checks_current_balance(monkeypatch) -> None:
    monkeypatch.setenv("BALANCE_SERVICE_URL", "http://balance.test")
    monkeypatch.setattr(
        orchestrator_guardrails, "append_audit_event", lambda event: None
    )
    monkeypatch.setattr(
        orchestrator_main,
        "_classify_chat_intent",
        _intent_classifier(
            "balance_sufficiency",
            amount=500,
            account_hint="Savings",
        ),
    )

    class FakeRails:
        async def generate_async(self, prompt, **kwargs):
            return '{"is_blocked": false, "reason": "Bank balance feasibility question."}'

        async def check_async(self, messages, **kwargs):
            assert messages[-1]["role"] == "user"
            assert "Latest user message to evaluate:\nIs this enough to transfer 500" in messages[-1]["content"]
            assert messages[0] == {
                "role": "user",
                "content": "Check my savings account balance",
            }
            return SimpleNamespace(
                status=SimpleNamespace(name="PASSED"),
                content=messages[-1]["content"],
                rail=None,
            )

    async def passthrough_output(
        rails, content, user_id, request_path="/chat", context=()
    ):
        return content

    monkeypatch.setattr(orchestrator_main, "check_output", passthrough_output)
    requested_urls = []

    class FakeAsyncClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def get(self, url, headers):
            assert headers == {"Authorization": "Bearer token"}
            requested_urls.append(url)
            request = httpx.Request("GET", url)
            if url == "http://balance.test/sandbox/accounts":
                return httpx.Response(
                    200,
                    request=request,
                    json=[
                        {"account_id": "checking-001", "label": "Checking account"},
                        {"account_id": "savings-001", "label": "Savings account"},
                    ],
                )
            assert url == "http://balance.test/sandbox/balances/savings-001"
            return httpx.Response(
                200,
                request=request,
                json={
                    "account_id": "savings-001",
                    "balance": "3200.00",
                    "currency": "USD",
                },
            )

    monkeypatch.setattr(orchestrator_main.httpx, "AsyncClient", FakeAsyncClient)

    async def collect_events():
        return [
            event
            async for event in orchestrator_main._stream_response(
                "token",
                "customer-001",
                "Is this enough to transfer 500",
                guardrails=FakeRails(),
                prior_messages=[
                    HumanMessage(content="Check my savings account balance"),
                    AIMessage(
                        content="Your savings account balance is USD 3,200.00."
                    ),
                ],
            )
        ]

    events = asyncio.run(collect_events())

    assert events == [
        {
            "type": "content",
            "content": (
                "Based on the current Savings account balance, there is enough "
                "for a USD 500.00 transfer. This does not initiate a transfer."
            ),
        },
        {"type": "completion"},
    ]
    assert requested_urls == [
        "http://balance.test/sandbox/accounts",
        "http://balance.test/sandbox/balances/savings-001",
    ]


def test_incomplete_transfer_request_prompts_for_accounts(monkeypatch) -> None:
    monkeypatch.setattr(
        orchestrator_main, "check_advice_boundary", _allow_advice_boundary
    )
    monkeypatch.setattr(
        orchestrator_main,
        "_classify_chat_intent",
        _intent_classifier("transfer_request"),
    )
    async def passthrough_check(
        rails, content, user_id, request_path="/chat", context=()
    ):
        return content

    monkeypatch.setattr(orchestrator_main, "check_input", passthrough_check)
    monkeypatch.setattr(orchestrator_main, "check_output", passthrough_check)
    monkeypatch.setattr(
        orchestrator_main,
        "build_agent_graph",
        lambda *args, **kwargs: pytest.fail("Incomplete transfer reached the model"),
    )

    async def collect():
        return [
            event
            async for event in orchestrator_main._stream_response(
                "validated-test-token",
                "sandbox-checking-001",
                "Transfer 1USD",
            )
        ]

    events = asyncio.run(collect())

    assert events == [
        {
            "type": "content",
            "content": (
                "Please specify both the source and recipient accounts. "
                "For example: 'Transfer $10 from Checking account to Savings account.'"
            ),
        },
        {"type": "completion"},
    ]


def test_langgraph_runs_balance_tool_over_http(monkeypatch) -> None:
    token = "validated-test-token"
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("BALANCE_SERVICE_URL", "http://balance.test")

    class FakeModel:
        def bind_tools(self, tools):
            assert [registered_tool.name for registered_tool in tools] == [
                "check_balance",
            ]
            return self

        async def astream(self, messages):
            if any(isinstance(message, ToolMessage) for message in messages):
                yield AIMessageChunk(content="Your balance is USD 50.00.")
            else:
                yield AIMessageChunk(
                    content="",
                    tool_calls=[
                        {
                            "name": "check_balance",
                            "args": {},
                            "id": "balance-call-1",
                            "type": "tool_call",
                        }
                    ],
                )

    class FakeAsyncClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def get(self, url, headers):
            assert url == "http://balance.test/sandbox/balances/trusted-account"
            assert headers == {"Authorization": f"Bearer {token}"}
            request = httpx.Request("GET", url, headers=headers)
            return httpx.Response(
                200,
                request=request,
                json={"balance": "50.00", "currency": "USD"},
            )

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        lambda **kwargs: FakeModel(),
    )
    monkeypatch.setattr(orchestrator_agent.httpx, "AsyncClient", FakeAsyncClient)

    graph = orchestrator_agent.build_agent_graph(token, "trusted-account")
    state = orchestrator_agent.initial_chat_state(
        "Check my balance", "trusted-account"
    )

    async def collect_events():
        return [event async for event in orchestrator_agent.stream_agent_events(graph, state)]

    events = asyncio.run(collect_events())

    assert events[-1] == {"type": "completion"}
    assert {
        "type": "financial",
        "payload": {
            "kind": "balance",
            "account_id": "trusted-account",
            "balance": 50.0,
            "currency": "USD",
        },
    } in events
    assert not any(
        event.get("type") == "content"
        and event.get("content") == "Your balance is USD 50.00."
        for event in events
    )