import asyncio
from datetime import date, timedelta
from time import time

import httpx
import jwt
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk, ToolMessage

from services.balance.main import app as balance_app
from services.orchestrator import agent
from services.orchestrator import main as orchestrator_main


def _access_token(subject: str = "sandbox-checking-001") -> str:
    return jwt.encode(
        {
            "sub": subject,
            "iat": int(time()),
            "exp": int(time()) + 300,
            "iss": "banking-auth",
            "aud": "banking-api",
            "typ": "access",
        },
        "test-secret-key-with-at-least-32-bytes",
        algorithm="HS256",
    )


def test_sandbox_transactions_require_authentication_and_sandbox(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    monkeypatch.setenv("SANDBOX_MODE", "true")
    client = TestClient(balance_app)

    assert client.get(
        "/accounts/sandbox-checking-001/transactions"
    ).status_code == 401

    monkeypatch.setenv("SANDBOX_MODE", "false")
    response = client.get(
        "/accounts/sandbox-checking-001/transactions",
        headers={"Authorization": f"Bearer {_access_token()}"},
    )
    assert response.status_code == 404


def test_sandbox_transactions_enforce_entitlement_and_filter_results(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    monkeypatch.setenv("SANDBOX_MODE", "true")
    client = TestClient(balance_app)
    headers = {"Authorization": f"Bearer {_access_token()}"}

    unauthorized = client.get(
        "/accounts/unlinked-account/transactions", headers=headers
    )
    assert unauthorized.status_code == 403
    checking_defaults = client.get(
        "/accounts/sandbox-checking-001/transactions", headers=headers
    )
    assert checking_defaults.status_code == 200
    assert len(checking_defaults.json()) == 5
    savings = client.get(
        "/accounts/sandbox-savings-001/transactions", headers=headers
    )
    assert savings.status_code == 200
    assert len(savings.json()) == 2

    response = client.get(
        "/accounts/sandbox-checking-001/transactions",
        headers=headers,
        params={
            "limit": 1,
            "date_from": (date.today() - timedelta(days=2)).isoformat(),
            "date_to": date.today().isoformat(),
            "merchant": "coffee",
        },
    )
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["merchant"] == "Northstar Coffee"
    assert set(response.json()[0]) == {
        "transaction_id",
        "date",
        "amount",
        "currency",
        "description",
        "merchant",
    }

    invalid_range = client.get(
        "/accounts/sandbox-checking-001/transactions",
        headers=headers,
        params={
            "date_from": date.today().isoformat(),
            "date_to": "2020-01-01",
        },
    )
    assert invalid_range.status_code == 422


def test_recent_transactions_tool_forwards_auth_and_filters(monkeypatch) -> None:
    token = _access_token()
    monkeypatch.setenv("BALANCE_SERVICE_URL", "http://balance.test")
    observed_requests = []
    today = date.today()

    class FakeAsyncClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def get(self, url, headers, params=None):
            observed_requests.append((url, headers, params))
            request = httpx.Request("GET", url)
            if url == "http://balance.test/sandbox/accounts":
                return httpx.Response(
                    200,
                    request=request,
                    json=[
                        {
                            "account_id": "sandbox-checking-001",
                            "label": "Checking account",
                        },
                        {
                            "account_id": "sandbox-savings-001",
                            "label": "Savings account",
                        },
                    ],
                )
            return httpx.Response(
                200,
                request=request,
                json=[
                    {
                        "transaction_id": "savings-tx-001",
                        "date": today.isoformat(),
                        "amount": "25.00",
                        "currency": "USD",
                        "description": "Monthly interest",
                        "merchant": "Bank Interest",
                    }
                ],
            )

    monkeypatch.setattr(agent.httpx, "AsyncClient", FakeAsyncClient)
    tool = agent.build_agent_tools(token, "sandbox-checking-001")[2]
    result = asyncio.run(
        tool.ainvoke(
            {
                "account_id": "savings",
                "limit": 5,
                "date_hint": "this month",
                "merchant_hint": "interest",
            }
        )
    )

    assert observed_requests == [
        (
            "http://balance.test/sandbox/accounts",
            {"Authorization": f"Bearer {token}"},
            None,
        ),
        (
            "http://balance.test/accounts/sandbox-savings-001/transactions",
            {"Authorization": f"Bearer {token}"},
            {
                "limit": 5,
                "date_from": today.replace(day=1).isoformat(),
                "date_to": today.isoformat(),
                "merchant": "interest",
            },
        ),
    ]
    assert '"transaction_id":"savings-tx-001"' in result


def test_transaction_history_intent_extracts_entities() -> None:
    class FakeRails:
        async def generate_async(self, prompt, options):
            assert "transaction_history" in prompt
            assert "merchant_hint" in prompt
            return (
                '{"intent":"transaction_history","limit":5,'
                '"account_hint":"checking","date_hint":"this month",'
                '"merchant_hint":"coffee"}'
            )

    classification = asyncio.run(
        orchestrator_main._classify_chat_intent(
            FakeRails(), "Show my last 5 coffee transactions this month", []
        )
    )

    assert classification.intent == "transaction_history"
    assert classification.limit == 5
    assert classification.account_hint == "checking"
    assert classification.date_hint == "this month"
    assert classification.merchant_hint == "coffee"


def test_transaction_intent_enables_recent_transactions_tool(monkeypatch) -> None:
    registered_tools = []

    class FakeModel:
        def bind_tools(self, tools):
            registered_tools.extend(tool.name for tool in tools)
            return self

        async def astream(self, messages):
            yield AIMessageChunk(content="I'll check those transactions.")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        lambda **kwargs: FakeModel(),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    state = agent.initial_chat_state(
        "Show my last 5 transactions", "sandbox-checking-001"
    )
    state["intent"] = "transaction_history"

    asyncio.run(
        agent.build_agent_graph("validated-token", "sandbox-checking-001").ainvoke(
            state
        )
    )

    assert registered_tools == ["get_recent_transactions"]


def test_recent_transaction_tool_output_is_plain_and_clear() -> None:
    transaction = {
        "transaction_id": "checking-tx-001",
        "date": date.today().isoformat(),
        "amount": "-5.75",
        "currency": "USD",
        "description": "Coffee",
        "merchant": "Northstar Coffee",
    }

    class FakeGraph:
        async def astream(self, state, stream_mode):
            yield (
                ToolMessage(
                    content=agent.RecentTransactionsPayload(
                        account_id="sandbox-checking-001",
                        transactions=[transaction],
                    ).model_dump_json(),
                    name="get_recent_transactions",
                    tool_call_id="transactions-call-1",
                ),
                {},
            )
            yield (AIMessageChunk(content="Here are your transactions."), {})

    state = agent.initial_chat_state(
        "Show my recent transactions", "sandbox-checking-001"
    )
    state["intent"] = "transaction_history"

    async def collect():
        return [
            event async for event in agent.stream_agent_events(FakeGraph(), state)
        ]

    events = asyncio.run(collect())
    assert events[0]["type"] == "content"
    assert "Northstar Coffee: Coffee" in events[0]["content"]
    assert "USD -5.75" in events[0]["content"]
    assert events[1] == {"type": "completion"}
