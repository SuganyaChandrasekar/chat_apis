import asyncio

from langchain_core.messages import AIMessage

from services.authentication import create_access_token
from services.orchestrator import agent, fraud, main


def test_chat_classifier_detects_scam_reports_semantically() -> None:
    class FakeRails:
        async def generate_async(self, *, prompt, options):
            assert "suspected scam" in prompt
            assert "'my card'" in prompt
            return '{"intent":"fraud_report","card_hint":"debit"}'

    result = asyncio.run(
        main._classify_chat_intent(
            FakeRails(),
            "Someone stole my money using my debit card",
            [],
        )
    )

    assert result.intent == "fraud_report"
    assert result.card_hint == "debit"


def test_langgraph_routes_fraud_reports_to_the_risk_distress_action(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", "orchestrator-fraud-test-secret-32")
    monkeypatch.setattr(agent, "record_authorization_decision", lambda **kwargs: None)
    recorded = []

    async def report_sandbox_fraud(**kwargs):
        recorded.append(kwargs)
        return "I froze your card and opened Fraud Case #FRD-2026-1234ABCD."

    monkeypatch.setattr(agent, "report_sandbox_fraud", report_sandbox_fraud)
    state = agent.initial_chat_state(
        "I've been scammed",
        "sandbox-checking-001",
    )
    state["intent"] = "fraud_report"
    state["fraud_card_hint"] = "debit"
    token = create_access_token(
        "sandbox-checking-001",
        scopes=("fraud:report",),
        consent_state="not_required",
        sca_state="not_required",
        token_kind="delegated",
    )
    graph = agent.build_agent_graph(token, "sandbox-checking-001")

    events = asyncio.run(_collect(agent.stream_agent_events(graph, state)))

    assert events[0]["type"] == "content"
    assert "FRD-2026-1234ABCD" in events[0]["content"]
    assert events[-1]["type"] == "completion"
    assert recorded == [
        {
            "access_token": token,
            "account_id": "sandbox-checking-001",
            "reason": "I've been scammed",
            "card_hint": "debit",
        }
    ]


def test_fraud_report_freezes_all_active_cards_without_confirmation(
    monkeypatch,
) -> None:
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def get(self, url, headers):
            raise AssertionError("All cards can be frozen without card lookup")

        async def post(self, url, headers, json):
            calls.append((url, headers, json))
            if url.endswith("/sandbox/fraud/freeze"):
                return FakeResponse(
                    {
                        "frozen_card_ids": ["card_debit_4321", "card_credit_8899"],
                        "newly_frozen_card_ids": [
                            "card_debit_4321",
                            "card_credit_8899",
                        ],
                        "already_frozen_card_ids": [],
                    }
                )
            return FakeResponse(
                {
                    "case_id": "FRD-2026-1234ABCD",
                    "user_id": "sandbox-checking-001",
                    "status": "OPEN",
                    "frozen_card_ids": ["card_debit_4321", "card_credit_8899"],
                }
            )

    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    monkeypatch.setenv("CARDS_SERVICE_URL", "http://cards.test")
    monkeypatch.setenv("FRAUD_SERVICE_URL", "http://fraud.test")
    monkeypatch.setitem(
        fraud.SANDBOX_CARD_OWNER_BY_ACCOUNT_ID,
        "sandbox-checking-001",
        "usr_sandbox_01",
    )
    monkeypatch.setattr(fraud, "create_access_token", lambda subject, ttl_seconds: "card-token")
    monkeypatch.setattr(fraud.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(fraud, "uuid4", lambda: "stable-key")

    reply = asyncio.run(
        fraud.report_sandbox_fraud(
            access_token="customer-token",
            account_id="sandbox-checking-001",
            reason="Someone stole my money",
            card_hint=None,
        )
    )

    assert "immediately frozen" in reply
    assert "FRD-2026-1234ABCD" in reply
    assert "tap the button" not in reply
    assert "https://albionbank.com/sandbox/fraud-support" in reply
    assert "PHONE_NUMBER" not in reply
    assert "1-800-" not in reply
    assert calls[0][0] == "http://cards.test/sandbox/fraud/freeze"
    assert calls[0][2] == {"card_ids": None}
    assert calls[1][0] == "http://fraud.test/fraud/cases"
    assert calls[1][1]["Authorization"] == "Bearer customer-token"
    assert calls[1][1]["Idempotency-Key"] == "stable-key"


def test_fraud_report_says_cards_were_already_frozen(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeClient:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def post(self, url, headers, json):
            if url.endswith("/sandbox/fraud/freeze"):
                return FakeResponse(
                    {
                        "frozen_card_ids": ["card_debit_4321"],
                        "newly_frozen_card_ids": [],
                        "already_frozen_card_ids": ["card_debit_4321"],
                    }
                )
            return FakeResponse(
                {
                    "case_id": "FRD-2026-5678ABCD",
                    "user_id": "sandbox-checking-001",
                    "status": "OPEN",
                    "frozen_card_ids": ["card_debit_4321"],
                }
            )

    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("CARDS_SERVICE_URL", "http://cards.test")
    monkeypatch.setenv("FRAUD_SERVICE_URL", "http://fraud.test")
    monkeypatch.setitem(
        fraud.SANDBOX_CARD_OWNER_BY_ACCOUNT_ID,
        "sandbox-checking-001",
        "usr_sandbox_01",
    )
    monkeypatch.setattr(fraud, "create_access_token", lambda subject, ttl_seconds: "card-token")
    monkeypatch.setattr(fraud.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(fraud, "uuid4", lambda: "stable-key")

    reply = asyncio.run(
        fraud.report_sandbox_fraud(
            access_token="customer-token",
            account_id="sandbox-checking-001",
            reason="I was scammed",
            card_hint=None,
        )
    )

    assert "already frozen" in reply
    assert "immediately frozen" not in reply
    assert "FRD-2026-5678ABCD" in reply


def test_fraud_report_freezes_only_the_explicitly_named_card(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeClient:
        def __init__(self, timeout):
            assert timeout == 5.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exception_type, exception, traceback):
            return False

        async def get(self, url, headers):
            calls.append(("get", url, headers, None))
            return FakeResponse(
                [
                    {
                        "card_id": "card_debit_4321",
                        "name": "Albion Debit Card",
                        "last_4": "4321",
                        "status": "active",
                    },
                    {
                        "card_id": "card_credit_8899",
                        "name": "Albion Credit Card",
                        "last_4": "8899",
                        "status": "active",
                    },
                ]
            )

        async def post(self, url, headers, json):
            calls.append(("post", url, headers, json))
            if url.endswith("/sandbox/fraud/freeze"):
                return FakeResponse({
                    "frozen_card_ids": ["card_credit_8899"],
                    "newly_frozen_card_ids": ["card_credit_8899"],
                    "already_frozen_card_ids": [],
                })
            return FakeResponse(
                {
                    "case_id": "FRD-2026-5678ABCD",
                    "user_id": "sandbox-checking-001",
                    "status": "OPEN",
                    "frozen_card_ids": ["card_credit_8899"],
                }
            )

    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    monkeypatch.setenv("CARDS_SERVICE_URL", "http://cards.test")
    monkeypatch.setenv("FRAUD_SERVICE_URL", "http://fraud.test")
    monkeypatch.setitem(
        fraud.SANDBOX_CARD_OWNER_BY_ACCOUNT_ID,
        "sandbox-checking-001",
        "usr_sandbox_01",
    )
    monkeypatch.setattr(
        fraud, "create_access_token", lambda subject, ttl_seconds: "card-token"
    )
    monkeypatch.setattr(fraud.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(fraud, "uuid4", lambda: "stable-key")

    reply = asyncio.run(
        fraud.report_sandbox_fraud(
            access_token="customer-token",
            account_id="sandbox-checking-001",
            reason="Someone stole money from my credit card",
            card_hint="credit",
        )
    )

    assert "FRD-2026-5678ABCD" in reply
    assert calls[1][3] == {"card_ids": ["card_credit_8899"]}
    assert calls[2][3]["frozen_card_ids"] == ["card_credit_8899"]


def test_fraud_report_resolves_a_specific_card_hint(monkeypatch) -> None:
    cards = [
        fraud._CardSummary(
            card_id="card_debit_4321",
            name="Albion Debit Card",
            last_4="4321",
            status="active",
        ),
        fraud._CardSummary(
            card_id="card_credit_8899",
            name="Albion Credit Card",
            last_4="8899",
            status="active",
        ),
    ]

    assert fraud._matching_card_ids("credit", cards) == ["card_credit_8899"]
    assert fraud._matching_card_ids("8899", cards) == ["card_credit_8899"]


def test_fraud_safety_action_is_not_available_outside_sandbox(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "false")

    reply = asyncio.run(
        fraud.report_sandbox_fraud(
            access_token="customer-token",
            account_id="sandbox-checking-001",
            reason="I was scammed",
            card_hint=None,
        )
    )

    assert "can't open a fraud case" in reply
    assert "official support channels" in reply


async def _collect(events):
    return [event async for event in events]
