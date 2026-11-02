import asyncio

import pytest

from langchain_core.messages import ToolMessage

from services.orchestrator.agent import ChatState, stream_agent_events
from services.orchestrator.cache import (
    DEFAULT_FAQ_CACHE_TTL_SECONDS,
    NoOpFAQCache,
    RedisFAQCache,
    create_faq_cache,
    is_cacheable_public_intent,
)
from services.orchestrator import main as orchestrator_main


async def _allow_advice_boundary(*args, **kwargs):
    return {
        "is_blocked": False,
        "reason": "No regulated advice request.",
        "refusal_message": "I can't provide personal investment advice or recommend whether to buy or sell investments.",
    }


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("public_faq", True),
        ("balance_check", False),
        ("balance_sufficiency", False),
        ("transaction_history", False),
        ("transfer_history", False),
        ("transfer_request", False),
        ("account_number", False),
        ("other", False),
    ],
)
def test_only_public_faq_intent_is_cacheable(intent: str, expected: bool) -> None:
    assert is_cacheable_public_intent(intent) is expected


def test_public_faq_intent_is_classified_semantically() -> None:
    class FakeRails:
        async def generate_async(self, prompt, options):
            assert "public_faq" in prompt
            assert "general informational questions about banking concepts" in prompt
            return '{"intent":"public_faq"}'

    classification = asyncio.run(
        orchestrator_main._classify_chat_intent(
            FakeRails(), "Could you walk me through how overdrafts work?", []
        )
    )

    assert classification.intent == "public_faq"
    assert is_cacheable_public_intent(classification.intent)


def test_faq_cache_uses_configured_ttl(monkeypatch) -> None:
    monkeypatch.setenv("FAQ_CACHE_TTL_SECONDS", "600")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    cache = create_faq_cache()

    assert isinstance(cache, RedisFAQCache)
    assert cache.ttl_seconds == 600


def test_faq_cache_without_redis_is_a_safe_noop(monkeypatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("VALKEY_URL", raising=False)

    cache = create_faq_cache()

    assert isinstance(cache, NoOpFAQCache)
    assert cache.ttl_seconds == 0


def test_redis_cache_uses_default_ttl_when_configuration_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("FAQ_CACHE_TTL_SECONDS", raising=False)
    monkeypatch.setenv("VALKEY_URL", "redis://localhost:6379/0")

    cache = create_faq_cache()

    assert isinstance(cache, RedisFAQCache)
    assert cache.ttl_seconds == DEFAULT_FAQ_CACHE_TTL_SECONDS


def test_authenticated_balance_query_bypasses_faq_cache(monkeypatch) -> None:
    class FailingCache:
        ttl_seconds = 300

        async def get(self, question: str) -> str | None:
            raise AssertionError("authenticated query must not read the FAQ cache")

        async def set(self, question: str, answer: str, ttl_seconds: int) -> None:
            raise AssertionError("authenticated query must not write the FAQ cache")

        async def close(self) -> None:
            return None

    async def fake_agent_events(graph, state):
        yield {"type": "content", "content": "Your current balance is available."}
        yield {"type": "completion"}

    async def collect_events() -> list[dict]:
        monkeypatch.setattr(
            orchestrator_main, "check_advice_boundary", _allow_advice_boundary
        )
        async def classify_intent(rails, content, context):
            return orchestrator_main.ChatIntentClassification(intent="other")

        monkeypatch.setattr(
            orchestrator_main, "_classify_chat_intent", classify_intent
        )

        async def passthrough_check(
            rails, content, user_id, request_path="/chat", context=()
        ):
            return content

        monkeypatch.setattr(orchestrator_main, "check_input", passthrough_check)
        monkeypatch.setattr(orchestrator_main, "check_output", passthrough_check)
        monkeypatch.setattr(
            orchestrator_main, "_prepare_message", lambda content, account_id: {}
        )
        monkeypatch.setattr(
            orchestrator_main, "build_agent_graph", lambda token, account_id: object()
        )
        monkeypatch.setattr(orchestrator_main, "stream_agent_events", fake_agent_events)
        return [
            event
            async for event in orchestrator_main._stream_response(
                "token",
                "account-123",
                "What is my balance?",
                FailingCache(),
                guardrails=object(),
            )
        ]

    events = asyncio.run(collect_events())

    assert events == [
        {"type": "content", "content": "Your current balance is available."},
        {"type": "completion"},
    ]


def test_agent_streams_financial_payload_as_one_atomic_event() -> None:
    class FakeGraph:
        def __init__(self) -> None:
            self.events = [
                (
                    ToolMessage(
                        content='{"kind":"balance","account_id":"account-123","balance":25.5,"currency":"USD"}',
                        name="check_balance",
                        tool_call_id="balance-call-1",
                    ),
                    None,
                ),
                (
                    ToolMessage(
                        content="not-json",
                        name="check_balance",
                        tool_call_id="balance-call-2",
                    ),
                    None,
                ),
            ]

        def astream(self, state, stream_mode):
            async def generator():
                for event in self.events:
                    yield event
            return generator()

    state = ChatState(
        messages=[],
        account_id="account-123",
        tool_states={},
    )

    events = asyncio.run(_collect_events(FakeGraph(), state))

    assert events == [
        {
            "type": "financial",
            "payload": {
                "kind": "balance",
                "account_id": "account-123",
                "balance": 25.5,
                "currency": "USD",
            },
        },
        {"type": "completion"},
    ]


async def _collect_events(graph, state):
    return [event async for event in stream_agent_events(graph, state)]
