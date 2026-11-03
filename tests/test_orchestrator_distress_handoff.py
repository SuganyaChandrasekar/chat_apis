import asyncio
from uuid import uuid4

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from services.orchestrator import distress, handoff, main


class _FakeDistressClassifier:
    def __init__(
        self,
        *,
        score: float,
        risk_level: str,
        requests_human_support: bool = False,
    ) -> None:
        self.score = score
        self.risk_level = risk_level
        self.requests_human_support = requests_human_support
        self.prompt = ""

    async def generate_async(self, *, prompt, options):
        self.prompt = prompt
        return {
            "distress_score": self.score,
            "risk_level": self.risk_level,
            "requests_human_support": self.requests_human_support,
        }


def test_distress_classifier_uses_semantics_and_normalizes_tiers() -> None:
    async def classify(score: float, risk_level: str):
        return await distress.evaluate_distress(
            "The meaning depends on what the user is experiencing.",
            rails=_FakeDistressClassifier(score=score, risk_level=risk_level),
        )

    tier_one = asyncio.run(classify(0.55, "elevated"))
    tier_two = asyncio.run(classify(0.75, "high"))
    tier_three = asyncio.run(classify(0.86, "crisis"))
    tier_four = asyncio.run(classify(0.96, "severe"))
    crisis_floor = asyncio.run(classify(0.30, "crisis"))

    assert 0.4 <= tier_one.score < 0.7
    assert tier_one.tier == 1
    assert 0.7 <= tier_two.score < 0.85
    assert tier_two.tier == 2
    assert tier_three.score >= 0.85
    assert tier_three.tier == 3
    assert tier_four.score >= 0.95
    assert tier_four.tier == 4
    assert crisis_floor.score == 0.85
    assert crisis_floor.tier == 3
    requested_support = asyncio.run(
        distress.evaluate_distress(
            "Could you connect me with someone?",
            rails=_FakeDistressClassifier(
                score=0.2,
                risk_level="low",
                requests_human_support=True,
            ),
        )
    )
    assert requested_support.tier == 0
    assert requested_support.requests_human_support is True


def test_distress_classifier_includes_context_and_fails_closed() -> None:
    classifier = _FakeDistressClassifier(score=0.2, risk_level="none")
    prior = [HumanMessage(content="Earlier I said my situation was getting worse.")]

    result = asyncio.run(
        distress.evaluate_distress(
            "I don't know how much longer I can keep this up.",
            prior,
            rails=classifier,
        )
    )

    assert result.tier == 0
    assert "my situation was getting worse" in classifier.prompt
    assert "meaning rather than keywords" in classifier.prompt
    assert "requests_human_support" in classifier.prompt

    async def classify_without_model():
        return await distress.evaluate_distress("Help.", rails=None)

    try:
        asyncio.run(classify_without_model())
    except distress.DistressClassifierUnavailable:
        pass
    else:
        raise AssertionError("Missing classifier must fail closed")


def test_distress_state_modulates_agent_prompt() -> None:
    state = main._prepare_message(
        "I need help.", "sandbox-checking-001", distress_tier=2
    )
    prompt = state["messages"][0].content

    assert "extra empathy" in prompt
    assert "short and simple sentences" in prompt
    assert "warm hand-off" in prompt


def test_crisis_turn_signposts_from_structured_semantic_result(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setattr(main, "sanitize_content", lambda content: content)
    turn_messages = []
    classifier = _FakeDistressClassifier(score=0.6, risk_level="severe")

    async def collect_events():
        return [
            event
            async for event in main._stream_response(
                token="unused",
                account_id="sandbox-user",
                content="I am going to kill myself.",
                turn_messages=turn_messages,
                guardrails=classifier,
            )
        ]

    events = asyncio.run(collect_events())

    assert events[0]["type"] == "content"
    assert "local emergency services" in events[0]["content"]
    assert events[-1]["type"] == "completion"
    assert turn_messages[0].content == "I am going to kill myself."
    assert turn_messages[0].additional_kwargs["sandbox_distress_tier"] == 4


def test_tier_two_distress_gets_support_offer_not_generic_refusal(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setattr(main, "sanitize_content", lambda content: content)
    classifier = _FakeDistressClassifier(score=0.76, risk_level="high")

    async def unexpected_advice_check(*args, **kwargs):
        raise AssertionError("Distress support must not enter advice classification")

    monkeypatch.setattr(main, "check_advice_boundary", unexpected_advice_check)

    class _EmptyStore:
        appended = []

        async def load(self, thread_id):
            return []

        async def append(self, thread_id, messages):
            self.appended.extend(messages)

    store = _EmptyStore()

    async def collect_events():
        return [
            event
            async for event in main._stream_persisted_response(
                store=store,
                thread_id=uuid4(),
                token="unused",
                account_id="sandbox-user",
                content="I’m overwhelmed by this financial situation.",
                guardrails=classifier,
            )
        ]

    events = asyncio.run(collect_events())
    response_text = "".join(
        event["content"] for event in events if event["type"] == "content"
    )

    assert "I’m sorry this feels difficult." in response_text
    assert "human support specialist" in response_text
    assert "Reply yes to share your identity" in response_text
    assert "can't assist with that request" not in response_text
    assert any(
        isinstance(message, AIMessage)
        and "human support specialist" in message.content
        for message in store.appended
    )


def test_direct_human_support_request_gets_consent_offer_below_tier_two(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setattr(main, "sanitize_content", lambda content: content)
    classifier = _FakeDistressClassifier(
        score=0.2,
        risk_level="low",
        requests_human_support=True,
    )

    class _EmptyStore:
        appended = []

        async def load(self, thread_id):
            return []

        async def append(self, thread_id, messages):
            self.appended.extend(messages)

    store = _EmptyStore()

    async def collect_events():
        return [
            event
            async for event in main._stream_persisted_response(
                store=store,
                thread_id=uuid4(),
                token="unused",
                account_id="sandbox-user",
                content=(
                    "I’m overwhelmed by this financial situation. "
                    "Could you connect me with someone?"
                ),
                guardrails=classifier,
            )
        ]

    events = asyncio.run(collect_events())
    response_text = "".join(
        event["content"] for event in events if event["type"] == "content"
    )

    assert "I can also connect you with a human support specialist." in response_text
    assert "Reply yes to share your identity" in response_text
    assert any(
        isinstance(message, AIMessage)
        and main._HANDOFF_OFFER in message.content
        for message in store.appended
    )


class _MemoryStore:
    def __init__(self) -> None:
        self.appended = []

    async def append(self, thread_id, messages) -> None:
        self.appended.extend(messages)

    async def resolve_thread(self, owner_id, requested_thread_id):
        assert owner_id == "sandbox-user"
        return requested_thread_id

    async def load(self, thread_id):
        return [
            HumanMessage(
                content=(
                    "I feel desperate, I'm panicking, please help, "
                    "I can't manage and I'm scared."
                )
            ),
            AIMessage(content=f"I hear you. {main._HANDOFF_OFFER}"),
        ]


def test_accept_handoff_persists_masked_session_package(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(
        main,
        "sanitize_content",
        lambda content: content.replace("sensitive@example.test", "[EMAIL]"),
    )
    handoff._mock_packages.clear()
    session_id = uuid4()
    prior_messages = [
        HumanMessage(
            content=(
                "I feel desperate, I'm panicking, please help, I can't manage "
                "and I'm scared."
            )
        ),
        AIMessage(content=f"I hear you. {main._HANDOFF_OFFER}"),
    ]
    store = _MemoryStore()
    classifier = _FakeDistressClassifier(score=0.75, risk_level="high")

    package, acknowledgement = asyncio.run(
        main._accept_handoff(
            store=store,
            thread_id=session_id,
            user_id="sandbox-user",
            prior_messages=prior_messages,
            accepted_content="Yes, and my email is sensitive@example.test.",
            guardrails=classifier,
        )
    )

    assert package.session_id == str(session_id)
    assert package.user_id == "sandbox-user"
    assert package.distress_score >= 0.7
    assert package.identity_consent_state == "consented"
    assert package.detected_intent == "distress_support"
    assert "[EMAIL]" in package.transcript[-1].content
    assert "sensitive@example.test" not in package.transcript[-1].content
    assert store.appended[-1].content == acknowledgement
    assert "recorded in this sandbox" in acknowledgement
    assert "not been sent to or assigned" in acknowledgement
    assert "queue place" not in acknowledgement


def test_concurrent_handoff_acceptance_does_not_append_second_confirmation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setattr(main, "sanitize_content", lambda content: content)
    thread_id = uuid4()
    existing_package = handoff.HandoffPackage(
        package_id="b" * 32,
        session_id=str(thread_id),
        user_id="sandbox-user",
        transcript=[],
        detected_intent="distress_support",
        distress_score=0.75,
        identity_consent_state="consented",
        actions_taken=[],
    )
    monkeypatch.setattr(main, "new_package_id", lambda: "a" * 32)
    monkeypatch.setattr(
        main, "create_handoff_package", lambda _package: existing_package
    )

    store = _MemoryStore()
    prior_messages = [
        HumanMessage(
            content="I am overwhelmed.",
            additional_kwargs={
                "sandbox_distress_score": 0.75,
                "sandbox_distress_tier": 2,
                "sandbox_requests_human_support": False,
            },
        ),
        AIMessage(content=main._HANDOFF_OFFER),
    ]

    package, acknowledgement = asyncio.run(
        main._accept_handoff(
            store=store,
            thread_id=thread_id,
            user_id="sandbox-user",
            prior_messages=prior_messages,
            accepted_content="yes",
            guardrails=None,
        )
    )

    assert package == existing_package
    assert "already recorded" in acknowledgement
    assert store.appended == []


def test_handoff_uses_persisted_distress_assessment_without_reclassifying() -> None:
    evaluation = asyncio.run(
        main._latest_user_distress(
            [
                HumanMessage(
                    content="I am overwhelmed.",
                    additional_kwargs={
                        "sandbox_distress_score": 0.76,
                        "sandbox_distress_tier": 2,
                        "sandbox_requests_human_support": False,
                    },
                )
            ],
            guardrails=None,
        )
    )

    assert evaluation == distress.DistressEvaluation(score=0.76, tier=2)


def test_repeated_handoff_acceptance_returns_saved_package(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setattr(main, "sanitize_content", lambda content: content)
    thread_id = uuid4()
    package = handoff.HandoffPackage(
        package_id="a" * 32,
        session_id=str(thread_id),
        user_id="sandbox-user",
        transcript=[],
        detected_intent="distress_support",
        distress_score=0.75,
        identity_consent_state="consented",
        actions_taken=[],
    )
    monkeypatch.setattr(main, "get_handoff_package", lambda **_kwargs: package)

    class _CompletedStore:
        appended = []

        async def load(self, _thread_id):
            return [AIMessage(content="Your hand-off is already recorded.")]

        async def append(self, _thread_id, messages):
            self.appended.extend(messages)

    store = _CompletedStore()

    async def collect_events():
        return [
            event
            async for event in main._stream_persisted_response(
                store=store,
                thread_id=thread_id,
                token="unused",
                account_id="sandbox-user",
                content="yes",
            )
        ]

    events = asyncio.run(collect_events())

    assert "already recorded" in events[0]["content"]
    assert package.package_id in events[0]["content"]
    assert events[-1]["type"] == "completion"
    assert isinstance(store.appended[-1], AIMessage)


def test_handoff_package_endpoint_authenticates_and_reports_no_delivery_integration(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SANDBOX_MODE", "true")
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("JWT_SECRET", "test-secret-with-more-than-32-characters")
    monkeypatch.setattr(main, "sanitize_content", lambda content: content)

    async def classify_pending_turn(messages, *, guardrails):
        return distress.DistressEvaluation(score=0.75, tier=2)

    monkeypatch.setattr(main, "_latest_user_distress", classify_pending_turn)
    store = _MemoryStore()
    monkeypatch.setattr(main, "_conversation_store", lambda _state: store)
    handoff._mock_packages.clear()
    token = main.create_access_token("sandbox-user")
    session_id = str(uuid4())

    with TestClient(main.app) as client:
        response = client.post(
            "/hand-off/package",
            headers={"Authorization": f"Bearer {token}"},
            json={"session_id": session_id},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["session_id"] == session_id
    assert result["user_id"] == "sandbox-user"
    assert result["identity_consent_state"] == "consented"
    assert result["delivery"] == {"status": "not_configured"}
