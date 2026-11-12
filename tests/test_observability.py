import asyncio

from prometheus_client import generate_latest
from fastapi.testclient import TestClient

from services.orchestrator import main, memory, observability
from services.orchestrator.distress import DistressEvaluation


def test_classifier_accepts_runtime_semantic_dimensions() -> None:
    class SemanticRails:
        async def generate_async(self, prompt, options):
            assert "broad semantic category" in prompt
            return (
                '{"intent":"cross_border_remittance","confidence":0.84,'
                '"workflow":"international_payments",'
                '"capability":"cross_border_transfer"}'
            )

    classification = asyncio.run(
        main._classify_chat_intent(
            SemanticRails(), "Send money to another country", []
        )
    )

    assert observability.dimensions_for_intent(
        classification.intent,
        classification.workflow,
        classification.capability,
    ) == ("international_payments", "cross_border_transfer")


def test_intent_dimensions_use_semantic_neighbors_for_unseen_intents() -> None:
    assert observability.dimensions_for_intent(
        "new international remittance",
        "fund_transfer",
        "cross_border_payment",
    ) == ("fund_transfer", "cross_border_payment")
    assert observability.dimensions_for_intent(
        "unrelated_new_capability"
    ) == ("general_banking", "unrelated_new_capability")
    assert observability.dimensions_for_intent("") == ("general_banking", "unknown")
    assert observability.dimensions_for_intent(
        "future_intent", "Payments & Transfers", "Cross-Border / Remittance"
    ) == ("payments_transfers", "cross_border_remittance")


def test_distress_turn_emits_not_applicable_grounding_metric(monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(main, "sanitize_content", lambda content: content)
    monkeypatch.setattr(
        main,
        "_sandbox_distress_evaluation",
        lambda *args, **kwargs: asyncio.sleep(
            0,
            result=DistressEvaluation(
                score=0.96, tier=4, requests_human_support=False
            ),
        ),
    )
    monkeypatch.setattr(
        main,
        "observe_grounding_status",
        lambda status, workflow, capability: observed.append(
            (status, workflow, capability)
        ),
    )

    async def collect():
        return [
            event
            async for event in main._stream_response(
                token="unused",
                account_id="test-account",
                content="I need urgent support",
            )
        ]

    events = asyncio.run(collect())

    assert events[-1]["type"] == "completion"
    assert observed == [
        ("not_applicable", "customer_support", "crisis_signposting")
    ]


def test_intent_metrics_are_tagged_by_workflow_and_capability() -> None:
    workflow, capability = observability.dimensions_for_intent(
        "transfer_request", "fund_transfer", "transfer_preparation"
    )

    observability.observe_intent_confidence(0.91, workflow, capability)
    observability.observe_intent_outcome("accepted", workflow, capability)
    observability.observe_grounding_status(
        "not_applicable", workflow, capability
    )
    dimensions_token = observability.set_metric_dimensions(workflow, capability)
    observability.observe_turn("resolved", 0.25)
    observability.reset_metric_dimensions(dimensions_token)
    metrics = generate_latest(observability.METRICS_REGISTRY).decode()

    assert 'capability="transfer_preparation",workflow="fund_transfer"' in metrics
    assert "chat_intent_confidence_bucket" in metrics
    assert "chat_intent_accuracy_proxy_total" in metrics
    assert "chat_grounding_faithfulness_total" in metrics
    assert "chat_turn_latency_seconds_bucket" in metrics
    assert 'outcome="accepted"' in metrics
    assert 'outcome="resolved"' in metrics


def test_decision_trace_redacts_financial_and_pii_content(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        memory,
        "anonymize_message",
        lambda value: value.replace("alice@example.com", "[EMAIL]"),
    )
    monkeypatch.setattr(
        memory,
        "append_decision_trace_event",
        lambda **kwargs: captured.update(kwargs),
    )

    memory.record_decision_trace(
        conversation_id="00000000-0000-0000-0000-000000000001",
        correlation_id="request-123",
        event_type="user_input",
        payload={
            "content": "My email is alice@example.com; my balance is USD 500.00 "
            "and account number 12345678.",
            "pin": "1234",
            "account_id": "checking-12345678",
        },
    )

    assert "[EMAIL]" in captured["payload"]["content"]
    assert captured["payload"]["content"].find("500.00") == -1
    assert "12345678" not in captured["payload"]["content"]
    assert captured["payload"]["pin"] == "[REDACTED]"
    assert captured["payload"]["account_id"] == "[REDACTED]"


def test_persisted_response_traces_masked_input_before_implementation(
    monkeypatch,
) -> None:
    import asyncio
    from uuid import uuid4

    from services.orchestrator import main as orchestrator_main

    traced = []
    monkeypatch.setattr(
        orchestrator_main,
        "sanitize_content",
        lambda value: value.replace("12345678", "[REDACTED]"),
    )

    async def capture_trace(event_type, payload, **_kwargs) -> None:
        traced.append((event_type, payload))

    async def fail_before_processing(**_kwargs):
        raise RuntimeError("storage unavailable")
        yield

    monkeypatch.setattr(orchestrator_main, "_trace_decision", capture_trace)
    monkeypatch.setattr(
        orchestrator_main,
        "_stream_persisted_response_impl",
        fail_before_processing,
    )

    async def consume() -> None:
        try:
            async for _event in orchestrator_main._stream_persisted_response(
                store=object(),
                thread_id=uuid4(),
                token="token",
                account_id="customer",
                content="Account number 12345678",
                correlation_id="request-correlation",
            ):
                pass
        except RuntimeError:
            pass

    asyncio.run(consume())

    assert traced[0] == (
        "user_input",
        {"content": "Account number [REDACTED]"},
    )


def test_prometheus_metrics_endpoint_is_mounted() -> None:
    from services.orchestrator.main import app

    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert "chat_intent_confidence" in response.text
