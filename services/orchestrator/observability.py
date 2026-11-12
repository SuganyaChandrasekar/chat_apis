from contextvars import ContextVar, Token
import re

from prometheus_client import CollectorRegistry, Counter, Histogram, make_asgi_app


_DEFAULT_DIMENSIONS = ("general_banking", "unknown")
_metric_dimensions: ContextVar[tuple[str, str]] = ContextVar(
    "chat_metric_dimensions", default=_DEFAULT_DIMENSIONS
)

METRICS_REGISTRY = CollectorRegistry(auto_describe=True)
INTENT_CONFIDENCE = Histogram(
    "chat_intent_confidence",
    "Distribution of model-reported intent classification confidence.",
    labelnames=("workflow", "capability"),
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
    registry=METRICS_REGISTRY,
)
INTENT_ACCURACY_PROXY = Counter(
    "chat_intent_accuracy_proxy",
    "Policy/tool acceptance proxy; not ground-truth intent accuracy.",
    labelnames=("workflow", "capability", "outcome"),
    registry=METRICS_REGISTRY,
)
CONTAINMENT = Counter(
    "chat_containment",
    "Completed AI-resolved and escalated chat turns.",
    labelnames=("workflow", "capability", "outcome"),
    registry=METRICS_REGISTRY,
)
GROUNDING_FAITHFULNESS = Counter(
    "chat_grounding_faithfulness",
    "Grounding validation status; not_applicable until retrieval is available.",
    labelnames=("workflow", "capability", "status"),
    registry=METRICS_REGISTRY,
)
TURN_LATENCY = Histogram(
    "chat_turn_latency_seconds",
    "End-to-end chat turn execution latency in seconds.",
    labelnames=("workflow", "capability"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
    registry=METRICS_REGISTRY,
)


_METRIC_LABEL_PATTERN = re.compile(r"[^a-zA-Z0-9_:]+")


def _metric_label(value: str | None, fallback: str) -> str:
    if not isinstance(value, str):
        value = ""
    label = _METRIC_LABEL_PATTERN.sub("_", value.strip().casefold()).strip("_")
    return label[:64] or fallback


def dimensions_for_intent(
    intent: str,
    workflow: str | None = None,
    capability: str | None = None,
) -> tuple[str, str]:
    """Return sanitized semantic dimensions supplied by the intent classifier.

    The configured chat classifier determines the semantic taxonomy at runtime.
    No local intent-to-category registry is maintained; absent classifier labels
    fall back to a general workflow and a normalized intent label.
    """
    normalized_intent = _metric_label(intent, "unknown")
    return (
        _metric_label(workflow, "general_banking"),
        _metric_label(capability, normalized_intent),
    )


def set_metric_dimensions(
    workflow: str, capability: str
) -> Token[tuple[str, str]]:
    return _metric_dimensions.set((workflow, capability))


def reset_metric_dimensions(token: Token[tuple[str, str]]) -> None:
    _metric_dimensions.reset(token)


def get_metric_dimensions() -> tuple[str, str]:
    return _metric_dimensions.get()


def observe_intent_confidence(
    confidence: float, workflow: str, capability: str
) -> None:
    INTENT_CONFIDENCE.labels(workflow, capability).observe(confidence)


def observe_intent_outcome(
    outcome: str, workflow: str, capability: str
) -> None:
    INTENT_ACCURACY_PROXY.labels(workflow, capability, outcome).inc()


def observe_grounding_status(status: str, workflow: str, capability: str) -> None:
    GROUNDING_FAITHFULNESS.labels(workflow, capability, status).inc()


def observe_turn(outcome: str, duration_seconds: float) -> None:
    workflow, capability = get_metric_dimensions()
    CONTAINMENT.labels(workflow, capability, outcome).inc()
    TURN_LATENCY.labels(workflow, capability).observe(max(0.0, duration_seconds))


def metrics_asgi_app():
    return make_asgi_app(registry=METRICS_REGISTRY)
