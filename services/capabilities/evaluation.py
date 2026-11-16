from dataclasses import dataclass
import re
from collections.abc import Sequence

from services.capabilities.contracts import EvaluationCase


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    passed: int
    total: int


def _normalize_response(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def run_evaluation(cases: Sequence[EvaluationCase]) -> EvaluationResult:
    if not cases:
        raise ValueError("At least one evaluation case is required")

    passed = sum(
        _normalize_response(case.actual_response)
        == _normalize_response(case.expected_response)
        for case in cases
    )
    return EvaluationResult(score=passed / len(cases), passed=passed, total=len(cases))
