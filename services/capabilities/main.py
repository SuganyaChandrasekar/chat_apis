import asyncio
import json
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from services.api_errors import register_exception_handlers
from services.authentication import require_any_scope
from services.capabilities.contracts import (
    CapabilityManifest,
    CapabilityPromotionResponse,
    CapabilityRegistrationRequest,
    CapabilityRegistrationResponse,
    CapabilityRouteResponse,
    EvaluationCase,
)
from services.capabilities.evaluation import run_evaluation
from services.capabilities.store import (
    CapabilityNotFound,
    CapabilityPromotionBlocked,
    CapabilityRecord,
    CapabilityStoreUnavailable,
    CapabilityVersionConflict,
    create_version,
    get_version,
    promote_version,
    resolve_capability,
)
from services.orchestrator.guardrails import (
    GuardrailViolation,
    GuardrailsUnavailable,
    check_input,
    check_output,
    initialize_guardrails,
)

logger = logging.getLogger("uvicorn.error")
_REGISTRY_SCOPES = ("admin:capabilities", "lob:publisher")
_MINIMUM_EVALUATION_SCORE = 0.95
_CERTIFICATION_TIMEOUT_SECONDS = 60


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.guardrails = initialize_guardrails()
    yield


app = FastAPI(title="Capability Registry", version="0.1.0", lifespan=lifespan)
register_exception_handlers(app)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _request: Request, exception: RequestValidationError
) -> JSONResponse:
    errors = [
        {
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"],
        }
        for error in exception.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


def _record_response(
    record: CapabilityRecord, *, reasons: list[str] | None = None
) -> CapabilityRegistrationResponse:
    return CapabilityRegistrationResponse(
        capability_id=record.capability_id,
        version=record.version,
        certification_status=record.certification_status,
        guardrails_passed=record.guardrails_passed,
        evaluation_score=record.evaluation_score,
        rollout_status=record.rollout_status,
        feature_flag=record.feature_flag,
        feature_flag_enabled=record.feature_flag_enabled,
        cohort=record.cohort,
        general_traffic_percentage=record.general_traffic_percentage,
        reasons=record.reasons if reasons is None else reasons,
    )


async def _check_guardrail_text(
    rails: Any,
    text: str,
    *,
    role: str,
    user_id: str,
) -> bool:
    try:
        if role == "input":
            checked_text = await check_input(
                rails, text, user_id, request_path="/capabilities"
            )
        else:
            checked_text = await check_output(
                rails, text, user_id, request_path="/capabilities"
            )
    except GuardrailViolation:
        return False
    if checked_text != text:
        logger.warning("Capability certification blocked masked content role=%s", role)
        return False
    return True


async def _run_certification(
    *,
    rails: Any,
    manifest: CapabilityManifest,
    cases: list[EvaluationCase],
    user_id: str,
) -> tuple[bool, float | None, list[str]]:
    if rails is None:
        raise GuardrailsUnavailable

    manifest_text = json.dumps(
        manifest.model_dump(mode="json", by_alias=True), ensure_ascii=True
    )
    if not await _check_guardrail_text(
        rails, manifest_text, role="input", user_id=user_id
    ):
        return False, None, ["manifest_guardrail_failed_or_pii_detected"]

    for case_number, case in enumerate(cases, start=1):
        for role, value in (
            ("input", case.prompt),
            ("output", case.expected_response),
            ("output", case.actual_response),
        ):
            if not await _check_guardrail_text(
                rails, value, role=role, user_id=user_id
            ):
                return False, None, [
                    f"evaluation_case_{case_number}_{role}_guardrail_failed_or_pii_detected"
                ]

    result = run_evaluation(cases)
    reasons = (
        []
        if result.score >= _MINIMUM_EVALUATION_SCORE
        else ["evaluation_below_0.95"]
    )
    return True, result.score, reasons


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {"service": "capabilities", "status": "ok"}


@app.get(
    "/capabilities/{capability_id}/route",
    response_model=CapabilityRouteResponse,
    tags=["capabilities"],
)
async def route_capability(
    capability_id: str,
    cohort: Literal["general", "internal_beta"] = Query(default="general"),
    _: str = Depends(require_any_scope(_REGISTRY_SCOPES)),
) -> CapabilityRouteResponse:
    try:
        record = await asyncio.to_thread(
            resolve_capability, capability_id, cohort=cohort
        )
    except CapabilityStoreUnavailable:
        raise HTTPException(
            status_code=503, detail="Capability registry storage is unavailable"
        ) from None
    if record is None:
        raise HTTPException(
            status_code=404,
            detail="No certified capability version is routable for this cohort",
        )
    return CapabilityRouteResponse(
        capability_id=record.capability_id,
        version=record.version,
        cohort=record.cohort,
        feature_flag=record.feature_flag,
        manifest=record.manifest,
    )


@app.post(
    "/capabilities",
    response_model=CapabilityRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["capabilities"],
)
async def register_capability(
    body: CapabilityRegistrationRequest,
    request: Request,
    user_id: str = Depends(require_any_scope(_REGISTRY_SCOPES)),
) -> CapabilityRegistrationResponse:
    try:
        rails = request.app.state.guardrails
        guardrails_passed, evaluation_score, reasons = await asyncio.wait_for(
            _run_certification(
                rails=rails,
                manifest=body.manifest,
                cases=body.evaluation_cases,
                user_id=user_id,
            ),
            timeout=_CERTIFICATION_TIMEOUT_SECONDS,
        )
    except GuardrailsUnavailable:
        raise HTTPException(
            status_code=503, detail="Capability guardrail certification is unavailable"
        ) from None
    except TimeoutError:
        raise HTTPException(
            status_code=503, detail="Capability certification timed out"
        ) from None

    try:
        record = await asyncio.to_thread(
            create_version,
            manifest=body.manifest.model_dump(mode="json", by_alias=True),
            guardrails_passed=guardrails_passed,
            evaluation_score=evaluation_score,
            reasons=reasons,
        )
    except CapabilityVersionConflict:
        raise HTTPException(
            status_code=409, detail="This capability version is already registered"
        ) from None
    except CapabilityStoreUnavailable:
        raise HTTPException(
            status_code=503, detail="Capability registry storage is unavailable"
        ) from None
    return _record_response(record)


@app.post(
    "/capabilities/{capability_id}/versions/{version}/promote",
    response_model=CapabilityPromotionResponse,
    tags=["capabilities"],
)
async def promote_capability(
    capability_id: str,
    version: str,
    response: Response,
    _: str = Depends(require_any_scope(_REGISTRY_SCOPES)),
) -> CapabilityPromotionResponse:
    try:
        candidate = await asyncio.to_thread(get_version, capability_id, version)
        if candidate is None:
            raise HTTPException(status_code=404, detail="Capability version not found")
        active = await asyncio.to_thread(
            resolve_capability, capability_id, cohort="general"
        )
        promoted = await asyncio.to_thread(promote_version, capability_id, version)
    except CapabilityPromotionBlocked as error:
        response.status_code = status.HTTP_409_CONFLICT
        return CapabilityPromotionResponse(
            capability_id=capability_id,
            version=version,
            promoted=False,
            rollout_status=candidate.rollout_status,
            candidate_score=candidate.evaluation_score or 0.0,
            active_version=error.active_version,
            reasons=error.reasons,
        )
    except CapabilityNotFound:
        raise HTTPException(status_code=404, detail="Capability version not found") from None
    except CapabilityVersionConflict:
        raise HTTPException(
            status_code=409, detail="Another capability version was promoted concurrently"
        ) from None
    except CapabilityStoreUnavailable:
        raise HTTPException(
            status_code=503, detail="Capability registry storage is unavailable"
        ) from None

    return CapabilityPromotionResponse(
        capability_id=promoted.capability_id,
        version=promoted.version,
        promoted=True,
        rollout_status=promoted.rollout_status,
        candidate_score=promoted.evaluation_score or 0.0,
        active_version=promoted.version,
        reasons=[],
    )
