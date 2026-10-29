from __future__ import annotations

import logging
import uuid
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from services.security.audit import (
    AuditStoreUnavailable,
    record_application_error,
    record_security_event,
)

logger = logging.getLogger(__name__)


class ApplicationError(Exception):
    """A controlled application failure with a stable client-facing code."""

    def __init__(self, message: str, *, status_code: int = 500, code: str = "APPLICATION_ERROR") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class SecurityError(Exception):
    """A controlled security failure that should be recorded as an audit event."""

    def __init__(self, message: str, *, status_code: int = 403, code: str = "SECURITY_ERROR") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class ApiErrorResponse:
    """Stable JSON shape returned to API clients."""

    def as_dict(self, code: str, message: str, request_id: str | None = None) -> dict[str, Any]:
        return {
            "error": {
                "code": code,
                "message": message,
                **({"request_id": request_id} if request_id else {}),
            }
        }


def _request_id(request: Request) -> str:
    request_id = request.headers.get("x-request-id", "").strip()
    if not request_id:
        request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    return request_id


async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Populate a request ID and convert route failures into stable responses."""

    request_id = _request_id(request)
    try:
        response = await call_next(request)
    except HTTPException as exception:
        response = _http_exception_response(request, exception)
    except ApplicationError as exception:
        response = _exception_response(
            request,
            exception,
            code=exception.code,
            message=str(exception),
            status_code=exception.status_code,
            event_type="APPLICATION_ERROR",
            severity="ERROR",
        )
    except SecurityError as exception:
        response = _exception_response(
            request,
            exception,
            code=exception.code,
            message=str(exception),
            status_code=exception.status_code,
            event_type="SECURITY",
            severity="ERROR",
        )
    except Exception as exception:
        response = _exception_response(
            request,
            exception,
            code="INTERNAL_SERVER_ERROR",
            message="An internal error occurred",
            status_code=500,
            event_type="APPLICATION_ERROR",
            severity="CRITICAL",
        )
    response.headers["x-request-id"] = request_id
    return response


def _safe_request_details(request: Request) -> dict[str, Any]:
    return {
        "method": request.method,
        "path": request.url.path,
        "request_id": getattr(request.state, "request_id", None),
    }


def _audit_exception(
    exception: BaseException,
    request: Request,
    *,
    event_type: str,
    error_code: str,
    severity: str,
) -> None:
    try:
        if event_type == "SECURITY":
            record_security_event(
                exception=exception,
                request_path=request.url.path,
                request_id=getattr(request.state, "request_id", None),
                error_code=error_code,
                details=_safe_request_details(request),
                source="api_exception_handler",
            )
        else:
            record_application_error(
                exception=exception,
                request_path=request.url.path,
                request_id=getattr(request.state, "request_id", None),
                error_code=error_code,
                error_message="An internal error occurred",
                details=_safe_request_details(request),
                source="api_exception_handler",
            )
    except Exception:
        logger.error(
            "Failed to record %s audit event for request %s",
            event_type,
            getattr(request.state, "request_id", None),
        )


def _exception_response(
    request: Request,
    exception: Exception,
    *,
    code: str,
    message: str,
    status_code: int,
    event_type: str,
    severity: str,
) -> JSONResponse:
    _audit_exception(
        exception,
        request,
        event_type=event_type,
        error_code=code,
        severity=severity,
    )
    return JSONResponse(
        status_code=status_code,
        content=ApiErrorResponse().as_dict(code, message, request.state.request_id),
    )


def _http_exception_response(request: Request, exception: HTTPException) -> JSONResponse:
    if exception.status_code in {401, 403}:
        _audit_exception(
            exception,
            request,
            event_type="SECURITY",
            error_code=f"HTTP_{exception.status_code}",
            severity="ERROR",
        )
    return JSONResponse(
        status_code=exception.status_code,
        content={"detail": exception.detail},
        headers=dict(exception.headers or {}),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register stable, audited exception handling for a service application."""

    app.middleware("http")(request_id_middleware)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exception: HTTPException) -> JSONResponse:
        if exception.status_code in {401, 403}:
            _audit_exception(
                exception,
                request,
                event_type="SECURITY",
                error_code=f"HTTP_{exception.status_code}",
                severity="ERROR",
            )
        return JSONResponse(
            status_code=exception.status_code,
            content={"detail": exception.detail},
            headers=dict(exception.headers or {}),
        )

    @app.exception_handler(ApplicationError)
    async def application_error_handler(
        request: Request, exception: ApplicationError
    ) -> JSONResponse:
        return _exception_response(
            request,
            exception,
            code=exception.code,
            message=str(exception),
            status_code=exception.status_code,
            event_type="APPLICATION_ERROR",
            severity="ERROR",
        )

    @app.exception_handler(SecurityError)
    async def security_error_handler(request: Request, exception: SecurityError) -> JSONResponse:
        return _exception_response(
            request,
            exception,
            code=exception.code,
            message=str(exception),
            status_code=exception.status_code,
            event_type="SECURITY",
            severity="ERROR",
        )

    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, exception: Exception) -> JSONResponse:
        return _exception_response(
            request,
            exception,
            code="INTERNAL_SERVER_ERROR",
            message="An internal error occurred",
            status_code=500,
            event_type="APPLICATION_ERROR",
            severity="CRITICAL",
        )
