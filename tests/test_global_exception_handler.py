from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from services.api_errors import ApplicationError, SecurityError, register_exception_handlers


def _app_with_route(exception: Exception) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/fail")
    async def fail() -> None:
        raise exception

    return app


def test_global_handler_records_and_returns_sanitized_application_error(monkeypatch) -> None:
    recorded = {}

    def fake_record_application_error(**kwargs) -> None:
        recorded.update(kwargs)

    monkeypatch.setattr(
        "services.api_errors.record_application_error",
        fake_record_application_error,
    )
    app = _app_with_route(RuntimeError("password=secret-token"))

    response = TestClient(app).get("/fail")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert response.json()["error"]["message"] == "An internal error occurred"
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]
    assert recorded["exception"] is not None
    assert recorded["error_message"] == "An internal error occurred"
    assert recorded["request_path"] == "/fail"
    assert recorded["request_path"] == "/fail"
    assert "password" not in recorded["error_message"]
    assert "secret-token" not in recorded["error_message"]


def test_custom_security_error_is_audited_as_security_event(monkeypatch) -> None:
    recorded = {}

    def fake_record_security_event(**kwargs) -> None:
        recorded.update(kwargs)

    monkeypatch.setattr(
        "services.api_errors.record_security_event",
        fake_record_security_event,
    )
    app = _app_with_route(SecurityError("Access denied for this account"))

    response = TestClient(app).get("/fail")

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "SECURITY_ERROR"
    assert response.json()["error"]["message"] == "Access denied for this account"
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]
    assert recorded["exception"] is not None
    assert recorded["error_code"] == "SECURITY_ERROR"
    assert recorded["request_path"] == "/fail"


def test_custom_application_error_uses_controlled_status_and_code(monkeypatch) -> None:
    recorded = {}

    def fake_record_application_error(**kwargs) -> None:
        recorded.update(kwargs)

    monkeypatch.setattr(
        "services.api_errors.record_application_error",
        fake_record_application_error,
    )
    app = _app_with_route(
        ApplicationError(
            "Service temporarily unavailable",
            status_code=503,
            code="SERVICE_UNAVAILABLE",
        )
    )

    response = TestClient(app).get("/fail")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "SERVICE_UNAVAILABLE"
    assert response.json()["error"]["message"] == "Service temporarily unavailable"
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]
    assert recorded["error_code"] == "SERVICE_UNAVAILABLE"


def test_audit_failure_does_not_replace_client_response(monkeypatch) -> None:
    def fail_audit(**kwargs) -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        "services.api_errors.record_application_error",
        fail_audit,
    )
    app = _app_with_route(RuntimeError("unexpected failure"))

    response = TestClient(app).get("/fail")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_SERVER_ERROR"
