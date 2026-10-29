import pytest
from sqlalchemy import inspect

from services.security.audit import (
    AuditBase,
    AuditStoreUnavailable,
    SecurityAuditEvent,
    append_audit_event,
    create_audit_event,
    record_application_error,
)


def test_security_audit_model_matches_postgresql_schema() -> None:
    table = SecurityAuditEvent.__table__

    assert table.name == "security_audit_events"
    assert table.c.event_id.primary_key
    assert table.c.occurred_at.nullable is False
    assert table.c.event_type.nullable is False
    assert table.c.severity.nullable is False
    assert table.c.request_path.nullable is False
    assert table.c.source.nullable is False

    assert set(AuditBase.metadata.tables) == {
        "security_audit_events",
        "transfer_audit_events",
    }


def test_create_audit_event_validates_fields() -> None:
    event = create_audit_event(
        event_type="SECURITY",
        severity="WARNING",
        request_path="/chat",
        user_id="user-123",
        request_id="request-123",
        error_type=None,
        error_message=None,
        details={"ip_address": "198.51.100.10"},
    )

    assert event.user_id == "user-123"
    assert event.request_path == "/chat"
    assert event.details == {"ip_address": "198.51.100.10"}

    with pytest.raises(ValueError):
        create_audit_event(
            event_type="SECURITY",
            severity="WARNING",
            request_path="",
        )


def test_audit_details_redact_sensitive_values() -> None:
    event = create_audit_event(
        event_type="SECURITY",
        severity="WARNING",
        request_path="/chat",
        details={
            "authorization": "Bearer secret-token",
            "password": "plaintext-password",
            "request": {"otp": "123456"},
        },
    )

    assert event.details == {
        "authorization": "[REDACTED]",
        "password": "[REDACTED]",
        "request": {"otp": "[REDACTED]"},
    }


def test_record_application_error_uses_safe_metadata(monkeypatch) -> None:
    captured = {}

    def fake_append(event) -> None:
        captured["event"] = event

    monkeypatch.setattr(
        "services.security.audit.append_audit_event",
        fake_append,
    )

    event = record_application_error(
        exception=RuntimeError("database unavailable"),
        request_path="/chat",
        user_id="user-123",
        request_id="request-123",
        details={"retry_count": 1},
    )

    assert event.event_type == "APPLICATION_ERROR"
    assert event.severity == "CRITICAL"
    assert event.error_type == "RuntimeError"
    assert event.error_message == "database unavailable"
    assert captured["event"] is event


def test_append_audit_event_fails_closed_without_database(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "")

    with pytest.raises(AuditStoreUnavailable):
        append_audit_event(
            create_audit_event(
                event_type="SECURITY",
                severity="ERROR",
                request_path="/chat",
            )
        )
