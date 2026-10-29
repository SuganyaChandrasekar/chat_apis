from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import CheckConstraint, DateTime, Index, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from services.transfer.audit import AuditBase


class SecurityAuditEvent(AuditBase):
    """Immutable security and critical application audit events."""

    __tablename__ = "security_audit_events"
    __table_args__ = (
        CheckConstraint(
            "severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')",
            name="ck_security_audit_severity",
        ),
        CheckConstraint(
            "event_type IN ('SECURITY', 'APPLICATION_ERROR')",
            name="ck_security_audit_event_type",
        ),
        Index(
            "ix_security_audit_events_occurred_at",
            "occurred_at",
        ),
        Index(
            "ix_security_audit_events_user_id",
            "user_id",
        ),
        Index(
            "ix_security_audit_events_request_path",
            "request_path",
        ),
        Index(
            "ix_security_audit_events_event_type_severity",
            "event_type",
            "severity",
        ),
    )

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    request_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    source: Mapped[str] = mapped_column(String(128), nullable=False)


class AuditStoreUnavailable(Exception):
    pass


def _database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise AuditStoreUnavailable
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url


def _redact_sensitive_values(value: Any) -> Any:
    sensitive_key_pattern = re.compile(
        r"(?:password|passwd|secret|token|authorization|credential|otp|ssn|pan)",
        re.IGNORECASE,
    )

    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if sensitive_key_pattern.search(str(key))
            else _redact_sensitive_values(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_sensitive_values(item) for item in value)
    return value


def create_audit_event(
    *,
    event_type: str,
    severity: str,
    request_path: str,
    user_id: str | None = None,
    request_id: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    details: dict[str, Any] | None = None,
    source: str = "application",
) -> SecurityAuditEvent:
    """Build a validated audit event without exposing raw exception traces."""

    if event_type not in {"SECURITY", "APPLICATION_ERROR"}:
        raise ValueError("Unsupported audit event type")
    if severity not in {"INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("Unsupported audit severity")
    if not request_path or len(request_path) > 2048:
        raise ValueError("Request path is required and must be at most 2048 characters")
    if user_id is not None and len(user_id) > 256:
        raise ValueError("User ID must be at most 256 characters")
    if request_id is not None and len(request_id) > 128:
        raise ValueError("Request ID must be at most 128 characters")
    if error_type is not None and len(error_type) > 255:
        raise ValueError("Error type must be at most 255 characters")
    if error_message is not None and len(error_message) > 4000:
        raise ValueError("Error message must be at most 4000 characters")
    if source is not None and len(source) > 128:
        raise ValueError("Audit source must be at most 128 characters")

    return SecurityAuditEvent(
        event_id=str(uuid4()),
        occurred_at=datetime.now(timezone.utc),
        event_type=event_type,
        severity=severity,
        user_id=user_id,
        request_path=request_path,
        request_id=request_id,
        error_type=error_type,
        error_message=error_message,
        details=_redact_sensitive_values(details),
        source=source,
    )


def append_audit_event(event: SecurityAuditEvent) -> None:
    """Persist one audit event using the configured PostgreSQL database."""

    try:
        with Session(create_engine(_database_url(), pool_pre_ping=True)) as session:
            session.add(event)
            session.commit()
    except Exception as exc:
        raise AuditStoreUnavailable from exc


def record_application_error(
    *,
    exception: BaseException,
    request_path: str,
    user_id: str | None = None,
    request_id: str | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    details: dict[str, Any] | None = None,
    source: str = "application",
) -> SecurityAuditEvent:
    """Record a critical application error without storing a traceback."""

    event = create_audit_event(
        event_type="APPLICATION_ERROR",
        severity="CRITICAL",
        request_path=request_path,
        user_id=user_id,
        request_id=request_id,
        error_type=type(exception).__name__,
        error_message=error_message if error_message is not None else str(exception)[:4000],
        details={**(details or {}), **({"error_code": error_code} if error_code else {})},
        source=source,
    )
    append_audit_event(event)
    return event


def record_security_event(
    *,
    exception: BaseException,
    request_path: str,
    user_id: str | None = None,
    request_id: str | None = None,
    error_code: str,
    error_message: str | None = None,
    details: dict[str, Any] | None = None,
    source: str = "application",
) -> SecurityAuditEvent:
    """Record a security failure without storing raw exception details."""

    event = create_audit_event(
        event_type="SECURITY",
        severity="ERROR",
        request_path=request_path,
        user_id=user_id,
        request_id=request_id,
        error_type=type(exception).__name__,
        error_message=error_message if error_message is not None else str(exception)[:4000],
        details={**(details or {}), "error_code": error_code},
        source=source,
    )
    append_audit_event(event)
    return event
