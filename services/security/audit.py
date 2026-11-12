from __future__ import annotations

import os
import re
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Identity,
    Index,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from services.orchestrator.pii import PiiMaskingUnavailable, anonymize_message
from services.transfer.audit import AuditBase


_audit_context: ContextVar[tuple[str | None, str | None]] = ContextVar(
    "audit_context", default=(None, None)
)
_SENSITIVE_VALUE_PATTERN = re.compile(
    r"\b(password|passcode|pin|otp|one[-_ ]time code|cvv|cvc|security[_ ]code|"
    r"access[_ ]token|refresh[_ ]token|bearer[_ ]token)\b\s*(?:is|:|=)?\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)
_SENSITIVE_FIELD_PATTERN = re.compile(
    r"(?:password|passwd|secret|token|authorization|credential|otp|pin|cvv|cvc|"
    r"security[_ -]?code|ssn|pan|account[_ -]?(?:number|no|id))",
    re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(
    r"\bBearer\s+(?!token\b)[A-Za-z0-9._~+/-]+=*", re.IGNORECASE
)
_TOKEN_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{30,})\b"
)
_JWT_PATTERN = re.compile(
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
)


def set_audit_context(
    correlation_id: str | None, conversation_id: str | None
) -> Token[tuple[str | None, str | None]]:
    return _audit_context.set((correlation_id, conversation_id))


def reset_audit_context(token: Token[tuple[str | None, str | None]]) -> None:
    _audit_context.reset(token)


def get_audit_context() -> tuple[str | None, str | None]:
    return _audit_context.get()


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
            "ix_security_audit_events_correlation",
            "correlation_id",
        ),
        Index(
            "ix_security_audit_events_conversation",
            "conversation_id",
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
    correlation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    conversation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    source: Mapped[str] = mapped_column(String(128), nullable=False)


class DecisionTraceEvent(AuditBase):
    """Append-only, PII-scrubbed events used to reconstruct chat decisions."""

    __tablename__ = "decision_trace_events"
    __table_args__ = (
        Index(
            "ix_decision_trace_conversation_sequence",
            "conversation_id",
            "sequence_id",
        ),
        Index(
            "ix_decision_trace_correlation",
            "correlation_id",
        ),
    )

    sequence_id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    conversation_id: Mapped[str] = mapped_column(String(36), nullable=False)
    correlation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
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
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if _SENSITIVE_FIELD_PATTERN.search(str(key))
            else _redact_sensitive_values(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_sensitive_values(item) for item in value)
    return value


def _redact_free_text(value: str) -> str:
    try:
        redacted = anonymize_message(value)
    except PiiMaskingUnavailable as exc:
        raise AuditStoreUnavailable from exc
    redacted = _BEARER_PATTERN.sub("******", redacted)
    redacted = _SENSITIVE_VALUE_PATTERN.sub(
        lambda match: f"{match.group(1)} [REDACTED]", redacted
    )
    redacted = _TOKEN_PATTERN.sub("[REDACTED_TOKEN]", redacted)
    return _JWT_PATTERN.sub("[REDACTED_TOKEN]", redacted)


def _redact_audit_text_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if _SENSITIVE_FIELD_PATTERN.search(str(key))
            else _redact_audit_text_values(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_audit_text_values(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_audit_text_values(item) for item in value)
    if isinstance(value, str):
        return _redact_free_text(value)
    return value


def _prepare_security_event_for_persistence(event: SecurityAuditEvent) -> None:
    if event.error_message is not None:
        event.error_message = _redact_free_text(event.error_message)
    if event.details is not None:
        event.details = _redact_audit_text_values(event.details)


def create_audit_event(
    *,
    event_type: str,
    severity: str,
    request_path: str,
    user_id: str | None = None,
    request_id: str | None = None,
    correlation_id: str | None = None,
    conversation_id: str | None = None,
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
    context_correlation_id, context_conversation_id = get_audit_context()
    correlation_id = correlation_id or context_correlation_id or request_id
    conversation_id = conversation_id or context_conversation_id
    if correlation_id is not None and len(correlation_id) > 128:
        raise ValueError("Correlation ID must be at most 128 characters")
    if conversation_id is not None and len(conversation_id) > 36:
        raise ValueError("Conversation ID must be at most 36 characters")
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
        request_id=request_id or correlation_id,
        correlation_id=correlation_id,
        conversation_id=conversation_id,
        error_type=error_type,
        error_message=error_message,
        details=_redact_sensitive_values(details),
        source=source,
    )


def append_audit_event(event: SecurityAuditEvent) -> None:
    """Persist one audit event using the configured PostgreSQL database."""

    try:
        database_url = _database_url()
        _prepare_security_event_for_persistence(event)
        with Session(create_engine(database_url, pool_pre_ping=True)) as session:
            session.add(event)
            session.commit()
    except Exception as exc:
        raise AuditStoreUnavailable from exc


def append_decision_trace_event(
    *,
    conversation_id: str,
    correlation_id: str,
    event_type: str,
    payload: dict[str, Any],
    source: str = "chat_orchestrator",
) -> None:
    if not conversation_id or len(conversation_id) > 36:
        raise ValueError("Conversation ID is required and must be at most 36 characters")
    if not correlation_id or len(correlation_id) > 128:
        raise ValueError("Correlation ID is required and must be at most 128 characters")
    if not event_type or len(event_type) > 64:
        raise ValueError("Trace event type is required and must be at most 64 characters")
    if len(source) > 128:
        raise ValueError("Trace source must be at most 128 characters")

    try:
        safe_payload = _redact_audit_text_values(payload)
        with Session(create_engine(_database_url(), pool_pre_ping=True)) as session:
            session.add(
                DecisionTraceEvent(
                    occurred_at=datetime.now(timezone.utc),
                    conversation_id=conversation_id,
                    correlation_id=correlation_id,
                    event_type=event_type,
                    payload=safe_payload,
                    source=source,
                )
            )
            session.commit()
    except Exception as exc:
        raise AuditStoreUnavailable from exc


def load_decision_trace(
    conversation_id: str, *, after_sequence: int = 0, limit: int = 1000
) -> list[dict[str, Any]]:
    if not conversation_id or len(conversation_id) > 36:
        raise ValueError("Conversation ID is required and must be at most 36 characters")
    if after_sequence < 0:
        raise ValueError("Trace sequence cursor must be non-negative")
    if not 1 <= limit <= 10000:
        raise ValueError("Trace limit must be between 1 and 10000")
    try:
        with Session(create_engine(_database_url(), pool_pre_ping=True)) as session:
            events = session.scalars(
                select(DecisionTraceEvent)
                .where(
                    DecisionTraceEvent.conversation_id == conversation_id,
                    DecisionTraceEvent.sequence_id > after_sequence,
                )
                .order_by(DecisionTraceEvent.sequence_id)
                .limit(limit)
            ).all()
            return [
                {
                    "sequence_id": event.sequence_id,
                    "occurred_at": event.occurred_at.isoformat(),
                    "conversation_id": event.conversation_id,
                    "correlation_id": event.correlation_id,
                    "event_type": event.event_type,
                    "payload": event.payload,
                    "source": event.source,
                }
                for event in events
            ]
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
