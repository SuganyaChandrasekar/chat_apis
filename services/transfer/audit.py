import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Numeric,
    String,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class AuditBase(DeclarativeBase):
    pass


class TransferAuditEvent(AuditBase):
    __tablename__ = "transfer_audit_events"
    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING_VALIDATION', 'SUCCESS', 'FAILED')",
            name="ck_transfer_audit_status",
        ),
        UniqueConstraint(
            "account_id", "idempotency_key", "status", name="uq_transfer_audit_state"
        ),
    )

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String(36), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), index=True)
    account_id: Mapped[str] = mapped_column(String(256), index=True)
    source_account_id: Mapped[str] = mapped_column(String(256))
    destination_account_id: Mapped[str] = mapped_column(String(256))
    amount: Mapped[object] = mapped_column(Numeric(12, 2))
    currency: Mapped[str] = mapped_column(String(3))
    status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class AuditStoreUnavailable(Exception):
    pass


class IdempotencyConflict(Exception):
    pass


@dataclass(frozen=True)
class TransferStart:
    transaction_id: str
    status: str
    replayed: bool


@lru_cache(maxsize=1)
def _engine():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise AuditStoreUnavailable
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return create_engine(database_url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def _matching_event(
    session: Session,
    account_id: str,
    idempotency_key: str,
) -> TransferAuditEvent | None:
    return session.scalar(
        select(TransferAuditEvent)
        .where(
            TransferAuditEvent.account_id == account_id,
            TransferAuditEvent.idempotency_key == idempotency_key,
        )
        .order_by(TransferAuditEvent.created_at.desc())
        .limit(1)
    )


def begin_transfer(
    *,
    account_id: str,
    idempotency_key: str,
    source_account_id: str,
    destination_account_id: str,
    amount,
    currency: str,
) -> TransferStart:
    try:
        with Session(_engine()) as session:
            existing = _matching_event(session, account_id, idempotency_key)
            if existing is not None:
                if (
                    existing.source_account_id != source_account_id
                    or existing.destination_account_id != destination_account_id
                    or existing.amount != amount
                    or existing.currency != currency
                ):
                    raise IdempotencyConflict
                if existing.status == "SUCCESS":
                    return TransferStart(existing.transaction_id, "SUCCESS", True)
                raise IdempotencyConflict

            transaction_id = str(uuid4())
            session.add(
                TransferAuditEvent(
                    event_id=str(uuid4()),
                    transaction_id=transaction_id,
                    idempotency_key=idempotency_key,
                    account_id=account_id,
                    source_account_id=source_account_id,
                    destination_account_id=destination_account_id,
                    amount=amount,
                    currency=currency,
                    status="PENDING_VALIDATION",
                )
            )
            session.commit()
            return TransferStart(transaction_id, "PENDING_VALIDATION", False)
    except IdempotencyConflict:
        raise
    except IntegrityError:
        raise IdempotencyConflict from None
    except Exception:
        raise AuditStoreUnavailable from None


def append_transfer_status(
    *,
    transaction_id: str,
    account_id: str,
    idempotency_key: str,
    source_account_id: str,
    destination_account_id: str,
    amount,
    currency: str,
    status: str,
) -> None:
    try:
        with Session(_engine()) as session:
            session.add(
                TransferAuditEvent(
                    event_id=str(uuid4()),
                    transaction_id=transaction_id,
                    idempotency_key=idempotency_key,
                    account_id=account_id,
                    source_account_id=source_account_id,
                    destination_account_id=destination_account_id,
                    amount=amount,
                    currency=currency,
                    status=status,
                )
            )
            session.commit()
    except Exception:
        raise AuditStoreUnavailable from None