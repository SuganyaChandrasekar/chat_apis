import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class FraudBase(DeclarativeBase):
    pass


class FraudCase(FraudBase):
    __tablename__ = "fraud_cases"
    __table_args__ = (
        CheckConstraint("status = 'OPEN'", name="ck_fraud_cases_status"),
        UniqueConstraint(
            "user_id",
            "idempotency_key",
            name="uq_fraud_cases_user_idempotency_key",
        ),
    )

    case_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="OPEN", server_default="OPEN"
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    frozen_card_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class FraudCaseStoreUnavailable(Exception):
    pass


class FraudCaseIdempotencyConflict(Exception):
    pass


@dataclass(frozen=True)
class FraudCaseRecord:
    case_id: str
    user_id: str
    status: str
    reason: str
    frozen_card_ids: list[str]
    idempotency_key: str | None
    created_at: datetime


def _database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise FraudCaseStoreUnavailable
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url


@lru_cache(maxsize=1)
def _engine():
    return create_engine(
        _database_url(),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
    )


def _record(case: FraudCase) -> FraudCaseRecord:
    return FraudCaseRecord(
        case_id=case.case_id,
        user_id=case.user_id,
        status=case.status,
        reason=case.reason,
        frozen_card_ids=list(case.frozen_card_ids),
        idempotency_key=case.idempotency_key,
        created_at=case.created_at,
    )


def _matching_idempotent_case(
    session: Session,
    user_id: str,
    idempotency_key: str,
    reason: str,
    frozen_card_ids: list[str],
) -> FraudCaseRecord | None:
    existing = session.scalar(
        select(FraudCase).where(
            FraudCase.user_id == user_id,
            FraudCase.idempotency_key == idempotency_key,
        )
    )
    if existing is None:
        return None
    if existing.reason != reason or existing.frozen_card_ids != frozen_card_ids:
        raise FraudCaseIdempotencyConflict
    return _record(existing)


def get_idempotent_fraud_case(
    *,
    user_id: str,
    idempotency_key: str,
) -> FraudCaseRecord | None:
    try:
        with Session(_engine()) as session:
            existing = session.scalar(
                select(FraudCase).where(
                    FraudCase.user_id == user_id,
                    FraudCase.idempotency_key == idempotency_key,
                )
            )
            return _record(existing) if existing is not None else None
    except Exception:
        raise FraudCaseStoreUnavailable from None


def create_fraud_case(
    *,
    user_id: str,
    reason: str,
    frozen_card_ids: list[str],
    idempotency_key: str | None,
) -> FraudCaseRecord:
    try:
        with Session(_engine()) as session:
            if idempotency_key:
                existing = _matching_idempotent_case(
                    session, user_id, idempotency_key, reason, frozen_card_ids
                )
                if existing is not None:
                    return existing

            now = datetime.now(timezone.utc)
            fraud_case = FraudCase(
                case_id=f"FRD-{now.year}-{uuid4().hex[:8].upper()}",
                user_id=user_id,
                status="OPEN",
                reason=reason,
                frozen_card_ids=frozen_card_ids,
                idempotency_key=idempotency_key,
                created_at=now,
            )
            session.add(fraud_case)
            session.commit()
            return _record(fraud_case)
    except FraudCaseIdempotencyConflict:
        raise
    except IntegrityError:
        if not idempotency_key:
            raise FraudCaseStoreUnavailable from None
        try:
            with Session(_engine()) as session:
                existing = _matching_idempotent_case(
                    session, user_id, idempotency_key, reason, frozen_card_ids
                )
                if existing is not None:
                    return existing
        except FraudCaseIdempotencyConflict:
            raise
        except Exception:
            raise FraudCaseStoreUnavailable from None
        raise FraudCaseStoreUnavailable from None
    except Exception:
        raise FraudCaseStoreUnavailable from None
