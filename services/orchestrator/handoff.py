import os
from datetime import datetime, timezone
from functools import lru_cache
from threading import RLock
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    DateTime,
    Float,
    JSON,
    String,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class HandoffTranscriptMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class HandoffPackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    package_id: str = Field(min_length=32, max_length=32)
    session_id: str = Field(min_length=1, max_length=36)
    user_id: str = Field(min_length=1, max_length=256)
    transcript: list[HandoffTranscriptMessage] = Field(max_length=20)
    detected_intent: str = Field(min_length=1, max_length=64)
    distress_score: float = Field(ge=0.0, le=1.0)
    identity_consent_state: Literal["consented"]
    actions_taken: list[str] = Field(max_length=20)


class HandoffDeliveryStatus(BaseModel):
    status: Literal["not_configured"] = "not_configured"


class HandoffPackageResponse(HandoffPackage):
    delivery: HandoffDeliveryStatus


class HandoffPackageStoreUnavailable(Exception):
    pass


class HandoffBase(DeclarativeBase):
    pass


class HandoffPackageRow(HandoffBase):
    __tablename__ = "handoff_packages"
    __table_args__ = (
        UniqueConstraint("session_id", name="uq_handoff_packages_session_id"),
    )

    package_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False)
    user_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    transcript: Mapped[list[dict[str, str]]] = mapped_column(JSON, nullable=False)
    detected_intent: Mapped[str] = mapped_column(String(64), nullable=False)
    distress_score: Mapped[float] = mapped_column(Float, nullable=False)
    identity_consent_state: Mapped[str] = mapped_column(String(32), nullable=False)
    actions_taken: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


_mock_packages: dict[str, HandoffPackage] = {}
_mock_lock = RLock()


def _database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise HandoffPackageStoreUnavailable
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url


@lru_cache(maxsize=1)
def _engine():
    return create_engine(
        _database_url(), pool_pre_ping=True, pool_size=5, max_overflow=5
    )


def _package_from_row(row: HandoffPackageRow) -> HandoffPackage:
    return HandoffPackage(
        package_id=row.package_id,
        session_id=row.session_id,
        user_id=row.user_id,
        transcript=row.transcript,
        detected_intent=row.detected_intent,
        distress_score=row.distress_score,
        identity_consent_state=row.identity_consent_state,
        actions_taken=row.actions_taken,
    )


def get_handoff_package(*, session_id: str, user_id: str) -> HandoffPackage | None:
    if not os.getenv("DATABASE_URL"):
        with _mock_lock:
            package = _mock_packages.get(session_id)
        return package if package and package.user_id == user_id else None
    try:
        with Session(_engine()) as session:
            row = session.scalar(
                select(HandoffPackageRow).where(
                    HandoffPackageRow.session_id == session_id,
                    HandoffPackageRow.user_id == user_id,
                )
            )
            return _package_from_row(row) if row is not None else None
    except Exception:
        raise HandoffPackageStoreUnavailable from None


def create_handoff_package(package: HandoffPackage) -> HandoffPackage:
    if os.getenv("SANDBOX_MODE", "false").lower() != "true":
        raise HandoffPackageStoreUnavailable
    if not os.getenv("DATABASE_URL"):
        with _mock_lock:
            existing = _mock_packages.get(package.session_id)
            if existing is not None:
                if existing.user_id != package.user_id:
                    raise HandoffPackageStoreUnavailable
                return existing
            _mock_packages[package.session_id] = package
        return package

    try:
        with Session(_engine()) as session:
            existing = session.scalar(
                select(HandoffPackageRow).where(
                    HandoffPackageRow.session_id == package.session_id
                )
            )
            if existing is not None:
                if existing.user_id != package.user_id:
                    raise HandoffPackageStoreUnavailable
                return _package_from_row(existing)
            row = HandoffPackageRow(
                package_id=package.package_id,
                session_id=package.session_id,
                user_id=package.user_id,
                transcript=[
                    message.model_dump() for message in package.transcript
                ],
                detected_intent=package.detected_intent,
                distress_score=package.distress_score,
                identity_consent_state=package.identity_consent_state,
                actions_taken=package.actions_taken,
                created_at=datetime.now(timezone.utc),
            )
            session.add(row)
            session.commit()
            return _package_from_row(row)
    except HandoffPackageStoreUnavailable:
        raise
    except IntegrityError:
        existing = get_handoff_package(
            session_id=package.session_id, user_id=package.user_id
        )
        if existing is not None:
            return existing
        raise HandoffPackageStoreUnavailable from None
    except Exception:
        raise HandoffPackageStoreUnavailable from None


def new_package_id() -> str:
    return uuid4().hex
