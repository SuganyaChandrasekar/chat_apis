import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import logging
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Index,
    JSON,
    String,
    create_engine,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

logger = logging.getLogger("uvicorn.error")


class CapabilityBase(DeclarativeBase):
    pass


class CapabilityVersion(CapabilityBase):
    __tablename__ = "capability_versions"
    __table_args__ = (
        CheckConstraint(
            "certification_status != 'CERTIFIED' OR "
            "(guardrails_passed AND evaluation_score >= 0.95)",
            name="ck_capability_certification_gate",
        ),
        CheckConstraint(
            "rollout_status NOT IN ('BETA', 'PRODUCTION') OR "
            "(certification_status = 'CERTIFIED' AND guardrails_passed "
            "AND feature_flag_enabled AND evaluation_score >= 0.95)",
            name="ck_capability_routable_certified",
        ),
        Index(
            "uq_capability_versions_production",
            "capability_id",
            unique=True,
            postgresql_where=text("rollout_status = 'PRODUCTION'"),
            sqlite_where=text("rollout_status = 'PRODUCTION'"),
        ),
    )

    capability_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    version: Mapped[str] = mapped_column(String(128), primary_key=True)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    certification_status: Mapped[str] = mapped_column(String(16), nullable=False)
    guardrails_passed: Mapped[bool] = mapped_column(nullable=False)
    evaluation_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rollout_status: Mapped[str] = mapped_column(String(16), nullable=False)
    feature_flag: Mapped[str] = mapped_column(String(320), nullable=False)
    feature_flag_enabled: Mapped[bool] = mapped_column(nullable=False)
    cohort: Mapped[str] = mapped_column(String(64), nullable=False)
    general_traffic_percentage: Mapped[int] = mapped_column(nullable=False)
    reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class CapabilityStoreUnavailable(Exception):
    pass


class CapabilityVersionConflict(Exception):
    pass


class CapabilityPromotionBlocked(Exception):
    def __init__(self, reasons: list[str], active_version: str | None):
        super().__init__("Capability promotion was blocked")
        self.reasons = reasons
        self.active_version = active_version


class CapabilityNotFound(Exception):
    pass


@dataclass(frozen=True)
class CapabilityRecord:
    capability_id: str
    version: str
    manifest: dict[str, Any]
    certification_status: str
    guardrails_passed: bool
    evaluation_score: float | None
    rollout_status: str
    feature_flag: str
    feature_flag_enabled: bool
    cohort: str
    general_traffic_percentage: int
    reasons: list[str]
    created_at: datetime
    promoted_at: datetime | None


def _database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise CapabilityStoreUnavailable
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


def _record(version: CapabilityVersion) -> CapabilityRecord:
    return CapabilityRecord(
        capability_id=version.capability_id,
        version=version.version,
        manifest=version.manifest,
        certification_status=version.certification_status,
        guardrails_passed=version.guardrails_passed,
        evaluation_score=version.evaluation_score,
        rollout_status=version.rollout_status,
        feature_flag=version.feature_flag,
        feature_flag_enabled=version.feature_flag_enabled,
        cohort=version.cohort,
        general_traffic_percentage=version.general_traffic_percentage,
        reasons=list(version.reasons),
        created_at=version.created_at,
        promoted_at=version.promoted_at,
    )


def create_version(
    *,
    manifest: dict[str, Any],
    guardrails_passed: bool,
    evaluation_score: float | None,
    reasons: list[str],
) -> CapabilityRecord:
    capability_id = manifest["capability_id"]
    version = manifest["version"]
    certified = (
        guardrails_passed
        and evaluation_score is not None
        and evaluation_score >= 0.95
    )
    record = CapabilityVersion(
        capability_id=capability_id,
        version=version,
        manifest=manifest,
        certification_status="CERTIFIED" if certified else "FAILED",
        guardrails_passed=guardrails_passed,
        evaluation_score=evaluation_score,
        rollout_status="BETA" if certified else "BLOCKED",
        feature_flag=f"capability.{capability_id}.{version}",
        feature_flag_enabled=certified,
        cohort="internal_beta",
        general_traffic_percentage=0,
        reasons=reasons,
    )
    try:
        with Session(_engine()) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return _record(record)
    except IntegrityError:
        raise CapabilityVersionConflict from None
    except SQLAlchemyError as error:
        logger.error(
            "Capability version registration failed (%s)", type(error).__name__
        )
        raise CapabilityStoreUnavailable from None


def get_version(capability_id: str, version: str) -> CapabilityRecord | None:
    try:
        with Session(_engine()) as session:
            record = session.get(CapabilityVersion, (capability_id, version))
            return _record(record) if record is not None else None
    except SQLAlchemyError as error:
        logger.error("Capability version lookup failed (%s)", type(error).__name__)
        raise CapabilityStoreUnavailable from None


def promote_version(capability_id: str, version: str) -> CapabilityRecord:
    try:
        with Session(_engine()) as session:
            candidate = session.scalar(
                select(CapabilityVersion)
                .where(
                    CapabilityVersion.capability_id == capability_id,
                    CapabilityVersion.version == version,
                )
                .with_for_update()
            )
            if candidate is None:
                raise CapabilityNotFound

            active = session.scalar(
                select(CapabilityVersion)
                .where(
                    CapabilityVersion.capability_id == capability_id,
                    CapabilityVersion.rollout_status == "PRODUCTION",
                )
                .with_for_update()
            )
            reasons: list[str] = []
            if candidate.certification_status != "CERTIFIED":
                reasons.append("capability_is_not_certified")
            if not candidate.guardrails_passed:
                reasons.append("guardrails_failed")
            if candidate.evaluation_score is None or candidate.evaluation_score < 0.95:
                reasons.append("evaluation_below_0.95")
            if active is not None and active.version != candidate.version:
                if (
                    candidate.evaluation_score is not None
                    and active.evaluation_score is not None
                    and active.evaluation_score > 0
                    and (active.evaluation_score - candidate.evaluation_score)
                    / active.evaluation_score
                    > 0.02
                ):
                    reasons.append("evaluation_regression_exceeds_2_percent")
                if (
                    candidate.evaluation_score is None
                    or active.evaluation_score is None
                    or candidate.evaluation_score < active.evaluation_score
                ):
                    reasons.append("evaluation_below_active_version")
            if reasons:
                raise CapabilityPromotionBlocked(
                    reasons, active.version if active is not None else None
                )
            if active is not None and active.version == candidate.version:
                return _record(candidate)

            candidate.rollout_status = "PRODUCTION"
            candidate.feature_flag_enabled = True
            candidate.cohort = "general"
            candidate.general_traffic_percentage = 100
            candidate.promoted_at = datetime.now(timezone.utc)
            if active is not None:
                active.rollout_status = "RETIRED"
                active.feature_flag_enabled = False
                active.cohort = "retired"
                active.general_traffic_percentage = 0
            session.commit()
            session.refresh(candidate)
            return _record(candidate)
    except CapabilityNotFound:
        raise
    except IntegrityError:
        raise CapabilityVersionConflict from None
    except SQLAlchemyError as error:
        logger.error("Capability version promotion failed (%s)", type(error).__name__)
        raise CapabilityStoreUnavailable from None


def resolve_capability(
    capability_id: str, *, cohort: str = "general"
) -> CapabilityRecord | None:
    if cohort not in {"general", "internal_beta"}:
        return None
    try:
        with Session(_engine()) as session:
            rollout_status = "PRODUCTION" if cohort == "general" else "BETA"
            record = session.scalar(
                select(CapabilityVersion)
                .where(
                    CapabilityVersion.capability_id == capability_id,
                    CapabilityVersion.rollout_status == rollout_status,
                    CapabilityVersion.cohort == cohort,
                    CapabilityVersion.certification_status == "CERTIFIED",
                    CapabilityVersion.guardrails_passed.is_(True),
                    CapabilityVersion.feature_flag_enabled.is_(True),
                    CapabilityVersion.general_traffic_percentage
                    == (100 if cohort == "general" else 0),
                )
                .order_by(CapabilityVersion.created_at.desc())
            )
            return _record(record) if record is not None else None
    except SQLAlchemyError as error:
        logger.error("Capability route resolution failed (%s)", type(error).__name__)
        raise CapabilityStoreUnavailable from None


def list_routable_capabilities(
    *, cohort: str = "general"
) -> list[CapabilityRecord]:
    if cohort not in {"general", "internal_beta"}:
        return []
    try:
        with Session(_engine()) as session:
            rollout_status = "PRODUCTION" if cohort == "general" else "BETA"
            versions = session.scalars(
                select(CapabilityVersion)
                .where(
                    CapabilityVersion.rollout_status == rollout_status,
                    CapabilityVersion.cohort == cohort,
                    CapabilityVersion.certification_status == "CERTIFIED",
                    CapabilityVersion.guardrails_passed.is_(True),
                    CapabilityVersion.feature_flag_enabled.is_(True),
                    CapabilityVersion.general_traffic_percentage
                    == (100 if cohort == "general" else 0),
                )
                .order_by(CapabilityVersion.capability_id)
            )
            return [_record(version) for version in versions]
    except SQLAlchemyError as error:
        logger.error(
            "Capability catalog lookup failed (%s)", type(error).__name__
        )
        raise CapabilityStoreUnavailable from None
