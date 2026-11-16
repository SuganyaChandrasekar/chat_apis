from alembic import op
import sqlalchemy as sa


revision = "20261113_0001"
down_revision = "20261106_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "capability_versions",
        sa.Column("capability_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("certification_status", sa.String(length=16), nullable=False),
        sa.Column("guardrails_passed", sa.Boolean(), nullable=False),
        sa.Column("evaluation_score", sa.Float(), nullable=True),
        sa.Column("rollout_status", sa.String(length=16), nullable=False),
        sa.Column("feature_flag", sa.String(length=320), nullable=False),
        sa.Column("feature_flag_enabled", sa.Boolean(), nullable=False),
        sa.Column("cohort", sa.String(length=64), nullable=False),
        sa.Column("general_traffic_percentage", sa.Integer(), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("capability_id", "version"),
        sa.CheckConstraint(
            "certification_status IN ('CERTIFIED', 'FAILED')",
            name="ck_capability_certification_status",
        ),
        sa.CheckConstraint(
            "rollout_status IN ('BETA', 'PRODUCTION', 'BLOCKED', 'RETIRED')",
            name="ck_capability_rollout_status",
        ),
        sa.CheckConstraint(
            "general_traffic_percentage >= 0 AND general_traffic_percentage <= 100",
            name="ck_capability_general_traffic_percentage",
        ),
        sa.CheckConstraint(
            "evaluation_score IS NULL OR "
            "(evaluation_score >= 0 AND evaluation_score <= 1)",
            name="ck_capability_evaluation_score",
        ),
        sa.CheckConstraint(
            "certification_status != 'CERTIFIED' OR "
            "(guardrails_passed AND evaluation_score >= 0.95)",
            name="ck_capability_certification_gate",
        ),
        sa.CheckConstraint(
            "rollout_status NOT IN ('BETA', 'PRODUCTION') OR "
            "(certification_status = 'CERTIFIED' AND guardrails_passed "
            "AND feature_flag_enabled AND evaluation_score >= 0.95)",
            name="ck_capability_routable_certified",
        ),
    )
    op.create_index(
        "uq_capability_versions_production",
        "capability_versions",
        ["capability_id"],
        unique=True,
        postgresql_where=sa.text("rollout_status = 'PRODUCTION'"),
        sqlite_where=sa.text("rollout_status = 'PRODUCTION'"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_capability_versions_production", table_name="capability_versions"
    )
    op.drop_table("capability_versions")
