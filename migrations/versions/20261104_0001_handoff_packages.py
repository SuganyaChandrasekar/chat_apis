from alembic import op
import sqlalchemy as sa


revision = "20261104_0001"
down_revision = "20261103_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "handoff_packages",
        sa.Column("package_id", sa.String(length=32), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("transcript", sa.JSON(), nullable=False),
        sa.Column("detected_intent", sa.String(length=64), nullable=False),
        sa.Column("distress_score", sa.Float(), nullable=False),
        sa.Column("identity_consent_state", sa.String(length=32), nullable=False),
        sa.Column("actions_taken", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("package_id"),
        sa.UniqueConstraint("session_id", name="uq_handoff_packages_session_id"),
    )
    op.create_index("ix_handoff_packages_user_id", "handoff_packages", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_handoff_packages_user_id", table_name="handoff_packages")
    op.drop_table("handoff_packages")
