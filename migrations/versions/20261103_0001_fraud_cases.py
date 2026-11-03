from alembic import op
import sqlalchemy as sa


revision = "20261103_0001"
down_revision = "20261102_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "fraud_cases",
        sa.Column("case_id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="OPEN", nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("frozen_card_ids", sa.JSON(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status = 'OPEN'", name="ck_fraud_cases_status"),
        sa.PrimaryKeyConstraint("case_id"),
        sa.UniqueConstraint(
            "user_id",
            "idempotency_key",
            name="uq_fraud_cases_user_idempotency_key",
        ),
    )
    op.create_index("ix_fraud_cases_user_id", "fraud_cases", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_fraud_cases_user_id", table_name="fraud_cases")
    op.drop_table("fraud_cases")
