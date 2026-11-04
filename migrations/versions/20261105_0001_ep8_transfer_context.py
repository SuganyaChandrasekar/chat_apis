from alembic import op
import sqlalchemy as sa


revision = "20261105_0001"
down_revision = "20261104_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "transfer_audit_events",
        sa.Column(
            "policy_decision",
            sa.String(length=8),
            server_default="ALLOW",
            nullable=False,
        ),
    )
    op.add_column(
        "transfer_audit_events",
        sa.Column(
            "consent_state",
            sa.String(length=16),
            server_default="GRANTED",
            nullable=False,
        ),
    )
    op.add_column(
        "transfer_audit_events",
        sa.Column(
            "sca_state",
            sa.String(length=16),
            server_default="NOT_REQUIRED",
            nullable=False,
        ),
    )
    op.create_check_constraint(
        "ck_transfer_audit_policy_decision",
        "transfer_audit_events",
        "policy_decision IN ('ALLOW', 'DENY')",
    )
    op.create_check_constraint(
        "ck_transfer_audit_consent_state",
        "transfer_audit_events",
        "consent_state IN ('GRANTED', 'NOT_REQUIRED', 'DENIED')",
    )
    op.create_check_constraint(
        "ck_transfer_audit_sca_state",
        "transfer_audit_events",
        "sca_state IN ('NOT_REQUIRED', 'REQUIRED', 'VERIFIED', 'FAILED', "
        "'ABANDONED', 'UNAVAILABLE')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_transfer_audit_sca_state", "transfer_audit_events", type_="check"
    )
    op.drop_constraint(
        "ck_transfer_audit_consent_state", "transfer_audit_events", type_="check"
    )
    op.drop_constraint(
        "ck_transfer_audit_policy_decision", "transfer_audit_events", type_="check"
    )
    op.drop_column("transfer_audit_events", "sca_state")
    op.drop_column("transfer_audit_events", "consent_state")
    op.drop_column("transfer_audit_events", "policy_decision")
