from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = "20261007_0001"
down_revision = "20261005_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "security_audit_events",
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("user_id", sa.String(length=256), nullable=True),
        sa.Column("request_path", sa.String(length=2048), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("error_type", sa.String(length=255), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("details", JSONB(), nullable=True),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.PrimaryKeyConstraint("event_id"),
        sa.CheckConstraint(
            "severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')",
            name="ck_security_audit_severity",
        ),
        sa.CheckConstraint(
            "event_type IN ('SECURITY', 'APPLICATION_ERROR')",
            name="ck_security_audit_event_type",
        ),
    )
    op.create_index(
        "ix_security_audit_events_occurred_at",
        "security_audit_events",
        ["occurred_at"],
    )
    op.create_index(
        "ix_security_audit_events_user_id",
        "security_audit_events",
        ["user_id"],
    )
    op.create_index(
        "ix_security_audit_events_request_path",
        "security_audit_events",
        ["request_path"],
    )
    op.create_index(
        "ix_security_audit_events_event_type_severity",
        "security_audit_events",
        ["event_type", "severity"],
    )
    op.execute(
        "CREATE FUNCTION reject_security_audit_mutation() RETURNS trigger AS $$ "
        "BEGIN RAISE EXCEPTION 'security audit events are immutable'; END; "
        "$$ LANGUAGE plpgsql"
    )
    op.execute(
        """
        CREATE TRIGGER security_audit_events_immutable
        BEFORE UPDATE OR DELETE ON security_audit_events
        FOR EACH ROW EXECUTE FUNCTION reject_security_audit_mutation()
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS security_audit_events_immutable ON security_audit_events"
    )
    op.execute("DROP FUNCTION IF EXISTS reject_security_audit_mutation()")
    op.drop_table("security_audit_events")
