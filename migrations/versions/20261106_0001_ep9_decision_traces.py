from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision = "20261106_0001"
down_revision = "20261105_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "security_audit_events",
        sa.Column("correlation_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "security_audit_events",
        sa.Column("conversation_id", sa.String(length=36), nullable=True),
    )
    op.create_index(
        "ix_security_audit_events_correlation",
        "security_audit_events",
        ["correlation_id"],
    )
    op.create_index(
        "ix_security_audit_events_conversation",
        "security_audit_events",
        ["conversation_id"],
    )
    op.create_table(
        "decision_trace_events",
        sa.Column(
            "sequence_id",
            sa.BigInteger(),
            sa.Identity(always=True),
            primary_key=True,
        ),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("conversation_id", sa.String(length=36), nullable=False),
        sa.Column("correlation_id", sa.String(length=128), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", JSONB(), nullable=False),
        sa.Column("source", sa.String(length=128), nullable=False),
    )
    op.create_index(
        "ix_decision_trace_conversation_sequence",
        "decision_trace_events",
        ["conversation_id", "sequence_id"],
    )
    op.create_index(
        "ix_decision_trace_correlation",
        "decision_trace_events",
        ["correlation_id"],
    )
    op.execute(
        "CREATE FUNCTION reject_decision_trace_mutation() RETURNS trigger AS $$ "
        "BEGIN RAISE EXCEPTION 'decision trace events are immutable'; END; "
        "$$ LANGUAGE plpgsql"
    )
    op.execute(
        """
        CREATE TRIGGER decision_trace_events_immutable
        BEFORE UPDATE OR DELETE ON decision_trace_events
        FOR EACH ROW EXECUTE FUNCTION reject_decision_trace_mutation()
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS decision_trace_events_immutable "
        "ON decision_trace_events"
    )
    op.execute("DROP FUNCTION IF EXISTS reject_decision_trace_mutation()")
    op.drop_index(
        "ix_decision_trace_correlation", table_name="decision_trace_events"
    )
    op.drop_index(
        "ix_decision_trace_conversation_sequence",
        table_name="decision_trace_events",
    )
    op.drop_table("decision_trace_events")
    op.drop_index(
        "ix_security_audit_events_conversation",
        table_name="security_audit_events",
    )
    op.drop_index(
        "ix_security_audit_events_correlation",
        table_name="security_audit_events",
    )
    op.drop_column("security_audit_events", "conversation_id")
    op.drop_column("security_audit_events", "correlation_id")
