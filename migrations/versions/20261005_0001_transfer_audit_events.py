from alembic import op
import sqlalchemy as sa


revision = "20261005_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "transfer_audit_events",
        sa.Column("event_id", sa.String(length=36), primary_key=True),
        sa.Column("transaction_id", sa.String(length=36), nullable=False),
        sa.Column("idempotency_key", sa.String(length=128), nullable=False),
        sa.Column("account_id", sa.String(length=256), nullable=False),
        sa.Column("source_account_id", sa.String(length=256), nullable=False),
        sa.Column("destination_account_id", sa.String(length=256), nullable=False),
        sa.Column("amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "status IN ('PENDING_VALIDATION', 'SUCCESS', 'FAILED')",
            name="ck_transfer_audit_status",
        ),
        sa.UniqueConstraint(
            "account_id",
            "idempotency_key",
            "status",
            name="uq_transfer_audit_state",
        ),
    )
    op.create_index(
        "ix_transfer_audit_events_transaction_id",
        "transfer_audit_events",
        ["transaction_id"],
    )
    op.create_index(
        "ix_transfer_audit_events_idempotency_key",
        "transfer_audit_events",
        ["idempotency_key"],
    )
    op.create_index(
        "ix_transfer_audit_events_account_id",
        "transfer_audit_events",
        ["account_id"],
    )
    op.execute(
        "CREATE FUNCTION reject_transfer_audit_mutation() RETURNS trigger AS $$ "
        "BEGIN RAISE EXCEPTION 'transfer audit events are immutable'; END; "
        "$$ LANGUAGE plpgsql"
    )
    op.execute(
        """
        CREATE TRIGGER transfer_audit_events_immutable
        BEFORE UPDATE OR DELETE ON transfer_audit_events
        FOR EACH ROW EXECUTE FUNCTION reject_transfer_audit_mutation()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS transfer_audit_events_immutable ON transfer_audit_events")
    op.execute("DROP FUNCTION IF EXISTS reject_transfer_audit_mutation()")
    op.drop_table("transfer_audit_events")