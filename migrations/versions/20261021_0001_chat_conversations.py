from alembic import op
import sqlalchemy as sa


revision = "20261021_0001"
down_revision = "20261007_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_conversations",
        sa.Column("thread_id", sa.Uuid(), nullable=False),
        sa.Column("owner_key", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_activity_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("thread_id"),
    )
    op.create_index(
        "ix_chat_conversations_owner_activity",
        "chat_conversations",
        ["owner_key", "last_activity_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_chat_conversations_owner_activity", table_name="chat_conversations"
    )
    op.drop_table("chat_conversations")