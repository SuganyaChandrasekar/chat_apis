from alembic import op
import sqlalchemy as sa


revision = "20261102_0001"
down_revision = "20261028_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cards",
        sa.Column("card_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("card_name", sa.String(length=256), nullable=False),
        sa.Column("last_4", sa.String(length=4), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default="active",
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('active', 'frozen')",
            name="ck_cards_status",
        ),
        sa.PrimaryKeyConstraint("card_id"),
    )
    op.create_index("ix_cards_user_id", "cards", ["user_id"])
    op.bulk_insert(
        sa.table(
            "cards",
            sa.column("card_id", sa.String(length=128)),
            sa.column("user_id", sa.String(length=256)),
            sa.column("card_name", sa.String(length=256)),
            sa.column("last_4", sa.String(length=4)),
            sa.column("status", sa.String(length=16)),
        ),
        [
            {
                "card_id": "card_debit_4321",
                "user_id": "usr_sandbox_01",
                "card_name": "Albion Debit Card",
                "last_4": "4321",
                "status": "active",
            },
            {
                "card_id": "card_credit_8899",
                "user_id": "usr_sandbox_01",
                "card_name": "Albion Credit Card",
                "last_4": "8899",
                "status": "active",
            },
        ],
    )


def downgrade() -> None:
    op.drop_index("ix_cards_user_id", table_name="cards")
    op.drop_table("cards")
