from decimal import Decimal

from alembic import op
import sqlalchemy as sa


revision = "20261117_0001"
down_revision = "20261113_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sandbox_accounts",
        sa.Column("account_id", sa.String(length=256), primary_key=True),
        sa.Column("balance", sa.Numeric(12, 2), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
    )
    op.bulk_insert(
        sa.table(
            "sandbox_accounts",
            sa.column("account_id", sa.String(length=256)),
            sa.column("balance", sa.Numeric(12, 2)),
            sa.column("currency", sa.String(length=3)),
        ),
        [
            {
                "account_id": "sandbox-checking-001",
                "balance": Decimal("1500.00"),
                "currency": "USD",
            },
            {
                "account_id": "sandbox-savings-001",
                "balance": Decimal("3200.00"),
                "currency": "USD",
            },
        ],
    )


def downgrade() -> None:
    op.drop_table("sandbox_accounts")
