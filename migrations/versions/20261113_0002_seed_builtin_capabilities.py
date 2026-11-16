from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa


revision = "20261113_0002"
down_revision = "20261113_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    capability_versions = sa.table(
        "capability_versions",
        sa.column("capability_id", sa.String()),
        sa.column("version", sa.String()),
        sa.column("manifest", sa.JSON()),
        sa.column("certification_status", sa.String()),
        sa.column("guardrails_passed", sa.Boolean()),
        sa.column("evaluation_score", sa.Float()),
        sa.column("rollout_status", sa.String()),
        sa.column("feature_flag", sa.String()),
        sa.column("feature_flag_enabled", sa.Boolean()),
        sa.column("cohort", sa.String()),
        sa.column("general_traffic_percentage", sa.Integer()),
        sa.column("reasons", sa.JSON()),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("promoted_at", sa.DateTime(timezone=True)),
    )
    now = datetime.now(timezone.utc)
    builtin_capabilities = (
        (
            "check_balance",
            "Get authenticated account balance",
            "Returns the authenticated customer's account balance.",
            "services.orchestrator.agent:check_balance",
            ["balance_check"],
            [],
            "BalanceFinancialPayload",
        ),
        (
            "prepare_transfer",
            "Prepare a transfer",
            "Prepares an authorized transfer preview for linked accounts.",
            "services.orchestrator.agent:prepare_transfer",
            ["transfer_request"],
            ["source_account", "recipient", "amount"],
            "TransferPreparationResult",
        ),
        (
            "get_recent_transactions",
            "Get recent transactions",
            "Returns recent transactions for an authenticated linked account.",
            "services.orchestrator.agent:get_recent_transactions",
            ["transaction_history"],
            [],
            "RecentTransactionsPayload",
        ),
    )
    rows = [
        {
            "capability_id": capability_id,
            "version": "1.0.0",
            "manifest": {
                "capability_id": capability_id,
                "version": "1.0.0",
                "display_name": display_name,
                "description": description,
                "entrypoint": entrypoint,
                "routing": {"intents": intents, "keywords": []},
                "schema": {
                    "required_inputs": required_inputs,
                    "output_type": output_type,
                },
            },
            "certification_status": "CERTIFIED",
            "guardrails_passed": True,
            "evaluation_score": 1.0,
            "rollout_status": "PRODUCTION",
            "feature_flag": f"capability.{capability_id}.1.0.0",
            "feature_flag_enabled": True,
            "cohort": "general",
            "general_traffic_percentage": 100,
            "reasons": ["preapproved_builtin_capability"],
            "created_at": now,
            "promoted_at": now,
        }
        for (
            capability_id,
            display_name,
            description,
            entrypoint,
            intents,
            required_inputs,
            output_type,
        ) in builtin_capabilities
    ]
    connection = op.get_bind()
    for row in rows:
        existing = connection.execute(
            sa.select(capability_versions.c.capability_id).where(
                capability_versions.c.capability_id == row["capability_id"],
                capability_versions.c.version == row["version"],
            )
        ).first()
        if existing is None:
            connection.execute(capability_versions.insert().values(**row))


def downgrade() -> None:
    capability_versions = sa.table(
        "capability_versions",
        sa.column("capability_id", sa.String()),
        sa.column("version", sa.String()),
        sa.column("reasons", sa.JSON()),
    )
    op.execute(
        capability_versions.delete().where(
            capability_versions.c.capability_id.in_(
                [
                    "check_balance",
                    "prepare_transfer",
                    "get_recent_transactions",
                ]
            ),
            capability_versions.c.version == "1.0.0",
            capability_versions.c.reasons == ["preapproved_builtin_capability"],
        )
    )
