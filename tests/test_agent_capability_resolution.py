import importlib.util
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine

from services.capabilities import store as capability_store
from services.capabilities.store import CapabilityBase, create_version, promote_version
from services.orchestrator import agent


def _manifest(
    capability_id: str,
    entrypoint: str,
    *,
    intent: str = "balance_check",
) -> dict:
    return {
        "capability_id": capability_id,
        "version": "1.0.0",
        "display_name": capability_id,
        "description": f"Capability for {intent}.",
        "entrypoint": entrypoint,
        "routing": {"intents": [intent], "keywords": []},
        "schema": {"required_inputs": [], "output_type": "TestResult"},
    }


def _register_active(manifest: dict) -> None:
    create_version(
        manifest=manifest,
        guardrails_passed=True,
        evaluation_score=1.0,
        reasons=[],
    )
    promote_version(manifest["capability_id"], manifest["version"])


def test_agent_tools_are_loaded_from_active_registry_manifests(
    monkeypatch, tmp_path
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'capabilities.db'}")
    CapabilityBase.metadata.create_all(engine)
    monkeypatch.setattr(capability_store, "_engine", lambda: engine)
    for name, intent, scope in (
        ("check_balance", "balance_check", "account:read"),
        ("prepare_transfer", "transfer_request", "transfer:prepare"),
        (
            "get_recent_transactions",
            "transaction_history",
            "transactions:read",
        ),
    ):
        function = getattr(agent, name)
        assert function.capability_required_scope == scope
        _register_active(
            _manifest(
                name,
                f"services.orchestrator.agent:{name}",
                intent=intent,
            )
        )

    tools = agent.build_agent_tools("delegated-token", "authenticated-customer")
    tools_by_name = {registered_tool.name: registered_tool for registered_tool in tools}

    assert set(tools_by_name) == {
        "check_balance",
        "prepare_transfer",
        "get_recent_transactions",
    }
    assert tools_by_name["prepare_transfer"].metadata == {
        "required_scope": "transfer:prepare",
        "routing_intents": ("transfer_request",),
    }
    assert set(tools_by_name["prepare_transfer"].args) == {
        "source_account",
        "recipient",
        "amount",
    }
    assert "access_token" not in tools_by_name["prepare_transfer"].args
    assert "account_id" not in tools_by_name["prepare_transfer"].args
    engine.dispose()


def test_non_trusted_entrypoint_is_rejected(monkeypatch, tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'untrusted.db'}")
    CapabilityBase.metadata.create_all(engine)
    monkeypatch.setattr(capability_store, "_engine", lambda: engine)
    manifest = _manifest(
        "untrusted_tool",
        "services.orchestrator.fraud:report_sandbox_fraud",
    )
    _register_active(manifest)

    with pytest.raises(RuntimeError, match="outside the trusted tool module"):
        agent.build_agent_tools("delegated-token", "authenticated-customer")
    engine.dispose()


def test_seed_migration_registers_existing_tools_as_production(
    monkeypatch, tmp_path
) -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "migrations"
        / "versions"
        / "20261113_0002_seed_builtin_capabilities.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_builtin_capabilities", migration_path
    )
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)

    engine = create_engine(f"sqlite:///{tmp_path / 'seeded.db'}")
    CapabilityBase.metadata.create_all(engine)
    with engine.begin() as connection:
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()
    monkeypatch.setattr(capability_store, "_engine", lambda: engine)

    records = capability_store.list_routable_capabilities(cohort="general")

    assert {record.capability_id for record in records} == {
        "check_balance",
        "prepare_transfer",
        "get_recent_transactions",
    }
    assert all(
        record.certification_status == "CERTIFIED"
        and record.rollout_status == "PRODUCTION"
        and record.general_traffic_percentage == 100
        for record in records
    )
    assert {tool.name for tool in agent.build_agent_tools("token", "customer")} == {
        "check_balance",
        "prepare_transfer",
        "get_recent_transactions",
    }
    engine.dispose()
