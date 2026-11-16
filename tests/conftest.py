import pytest
from sqlalchemy import create_engine

from services.capabilities import store as capability_store
from services.capabilities.store import CapabilityBase, create_version, promote_version


@pytest.fixture(autouse=True)
def active_builtin_capabilities(request, monkeypatch, tmp_path):
    if request.node.path.name not in {
        "test_balance_routes.py",
        "test_orchestrator_fraud.py",
        "test_orchestrator_transfer_tool.py",
        "test_transactions.py",
    }:
        yield
        return

    engine = create_engine(f"sqlite:///{tmp_path / 'capabilities.db'}")
    CapabilityBase.metadata.create_all(engine)
    monkeypatch.setattr(capability_store, "_engine", lambda: engine)
    capabilities = (
        ("check_balance", "balance_check"),
        ("prepare_transfer", "transfer_request"),
        ("get_recent_transactions", "transaction_history"),
    )
    for capability_id, intent in capabilities:
        create_version(
            manifest={
                "capability_id": capability_id,
                "version": "1.0.0",
                "display_name": capability_id,
                "description": f"Test capability for {intent}.",
                "entrypoint": (
                    "services.orchestrator.agent:" + capability_id
                ),
                "routing": {"intents": [intent], "keywords": []},
                "schema": {"required_inputs": [], "output_type": "TestResult"},
            },
            guardrails_passed=True,
            evaluation_score=1.0,
            reasons=[],
        )
        promote_version(capability_id, "1.0.0")

    yield
    engine.dispose()
