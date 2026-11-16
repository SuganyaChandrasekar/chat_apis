from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from services.authentication import create_access_token
from services.capabilities import main as capability_main
from services.capabilities import store as capability_store
from services.capabilities.store import CapabilityBase, CapabilityVersion
from services.orchestrator.guardrails import GuardrailViolation


def _manifest(capability_id: str, version: str) -> dict:
    return {
        "capability_id": capability_id,
        "version": version,
        "display_name": "Account summary",
        "description": "Summarize an authenticated customer's account balances.",
        "entrypoint": "banking.workflows.balance:get_summary",
        "routing": {
            "intents": ["customer asks for a comprehensive account summary"],
            "keywords": ["account overview"],
        },
        "schema": {
            "required_inputs": ["account_id"],
            "output_type": "AccountSummary",
        },
    }


def _request(manifest: dict, *, mismatches: int = 0) -> dict:
    count = 20
    cases = [
        {
            "prompt": f"Prompt {index}",
            "expected_response": f"Expected response {index}",
            "actual_response": (
                f"Different response {index}"
                if index < mismatches
                else f"Expected response {index}"
            ),
        }
        for index in range(count)
    ]
    return {"manifest": manifest, "evaluation_cases": cases}


def _prepare_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    engine = create_engine(f"sqlite:///{tmp_path / 'capabilities.db'}")
    CapabilityBase.metadata.create_all(engine)
    monkeypatch.setattr(capability_store, "_engine", lambda: engine)
    monkeypatch.setattr(capability_main, "initialize_guardrails", lambda: object())

    async def allow_text(_rails, value, *_args, **_kwargs):
        return value

    monkeypatch.setattr(capability_main, "check_input", allow_text)
    monkeypatch.setattr(capability_main, "check_output", allow_text)
    capability_main.app.state.guardrails = object()
    headers = {
        "Authorization": "Bearer "
        + create_access_token(
            "capability-publisher", scopes=["lob:publisher"]
        )
    }
    return engine, TestClient(capability_main.app), headers


def test_registry_requires_authentication_and_publisher_scope(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-with-at-least-32-bytes")
    client = TestClient(capability_main.app)
    body = _request(_manifest("new_dynamic_intent", "1.0.0"))

    unauthenticated = client.post("/capabilities", json=body)
    wrong_scope = client.post(
        "/capabilities",
        headers={
            "Authorization": "Bearer "
            + create_access_token("reader", scopes=["chat:respond"])
        },
        json=body,
    )

    assert unauthenticated.status_code == 401
    assert wrong_scope.status_code == 403


def test_registry_openapi_uses_bearer_authorization() -> None:
    schema = capability_main.app.openapi()
    operation = schema["paths"]["/capabilities"]["post"]

    assert operation["security"] == [{"HTTPBearer": []}]
    assert "authorization" not in {
        parameter["name"] for parameter in operation.get("parameters", [])
    }
    assert schema["components"]["securitySchemes"]["HTTPBearer"] == {
        "type": "http",
        "scheme": "bearer",
    }


def test_manifest_validation_returns_specific_schema_errors(
    monkeypatch, tmp_path
) -> None:
    engine, client, headers = _prepare_registry(monkeypatch, tmp_path)
    invalid = _request(_manifest("bad_version", "not-semver"))
    invalid["manifest"]["routing"]["unrecognized"] = True

    response = client.post("/capabilities", headers=headers, json=invalid)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("version" in ".".join(error["loc"]) for error in errors)
    assert any("unrecognized" in ".".join(error["loc"]) for error in errors)
    engine.dispose()


def test_registers_dynamic_manifest_and_only_routes_certified_version(
    monkeypatch, tmp_path
) -> None:
    engine, client, headers = _prepare_registry(monkeypatch, tmp_path)

    response = client.post(
        "/capabilities",
        headers=headers,
        json=_request(_manifest("customer_specific_workflow", "1.0.0")),
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["certification_status"] == "CERTIFIED"
    assert payload["rollout_status"] == "BETA"
    assert payload["cohort"] == "internal_beta"
    assert payload["general_traffic_percentage"] == 0
    assert payload["feature_flag_enabled"] is True
    assert capability_store.resolve_capability(
        "customer_specific_workflow", cohort="internal_beta"
    ) is not None
    assert capability_store.resolve_capability("customer_specific_workflow") is None
    beta_route = client.get(
        "/capabilities/customer_specific_workflow/route",
        params={"cohort": "internal_beta"},
        headers=headers,
    )
    production_route = client.get(
        "/capabilities/customer_specific_workflow/route",
        headers=headers,
    )
    assert beta_route.status_code == 200
    assert beta_route.json()["manifest"]["capability_id"] == (
        "customer_specific_workflow"
    )
    assert production_route.status_code == 404

    duplicate = client.post(
        "/capabilities",
        headers=headers,
        json=_request(_manifest("customer_specific_workflow", "1.0.0")),
    )
    assert duplicate.status_code == 409
    engine.dispose()


def test_guardrail_failure_is_saved_blocked_and_never_routable(
    monkeypatch, tmp_path
) -> None:
    engine, client, headers = _prepare_registry(monkeypatch, tmp_path)

    async def reject_manifest(_rails, value, *_args, **_kwargs):
        if "unsafe capability" in value:
            raise GuardrailViolation("safety")
        return value

    monkeypatch.setattr(capability_main, "check_input", reject_manifest)
    manifest = _manifest("unsafe_capability", "1.0.0")
    manifest["description"] = "unsafe capability"

    response = client.post(
        "/capabilities", headers=headers, json=_request(manifest)
    )

    assert response.status_code == 201
    assert response.json()["certification_status"] == "FAILED"
    assert response.json()["guardrails_passed"] is False
    assert capability_store.resolve_capability("unsafe_capability", cohort="internal_beta") is None
    engine.dispose()


def test_eval_score_below_threshold_is_blocked(
    monkeypatch, tmp_path
) -> None:
    engine, client, headers = _prepare_registry(monkeypatch, tmp_path)

    response = client.post(
        "/capabilities",
        headers=headers,
        json=_request(_manifest("low_accuracy", "1.0.0"), mismatches=2),
    )

    assert response.status_code == 201
    assert response.json()["evaluation_score"] == 0.9
    assert response.json()["certification_status"] == "FAILED"
    assert response.json()["rollout_status"] == "BLOCKED"
    engine.dispose()


def test_promotion_blocks_eval_regression_and_keeps_active_version(
    monkeypatch, tmp_path
) -> None:
    engine, client, headers = _prepare_registry(monkeypatch, tmp_path)
    capability_id = "rollout_workflow"
    first = client.post(
        "/capabilities",
        headers=headers,
        json=_request(_manifest(capability_id, "1.0.0")),
    )
    first_promotion = client.post(
        f"/capabilities/{capability_id}/versions/1.0.0/promote",
        headers=headers,
    )
    candidate = client.post(
        "/capabilities",
        headers=headers,
        json=_request(_manifest(capability_id, "2.0.0"), mismatches=1),
    )
    blocked = client.post(
        f"/capabilities/{capability_id}/versions/2.0.0/promote",
        headers=headers,
    )

    assert first.status_code == candidate.status_code == 201
    assert first_promotion.json()["promoted"] is True
    assert blocked.status_code == 409
    assert blocked.json()["promoted"] is False
    assert "evaluation_regression_exceeds_2_percent" in blocked.json()["reasons"]
    assert capability_store.resolve_capability(capability_id).version == "1.0.0"
    with Session(engine) as session:
        versions = session.scalars(
            select(CapabilityVersion).where(
                CapabilityVersion.capability_id == capability_id
            )
        ).all()
    assert {version.version: version.rollout_status for version in versions} == {
        "1.0.0": "PRODUCTION",
        "2.0.0": "BETA",
    }
    engine.dispose()


def test_successful_promotion_moves_general_route_to_new_version(
    monkeypatch, tmp_path
) -> None:
    engine, client, headers = _prepare_registry(monkeypatch, tmp_path)
    capability_id = "promotable_workflow"

    for version in ("1.0.0", "2.0.0"):
        registered = client.post(
            "/capabilities",
            headers=headers,
            json=_request(_manifest(capability_id, version)),
        )
        assert registered.status_code == 201

    promoted = client.post(
        f"/capabilities/{capability_id}/versions/1.0.0/promote",
        headers=headers,
    )
    replaced = client.post(
        f"/capabilities/{capability_id}/versions/2.0.0/promote",
        headers=headers,
    )

    assert promoted.json()["promoted"] is True
    assert replaced.json()["promoted"] is True
    assert replaced.json()["active_version"] == "2.0.0"
    assert capability_store.resolve_capability(capability_id).version == "2.0.0"
    assert capability_store.resolve_capability(
        capability_id, cohort="internal_beta"
    ) is None
    engine.dispose()
