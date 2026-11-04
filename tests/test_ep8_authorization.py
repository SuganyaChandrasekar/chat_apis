from decimal import Decimal

import pytest

from services.authorization import (
    HIGH_VALUE_TRANSFER_THRESHOLD,
    InvalidScaAttestation,
    create_sca_attestation,
    record_authorization_decision,
    transfer_action_id,
    validate_sca_attestation,
)


def test_sca_attestation_is_bound_to_customer_and_transfer(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "ep8-test-secret-with-at-least-32-bytes")
    amount = Decimal("1000.00")
    action_id = transfer_action_id(
        account_id="customer-1",
        source_account_id="checking-1",
        destination_account_id="savings-1",
        amount=amount,
        currency="USD",
        idempotency_key="request-1",
    )
    token = create_sca_attestation(
        subject="customer-1",
        action_id=action_id,
        transaction_id="transaction-1",
        amount=amount,
        currency="USD",
    )

    validate_sca_attestation(
        token,
        subject="customer-1",
        action_id=action_id,
        transaction_id="transaction-1",
        amount=amount,
        currency="USD",
    )
    with pytest.raises(InvalidScaAttestation):
        validate_sca_attestation(
            token,
            subject="customer-1",
            action_id=action_id,
            transaction_id="transaction-1",
            amount=Decimal("1000.01"),
            currency="USD",
        )


def test_sca_threshold_is_inclusive_at_configured_boundary() -> None:
    assert Decimal("1000.00") >= HIGH_VALUE_TRANSFER_THRESHOLD
    assert Decimal("999.99") < HIGH_VALUE_TRANSFER_THRESHOLD


def test_pdp_audit_records_policy_context_without_tokens(monkeypatch) -> None:
    recorded = []
    monkeypatch.setattr(
        "services.authorization.append_audit_event",
        lambda event: recorded.append(event),
    )

    record_authorization_decision(
        subject="customer-1",
        request_path="/chat",
        scope="transfer:prepare",
        decision="allow",
        consent_state="not_required",
        sca_state="not_required",
        action="transfer_request",
    )

    event = recorded[0]
    assert event.source == "ep8_pdp"
    assert event.event_type == "SECURITY"
    assert event.details == {
        "decision": "allow",
        "scope": "transfer:prepare",
        "consent_state": "not_required",
        "sca_state": "not_required",
        "action": "transfer_request",
    }
