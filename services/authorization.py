import hashlib
import json
import os
import time
from decimal import Decimal

import jwt

from services.authentication import (
    AuthorizationDenied,
    _jwt_secret,
    authorize_access_token,
)
from services.security.audit import append_audit_event, create_audit_event

HIGH_VALUE_TRANSFER_THRESHOLD = Decimal(
    os.getenv("HIGH_VALUE_TRANSFER_THRESHOLD", "1000.00")
)
SCA_ISSUER = "banking-auth"
SCA_AUDIENCE = "banking-transfer"
SCA_TOKEN_MAX_LENGTH = 4096


class InvalidScaAttestation(Exception):
    pass


def record_authorization_decision(
    *,
    subject: str,
    request_path: str,
    scope: str,
    decision: str,
    consent_state: str,
    sca_state: str,
    action: str,
) -> None:
    if decision not in {"allow", "deny"}:
        raise ValueError("Unsupported authorization decision")
    event = create_audit_event(
        event_type="SECURITY",
        severity="INFO" if decision == "allow" else "WARNING",
        request_path=request_path,
        user_id=subject,
        error_type="PolicyDecision" if decision == "deny" else None,
        error_message="Policy decision denied the requested action"
        if decision == "deny"
        else None,
        details={
            "decision": decision,
            "scope": scope,
            "consent_state": consent_state,
            "sca_state": sca_state,
            "action": action,
        },
        source="ep8_pdp",
    )
    append_audit_event(event)


def transfer_action_id(
    *,
    account_id: str,
    source_account_id: str,
    destination_account_id: str,
    amount: Decimal,
    currency: str,
    idempotency_key: str,
) -> str:
    action = json.dumps(
        [
            account_id,
            source_account_id,
            destination_account_id,
            f"{amount:.2f}",
            currency,
            idempotency_key,
        ],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(action.encode("utf-8")).hexdigest()


def authorize_transfer_context(
    token: str,
    *,
    account_id: str,
    source_account_id: str,
    destination_account_id: str,
    amount: Decimal,
    currency: str,
    idempotency_key: str,
):
    action_id = transfer_action_id(
        account_id=account_id,
        source_account_id=source_account_id,
        destination_account_id=destination_account_id,
        amount=amount,
        currency=currency,
        idempotency_key=idempotency_key,
    )
    context = authorize_access_token(
        token,
        required_scope="transfer:execute",
        consent_action=action_id,
        required_sca_state=(
            "required"
            if amount >= HIGH_VALUE_TRANSFER_THRESHOLD
            else "not_required"
        ),
    )
    if context.subject != account_id or context.token_kind != "delegated":
        raise AuthorizationDenied
    return action_id, context


def card_freeze_action_id(*, subject: str, card_id: str) -> str:
    action = json.dumps(
        ["card_freeze", subject, card_id],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(action.encode("utf-8")).hexdigest()


def authorize_card_freeze_context(token: str, *, subject: str, card_id: str):
    action_id = card_freeze_action_id(subject=subject, card_id=card_id)
    context = authorize_access_token(
        token,
        required_scope="cards:freeze",
        consent_action=action_id,
        required_sca_state="required",
    )
    if context.subject != subject or context.token_kind != "delegated":
        raise AuthorizationDenied
    return action_id, context


def create_sca_attestation(
    *,
    subject: str,
    action_id: str,
    transaction_id: str,
    purpose: str = "transfer",
    amount: Decimal | None = None,
    currency: str | None = None,
    ttl_seconds: int = 120,
) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "sub": subject,
            "action_id": action_id,
            "transaction_id": transaction_id,
            "amount": f"{amount:.2f}" if amount is not None else None,
            "currency": currency,
            "purpose": purpose,
            "iat": now,
            "exp": now + ttl_seconds,
            "iss": SCA_ISSUER,
            "aud": SCA_AUDIENCE,
            "typ": "sca",
        },
        _jwt_secret(),
        algorithm="HS256",
    )


def validate_sca_attestation(
    token: str,
    *,
    subject: str,
    action_id: str,
    transaction_id: str,
    purpose: str = "transfer",
    amount: Decimal | None = None,
    currency: str | None = None,
) -> None:
    if not token or len(token) > SCA_TOKEN_MAX_LENGTH:
        raise InvalidScaAttestation
    try:
        claims = jwt.decode(
            token,
            _jwt_secret(),
            algorithms=["HS256"],
            issuer=SCA_ISSUER,
            audience=SCA_AUDIENCE,
            options={
                "require": [
                    "sub",
                    "action_id",
                    "transaction_id",
                    "purpose",
                    "iat",
                    "exp",
                    "iss",
                    "aud",
                    "typ",
                ]
            },
        )
        if amount is not None or currency is not None:
            if not {"amount", "currency"}.issubset(claims):
                raise InvalidScaAttestation
    except jwt.InvalidTokenError:
        raise InvalidScaAttestation from None
    if (
        claims.get("typ") != "sca"
        or claims.get("purpose") != purpose
        or claims.get("sub") != subject
        or claims.get("action_id") != action_id
        or claims.get("transaction_id") != transaction_id
        or claims.get("amount") != (f"{amount:.2f}" if amount is not None else None)
        or claims.get("currency") != currency
    ):
        raise InvalidScaAttestation
