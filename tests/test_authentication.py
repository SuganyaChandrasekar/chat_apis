import time

import jwt
import pytest

from services.authentication import (
    AuthorizationDenied,
    JwtConfigurationError,
    InvalidAccessToken,
    authorize_access_token,
    authenticated_subject,
    bearer_token,
    create_access_token,
    exchange_access_token,
    validate_access_token,
)


SECRET = "a" * 64


def test_create_access_token_uses_expected_claims(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)

    token = create_access_token("customer-123", ttl_seconds=300, scopes=())
    claims = validate_access_token(token)

    assert claims == "customer-123"
    decoded = jwt.decode(
        token,
        SECRET,
        algorithms=["HS256"],
        issuer="banking-auth",
        audience="banking-api",
    )
    assert decoded["sub"] == "customer-123"
    assert decoded["exp"] > int(time.time())
    assert decoded["iat"] <= int(time.time())
    assert decoded["scope"] == []
    assert decoded["consent"] == {"state": "not_granted", "action": None}
    assert decoded["sca"] == {"state": "not_performed"}


def test_scoped_exchange_attenuates_permissions_and_carries_action_context(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)
    source = create_access_token(
        "customer-123",
        scopes=("account:read", "transfer:execute"),
    )

    delegated = exchange_access_token(
        source,
        scopes=("transfer:execute",),
        consent_state="granted",
        consent_action="transfer-hash",
        sca_state="required",
    )
    claims = jwt.decode(
        delegated,
        SECRET,
        algorithms=["HS256"],
        issuer="banking-auth",
        audience="banking-api",
    )

    assert claims["sub"] == "customer-123"
    assert claims["scope"] == ["transfer:execute"]
    assert claims["consent"] == {"state": "granted", "action": "transfer-hash"}
    assert claims["sca"] == {"state": "required"}
    assert claims["token_kind"] == "delegated"
    assert authorize_access_token(
        delegated,
        required_scope="transfer:execute",
        consent_action="transfer-hash",
        required_sca_state="required",
    ).subject == "customer-123"


def test_policy_denies_missing_scope_and_mismatched_consent(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)
    token = create_access_token(
        "customer-123",
        scopes=("account:read",),
    )

    with pytest.raises(AuthorizationDenied):
        exchange_access_token(token, scopes=("transfer:execute",))

    delegated = create_access_token(
        "customer-123",
        scopes=("transfer:execute",),
        consent_state="granted",
        consent_action="original-transfer",
        sca_state="required",
    )
    with pytest.raises(AuthorizationDenied):
        authorize_access_token(
            delegated,
            required_scope="transfer:execute",
            consent_action="different-transfer",
            required_sca_state="required",
        )


def test_create_access_token_rejects_short_secret(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "too-short")

    with pytest.raises(JwtConfigurationError):
        create_access_token("customer-123")


def test_validate_access_token_rejects_invalid_signature(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)
    token = jwt.encode(
        {
            "sub": "customer-123",
            "exp": int(time.time()) + 300,
            "iss": "banking-auth",
            "aud": "banking-api",
        },
        "b" * 64,
        algorithm="HS256",
    )

    with pytest.raises(InvalidAccessToken):
        validate_access_token(token)


def test_validate_access_token_rejects_expired_token(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)
    token = jwt.encode(
        {
            "sub": "customer-123",
            "iat": int(time.time()) - 300,
            "exp": int(time.time()) - 60,
            "iss": "banking-auth",
            "aud": "banking-api",
            "typ": "access",
        },
        SECRET,
        algorithm="HS256",
    )

    with pytest.raises(InvalidAccessToken):
        validate_access_token(token)


def test_bearer_token_requires_bearer_scheme() -> None:
    with pytest.raises(InvalidAccessToken):
        bearer_token("Basic abc")

    assert bearer_token("Bearer abc") == "abc"


def test_authenticated_subject_returns_verified_subject(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)
    token = create_access_token("customer-123")

    assert authenticated_subject(f"Bearer {token}") == "customer-123"
