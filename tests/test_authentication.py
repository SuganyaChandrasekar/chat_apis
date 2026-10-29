import time

import jwt
import pytest

from services.authentication import (
    JwtConfigurationError,
    InvalidAccessToken,
    authenticated_subject,
    bearer_token,
    create_access_token,
    validate_access_token,
)


SECRET = "a" * 64


def test_create_access_token_uses_expected_claims(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)

    token = create_access_token("customer-123", ttl_seconds=300)
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
