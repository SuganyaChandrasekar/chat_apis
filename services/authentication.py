import os
import time
from dataclasses import dataclass
from typing import Iterable

import jwt
from fastapi import Header, HTTPException

JWT_ISSUER = "banking-auth"
JWT_AUDIENCE = "banking-api"
JWT_TOKEN_MAX_LENGTH = 8192
SANDBOX_SCOPES = frozenset(
    {
        "account:read",
        "transactions:read",
        "transfer:prepare",
        "transfer:read",
        "transfer:execute",
        "cards:read",
        "cards:freeze",
        "fraud:report",
        "fraud:respond",
        "chat:respond",
    }
)


class JwtConfigurationError(Exception):
    pass


class InvalidAccessToken(Exception):
    pass


class AuthorizationDenied(Exception):
    pass


@dataclass(frozen=True)
class AccessContext:
    subject: str
    scopes: frozenset[str]
    consent_state: str
    consent_action: str | None
    sca_state: str
    token_kind: str


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET")
    if not secret or len(secret.encode("utf-8")) < 32:
        raise JwtConfigurationError
    return secret


def create_access_token(
    subject: str,
    ttl_seconds: int = 1800,
    *,
    scopes: Iterable[str] | None = None,
    consent_state: str = "not_granted",
    consent_action: str | None = None,
    sca_state: str = "not_performed",
    token_kind: str = "customer",
) -> str:
    if not isinstance(subject, str) or not subject or len(subject) > 256:
        raise ValueError("JWT subject must be a non-empty string of at most 256 characters")
    if ttl_seconds <= 0:
        raise ValueError("JWT lifetime must be greater than zero")

    if scopes is None:
        scopes = (
            SANDBOX_SCOPES
            if os.getenv("SANDBOX_MODE", "false").lower() == "true"
            else ()
        )
    normalized_scopes = sorted(set(scopes))
    if any(
        not isinstance(scope, str) or not scope or len(scope) > 128
        for scope in normalized_scopes
    ):
        raise ValueError("JWT scopes must be non-empty strings of at most 128 characters")
    if consent_state not in {"not_granted", "not_required", "granted"}:
        raise ValueError("Unsupported consent state")
    if consent_action is not None and (
        not isinstance(consent_action, str)
        or not consent_action
        or len(consent_action) > 128
    ):
        raise ValueError("Consent action must be a non-empty string of at most 128 characters")
    if sca_state not in {"not_performed", "not_required", "required", "verified"}:
        raise ValueError("Unsupported SCA state")
    if token_kind not in {"customer", "delegated"}:
        raise ValueError("Unsupported access token kind")

    now = int(time.time())
    return jwt.encode(
        {
            "sub": subject,
            "iat": now,
            "exp": now + ttl_seconds,
            "iss": JWT_ISSUER,
            "aud": JWT_AUDIENCE,
            "typ": "access",
            "scope": normalized_scopes,
            "consent": {"state": consent_state, "action": consent_action},
            "sca": {"state": sca_state},
            "token_kind": token_kind,
        },
        _jwt_secret(),
        algorithm="HS256",
    )


def _decode_access_token(token: str) -> dict:
    if not token or len(token) > JWT_TOKEN_MAX_LENGTH:
        raise InvalidAccessToken

    try:
        claims = jwt.decode(
            token,
            _jwt_secret(),
            algorithms=["HS256"],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            options={"require": ["exp", "iat", "iss", "aud", "sub", "typ"]},
        )
    except jwt.InvalidTokenError:
        raise InvalidAccessToken from None

    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject or len(subject) > 256:
        raise InvalidAccessToken
    if claims.get("typ") != "access":
        raise InvalidAccessToken
    scopes = claims.get("scope")
    if scopes is None and os.getenv("SANDBOX_MODE", "false").lower() == "true":
        scopes = sorted(SANDBOX_SCOPES)
    if scopes is None:
        scopes = []
    consent = claims.get("consent", {"state": "not_granted", "action": None})
    sca = claims.get("sca", {"state": "not_performed"})
    token_kind = claims.get("token_kind", "customer")
    if (
        not isinstance(scopes, list)
        or any(not isinstance(scope, str) or not scope for scope in scopes)
        or not isinstance(consent, dict)
        or consent.get("state") not in {"not_granted", "not_required", "granted"}
        or (
            consent.get("action") is not None
            and not isinstance(consent.get("action"), str)
        )
        or not isinstance(sca, dict)
        or sca.get("state")
        not in {"not_performed", "not_required", "required", "verified"}
        or token_kind not in {"customer", "delegated"}
    ):
        raise InvalidAccessToken
    claims["scope"] = scopes
    claims["consent"] = consent
    claims["sca"] = sca
    claims["token_kind"] = token_kind
    return claims


def validate_access_token(token: str) -> str:
    return _decode_access_token(token)["sub"]


def validate_access_context(token: str) -> AccessContext:
    claims = _decode_access_token(token)
    consent = claims.get("consent", {"state": "not_granted", "action": None})
    sca = claims.get("sca", {"state": "not_performed"})
    return AccessContext(
        subject=claims["sub"],
        scopes=frozenset(claims.get("scope", [])),
        consent_state=consent["state"],
        consent_action=consent.get("action"),
        sca_state=sca["state"],
        token_kind=claims["token_kind"],
    )


def exchange_access_token(
    token: str,
    *,
    scopes: Iterable[str],
    consent_state: str = "not_required",
    consent_action: str | None = None,
    sca_state: str = "not_required",
    ttl_seconds: int = 60,
) -> str:
    context = validate_access_context(token)
    requested_scopes = frozenset(scopes)
    if not requested_scopes or not requested_scopes.issubset(context.scopes):
        raise AuthorizationDenied
    return create_access_token(
        context.subject,
        ttl_seconds=ttl_seconds,
        scopes=requested_scopes,
        consent_state=consent_state,
        consent_action=consent_action,
        sca_state=sca_state,
        token_kind="delegated",
    )


def authorize_access_token(
    token: str,
    *,
    required_scope: str,
    consent_action: str | None = None,
    required_sca_state: str | None = None,
) -> AccessContext:
    context = validate_access_context(token)
    if required_scope not in context.scopes:
        raise AuthorizationDenied
    if consent_action is not None and (
        context.consent_state != "granted"
        or context.consent_action != consent_action
    ):
        raise AuthorizationDenied
    if required_sca_state is not None and context.sca_state != required_sca_state:
        raise AuthorizationDenied
    return context


def require_scope(scope: str):
    def dependency(authorization: str | None = Header(default=None)) -> str:
        try:
            context = authorize_access_token(
                bearer_token(authorization),
                required_scope=scope,
            )
        except JwtConfigurationError:
            raise HTTPException(
                status_code=503, detail="Authentication is not configured"
            ) from None
        except InvalidAccessToken:
            raise HTTPException(
                status_code=401,
                detail="Invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None
        except AuthorizationDenied:
            raise HTTPException(
                status_code=403, detail="Required permission is not granted"
            ) from None
        return context.subject

    return dependency


def bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise InvalidAccessToken
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token:
        raise InvalidAccessToken
    return token


def authenticated_subject(authorization: str | None = Header(default=None)) -> str:
    try:
        return validate_access_token(bearer_token(authorization))
    except JwtConfigurationError:
        raise HTTPException(status_code=503, detail="Authentication is not configured") from None
    except InvalidAccessToken:
        raise HTTPException(
            status_code=401,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None