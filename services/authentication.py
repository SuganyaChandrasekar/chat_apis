import os
import time

import jwt
from fastapi import Header, HTTPException

JWT_ISSUER = "banking-auth"
JWT_AUDIENCE = "banking-api"
JWT_TOKEN_MAX_LENGTH = 8192


class JwtConfigurationError(Exception):
    pass


class InvalidAccessToken(Exception):
    pass


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET")
    if not secret or len(secret.encode("utf-8")) < 32:
        raise JwtConfigurationError
    return secret


def create_access_token(subject: str, ttl_seconds: int = 1800) -> str:
    if not isinstance(subject, str) or not subject or len(subject) > 256:
        raise ValueError("JWT subject must be a non-empty string of at most 256 characters")
    if ttl_seconds <= 0:
        raise ValueError("JWT lifetime must be greater than zero")

    now = int(time.time())
    return jwt.encode(
        {
            "sub": subject,
            "iat": now,
            "exp": now + ttl_seconds,
            "iss": JWT_ISSUER,
            "aud": JWT_AUDIENCE,
            "typ": "access",
        },
        _jwt_secret(),
        algorithm="HS256",
    )


def validate_access_token(token: str) -> str:
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
    return subject


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