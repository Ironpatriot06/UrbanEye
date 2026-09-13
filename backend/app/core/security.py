"""
Security utilities: password hashing and JWT token operations.

Design
------
- Passwords are hashed with bcrypt via passlib.  We never store or return
  the plain-text password.
- JWTs are signed with the SECRET_KEY using HS256.  The token payload
  contains the user's email (sub) and their role.  Expiry is configurable.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Return the bcrypt hash of a plain-text password."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if `plain` matches the stored `hashed` password."""
    return _pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a signed JWT.

    `data` must contain at least {"sub": email, "role": role_value}.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode and validate a JWT.

    Returns the payload dict on success, or None if the token is invalid /
    expired.
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError:
        return None


# ---------------------------------------------------------------------------
# OAuth state tokens
# ---------------------------------------------------------------------------
#
# The `state` parameter of an OAuth authorization request is the CSRF defence:
# without it an attacker can feed their own authorization code to our callback
# and land the victim in the attacker's account.
#
# We make state a short-lived JWT signed with SECRET_KEY, so the callback can
# verify it is one we issued and has not expired without keeping any
# server-side session store.  The same value is also set as an HttpOnly cookie
# and compared on return (a double-submit), so a state token stolen from a
# browser URL cannot be replayed from a different browser.

#: Marks a token as an OAuth state token so it can never be presented as an
#: access token, and vice versa.
_OAUTH_STATE_TOKEN_TYPE = "oauth_state"


def create_oauth_state_token(nonce: str, next_path: str | None = None) -> str:
    """Create the signed, short-lived `state` value for an OAuth redirect."""
    payload = {
        "typ": _OAUTH_STATE_TOKEN_TYPE,
        "nonce": nonce,
        "exp": datetime.now(timezone.utc)
        + timedelta(seconds=settings.OAUTH_STATE_EXPIRE_SECONDS),
    }
    if next_path:
        payload["next"] = next_path
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_oauth_state_token(token: str) -> Optional[dict]:
    """
    Validate an OAuth `state` token.

    Returns the payload, or None if the token is invalid, expired, or is some
    other kind of token (an access token replayed as state, for instance).
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
    except JWTError:
        return None
    if payload.get("typ") != _OAUTH_STATE_TOKEN_TYPE:
        return None
    return payload
