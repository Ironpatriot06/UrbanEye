"""
Google Sign-In — OAuth 2.0 Authorization Code flow.

Why the authorization-code flow and not the one-tap ID token
------------------------------------------------------------
The code is exchanged for tokens by the BACKEND, over a direct TLS connection
to Google, using the client secret.  The secret never reaches the browser, and
the browser never hands us an identity assertion we have to take on trust.

What is verified before we believe an identity
----------------------------------------------
1. `state` — signed by us, unexpired, and matching the HttpOnly cookie set when
   the flow started (see app/api/auth.py).  This is the CSRF control.
2. The ID token's *signature*, issuer and audience, checked by google-auth
   against Google's published certificates.  We do this even though the token
   arrived straight from Google's token endpoint, so a future refactor that
   changes how the token reaches us cannot silently remove the check.
3. `email_verified` — an unverified Google email proves nothing about who owns
   that mailbox, and we key account linking on the email.  Refused outright.

What this module deliberately does not do
-----------------------------------------
It does not assign roles, look at the email's domain, or touch the database.
It returns a verified identity; app/services/user_service.py decides what
account that identity maps to, and every new account it creates is a USER.

Tokens
------
Google's access and refresh tokens are used for nothing beyond the exchange —
we ask only for the user's identity, not for access to their Google data — so
they are never stored, logged, or returned.
"""

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode

import httpx
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

from app.core.config import get_settings

settings = get_settings()

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"

#: Identity only.  We ask for no Google API access, so there is nothing to
#: store a refresh token for.
GOOGLE_SCOPES = ["openid", "email", "profile"]

#: Issuers Google signs ID tokens with.  google-auth checks this itself; the
#: constant is kept for the explicit re-check below.
_VALID_ISSUERS = ("accounts.google.com", "https://accounts.google.com")


class GoogleAuthError(RuntimeError):
    """Raised when a Google sign-in attempt cannot be trusted or completed."""


class GoogleAuthNotConfigured(GoogleAuthError):
    """Raised when GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET are not set."""


@dataclass(frozen=True)
class GoogleIdentity:
    """A Google identity that has passed every check above."""

    subject: str          # Google's stable `sub` claim
    email: str            # verified email address
    email_verified: bool  # always True by the time this is constructed
    name: str             # display name, falling back to the email local part


def is_configured() -> bool:
    """True when both Google OAuth credentials are present in the environment."""
    return bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET)


def _require_configured() -> None:
    if not is_configured():
        raise GoogleAuthNotConfigured(
            "Google sign-in is not configured on this server. "
            "Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."
        )


def build_authorization_url(state: str) -> str:
    """
    Build the URL the browser is sent to for Google's consent screen.

    `prompt=select_account` is used rather than `consent`: it lets a user pick
    which Google account to sign in with (useful when several are signed in)
    without re-prompting for permissions we already hold.
    """
    _require_configured()
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": settings.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(GOOGLE_SCOPES),
        "state": state,
        "access_type": "online",
        "include_granted_scopes": "true",
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTH_ENDPOINT}?{urlencode(params)}"


def _exchange_code_for_tokens(code: str) -> dict:
    """POST the authorization code to Google's token endpoint and return its JSON."""
    _require_configured()
    try:
        response = httpx.post(
            GOOGLE_TOKEN_ENDPOINT,
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        raise GoogleAuthError(f"Could not reach Google to complete sign-in: {exc}") from exc

    if response.status_code != 200:
        # Google's error body names the cause (redirect_uri_mismatch,
        # invalid_grant on a reused code, ...).  Surfacing it is what makes a
        # misconfigured redirect URI debuggable; it contains no secret.
        raise GoogleAuthError(
            f"Google rejected the authorization code ({response.status_code}): "
            f"{response.text[:300]}"
        )
    return response.json()


def _verify_id_token(raw_id_token: str, *, request=None) -> dict:
    """
    Verify an ID token's signature, issuer, audience and expiry.

    `request` is injectable so tests can supply a transport; production passes
    None and google-auth fetches Google's certificates itself.
    """
    transport = request or google_requests.Request()
    try:
        claims = google_id_token.verify_oauth2_token(
            raw_id_token, transport, settings.GOOGLE_CLIENT_ID
        )
    except ValueError as exc:
        raise GoogleAuthError(f"Google ID token failed verification: {exc}") from exc

    if claims.get("iss") not in _VALID_ISSUERS:
        raise GoogleAuthError("Google ID token has an unexpected issuer.")
    return claims


def identity_from_claims(claims: dict) -> GoogleIdentity:
    """
    Turn verified ID token claims into a GoogleIdentity.

    Split out from `complete_sign_in` so the account-resolution logic can be
    tested end to end without a live Google exchange — the claims are exactly
    what Google returns, minus the network.
    """
    subject = claims.get("sub")
    email = claims.get("email")
    # Google sends this as a bool, but has historically sent the strings
    # "true"/"false" through some endpoints, so both are accepted and anything
    # else is treated as unverified.
    verified = claims.get("email_verified")
    verified = verified is True or verified == "true"

    if not subject:
        raise GoogleAuthError("Google ID token is missing the subject claim.")
    if not email:
        raise GoogleAuthError("Google did not return an email address.")
    if not verified:
        raise GoogleAuthError(
            "This Google account's email address is not verified, so it cannot "
            "be used to sign in."
        )

    name = (claims.get("name") or "").strip() or email.split("@")[0]
    return GoogleIdentity(
        subject=str(subject),
        email=email.lower().strip(),
        email_verified=True,
        name=name[:100],
    )


def complete_sign_in(code: str, *, request=None) -> GoogleIdentity:
    """
    Exchange an authorization code and return the verified Google identity.

    Raises GoogleAuthError for anything that leaves the identity in doubt; the
    caller turns that into an error redirect rather than a signed-in session.
    """
    tokens = _exchange_code_for_tokens(code)
    raw_id_token: Optional[str] = tokens.get("id_token")
    if not raw_id_token:
        raise GoogleAuthError("Google's response did not include an ID token.")
    return identity_from_claims(_verify_id_token(raw_id_token, request=request))
