"""
Authentication API router.

Endpoints
---------
POST /api/v1/auth/register          — Create an account with email + password
POST /api/v1/auth/login             — Log in with email + password
GET  /api/v1/auth/me                — Return the currently authenticated user
GET  /api/v1/auth/config            — Which sign-in methods this server offers
GET  /api/v1/auth/google/login      — Start Google sign-in (302 to Google)
GET  /api/v1/auth/google/callback   — Google's redirect back; issues the session

Roles
-----
No endpoint here accepts a role.  Registration and Google sign-in both create
USER accounts unconditionally, and the schemas have no role field for a client
to populate.  Roles are changed only through /api/v1/admin/users/{id}/role,
which requires an authenticated ADMIN.

Sessions
--------
The existing JWT scheme is used unchanged: a bearer token signed with
SECRET_KEY, carrying the subject email.  Authorization is not read from that
token — see app/core/deps.py.

Google CSRF protection
----------------------
/google/login mints a signed, short-lived `state` token, sends it to Google in
the authorization URL, and also sets it as an HttpOnly, SameSite=Lax cookie.
/google/callback requires both to be present and equal.  An attacker who can
make the browser visit our callback with their own authorization code cannot
also produce the matching cookie, so the forged callback is rejected.
"""

import secrets
from urllib.parse import urlencode

from fastapi import APIRouter, Cookie, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.deps import get_current_user
from app.core.security import (
    create_access_token,
    create_oauth_state_token,
    decode_oauth_state_token,
)
from app.db.database import get_db
from app.models.user import UserRole
from app.schemas.user import (
    RegisterResponse,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserRead,
)
from app.services import google_oauth, user_service
from app.services.user_service import RoleChangeError

router = APIRouter(prefix="/auth", tags=["Authentication"])

settings = get_settings()

#: Name of the HttpOnly cookie holding the OAuth state for the duration of the
#: round trip to Google.
OAUTH_STATE_COOKIE = "ue_oauth_state"


def _issue_session(db: Session, user) -> str:
    """Stamp the sign-in and return a signed access token for the user."""
    user_service.touch_last_login(db, user)
    return create_access_token({"sub": user.email, "role": user.role})


def _safe_next_path(next_path: str | None) -> str:
    """
    Reduce a caller-supplied post-login target to a safe same-site path.

    Anything absolute, protocol-relative, or otherwise off-site is discarded and
    replaced by the default landing page.  Without this, `?next=https://evil...`
    would hand a freshly minted token to an attacker's site.
    """
    default = "/auth/callback"
    if not next_path or not next_path.startswith("/") or next_path.startswith("//"):
        return default
    return next_path


def _frontend_redirect(path: str, **params: str) -> str:
    """Build an absolute URL back into the configured frontend origin."""
    base = settings.FRONTEND_URL.rstrip("/")
    query = f"?{urlencode(params)}" if params else ""
    return f"{base}{path}{query}"


# ---------------------------------------------------------------------------
# Email + password
# ---------------------------------------------------------------------------

@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new account",
    description=(
        "Creates a new account with the USER role and returns a session token "
        "so the browser can continue straight to the dashboard.\n\n"
        "**The role cannot be chosen by the client.** A `role` field in the "
        "request body is ignored; new accounts are always USER. An "
        "administrator grants AGENT or ADMIN afterwards from User Management."
    ),
)
def register(payload: UserCreate, db: Session = Depends(get_db)) -> RegisterResponse:
    existing = user_service.get_user_by_email(db, payload.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )
    # role is hard-coded here; it is not read from `payload`, which has no such
    # field.  This is the only registration path.
    user = user_service.create_user(db, payload, role=UserRole.USER)
    token = _issue_session(db, user)
    return RegisterResponse(
        **UserRead.model_validate(user).model_dump(),
        access_token=token,
        user_id=user.id,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in and receive a JWT access token",
)
def login(payload: UserLogin, db: Session = Depends(get_db)) -> TokenResponse:
    user = user_service.authenticate_user(db, payload.email, payload.password)
    if user is None:
        # One message for a wrong password, an unknown address, and an account
        # that has only ever signed in with Google — so the response cannot be
        # used to enumerate which addresses are registered.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        # Said plainly only to someone who has already proved they hold the
        # password, so it reveals nothing to an enumerator.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account has been deactivated. Contact an administrator.",
        )
    token = _issue_session(db, user)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user.role,
        user_id=user.id,
        name=user.name,
    )


@router.get(
    "/me",
    response_model=UserRead,
    summary="Get the currently authenticated user",
)
def me(current_user=Depends(get_current_user)) -> UserRead:
    return current_user


@router.get(
    "/config",
    summary="Sign-in methods available on this server",
    description=(
        "Lets the login page show or hide the Google button according to what "
        "the server is actually configured for, instead of offering a button "
        "that can only fail. Exposes no credentials — a boolean only."
    ),
)
def auth_config() -> dict:
    return {"google_enabled": google_oauth.is_configured()}


# ---------------------------------------------------------------------------
# Google Sign-In
# ---------------------------------------------------------------------------

@router.get(
    "/google/login",
    summary="Start Google sign-in",
    description=(
        "Redirects the browser to Google's consent screen. Sets a short-lived "
        "HttpOnly `state` cookie that the callback requires."
    ),
)
def google_login(
    next: str | None = Query(
        default=None,
        description="Same-site path to return to after sign-in. Off-site values are ignored.",
    ),
) -> RedirectResponse:
    if not google_oauth.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Google sign-in is not configured on this server. "
                "Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."
            ),
        )

    nonce = secrets.token_urlsafe(24)
    state = create_oauth_state_token(nonce, _safe_next_path(next))
    response = RedirectResponse(
        url=google_oauth.build_authorization_url(state),
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
    )
    response.set_cookie(
        OAUTH_STATE_COOKIE,
        state,
        max_age=settings.OAUTH_STATE_EXPIRE_SECONDS,
        httponly=True,
        # Lax, not Strict: Google's callback is a top-level cross-site
        # navigation, and a Strict cookie would not be sent with it.
        samesite="lax",
        secure=settings.COOKIE_SECURE,
        path="/",
    )
    return response


@router.get(
    "/google/callback",
    summary="Google sign-in callback",
    description=(
        "Google redirects here with an authorization code. The code is "
        "exchanged server-side, the returned ID token is verified, and the "
        "browser is sent back to the frontend with a session token.\n\n"
        "New Google accounts are created with the USER role. An existing "
        "account with the same verified email is linked, keeping whatever role "
        "it already has. The email's domain never affects the role."
    ),
)
def google_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    state_cookie: str | None = Cookie(default=None, alias=OAUTH_STATE_COOKIE),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    def failure(message: str) -> RedirectResponse:
        """Send the browser back to the login page with a readable reason."""
        resp = RedirectResponse(
            url=_frontend_redirect("/login", auth_error=message),
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        )
        resp.delete_cookie(OAUTH_STATE_COOKIE, path="/")
        return resp

    # The user declined consent, or Google refused the request.
    if error:
        return failure(f"Google sign-in was cancelled or refused ({error}).")

    if not code or not state:
        return failure("Google sign-in returned an incomplete response.")

    # CSRF: the state must be one we signed, unexpired, AND identical to the
    # cookie set when this browser started the flow.
    if not state_cookie or not secrets.compare_digest(state, state_cookie):
        return failure(
            "Google sign-in could not be verified. Please start again from the "
            "login page in the same browser."
        )
    state_payload = decode_oauth_state_token(state)
    if state_payload is None:
        return failure("The Google sign-in request expired. Please try again.")

    try:
        identity = google_oauth.complete_sign_in(code)
    except google_oauth.GoogleAuthError as exc:
        return failure(str(exc))

    try:
        user, _created, _linked = user_service.resolve_google_user(db, identity)
    except RoleChangeError as exc:
        return failure(str(exc))

    if not user.is_active:
        return failure("This account has been deactivated. Contact an administrator.")

    token = _issue_session(db, user)

    destination = _safe_next_path(state_payload.get("next"))
    response = RedirectResponse(
        url=_frontend_redirect(
            destination,
            token=token,
            role=user.role.value,
            user_id=str(user.id),
            name=user.name,
        ),
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
    )
    # The state has done its job; leaving it around only widens the window in
    # which it could be replayed.
    response.delete_cookie(OAUTH_STATE_COOKIE, path="/")
    return response
