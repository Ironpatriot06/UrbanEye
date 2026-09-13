"""
UrbanEye+ — authentication and role-management tests.

Scope
-----
Email/password registration and login, Google sign-in identity resolution,
admin-only authorization on the user-management API, role changes, and the
lockout safeguards that keep at least one administrator able to sign in.

Testing strategy
----------------
Same as the existing suites: FastAPI TestClient against the REAL
PostgreSQL/PostGIS database, with every test wrapped in a transaction that is
rolled back afterwards, so nothing persists in the development database.

Google sign-in
--------------
The network round trip to Google is NOT exercised here — completing a real
OAuth flow needs an interactive browser login against Google's servers, which
cannot run unattended.  What IS exercised is everything on our side of that
boundary:

  - the CSRF state check on the callback,
  - the refusal of an unverified Google email,
  - and the account resolution that turns a verified identity into an account,

by feeding `resolve_google_user` exactly the identity that
`google_oauth.complete_sign_in` returns.  Those tests are real tests of real
code; the remaining manual step is documented in docs/authentication.md.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.auth import OAUTH_STATE_COOKIE
from app.core.security import (
    create_access_token,
    create_oauth_state_token,
    hash_password,
)
from app.db.base import Base
from app.db.database import engine, get_db
from app.main import app
from app.models.user import User, UserRole
from app.models.user_audit import UserAuditAction, UserAuditLog
from app.services import google_oauth, user_service


# ---------------------------------------------------------------------------
# Fixtures — mirror tests/test_incidents.py so the suites behave identically
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def create_tables():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture()
def db_session():
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    session.begin_nested()
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique_email(prefix: str = "t") -> str:
    """A fresh address per test, so reruns never collide with committed rows."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}@urbaneye-test.com"


def _make_user(db, email=None, role=UserRole.USER, name="Test User", **kwargs) -> User:
    user = User(
        name=name,
        email=email or _unique_email(),
        password_hash=hash_password("TestPass123"),
        role=role,
        is_active=kwargs.pop("is_active", True),
        is_available=True,
        **kwargs,
    )
    db.add(user)
    db.flush()
    return user


def _auth(user) -> dict:
    return {"Authorization": f"Bearer {create_access_token({'sub': user.email, 'role': user.role})}"}


def _hide_other_admins(db) -> None:
    """
    Make the user under test the only ADMIN visible to the last-admin guard.

    The development database holds a permanent demo admin, so without this the
    "last remaining admin" rules could never fire inside a test.  The change is
    confined to this test's rolled-back transaction.
    """
    db.query(User).filter(User.role == UserRole.ADMIN).update(
        {"is_active": False}, synchronize_session="fetch"
    )
    db.flush()


class _GoogleIdentity:
    """Stand-in for google_oauth.GoogleIdentity — same attributes, no network."""

    def __init__(self, subject, email, name="Google User"):
        self.subject = subject
        self.email = email
        self.email_verified = True
        self.name = name


# ===========================================================================
# 1. Email registration
# ===========================================================================

def test_register_creates_account_and_returns_session(client):
    email = _unique_email("reg")
    resp = client.post(
        "/api/v1/auth/register",
        json={"name": "New Citizen", "email": email, "password": "GoodPass123"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == email
    assert body["role"] == "USER"
    assert body["access_token"]
    assert body["token_type"] == "bearer"

    # The returned token is immediately usable — registration signs you in.
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200
    assert me.json()["email"] == email


def test_register_always_assigns_user_role(client):
    resp = client.post(
        "/api/v1/auth/register",
        json={"name": "Plain", "email": _unique_email("plain"), "password": "GoodPass123"},
    )
    assert resp.status_code == 201
    assert resp.json()["role"] == "USER"


def test_register_ignores_client_supplied_role(client, db_session):
    """The critical rule: a client cannot choose its own role at signup."""
    email = _unique_email("escalate")
    resp = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Would-be Admin",
            "email": email,
            "password": "GoodPass123",
            "role": "ADMIN",           # ignored
            "is_active": True,
            "google_subject_id": "forged-sub",
        },
    )
    assert resp.status_code == 201
    assert resp.json()["role"] == "USER"

    # And the database agrees — nothing was smuggled through to the row.
    row = db_session.query(User).filter(User.email == email).one()
    assert row.role == UserRole.USER
    assert row.google_subject_id is None


def test_register_ignores_role_in_query_string_too(client, db_session):
    email = _unique_email("qs")
    resp = client.post(
        "/api/v1/auth/register?role=ADMIN",
        json={"name": "QS", "email": email, "password": "GoodPass123"},
    )
    assert resp.status_code == 201
    assert db_session.query(User).filter(User.email == email).one().role == UserRole.USER


def test_register_duplicate_email_rejected(client, db_session):
    user = _make_user(db_session, email=_unique_email("dup"))
    resp = client.post(
        "/api/v1/auth/register",
        json={"name": "Impostor", "email": user.email, "password": "GoodPass123"},
    )
    assert resp.status_code == 409


def test_register_duplicate_email_is_case_insensitive(client, db_session):
    user = _make_user(db_session, email=_unique_email("case"))
    resp = client.post(
        "/api/v1/auth/register",
        json={"name": "Impostor", "email": user.email.upper(), "password": "GoodPass123"},
    )
    assert resp.status_code == 409


def test_password_is_never_stored_in_plaintext(client, db_session):
    email = _unique_email("hash")
    secret = "PlainSecret123"
    resp = client.post(
        "/api/v1/auth/register",
        json={"name": "Hashed", "email": email, "password": secret},
    )
    assert resp.status_code == 201
    # Not echoed back in the response, in any field.
    assert secret not in resp.text

    row = db_session.query(User).filter(User.email == email).one()
    assert row.password_hash != secret
    assert secret not in row.password_hash
    assert row.password_hash.startswith("$2")      # bcrypt
    assert len(row.password_hash) >= 55


def test_register_rejects_weak_password(client):
    resp = client.post(
        "/api/v1/auth/register",
        json={"name": "Weak", "email": _unique_email("weak"), "password": "short"},
    )
    assert resp.status_code == 422


def test_register_rejects_password_without_a_digit(client):
    resp = client.post(
        "/api/v1/auth/register",
        json={"name": "Weak", "email": _unique_email("nodigit"), "password": "alllettershere"},
    )
    assert resp.status_code == 422


def test_register_rejects_malformed_email(client):
    resp = client.post(
        "/api/v1/auth/register",
        json={"name": "Bad", "email": "not-an-email", "password": "GoodPass123"},
    )
    assert resp.status_code == 422


def test_registration_is_recorded_in_the_account_audit_trail(client, db_session):
    email = _unique_email("audit")
    client.post(
        "/api/v1/auth/register",
        json={"name": "Audited", "email": email, "password": "GoodPass123"},
    )
    user = db_session.query(User).filter(User.email == email).one()
    entry = (
        db_session.query(UserAuditLog)
        .filter(
            UserAuditLog.target_user_id == user.id,
            UserAuditLog.action == UserAuditAction.USER_REGISTERED,
        )
        .one()
    )
    assert entry.new_value == "USER"
    # No credential material in the audit trail.
    assert "GoodPass123" not in (entry.description or "")


# ===========================================================================
# 2. Email login
# ===========================================================================

def test_login_with_correct_password_succeeds(client, db_session):
    user = _make_user(db_session, email=_unique_email("ok"))
    resp = client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": "TestPass123"}
    )
    assert resp.status_code == 200
    assert resp.json()["access_token"]
    assert resp.json()["role"] == "USER"


def test_login_with_wrong_password_fails(client, db_session):
    user = _make_user(db_session, email=_unique_email("bad"))
    resp = client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": "NotThePass1"}
    )
    assert resp.status_code == 401


def test_login_with_unknown_account_fails(client):
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": _unique_email("ghost"), "password": "Whatever123"},
    )
    assert resp.status_code == 401


def test_login_does_not_reveal_whether_an_account_exists(client, db_session):
    """Unknown address and wrong password must be indistinguishable."""
    user = _make_user(db_session, email=_unique_email("enum"))
    known = client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": "WrongPass123"}
    )
    unknown = client.post(
        "/api/v1/auth/login",
        json={"email": _unique_email("nobody"), "password": "WrongPass123"},
    )
    assert known.status_code == unknown.status_code == 401
    assert known.json()["detail"] == unknown.json()["detail"]


def test_login_records_last_login_time(client, db_session):
    user = _make_user(db_session, email=_unique_email("stamp"))
    assert user.last_login_at is None
    client.post("/api/v1/auth/login", json={"email": user.email, "password": "TestPass123"})
    db_session.refresh(user)
    assert user.last_login_at is not None


def test_login_is_case_insensitive_on_email(client, db_session):
    user = _make_user(db_session, email=_unique_email("mixed"))
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": user.email.upper(), "password": "TestPass123"},
    )
    assert resp.status_code == 200


def test_deactivated_account_cannot_log_in(client, db_session):
    user = _make_user(db_session, email=_unique_email("off"), is_active=False)
    resp = client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": "TestPass123"}
    )
    assert resp.status_code == 403


def test_deactivated_accounts_existing_token_stops_working(client, db_session):
    user = _make_user(db_session, email=_unique_email("revoked"))
    headers = _auth(user)
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 200

    user.is_active = False
    db_session.flush()
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 401


def test_google_only_account_cannot_log_in_with_a_password(client, db_session):
    """An account with no password hash must not be signed in by any password."""
    user = User(
        name="Google Only",
        email=_unique_email("gonly"),
        password_hash=None,
        google_subject_id=f"sub-{uuid.uuid4().hex}",
        role=UserRole.USER,
        is_active=True,
        is_available=True,
    )
    db_session.add(user)
    db_session.flush()

    for attempt in ("", "anything", "TestPass123"):
        resp = client.post(
            "/api/v1/auth/login", json={"email": user.email, "password": attempt}
        )
        assert resp.status_code == 401, f"password {attempt!r} was accepted"


# ===========================================================================
# 3. Google authentication
# ===========================================================================

def test_google_identity_creates_a_user_account(client, db_session):
    identity = _GoogleIdentity(f"sub-{uuid.uuid4().hex}", _unique_email("g"), "Gina G")
    user, created, linked = user_service.resolve_google_user(db_session, identity)
    assert created is True and linked is False
    assert user.role == UserRole.USER
    assert user.password_hash is None
    assert user.google_subject_id == identity.subject


def test_google_login_never_grants_an_elevated_role(db_session):
    """No domain, and no Google claim, may produce an AGENT or ADMIN account."""
    for domain in ("gmail.com", "vit.ac.in", "company.com", "urbaneye.local"):
        identity = _GoogleIdentity(
            f"sub-{uuid.uuid4().hex}", f"new_{uuid.uuid4().hex[:8]}@{domain}"
        )
        user, created, _ = user_service.resolve_google_user(db_session, identity)
        assert created is True
        assert user.role == UserRole.USER, f"{domain} produced {user.role}"


def test_google_sign_in_reuses_the_existing_account_on_second_visit(db_session):
    identity = _GoogleIdentity(f"sub-{uuid.uuid4().hex}", _unique_email("again"))
    first, created_1, _ = user_service.resolve_google_user(db_session, identity)
    second, created_2, _ = user_service.resolve_google_user(db_session, identity)
    assert created_1 is True and created_2 is False
    assert first.id == second.id


def test_google_sign_in_links_to_an_existing_password_account(db_session):
    """Account linking: same verified email must not create a second account."""
    existing = _make_user(db_session, email=_unique_email("link"), name="Already Here")
    before = db_session.query(User).count()

    identity = _GoogleIdentity(f"sub-{uuid.uuid4().hex}", existing.email)
    user, created, linked = user_service.resolve_google_user(db_session, identity)

    assert created is False and linked is True
    assert user.id == existing.id
    assert user.google_subject_id == identity.subject
    assert user.password_hash is not None       # password still works too
    assert db_session.query(User).count() == before


def test_account_linking_preserves_the_existing_role(db_session):
    """An admin who signs in with Google stays an admin; they are not reset to USER."""
    admin = _make_user(db_session, email=_unique_email("gadmin"), role=UserRole.ADMIN)
    identity = _GoogleIdentity(f"sub-{uuid.uuid4().hex}", admin.email)
    user, created, linked = user_service.resolve_google_user(db_session, identity)
    assert created is False and linked is True
    assert user.role == UserRole.ADMIN


def test_account_linking_is_recorded_in_the_audit_trail(db_session):
    existing = _make_user(db_session, email=_unique_email("linkaudit"))
    identity = _GoogleIdentity(f"sub-{uuid.uuid4().hex}", existing.email)
    user_service.resolve_google_user(db_session, identity)

    entry = (
        db_session.query(UserAuditLog)
        .filter(
            UserAuditLog.target_user_id == existing.id,
            UserAuditLog.action == UserAuditAction.GOOGLE_ACCOUNT_LINKED,
        )
        .one()
    )
    # The trail records that an identity was linked, never the credential.
    assert identity.subject not in (entry.description or "")


def test_google_identity_matches_by_subject_after_an_email_change(db_session):
    """A user changing their Google email must not fork a second account."""
    subject = f"sub-{uuid.uuid4().hex}"
    original = _GoogleIdentity(subject, _unique_email("before"))
    user, created, _ = user_service.resolve_google_user(db_session, original)
    assert created is True

    renamed = _GoogleIdentity(subject, _unique_email("after"))
    same_user, created_again, _ = user_service.resolve_google_user(db_session, renamed)
    assert created_again is False
    assert same_user.id == user.id


def test_unverified_google_email_is_refused(db_session):
    with pytest.raises(google_oauth.GoogleAuthError):
        google_oauth.identity_from_claims(
            {"sub": "123", "email": "unverified@example.com", "email_verified": False}
        )


def test_google_claims_without_an_email_are_refused():
    with pytest.raises(google_oauth.GoogleAuthError):
        google_oauth.identity_from_claims({"sub": "123", "email_verified": True})


def test_google_callback_rejects_a_missing_state_cookie(client):
    """Without the cookie set at /google/login, a callback is a forged request."""
    state = create_oauth_state_token("nonce")
    resp = client.get(
        f"/api/v1/auth/google/callback?code=fake&state={state}",
        follow_redirects=False,
    )
    assert resp.status_code in (302, 307)
    assert "auth_error" in resp.headers["location"]
    assert "token=" not in resp.headers["location"]


def test_google_callback_rejects_a_mismatched_state(client):
    client.cookies.set(OAUTH_STATE_COOKIE, create_oauth_state_token("cookie-nonce"))
    other = create_oauth_state_token("url-nonce")
    resp = client.get(
        f"/api/v1/auth/google/callback?code=fake&state={other}",
        follow_redirects=False,
    )
    assert resp.status_code in (302, 307)
    assert "auth_error" in resp.headers["location"]
    assert "token=" not in resp.headers["location"]


def test_google_callback_reports_a_declined_consent(client):
    resp = client.get(
        "/api/v1/auth/google/callback?error=access_denied", follow_redirects=False
    )
    assert resp.status_code in (302, 307)
    assert "auth_error" in resp.headers["location"]


def test_google_login_returns_503_when_not_configured(client):
    """With no credentials set, the endpoint says so rather than half-working."""
    if google_oauth.is_configured():
        pytest.skip("Google OAuth is configured in this environment.")
    resp = client.get("/api/v1/auth/google/login", follow_redirects=False)
    assert resp.status_code == 503


def test_auth_config_reports_google_availability(client):
    resp = client.get("/api/v1/auth/config")
    assert resp.status_code == 200
    assert resp.json()["google_enabled"] is google_oauth.is_configured()


def test_oauth_state_url_rejects_an_offsite_next_target(client):
    """A crafted `next` must never bounce a fresh token off-site."""
    from app.api.auth import _safe_next_path

    for hostile in ("https://evil.example.com", "//evil.example.com", "http://x"):
        assert _safe_next_path(hostile) == "/auth/callback"
    assert _safe_next_path("/admin/users") == "/admin/users"


# ===========================================================================
# 4. Authorization on the admin user API
# ===========================================================================

def test_unauthenticated_request_to_admin_users_is_401(client):
    assert client.get("/api/v1/admin/users").status_code == 401


def test_user_cannot_access_admin_users(client, db_session):
    user = _make_user(db_session, role=UserRole.USER)
    assert client.get("/api/v1/admin/users", headers=_auth(user)).status_code == 403


def test_agent_cannot_access_admin_users(client, db_session):
    agent = _make_user(db_session, role=UserRole.AGENT)
    assert client.get("/api/v1/admin/users", headers=_auth(agent)).status_code == 403


def test_admin_can_access_admin_users(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    resp = client.get("/api/v1/admin/users", headers=_auth(admin))
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_user_cannot_change_any_role(client, db_session):
    user = _make_user(db_session, role=UserRole.USER)
    victim = _make_user(db_session, role=UserRole.USER)
    resp = client.patch(
        f"/api/v1/admin/users/{victim.id}/role",
        json={"role": "ADMIN"},
        headers=_auth(user),
    )
    assert resp.status_code == 403
    db_session.refresh(victim)
    assert victim.role == UserRole.USER


def test_user_cannot_promote_themselves(client, db_session):
    """The obvious attack: a citizen PATCHing their own row."""
    user = _make_user(db_session, role=UserRole.USER)
    resp = client.patch(
        f"/api/v1/admin/users/{user.id}/role",
        json={"role": "ADMIN"},
        headers=_auth(user),
    )
    assert resp.status_code == 403
    db_session.refresh(user)
    assert user.role == UserRole.USER


def test_agent_cannot_change_any_role(client, db_session):
    agent = _make_user(db_session, role=UserRole.AGENT)
    victim = _make_user(db_session, role=UserRole.USER)
    resp = client.patch(
        f"/api/v1/admin/users/{victim.id}/role",
        json={"role": "AGENT"},
        headers=_auth(agent),
    )
    assert resp.status_code == 403


def test_unauthenticated_role_change_is_401(client, db_session):
    victim = _make_user(db_session, role=UserRole.USER)
    resp = client.patch(f"/api/v1/admin/users/{victim.id}/role", json={"role": "ADMIN"})
    assert resp.status_code == 401


def test_admin_user_list_exposes_no_credentials(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    _make_user(db_session, email=_unique_email("listed"))
    resp = client.get("/api/v1/admin/users", headers=_auth(admin))
    assert resp.status_code == 200
    body = resp.text
    assert "password_hash" not in body
    assert "google_subject_id" not in body
    assert "$2b$" not in body and "$2a$" not in body

    row = resp.json()[0]
    assert set(row) == {
        "id", "name", "email", "role", "is_active", "auth_methods",
        "created_at", "last_login_at", "is_available",
    }


def test_admin_user_list_reports_the_authentication_method(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    pw_user = _make_user(db_session, email=_unique_email("pw"))
    linked = _make_user(db_session, email=_unique_email("both"))
    linked.google_subject_id = f"sub-{uuid.uuid4().hex}"
    db_session.flush()

    rows = {r["id"]: r for r in client.get("/api/v1/admin/users", headers=_auth(admin)).json()}
    assert rows[pw_user.id]["auth_methods"] == ["EMAIL"]
    assert rows[linked.id]["auth_methods"] == ["EMAIL", "GOOGLE"]


def test_newly_registered_user_appears_in_admin_user_list(client, db_session):
    """A registration must show up for the admin without any script being run."""
    admin = _make_user(db_session, role=UserRole.ADMIN)
    email = _unique_email("fresh")
    client.post(
        "/api/v1/auth/register",
        json={"name": "John Doe", "email": email, "password": "GoodPass123"},
    )
    rows = client.get("/api/v1/admin/users", headers=_auth(admin)).json()
    match = [r for r in rows if r["email"] == email]
    assert len(match) == 1
    assert match[0]["role"] == "USER"
    assert match[0]["is_active"] is True
    assert match[0]["auth_methods"] == ["EMAIL"]


def test_admin_can_search_and_filter_users(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    target = _make_user(db_session, email=_unique_email("needle"), name="Findable Person")

    by_search = client.get(
        f"/api/v1/admin/users?q={target.name[:8]}", headers=_auth(admin)
    ).json()
    assert any(r["id"] == target.id for r in by_search)

    by_role = client.get("/api/v1/admin/users?role=AGENT", headers=_auth(admin)).json()
    assert all(r["role"] == "AGENT" for r in by_role)


# ===========================================================================
# 5. Role changes
# ===========================================================================

def test_admin_can_promote_user_to_agent(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    citizen = _make_user(db_session, role=UserRole.USER)
    resp = client.patch(
        f"/api/v1/admin/users/{citizen.id}/role",
        json={"role": "AGENT"},
        headers=_auth(admin),
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "AGENT"


def test_admin_can_promote_user_to_admin(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    citizen = _make_user(db_session, role=UserRole.USER)
    resp = client.patch(
        f"/api/v1/admin/users/{citizen.id}/role",
        json={"role": "ADMIN"},
        headers=_auth(admin),
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "ADMIN"


def test_admin_can_demote_agent_to_user(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    agent = _make_user(db_session, role=UserRole.AGENT)
    resp = client.patch(
        f"/api/v1/admin/users/{agent.id}/role",
        json={"role": "USER"},
        headers=_auth(admin),
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "USER"


def test_admin_can_promote_agent_to_admin(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    agent = _make_user(db_session, role=UserRole.AGENT)
    resp = client.patch(
        f"/api/v1/admin/users/{agent.id}/role",
        json={"role": "ADMIN"},
        headers=_auth(admin),
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "ADMIN"


def test_role_change_persists_in_the_database(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    citizen = _make_user(db_session, role=UserRole.USER)
    client.patch(
        f"/api/v1/admin/users/{citizen.id}/role",
        json={"role": "AGENT"},
        headers=_auth(admin),
    )
    db_session.expire(citizen)
    assert db_session.query(User).filter(User.id == citizen.id).one().role == UserRole.AGENT


def test_new_role_takes_effect_on_the_next_request_with_the_old_token(client, db_session):
    """
    The point of reading the role from the database rather than the JWT claim.

    The promoted user's token was minted while they were a USER and still says
    so, yet the agent-only endpoint now admits them.
    """
    admin = _make_user(db_session, role=UserRole.ADMIN)
    citizen = _make_user(db_session, role=UserRole.USER)
    stale_headers = _auth(citizen)          # token claims role=USER

    assert client.get("/api/v1/agents/me", headers=stale_headers).status_code == 403

    client.patch(
        f"/api/v1/admin/users/{citizen.id}/role",
        json={"role": "AGENT"},
        headers=_auth(admin),
    )
    assert client.get("/api/v1/agents/me", headers=stale_headers).status_code == 200


def test_demotion_takes_effect_immediately_on_an_existing_admin_token(client, db_session):
    """The reverse: a revoked admin's outstanding token loses admin access at once."""
    keeper = _make_user(db_session, role=UserRole.ADMIN, name="Keeper")
    doomed = _make_user(db_session, role=UserRole.ADMIN, name="Doomed")
    doomed_headers = _auth(doomed)

    assert client.get("/api/v1/admin/users", headers=doomed_headers).status_code == 200
    client.patch(
        f"/api/v1/admin/users/{doomed.id}/role",
        json={"role": "USER"},
        headers=_auth(keeper),
    )
    assert client.get("/api/v1/admin/users", headers=doomed_headers).status_code == 403


def test_role_change_is_recorded_in_the_account_audit_trail(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN, name="Auditing Admin")
    citizen = _make_user(db_session, role=UserRole.USER)
    client.patch(
        f"/api/v1/admin/users/{citizen.id}/role",
        json={"role": "AGENT"},
        headers=_auth(admin),
    )
    entry = (
        db_session.query(UserAuditLog)
        .filter(
            UserAuditLog.target_user_id == citizen.id,
            UserAuditLog.action == UserAuditAction.USER_ROLE_CHANGED,
        )
        .one()
    )
    assert entry.old_value == "USER"
    assert entry.new_value == "AGENT"
    assert entry.actor_id == admin.id
    assert entry.actor_email == admin.email


def test_role_change_does_not_touch_incident_history(client, db_session):
    """User-role events belong in the account trail, not the incident trail."""
    from app.models.history import IncidentHistory

    admin = _make_user(db_session, role=UserRole.ADMIN)
    citizen = _make_user(db_session, role=UserRole.USER)
    before = db_session.query(IncidentHistory).count()

    client.patch(
        f"/api/v1/admin/users/{citizen.id}/role",
        json={"role": "AGENT"},
        headers=_auth(admin),
    )
    assert db_session.query(IncidentHistory).count() == before


def test_admin_can_read_the_account_audit_trail(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    citizen = _make_user(db_session, role=UserRole.USER)
    client.patch(
        f"/api/v1/admin/users/{citizen.id}/role",
        json={"role": "ADMIN"},
        headers=_auth(admin),
    )
    resp = client.get(
        f"/api/v1/admin/audit?user_id={citizen.id}", headers=_auth(admin)
    )
    assert resp.status_code == 200
    actions = [e["action"] for e in resp.json()]
    assert "USER_ROLE_CHANGED" in actions


def test_non_admin_cannot_read_the_account_audit_trail(client, db_session):
    agent = _make_user(db_session, role=UserRole.AGENT)
    assert client.get("/api/v1/admin/audit", headers=_auth(agent)).status_code == 403
    assert client.get("/api/v1/admin/audit").status_code == 401


def test_role_change_on_a_missing_user_is_404(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    resp = client.patch(
        "/api/v1/admin/users/99999999/role", json={"role": "AGENT"}, headers=_auth(admin)
    )
    assert resp.status_code == 404


def test_role_change_rejects_an_unknown_role(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    citizen = _make_user(db_session, role=UserRole.USER)
    resp = client.patch(
        f"/api/v1/admin/users/{citizen.id}/role",
        json={"role": "SUPERUSER"},
        headers=_auth(admin),
    )
    assert resp.status_code == 422


# ===========================================================================
# 6. Lockout protection
# ===========================================================================

def test_admin_cannot_change_their_own_role(client, db_session):
    """Even with other admins around — self-edit is the shortest path to lockout."""
    admin = _make_user(db_session, role=UserRole.ADMIN)
    _make_user(db_session, role=UserRole.ADMIN)      # a second admin exists
    resp = client.patch(
        f"/api/v1/admin/users/{admin.id}/role",
        json={"role": "USER"},
        headers=_auth(admin),
    )
    assert resp.status_code == 409
    db_session.refresh(admin)
    assert admin.role == UserRole.ADMIN


def test_admin_cannot_deactivate_themselves(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    _make_user(db_session, role=UserRole.ADMIN)
    resp = client.patch(
        f"/api/v1/admin/users/{admin.id}/status",
        json={"is_active": False},
        headers=_auth(admin),
    )
    assert resp.status_code == 409
    db_session.refresh(admin)
    assert admin.is_active is True


def test_demoting_the_only_remaining_admin_is_refused_over_http(client, db_session):
    """
    Over HTTP the only way to reach "this is the last admin" is to target
    yourself, so the self-rule is what answers — and it answers 409 either way.
    The assertion is on the status and on the account surviving, not on which
    of the two messages came back.
    """
    _hide_other_admins(db_session)
    only_admin = _make_user(db_session, role=UserRole.ADMIN, name="Only Admin")
    # A second ADMIN account exists but is deactivated, so it cannot sign in and
    # must not count towards "an administrator remains".
    _make_user(db_session, role=UserRole.ADMIN, name="Disabled", is_active=False)

    resp = client.patch(
        f"/api/v1/admin/users/{only_admin.id}/role",
        json={"role": "USER"},
        headers=_auth(only_admin),
    )
    assert resp.status_code == 409
    db_session.refresh(only_admin)
    assert only_admin.role == UserRole.ADMIN
    assert user_service.count_active_admins(db_session) == 1


def test_the_last_admin_rule_holds_for_a_non_http_caller(db_session):
    """
    The last-admin rule lives in the service layer, not the router.

    Through the API it is shadowed by the self-rule — an admin performing the
    change is themselves an active admin, so the target can only be the last one
    if it IS them.  It still has to hold for any other caller: a seeding script,
    a management command, or a future endpoint.  This exercises that branch
    directly and pins both messages.
    """
    _hide_other_admins(db_session)
    last = _make_user(db_session, role=UserRole.ADMIN, name="Last Admin")
    other = _make_user(db_session, role=UserRole.USER, name="Some Caller")

    assert user_service.count_active_admins(db_session, exclude_user_id=last.id) == 0

    with pytest.raises(user_service.RoleChangeError, match="last active administrator"):
        user_service.set_user_role(db_session, last, UserRole.USER, actor=other)

    with pytest.raises(user_service.RoleChangeError, match="last active administrator"):
        user_service.set_user_active(db_session, last, False, actor=other)

    db_session.refresh(last)
    assert last.role == UserRole.ADMIN
    assert last.is_active is True


def test_a_deactivated_admin_does_not_count_as_a_surviving_admin(db_session):
    """The guard counts admins who can actually sign in, not ADMIN rows."""
    _hide_other_admins(db_session)
    active = _make_user(db_session, role=UserRole.ADMIN, name="Active")
    _make_user(db_session, role=UserRole.ADMIN, name="Deactivated", is_active=False)

    assert user_service.count_active_admins(db_session) == 1
    assert user_service.count_active_admins(db_session, exclude_user_id=active.id) == 0


def test_demoting_a_peer_admin_is_allowed_while_another_remains(client, db_session):
    """The rule is "never zero admins", not "admins can never be demoted"."""
    _hide_other_admins(db_session)
    keeper = _make_user(db_session, role=UserRole.ADMIN, name="Keeper")
    peer = _make_user(db_session, role=UserRole.ADMIN, name="Peer")

    resp = client.patch(
        f"/api/v1/admin/users/{peer.id}/role",
        json={"role": "USER"},
        headers=_auth(keeper),
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "USER"
    assert user_service.count_active_admins(db_session) == 1


def test_an_admin_can_deactivate_a_peer_admin(client, db_session):
    _hide_other_admins(db_session)
    keeper = _make_user(db_session, role=UserRole.ADMIN, name="Keeper")
    peer = _make_user(db_session, role=UserRole.ADMIN, name="Peer")

    resp = client.patch(
        f"/api/v1/admin/users/{peer.id}/status",
        json={"is_active": False},
        headers=_auth(keeper),
    )
    assert resp.status_code == 200
    assert resp.json()["is_active"] is False
    assert user_service.count_active_admins(db_session) == 1


def test_a_system_always_retains_at_least_one_admin(client, db_session):
    """The invariant itself, stated as a test."""
    _hide_other_admins(db_session)
    a = _make_user(db_session, role=UserRole.ADMIN, name="A")
    b = _make_user(db_session, role=UserRole.ADMIN, name="B")

    client.patch(f"/api/v1/admin/users/{b.id}/role", json={"role": "USER"}, headers=_auth(a))
    # `a` cannot now remove themselves by either route.
    assert client.patch(
        f"/api/v1/admin/users/{a.id}/role", json={"role": "USER"}, headers=_auth(a)
    ).status_code == 409
    assert client.patch(
        f"/api/v1/admin/users/{a.id}/status", json={"is_active": False}, headers=_auth(a)
    ).status_code == 409
    assert user_service.count_active_admins(db_session) >= 1


def test_a_no_op_role_change_is_accepted_and_records_nothing(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    agent = _make_user(db_session, role=UserRole.AGENT)
    before = db_session.query(UserAuditLog).count()
    resp = client.patch(
        f"/api/v1/admin/users/{agent.id}/role",
        json={"role": "AGENT"},
        headers=_auth(admin),
    )
    assert resp.status_code == 200
    assert db_session.query(UserAuditLog).count() == before


# ===========================================================================
# 7. Deactivation
# ===========================================================================

def test_admin_can_deactivate_and_reactivate_a_user(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    citizen = _make_user(db_session, role=UserRole.USER)

    off = client.patch(
        f"/api/v1/admin/users/{citizen.id}/status",
        json={"is_active": False},
        headers=_auth(admin),
    )
    assert off.status_code == 200 and off.json()["is_active"] is False

    on = client.patch(
        f"/api/v1/admin/users/{citizen.id}/status",
        json={"is_active": True},
        headers=_auth(admin),
    )
    assert on.status_code == 200 and on.json()["is_active"] is True


def test_non_admin_cannot_change_account_status(client, db_session):
    agent = _make_user(db_session, role=UserRole.AGENT)
    victim = _make_user(db_session, role=UserRole.USER)
    resp = client.patch(
        f"/api/v1/admin/users/{victim.id}/status",
        json={"is_active": False},
        headers=_auth(agent),
    )
    assert resp.status_code == 403


# ===========================================================================
# 8. Existing accounts keep working
# ===========================================================================

def test_pre_existing_accounts_keep_their_role_and_password(client, db_session):
    """
    An account created the old way — by scripts/create_demo_data.py, before any
    of this existed — must still sign in and keep its elevated role.
    """
    legacy = User(
        name="Legacy Admin",
        email=_unique_email("legacy"),
        password_hash=hash_password("Admin@1234"),
        role=UserRole.ADMIN,
        is_available=True,
    )
    db_session.add(legacy)
    db_session.flush()

    resp = client.post(
        "/api/v1/auth/login", json={"email": legacy.email, "password": "Admin@1234"}
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "ADMIN"
    assert client.get(
        "/api/v1/admin/users",
        headers={"Authorization": f"Bearer {resp.json()['access_token']}"},
    ).status_code == 200


def test_the_seeded_demo_admin_still_works(client, db_session):
    """Guards the account the manual test instructions rely on."""
    from app.core.config import get_settings

    settings = get_settings()
    demo = (
        db_session.query(User)
        .filter(User.email == settings.DEMO_ADMIN_EMAIL)
        .first()
    )
    if demo is None:
        pytest.skip("Demo admin has not been seeded in this database.")
    assert demo.role == UserRole.ADMIN
    assert demo.is_active is True

    resp = client.post(
        "/api/v1/auth/login",
        json={"email": demo.email, "password": settings.DEMO_ADMIN_PASSWORD},
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "ADMIN"


# ===========================================================================
# 9. Audit immutability
# ===========================================================================

def test_account_audit_entries_cannot_be_modified(client, db_session):
    from app.models.user_audit import UserAuditImmutableError

    admin = _make_user(db_session, role=UserRole.ADMIN)
    citizen = _make_user(db_session, role=UserRole.USER)
    client.patch(
        f"/api/v1/admin/users/{citizen.id}/role",
        json={"role": "AGENT"},
        headers=_auth(admin),
    )
    entry = (
        db_session.query(UserAuditLog)
        .filter(UserAuditLog.target_user_id == citizen.id)
        .order_by(UserAuditLog.id.desc())
        .first()
    )
    entry.new_value = "ADMIN"
    with pytest.raises(UserAuditImmutableError):
        db_session.flush()
    db_session.rollback()


def test_account_audit_entries_cannot_be_deleted(client, db_session):
    from app.models.user_audit import UserAuditImmutableError

    admin = _make_user(db_session, role=UserRole.ADMIN)
    citizen = _make_user(db_session, role=UserRole.USER)
    client.patch(
        f"/api/v1/admin/users/{citizen.id}/role",
        json={"role": "AGENT"},
        headers=_auth(admin),
    )
    entry = (
        db_session.query(UserAuditLog)
        .filter(UserAuditLog.target_user_id == citizen.id)
        .order_by(UserAuditLog.id.desc())
        .first()
    )
    db_session.delete(entry)
    with pytest.raises(UserAuditImmutableError):
        db_session.flush()
    db_session.rollback()


def test_there_is_no_write_endpoint_for_the_account_audit_trail(client, db_session):
    admin = _make_user(db_session, role=UserRole.ADMIN)
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        resp = client.request(method, "/api/v1/admin/audit", headers=_auth(admin))
        assert resp.status_code == 405, f"{method} /admin/audit is exposed"
