"""
UrbanEye+ test suite — Phase 1 + Review III

Testing strategy
----------------
All tests use FastAPI's TestClient against the REAL PostgreSQL/PostGIS database.
Each test that writes data uses a SAVEPOINT nested transaction that is rolled
back after the test — no test data persists in the development database.

The existing 15 Phase 1 tests are preserved with minimal changes:
- Incident creation now requires authentication (JWT token).
- A test_user fixture creates a transient user and token inside the rollback
  transaction.  All existing assertions are unchanged.
"""

import io
import json
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.database import engine, get_db
from app.db.base import Base
from app.main import app
from app.models.user import UserRole
from app.core.security import create_access_token, hash_password


# ---------------------------------------------------------------------------
# Session-level fixture: ensure all tables exist
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def create_tables():
    Base.metadata.create_all(bind=engine)
    yield


# ---------------------------------------------------------------------------
# Per-test rollback session + client
# ---------------------------------------------------------------------------

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
# Helper: create a user directly in the test session
# ---------------------------------------------------------------------------

def _create_user_in_db(db_session, email, role=UserRole.USER, name="Test User"):
    from app.models.user import User
    user = User(
        name=name,
        email=email,
        password_hash=hash_password("TestPass123"),
        role=role,
        is_available=True,
    )
    db_session.add(user)
    db_session.flush()  # assign id without committing to outer tx
    return user


def _token_for(user) -> str:
    return create_access_token({"sub": user.email, "role": user.role})


def _auth(user) -> dict:
    return {"Authorization": f"Bearer {_token_for(user)}"}


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

VALID_INCIDENT = {
    "title": "Test pothole on MG Road",
    "description": "30 cm deep pothole near bus stop 14.",
    "category": "POTHOLE",
    "source": "CITIZEN",
    "severity": "HIGH",
    "priority": 3,
    "latitude": 12.9716,
    "longitude": 77.5946,
}


def _multipart_incident(extra=None):
    payload = {**VALID_INCIDENT, **(extra or {})}
    return {"data": (None, json.dumps(payload), "application/json")}


# ---------------------------------------------------------------------------
# Health tests (no auth needed)
# ---------------------------------------------------------------------------

@pytest.fixture()
def plain_client():
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(plain_client):
    resp = plain_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "service" in resp.json()


def test_health_db_returns_ok(plain_client):
    resp = plain_client.get("/health/db")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "database" in resp.json()


# ---------------------------------------------------------------------------
# Phase 1 incident tests (updated to use auth token)
# ---------------------------------------------------------------------------

def test_create_incident_success(client, db_session):
    user = _create_user_in_db(db_session, "create_ok@test.com")
    resp = client.post(
        "/api/v1/incidents/",
        files=_multipart_incident(),
        headers=_auth(user),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["id"] is not None
    assert body["title"] == VALID_INCIDENT["title"]
    assert body["category"] == "POTHOLE"
    assert body["status"] in ("REPORTED", "ASSIGNED")
    assert body["latitude"] == pytest.approx(12.9716, abs=1e-4)
    assert body["longitude"] == pytest.approx(77.5946, abs=1e-4)


def test_create_incident_missing_title(client, db_session):
    user = _create_user_in_db(db_session, "missing_title@test.com")
    payload = {k: v for k, v in VALID_INCIDENT.items() if k != "title"}
    resp = client.post(
        "/api/v1/incidents/",
        files={"data": (None, json.dumps(payload), "application/json")},
        headers=_auth(user),
    )
    assert resp.status_code == 422


def test_create_incident_invalid_latitude(client, db_session):
    user = _create_user_in_db(db_session, "bad_lat@test.com")
    bad = {**VALID_INCIDENT, "latitude": 91.0}
    resp = client.post(
        "/api/v1/incidents/",
        files={"data": (None, json.dumps(bad), "application/json")},
        headers=_auth(user),
    )
    assert resp.status_code == 422


def test_create_incident_invalid_longitude(client, db_session):
    user = _create_user_in_db(db_session, "bad_lon@test.com")
    bad = {**VALID_INCIDENT, "longitude": -181.0}
    resp = client.post(
        "/api/v1/incidents/",
        files={"data": (None, json.dumps(bad), "application/json")},
        headers=_auth(user),
    )
    assert resp.status_code == 422


def test_create_incident_invalid_category(client, db_session):
    user = _create_user_in_db(db_session, "bad_cat@test.com")
    bad = {**VALID_INCIDENT, "category": "ALIEN_INVASION"}
    resp = client.post(
        "/api/v1/incidents/",
        files={"data": (None, json.dumps(bad), "application/json")},
        headers=_auth(user),
    )
    assert resp.status_code == 422


def test_create_incident_invalid_status_in_body(client, db_session):
    user = _create_user_in_db(db_session, "extra_status@test.com")
    payload = {**VALID_INCIDENT, "status": "CLOSED"}
    resp = client.post(
        "/api/v1/incidents/",
        files={"data": (None, json.dumps(payload), "application/json")},
        headers=_auth(user),
    )
    assert resp.status_code == 201
    assert resp.json()["status"] in ("REPORTED", "ASSIGNED")


def test_get_incident_by_id(client, db_session):
    user = _create_user_in_db(db_session, "get_by_id@test.com")
    create_resp = client.post(
        "/api/v1/incidents/",
        files=_multipart_incident(),
        headers=_auth(user),
    )
    assert create_resp.status_code == 201
    incident_id = create_resp.json()["id"]
    get_resp = client.get(f"/api/v1/incidents/{incident_id}", headers=_auth(user))
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == incident_id


def test_get_incident_not_found(client, db_session):
    user = _create_user_in_db(db_session, "get_404@test.com")
    resp = client.get("/api/v1/incidents/999999", headers=_auth(user))
    assert resp.status_code == 404


def test_list_incidents(client, db_session):
    user = _create_user_in_db(db_session, "list@test.com")
    resp = client.get("/api/v1/incidents/", headers=_auth(user))
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_list_incidents_after_create(client, db_session):
    user = _create_user_in_db(db_session, "list_after@test.com")
    client.post("/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user))
    resp = client.get("/api/v1/incidents/", headers=_auth(user))
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_update_incident_status(client, db_session):
    # Admin can update any incident
    admin = _create_user_in_db(db_session, "admin_upd@test.com", role=UserRole.ADMIN)
    user = _create_user_in_db(db_session, "user_upd@test.com")
    create_resp = client.post(
        "/api/v1/incidents/",
        files=_multipart_incident(),
        headers=_auth(user),
    )
    assert create_resp.status_code == 201
    incident_id = create_resp.json()["id"]
    patch_resp = client.patch(
        f"/api/v1/incidents/{incident_id}/status",
        json={"status": "IN_PROGRESS"},
        headers=_auth(admin),
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "IN_PROGRESS"


def test_update_status_invalid_value(client, db_session):
    admin = _create_user_in_db(db_session, "admin_inv@test.com", role=UserRole.ADMIN)
    user = _create_user_in_db(db_session, "user_inv@test.com")
    create_resp = client.post(
        "/api/v1/incidents/",
        files=_multipart_incident(),
        headers=_auth(user),
    )
    incident_id = create_resp.json()["id"]
    patch_resp = client.patch(
        f"/api/v1/incidents/{incident_id}/status",
        json={"status": "DELETED"},
        headers=_auth(admin),
    )
    assert patch_resp.status_code == 422


def test_update_status_not_found(client, db_session):
    admin = _create_user_in_db(db_session, "admin_nf@test.com", role=UserRole.ADMIN)
    resp = client.patch(
        "/api/v1/incidents/999999/status",
        json={"status": "RESOLVED"},
        headers=_auth(admin),
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

def test_register_success(client):
    resp = client.post("/api/v1/auth/register", json={
        "name": "New Citizen",
        "email": "newcit@test.com",
        "password": "NewPass123",
    })
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "newcit@test.com"
    assert body["role"] == "USER"
    assert "password_hash" not in body


def test_register_duplicate_email(client, db_session):
    _create_user_in_db(db_session, "dup@test.com")
    # Try to register same email via API — should conflict
    resp = client.post("/api/v1/auth/register", json={
        "name": "Dup User",
        "email": "dup@test.com",
        "password": "SomePass123",
    })
    assert resp.status_code == 409


def test_login_success(client, db_session):
    _create_user_in_db(db_session, "login_ok@test.com")
    resp = client.post("/api/v1/auth/login", json={
        "email": "login_ok@test.com",
        "password": "TestPass123",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["role"] == "USER"


def test_login_invalid_password(client, db_session):
    _create_user_in_db(db_session, "wrong_pw@test.com")
    resp = client.post("/api/v1/auth/login", json={
        "email": "wrong_pw@test.com",
        "password": "WrongPassword",
    })
    assert resp.status_code == 401


def test_login_nonexistent_user(client):
    resp = client.post("/api/v1/auth/login", json={
        "email": "nobody@test.com",
        "password": "Whatever123",
    })
    assert resp.status_code == 401


def test_me_endpoint(client, db_session):
    user = _create_user_in_db(db_session, "me@test.com")
    resp = client.get("/api/v1/auth/me", headers=_auth(user))
    assert resp.status_code == 200
    assert resp.json()["email"] == "me@test.com"


def test_me_no_token(client):
    resp = client.get("/api/v1/auth/me")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Role authorization tests
# ---------------------------------------------------------------------------

def test_user_cannot_see_other_users_incident(client, db_session):
    user1 = _create_user_in_db(db_session, "owner@test.com")
    user2 = _create_user_in_db(db_session, "intruder@test.com")
    create_resp = client.post(
        "/api/v1/incidents/",
        files=_multipart_incident(),
        headers=_auth(user1),
    )
    incident_id = create_resp.json()["id"]
    resp = client.get(f"/api/v1/incidents/{incident_id}", headers=_auth(user2))
    assert resp.status_code == 404  # looks like not found to prevent enumeration


def test_user_cannot_update_status(client, db_session):
    user = _create_user_in_db(db_session, "user_no_upd@test.com")
    create_resp = client.post(
        "/api/v1/incidents/",
        files=_multipart_incident(),
        headers=_auth(user),
    )
    incident_id = create_resp.json()["id"]
    resp = client.patch(
        f"/api/v1/incidents/{incident_id}/status",
        json={"status": "RESOLVED"},
        headers=_auth(user),
    )
    assert resp.status_code == 403


def test_admin_can_see_all_incidents(client, db_session):
    user = _create_user_in_db(db_session, "vis_user@test.com")
    admin = _create_user_in_db(db_session, "vis_admin@test.com", role=UserRole.ADMIN)
    client.post("/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user))
    resp = client.get("/api/v1/incidents/", headers=_auth(admin))
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_agent_cannot_access_unrelated_incident(client, db_session):
    user = _create_user_in_db(db_session, "rel_user@test.com")
    agent = _create_user_in_db(db_session, "unrel_agent@test.com", role=UserRole.AGENT)

    # Disable ALL agents (including demo agents) so incident stays unassigned
    _disable_all_existing_agents_inline(db_session)

    create_resp = client.post(
        "/api/v1/incidents/",
        files=_multipart_incident(),
        headers=_auth(user),
    )
    incident_id = create_resp.json()["id"]
    # Incident is REPORTED/unassigned; agent is not assigned to it → 403
    resp = client.get(f"/api/v1/incidents/{incident_id}", headers=_auth(agent))
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Agent queue tests
# ---------------------------------------------------------------------------

def _disable_all_existing_agents(db_session):
    """Set all pre-existing DB agents to unavailable so test agents are isolated."""
    from app.models.user import User
    db_session.query(User).filter(User.role == UserRole.AGENT).update(
        {"is_available": False}, synchronize_session="fetch"
    )
    db_session.flush()


def test_available_agent_receives_incident(client, db_session):
    # Disable any real/demo agents already in the DB to isolate this test
    _disable_all_existing_agents(db_session)

    agent = _create_user_in_db(db_session, "avail_agent@test.com", role=UserRole.AGENT, name="Agent Available")
    agent.is_available = True
    db_session.flush()

    user = _create_user_in_db(db_session, "queue_user@test.com")
    resp = client.post(
        "/api/v1/incidents/",
        files=_multipart_incident(),
        headers=_auth(user),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "ASSIGNED"
    assert body["assigned_agent_id"] == agent.id


def test_unavailable_agent_skipped(client, db_session):
    # Disable all real agents, create one unavailable test agent
    _disable_all_existing_agents(db_session)

    agent = _create_user_in_db(db_session, "unavail_agent@test.com", role=UserRole.AGENT)
    agent.is_available = False
    db_session.flush()

    user = _create_user_in_db(db_session, "no_agent_user@test.com")
    resp = client.post(
        "/api/v1/incidents/",
        files=_multipart_incident(),
        headers=_auth(user),
    )
    assert resp.status_code == 201
    body = resp.json()
    # No available agents → REPORTED, unassigned
    assert body["status"] == "REPORTED"
    assert body["assigned_agent_id"] is None


def test_agent_can_update_assigned_incident(client, db_session):
    # Disable all real agents first, create one available test agent
    _disable_all_existing_agents(db_session)

    agent = _create_user_in_db(db_session, "assigned_agt@test.com", role=UserRole.AGENT)
    agent.is_available = True
    db_session.flush()

    user = _create_user_in_db(db_session, "user_for_agent@test.com")
    create_resp = client.post(
        "/api/v1/incidents/",
        files=_multipart_incident(),
        headers=_auth(user),
    )
    assert create_resp.status_code == 201
    body = create_resp.json()
    assert body["assigned_agent_id"] == agent.id

    patch_resp = client.patch(
        f"/api/v1/incidents/{body['id']}/status",
        json={"status": "IN_PROGRESS"},
        headers=_auth(agent),
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "IN_PROGRESS"


# ---------------------------------------------------------------------------
# Image upload tests
# ---------------------------------------------------------------------------

def _make_jpeg_bytes():
    """Return minimal valid JPEG magic bytes padded to pass validation."""
    return b"\xff\xd8\xff" + b"\x00" * 100


def test_image_upload_valid(client, db_session):
    user = _create_user_in_db(db_session, "img_user@test.com")
    jpeg_bytes = _make_jpeg_bytes()
    resp = client.post(
        "/api/v1/incidents/",
        files={
            "data": (None, json.dumps(VALID_INCIDENT), "application/json"),
            "images": ("test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg"),
        },
        headers=_auth(user),
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["image_count"] == 1


def test_image_invalid_type_rejected(client, db_session):
    user = _create_user_in_db(db_session, "bad_img@test.com")
    resp = client.post(
        "/api/v1/incidents/",
        files={
            "data": (None, json.dumps(VALID_INCIDENT), "application/json"),
            "images": ("script.exe", io.BytesIO(b"MZ\x00\x00bad exe"), "application/octet-stream"),
        },
        headers=_auth(user),
    )
    assert resp.status_code == 422


def test_image_oversized_rejected(client, db_session):
    user = _create_user_in_db(db_session, "big_img@test.com")
    big_data = b"\xff\xd8\xff" + b"\x00" * (6 * 1024 * 1024)  # 6 MB
    resp = client.post(
        "/api/v1/incidents/",
        files={
            "data": (None, json.dumps(VALID_INCIDENT), "application/json"),
            "images": ("big.jpg", io.BytesIO(big_data), "image/jpeg"),
        },
        headers=_auth(user),
    )
    assert resp.status_code == 422


def test_image_retrieval(client, db_session):
    user = _create_user_in_db(db_session, "img_ret@test.com")
    jpeg_bytes = _make_jpeg_bytes()
    create_resp = client.post(
        "/api/v1/incidents/",
        files={
            "data": (None, json.dumps(VALID_INCIDENT), "application/json"),
            "images": ("photo.jpg", io.BytesIO(jpeg_bytes), "image/jpeg"),
        },
        headers=_auth(user),
    )
    incident_id = create_resp.json()["id"]

    list_resp = client.get(f"/api/v1/incidents/{incident_id}/images", headers=_auth(user))
    assert list_resp.status_code == 200
    images = list_resp.json()
    assert len(images) == 1
    assert images[0]["filename"] == "photo.jpg"

    img_id = images[0]["id"]
    raw_resp = client.get(f"/api/v1/incidents/{incident_id}/images/{img_id}", headers=_auth(user))
    assert raw_resp.status_code == 200
    assert raw_resp.headers["content-type"].startswith("image/jpeg")
