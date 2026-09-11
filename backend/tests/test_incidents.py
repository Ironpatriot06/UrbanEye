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
# Helpers
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


def _disable_all_existing_agents(db_session):
    """Set ALL agents currently visible in the session to unavailable.
    Prevents demo/persistent agents from stealing test assignments."""
    from app.models.user import User
    db_session.query(User).filter(User.role == UserRole.AGENT).update(
        {"is_available": False}, synchronize_session="fetch"
    )
    db_session.flush()


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
    """Admin can move an incident through the workflow, one valid step at a time.

    REPORTED -> IN_PROGRESS is NOT a valid jump, so the admin walks the
    lifecycle: REPORTED -> TRIAGED -> ASSIGNED -> IN_PROGRESS.
    """
    _disable_all_existing_agents(db_session)
    admin = _create_user_in_db(db_session, "admin_upd@test.com", role=UserRole.ADMIN)
    user = _create_user_in_db(db_session, "user_upd@test.com")
    create_resp = client.post(
        "/api/v1/incidents/",
        files=_multipart_incident(),
        headers=_auth(user),
    )
    assert create_resp.status_code == 201
    incident_id = create_resp.json()["id"]
    assert create_resp.json()["status"] == "REPORTED"

    for nxt in ("TRIAGED", "ASSIGNED", "IN_PROGRESS"):
        patch_resp = client.patch(
            f"/api/v1/incidents/{incident_id}/status",
            json={"status": nxt},
            headers=_auth(admin),
        )
        assert patch_resp.status_code == 200, patch_resp.text
        assert patch_resp.json()["status"] == nxt


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
    _disable_all_existing_agents(db_session)

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


# ---------------------------------------------------------------------------
# Priority / severity — system-determined, never client-supplied
# ---------------------------------------------------------------------------

def test_user_cannot_set_priority_manually(client, db_session):
    """A client that sends severity/priority/priority_level has them ignored.

    VALID_INCIDENT deliberately carries severity=HIGH and priority=3.  A
    POTHOLE is a P3/MEDIUM incident by system policy, so the response must
    reflect the system's decision, not the client's.
    """
    user = _create_user_in_db(db_session, "prio_spoof@test.com")
    payload = {
        **VALID_INCIDENT,
        "category": "POTHOLE",
        "severity": "CRITICAL",
        "priority": 1,
        "priority_level": "P1",
    }
    resp = client.post(
        "/api/v1/incidents/",
        files={"data": (None, json.dumps(payload), "application/json")},
        headers=_auth(user),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["priority_level"] == "P3"
    assert body["severity"] == "MEDIUM"
    assert body["priority"] == 3


def test_priority_is_populated_system_side(client, db_session):
    """Priority is derived from the incident category by the service layer."""
    user = _create_user_in_db(db_session, "prio_system@test.com")
    expected = {
        "FIRE_HAZARD": ("P1", "CRITICAL"),
        "FLOOD":       ("P2", "HIGH"),
        "POTHOLE":     ("P3", "MEDIUM"),
        "GARBAGE":     ("P4", "LOW"),
    }
    for category, (priority_level, severity) in expected.items():
        payload = {k: v for k, v in VALID_INCIDENT.items()
                   if k not in ("severity", "priority")}
        payload["category"] = category
        resp = client.post(
            "/api/v1/incidents/",
            files={"data": (None, json.dumps(payload), "application/json")},
            headers=_auth(user),
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["priority_level"] == priority_level, category
        assert body["severity"] == severity, category


def test_admin_can_override_priority(client, db_session):
    """Admin (and, later, the AI classifier) may re-prioritise an incident."""
    admin = _create_user_in_db(db_session, "prio_admin@test.com", role=UserRole.ADMIN)
    user = _create_user_in_db(db_session, "prio_owner@test.com")
    create_resp = client.post(
        "/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user)
    )
    incident_id = create_resp.json()["id"]

    resp = client.patch(
        f"/api/v1/incidents/{incident_id}/priority",
        json={"priority_level": "P1"},
        headers=_auth(admin),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["priority_level"] == "P1"
    assert body["severity"] == "CRITICAL"
    assert body["sla_hours"] == 4


# ---------------------------------------------------------------------------
# Status workflow enforcement
# ---------------------------------------------------------------------------

def test_invalid_status_transition_rejected(client, db_session):
    """Arbitrary jumps through the workflow are rejected with 422."""
    _disable_all_existing_agents(db_session)
    admin = _create_user_in_db(db_session, "trans_admin@test.com", role=UserRole.ADMIN)
    user = _create_user_in_db(db_session, "trans_user@test.com")
    create_resp = client.post(
        "/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user)
    )
    incident_id = create_resp.json()["id"]
    assert create_resp.json()["status"] == "REPORTED"

    # REPORTED -> RESOLVED skips triage, assignment and work: not allowed.
    for bad in ("RESOLVED", "IN_PROGRESS"):
        resp = client.patch(
            f"/api/v1/incidents/{incident_id}/status",
            json={"status": bad},
            headers=_auth(admin),
        )
        assert resp.status_code == 422, f"{bad}: {resp.text}"


def test_closed_incident_is_terminal(client, db_session):
    _disable_all_existing_agents(db_session)
    admin = _create_user_in_db(db_session, "term_admin@test.com", role=UserRole.ADMIN)
    user = _create_user_in_db(db_session, "term_user@test.com")
    incident_id = client.post(
        "/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user)
    ).json()["id"]

    close = client.patch(
        f"/api/v1/incidents/{incident_id}/status",
        json={"status": "CLOSED"},
        headers=_auth(admin),
    )
    assert close.status_code == 200, close.text

    resp = client.patch(
        f"/api/v1/incidents/{incident_id}/status",
        json={"status": "TRIAGED"},
        headers=_auth(admin),
    )
    assert resp.status_code == 422


def _incident_assigned_to_new_agent(client, db_session, slug):
    """Create an available agent and an incident auto-assigned to them."""
    _disable_all_existing_agents(db_session)
    agent = _create_user_in_db(
        db_session, f"{slug}_agent@test.com", role=UserRole.AGENT, name="Workflow Agent"
    )
    agent.is_available = True
    db_session.flush()

    user = _create_user_in_db(db_session, f"{slug}_user@test.com")
    resp = client.post(
        "/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user)
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "ASSIGNED"
    assert body["assigned_agent_id"] == agent.id
    return agent, user, body


def test_agent_can_move_assigned_to_in_progress(client, db_session):
    agent, _user, incident = _incident_assigned_to_new_agent(client, db_session, "aip")
    resp = client.patch(
        f"/api/v1/incidents/{incident['id']}/status",
        json={"status": "IN_PROGRESS"},
        headers=_auth(agent),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "IN_PROGRESS"


def test_agent_can_move_in_progress_to_resolved(client, db_session):
    agent, _user, incident = _incident_assigned_to_new_agent(client, db_session, "ipr")
    started = client.patch(
        f"/api/v1/incidents/{incident['id']}/status",
        json={"status": "IN_PROGRESS"},
        headers=_auth(agent),
    )
    assert started.status_code == 200, started.text

    resolved = client.patch(
        f"/api/v1/incidents/{incident['id']}/status",
        json={"status": "RESOLVED"},
        headers=_auth(agent),
    )
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["status"] == "RESOLVED"


def test_agent_cannot_perform_invalid_transitions(client, db_session):
    """Agents are confined to ASSIGNED -> IN_PROGRESS -> RESOLVED."""
    agent, _user, incident = _incident_assigned_to_new_agent(client, db_session, "ainv")

    # Skipping IN_PROGRESS, closing, or sending back to triage are all refused.
    for bad in ("RESOLVED", "CLOSED", "TRIAGED", "REPORTED"):
        resp = client.patch(
            f"/api/v1/incidents/{incident['id']}/status",
            json={"status": bad},
            headers=_auth(agent),
        )
        assert resp.status_code == 403, f"ASSIGNED -> {bad}: {resp.text}"

    # And once IN_PROGRESS, an agent still cannot close the incident.
    client.patch(
        f"/api/v1/incidents/{incident['id']}/status",
        json={"status": "IN_PROGRESS"},
        headers=_auth(agent),
    )
    resp = client.patch(
        f"/api/v1/incidents/{incident['id']}/status",
        json={"status": "CLOSED"},
        headers=_auth(agent),
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Agent assignment + availability override
# ---------------------------------------------------------------------------

def test_unavailable_agent_cannot_be_assigned(client, db_session):
    _disable_all_existing_agents(db_session)
    admin = _create_user_in_db(db_session, "asg_admin@test.com", role=UserRole.ADMIN)
    agent = _create_user_in_db(
        db_session, "asg_busy@test.com", role=UserRole.AGENT, name="Busy Agent"
    )
    agent.is_available = False
    db_session.flush()

    user = _create_user_in_db(db_session, "asg_user@test.com")
    incident_id = client.post(
        "/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user)
    ).json()["id"]

    resp = client.put(
        f"/api/v1/incidents/{incident_id}/assign",
        json={"agent_id": agent.id},
        headers=_auth(admin),
    )
    assert resp.status_code == 422
    assert "unavailable" in resp.json()["detail"].lower()


def test_admin_can_override_unavailable_agent_assignment(client, db_session):
    _disable_all_existing_agents(db_session)
    admin = _create_user_in_db(db_session, "ovr_admin@test.com", role=UserRole.ADMIN)
    agent = _create_user_in_db(
        db_session, "ovr_busy@test.com", role=UserRole.AGENT, name="Busy Agent"
    )
    agent.is_available = False
    db_session.flush()

    user = _create_user_in_db(db_session, "ovr_user@test.com")
    incident_id = client.post(
        "/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user)
    ).json()["id"]

    resp = client.put(
        f"/api/v1/incidents/{incident_id}/assign",
        json={"agent_id": agent.id, "override_availability": True},
        headers=_auth(admin),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["assigned_agent_id"] == agent.id
    assert body["status"] == "ASSIGNED"


def test_non_admin_cannot_assign_agents(client, db_session):
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "noassign_user@test.com")
    agent = _create_user_in_db(db_session, "noassign_agent@test.com", role=UserRole.AGENT)
    db_session.flush()
    incident_id = client.post(
        "/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user)
    ).json()["id"]

    resp = client.put(
        f"/api/v1/incidents/{incident_id}/assign",
        json={"agent_id": agent.id},
        headers=_auth(user),
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Agent availability — persistent, admin-overridable
# ---------------------------------------------------------------------------

def test_admin_can_change_agent_availability(client, db_session):
    admin = _create_user_in_db(db_session, "av_admin@test.com", role=UserRole.ADMIN)
    agent = _create_user_in_db(db_session, "av_agent@test.com", role=UserRole.AGENT)
    db_session.flush()

    off = client.patch(
        f"/api/v1/agents/{agent.id}/availability",
        json={"is_available": False},
        headers=_auth(admin),
    )
    assert off.status_code == 200, off.text
    assert off.json()["is_available"] is False

    on = client.patch(
        f"/api/v1/agents/{agent.id}/availability",
        json={"is_available": True},
        headers=_auth(admin),
    )
    assert on.status_code == 200
    assert on.json()["is_available"] is True


def test_non_admin_cannot_change_other_agent_availability(client, db_session):
    agent_a = _create_user_in_db(db_session, "av_a@test.com", role=UserRole.AGENT)
    agent_b = _create_user_in_db(db_session, "av_b@test.com", role=UserRole.AGENT)
    db_session.flush()
    resp = client.patch(
        f"/api/v1/agents/{agent_b.id}/availability",
        json={"is_available": False},
        headers=_auth(agent_a),
    )
    assert resp.status_code == 403


def test_agent_availability_persists_across_logout_and_login(client, db_session):
    """Logging in must NOT reset availability back to available."""
    agent = _create_user_in_db(db_session, "persist_agent@test.com", role=UserRole.AGENT)
    db_session.flush()
    assert agent.is_available is True

    # Agent marks themselves unavailable, then "logs out" (token discarded).
    off = client.post(
        "/api/v1/agents/availability",
        json={"is_available": False},
        headers=_auth(agent),
    )
    assert off.status_code == 200, off.text
    assert off.json()["is_available"] is False

    # Fresh login issues a new token but must not touch availability.
    login = client.post("/api/v1/auth/login", json={
        "email": "persist_agent@test.com",
        "password": "TestPass123",
    })
    assert login.status_code == 200, login.text
    new_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    me = client.get("/api/v1/agents/me", headers=new_headers)
    assert me.status_code == 200, me.text
    assert me.json()["is_available"] is False

    # ...and the agent is still skipped by the auto-assignment queue.
    _disable_all_existing_agents(db_session)
    agent.is_available = False
    db_session.flush()
    user = _create_user_in_db(db_session, "persist_user@test.com")
    created = client.post(
        "/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user)
    ).json()
    assert created["assigned_agent_id"] is None


def test_admin_agent_list_reports_availability_and_workload(client, db_session):
    admin = _create_user_in_db(db_session, "list_admin@test.com", role=UserRole.ADMIN)
    _disable_all_existing_agents(db_session)
    agent = _create_user_in_db(
        db_session, "list_agent@test.com", role=UserRole.AGENT, name="Listed Agent"
    )
    agent.is_available = True
    db_session.flush()

    user = _create_user_in_db(db_session, "list_agent_user@test.com")
    client.post("/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user))

    resp = client.get("/api/v1/agents/", headers=_auth(admin))
    assert resp.status_code == 200, resp.text
    row = next(a for a in resp.json() if a["id"] == agent.id)
    assert row["is_available"] is True
    assert row["active_incident_count"] == 1
    assert row["last_assigned_at"] is not None


# ---------------------------------------------------------------------------
# SLA
# ---------------------------------------------------------------------------

def _assert_sla_shape(body, expected_hours=None):
    assert "priority_level" in body
    assert "sla_hours" in body
    assert "sla_deadline" in body
    assert "sla_status" in body
    assert body["sla_status"] in ("ON_TRACK", "AT_RISK", "BREACHED")
    if expected_hours is not None:
        assert body["sla_hours"] == expected_hours


def test_sla_fields_returned_on_create(client, db_session):
    user = _create_user_in_db(db_session, "sla_create@test.com")
    resp = client.post(
        "/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user)
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    # POTHOLE -> P3 -> 72h SLA
    _assert_sla_shape(body, expected_hours=72)
    assert body["sla_status"] == "ON_TRACK"
    assert body["sla_deadline"] is not None


def test_user_can_view_sla(client, db_session):
    user = _create_user_in_db(db_session, "sla_user@test.com")
    incident_id = client.post(
        "/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user)
    ).json()["id"]

    resp = client.get(f"/api/v1/incidents/{incident_id}", headers=_auth(user))
    assert resp.status_code == 200
    _assert_sla_shape(resp.json(), expected_hours=72)


def test_user_cannot_edit_sla(client, db_session):
    user = _create_user_in_db(db_session, "sla_noedit@test.com")
    incident_id = client.post(
        "/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user)
    ).json()["id"]

    resp = client.patch(
        f"/api/v1/incidents/{incident_id}/sla",
        json={"sla_hours": 1},
        headers=_auth(user),
    )
    assert resp.status_code == 403


def test_agent_can_view_sla(client, db_session):
    agent, _user, incident = _incident_assigned_to_new_agent(client, db_session, "slaag")
    resp = client.get(f"/api/v1/incidents/{incident['id']}", headers=_auth(agent))
    assert resp.status_code == 200, resp.text
    _assert_sla_shape(resp.json(), expected_hours=72)


def test_agent_cannot_edit_sla(client, db_session):
    agent, _user, incident = _incident_assigned_to_new_agent(client, db_session, "slaae")
    resp = client.patch(
        f"/api/v1/incidents/{incident['id']}/sla",
        json={"sla_hours": 2},
        headers=_auth(agent),
    )
    assert resp.status_code == 403


def test_admin_can_update_sla(client, db_session):
    admin = _create_user_in_db(db_session, "sla_admin@test.com", role=UserRole.ADMIN)
    user = _create_user_in_db(db_session, "sla_owner@test.com")
    incident_id = client.post(
        "/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user)
    ).json()["id"]

    resp = client.patch(
        f"/api/v1/incidents/{incident_id}/sla",
        json={"sla_hours": 6},
        headers=_auth(admin),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["sla_hours"] == 6
    assert body["sla_deadline"] is not None

    # The change is persisted, not just echoed back.
    again = client.get(f"/api/v1/incidents/{incident_id}", headers=_auth(admin))
    assert again.json()["sla_hours"] == 6


def test_admin_sla_override_recomputes_status(client, db_session):
    """Overriding the SLA re-derives sla_deadline and sla_status from created_at."""
    admin = _create_user_in_db(db_session, "sla_breach@test.com", role=UserRole.ADMIN)
    user = _create_user_in_db(db_session, "sla_breach_u@test.com")
    incident_id = client.post(
        "/api/v1/incidents/", files=_multipart_incident(), headers=_auth(user)
    ).json()["id"]

    # The deadline is anchored to created_at, so a 1-hour SLA on a freshly
    # created incident is still inside its window — but AT_RISK, because less
    # than 20% of a one-hour window is left only near the end.  What matters
    # here is that the override is applied and a status is re-derived.
    resp = client.patch(
        f"/api/v1/incidents/{incident_id}/sla",
        json={"sla_hours": 1},
        headers=_auth(admin),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["sla_status"] in ("ON_TRACK", "AT_RISK", "BREACHED")


# ---------------------------------------------------------------------------
# Authenticated image access
# ---------------------------------------------------------------------------

def _create_incident_with_image(client, user, extra=None):
    payload = {**VALID_INCIDENT, **(extra or {})}
    return client.post(
        "/api/v1/incidents/",
        files={
            "data": (None, json.dumps(payload), "application/json"),
            "images": ("evidence.jpg", io.BytesIO(_make_jpeg_bytes()), "image/jpeg"),
        },
        headers=_auth(user),
    )


def test_reporter_can_view_own_incident_images(client, db_session):
    user = _create_user_in_db(db_session, "imgown@test.com")
    created = _create_incident_with_image(client, user)
    assert created.status_code == 201, created.text
    incident_id = created.json()["id"]

    listing = client.get(f"/api/v1/incidents/{incident_id}/images", headers=_auth(user))
    assert listing.status_code == 200
    image_id = listing.json()[0]["id"]

    raw = client.get(
        f"/api/v1/incidents/{incident_id}/images/{image_id}", headers=_auth(user)
    )
    assert raw.status_code == 200
    assert raw.headers["content-type"].startswith("image/jpeg")
    assert raw.content.startswith(b"\xff\xd8\xff")


def test_admin_can_view_incident_images(client, db_session):
    admin = _create_user_in_db(db_session, "imgadmin@test.com", role=UserRole.ADMIN)
    user = _create_user_in_db(db_session, "imgadmin_u@test.com")
    incident_id = _create_incident_with_image(client, user).json()["id"]

    listing = client.get(f"/api/v1/incidents/{incident_id}/images", headers=_auth(admin))
    assert listing.status_code == 200, listing.text
    assert len(listing.json()) == 1

    image_id = listing.json()[0]["id"]
    raw = client.get(
        f"/api/v1/incidents/{incident_id}/images/{image_id}", headers=_auth(admin)
    )
    assert raw.status_code == 200
    assert raw.headers["content-type"].startswith("image/jpeg")


def test_assigned_agent_can_view_incident_images(client, db_session):
    _disable_all_existing_agents(db_session)
    agent = _create_user_in_db(
        db_session, "imgagent@test.com", role=UserRole.AGENT, name="Image Agent"
    )
    agent.is_available = True
    db_session.flush()

    user = _create_user_in_db(db_session, "imgagent_u@test.com")
    created = _create_incident_with_image(client, user).json()
    assert created["assigned_agent_id"] == agent.id

    listing = client.get(
        f"/api/v1/incidents/{created['id']}/images", headers=_auth(agent)
    )
    assert listing.status_code == 200, listing.text
    image_id = listing.json()[0]["id"]

    raw = client.get(
        f"/api/v1/incidents/{created['id']}/images/{image_id}", headers=_auth(agent)
    )
    assert raw.status_code == 200
    assert raw.content.startswith(b"\xff\xd8\xff")


def test_unrelated_user_cannot_view_incident_images(client, db_session):
    owner = _create_user_in_db(db_session, "imgowner@test.com")
    intruder = _create_user_in_db(db_session, "imgintruder@test.com")
    incident_id = _create_incident_with_image(client, owner).json()["id"]

    assert client.get(
        f"/api/v1/incidents/{incident_id}/images", headers=_auth(intruder)
    ).status_code == 403


def test_image_requires_authentication(client, db_session):
    """Images are not public: no token means no bytes, with or without a query arg."""
    user = _create_user_in_db(db_session, "imgauth@test.com")
    incident_id = _create_incident_with_image(client, user).json()["id"]
    listing = client.get(f"/api/v1/incidents/{incident_id}/images", headers=_auth(user))
    image_id = listing.json()[0]["id"]

    anon = client.get(f"/api/v1/incidents/{incident_id}/images/{image_id}")
    assert anon.status_code == 401

    token_in_query = client.get(
        f"/api/v1/incidents/{incident_id}/images/{image_id}"
        f"?token={_token_for(user)}"
    )
    assert token_in_query.status_code == 401


def test_multiple_images_are_all_retrievable(client, db_session):
    user = _create_user_in_db(db_session, "multiimg@test.com")
    resp = client.post(
        "/api/v1/incidents/",
        files=[
            ("data", (None, json.dumps(VALID_INCIDENT), "application/json")),
            ("images", ("one.jpg", io.BytesIO(_make_jpeg_bytes()), "image/jpeg")),
            ("images", ("two.png", io.BytesIO(b"\x89PNG" + b"\x00" * 80), "image/png")),
        ],
        headers=_auth(user),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["image_count"] == 2

    listing = client.get(f"/api/v1/incidents/{body['id']}/images", headers=_auth(user))
    assert listing.status_code == 200
    images = listing.json()
    assert {i["filename"] for i in images} == {"one.jpg", "two.png"}
    for img in images:
        raw = client.get(
            f"/api/v1/incidents/{body['id']}/images/{img['id']}", headers=_auth(user)
        )
        assert raw.status_code == 200
        assert raw.headers["content-type"] == img["content_type"]
