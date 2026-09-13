"""
UrbanEye+ test suite — incident history / audit trail.

Testing strategy
----------------
Identical to tests/test_incidents.py: FastAPI's TestClient against the real
PostgreSQL/PostGIS database, with every test inside a SAVEPOINT that is rolled
back afterwards, so nothing persists in the development database.

One consequence matters here.  The whole test runs inside a single
transaction, so PostgreSQL's now() is frozen for its duration — which is
exactly why IncidentHistory.created_at uses clock_timestamp() and the service
layer stamps an explicit wall-clock time.  The chronology test below asserts
that this actually holds rather than trusting it.
"""

import io
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password
from app.db.base import Base
from app.db.database import engine, get_db
from app.main import app
from app.models.history import (
    ActorRole,
    HistoryAction,
    HistoryImmutableError,
    IncidentHistory,
)
from app.models.user import UserRole
from app.services import history_service


# ---------------------------------------------------------------------------
# Fixtures — mirror tests/test_incidents.py
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

def _create_user_in_db(db_session, email, role=UserRole.USER, name="Test User",
                       is_available=True):
    from app.models.user import User

    user = User(
        name=name,
        email=email,
        password_hash=hash_password("TestPass123"),
        role=role,
        is_available=is_available,
    )
    db_session.add(user)
    db_session.flush()
    return user


def _disable_all_existing_agents(db_session):
    """Take every pre-existing agent out of the queue.

    Without this, a demo agent left available in the development database would
    win the automatic assignment and the test's own agent would never be
    picked.
    """
    from app.models.user import User

    db_session.query(User).filter(User.role == UserRole.AGENT).update(
        {"is_available": False}, synchronize_session="fetch"
    )
    db_session.flush()


def _auth(user) -> dict:
    return {"Authorization": f"Bearer {create_access_token({'sub': user.email, 'role': user.role})}"}


VALID_INCIDENT = {
    "title": "Audit trail test incident",
    "description": "Used by the incident history tests.",
    "category": "POTHOLE",
    "source": "CITIZEN",
    "latitude": 12.9716,
    "longitude": 77.5946,
}


def _png_bytes() -> bytes:
    """Smallest byte string that passes the image magic-byte check."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _create_incident(client, user, extra=None, with_image=False):
    """Create an incident through the API and return the response body."""
    payload = {**VALID_INCIDENT, **(extra or {})}
    files = {"data": (None, json.dumps(payload), "application/json")}
    if with_image:
        files["images"] = ("evidence.png", io.BytesIO(_png_bytes()), "image/png")
    resp = client.post("/api/v1/incidents/", files=files, headers=_auth(user))
    assert resp.status_code == 201, resp.text
    return resp.json()


def _history(client, user, incident_id):
    """Fetch an incident's history as the given user, asserting success."""
    resp = client.get(f"/api/v1/incidents/{incident_id}/history", headers=_auth(user))
    assert resp.status_code == 200, resp.text
    return resp.json()


def _actions(events) -> list:
    return [e["action"] for e in events]


# ---------------------------------------------------------------------------
# 1-8: operations generate the right history
# ---------------------------------------------------------------------------

def test_incident_creation_creates_history(client, db_session):
    """Reporting an incident opens its trail with INCIDENT_CREATED by the reporter."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_create@test.com", name="Ratish Kapoor")

    incident = _create_incident(client, user)
    events = _history(client, user, incident["id"])

    assert len(events) >= 1
    first = events[0]
    assert first["action"] == HistoryAction.INCIDENT_CREATED.value
    assert first["actor_name"] == "Ratish Kapoor"
    assert first["actor_role"] == ActorRole.USER.value
    assert first["actor_id"] == user.id
    assert first["new_value"] == "REPORTED"
    assert first["incident_id"] == incident["id"]
    assert "Ratish Kapoor" in first["description"]

    # The SLA the incident was born with is recorded too, so a later reader can
    # see where the current response target came from.
    assert HistoryAction.SLA_CREATED.value in _actions(events)


def test_status_change_creates_history(client, db_session):
    """Each status transition records old and new value, attributed to the actor."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_status_u@test.com")
    admin = _create_user_in_db(
        db_session, "hist_status_a@test.com", role=UserRole.ADMIN, name="City Admin"
    )

    incident = _create_incident(client, user)
    assert incident["status"] == "REPORTED"  # no agent was available

    resp = client.patch(
        f"/api/v1/incidents/{incident['id']}/status",
        json={"status": "TRIAGED"},
        headers=_auth(admin),
    )
    assert resp.status_code == 200, resp.text

    events = _history(client, admin, incident["id"])
    triaged = [e for e in events if e["action"] == HistoryAction.INCIDENT_TRIAGED.value]
    assert len(triaged) == 1
    assert triaged[0]["old_value"] == "REPORTED"
    assert triaged[0]["new_value"] == "TRIAGED"
    assert triaged[0]["actor_role"] == ActorRole.ADMIN.value
    assert triaged[0]["actor_name"] == "City Admin"


def test_assignment_creates_history(client, db_session):
    """An admin assigning an agent records AGENT_ASSIGNED plus the status move."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_assign_u@test.com")
    admin = _create_user_in_db(
        db_session, "hist_assign_a@test.com", role=UserRole.ADMIN, name="City Admin"
    )

    # The incident is created while no agent is available, so it stays
    # REPORTED and the admin's manual assignment is what the test observes.
    incident = _create_incident(client, user)
    assert incident["assigned_agent_id"] is None

    agent = _create_user_in_db(
        db_session, "hist_assign_ag@test.com", role=UserRole.AGENT, name="Priya Sharma"
    )

    resp = client.put(
        f"/api/v1/incidents/{incident['id']}/assign",
        json={"agent_id": agent.id, "override_availability": False},
        headers=_auth(admin),
    )
    assert resp.status_code == 200, resp.text

    events = _history(client, admin, incident["id"])
    assigned = [e for e in events if e["action"] == HistoryAction.AGENT_ASSIGNED.value]
    assert len(assigned) == 1
    assert assigned[0]["new_value"] == "Priya Sharma"
    assert assigned[0]["actor_role"] == ActorRole.ADMIN.value

    # Assignment moved REPORTED -> ASSIGNED; that side effect is recorded too.
    status_events = [
        e for e in events
        if e["action"] == HistoryAction.STATUS_CHANGED.value
        and e["new_value"] == "ASSIGNED"
    ]
    assert len(status_events) == 1
    assert status_events[0]["old_value"] == "REPORTED"


def test_reassignment_creates_history(client, db_session):
    """Moving an incident between agents records both the old and new holder."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_reassign_u@test.com")
    admin = _create_user_in_db(
        db_session, "hist_reassign_a@test.com", role=UserRole.ADMIN, name="City Admin"
    )

    incident = _create_incident(client, user)

    first = _create_user_in_db(
        db_session, "hist_reassign_1@test.com", role=UserRole.AGENT, name="Agent One"
    )
    second = _create_user_in_db(
        db_session, "hist_reassign_2@test.com", role=UserRole.AGENT, name="Agent Two"
    )

    for agent in (first, second):
        resp = client.put(
            f"/api/v1/incidents/{incident['id']}/assign",
            json={"agent_id": agent.id, "override_availability": False},
            headers=_auth(admin),
        )
        assert resp.status_code == 200, resp.text

    events = _history(client, admin, incident["id"])
    reassigned = [
        e for e in events if e["action"] == HistoryAction.AGENT_REASSIGNED.value
    ]
    assert len(reassigned) == 1
    assert reassigned[0]["old_value"] == "Agent One"
    assert reassigned[0]["new_value"] == "Agent Two"


def test_unassignment_creates_history(client, db_session):
    """Removing the agent records AGENT_UNASSIGNED naming who was removed."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_unassign_u@test.com")
    admin = _create_user_in_db(
        db_session, "hist_unassign_a@test.com", role=UserRole.ADMIN
    )

    incident = _create_incident(client, user)

    agent = _create_user_in_db(
        db_session, "hist_unassign_ag@test.com", role=UserRole.AGENT, name="Agent One"
    )
    client.put(
        f"/api/v1/incidents/{incident['id']}/assign",
        json={"agent_id": agent.id, "override_availability": False},
        headers=_auth(admin),
    )
    resp = client.put(
        f"/api/v1/incidents/{incident['id']}/assign",
        json={"agent_id": None, "override_availability": False},
        headers=_auth(admin),
    )
    assert resp.status_code == 200, resp.text

    events = _history(client, admin, incident["id"])
    unassigned = [
        e for e in events if e["action"] == HistoryAction.AGENT_UNASSIGNED.value
    ]
    assert len(unassigned) == 1
    assert unassigned[0]["old_value"] == "Agent One"


def test_priority_change_creates_history(client, db_session):
    """A priority override records P-level old/new, and the SLA move it causes."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_prio_u@test.com")
    admin = _create_user_in_db(db_session, "hist_prio_a@test.com", role=UserRole.ADMIN)

    # POTHOLE defaults to P3, so P1 is a genuine change.
    incident = _create_incident(client, user)
    assert incident["priority_level"] == "P3"

    resp = client.patch(
        f"/api/v1/incidents/{incident['id']}/priority",
        json={"priority_level": "P1"},
        headers=_auth(admin),
    )
    assert resp.status_code == 200, resp.text

    events = _history(client, admin, incident["id"])
    changed = [e for e in events if e["action"] == HistoryAction.PRIORITY_CHANGED.value]
    assert len(changed) == 1
    assert changed[0]["old_value"] == "P3"
    assert changed[0]["new_value"] == "P1"
    assert changed[0]["actor_role"] == ActorRole.ADMIN.value

    # Re-prioritising resets the SLA window (72h -> 4h); the trail explains it
    # rather than leaving the reader with an SLA that changed by itself.
    sla_events = [e for e in events if e["action"] == HistoryAction.SLA_UPDATED.value]
    assert len(sla_events) == 1
    assert sla_events[0]["old_value"] == "72 hours"
    assert sla_events[0]["new_value"] == "4 hours"


def test_sla_update_creates_history(client, db_session):
    """An admin SLA override records the old and new window as readable text."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_sla_u@test.com")
    admin = _create_user_in_db(db_session, "hist_sla_a@test.com", role=UserRole.ADMIN)

    incident = _create_incident(client, user)
    assert incident["sla_hours"] == 72

    resp = client.patch(
        f"/api/v1/incidents/{incident['id']}/sla",
        json={"sla_hours": 8},
        headers=_auth(admin),
    )
    assert resp.status_code == 200, resp.text

    events = _history(client, admin, incident["id"])
    sla_events = [e for e in events if e["action"] == HistoryAction.SLA_UPDATED.value]
    assert len(sla_events) == 1
    assert sla_events[0]["old_value"] == "72 hours"
    assert sla_events[0]["new_value"] == "8 hours"
    assert sla_events[0]["actor_role"] == ActorRole.ADMIN.value


def test_image_upload_creates_history(client, db_session):
    """A stored image records IMAGE_ADDED with metadata, never the bytes."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_img@test.com", name="Ratish Kapoor")

    incident = _create_incident(client, user, with_image=True)
    assert incident["image_count"] == 1

    events = _history(client, user, incident["id"])
    images = [e for e in events if e["action"] == HistoryAction.IMAGE_ADDED.value]
    assert len(images) == 1
    assert images[0]["new_value"] == "evidence.png"
    assert images[0]["actor_role"] == ActorRole.USER.value
    assert "image/png" in images[0]["description"]

    # The history row carries metadata only — no image payload leaks into it.
    row = (
        db_session.query(IncidentHistory)
        .filter(IncidentHistory.id == images[0]["id"])
        .one()
    )
    assert b"PNG" not in (row.description or "").encode()
    assert all(
        not isinstance(getattr(row, col), (bytes, bytearray))
        for col in ("old_value", "new_value", "description")
    )


def test_rejected_image_upload_creates_no_history(client, db_session):
    """A rejected upload attached nothing, so it records nothing."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_img_bad@test.com")
    incident = _create_incident(client, user)

    before = len(_history(client, user, incident["id"]))

    resp = client.post(
        "/api/v1/incidents/",
        files={
            "data": (None, json.dumps(VALID_INCIDENT), "application/json"),
            "images": ("evil.txt", io.BytesIO(b"not an image"), "text/plain"),
        },
        headers=_auth(user),
    )
    assert resp.status_code == 422

    assert len(_history(client, user, incident["id"])) == before


def test_admin_override_creates_history(client, db_session):
    """Force-assigning an unavailable agent records ADMIN_OVERRIDE and the assignment."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_ovr_u@test.com")
    admin = _create_user_in_db(
        db_session, "hist_ovr_a@test.com", role=UserRole.ADMIN, name="City Admin"
    )
    agent = _create_user_in_db(
        db_session,
        "hist_ovr_ag@test.com",
        role=UserRole.AGENT,
        name="Busy Agent",
        is_available=False,
    )

    incident = _create_incident(client, user)

    # Without the override flag the backend refuses, and records nothing.
    refused = client.put(
        f"/api/v1/incidents/{incident['id']}/assign",
        json={"agent_id": agent.id, "override_availability": False},
        headers=_auth(admin),
    )
    assert refused.status_code == 422
    assert HistoryAction.ADMIN_OVERRIDE.value not in _actions(
        _history(client, admin, incident["id"])
    )

    resp = client.put(
        f"/api/v1/incidents/{incident['id']}/assign",
        json={"agent_id": agent.id, "override_availability": True},
        headers=_auth(admin),
    )
    assert resp.status_code == 200, resp.text

    events = _history(client, admin, incident["id"])
    overrides = [e for e in events if e["action"] == HistoryAction.ADMIN_OVERRIDE.value]
    assert len(overrides) == 1
    assert overrides[0]["actor_role"] == ActorRole.ADMIN.value
    assert "Busy Agent" in overrides[0]["description"]
    assert "unavailable" in overrides[0]["description"].lower()

    # The override and the assignment it enabled are both recorded.
    assert HistoryAction.AGENT_ASSIGNED.value in _actions(events)


def test_agent_availability_change_creates_history(client, db_session):
    """Availability changes are recorded against the agent's open incidents."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_avail_u@test.com")
    admin = _create_user_in_db(
        db_session, "hist_avail_a@test.com", role=UserRole.ADMIN, name="City Admin"
    )
    agent = _create_user_in_db(
        db_session, "hist_avail_ag@test.com", role=UserRole.AGENT, name="Priya Sharma"
    )

    # Created while the agent is available, so it is auto-assigned to them.
    incident = _create_incident(client, user)
    assert incident["assigned_agent_id"] == agent.id

    resp = client.patch(
        f"/api/v1/agents/{agent.id}/availability",
        json={"is_available": False},
        headers=_auth(admin),
    )
    assert resp.status_code == 200, resp.text

    events = _history(client, admin, incident["id"])
    avail = [
        e for e in events
        if e["action"] == HistoryAction.AGENT_AVAILABILITY_CHANGED.value
    ]
    assert len(avail) == 1
    assert avail[0]["new_value"] == "Unavailable"
    assert avail[0]["actor_role"] == ActorRole.ADMIN.value
    assert "Priya Sharma" in avail[0]["description"]


# ---------------------------------------------------------------------------
# 9-12: authorization
# ---------------------------------------------------------------------------

def test_user_can_view_own_history(client, db_session):
    """A citizen reads the trail of an incident they reported."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_own@test.com", name="Ratish Kapoor")
    incident = _create_incident(client, user)

    resp = client.get(
        f"/api/v1/incidents/{incident['id']}/history", headers=_auth(user)
    )
    assert resp.status_code == 200
    assert len(resp.json()) >= 1
    assert resp.json()[0]["action"] == HistoryAction.INCIDENT_CREATED.value


def test_user_cannot_view_other_users_history(client, db_session):
    """One citizen cannot read another's trail, and is not told it exists."""
    _disable_all_existing_agents(db_session)
    owner = _create_user_in_db(db_session, "hist_owner@test.com")
    intruder = _create_user_in_db(db_session, "hist_intruder@test.com")

    incident = _create_incident(client, owner)

    resp = client.get(
        f"/api/v1/incidents/{incident['id']}/history", headers=_auth(intruder)
    )
    # 404, not 403: the same answer they get for the incident itself, so the
    # endpoint never confirms that an incident they cannot see exists.
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_history_requires_authentication(client, db_session):
    """An unauthenticated request is rejected outright."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_anon@test.com")
    incident = _create_incident(client, user)

    assert client.get(f"/api/v1/incidents/{incident['id']}/history").status_code == 401


def test_agent_can_view_assigned_incident_history(client, db_session):
    """An agent reads the trail of an incident assigned to them."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_agent_u@test.com")
    agent = _create_user_in_db(
        db_session, "hist_agent_ag@test.com", role=UserRole.AGENT, name="Priya Sharma"
    )

    incident = _create_incident(client, user)
    assert incident["assigned_agent_id"] == agent.id

    events = _history(client, agent, incident["id"])
    assert HistoryAction.INCIDENT_CREATED.value in _actions(events)
    assert HistoryAction.AGENT_ASSIGNED.value in _actions(events)


def test_agent_cannot_view_unassigned_incident_history(client, db_session):
    """An agent stays inside their existing authorization scope."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_other_u@test.com")
    outsider = _create_user_in_db(
        db_session, "hist_outsider@test.com", role=UserRole.AGENT, is_available=False
    )

    incident = _create_incident(client, user)
    assert incident["assigned_agent_id"] is None

    resp = client.get(
        f"/api/v1/incidents/{incident['id']}/history", headers=_auth(outsider)
    )
    assert resp.status_code == 403


def test_admin_can_view_any_history(client, db_session):
    """An admin reads any incident's trail, including events hidden from the reporter."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_admin_u@test.com")
    admin = _create_user_in_db(db_session, "hist_admin_a@test.com", role=UserRole.ADMIN)
    agent = _create_user_in_db(
        db_session, "hist_admin_ag@test.com", role=UserRole.AGENT, name="Priya Sharma"
    )

    incident = _create_incident(client, user)
    client.patch(
        f"/api/v1/agents/{agent.id}/availability",
        json={"is_available": False},
        headers=_auth(admin),
    )

    admin_events = _history(client, admin, incident["id"])
    assert HistoryAction.AGENT_AVAILABILITY_CHANGED.value in _actions(admin_events)

    # The reporter sees their incident's progress, but not internal staffing.
    user_events = _history(client, user, incident["id"])
    assert HistoryAction.AGENT_AVAILABILITY_CHANGED.value not in _actions(user_events)
    assert HistoryAction.INCIDENT_CREATED.value in _actions(user_events)
    assert HistoryAction.AGENT_ASSIGNED.value in _actions(user_events)


def test_history_is_not_writable_through_the_api(client, db_session):
    """No role can post, edit or delete an audit entry — the routes do not exist."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_write_u@test.com")
    admin = _create_user_in_db(db_session, "hist_write_a@test.com", role=UserRole.ADMIN)
    incident = _create_incident(client, user)

    path = f"/api/v1/incidents/{incident['id']}/history"
    forged = {
        "action": "INCIDENT_CLOSED",
        "actor_role": "ADMIN",
        "description": "forged",
    }

    # Even an admin — the most privileged role — has no write path.
    for actor in (admin, user):
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            resp = client.request(method, path, json=forged, headers=_auth(actor))
            assert resp.status_code == 405, (
                f"{method} {path} as {actor.role} -> {resp.status_code}"
            )

    # Nothing was appended by any of those attempts.
    assert HistoryAction.INCIDENT_CLOSED.value not in _actions(
        _history(client, admin, incident["id"])
    )


def test_client_cannot_forge_actor_on_incident_creation(client, db_session):
    """Actor fields sent by a client are ignored; identity comes from the JWT."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_forge@test.com", name="Real Reporter")
    admin = _create_user_in_db(db_session, "hist_forge_a@test.com", role=UserRole.ADMIN)

    incident = _create_incident(
        client,
        user,
        extra={
            "actor_id": admin.id,
            "actor_role": "ADMIN",
            "actor_name": "Somebody Else",
            "created_at": "1999-01-01T00:00:00Z",
        },
    )

    first = _history(client, user, incident["id"])[0]
    assert first["actor_id"] == user.id
    assert first["actor_role"] == ActorRole.USER.value
    assert first["actor_name"] == "Real Reporter"
    assert not first["created_at"].startswith("1999")


# ---------------------------------------------------------------------------
# 13-15: immutability, ordering, backfill
# ---------------------------------------------------------------------------

def test_history_is_immutable(client, db_session):
    """Backend code cannot rewrite or remove an audit entry either."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_immutable@test.com")
    incident = _create_incident(client, user)

    def first_entry():
        return (
            db_session.query(IncidentHistory)
            .filter(IncidentHistory.incident_id == incident["id"])
            .order_by(IncidentHistory.id)
            .first()
        )

    entry = first_entry()
    assert entry is not None
    entry_id = entry.id
    original_action = entry.action
    original_description = entry.description

    # Each attempt gets its own savepoint: the guard raises mid-flush, which
    # leaves the session needing a rollback, and rolling the whole session back
    # would discard the incident this test just created.
    savepoint = db_session.begin_nested()
    entry.description = "rewritten by a bug"
    with pytest.raises(HistoryImmutableError):
        db_session.flush()
    savepoint.rollback()

    savepoint = db_session.begin_nested()
    db_session.delete(db_session.get(IncidentHistory, entry_id))
    with pytest.raises(HistoryImmutableError):
        db_session.flush()
    savepoint.rollback()

    # The entry survived both attempts, unchanged and still present.
    db_session.expire_all()
    survivor = first_entry()
    assert survivor is not None
    assert survivor.id == entry_id
    assert survivor.action == original_action
    assert survivor.description == original_description


def test_history_is_chronological(client, db_session):
    """Events come back oldest first, including several written in one transaction."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_chrono_u@test.com")
    admin = _create_user_in_db(db_session, "hist_chrono_a@test.com", role=UserRole.ADMIN)

    incident = _create_incident(client, user)

    agent = _create_user_in_db(
        db_session, "hist_chrono_ag@test.com", role=UserRole.AGENT, name="Agent One"
    )
    client.patch(
        f"/api/v1/incidents/{incident['id']}/priority",
        json={"priority_level": "P1"},
        headers=_auth(admin),
    )
    client.put(
        f"/api/v1/incidents/{incident['id']}/assign",
        json={"agent_id": agent.id, "override_availability": False},
        headers=_auth(admin),
    )

    events = _history(client, admin, incident["id"])
    assert len(events) >= 5

    timestamps = [e["created_at"] for e in events]
    assert timestamps == sorted(timestamps), "history is not in chronological order"

    ids = [e["id"] for e in events]
    assert ids == sorted(ids), "history order disagrees with insertion order"

    # The first event is always the opening one.
    assert events[0]["action"] == HistoryAction.INCIDENT_CREATED.value

    # The whole test runs in one transaction, where PostgreSQL's now() is
    # frozen.  Distinct timestamps prove clock_timestamp()/wall-clock stamping
    # is doing its job rather than collapsing every event onto one instant.
    assert len(set(timestamps)) > 1


def test_existing_incidents_have_initial_history(client, db_session):
    """An incident predating the audit trail is backfilled with its opening event."""
    from app.models.incident import Incident, IncidentPriority, IncidentStatus

    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_legacy@test.com", name="Legacy Reporter")

    # A row written straight to the table, as a pre-feature incident would be:
    # no history, because the service layer never saw it.
    legacy = Incident(
        title="Legacy incident with no audit trail",
        description="Created before the history feature existed.",
        category="POTHOLE",
        source="CITIZEN",
        status=IncidentStatus.IN_PROGRESS,
        priority_level=IncidentPriority.P3,
        severity="MEDIUM",
        priority=3,
        latitude=12.9716,
        longitude=77.5946,
        location="SRID=4326;POINT(77.5946 12.9716)",
        reported_by=user.id,
    )
    db_session.add(legacy)
    db_session.flush()

    assert not history_service.has_history(db_session, legacy.id)

    written = history_service.backfill_initial_history(db_session)
    assert written >= 1

    events = _history(client, user, legacy.id)
    assert len(events) == 1
    assert events[0]["action"] == HistoryAction.INCIDENT_CREATED.value
    assert events[0]["actor_name"] == "Legacy Reporter"
    assert events[0]["new_value"] == "REPORTED"

    # The backfill states only what is known. It does not invent the triage,
    # assignment or start-of-work that must have happened to reach IN_PROGRESS.
    assert _actions(events) == [HistoryAction.INCIDENT_CREATED.value]

    # Idempotent: a second run adds nothing to this incident.
    history_service.backfill_initial_history(db_session)
    assert len(_history(client, user, legacy.id)) == 1


def test_backfill_attributes_unknown_reporter_to_system(client, db_session):
    """With no reporter on record, the opening event is SYSTEM, not a guess."""
    from app.models.incident import Incident, IncidentPriority, IncidentStatus

    _disable_all_existing_agents(db_session)
    admin = _create_user_in_db(db_session, "hist_orphan_a@test.com", role=UserRole.ADMIN)

    orphan = Incident(
        title="Legacy incident with no known reporter",
        category="OTHER",
        source="CAMERA",
        status=IncidentStatus.REPORTED,
        priority_level=IncidentPriority.P3,
        severity="MEDIUM",
        priority=3,
        latitude=12.9716,
        longitude=77.5946,
        location="SRID=4326;POINT(77.5946 12.9716)",
        reported_by=None,
    )
    db_session.add(orphan)
    db_session.flush()

    history_service.backfill_initial_history(db_session)

    events = _history(client, admin, orphan.id)
    assert len(events) == 1
    assert events[0]["actor_role"] == ActorRole.SYSTEM.value
    assert events[0]["actor_id"] is None
    assert events[0]["actor_name"] == "System"


# ---------------------------------------------------------------------------
# 16: the whole lifecycle
# ---------------------------------------------------------------------------

def test_full_lifecycle_generates_expected_history(client, db_session):
    """
    Walk an incident from report to closure and check the trail tells that story.

    REPORTED -> TRIAGED -> ASSIGNED -> IN_PROGRESS -> RESOLVED -> CLOSED,
    with each actor doing only what their role permits.
    """
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "life_u@test.com", name="Ratish Kapoor")
    admin = _create_user_in_db(
        db_session, "life_a@test.com", role=UserRole.ADMIN, name="City Admin"
    )

    # 1. Citizen reports it. No agent is available yet, so it stays REPORTED
    #    and every later step is an explicit act by a named person.
    incident = _create_incident(client, user, with_image=True)
    incident_id = incident["id"]
    assert incident["status"] == "REPORTED"

    agent = _create_user_in_db(
        db_session, "life_ag@test.com", role=UserRole.AGENT, name="Priya Sharma"
    )

    def patch_status(new_status, actor):
        resp = client.patch(
            f"/api/v1/incidents/{incident_id}/status",
            json={"status": new_status},
            headers=_auth(actor),
        )
        assert resp.status_code == 200, resp.text

    # 2. Admin triages.
    patch_status("TRIAGED", admin)

    # 3. Admin assigns the agent, which also moves TRIAGED -> ASSIGNED.
    resp = client.put(
        f"/api/v1/incidents/{incident_id}/assign",
        json={"agent_id": agent.id, "override_availability": False},
        headers=_auth(admin),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "ASSIGNED"

    # 4-5. The agent works it and resolves it.
    patch_status("IN_PROGRESS", agent)
    patch_status("RESOLVED", agent)

    # 6. Admin closes it.
    patch_status("CLOSED", admin)

    events = _history(client, admin, incident_id)
    actions = _actions(events)

    # The spine of the story, in order, ignoring the SLA and image events that
    # sit between these.
    expected_spine = [
        HistoryAction.INCIDENT_CREATED.value,
        HistoryAction.IMAGE_ADDED.value,
        HistoryAction.INCIDENT_TRIAGED.value,
        HistoryAction.AGENT_ASSIGNED.value,
        HistoryAction.STATUS_CHANGED.value,   # TRIAGED -> ASSIGNED
        HistoryAction.STATUS_CHANGED.value,   # ASSIGNED -> IN_PROGRESS
        HistoryAction.INCIDENT_RESOLVED.value,
        HistoryAction.INCIDENT_CLOSED.value,
    ]
    spine = [a for a in actions if a in set(expected_spine)]
    assert spine == expected_spine, f"unexpected lifecycle trail: {actions}"

    # Every status transition is present with both endpoints recorded.
    transitions = [
        (e["old_value"], e["new_value"])
        for e in events
        if e["action"] in {
            HistoryAction.STATUS_CHANGED.value,
            HistoryAction.INCIDENT_TRIAGED.value,
            HistoryAction.INCIDENT_RESOLVED.value,
            HistoryAction.INCIDENT_CLOSED.value,
        }
    ]
    assert transitions == [
        ("REPORTED", "TRIAGED"),
        ("TRIAGED", "ASSIGNED"),
        ("ASSIGNED", "IN_PROGRESS"),
        ("IN_PROGRESS", "RESOLVED"),
        ("RESOLVED", "CLOSED"),
    ]

    # Each event is attributed to whoever actually performed it.
    by_action = {e["action"]: e for e in events}
    assert by_action[HistoryAction.INCIDENT_CREATED.value]["actor_name"] == "Ratish Kapoor"
    assert by_action[HistoryAction.INCIDENT_TRIAGED.value]["actor_name"] == "City Admin"
    assert by_action[HistoryAction.INCIDENT_RESOLVED.value]["actor_name"] == "Priya Sharma"
    assert by_action[HistoryAction.INCIDENT_RESOLVED.value]["actor_role"] == ActorRole.AGENT.value
    assert by_action[HistoryAction.INCIDENT_CLOSED.value]["actor_role"] == ActorRole.ADMIN.value

    # The system's own work is attributed to SYSTEM, not to the citizen.
    sla_created = by_action[HistoryAction.SLA_CREATED.value]
    assert sla_created["actor_role"] == ActorRole.SYSTEM.value
    assert sla_created["actor_id"] is None

    assert [e["created_at"] for e in events] == sorted(e["created_at"] for e in events)


def test_reopening_a_resolved_incident_is_recorded(client, db_session):
    """Sending finished work back records INCIDENT_REOPENED, not a plain status change."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_reopen_u@test.com")
    admin = _create_user_in_db(db_session, "hist_reopen_a@test.com", role=UserRole.ADMIN)
    agent = _create_user_in_db(
        db_session, "hist_reopen_ag@test.com", role=UserRole.AGENT, name="Agent One"
    )

    incident = _create_incident(client, user)
    incident_id = incident["id"]
    assert incident["assigned_agent_id"] == agent.id

    for status_value, actor in (("IN_PROGRESS", agent), ("RESOLVED", agent)):
        resp = client.patch(
            f"/api/v1/incidents/{incident_id}/status",
            json={"status": status_value},
            headers=_auth(actor),
        )
        assert resp.status_code == 200, resp.text

    # Admin sends it back for more work.
    resp = client.patch(
        f"/api/v1/incidents/{incident_id}/status",
        json={"status": "IN_PROGRESS"},
        headers=_auth(admin),
    )
    assert resp.status_code == 200, resp.text

    events = _history(client, admin, incident_id)
    reopened = [e for e in events if e["action"] == HistoryAction.INCIDENT_REOPENED.value]
    assert len(reopened) == 1
    assert reopened[0]["old_value"] == "RESOLVED"
    assert reopened[0]["new_value"] == "IN_PROGRESS"
    assert reopened[0]["actor_role"] == ActorRole.ADMIN.value


def test_no_op_operations_record_nothing(client, db_session):
    """Saving a value that is already set is not an event."""
    _disable_all_existing_agents(db_session)
    user = _create_user_in_db(db_session, "hist_noop_u@test.com")
    admin = _create_user_in_db(db_session, "hist_noop_a@test.com", role=UserRole.ADMIN)

    incident = _create_incident(client, user)
    incident_id = incident["id"]

    agent = _create_user_in_db(
        db_session, "hist_noop_ag@test.com", role=UserRole.AGENT, name="Agent One"
    )

    client.put(
        f"/api/v1/incidents/{incident_id}/assign",
        json={"agent_id": agent.id, "override_availability": False},
        headers=_auth(admin),
    )
    baseline = _history(client, admin, incident_id)

    # Re-assign the same agent, re-save the same SLA, re-set the same priority.
    client.put(
        f"/api/v1/incidents/{incident_id}/assign",
        json={"agent_id": agent.id, "override_availability": False},
        headers=_auth(admin),
    )
    client.patch(
        f"/api/v1/incidents/{incident_id}/sla",
        json={"sla_hours": 72},
        headers=_auth(admin),
    )
    client.patch(
        f"/api/v1/incidents/{incident_id}/priority",
        json={"priority_level": "P3"},
        headers=_auth(admin),
    )

    assert _history(client, admin, incident_id) == baseline
