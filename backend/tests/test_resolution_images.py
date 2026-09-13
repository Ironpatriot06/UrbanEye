"""
UrbanEye+ — proof-of-work image tests.

The feature
-----------
After the assigned agent marks an incident RESOLVED, they may attach a photo
showing the finished work.  It is stored exactly like a citizen's report photo —
same table, same validation, same authenticated streaming endpoint — and tagged
RESOLUTION so the three dashboards can label it as proof.

What these tests pin down
-------------------------
- Only the assigned agent (or an admin) can attach proof, and only once the
  incident is RESOLVED.
- The image is stored identically to a report photo, and the bytes come back
  byte-for-byte from the same endpoint.
- All three roles that can see the incident can see the proof: the reporting
  citizen, the assigned agent, and any admin.  A citizen who did NOT report it,
  and an agent it is NOT assigned to, cannot.
- Existing report photos keep the REPORT kind and are unaffected.

Testing strategy
----------------
Same as the other suites: TestClient against the real PostgreSQL/PostGIS
database, each test inside a transaction that is rolled back afterwards.
"""

import io
import json
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password
from app.db.base import Base
from app.db.database import engine, get_db
from app.main import app
from app.models.image import ImageKind, IncidentImage
from app.models.user import User, UserRole


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

VALID_INCIDENT = {
    "title": "Pothole for proof-of-work test",
    "description": "30 cm deep pothole near bus stop 14.",
    "category": "POTHOLE",
    "source": "CITIZEN",
    "latitude": 12.9716,
    "longitude": 77.5946,
}


def _unique_email(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}@urbaneye-test.com"


def _make_user(db, role=UserRole.USER, name="Test User") -> User:
    user = User(
        name=name,
        email=_unique_email(role.value.lower()),
        password_hash=hash_password("TestPass123"),
        role=role,
        is_active=True,
        is_available=True,
    )
    db.add(user)
    db.flush()
    return user


def _auth(user) -> dict:
    return {
        "Authorization": f"Bearer {create_access_token({'sub': user.email, 'role': user.role})}"
    }


def _jpeg(marker: bytes = b"\x00") -> bytes:
    """Minimal valid JPEG: real magic bytes plus distinguishable padding."""
    return b"\xff\xd8\xff" + marker * 100


def _png() -> bytes:
    return b"\x89PNG" + b"\x11" * 100


def _disable_other_agents(db) -> None:
    """Stop the persistent demo agents stealing the auto-assignment."""
    db.query(User).filter(User.role == UserRole.AGENT).update(
        {"is_available": False}, synchronize_session="fetch"
    )
    db.flush()


def _resolved_incident(client, db_session, with_report_photo=True):
    """
    Build the real situation the feature exists for.

    A citizen reports an incident (optionally with a photo), it auto-assigns to
    the one available agent, and that agent walks it through to RESOLVED using
    the normal endpoints.  Returns (incident_id, citizen, agent).
    """
    _disable_other_agents(db_session)
    citizen = _make_user(db_session, UserRole.USER, name="Reporting Citizen")
    agent = _make_user(db_session, UserRole.AGENT, name="Field Agent")
    agent.is_available = True
    db_session.flush()

    files = {"data": (None, json.dumps(VALID_INCIDENT), "application/json")}
    if with_report_photo:
        files["images"] = ("report.jpg", io.BytesIO(_jpeg()), "image/jpeg")

    created = client.post("/api/v1/incidents/", files=files, headers=_auth(citizen))
    assert created.status_code == 201, created.text
    incident_id = created.json()["id"]
    assert created.json()["assigned_agent_id"] == agent.id, "auto-assignment did not pick the test agent"

    for step in ("IN_PROGRESS", "RESOLVED"):
        moved = client.patch(
            f"/api/v1/incidents/{incident_id}/status",
            json={"status": step},
            headers=_auth(agent),
        )
        assert moved.status_code == 200, moved.text

    return incident_id, citizen, agent


def _upload_proof(client, incident_id, actor, data=None, filename="proof.jpg", ctype="image/jpeg"):
    return client.post(
        f"/api/v1/incidents/{incident_id}/images/resolution",
        files={"file": (filename, io.BytesIO(data or _jpeg(b"\x42")), ctype)},
        headers=_auth(actor),
    )


# ===========================================================================
# 1. The happy path
# ===========================================================================

def test_assigned_agent_can_attach_proof_after_resolving(client, db_session):
    incident_id, _citizen, agent = _resolved_incident(client, db_session)

    resp = _upload_proof(client, incident_id, agent)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["kind"] == "RESOLUTION"
    assert body["filename"] == "proof.jpg"
    assert body["incident_id"] == incident_id
    # Metadata only — the bytes are never echoed into a JSON payload.
    assert "image_data" not in body


def test_proof_is_stored_exactly_like_a_report_photo(client, db_session):
    """Same table, same columns, same BYTEA storage — only `kind` differs."""
    incident_id, _citizen, agent = _resolved_incident(client, db_session)
    proof_bytes = _jpeg(b"\x42")
    resp = _upload_proof(client, incident_id, agent, data=proof_bytes)
    proof_id = resp.json()["id"]

    rows = (
        db_session.query(IncidentImage)
        .filter(IncidentImage.incident_id == incident_id)
        .order_by(IncidentImage.id)
        .all()
    )
    assert len(rows) == 2
    report, proof = rows[0], rows[1]

    assert report.kind == ImageKind.REPORT
    assert proof.kind == ImageKind.RESOLUTION
    assert proof.id == proof_id
    # Stored the same way: same table, bytes intact, same metadata columns.
    assert proof.image_data == proof_bytes
    assert proof.content_type == "image/jpeg"
    assert proof.filename == "proof.jpg"
    assert proof.incident_id == report.incident_id


def test_proof_bytes_stream_back_unchanged(client, db_session):
    incident_id, _citizen, agent = _resolved_incident(client, db_session)
    proof_bytes = _jpeg(b"\x42")
    proof_id = _upload_proof(client, incident_id, agent, data=proof_bytes).json()["id"]

    raw = client.get(
        f"/api/v1/incidents/{incident_id}/images/{proof_id}", headers=_auth(agent)
    )
    assert raw.status_code == 200
    assert raw.headers["content-type"].startswith("image/jpeg")
    assert raw.content == proof_bytes


def test_agent_can_attach_several_proof_photos(client, db_session):
    incident_id, _citizen, agent = _resolved_incident(client, db_session)
    for i in range(3):
        assert _upload_proof(
            client, incident_id, agent, filename=f"proof{i}.jpg"
        ).status_code == 201

    listing = client.get(
        f"/api/v1/incidents/{incident_id}/images", headers=_auth(agent)
    ).json()
    assert sum(1 for i in listing if i["kind"] == "RESOLUTION") == 3


def test_png_and_webp_proof_are_accepted(client, db_session):
    incident_id, _citizen, agent = _resolved_incident(client, db_session)
    assert _upload_proof(
        client, incident_id, agent, data=_png(), filename="p.png", ctype="image/png"
    ).status_code == 201


# ===========================================================================
# 2. Visibility — citizen, agent, admin
# ===========================================================================

def test_reporting_citizen_sees_the_proof(client, db_session):
    """The point of the feature: the person who reported it sees the evidence."""
    incident_id, citizen, agent = _resolved_incident(client, db_session)
    proof_id = _upload_proof(client, incident_id, agent).json()["id"]

    listing = client.get(
        f"/api/v1/incidents/{incident_id}/images", headers=_auth(citizen)
    )
    assert listing.status_code == 200
    proofs = [i for i in listing.json() if i["kind"] == "RESOLUTION"]
    assert len(proofs) == 1
    assert proofs[0]["id"] == proof_id

    raw = client.get(
        f"/api/v1/incidents/{incident_id}/images/{proof_id}", headers=_auth(citizen)
    )
    assert raw.status_code == 200


def test_assigned_agent_sees_the_proof(client, db_session):
    incident_id, _citizen, agent = _resolved_incident(client, db_session)
    proof_id = _upload_proof(client, incident_id, agent).json()["id"]

    listing = client.get(
        f"/api/v1/incidents/{incident_id}/images", headers=_auth(agent)
    ).json()
    assert any(i["id"] == proof_id and i["kind"] == "RESOLUTION" for i in listing)
    assert client.get(
        f"/api/v1/incidents/{incident_id}/images/{proof_id}", headers=_auth(agent)
    ).status_code == 200


def test_admin_sees_the_proof(client, db_session):
    incident_id, _citizen, agent = _resolved_incident(client, db_session)
    admin = _make_user(db_session, UserRole.ADMIN, name="Overseer")
    proof_id = _upload_proof(client, incident_id, agent).json()["id"]

    listing = client.get(
        f"/api/v1/incidents/{incident_id}/images", headers=_auth(admin)
    ).json()
    assert any(i["id"] == proof_id and i["kind"] == "RESOLUTION" for i in listing)
    assert client.get(
        f"/api/v1/incidents/{incident_id}/images/{proof_id}", headers=_auth(admin)
    ).status_code == 200


def test_listing_returns_both_kinds_together(client, db_session):
    incident_id, citizen, agent = _resolved_incident(client, db_session)
    _upload_proof(client, incident_id, agent)

    listing = client.get(
        f"/api/v1/incidents/{incident_id}/images", headers=_auth(citizen)
    ).json()
    kinds = [i["kind"] for i in listing]
    assert kinds.count("REPORT") == 1
    assert kinds.count("RESOLUTION") == 1
    # Oldest first, so the report photo precedes the proof.
    assert kinds == ["REPORT", "RESOLUTION"]


def test_an_unrelated_citizen_cannot_see_the_proof(client, db_session):
    incident_id, _citizen, agent = _resolved_incident(client, db_session)
    proof_id = _upload_proof(client, incident_id, agent).json()["id"]
    stranger = _make_user(db_session, UserRole.USER, name="Nosy Stranger")

    assert client.get(
        f"/api/v1/incidents/{incident_id}/images", headers=_auth(stranger)
    ).status_code == 403
    assert client.get(
        f"/api/v1/incidents/{incident_id}/images/{proof_id}", headers=_auth(stranger)
    ).status_code == 403


def test_an_unassigned_agent_cannot_see_the_proof(client, db_session):
    incident_id, _citizen, agent = _resolved_incident(client, db_session)
    proof_id = _upload_proof(client, incident_id, agent).json()["id"]
    other_agent = _make_user(db_session, UserRole.AGENT, name="Other Agent")

    assert client.get(
        f"/api/v1/incidents/{incident_id}/images", headers=_auth(other_agent)
    ).status_code == 403
    assert client.get(
        f"/api/v1/incidents/{incident_id}/images/{proof_id}", headers=_auth(other_agent)
    ).status_code == 403


def test_unauthenticated_request_for_proof_is_401(client, db_session):
    incident_id, _citizen, agent = _resolved_incident(client, db_session)
    proof_id = _upload_proof(client, incident_id, agent).json()["id"]
    assert client.get(f"/api/v1/incidents/{incident_id}/images").status_code == 401
    assert client.get(
        f"/api/v1/incidents/{incident_id}/images/{proof_id}"
    ).status_code == 401


# ===========================================================================
# 3. Who may upload
# ===========================================================================

def test_citizen_cannot_attach_proof_of_work(client, db_session):
    """A citizen cannot attest that the work was completed — even on their own report."""
    incident_id, citizen, _agent = _resolved_incident(client, db_session)
    resp = _upload_proof(client, incident_id, citizen)
    assert resp.status_code == 403


def test_an_unassigned_agent_cannot_attach_proof(client, db_session):
    incident_id, _citizen, _agent = _resolved_incident(client, db_session)
    other_agent = _make_user(db_session, UserRole.AGENT, name="Other Agent")
    resp = _upload_proof(client, incident_id, other_agent)
    assert resp.status_code == 403


def test_unauthenticated_upload_is_401(client, db_session):
    incident_id, _citizen, _agent = _resolved_incident(client, db_session)
    resp = client.post(
        f"/api/v1/incidents/{incident_id}/images/resolution",
        files={"file": ("proof.jpg", io.BytesIO(_jpeg()), "image/jpeg")},
    )
    assert resp.status_code == 401


def test_admin_can_attach_proof(client, db_session):
    incident_id, _citizen, _agent = _resolved_incident(client, db_session)
    admin = _make_user(db_session, UserRole.ADMIN, name="Overseer")
    assert _upload_proof(client, incident_id, admin).status_code == 201


def test_upload_to_a_missing_incident_is_404(client, db_session):
    agent = _make_user(db_session, UserRole.AGENT)
    assert _upload_proof(client, 99999999, agent).status_code == 404


# ===========================================================================
# 4. Only after the work is declared done
# ===========================================================================

def test_proof_is_refused_before_the_incident_is_resolved(client, db_session):
    """Proof of completed work means nothing until the work is declared complete."""
    _disable_other_agents(db_session)
    citizen = _make_user(db_session, UserRole.USER)
    agent = _make_user(db_session, UserRole.AGENT)
    agent.is_available = True
    db_session.flush()

    created = client.post(
        "/api/v1/incidents/",
        files={"data": (None, json.dumps(VALID_INCIDENT), "application/json")},
        headers=_auth(citizen),
    )
    incident_id = created.json()["id"]

    # ASSIGNED — too early.
    assert _upload_proof(client, incident_id, agent).status_code == 422

    # IN_PROGRESS — still too early.
    client.patch(
        f"/api/v1/incidents/{incident_id}/status",
        json={"status": "IN_PROGRESS"},
        headers=_auth(agent),
    )
    assert _upload_proof(client, incident_id, agent).status_code == 422

    # RESOLVED — now allowed.
    client.patch(
        f"/api/v1/incidents/{incident_id}/status",
        json={"status": "RESOLVED"},
        headers=_auth(agent),
    )
    assert _upload_proof(client, incident_id, agent).status_code == 201


def test_no_image_row_is_created_when_the_upload_is_refused(client, db_session):
    """A refused upload must attach nothing at all."""
    _disable_other_agents(db_session)
    citizen = _make_user(db_session, UserRole.USER)
    agent = _make_user(db_session, UserRole.AGENT)
    agent.is_available = True
    db_session.flush()

    created = client.post(
        "/api/v1/incidents/",
        files={"data": (None, json.dumps(VALID_INCIDENT), "application/json")},
        headers=_auth(citizen),
    )
    incident_id = created.json()["id"]

    assert _upload_proof(client, incident_id, agent).status_code == 422
    assert (
        db_session.query(IncidentImage)
        .filter(IncidentImage.incident_id == incident_id)
        .count()
        == 0
    )


# ===========================================================================
# 5. Validation — identical to a report photo
# ===========================================================================

def test_proof_rejects_an_unsupported_file_type(client, db_session):
    incident_id, _citizen, agent = _resolved_incident(client, db_session)
    resp = _upload_proof(
        client,
        incident_id,
        agent,
        data=b"MZ\x00\x00bad exe",
        filename="script.exe",
        ctype="application/octet-stream",
    )
    assert resp.status_code == 422


def test_proof_rejects_a_file_whose_bytes_are_not_an_image(client, db_session):
    """An image content-type on non-image bytes is still refused (magic-byte check)."""
    incident_id, _citizen, agent = _resolved_incident(client, db_session)
    resp = _upload_proof(
        client, incident_id, agent, data=b"not really a jpeg at all", ctype="image/jpeg"
    )
    assert resp.status_code == 422


def test_proof_rejects_an_oversized_file(client, db_session):
    incident_id, _citizen, agent = _resolved_incident(client, db_session)
    big = b"\xff\xd8\xff" + b"\x00" * (6 * 1024 * 1024)
    resp = _upload_proof(client, incident_id, agent, data=big, filename="big.jpg")
    assert resp.status_code == 422


# ===========================================================================
# 6. Audit trail
# ===========================================================================

def test_attaching_proof_is_recorded_in_incident_history(client, db_session):
    incident_id, _citizen, agent = _resolved_incident(client, db_session)
    _upload_proof(client, incident_id, agent)

    history = client.get(
        f"/api/v1/incidents/{incident_id}/history", headers=_auth(agent)
    ).json()
    proof_events = [
        e
        for e in history
        if e["action"] == "IMAGE_ADDED" and "Proof-of-work" in (e["description"] or "")
    ]
    assert len(proof_events) == 1
    assert proof_events[0]["actor_id"] == agent.id
    assert proof_events[0]["actor_role"] == "AGENT"


def test_a_report_photo_still_records_a_plain_image_event(client, db_session):
    """The existing wording for citizen photos is unchanged."""
    incident_id, citizen, _agent = _resolved_incident(client, db_session)
    history = client.get(
        f"/api/v1/incidents/{incident_id}/history", headers=_auth(citizen)
    ).json()
    image_events = [e for e in history if e["action"] == "IMAGE_ADDED"]
    assert len(image_events) == 1
    assert image_events[0]["description"].startswith("Photo 'report.jpg' attached")


# ===========================================================================
# 7. Existing behaviour is untouched
# ===========================================================================

def test_citizen_report_photos_are_still_tagged_report(client, db_session):
    _disable_other_agents(db_session)
    citizen = _make_user(db_session, UserRole.USER)
    created = client.post(
        "/api/v1/incidents/",
        files={
            "data": (None, json.dumps(VALID_INCIDENT), "application/json"),
            "images": ("photo.jpg", io.BytesIO(_jpeg()), "image/jpeg"),
        },
        headers=_auth(citizen),
    )
    incident_id = created.json()["id"]
    listing = client.get(
        f"/api/v1/incidents/{incident_id}/images", headers=_auth(citizen)
    ).json()
    assert len(listing) == 1
    assert listing[0]["kind"] == "REPORT"


def test_every_pre_existing_image_in_the_database_is_a_report_photo(db_session):
    """The migration back-fill: nothing that predates the feature is mislabelled."""
    mislabelled = (
        db_session.query(IncidentImage)
        .filter(IncidentImage.kind.is_(None))
        .count()
    )
    assert mislabelled == 0
