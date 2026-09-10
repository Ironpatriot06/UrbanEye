"""
UrbanEye+ test suite — Phase 1.

Testing strategy
----------------
All tests use FastAPI's TestClient (which wraps httpx) and run against the
REAL development PostgreSQL/PostGIS database.  We chose this approach because:

  1. PostGIS is not available as an in-process SQLite substitute.
  2. A separate test database (URBANEYE_TEST_DATABASE_URL) would require
     additional Docker configuration that does not yet exist.

Isolation mechanism
-------------------
Each test that writes data is wrapped in a session-level transaction that is
rolled back after the test completes.  This prevents test data from
accumulating in the development database while still exercising the real
PostGIS engine.

The mechanism works by:
  1. Starting a raw DBAPI connection and opening a transaction.
  2. Binding a SQLAlchemy Session to that connection.
  3. Overriding the FastAPI `get_db` dependency to use that bound session.
  4. Rolling back the DBAPI transaction in the test's teardown fixture.

Because the outer transaction is never committed, all INSERT / UPDATE
statements are invisible to other connections and are discarded on rollback.

Limitation: tests that explicitly call `db.commit()` inside service functions
will commit *to the savepoint* created by the nested transaction, not to the
real database.  The outer rollback still discards everything.  This is the
standard "session-per-test with rollback" pattern.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.database import engine, SessionLocal
from app.db.base import Base
from app.main import app
from app.db.database import get_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def create_tables():
    """Ensure all tables exist before any test runs."""
    Base.metadata.create_all(bind=engine)
    yield
    # Tables are left in place; we do not drop them to preserve dev data.


@pytest.fixture()
def db_session():
    """
    Provide a database session that is always rolled back after the test.

    Uses SQLAlchemy's nested transaction / SAVEPOINT strategy so that
    service-layer commits are absorbed without touching the real database.
    """
    connection = engine.connect()
    transaction = connection.begin()

    session = Session(bind=connection)
    # Begin a nested (SAVEPOINT) transaction so that service-layer commits
    # do not escape to the outer transaction.
    session.begin_nested()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def client(db_session):
    """
    Return a TestClient whose `get_db` dependency is overridden to use the
    rollback-wrapped session from `db_session`.
    """

    def override_get_db():
        try:
            yield db_session
        finally:
            pass  # Rollback is handled by the db_session fixture.

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# Convenience payload for a valid incident.
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


# ---------------------------------------------------------------------------
# Health endpoint tests  (no DB writes, no rollback fixture needed)
# ---------------------------------------------------------------------------

@pytest.fixture()
def plain_client():
    """TestClient without dependency overrides — for health checks."""
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(plain_client):
    """GET /health must return 200 and status 'ok'."""
    resp = plain_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "service" in body


def test_health_db_returns_ok(plain_client):
    """GET /health/db must successfully query PostgreSQL."""
    resp = plain_client.get("/health/db")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "database" in body


# ---------------------------------------------------------------------------
# Incident creation tests
# ---------------------------------------------------------------------------

def test_create_incident_success(client):
    """POST /api/v1/incidents with valid data must return 201 and the created incident."""
    resp = client.post("/api/v1/incidents/", json=VALID_INCIDENT)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["id"] is not None
    assert body["title"] == VALID_INCIDENT["title"]
    assert body["category"] == "POTHOLE"
    assert body["status"] == "REPORTED"
    assert body["latitude"] == pytest.approx(12.9716, abs=1e-4)
    assert body["longitude"] == pytest.approx(77.5946, abs=1e-4)


def test_create_incident_missing_title(client):
    """POST without a title must return 422 Unprocessable Entity."""
    payload = {k: v for k, v in VALID_INCIDENT.items() if k != "title"}
    resp = client.post("/api/v1/incidents/", json=payload)
    assert resp.status_code == 422


def test_create_incident_invalid_latitude(client):
    """Latitude outside [-90, 90] must be rejected with 422."""
    bad = {**VALID_INCIDENT, "latitude": 91.0}
    resp = client.post("/api/v1/incidents/", json=bad)
    assert resp.status_code == 422


def test_create_incident_invalid_longitude(client):
    """Longitude outside [-180, 180] must be rejected with 422."""
    bad = {**VALID_INCIDENT, "longitude": -181.0}
    resp = client.post("/api/v1/incidents/", json=bad)
    assert resp.status_code == 422


def test_create_incident_invalid_category(client):
    """An unrecognised category string must be rejected with 422."""
    bad = {**VALID_INCIDENT, "category": "ALIEN_INVASION"}
    resp = client.post("/api/v1/incidents/", json=bad)
    assert resp.status_code == 422


def test_create_incident_invalid_status_in_body(client):
    """IncidentCreate has no status field — supplying one should be ignored (200-range)."""
    payload = {**VALID_INCIDENT, "status": "CLOSED"}
    resp = client.post("/api/v1/incidents/", json=payload)
    # Extra fields are ignored by default; creation still succeeds.
    assert resp.status_code == 201
    # The status must always start as REPORTED regardless.
    assert resp.json()["status"] == "REPORTED"


# ---------------------------------------------------------------------------
# Incident retrieval tests
# ---------------------------------------------------------------------------

def test_get_incident_by_id(client):
    """GET /api/v1/incidents/{id} must return the previously created incident."""
    create_resp = client.post("/api/v1/incidents/", json=VALID_INCIDENT)
    assert create_resp.status_code == 201
    incident_id = create_resp.json()["id"]

    get_resp = client.get(f"/api/v1/incidents/{incident_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == incident_id


def test_get_incident_not_found(client):
    """GET /api/v1/incidents/999999 for a non-existent ID must return 404."""
    resp = client.get("/api/v1/incidents/999999")
    assert resp.status_code == 404


def test_list_incidents(client):
    """GET /api/v1/incidents must return a list (possibly empty)."""
    resp = client.get("/api/v1/incidents/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_list_incidents_after_create(client):
    """After creating an incident, it must appear in the list."""
    client.post("/api/v1/incidents/", json=VALID_INCIDENT)
    resp = client.get("/api/v1/incidents/")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


# ---------------------------------------------------------------------------
# Status update tests
# ---------------------------------------------------------------------------

def test_update_incident_status(client):
    """PATCH /api/v1/incidents/{id}/status must update the status field."""
    create_resp = client.post("/api/v1/incidents/", json=VALID_INCIDENT)
    assert create_resp.status_code == 201
    incident_id = create_resp.json()["id"]

    patch_resp = client.patch(
        f"/api/v1/incidents/{incident_id}/status",
        json={"status": "IN_PROGRESS"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "IN_PROGRESS"


def test_update_status_invalid_value(client):
    """PATCH with an unrecognised status string must return 422."""
    create_resp = client.post("/api/v1/incidents/", json=VALID_INCIDENT)
    incident_id = create_resp.json()["id"]

    patch_resp = client.patch(
        f"/api/v1/incidents/{incident_id}/status",
        json={"status": "DELETED"},
    )
    assert patch_resp.status_code == 422


def test_update_status_not_found(client):
    """PATCH for a non-existent incident must return 404."""
    resp = client.patch(
        "/api/v1/incidents/999999/status",
        json={"status": "RESOLVED"},
    )
    assert resp.status_code == 404
