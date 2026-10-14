"""
Phase 2C ML endpoints — API tests against the real PostgreSQL/PostGIS database,
inside a rolled-back transaction (same strategy as test_incidents.py).

Covers authentication and RBAC, response schemas, SLA override semantics,
prediction logging, failure behaviour with missing artifacts, and that the
ML layer never changes an incident or blocks incident workflows.
"""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password
from app.db.base import Base
from app.db.database import engine, get_db
from app.main import app
from app.ml import registry as R
from app.models.user import UserRole

INCIDENT = {"title": "ML test pothole", "description": "for prediction tests",
            "category": "POTHOLE", "source": "CITIZEN", "latitude": 12.9716, "longitude": 77.5946}


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
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    R.set_registry(None)


def _user(db, email, role=UserRole.USER, available=False):
    from app.models.user import User

    u = User(name=email.split("@")[0], email=email, password_hash=hash_password("TestPass123"),
             role=role, is_available=available)
    db.add(u)
    db.flush()
    return u


def _auth(u) -> dict:
    return {"Authorization": f"Bearer {create_access_token({'sub': u.email, 'role': u.role})}"}


def _no_available_agents(db):
    from app.models.user import User

    db.query(User).filter(User.role == UserRole.AGENT).update(
        {"is_available": False}, synchronize_session="fetch")
    db.flush()


def _create(client, user, **extra) -> dict:
    r = client.post("/api/v1/incidents/", headers=_auth(user),
                    files={"data": (None, json.dumps({**INCIDENT, **extra}), "application/json")})
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture()
def people(db_session):
    _no_available_agents(db_session)
    return {"citizen": _user(db_session, "ml_citizen@test.com"),
            "other": _user(db_session, "ml_other@test.com"),
            "agent": _user(db_session, "ml_agent@test.com", UserRole.AGENT),
            "agent2": _user(db_session, "ml_agent2@test.com", UserRole.AGENT),
            "admin": _user(db_session, "ml_admin@test.com", UserRole.ADMIN)}


def _predict(client, user, iid, **params):
    return client.get(f"/api/v1/incidents/{iid}/predictions", headers=_auth(user), params=params)


# ---------------------------------------------------------------------------
# authentication / RBAC
# ---------------------------------------------------------------------------
def test_predictions_require_authentication(client, people):
    inc = _create(client, people["citizen"])
    assert client.get(f"/api/v1/incidents/{inc['id']}/predictions").status_code == 401


def test_predictions_follow_incident_access_rules(client, people):
    inc = _create(client, people["citizen"])
    iid = inc["id"]
    assert _predict(client, people["citizen"], iid).status_code == 200
    assert _predict(client, people["other"], iid).status_code == 404       # not theirs
    assert _predict(client, people["agent"], iid).status_code == 403       # not assigned
    assert _predict(client, people["admin"], iid).status_code == 200
    r = client.put(f"/api/v1/incidents/{iid}/assign", headers=_auth(people["admin"]),
                   json={"agent_id": people["agent"].id, "override_availability": True})
    assert r.status_code == 200, r.text
    assert _predict(client, people["agent"], iid).status_code == 200
    assert _predict(client, people["agent2"], iid).status_code == 403
    assert _predict(client, people["admin"], 999_999_999).status_code == 404


def test_model_inputs_are_admin_only(client, people):
    iid = _create(client, people["citizen"])["id"]
    assert _predict(client, people["citizen"], iid, include_inputs=True).json()["inputs"] is None
    body = _predict(client, people["admin"], iid, include_inputs=True).json()
    assert body["inputs"]["category"] == "POTHOLE" and "wl_backlog_open" in body["inputs"]


def test_ml_status_and_hotspots_rbac(client, people):
    assert client.get("/api/v1/ml/status", headers=_auth(people["citizen"])).status_code == 403
    assert client.get("/api/v1/ml/status", headers=_auth(people["agent"])).status_code == 403
    st = client.get("/api/v1/ml/status", headers=_auth(people["admin"])).json()
    assert st["components"]["sla"]["ready"] and st["bundle_version"] == "urbaneye-ml-2026.10.13"
    assert [a["name"] for a in st["service_areas"]] == ["Bengaluru", "Chennai"]
    q = {"category": "POTHOLE", "area": "Bengaluru"}
    assert client.get("/api/v1/hotspots", params=q, headers=_auth(people["citizen"])).status_code == 403
    for who in ("agent", "admin"):
        r = client.get("/api/v1/hotspots", params=q, headers=_auth(people[who]))
        assert r.status_code == 200, r.text
        assert r.json()["method"].startswith("EWMA")
        assert r.json()["available_areas"] == ["Bengaluru", "Chennai"]
    assert client.get("/api/v1/hotspots", params={"category": "NOPE"},
                      headers=_auth(people["admin"])).status_code == 422
    r = client.get("/api/v1/hotspots", params={"category": "POTHOLE", "area": "Atlantis"},
                   headers=_auth(people["admin"])).json()
    assert r["status"] == "UNAVAILABLE" and r["reasons"][0]["code"] == "UNKNOWN_AREA"


# ---------------------------------------------------------------------------
# schema, statuses, logging
# ---------------------------------------------------------------------------
def test_prediction_schema_and_honesty_fields(client, people):
    inc = _create(client, people["citizen"])
    body = _predict(client, people["citizen"], inc["id"]).json()
    assert body["features_as_of"].startswith(inc["created_at"][:19])
    assert body["service_area"] == "Bengaluru"
    for task in ("resolution", "sla"):
        p = body[task]
        assert p["task"] == task
        assert p["status"] in ("AVAILABLE", "EXPERIMENTAL", "FALLBACK", "UNAVAILABLE")
        assert p["local_reliability_established"] is False
        if p["method"] == "ML_MODEL":
            assert p["model_version"] and p["warnings"]
    s = body["sla"]
    assert s["sla_hours"] == 72                              # POTHOLE -> P3 -> 72 h
    if s["method"] == "ML_MODEL":
        assert 0 <= s["breach_probability"] <= 1 and s["risk_band"] in ("LOW", "MEDIUM", "HIGH")
    r = body["resolution"]
    if r["method"] == "ML_MODEL":
        assert r["range_calibrated"] is False and r["range_p10_hours"] <= r["range_p90_hours"]


def test_predictions_are_logged_once(client, people, db_session):
    from app.models.ml_prediction import MLPrediction

    iid = _create(client, people["citizen"])["id"]
    first = _predict(client, people["citizen"], iid).json()
    assert first["logged"] is True
    _predict(client, people["admin"], iid)
    rows = db_session.query(MLPrediction).filter(MLPrediction.incident_id == iid).all()
    assert sorted(r.task for r in rows) == ["resolution", "sla"]
    for row in rows:
        assert row.status == first[row.task]["status"] and row.method == first[row.task]["method"]
        assert row.bundle_version == "urbaneye-ml-2026.10.13"


def test_predictions_do_not_change_the_incident(client, people):
    iid = _create(client, people["citizen"])["id"]
    before = client.get(f"/api/v1/incidents/{iid}", headers=_auth(people["admin"])).json()
    _predict(client, people["admin"], iid)
    after = client.get(f"/api/v1/incidents/{iid}", headers=_auth(people["admin"])).json()
    for k in ("priority_level", "severity", "sla_hours", "sla_deadline", "status",
              "assigned_agent_id"):
        assert before[k] == after[k], k


# ---------------------------------------------------------------------------
# SLA override semantics (admin PATCH /sla is the only way sla_hours changes)
# ---------------------------------------------------------------------------
def _set_sla(client, admin, iid, hours):
    r = client.patch(f"/api/v1/incidents/{iid}/sla", headers=_auth(admin), json={"sla_hours": hours})
    assert r.status_code == 200, r.text


def test_sla_override_is_respected(client, people):
    admin = people["admin"]
    iid = _create(client, people["citizen"])["id"]
    _set_sla(client, admin, iid, 12)
    s = _predict(client, admin, iid).json()["sla"]
    assert s["sla_hours"] == 12
    if s["method"] == "ML_MODEL":
        assert any(r["code"] == "FEATURES_OUTSIDE_TRAINING_RANGE" for r in s["reasons"])
    _set_sla(client, admin, iid, 200)
    s = _predict(client, admin, iid).json()["sla"]
    assert s["sla_hours"] == 200 and s["fallback_reason"] == "SLA_OUT_OF_RANGE"
    assert s["breach_probability"] is None and s["risk_band"] is None
    _set_sla(client, admin, iid, None)
    s = _predict(client, admin, iid).json()["sla"]
    assert s["status"] == "UNAVAILABLE" and s["reasons"][0]["code"] == "NO_SLA"


def test_unsupported_category(client, people):
    iid = _create(client, people["citizen"], category="FIRE_HAZARD")["id"]
    body = _predict(client, people["admin"], iid).json()
    for task in ("resolution", "sla"):
        assert body[task]["method"] != "ML_MODEL"
        assert body[task]["fallback_reason"] == "UNSUPPORTED_CATEGORY"


# ---------------------------------------------------------------------------
# failure behaviour: missing artifacts never break incident workflows
# ---------------------------------------------------------------------------
def test_missing_artifacts_degrade_gracefully(client, people, tmp_path):
    R.set_registry(R.build_registry(True, str(tmp_path / "missing"), "0" * 64))
    citizen, admin, agent = people["citizen"], people["admin"], people["agent"]
    inc = _create(client, citizen)                           # creation still works
    iid = inc["id"]
    r = _predict(client, citizen, iid)
    assert r.status_code == 200
    for task in ("resolution", "sla"):
        p = r.json()[task]
        assert p["method"] != "ML_MODEL" and p["fallback_reason"] == "ARTIFACT_UNAVAILABLE"
        assert p["status"] in ("FALLBACK", "UNAVAILABLE")
    st = client.get("/api/v1/ml/status", headers=_auth(admin)).json()
    assert st["error"] == "BUNDLE_NOT_FOUND" and not st["components"]["resolution"]["ready"]
    # the normal workflow is unaffected
    assert client.put(f"/api/v1/incidents/{iid}/assign", headers=_auth(admin),
                      json={"agent_id": agent.id, "override_availability": True}).status_code == 200
    assert client.patch(f"/api/v1/incidents/{iid}/status", headers=_auth(agent),
                        json={"status": "IN_PROGRESS"}).status_code == 200
    assert client.get(f"/api/v1/incidents/{iid}/history", headers=_auth(admin)).status_code == 200


def test_ml_disabled(client, people):
    R.set_registry(R.build_registry(False, "", ""))
    iid = _create(client, people["citizen"])["id"]
    body = _predict(client, people["citizen"], iid).json()
    assert body["resolution"]["status"] == body["sla"]["status"] == "UNAVAILABLE"
    assert body["resolution"]["fallback_reason"] == "ML_DISABLED"


# ---------------------------------------------------------------------------
# automatic prediction logging (shadow mode)
# ---------------------------------------------------------------------------
def _rows(db, iid):
    from app.models.ml_prediction import MLPrediction

    return db.query(MLPrediction).filter(MLPrediction.incident_id == iid).all()


def test_creation_logs_predictions_without_viewing(client, people, db_session):
    iid = _create(client, people["citizen"])["id"]
    rows = _rows(db_session, iid)
    assert sorted(r.task for r in rows) == ["resolution", "sla"]
    assert all(r.bundle_version == "urbaneye-ml-2026.10.13" and r.status for r in rows)
    sla = next(r for r in rows if r.task == "sla")
    assert sla.sla_hours == 72 and sla.output["task"] == "sla"
    # a later view of the same prediction is not logged twice
    _predict(client, people["citizen"], iid)
    assert len(_rows(db_session, iid)) == 2


def test_logging_failure_never_blocks_incident_writes(client, people, db_session, monkeypatch):
    from app.ml import service as ml_service

    def boom(*a, **k):
        raise RuntimeError("simulated prediction failure")

    monkeypatch.setattr(ml_service, "predict_incident", boom)
    inc = _create(client, people["citizen"])                 # still 201
    assert _rows(db_session, inc["id"]) == []
    r = client.patch(f"/api/v1/incidents/{inc['id']}/sla", headers=_auth(people["admin"]),
                     json={"sla_hours": 12})
    assert r.status_code == 200 and r.json()["sla_hours"] == 12
    r = client.patch(f"/api/v1/incidents/{inc['id']}/priority", headers=_auth(people["admin"]),
                     json={"priority_level": "P1"})
    assert r.status_code == 200 and r.json()["priority_level"] == "P1"


def test_sla_override_adds_a_prediction_and_keeps_the_old_one(client, people, db_session):
    iid = _create(client, people["citizen"])["id"]
    _set_sla(client, people["admin"], iid, 12)
    r = client.patch(f"/api/v1/incidents/{iid}/priority", headers=_auth(people["admin"]),
                     json={"priority_level": "P2"})                # resets SLA to 24 h
    assert r.status_code == 200 and r.json()["sla_hours"] == 24
    sla_rows = sorted(r.sla_hours for r in _rows(db_session, iid) if r.task == "sla")
    assert sla_rows == [12, 24, 72]
    assert len([r for r in _rows(db_session, iid) if r.task == "resolution"]) == 1


def test_log_on_write_can_be_switched_off(client, people, db_session, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "ML_LOG_ON_WRITE", False)
    iid = _create(client, people["citizen"])["id"]
    assert _rows(db_session, iid) == []


def test_backfill_is_idempotent(client, people, db_session):
    from app.ml.service import backfill_predictions

    _create(client, people["citizen"])
    first = backfill_predictions(db_session)
    assert first["failed"] == []
    second = backfill_predictions(db_session)
    assert second["rows_added"] == 0 and second["failed"] == []


def test_shadow_evaluation_tracks_observed_and_censored_outcomes(client, people):
    admin, agent = people["admin"], people["agent"]
    iid = _create(client, people["citizen"])["id"]
    open_id = _create(client, people["citizen"])["id"]
    client.put(f"/api/v1/incidents/{iid}/assign", headers=_auth(admin),
               json={"agent_id": agent.id, "override_availability": True})
    for s in ("IN_PROGRESS", "RESOLVED"):
        assert client.patch(f"/api/v1/incidents/{iid}/status", headers=_auth(agent),
                            json={"status": s}).status_code == 200
    assert client.get("/api/v1/ml/shadow-evaluation", headers=_auth(agent)).status_code == 403
    assert client.get("/api/v1/ml/shadow-evaluation",
                      headers=_auth(people["citizen"])).status_code == 403
    ev = client.get("/api/v1/ml/shadow-evaluation", headers=_auth(admin)).json()
    res = [g for g in ev["groups"] if g["task"] == "resolution"]
    assert sum(g["outcomes"].get("OBSERVED", 0) for g in res) >= 1
    assert sum(g["outcomes"].get("CENSORED", 0) for g in res) >= 1
    assert "not model validation" in ev["notes"][0]
    assert "incident_id" not in json.dumps(ev)                 # aggregates only
    assert open_id != iid


def test_database_error_inside_prediction_is_contained(client, people, db_session, monkeypatch):
    """A failing SQL statement in the ML read must not abort the caller's transaction."""
    from sqlalchemy import text

    from app.ml import service as ml_service

    def bad_fetch(db, *a, **k):
        db.execute(text("SELECT * FROM table_that_does_not_exist"))

    monkeypatch.setattr(ml_service, "fetch_records", bad_fetch)
    inc = _create(client, people["citizen"])                          # still 201
    body = _predict(client, people["admin"], inc["id"]).json()
    assert body["resolution"]["status"] == body["sla"]["status"] == "UNAVAILABLE"
    assert body["sla"]["reasons"][0]["code"] == "INTERNAL_ERROR"
    r = client.get(f"/api/v1/incidents/{inc['id']}", headers=_auth(people["citizen"]))
    assert r.status_code == 200 and r.json()["title"] == INCIDENT["title"]
