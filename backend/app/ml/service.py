"""
Database access, orchestration and logging for the ML layer.

Reads only: incidents (category, created_at, latitude, longitude) and the first
INCIDENT_RESOLVED / INCIDENT_CLOSED event of each incident in incident_history.
Writes only: ml_predictions (best effort — a logging failure is reported in the
response as `logged: false` and never fails the request).  Incidents, their
priority, SLA and status are never modified here.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.ml.areas import ServiceArea, area_by_name, area_for, get_areas
from app.ml.features import HISTORY_FEATURES, IncidentRecord, calendar_features, history_features
from app.ml.hotspot import forecast
from app.ml.predictors import (UNAVAILABLE, Context, empty_result, predict_resolution,
                                predict_sla)
from app.ml.registry import get_registry
from app.models.history import HistoryAction, IncidentHistory
from app.models.incident import Incident
from app.models.ml_prediction import MLPrediction

log = logging.getLogger(__name__)
CLOSURE_ACTIONS = (HistoryAction.INCIDENT_RESOLVED, HistoryAction.INCIDENT_CLOSED)


def _enum_value(v: Any) -> Any:
    return getattr(v, "value", v)


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def json_safe(obj: Any) -> Any:
    """NaN/inf -> None, recursively (JSON and JSONB reject them)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item"):                      # numpy scalars
        return json_safe(obj.item())
    return obj


def fetch_records(db: Session, upto: Optional[datetime] = None,
                  area: Optional[ServiceArea] = None) -> list[IncidentRecord]:
    """Incidents (optionally inside `area`, created at or before `upto`) with their first closure."""
    closure = (select(IncidentHistory.incident_id,
                      func.min(IncidentHistory.created_at).label("first_closed_at"))
               .where(IncidentHistory.action.in_(CLOSURE_ACTIONS))
               .group_by(IncidentHistory.incident_id)
               .subquery())
    q = (db.query(Incident.id, Incident.category, Incident.created_at, Incident.latitude,
                  Incident.longitude, closure.c.first_closed_at)
         .outerjoin(closure, closure.c.incident_id == Incident.id))
    if upto is not None:
        q = q.filter(Incident.created_at <= upto)
    if area is not None:
        q = q.filter(Incident.latitude.between(area.lat_min, area.lat_max),
                     Incident.longitude.between(area.lon_min, area.lon_max))
    return [IncidentRecord(id=r[0], category=_enum_value(r[1]), created_at=_utc(r[2]),
                           latitude=float(r[3]), longitude=float(r[4]),
                           first_closed_at=None if r[5] is None else _utc(r[5]))
            for r in q.all()]


def _inputs(ctx: Context) -> dict:
    """The model inputs as served (calendar + history + breach rate at this SLA)."""
    d = {"category": ctx.category, **calendar_features(ctx.query.created_at)}
    d.update({k: ctx.features.get(k) for k in HISTORY_FEATURES})
    if ctx.sla_hours is not None:
        d["sla_hours"] = ctx.sla_hours
        d["hist_breach_rate_own"] = ctx.features.get("hist_breach_rate", {}).get(
            float(ctx.sla_hours), (None, 0))[0]
    return json_safe(d)


def _internal_error(task: str, bundle_version: Optional[str]) -> dict:
    out = empty_result(task, bundle_version)
    out["reasons"] = [{"code": "INTERNAL_ERROR",
                       "message": "The prediction could not be computed."}]
    return out


def _log(db: Session, incident: Incident, results: list[dict], inputs: dict,
         as_of: datetime) -> bool:
    """Insert the results; a failure rolls back only this savepoint, never the caller's work."""
    try:
        with db.begin_nested():
            for r in results:
                sla_hours = r.get("sla_hours")
                key = "|".join(str(x) for x in (r["task"], r["bundle_version"], r["method"],
                                                sla_hours, r["status"]))
                db.execute(
                    pg_insert(MLPrediction)
                    .values(incident_id=incident.id, task=r["task"], status=r["status"],
                            method=r["method"], model_version=r["model_version"],
                            bundle_version=r["bundle_version"],
                            sla_hours=None if sla_hours is None else int(sla_hours),
                            fallback_reason=r["fallback_reason"],
                            reasons=json_safe(r["reasons"]), output=json_safe(r),
                            features=inputs if r["method"] == "ML_MODEL" else None,
                            features_as_of=as_of, dedupe_key=key)
                    .on_conflict_do_nothing(constraint="uq_ml_predictions_incident_key"))
    except Exception:  # noqa: BLE001 — logging must never break a read or a write
        log.exception("could not log ML predictions for incident %s", incident.id)
        return False
    try:
        db.commit()
        return True
    except Exception:  # noqa: BLE001
        log.exception("could not commit ML prediction log for incident %s", incident.id)
        db.rollback()
        return False


def predict_incident(db: Session, incident: Incident, include_inputs: bool = False) -> dict:
    reg = get_registry()
    created = _utc(incident.created_at)
    category = _enum_value(incident.category)
    sla_hours = None if incident.sla_hours is None else float(incident.sla_hours)
    area = area_for(float(incident.latitude), float(incident.longitude))
    query = IncidentRecord(id=incident.id, category=category, created_at=created,
                           latitude=float(incident.latitude), longitude=float(incident.longitude),
                           first_closed_at=None)
    inputs: dict = {}
    try:
        if reg.enabled:
            with db.begin_nested():       # a failed read cannot poison the caller's session
                records = fetch_records(db, upto=created, area=area)
            feats = history_features(query, records if area else [], area,
                                     [sla_hours] if sla_hours else [])
        else:
            records, feats = [], {}
        ctx = Context(query=query, category=category, sla_hours=sla_hours, area=area,
                      features=feats, baseline_records=records)
        results = [predict_resolution(reg, ctx), predict_sla(reg, ctx)]
        inputs = _inputs(ctx) if reg.enabled else {}
    except Exception:  # noqa: BLE001 — a prediction must never break the incident view
        log.exception("ML prediction failed for incident %s", incident.id)
        results = [_internal_error("resolution", reg.bundle_version),
                   _internal_error("sla", reg.bundle_version)]
    results = [json_safe(r) for r in results]
    logged = _log(db, incident, results, inputs, created)
    return {"incident_id": incident.id, "generated_at": datetime.now(timezone.utc),
            "features_as_of": created, "incident_status": _enum_value(incident.status),
            "sla_deadline": incident.sla_deadline, "service_area": area.name if area else None,
            "bundle_version": reg.bundle_version, "resolution": results[0], "sla": results[1],
            "inputs": inputs if include_inputs else None, "logged": logged}


def record_predictions(db: Session, incident_id: int) -> bool:
    """
    Compute and log the predictions for one incident.  Best effort: returns
    False on any failure and never raises, so it can follow an incident write
    (creation, SLA or priority change) without being able to undo or block it.
    Call it after the write is committed; every ML read and insert runs in its
    own SAVEPOINT, so a failure rolls back ML work only.  Idempotent: the log's
    (incident_id, dedupe_key) constraint drops repeats.
    """
    from app.core.config import get_settings

    if not get_settings().ML_LOG_ON_WRITE:
        return False
    try:
        with db.begin_nested():
            incident = db.query(Incident).filter(Incident.id == incident_id).first()
        if incident is None:
            return False
        return bool(predict_incident(db, incident)["logged"])
    except Exception:  # noqa: BLE001
        log.exception("could not record ML predictions for incident %s", incident_id)
        return False


def backfill_predictions(db: Session, limit: Optional[int] = None) -> dict:
    """Log predictions for every incident (idempotent).  Used by scripts/ml_log_predictions.py."""
    ids = [i for (i,) in db.query(Incident.id).order_by(Incident.id).all()]
    if limit is not None:
        ids = ids[:limit]
    before = db.query(func.count(MLPrediction.id)).scalar()
    failed = [i for i in ids if not record_predictions(db, i)]
    after = db.query(func.count(MLPrediction.id)).scalar()
    return {"incidents": len(ids), "rows_added": int(after - before), "failed": failed}


def hotspots(db: Session, area_name: Optional[str], category: str, limit: int) -> dict:
    reg = get_registry()
    areas = get_areas()
    area = area_by_name(area_name, areas) if area_name else (areas[0] if areas else None)
    names = [a.name for a in areas]
    base = {"area": area_name, "category": category, "method": "EWMA (non-ML statistic)",
            "status": UNAVAILABLE, "reasons": [], "cells": [], "cells_total": 0,
            "available_areas": names, "generated_at": datetime.now(timezone.utc).isoformat()}
    if area is None:
        base["reasons"].append({"code": "UNKNOWN_AREA", "message": "No such service area.",
                                "detail": {"areas": [a.name for a in areas]}})
        return base
    if not reg.enabled:
        base["reasons"].append({"code": "ML_DISABLED",
                                "message": "Predictions are switched off on this server."})
        return base
    if not reg.ready("hotspot"):
        base["reasons"].append({"code": "ARTIFACT_UNAVAILABLE",
                                "message": "The hotspot settings are missing or failed verification."})
        return base
    meta = reg.hotspot_meta
    out = forecast(fetch_records(db, area=area), area, category, datetime.now(timezone.utc),
                   alpha=float(meta["ewma_alpha"]), min_weeks=int(meta["min_history_weeks"]),
                   limit=limit)
    out["available_areas"] = names
    out["model_version"] = meta.get("model_version")
    out["bundle_version"] = reg.bundle_version
    out["baseline_comparison"] = ("Phase 2B, Chicago 1 km grid (offline): EWMA MAE 1.453 vs "
                                  "rolling 4-week mean 1.506. Not evaluated on UrbanEye data.")
    return json_safe(out)
