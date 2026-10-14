"""
Shadow-mode evaluation: logged predictions joined to the outcomes UrbanEye
has observed since.  This is the only place a local accuracy figure can come
from, and it is NOT a validation until enough incidents have closed.

Outcome definitions (same closure rule as the features: the first
INCIDENT_RESOLVED or INCIDENT_CLOSED history event)

  resolution
    OBSERVED  closed: observed_hours = first closure − created_at
    CENSORED  still open: only a lower bound is known, elapsed_hours = now − created_at.
              Censored rows are never scored as if they had closed.  The one
              thing they do reveal is a certain UNDER-estimate when elapsed
              already exceeds the estimate; that count is reported separately.
  sla (scored against the SLA window the prediction was made for, which an
       admin may since have changed)
    BREACHED  closed at/after created_at + sla_hours, or still open past it
    MET       closed before the deadline
    PENDING   still open, deadline not reached — no outcome yet

Model rows (ML_MODEL) and baseline rows (FALLBACK) are summarised separately;
UNAVAILABLE rows only counted.
"""

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.history import HistoryAction, IncidentHistory
from app.models.incident import Incident
from app.models.ml_prediction import MLPrediction

MIN_OBSERVED_FOR_METRICS = 30


def _utc(ts: Optional[datetime]) -> Optional[datetime]:
    if ts is None:
        return None
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def outcome(task: str, created_at: datetime, first_closed_at: Optional[datetime],
            sla_hours: Optional[float], now: datetime) -> dict:
    """The observable outcome of one prediction at time `now` (pure function)."""
    created_at, first_closed_at, now = _utc(created_at), _utc(first_closed_at), _utc(now)
    if task == "resolution":
        if first_closed_at is not None:
            return {"state": "OBSERVED",
                    "observed_hours": max((first_closed_at - created_at).total_seconds(), 0) / 3600}
        return {"state": "CENSORED", "elapsed_hours": max((now - created_at).total_seconds(), 0) / 3600}
    if sla_hours is None:
        return {"state": "NO_SLA"}
    deadline = created_at + timedelta(hours=float(sla_hours))
    if first_closed_at is not None:
        return {"state": "BREACHED" if first_closed_at >= deadline else "MET"}
    return {"state": "BREACHED" if now >= deadline else "PENDING"}


def summarise(rows: Iterable[dict], now: datetime) -> dict:
    """
    rows: {task, status, method, output, sla_hours, created_at, first_closed_at}.
    Aggregates only — no incident identifiers or locations leave this function.
    """
    groups: dict = defaultdict(lambda: {"n": 0, "states": defaultdict(int), "abs_err": [],
                                        "under_certain": 0, "sq_err": []})
    for r in rows:
        key = (r["task"], r["method"] or "NONE", r["status"])
        g = groups[key]
        g["n"] += 1
        o = outcome(r["task"], r["created_at"], r["first_closed_at"], r.get("sla_hours"), now)
        g["states"][o["state"]] += 1
        out = r["output"] or {}
        if r["task"] == "resolution" and out.get("estimate_hours") is not None:
            est = float(out["estimate_hours"])
            if o["state"] == "OBSERVED":
                g["abs_err"].append(abs(o["observed_hours"] - est))
            elif o["state"] == "CENSORED" and o["elapsed_hours"] > est:
                g["under_certain"] += 1
        if r["task"] == "sla" and o["state"] in ("BREACHED", "MET"):
            p = out.get("breach_probability")
            if p is None:
                p = out.get("historical_breach_rate")
            if p is not None:
                g["sq_err"].append((float(p) - (1.0 if o["state"] == "BREACHED" else 0.0)) ** 2)

    out = []
    for (task, method, status), g in sorted(groups.items()):
        item = {"task": task, "method": method, "status": status, "predictions": g["n"],
                "outcomes": dict(g["states"])}
        if task == "resolution":
            n = len(g["abs_err"])
            item["observed_n"] = n
            item["mae_hours_observed_only"] = (sum(g["abs_err"]) / n) if n else None
            item["censored_already_underestimated"] = g["under_certain"]
        else:
            n = len(g["sq_err"])
            item["resolved_n"] = n
            item["brier_resolved_only"] = (sum(g["sq_err"]) / n) if n else None
        item["enough_for_evaluation"] = n >= MIN_OBSERVED_FOR_METRICS
        out.append(item)
    return {
        "generated_at": _utc(now).isoformat(),
        "min_observed_for_metrics": MIN_OBSERVED_FOR_METRICS,
        "groups": out,
        "notes": [
            "Shadow-mode monitoring, not model validation.",
            "Resolution MAE uses closed incidents only; long-running open incidents are "
            "right-censored, so this MAE is biased towards fast closures until they close.",
            "SLA scores use only incidents whose breach outcome is known.",
            "No metric is reliable below min_observed_for_metrics outcomes per group.",
        ],
    }


def collect(db: Session) -> list[dict]:
    closure = (select(IncidentHistory.incident_id,
                      func.min(IncidentHistory.created_at).label("first_closed_at"))
               .where(IncidentHistory.action.in_((HistoryAction.INCIDENT_RESOLVED,
                                                  HistoryAction.INCIDENT_CLOSED)))
               .group_by(IncidentHistory.incident_id).subquery())
    q = (db.query(MLPrediction.task, MLPrediction.status, MLPrediction.method,
                  MLPrediction.output, MLPrediction.sla_hours, Incident.created_at,
                  closure.c.first_closed_at)
         .join(Incident, Incident.id == MLPrediction.incident_id)
         .outerjoin(closure, closure.c.incident_id == MLPrediction.incident_id))
    return [{"task": r[0], "status": r[1], "method": r[2], "output": r[3], "sla_hours": r[4],
             "created_at": r[5], "first_closed_at": r[6]} for r in q.all()]


def shadow_evaluation(db: Session, now: Optional[datetime] = None) -> dict:
    return summarise(collect(db), now or datetime.now(timezone.utc))
