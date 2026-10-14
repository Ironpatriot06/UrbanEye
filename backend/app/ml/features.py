"""
Leak-free history features for one incident, ported from the ML pipeline.

Source of truth: UrbanEye_ML_Data_Pipeline/scripts/phase2b/features.py and
build_datasets.py, which produced the training features.  tests/test_ml_unit.py
replays a synthetic incident table through BOTH implementations (the pipeline's
output ships in the bundle as fixtures/feature_parity.json) and requires every
value to match.

THE ONE RULE.  A feature at the incident's report time t uses only
  * other incidents REPORTED strictly before t (minute resolution),
  * CLOSURES recorded strictly before t,
  * SLA outcomes at horizon X of incidents reported before t − X (their
    deadline has passed, so the outcome is known whatever happens later).
The incident's own status, closure and duration are never read.

UrbanEye mapping of training concepts
  city            the configured service area containing the incident (areas.py)
  model_category  the UrbanEye category (FIRE_HAZARD / OTHER incidents are history only)
  closed_at       the FIRST INCIDENT_RESOLVED or INCIDENT_CLOSED history event; a later
                  reopen is not undone (training data recorded a single closure)
  valid duration  1 minute <= closed_at − created_at <= 87,600 h; UrbanEye has no
                  duplicate flag, so no incident is excluded as a duplicate
  calendar        Asia/Kolkata local time (training used each US city's local time)
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import numpy as np

from app.ml.areas import ServiceArea

EPOCH = datetime(2009, 1, 1, tzinfo=timezone.utc)
DAY = 1440.0
MIN_EVENTS_FOR_STAT = 10       # recent duration statistics
MIN_EVENTS_FOR_RATE = 20       # recent breach rate
BACKLOG_CAP_DAYS = 90.0
MIN_RES_H, MAX_RES_H = 1.0 / 60.0, 87_600.0
CELL_M = 250.0
LOCAL_TZ = ZoneInfo("Asia/Kolkata")
NIGHT_START, NIGHT_END = 22, 6

WORKLOAD = ["wl_backlog_open", "wl_intake_7d", "wl_intake_30d", "wl_closed_7d",
            "wl_backlog_days", "wl_recent_closures_30d", "wl_recent_mean_log_res",
            "wl_recent_median_res_h", "wl_city_backlog_open", "wl_city_intake_7d"]
SPATIAL = ["sp_samecat_box750_7d", "sp_samecat_box750_30d", "sp_samecat_box750_90d",
           "sp_samecat_box750_open_now", "sp_all_box750_30d"]
HISTORY_FEATURES = WORKLOAD + SPATIAL
NAN = float("nan")


@dataclass(frozen=True)
class IncidentRecord:
    id: int
    category: str
    created_at: datetime
    latitude: float
    longitude: float
    first_closed_at: Optional[datetime]


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def to_minutes(ts: datetime) -> float:
    return (_utc(ts) - EPOCH).total_seconds() / 60.0


def calendar_features(created_at: datetime) -> dict:
    loc = _utc(created_at).astimezone(LOCAL_TZ)
    h = loc.hour
    return {"hour": h, "day_of_week": loc.weekday(), "month": loc.month,
            "is_weekend": loc.weekday() >= 5, "is_night": h >= NIGHT_START or h < NIGHT_END}


class _Table:
    """Column arrays for the incidents of one service area."""

    def __init__(self, records: list[IncidentRecord], area: ServiceArea):
        self.ids = np.array([r.id for r in records], dtype=np.int64)
        self.cat = np.array([r.category for r in records], dtype=object)
        self.rep = np.array([to_minutes(r.created_at) for r in records], dtype=np.float64)
        self.clo = np.array([NAN if r.first_closed_at is None else to_minutes(r.first_closed_at)
                             for r in records], dtype=np.float64)
        bad = ~np.isnan(self.clo) & (self.clo < self.rep)      # same fix as build_datasets
        self.clo[bad] = self.rep[bad]
        self.has_close = ~np.isnan(self.clo)
        res = (self.clo - self.rep) / 60.0
        self.valid = self.has_close & (res >= MIN_RES_H) & (res <= MAX_RES_H)
        self.log_res = np.where(self.valid, np.log1p(np.where(self.valid, res, 0.0)), NAN)
        self.dur_h = np.where(self.has_close, res, np.inf)
        self.exit = np.fmin(np.where(self.has_close, self.clo, np.inf),
                            self.rep + BACKLOG_CAP_DAYS * DAY)
        self.rep_f = np.floor(self.rep)
        self.exit_f = np.floor(self.exit)
        self.clo_f = np.floor(self.clo)
        self.warm = float(self.rep.min()) if len(records) else NAN
        lat = np.array([r.latitude for r in records], dtype=np.float64)
        lon = np.array([r.longitude for r in records], dtype=np.float64)
        x = (lon - area.lon_min) * 111_320.0 * math.cos(math.radians(area.ref_lat))
        y = (lat - area.lat_min) * 110_574.0
        self.ix = np.floor(x / CELL_M) + 2
        self.iy = np.floor(y / CELL_M) + 2


def _between(values_f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """values (already floored) in [floor(lo), floor(hi))."""
    return (values_f >= math.floor(lo)) & (values_f < math.floor(hi))


def _workload(T: _Table, m: np.ndarray, t: float) -> dict:
    tf = math.floor(t)
    backlog = float(np.sum(m & (T.rep_f < tf)) - np.sum(m & (T.exit_f < tf)))
    intake7 = float(np.sum(m & _between(T.rep_f, t - 7 * DAY, t)))
    intake30 = float(np.sum(m & _between(T.rep_f, t - 30 * DAY, t)))
    closed7 = float(np.sum(m & T.has_close & _between(T.clo_f, t - 7 * DAY, t)))
    win = m & T.valid & _between(T.clo_f, t - 30 * DAY, t)
    cnt = int(win.sum())
    mean_log = float(T.log_res[win].mean()) if cnt >= MIN_EVENTS_FOR_STAT else NAN
    dq = math.floor(t / DAY) * DAY             # snapshot at the UTC midnight starting t's day
    wd = m & T.valid & _between(T.clo_f, dq - 30 * DAY, dq)
    med = float(np.expm1(np.median(T.log_res[wd]))) if wd.sum() >= MIN_EVENTS_FOR_STAT else NAN
    rate = intake30 / 30.0
    return {"wl_backlog_open": backlog, "wl_intake_7d": intake7, "wl_intake_30d": intake30,
            "wl_closed_7d": closed7,
            "wl_backlog_days": backlog / rate if rate > 0 else NAN,
            "wl_recent_closures_30d": float(cnt), "wl_recent_mean_log_res": mean_log,
            "wl_recent_median_res_h": med}


def breach_rate(T: _Table, m: np.ndarray, t: float, horizon_h: float,
                min_events: int = MIN_EVENTS_FOR_RATE) -> tuple[float, int]:
    """
    Share of same-group incidents reported in [t − X − 30d, t − X) that were not
    resolved within X hours.  Never-closed incidents breach every horizon; a
    closure at or after t cannot be known at t, but such an incident was already
    open at its deadline (reported before t − X), so it is a breach either way.
    Returns (rate or NaN, n).
    """
    xm = horizon_h * 60.0
    elig = m & (~T.has_close | T.valid)
    win = elig & _between(T.rep_f, t - xm - 30 * DAY, t - xm)
    n = int(win.sum())
    if n < min_events:
        return NAN, n
    breach = ~T.has_close[win] | (T.dur_h[win] >= horizon_h)
    return float(breach.mean()), n


def _spatial(T: _Table, same: np.ndarray, area_all: np.ndarray, q: int, t: float) -> dict:
    tf = math.floor(t)
    block = (np.abs(T.ix - T.ix[q]) <= 1) & (np.abs(T.iy - T.iy[q]) <= 1)
    sb, ab = same & block, area_all & block
    out = {f"sp_samecat_box750_{w}d": float(np.sum(sb & _between(T.rep_f, t - w * DAY, t)))
           for w in (7, 30, 90)}
    out["sp_samecat_box750_open_now"] = float(np.sum(sb & (T.rep_f < tf)) - np.sum(sb & (T.exit_f < tf)))
    out["sp_all_box750_30d"] = float(np.sum(ab & _between(T.rep_f, t - 30 * DAY, t)))
    return out


def history_features(query: IncidentRecord, area_records: Iterable[IncidentRecord],
                     area: Optional[ServiceArea], breach_horizons: Iterable[float] = ()) -> dict:
    """
    All history features for `query` at its created_at.  `area_records` is every
    incident of the query's service area (the query itself included; anything
    not observable at t is ignored here).  Outside a service area every history
    feature is missing.

    Returns {feature: value, ..., "hist_breach_rate": {X: (rate, n)}}; missing = NaN.
    """
    horizons = list(breach_horizons)
    if area is None:
        out = {f: NAN for f in HISTORY_FEATURES}
        out["hist_breach_rate"] = {h: (NAN, 0) for h in horizons}
        return out
    recs = list(area_records)
    if not any(r.id == query.id for r in recs):
        recs.append(query)
    T = _Table(recs, area)
    q = int(np.flatnonzero(T.ids == query.id)[0])
    t = T.rep[q]
    same = T.cat == query.category
    everything = np.ones(len(recs), dtype=bool)

    out = _workload(T, same, t)
    city = _workload(T, everything, t)
    out["wl_city_backlog_open"] = city["wl_backlog_open"]
    out["wl_city_intake_7d"] = city["wl_intake_7d"]
    if t - T.warm < 30 * DAY:                     # warm-up: counts would be truncated
        out = {k: NAN for k in out}
    sp = _spatial(T, same, everything, q, t)
    if t - T.warm < 90 * DAY:
        sp["sp_samecat_box750_90d"] = NAN
    if t - T.warm < 30 * DAY:
        for k in ("sp_samecat_box750_7d", "sp_samecat_box750_30d",
                  "sp_samecat_box750_open_now", "sp_all_box750_30d"):
            sp[k] = NAN
    out.update(sp)
    out["hist_breach_rate"] = {h: breach_rate(T, same, t, h) for h in horizons}
    return out


def local_closed_durations(query: IncidentRecord, records: Iterable[IncidentRecord],
                           category: Optional[str]) -> np.ndarray:
    """Valid durations (hours) of incidents closed before the query's report time."""
    t = to_minutes(query.created_at)
    out = []
    for r in records:
        if r.id == query.id or r.first_closed_at is None:
            continue
        if category is not None and r.category != category:
            continue
        rep, clo = to_minutes(r.created_at), to_minutes(r.first_closed_at)
        clo = max(clo, rep)
        if math.floor(clo) >= math.floor(t):
            continue
        h = (clo - rep) / 60.0
        if MIN_RES_H <= h <= MAX_RES_H:
            out.append(h)
    return np.array(out, dtype=np.float64)


def local_breach_outcomes(query: IncidentRecord, records: Iterable[IncidentRecord],
                          category: Optional[str], horizon_h: float,
                          window_days: Optional[float] = None) -> np.ndarray:
    """
    Breach outcomes at horizon X of incidents whose deadline passed before t.
    With `window_days`, only those reported in [t − X − window, t − X) — the
    Phase 2B "recent violation rate" baseline; otherwise any age.
    """
    t = to_minutes(query.created_at)
    out = []
    for r in records:
        if r.id == query.id or (category is not None and r.category != category):
            continue
        rep = to_minutes(r.created_at)
        if math.floor(rep) >= math.floor(t - horizon_h * 60.0):
            continue
        if window_days is not None and \
                math.floor(rep) < math.floor(t - horizon_h * 60.0 - window_days * DAY):
            continue
        if r.first_closed_at is None:
            out.append(True)
            continue
        clo = max(to_minutes(r.first_closed_at), rep)
        h = (clo - rep) / 60.0
        if h < MIN_RES_H:
            continue                                   # not a valid resolution
        out.append(h >= horizon_h)                     # known at t: rep + X < t
    return np.array(out, dtype=bool)
