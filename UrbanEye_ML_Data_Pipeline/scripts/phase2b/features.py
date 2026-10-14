"""
Phase 2B — leak-free historical, workload and spatial features.

THE ONE RULE
------------
A feature for an incident reported at time t may only use facts that were
observable at t:

  * another incident's REPORT, if it was reported strictly before t;
  * another incident's CLOSURE (and therefore its duration), if it closed
    strictly before t;
  * another incident's SLA OUTCOME at horizon X, if it was reported before
    t − X — by then its deadline has passed, so whether it beat the deadline is
    already known, whatever happens to it later.

Nothing here reads the query incident's own closure, status or duration.
tests/test_phase2b.py checks each function against a brute-force reference and
checks that perturbing future events never changes a feature.

All counting is done with a sorted (group, time) key and two `searchsorted`
calls per window, so "how many events in [t − w, t)" is exact and O(log n).

UrbanEye reproducibility: every feature is defined on columns UrbanEye already
stores (category, created_at, location) plus resolution/closure times, which
UrbanEye records as `incident_history` rows (INCIDENT_RESOLVED /
INCIDENT_CLOSED). The PostGIS equivalents are in PHASE_2B_REPORT.md §13.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN = 60  # seconds
EPOCH = pd.Timestamp("2009-01-01", tz="UTC")
TBITS = 26            # minutes since EPOCH fit in 26 bits until ~2136
MIN_EVENTS_FOR_STAT = 10


def to_minutes(ts: pd.Series) -> np.ndarray:
    """UTC timestamps -> float minutes since EPOCH (NaN for NaT)."""
    ts = pd.to_datetime(ts, utc=True)
    out = (ts - EPOCH).dt.total_seconds() / MIN
    return out.to_numpy(dtype=float)


def _key(group_code: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    return (group_code.astype(np.int64) << TBITS) + np.floor(minutes).astype(np.int64)


def _count_before(sorted_keys: np.ndarray, g: np.ndarray, t: np.ndarray) -> np.ndarray:
    """# events in group g with time-minute strictly before minute floor(t)."""
    return np.searchsorted(sorted_keys, _key(g, t), side="left")


def _window_count(sorted_keys, g, t, width_min) -> np.ndarray:
    """# events in group g with time in [t − width, t)."""
    return _count_before(sorted_keys, g, t) - _count_before(sorted_keys, g, t - width_min)


def encode_groups(*cols: pd.Series) -> np.ndarray:
    """Dense int code for a tuple of columns (NaN-safe)."""
    frame = pd.concat([c.astype("string").fillna("<NA>").reset_index(drop=True) for c in cols],
                      axis=1)
    return frame.groupby(list(frame.columns), sort=True).ngroup().to_numpy()


# ---------------------------------------------------------------------------
# workload / recent-resolution features (per group, e.g. city x category)
# ---------------------------------------------------------------------------
def workload_features(group: np.ndarray, reported_min: np.ndarray, closed_min: np.ndarray,
                      res_hours_valid: np.ndarray, query_idx: np.ndarray | None = None,
                      warmup_start_min: np.ndarray | None = None,
                      warmup_min: float = 30 * 24 * 60,
                      backlog_cap_days: float = 90.0) -> pd.DataFrame:
    """
    For every row (or the rows in `query_idx`), at its own report time t:

      wl_backlog_open        reports before t, not closed before t, and less
                             than `backlog_cap_days` old at t
      wl_intake_7d / _30d    reports in [t−7d, t) / [t−30d, t)
      wl_closed_7d           closures in [t−7d, t)
      wl_backlog_days        backlog / (intake_30d / 30): days of intake queued
      wl_recent_closures_30d closures with a valid duration in [t−30d, t)
      wl_recent_median_res_h median duration of those closures
      wl_recent_mean_log_res mean log1p(duration) of those closures

    The age cap matters: ~2–5% of Chicago reports are still "open" in the 2026
    extract. Uncapped, they accumulate for ever and the backlog becomes a
    monotone proxy for calendar time. Each report therefore leaves the backlog
    at min(closure, report + cap), which is still known at t.

    `res_hours_valid` is NaN for closures that are not real resolutions
    (duplicates, instant closures); those still leave the backlog but do not
    enter the duration statistics. `warmup_start_min` (per row) blanks every
    feature for rows less than `warmup_min` after their source's first record,
    because the counts would be truncated there.
    """
    n = len(group)
    q = np.arange(n) if query_idx is None else np.asarray(query_idx)
    gq, tq = group[q], reported_min[q]

    day = 24 * 60.0
    rk = np.sort(_key(group, reported_min))
    has_close = ~np.isnan(closed_min)
    exit_min = np.fmin(np.where(has_close, closed_min, np.inf), reported_min + backlog_cap_days * day)
    ek = np.sort(_key(group, exit_min))
    ck = np.sort(_key(group[has_close], closed_min[has_close]))

    out = pd.DataFrame(index=q)
    out["wl_backlog_open"] = (_count_before(rk, gq, tq) - _count_before(ek, gq, tq)).astype(float)
    out["wl_intake_7d"] = _window_count(rk, gq, tq, 7 * day).astype(float)
    out["wl_intake_30d"] = _window_count(rk, gq, tq, 30 * day).astype(float)
    out["wl_closed_7d"] = _window_count(ck, gq, tq, 7 * day).astype(float)
    rate = out["wl_intake_30d"] / 30.0
    out["wl_backlog_days"] = np.where(rate > 0, out["wl_backlog_open"] / rate.where(rate > 0, 1), np.nan)

    # recent duration statistics from closures in [t − 30d, t)
    v = ~np.isnan(res_hours_valid) & has_close
    order = np.lexsort((closed_min[v], group[v]))
    vg, vc = group[v][order], closed_min[v][order]
    vlog = np.log1p(res_hours_valid[v][order])
    vk = _key(vg, vc)
    hi = np.searchsorted(vk, _key(gq, tq), side="left")
    lo = np.searchsorted(vk, _key(gq, tq - 30 * day), side="left")
    cnt = hi - lo
    csum = np.concatenate([[0.0], np.cumsum(vlog)])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_log = (csum[hi] - csum[lo]) / cnt
    out["wl_recent_closures_30d"] = cnt.astype(float)
    out["wl_recent_mean_log_res"] = np.where(cnt >= MIN_EVENTS_FOR_STAT, mean_log, np.nan)
    # The median is snapshotted once per (group, UTC day): closures in
    # [D − 30d, D) where D is the midnight starting the query's day. That is
    # strictly earlier than t, so still leak-free, and it bounds the number of
    # distinct medians to groups x days.
    dq = np.floor(tq / day) * day
    hi_d = np.searchsorted(vk, _key(gq, dq), side="left")
    lo_d = np.searchsorted(vk, _key(gq, dq - 30 * day), side="left")
    out["wl_recent_median_res_h"] = _window_medians(vlog, lo_d, hi_d)

    if warmup_start_min is not None:
        young = (tq - warmup_start_min[q]) < warmup_min
        out.loc[young, :] = np.nan
    return out


def _window_medians(vals: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Median of vals[lo:hi] per query, expm1'd back to hours; NaN if too few."""
    pairs = pd.DataFrame({"lo": lo, "hi": hi})
    uniq = pairs.drop_duplicates().reset_index(drop=True)
    meds = np.full(len(uniq), np.nan)
    for i, (a, b) in enumerate(zip(uniq["lo"].to_numpy(), uniq["hi"].to_numpy())):
        if b - a >= MIN_EVENTS_FOR_STAT:
            meds[i] = np.expm1(np.median(vals[a:b]))
    uniq["med"] = meds
    return pairs.merge(uniq, on=["lo", "hi"], how="left")["med"].to_numpy(float)


# ---------------------------------------------------------------------------
# recent SLA outcome rates (per group, per horizon)
# ---------------------------------------------------------------------------
def recent_breach_rate(group: np.ndarray, reported_min: np.ndarray, duration_h: np.ndarray,
                       is_open: np.ndarray, horizon_h: float, eligible: np.ndarray,
                       query_idx: np.ndarray | None = None, window_days: float = 30.0,
                       min_events: int = 20) -> np.ndarray:
    """
    Share of same-group incidents reported in [t − X − w, t − X) that were NOT
    resolved within X hours (X = horizon). All of them had their deadline
    expire before t, so the outcome is known at t.

    `eligible` restricts the reference population (e.g. non-duplicates);
    `is_open` marks incidents never closed (they breach every horizon);
    `duration_h` is the observed closure duration for the rest.
    """
    n = len(group)
    q = np.arange(n) if query_idx is None else np.asarray(query_idx)
    e = np.asarray(eligible, bool)
    breach = (is_open | (duration_h >= horizon_h))[e].astype(float)
    order = np.lexsort((reported_min[e], group[e]))
    k = _key(group[e][order], reported_min[e][order])
    b = np.concatenate([[0.0], np.cumsum(breach[order])])
    xm = horizon_h * 60.0
    tq, gq = reported_min[q], group[q]
    hi = np.searchsorted(k, _key(gq, tq - xm), side="left")
    lo = np.searchsorted(k, _key(gq, tq - xm - window_days * 1440.0), side="left")
    cnt = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = (b[hi] - b[lo]) / cnt
    return np.where(cnt >= min_events, rate, np.nan)


# ---------------------------------------------------------------------------
# spatial grid features
# ---------------------------------------------------------------------------
GRID_BITS = 11  # 2048 cells per axis


def grid_cells(lat: np.ndarray, lon: np.ndarray, origin_lat: np.ndarray, origin_lon: np.ndarray,
               ref_lat: np.ndarray, cell_m: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Local equirectangular projection onto a square grid anchored at a fixed
    per-city origin. Distortion is < 0.1% over a city-sized extent. In
    PostGIS this is ST_SnapToGrid(ST_Transform(location, <local UTM>), cell_m)
    with the same origin.
    """
    x = (lon - origin_lon) * 111_320.0 * np.cos(np.radians(ref_lat))
    y = (lat - origin_lat) * 110_574.0
    ix = np.floor(x / cell_m) + 2
    iy = np.floor(y / cell_m) + 2
    return ix, iy


def spatial_counts(cell_group: np.ndarray, ix: np.ndarray, iy: np.ndarray,
                   reported_min: np.ndarray, windows_days: list[float],
                   closed_min: np.ndarray | None = None, radius_cells: int = 1,
                   prefix: str = "sp") -> pd.DataFrame:
    """
    Counts of events reported in [t − w, t) within the (2r+1)x(2r+1) block of
    cells around each incident's cell, among rows sharing `cell_group` (e.g.
    city, or city x category). With `closed_min`, also the number of those
    block events still open at t. Rows without coordinates get NaN.
    """
    ok = ~(np.isnan(ix) | np.isnan(iy))
    n = len(ix)
    lim = (1 << GRID_BITS) - 1
    ixc = np.clip(np.nan_to_num(ix, nan=0), 0, lim).astype(np.int64)
    iyc = np.clip(np.nan_to_num(iy, nan=0), 0, lim).astype(np.int64)
    cid = (cell_group.astype(np.int64) << (2 * GRID_BITS)) + (ixc << GRID_BITS) + iyc
    rk = np.sort(_key(cid[ok], reported_min[ok]))
    if closed_min is not None:
        hc = ok & ~np.isnan(closed_min)
        ck = np.sort(_key(cid[hc], closed_min[hc]))

    acc = {w: np.zeros(n) for w in windows_days}
    open_now = np.zeros(n)
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            nb = (cell_group.astype(np.int64) << (2 * GRID_BITS)) + \
                 (np.clip(ixc + dx, 0, lim) << GRID_BITS) + np.clip(iyc + dy, 0, lim)
            before = _count_before(rk, nb, reported_min)
            for w in windows_days:
                acc[w] += before - _count_before(rk, nb, reported_min - w * 1440.0)
            if closed_min is not None:
                open_now += before - _count_before(ck, nb, reported_min)
    side = 2 * radius_cells + 1
    out = pd.DataFrame(index=np.arange(n))
    for w in windows_days:
        out[f"{prefix}_{int(w)}d"] = np.where(ok, acc[w], np.nan)
    if closed_min is not None:
        out[f"{prefix}_open_now"] = np.where(ok, open_now, np.nan)
    out.attrs["block_side_cells"] = side
    return out
