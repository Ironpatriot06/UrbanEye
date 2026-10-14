"""
Hotspot forecast: next week's report count per 1 km grid cell, by EWMA.

This is a STATISTIC, not a machine-learning model.  Phase 2B found the
gradient-boosted forecaster ~4% better than an EWMA inside Chicago but 37–52%
worse in a city it was not trained on, so no hotspot ML model is shipped.

Definition (matches the evaluated Phase 2B panel C as far as UrbanEye allows)
  input      reports of one category inside one service area (category, created_at, location)
  geography  1 km square cells, local equirectangular projection anchored at the
             area's south-west corner (lat_min, lon_min); x-scale at the area's mid latitude
  time       ISO weeks starting Monday 00:00 UTC.  The week containing the area's
             first report is partial and is dropped; the current week is incomplete
             and is never counted.  Weeks with no report in a cell count as 0.
  series     every cell with at least one report in the counted weeks.  (Phase 2B
             kept cells with >= 8 reports over 18 months; UrbanEye is too young
             for that filter.)
  forecast   s_0 = y_0 ; s_w = alpha * y_w + (1 - alpha) * s_(w-1) ; forecast for the
             current week = s_(last complete week), alpha = 0.3 from the bundle
  status     AVAILABLE with >= min_history_weeks (4) complete weeks, EXPERIMENTAL
             with 1–3, UNAVAILABLE with none.  Categories outside the four Phase 2B
             evaluated are EXPERIMENTAL (CATEGORY_NOT_EVALUATED).

Phase 2B comparison (Chicago, offline): EWMA MAE 1.453 vs rolling 4-week mean
1.506 (−3.5%).  It lags bursts and over-predicts quiet weeks.
"""

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from app.ml.areas import ServiceArea
from app.ml.features import IncidentRecord

CELL_M = 1000.0
EVALUATED_CATEGORIES = ("POTHOLE", "FLOOD", "GARBAGE", "STREETLIGHT")


def week_start(ts: datetime) -> datetime:
    ts = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    d = ts.date() - timedelta(days=ts.weekday())
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _cell(area: ServiceArea, lat: float, lon: float) -> tuple[int, int]:
    x = (lon - area.lon_min) * 111_320.0 * math.cos(math.radians(area.ref_lat))
    y = (lat - area.lat_min) * 110_574.0
    return int(math.floor(x / CELL_M)), int(math.floor(y / CELL_M))


def _cell_geometry(area: ServiceArea, ix: int, iy: int) -> dict:
    dlon = CELL_M / (111_320.0 * math.cos(math.radians(area.ref_lat)))
    dlat = CELL_M / 110_574.0
    w, s = area.lon_min + ix * dlon, area.lat_min + iy * dlat
    return {"center_lat": s + dlat / 2, "center_lon": w + dlon / 2,
            "bounds": {"south": s, "west": w, "north": s + dlat, "east": w + dlon}}


def ewma(series: list[float], alpha: float) -> float:
    s = float(series[0])
    for y in series[1:]:
        s = alpha * float(y) + (1.0 - alpha) * s
    return s


def forecast(records: Iterable[IncidentRecord], area: ServiceArea, category: str,
             now: datetime, alpha: float, min_weeks: int, limit: int = 50,
             area_first_report: Optional[datetime] = None) -> dict:
    """`records`: incidents of `area` (any category).  `area_first_report` defaults to theirs."""
    recs = [r for r in records if area.contains(r.latitude, r.longitude)]
    current = week_start(now)
    last_complete = current - timedelta(days=7)
    out = {"area": area.name, "category": category, "method": "EWMA (non-ML statistic)",
           "alpha": alpha, "min_history_weeks": min_weeks, "cell_size_m": CELL_M,
           "week_definition": "Monday 00:00 UTC", "status": "UNAVAILABLE", "reasons": [],
           "first_complete_week": None, "last_complete_week": last_complete.isoformat(),
           "forecast_week": current.isoformat(), "weeks_of_history": 0, "cells": [],
           "cells_total": 0, "generated_at": datetime.now(timezone.utc).isoformat()}
    first = area_first_report or (min(r.created_at for r in recs) if recs else None)
    if first is None:
        out["reasons"].append({"code": "NO_DATA", "message": "No reports in this area yet."})
        return out
    w0 = week_start(first) + timedelta(days=7)          # first week is partial
    n_weeks = (last_complete - w0).days // 7 + 1
    out["first_complete_week"] = w0.isoformat()
    if n_weeks <= 0:
        out["reasons"].append({"code": "NO_COMPLETE_WEEKS",
                               "message": "No complete week of reports exists yet."})
        return out
    out["weeks_of_history"] = n_weeks
    counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0] * n_weeks)
    for r in recs:
        if r.category != category:
            continue
        w = (week_start(r.created_at) - w0).days // 7
        if 0 <= w < n_weeks:
            counts[_cell(area, r.latitude, r.longitude)][w] += 1
    cells = []
    for (ix, iy), series in counts.items():
        cells.append({"cell_id": f"{ix}_{iy}", **_cell_geometry(area, ix, iy),
                      "predicted_next_week_count": ewma(series, alpha),
                      "last_week_count": series[-1],
                      "mean_last_4_weeks": sum(series[-4:]) / len(series[-4:]),
                      "reports_in_history": sum(series)})
    cells.sort(key=lambda c: (-c["predicted_next_week_count"], c["cell_id"]))
    out["cells_total"] = len(cells)
    out["cells"] = cells[:limit]
    status = "AVAILABLE"
    if n_weeks < min_weeks:
        status = "EXPERIMENTAL"
        out["reasons"].append({"code": "INSUFFICIENT_HISTORY_WEEKS",
                               "message": f"Only {n_weeks} complete week(s); the method was "
                                          f"evaluated with at least {min_weeks}."})
    if category not in EVALUATED_CATEGORIES:
        status = "EXPERIMENTAL"
        out["reasons"].append({"code": "CATEGORY_NOT_EVALUATED",
                               "message": "This category was not part of the offline evaluation."})
    out["status"] = status
    return out
