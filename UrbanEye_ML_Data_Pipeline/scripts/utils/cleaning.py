"""
Cleaning primitives.

Design rule that runs through this whole module: nothing is ever silently
discarded. Every rejection is counted, and (when quarantine is enabled) the
rejected rows are written to data/processed/quarantine/ with a reason column.
A cleaning step that cannot explain what it removed is not a cleaning step.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import load_config, p, ensure_dir

_WS = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")


@dataclass
class CleaningStats:
    """Accumulates one counter per rule so the quality report can render them."""
    dataset: str
    rows_in: int = 0
    rows_out: int = 0
    removed: dict[str, int] = field(default_factory=dict)
    flagged: dict[str, int] = field(default_factory=dict)
    nulled: dict[str, int] = field(default_factory=dict)

    def remove(self, rule: str, n: int) -> None:
        if n:
            self.removed[rule] = self.removed.get(rule, 0) + int(n)

    def flag(self, rule: str, n: int) -> None:
        if n:
            self.flagged[rule] = self.flagged.get(rule, 0) + int(n)

    def null(self, rule: str, n: int) -> None:
        if n:
            self.nulled[rule] = self.nulled.get(rule, 0) + int(n)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_removed_total": sum(self.removed.values()),
            "removed_by_rule": self.removed,
            "flagged_by_rule": self.flagged,
            "nulled_by_rule": self.nulled,
        }


def quarantine(df: pd.DataFrame, dataset: str, reason: str) -> None:
    """Persist rejected rows so a human can audit what the pipeline threw away."""
    cfg = load_config()["cleaning"].get("quarantine", {})
    if not cfg.get("enabled") or df.empty:
        return
    from .paths import repo_root
    out = repo_root() / cfg.get("dir", "data/processed/quarantine") / dataset
    out.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["_quarantine_reason"] = reason
    fn = out / f"{re.sub(r'[^a-z0-9_]+', '_', reason.lower())}.parquet"
    if fn.exists():
        try:
            df = pd.concat([pd.read_parquet(fn), df], ignore_index=True)
        except Exception:
            pass
    df.to_parquet(fn, index=False)


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------
def clean_coordinates(
    df: pd.DataFrame, lat_col: str, lon_col: str, city: str | None, stats: CleaningStats
) -> pd.DataFrame:
    """
    Null out invalid coordinates. Rows are NOT dropped — an incident with a bad
    coordinate is still a real incident with a real category, timestamp and text,
    and is still useful for the text and resolution-time models.
    """
    rules = load_config()["cleaning"]["coordinates"]
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    bad = pd.Series(False, index=df.index)

    if rules.get("reject_zero_island", True):
        tol = float(rules.get("zero_tolerance_deg", 1e-4))
        zero = (lat.abs() < tol) & (lon.abs() < tol)
        stats.null("coordinates_null_island_0_0", int(zero.fillna(False).sum()))
        bad |= zero.fillna(False)

    oob = (
        (lat < rules["lat_min"]) | (lat > rules["lat_max"])
        | (lon < rules["lon_min"]) | (lon > rules["lon_max"])
    ).fillna(False)
    stats.null("coordinates_out_of_global_range", int(oob.sum()))
    bad |= oob

    # City bbox violations are FLAGGED, not nulled. A genuine SF case can sit
    # just outside a hand-drawn box; we surface it rather than destroy it.
    boxes = rules.get("city_bbox", {})
    if city and city in boxes:
        b = boxes[city]
        outside = (
            (lat < b["lat_min"]) | (lat > b["lat_max"])
            | (lon < b["lon_min"]) | (lon > b["lon_max"])
        ).fillna(False) & ~bad & lat.notna() & lon.notna()
        stats.flag(f"coordinates_outside_{city.replace(' ', '_').lower()}_bbox", int(outside.sum()))
        df = df.copy()
        df["coord_outside_city_bbox"] = outside
    else:
        df = df.copy()
        df["coord_outside_city_bbox"] = False

    df[lat_col] = lat.where(~bad)
    df[lon_col] = lon.where(~bad)
    return df


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
def parse_timestamps(
    df: pd.DataFrame, cols: list[str], source_key: str, stats: CleaningStats
) -> pd.DataFrame:
    """
    Parse to tz-aware UTC.

    The source portals publish wall-clock local time with no offset. We attach
    the city's timezone from config and convert to UTC so that hour-of-day
    features are comparable across cities. Ambiguous/nonexistent local times
    (DST transitions) are resolved forward rather than dropped.
    """
    ccfg = load_config()["cleaning"]["timestamps"]
    tz = ccfg["source_timezones"].get(source_key)
    vmin = pd.Timestamp(ccfg["valid_min"], tz="UTC")
    vmax = pd.Timestamp(datetime.now(timezone.utc) + timedelta(days=int(ccfg["valid_max_offset_days"])))

    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_datetime(df[c], errors="coerce")
        n_unparseable = int(s.isna().sum() - df[c].isna().sum())
        stats.null(f"timestamp_unparseable__{c}", max(0, n_unparseable))

        if tz and getattr(s.dtype, "tz", None) is None:
            s = s.dt.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")
        s = s.dt.tz_convert("UTC") if getattr(s.dtype, "tz", None) is not None else s

        impossible = ((s < vmin) | (s > vmax)).fillna(False)
        stats.null(f"timestamp_out_of_range__{c}", int(impossible.sum()))
        df[c] = s.where(~impossible)
    return df


def _raw_duration_hours(df: pd.DataFrame, opened_col: str, closed_col: str) -> pd.Series:
    """closed - opened in hours, with no rules applied. NaN when either is missing."""
    opened = df[opened_col]
    closed = df[closed_col] if closed_col in df.columns else pd.Series(pd.NaT, index=df.index)
    return (closed - opened).dt.total_seconds() / 3600.0


def instant_closure_flag(df: pd.DataFrame, opened_col: str, closed_col: str) -> pd.Series:
    """
    True where the case was closed sooner than the data can actually measure.

    See cleaning.resolution_time.min_observable_seconds in dataset_config.yaml
    for the full justification. In short: one minute is the coarsest timestamp
    granularity across the publishers, and the sub-minute population is a
    machine signature (Chicago closes 18,279 cases at exactly 7 seconds; NYC's
    entire sub-minute population sits at exactly 0) rather than the short tail
    of a service-time distribution.

    NULL where the case is not closed at all — "we cannot tell" is not "False".
    """
    rules = load_config()["cleaning"]["resolution_time"]
    floor_s = rules.get("min_observable_seconds")
    hrs = _raw_duration_hours(df, opened_col, closed_col)
    if not floor_s:
        return pd.Series(pd.NA, index=df.index, dtype="boolean")
    # A NEGATIVE duration is a different defect — an internally inconsistent
    # record, already counted as resolution_time_negative — so it is False here
    # rather than being folded into the instant-closure population.
    floor_h = float(floor_s) / 3600.0
    return ((hrs >= 0) & (hrs < floor_h)).where(hrs.notna()).astype("boolean")


def compute_resolution_hours(
    df: pd.DataFrame, opened_col: str, closed_col: str, stats: CleaningStats
) -> pd.Series:
    """
    resolution_time_hours = closed - opened. NULL when closed is missing.

    Negative, absurd and UNMEASURABLY SHORT durations are nulled, not clipped.
    A negative duration means the record is internally inconsistent, and
    silently flooring it at zero would inject a fake "instantly resolved" case
    into the SLA model. A closure written seconds after intake is the same
    problem wearing a positive sign: it is a property of how the record was
    written, not of how long the work took, so it is removed from the target
    rather than left to teach a model clerical behaviour.

    Nothing is dropped here. The row survives with its status, closed_at and
    `resolution_instant_closure` flag intact; only the duration is unobserved.
    """
    rules = load_config()["cleaning"]["resolution_time"]
    hrs = _raw_duration_hours(df, opened_col, closed_col)

    if rules.get("reject_negative", True):
        neg = (hrs < 0).fillna(False)
        stats.null("resolution_time_negative", int(neg.sum()))
        hrs = hrs.where(~neg)

    floor_s = rules.get("min_observable_seconds")
    if floor_s:
        # negatives are already NULL from the rule above; this counts only the
        # genuinely-instant population so the two defects stay distinguishable
        instant = ((hrs >= 0) & (hrs < float(floor_s) / 3600.0)).fillna(False)
        stats.null("resolution_time_instant_closure", int(instant.sum()))
        hrs = hrs.where(~instant)

    cap = float(rules.get("max_hours", 87600))
    too_long = (hrs > cap).fillna(False)
    stats.null("resolution_time_exceeds_cap", int(too_long.sum()))
    hrs = hrs.where(~too_long)
    return hrs


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
def clean_text(s: pd.Series) -> pd.Series:
    cfg = load_config()["cleaning"]["text"]
    out = s.astype("string")
    if cfg.get("strip_html", True):
        out = out.map(lambda v: _TAG.sub(" ", html.unescape(v)) if isinstance(v, str) else v)
    if cfg.get("collapse_whitespace", True):
        out = out.map(lambda v: _WS.sub(" ", v).strip() if isinstance(v, str) else v)
    maxlen = int(cfg.get("max_length", 4000))
    out = out.map(lambda v: v[:maxlen] if isinstance(v, str) else v)
    if cfg.get("lowercase", False):
        out = out.str.lower()
    return out.replace({"": pd.NA, "N/A": pd.NA, "n/a": pd.NA, "NA": pd.NA, "-": pd.NA})


def drop_exact_duplicate_rows(df: pd.DataFrame, subset: list[str], stats: CleaningStats) -> pd.DataFrame:
    """Exact re-exports of the same source record. Distinct from *incident* duplicates."""
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    stats.remove("exact_duplicate_source_rows", before - len(df))
    return df


# ---------------------------------------------------------------------------
# Geo helper
# ---------------------------------------------------------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in metres."""
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
