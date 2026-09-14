#!/usr/bin/env python3
"""
Build resolution_dataset — target: resolution_time_hours (and sla_breach).

LEAKAGE IS THE WHOLE DIFFICULTY HERE. The incidents table carries several fields
that are only knowable after the case closes. Using any of them as a feature
produces a model that scores beautifully offline and cannot run in production,
because at report time none of them exists. This builder therefore emits an
explicit feature allow-list rather than "everything except the target":

  ALLOWED   category, subcategory, city, department, zone, report_channel,
            time-of-report features, backward-looking density features
  FORBIDDEN closed_at, status, sla_met, description (agency resolution text),
            resolution_time_hours itself

`department` deserves a note: it is assigned at intake by the 311 routing rules,
not at closure, so it is legitimately known at report time. If your municipality
assigns it later, drop it.

Two target framings are produced:
  resolution_time_hours  regression, on closed cases only
  sla_breach             binary, only where sla_target_hours exists (Boston, NYC)

Right-censoring: cases still open are excluded from the regression target. That
biases the sample toward faster resolutions. Survival analysis is the correct
treatment if long-running cases matter; the meta file records the count.

    python scripts/preprocess/build_resolution_dataset.py
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from scripts.preprocess._ml_common import (add_time_features, dual_chronological_split,
                                           in_scope, load_incidents, write)
from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import load_feature_config, p

log = get_logger("preprocess.resolution")

ALLOWED_FEATURES = ["category", "subcategory", "city", "department", "zone_id", "zone_type",
                    "report_channel", "hour", "day_of_week", "month", "year",
                    "is_weekend", "is_night", "hour_utc"]
FORBIDDEN = ["closed_at", "status", "sla_met", "description", "resolution_time_hours",
             "priority_reasons", "priority_score"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incidents", default=None)
    ap.add_argument("--geo-features", default=None)
    args = ap.parse_args()

    df = in_scope(load_incidents(args.incidents))
    total = len(df)
    no_target = df["resolution_time_hours"].isna()
    instant = df.get("resolution_instant_closure")
    instant = instant.fillna(False).astype(bool) if instant is not None else pd.Series(False, index=df.index)
    excluded_instant = int((no_target & instant).sum())
    excluded_open = int((no_target & ~instant).sum())
    df = df[df["resolution_time_hours"].notna()].copy()
    log.info("closed cases with a measurable duration: %d (excluded %d still-open/invalid, "
             "%d instant-closure artifacts)", len(df), excluded_open, excluded_instant)

    df = add_time_features(df)

    geo_fp = p("processed", "geo_features", "incident_geo_features.parquet") \
        if not args.geo_features else __import__("pathlib").Path(args.geo_features)
    density_cols: list[str] = []
    if geo_fp.exists():
        fc = load_feature_config()
        density_cols = list(fc["historical_density_features"].keys()) + \
            list(fc.get("historical_recency_features", {}).keys())
        geo = pd.read_parquet(geo_fp)[["incident_id"] + density_cols]
        df = df.merge(geo, on="incident_id", how="left")
        log.info("merged %d backward-looking density features", len(density_cols))

    cols = ["incident_id", "source_dataset", "reported_at"] + ALLOWED_FEATURES + density_cols + \
           ["sla_target_hours", "resolution_time_hours"]
    out = df[[c for c in cols if c in df.columns]].copy()

    # Secondary target: SLA breach, only where a real target timestamp existed.
    out["sla_breach"] = (out["resolution_time_hours"] > out["sla_target_hours"]).where(
        out["sla_target_hours"].notna()).astype("boolean")
    split, split_global, split_meta = dual_chronological_split(df)
    out["split"] = split
    out["split_global"] = split_global

    leaked = [c for c in FORBIDDEN if c in out.columns and c != "resolution_time_hours"]
    write(out, "resolution", "resolution_dataset", {
        "model": "Resolution time / SLA risk",
        "targets": {"regression": "resolution_time_hours",
                    "classification": "sla_breach (requires sla_target_hours)"},
        "feature_allow_list": ALLOWED_FEATURES + density_cols,
        "forbidden_and_excluded": FORBIDDEN,
        "leaked_columns_detected": leaked,
        "sla_breach_available_rows": int(out["sla_breach"].notna().sum()),
        "sla_breach_sources": "Boston (TARGET_DT) and NYC (Due Date) only",
        "excluded_rows": {
            "total_in_scope": total,
            "still_open_or_invalid": excluded_open,
            "instant_closure_artifact": excluded_instant,
            "right_censoring_note": ("Open cases are excluded, biasing the sample toward faster "
                                     "resolutions. Use survival analysis if long-running cases "
                                     "matter. Measured censoring on this corpus is mild (<1%)."),
            "instant_closure_note": ("Cases closed sooner than the source timestamps can measure "
                                     "(cleaning.resolution_time.min_observable_seconds). Their "
                                     "duration is unobserved, so they carry no regression target; "
                                     "the rows still exist in all_incidents with status, closed_at "
                                     "and the resolution_instant_closure flag intact.")},
        "split": split_meta,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
