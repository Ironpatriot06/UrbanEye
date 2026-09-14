#!/usr/bin/env python3
"""
Build `incident_time_features` — the report-time temporal block.

    data/processed/incidents/all_incidents.parquet
        -> data/processed/features/incident_time_features.parquet

WHY THIS IS ITS OWN STAGE
-------------------------
Time features used to be derived inline inside build_priority_features.py, which
meant (a) they existed only in the priority table, (b) every other builder
re-derived them slightly differently, and (c) they were taken off the UTC
timestamp, so `hour` and `is_night` encoded the city's longitude rather than the
time of day. Deriving them once, in the incident's own local timezone, and
joining the result by incident_id fixes all three.

The incident schema itself is deliberately NOT extended: INCIDENT_SCHEMA is
provenance-guarded and stable, and these are derived features, so they live in
their own table alongside incident_geo_features.

LEAKAGE
-------
Nil by construction. Every column is a pure function of the row's own
reported_at, which is known the moment the citizen presses submit.

    python scripts/preprocess/build_time_features.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import ensure_dir, p
from scripts.utils.timefeatures import (TIME_FEATURE_COLUMNS, TIME_MODEL_FEATURES,
                                        night_window, source_timezones, time_features)

log = get_logger("preprocess.time_features")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    args = ap.parse_args()

    src = Path(args.input) if args.input else p("processed", "incidents", "all_incidents.parquet")
    if not src.exists():
        log.error("not found: %s — run build_incidents.py first", src)
        return 2

    df = pd.read_parquet(src, columns=["incident_id", "source_dataset", "city", "reported_at"])
    log.info("loaded %d incidents", len(df))

    tf = time_features(df)
    out = pd.concat([df[["incident_id"]].reset_index(drop=True), tf.reset_index(drop=True)], axis=1)

    dest = ensure_dir(p("processed", "features", "incident_time_features.parquet"))
    out.to_parquet(dest, index=False)

    n_start, n_end = night_window()
    unmapped = sorted(set(df.loc[out["local_timezone"].isna(), "source_dataset"]
                          .astype("string").dropna().unique()))
    w = df.reset_index(drop=True).join(tf.reset_index(drop=True))
    w["_offset_h"] = ((w["reported_at_local"]
                       - pd.to_datetime(w["reported_at"], utc=True).dt.tz_localize(None))
                      .dt.total_seconds() / 3600.0)
    by_source = {}
    for ds, g in w.groupby("source_dataset"):
        tzs = g["local_timezone"].dropna().unique().tolist()
        by_source[str(ds)] = {
            "rows": int(len(g)),
            "timezone": str(tzs[0]) if tzs else None,
            "utc_offset_hours_observed": sorted(
                {round(float(v), 2) for v in g["_offset_h"].dropna().unique()}),
            "hour_min": int(g["hour"].min()) if g["hour"].notna().any() else None,
            "hour_max": int(g["hour"].max()) if g["hour"].notna().any() else None,
            "night_share": round(float(g["is_night"].mean()), 4),
            "weekend_share": round(float(g["is_weekend"].mean()), 4),
            "hour_histogram": {int(k): int(v) for k, v in
                               g["hour"].value_counts().sort_index().items()},
        }

    summary = {
        "rows": int(len(out)),
        "source": str(src),
        "columns": TIME_FEATURE_COLUMNS,
        "model_features": TIME_MODEL_FEATURES,
        "timezone_policy": {
            "timestamps_stored_as": "UTC",
            "features_derived_in": "source city local time",
            "mapping": source_timezones(),
            "night_window_local": f"{n_start:02d}:00-{n_end:02d}:00",
        },
        "null_counts": {c: int(out[c].isna().sum()) for c in TIME_FEATURE_COLUMNS},
        "by_source": by_source,
        "sources_without_timezone_mapping": unmapped,
        "leakage_note": ("Every column is a pure function of the row's own reported_at, "
                         "which is known at report time. No future information is used."),
    }
    ensure_dir(p("reports", "time_features_summary.json")).write_text(json.dumps(summary, indent=2))
    log.info("wrote %s (%d rows, %d cols)", dest, len(out), out.shape[1])
    for ds, d in by_source.items():
        log.info("  %-12s tz=%-20s utc_offset=%s night=%.1f%%",
                 ds, d["timezone"], d["utc_offset_hours_observed"], 100 * d["night_share"])
    if unmapped:
        log.error("sources with NO configured timezone (features left NULL): %s", unmapped)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
