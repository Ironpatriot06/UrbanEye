#!/usr/bin/env python3
"""
Union the per-city incident tables into data/processed/incidents/all_incidents.parquet.

This is a UNION, not a join. The four cities share a schema; they do not share
entities. No cross-city record linkage is attempted or implied.
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import ensure_dir, p
from scripts.utils.schema import finalise_incidents, INCIDENT_SCHEMA

log = get_logger("preprocess.build_incidents")
SOURCES = ["sf311", "boston311", "chicago311", "nyc311"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=SOURCES)
    args = ap.parse_args()

    frames, present, missing = [], [], []
    for s in args.sources:
        fp = p("processed", "incidents", f"{s}.parquet")
        if not fp.exists():
            missing.append(s); log.warning("missing %s — skipping", fp.name); continue
        df = pd.read_parquet(fp)
        log.info("%-12s %8d rows", s, len(df))
        frames.append(df); present.append(s)

    if not frames:
        log.error("no per-source incident tables found; run the process_*.py scripts first")
        return 2

    allx = pd.concat(frames, ignore_index=True)
    allx = finalise_incidents(allx)     # re-assert the provenance guard on the union

    dest = ensure_dir(p("processed", "incidents", "all_incidents.parquet"))
    allx.to_parquet(dest, index=False)

    summary = {
        "sources_included": present,
        "sources_missing": missing,
        "total_rows": int(len(allx)),
        "rows_by_source": {k: int(v) for k, v in allx["source_dataset"].value_counts().items()},
        "rows_by_city": {k: int(v) for k, v in allx["city"].value_counts().items()},
        "category_distribution": {str(k): int(v) for k, v in allx["category"].value_counts().items()},
        "null_by_design": {
            f: int(allx[f].notna().sum()) for f in ["severity", "occurred_at", "response_time_hours", "priority_label", "priority_label_source"]
        },
        "coverage": {
            "with_coordinates": int(allx["latitude"].notna().sum()),
            "with_description": int(allx["description"].notna().sum()),
            "with_image_url": int(allx["image_url"].notna().sum()),
            "with_after_image_url": int(allx["image_url_after"].notna().sum()),
            "with_sla_target": int(allx["sla_target_hours"].notna().sum()),
            "with_resolution_time": int(allx["resolution_time_hours"].notna().sum()),
            "with_parent_pointer": int(allx["parent_incident_id"].notna().sum()),
        },
        "note": "UNION of four independent city datasets. No cross-city entity resolution performed.",
    }
    ensure_dir(p("reports", "incidents_summary.json")).write_text(json.dumps(summary, indent=2))
    log.info("wrote %s (%d rows)", dest, len(allx))
    log.info("coverage: %s", summary["coverage"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
