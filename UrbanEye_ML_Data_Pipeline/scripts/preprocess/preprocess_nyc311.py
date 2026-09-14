#!/usr/bin/env python3
"""
Normalise NYC 311 (2010-2019) into the canonical `incidents` schema.

NYC specifics:
  * Due Date is documented by the publisher as SLA-derived, so
    sla_target_hours = Due Date - Created Date. It is present on ~8.66M of 22.6M
    rows; the rest get NULL.
  * Date hygiene is genuinely bad here: closed_date runs from 1899-12-31 to
    3027-03-30. The configured valid window nulls those.
  * NYC puts potholes in Descriptor, not Complaint Type ('Street Condition' +
    'Pothole'), so category mapping runs with the subcategory override tier.
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from scripts.preprocess._tabular import (DEFAULT_CHUNK, IncidentWriter, apply_common, clean_text,
                                         compute_resolution_hours, compute_sla, compute_sla_met,
                                         finish, get, normalise_channel, normalise_status,
                                         parse_timestamps, read_source_chunks, find_raw_file,
                                         missing_raw_message, resolve_col, write_unmapped)
from scripts.utils.cleaning import CleaningStats, drop_exact_duplicate_rows
from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import p
from scripts.utils.schema import INCIDENT_SCHEMA

log = get_logger("preprocess.nyc311")
DATASET, CITY = "nyc311", "New York City"
PATTERNS = ["*311*Service*Request*.csv", "nyc311*.csv", "rows.csv", "*.csv", "nyc311*.jsonl", "*.jsonl"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    args = ap.parse_args()

    src = find_raw_file(DATASET, PATTERNS, args.input)
    if src is None:
        print(missing_raw_message(DATASET, PATTERNS, str(p("raw", DATASET))), file=sys.stderr)
        return 2
    log.info("reading %s (%.1f MB)", src, src.stat().st_size / 1024**2)

    stats = CleaningStats(DATASET)
    writer = IncidentWriter(DATASET)
    unmapped: list = []
    due_present = 0

    for i, raw in enumerate(read_source_chunks(src, args.chunk_size), 1):
        stats.rows_in += len(raw)
        idc = resolve_col(raw, "unique_key", "Unique Key")
        raw = drop_exact_duplicate_rows(raw, [idc], stats) if idc else raw

        out = pd.DataFrame(index=raw.index)
        out["source_incident_id"] = get(raw, "unique_key").astype("string")
        out["source_category_raw"] = get(raw, "complaint_type").astype("string")
        out["subcategory"] = get(raw, "descriptor").astype("string")
        out["description"] = clean_text(get(raw, "resolution_description"))
        out["department"] = get(raw, "agency_name").astype("string")
        out["address"] = clean_text(get(raw, "incident_address"))
        out["latitude"] = get(raw, "latitude")
        out["longitude"] = get(raw, "longitude")
        out["zone_id"] = get(raw, "community_board").astype("string")
        out["zone_type"] = "community_board"
        out["status"] = normalise_status(get(raw, "status"))
        out["report_channel"] = normalise_channel(get(raw, "open_data_channel_type"))
        out["image_url"] = pd.NA                # NYC publishes no photos
        out["image_url_after"] = pd.NA
        out["reported_at"] = get(raw, "created_date")
        out["closed_at"] = get(raw, "closed_date")
        out["_target"] = get(raw, "due_date")

        out = parse_timestamps(out, ["reported_at", "closed_at", "_target"], DATASET, stats)
        out["resolution_time_hours"] = compute_resolution_hours(out, "reported_at", "closed_at", stats)
        out["sla_target_hours"] = compute_sla(out, "_target", "reported_at", stats)
        out["sla_met"] = compute_sla_met(out["resolution_time_hours"], out["sla_target_hours"])
        due_present += int(out["sla_target_hours"].notna().sum())

        out["is_duplicate"] = pd.NA              # no duplicate flag in NYC
        out["parent_incident_id"] = pd.NA

        out = apply_common(out, raw, dataset=DATASET, city=CITY,
                           category_col="source_category_raw", subcategory_col="subcategory",
                           stats=stats, unmapped_accum=unmapped)
        writer.write(out[[c for c in INCIDENT_SCHEMA if c in out.columns]])
        if i % 10 == 0:
            log.info("  chunk %d — %d rows in", i, stats.rows_in)

    write_unmapped(DATASET, unmapped, "source_category_raw")
    finish(DATASET, writer, stats, extra={
        "sla_available": True,
        "sla_derivation": "sla_target_hours = Due Date - Created Date (DERIVED; not a priority label)",
        "rows_with_sla_target": due_present,
        "note": "Publisher documents Due Date as based on complaint type and internal SLAs.",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
