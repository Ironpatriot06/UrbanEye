#!/usr/bin/env python3
"""
Normalise SF 311 Cases into the canonical `incidents` schema.

SF specifics handled here:
  * 551,131 rows sit at exactly lat=0, lon=0 (the null-island sentinel). Those
    coordinates are nulled; the rows are kept, because the category, text and
    resolution time are still valid training signal.
  * The 3.49M-row 'Street and Sidewalk Cleaning' category is genuinely ambiguous
    at the top level, so it is resolved through subcategory overrides
    (Category|Request Type) in config/category_mapping.csv rather than being
    force-mapped.
  * 'Case is a Duplicate' in Status Notes is a WEAK duplicate signal: it tells us
    a case was a duplicate, but not of WHAT. parent_incident_id stays NULL.
  * SF publishes no target/due timestamp, so sla_target_hours is NULL.
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from scripts.preprocess._tabular import (DEFAULT_CHUNK, IncidentWriter, apply_common, clean_text,
                                         compute_resolution_hours, extract_url, find_raw_file,
                                         finish, get, missing_raw_message, normalise_channel,
                                         normalise_status, parse_timestamps, read_source_chunks,
                                         write_unmapped)
from scripts.preprocess._tabular import resolve_col
from scripts.utils.cleaning import CleaningStats, drop_exact_duplicate_rows
from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import p
from scripts.utils.schema import INCIDENT_SCHEMA

log = get_logger("preprocess.sf311")
DATASET, CITY = "sf311", "San Francisco"
PATTERNS = ["*311*Cases*.csv", "sf311*.csv", "rows.csv", "*.csv", "sf311_cases.jsonl", "*.jsonl"]


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

    for i, raw in enumerate(read_source_chunks(src, args.chunk_size), 1):
        stats.rows_in += len(raw)
        idc = resolve_col(raw, "service_request_id", "CaseID")
        raw = drop_exact_duplicate_rows(raw, [idc], stats) \
            if resolve_col(raw, "service_request_id", "CaseID") else raw

        out = pd.DataFrame(index=raw.index)
        out["source_incident_id"] = get(raw, "service_request_id", "CaseID", "case_id").astype("string")
        out["source_category_raw"] = get(raw, "service_name", "Category").astype("string")
        out["subcategory"] = get(raw, "service_subtype", "Request Type").astype("string")
        out["_details"] = get(raw, "service_details", "Request Details").astype("string")
        out["description"] = clean_text(get(raw, "status_notes", "Status Notes"))
        out["department"] = get(raw, "agency_responsible", "Responsible Agency").astype("string")
        out["address"] = clean_text(get(raw, "address"))
        out["latitude"] = get(raw, "lat", "Latitude")
        out["longitude"] = get(raw, "long", "Longitude")
        out["zone_id"] = get(raw, "analysis_neighborhood", "Analysis Neighborhood").astype("string")
        out["zone_type"] = "analysis_neighborhood"
        out["status"] = normalise_status(get(raw, "status_description", "Status"))
        out["report_channel"] = normalise_channel(get(raw, "source"))
        out["image_url"] = extract_url(get(raw, "media_url", "Media URL"))
        out["image_url_after"] = pd.NA          # SF publishes no resolution photo
        out["reported_at"] = get(raw, "requested_datetime", "Opened")
        out["closed_at"] = get(raw, "closed_date", "Closed")

        out = parse_timestamps(out, ["reported_at", "closed_at"], DATASET, stats)
        out["resolution_time_hours"] = compute_resolution_hours(out, "reported_at", "closed_at", stats)

        # Weak duplicate signal. No parent pointer exists in SF, so we record the
        # flag and deliberately leave parent_incident_id NULL.
        notes = out["description"].fillna("")
        out["is_duplicate"] = notes.str.contains("is a Duplicate", case=False, na=False).astype("boolean")
        out["parent_incident_id"] = pd.NA
        stats.flag("sf_weak_duplicate_flagged", int(out["is_duplicate"].fillna(False).sum()))

        out["sla_target_hours"] = pd.NA          # SF publishes no target/due date
        out["sla_met"] = pd.NA

        out = apply_common(out, raw, dataset=DATASET, city=CITY,
                           category_col="source_category_raw", subcategory_col="subcategory",
                           stats=stats, unmapped_accum=unmapped)
        writer.write(out[[c for c in INCIDENT_SCHEMA if c in out.columns]])
        if i % 10 == 0:
            log.info("  chunk %d — %d rows in", i, stats.rows_in)

    write_unmapped(DATASET, unmapped, "source_category_raw")
    finish(DATASET, writer, stats, extra={
        "sla_available": False,
        "sla_note": "SF 311 publishes no target/due timestamp; sla_target_hours is NULL for all rows.",
        "duplicate_signal": "weak (Status Notes text match); no parent pointer available",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
