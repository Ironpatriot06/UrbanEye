#!/usr/bin/env python3
"""
Normalise Boston 311 yearly CSVs into the canonical `incidents` schema.

Boston is the only source with an explicit SLA target, so it carries most of the
weight for anything SLA-related:
  * sla_target_hours = TARGET_DT - OPEN_DT  (DERIVED, not a priority label)
  * OnTime_Status (ONTIME/OVERDUE) is retained as a source-reported compliance
    signal and cross-checked against our own computed sla_met. Disagreements are
    counted, not silently reconciled — if the city's flag and our arithmetic
    disagree, that is something a human should see.
  * SubmittedPhoto / ClosedPhoto are the only before/after photo pair anywhere in
    this corpus.

Column casing is inconsistent across the 2011-2024 files, so every field is
fetched through the tolerant resolver rather than by exact name.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from scripts.preprocess._tabular import (DEFAULT_CHUNK, IncidentWriter, apply_common, clean_text,
                                         compute_resolution_hours, compute_sla, compute_sla_met,
                                         extract_url, finish, get, normalise_channel,
                                         normalise_status, parse_timestamps, read_csv_chunks,
                                         write_unmapped)
from scripts.utils.cleaning import CleaningStats, drop_exact_duplicate_rows
from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import p
from scripts.utils.schema import INCIDENT_SCHEMA

log = get_logger("preprocess.boston311")
DATASET, CITY = "boston311", "Boston"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    args = ap.parse_args()

    raw_dir = Path(args.input_dir) if args.input_dir else p("raw", DATASET)
    files = sorted(raw_dir.glob("boston311_*.csv")) or sorted(raw_dir.glob("*.csv"))
    if not files:
        log.error("no *.csv found in %s — download the yearly Boston 311 CSVs manually "
                  "and place them there. See DOWNLOAD_INSTRUCTIONS.md.", raw_dir)
        return 2
    log.info("processing %d yearly files", len(files))

    stats = CleaningStats(DATASET)
    writer = IncidentWriter(DATASET)
    unmapped: list = []
    ontime_agree = ontime_disagree = ontime_unknown = 0
    photos_submitted = photos_closed = 0

    for f in files:
        log.info("  %s", f.name)
        for raw in read_csv_chunks(f, args.chunk_size):
            stats.rows_in += len(raw)
            idc = "case_enquiry_id"
            if any(c.lower() == idc for c in raw.columns):
                col = [c for c in raw.columns if c.lower() == idc][0]
                raw = drop_exact_duplicate_rows(raw, [col], stats)

            out = pd.DataFrame(index=raw.index)
            out["source_incident_id"] = get(raw, "case_enquiry_id", "CASE_ENQUIRY_ID").astype("string")
            out["source_category_raw"] = get(raw, "reason", "REASON").astype("string")
            out["subcategory"] = get(raw, "type", "TYPE").astype("string")
            title = clean_text(get(raw, "case_title", "CASE_TITLE"))
            closure = clean_text(get(raw, "closure_reason", "CLOSURE_REASON"))
            out["description"] = (title.fillna("") + ". " + closure.fillna("")).str.strip(" .").replace("", pd.NA)
            out["department"] = get(raw, "department", "Department").astype("string")
            out["address"] = clean_text(get(raw, "location", "Location"))
            out["latitude"] = get(raw, "latitude", "Latitude")
            out["longitude"] = get(raw, "longitude", "Longitude")
            out["zone_id"] = get(raw, "ward").astype("string")
            out["zone_type"] = "ward"
            out["status"] = normalise_status(get(raw, "case_status", "CASE_STATUS"))
            out["report_channel"] = normalise_channel(get(raw, "source", "Source"))

            sub = extract_url(get(raw, "submittedphoto", "SubmittedPhoto", "submitted_photo"))
            clo = extract_url(get(raw, "closedphoto", "ClosedPhoto", "closed_photo"))
            out["image_url"] = sub
            out["image_url_after"] = clo
            photos_submitted += int(sub.notna().sum())
            photos_closed += int(clo.notna().sum())

            out["reported_at"] = get(raw, "open_dt", "OPEN_DT")
            out["closed_at"] = get(raw, "closed_dt", "CLOSED_DT")
            out["_target"] = get(raw, "sla_target_dt", "target_dt", "TARGET_DT")
            out = parse_timestamps(out, ["reported_at", "closed_at", "_target"], DATASET, stats)

            out["resolution_time_hours"] = compute_resolution_hours(out, "reported_at", "closed_at", stats)
            out["sla_target_hours"] = compute_sla(out, "_target", "reported_at", stats)
            out["sla_met"] = compute_sla_met(out["resolution_time_hours"], out["sla_target_hours"])

            # Reconcile our computed sla_met against the city's own OnTime flag.
            src_ontime = get(raw, "ontime_status", "OnTime_Status", "on_time").astype("string").str.strip().str.upper()
            src_bool = src_ontime.map({"ONTIME": True, "ON TIME": True, "OVERDUE": False}).astype("boolean")
            comparable = src_bool.notna() & out["sla_met"].notna()
            ontime_agree += int((src_bool.eq(out["sla_met"]) & comparable).sum())
            ontime_disagree += int((src_bool.ne(out["sla_met"]) & comparable).sum())
            ontime_unknown += int((~comparable).sum())

            out["is_duplicate"] = pd.NA       # Boston publishes no duplicate flag
            out["parent_incident_id"] = pd.NA

            out = apply_common(out, raw, dataset=DATASET, city=CITY,
                               category_col="source_category_raw", subcategory_col="subcategory",
                               stats=stats, unmapped_accum=unmapped)
            writer.write(out[[c for c in INCIDENT_SCHEMA if c in out.columns]])

    write_unmapped(DATASET, unmapped, "source_category_raw")
    finish(DATASET, writer, stats, extra={
        "sla_available": True,
        "sla_derivation": "sla_target_hours = TARGET_DT - OPEN_DT (DERIVED; not a priority label)",
        "ontime_reconciliation": {
            "agree": ontime_agree, "disagree": ontime_disagree, "not_comparable": ontime_unknown,
            "note": ("Disagreements are reported, not reconciled. The city's OnTime flag may "
                     "reflect business-hours SLA arithmetic that a wall-clock difference does not."),
        },
        "photo_urls": {"submitted_non_null": photos_submitted, "closed_non_null": photos_closed},
        "years_excluded": [2025, 2026],
        "years_excluded_reason": "Boston 311 backend migration from Oct 2025; the NEW SYSTEM resource has a different schema.",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
