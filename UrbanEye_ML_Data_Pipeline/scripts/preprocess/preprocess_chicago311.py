#!/usr/bin/env python3
"""
Normalise Chicago 311 into the canonical `incidents` schema.

Chicago specifics:
  * This is the only source with a real duplicate POINTER (PARENT_SR_NUMBER,
    ~706,730 non-null). It becomes parent_incident_id and is what
    build_duplicate_pairs.py turns into supervised positive pairs.
  * There is NO free-text description field. description stays NULL rather than
    being back-filled from SR_TYPE, which would just be the category again.
  * No target/due timestamp, so sla_target_hours is NULL.
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from scripts.preprocess._tabular import (DEFAULT_CHUNK, IncidentWriter, apply_common, clean_text,
                                         compute_resolution_hours, finish, get, normalise_channel,
                                         normalise_status, parse_timestamps, read_source_chunks, find_raw_file,
                                         missing_raw_message, resolve_col,
                                         write_unmapped, build_incident_id)
from scripts.utils.cleaning import CleaningStats, drop_exact_duplicate_rows
from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import load_config, p
from scripts.utils.schema import INCIDENT_SCHEMA

log = get_logger("preprocess.chicago311")
DATASET, CITY = "chicago311", "Chicago"
PATTERNS = ["*311*Service*Request*.csv", "chicago311*.csv", "rows.csv", "*.csv", "chicago311.jsonl", "*.jsonl"]


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

    excluded = set(load_config()["datasets"][DATASET]["socrata"].get("exclude_sr_types", []))
    stats = CleaningStats(DATASET)
    writer = IncidentWriter(DATASET)
    unmapped: list = []

    for i, raw in enumerate(read_source_chunks(src, args.chunk_size), 1):
        stats.rows_in += len(raw)
        idc = resolve_col(raw, "sr_number", "SR_NUMBER")
        raw = drop_exact_duplicate_rows(raw, [idc], stats) if idc else raw

        # Belt and braces: the download filters these server-side, but a
        # --keep-excluded run or a pre-existing file would not be filtered.
        tcol = resolve_col(raw, "sr_type", "SR_TYPE")
        if excluded and tcol:
            mask = raw[tcol].isin(excluded)
            stats.remove("excluded_irrelevant_sr_type", int(mask.sum()))
            raw = raw[~mask]
        if raw.empty:
            continue

        out = pd.DataFrame(index=raw.index)
        out["source_incident_id"] = get(raw, "sr_number").astype("string")
        out["source_category_raw"] = get(raw, "sr_type").astype("string")
        out["subcategory"] = pd.NA               # Chicago has no true subcategory
        out["description"] = pd.NA               # Chicago has NO free-text field
        out["department"] = get(raw, "owner_department").astype("string")
        out["address"] = clean_text(get(raw, "street_address"))
        out["latitude"] = get(raw, "latitude")
        out["longitude"] = get(raw, "longitude")
        out["zone_id"] = get(raw, "ward").astype("string")
        out["zone_type"] = "ward"
        out["status"] = normalise_status(get(raw, "status"))
        out["report_channel"] = normalise_channel(get(raw, "origin"))
        out["image_url"] = pd.NA                # Chicago publishes no photos
        out["image_url_after"] = pd.NA
        out["reported_at"] = get(raw, "created_date")
        out["closed_at"] = get(raw, "closed_date")

        out = parse_timestamps(out, ["reported_at", "closed_at"], DATASET, stats)
        out["resolution_time_hours"] = compute_resolution_hours(out, "reported_at", "closed_at", stats)

        dup = get(raw, "duplicate").astype("string").str.lower().isin(["true", "1", "yes"])
        out["is_duplicate"] = dup.astype("boolean")
        parent = get(raw, "parent_sr_number").astype("string").str.strip()
        out["parent_incident_id"] = build_incident_id(DATASET, parent).where(parent.notna() & parent.ne(""))
        stats.flag("chicago_duplicate_true", int(dup.sum()))
        stats.flag("chicago_parent_pointer_present", int(out["parent_incident_id"].notna().sum()))

        out["sla_target_hours"] = pd.NA          # Chicago publishes no target/due date
        out["sla_met"] = pd.NA

        out = apply_common(out, raw, dataset=DATASET, city=CITY,
                           category_col="source_category_raw", subcategory_col=None,
                           stats=stats, unmapped_accum=unmapped)
        writer.write(out[[c for c in INCIDENT_SCHEMA if c in out.columns]])
        if i % 10 == 0:
            log.info("  chunk %d — %d rows in", i, stats.rows_in)

    write_unmapped(DATASET, unmapped, "source_category_raw")
    finish(DATASET, writer, stats, extra={
        "sla_available": False,
        "text_available": False,
        "text_note": "Chicago 311 has no free-text description column; description is NULL by design.",
        "duplicate_signal": "strong (PARENT_SR_NUMBER pointer)",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
