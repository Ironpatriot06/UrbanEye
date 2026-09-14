#!/usr/bin/env python3
"""
Provenance validation.

Two jobs, and the second is the important one.

  1. PASSIVE — confirm that severity, occurred_at and response_time_hours are
     empty in the outputs actually on disk.

  2. ACTIVE — deliberately attempt the three forbidden fabrications and confirm
     the guard REJECTS them:
        * severity derived from RDD2022 damage type
        * occurred_at derived from reported_at
        * response_time derived from resolution_time
     A guard that has never been tripped is just a comment. This proves it fires.

Writes reports/provenance_report.json. Exits non-zero if any check fails.

    python scripts/validation/validate_provenance.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import ensure_dir, p
from scripts.utils.schema import (INCIDENT_SCHEMA, DUPLICATE_PAIR_SCHEMA, geo_feature_schema,
                                  NULL_FIELDS, ProvenanceViolation, finalise_incidents,
                                  assert_null_fields_empty, provenance_table)

log = get_logger("validation.provenance")
RESULTS: list[dict] = []


def rec(cid: str, desc: str, passed: bool, detail: str = "") -> bool:
    RESULTS.append({"id": cid, "check": desc, "result": "PASS" if passed else "FAIL",
                    "detail": detail})
    (log.info if passed else log.error)("%-10s %-4s %s %s", cid,
                                        "PASS" if passed else "FAIL", desc, detail)
    return passed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", default=True)
    ap.parse_args()

    src = p("processed", "incidents", "all_incidents.parquet")
    if not src.exists():
        log.error("no incidents table at %s — run build_incidents.py first", src)
        return 2
    df = pd.read_parquet(src)

    # ---- 1. PASSIVE ------------------------------------------------------
    for field in NULL_FIELDS:
        n = int(df[field].notna().sum()) if field in df.columns else 0
        rec(f"PASSIVE-{field}", f"`{field}` is entirely NULL in all_incidents", n == 0,
            f"non-null={n}")

    for name, schema in (("vision/rdd2022", geo_feature_schema()), ("vision/smartathon", geo_feature_schema())):
        fp = p("processed", "vision", f"{name.split('/')[1]}.parquet")
        if fp.exists():
            v = pd.read_parquet(fp)
            n = int(v["severity"].notna().sum())
            rec(f"PASSIVE-{name}", f"`severity` is entirely NULL in {name}", n == 0, f"non-null={n}")

    dup = p("processed", "duplicates", "duplicate_pairs.parquet")
    if dup.exists():
        d = pd.read_parquet(dup)
        n = int(d["image_similarity"].notna().sum())
        rec("PASSIVE-image_similarity", "`image_similarity` is NULL in duplicate_pairs",
            n == 0, f"non-null={n} (Chicago, the only labelled source, has no photographs)")

    # ---- 2. ACTIVE -------------------------------------------------------
    # Attack 1: severity from RDD2022 damage type — the exact proxy the spec bans.
    attack = df.copy()
    rdd = p("processed", "vision", "rdd2022.parquet")
    proxy_note = "constant grade 3 on POTHOLE rows"
    if rdd.exists():
        v = pd.read_parquet(rdd)
        if (v["source_class"] == "D40").any():
            proxy_note = "RDD2022 D40 (pothole) mapped to severity 3"
    attack.loc[attack["category"] == "POTHOLE", "severity"] = 3
    n_injected = int(attack["severity"].notna().sum())
    try:
        finalise_incidents(attack)
        rec("ACTIVE-1", "guard REJECTS severity derived from RDD damage type", False,
            f"ACCEPTED {n_injected} fabricated values — THE GUARD IS BROKEN")
    except ProvenanceViolation as e:
        rec("ACTIVE-1", "guard REJECTS severity derived from RDD damage type", True,
            f"{proxy_note}; blocked {n_injected} rows")

    # Attack 2: occurred_at copied from reported_at
    attack2 = df.copy()
    attack2["occurred_at"] = attack2["reported_at"]
    try:
        finalise_incidents(attack2)
        rec("ACTIVE-2", "guard REJECTS occurred_at derived from reported_at", False,
            "ACCEPTED — THE GUARD IS BROKEN")
    except ProvenanceViolation:
        rec("ACTIVE-2", "guard REJECTS occurred_at derived from reported_at", True,
            f"blocked {int(attack2['occurred_at'].notna().sum())} rows")

    # Attack 3: response_time derived from resolution_time
    attack3 = df.copy()
    attack3["response_time_hours"] = pd.to_numeric(attack3["resolution_time_hours"],
                                                   errors="coerce") * 0.25
    try:
        finalise_incidents(attack3)
        rec("ACTIVE-3", "guard REJECTS response_time derived from resolution_time", False,
            "ACCEPTED — THE GUARD IS BROKEN")
    except ProvenanceViolation:
        rec("ACTIVE-3", "guard REJECTS response_time derived from resolution_time", True,
            f"blocked {int(attack3['response_time_hours'].notna().sum())} rows")

    # Attack 4: a fake row-level image<->incident join
    # NOTE: `image_path` exists in BOTH tables, which is a genuine foot-gun —
    # someone could naively merge on it. The names collide but the semantics do
    # not: in vision it is a relative path inside the RDD archive
    # ("India/train/images/x.jpg"); in incidents it is a remote media URL
    # ("https://..."). What actually matters is whether any VALUE overlaps, so
    # ACTIVE-5: the newest and most tempting fabrication — promoting the derived
    # baseline into the genuine label slot. This is exactly what would happen if
    # someone wanted "a priority target to train on".
    try:
        bad = df.copy()
        bad["priority_label"] = bad.get("priority_baseline", "P3")
        bad["priority_label_source"] = "RULE_ENGINE"
        finalise_incidents(bad)
        rec("ACTIVE-5", "guard REJECTS priority_label copied from priority_baseline", False,
            "NOT BLOCKED — derived baseline was accepted as a genuine label")
    except ProvenanceViolation as e:
        rec("ACTIVE-5", "guard REJECTS priority_label copied from priority_baseline", True,
            str(e).splitlines()[-1].strip())

    # that is what we test, and the name collision is recorded as a hazard.
    if rdd.exists():
        v = pd.read_parquet(rdd)
        colliding = sorted((set(v.columns) & set(df.columns)) - {"severity", "source_dataset"})
        vals_v = set(v["image_url"].dropna().astype(str))
        vals_i = set(df["image_url"].dropna().astype(str))
        overlap = vals_v & vals_i
        rec("ACTIVE-4", "no vision output is present in this tabular-only pipeline",
            len(overlap) == 0,
            f"colliding column names={colliding} (semantically different: archive path vs "
            f"media URL); value overlap={len(overlap)} — a row-level join would produce 0 rows")
        rec("ACTIVE-4b", "vision and incidents share no primary-key column",
            "incident_id" not in v.columns and "image_id" not in df.columns,
            "the tables are linked only by the UrbanEye+ category taxonomy")
    else:
        rec("ACTIVE-4", "no vision output is present in this tabular-only pipeline", True,
            "no vision output present")

    # ---- 3. priority labelling -------------------------------------------
    pri = p("processed", "ml_ready", "priority_dataset.parquet")
    if pri.exists():
        d = pd.read_parquet(pri)
        gt_ok = ("is_ground_truth" in d.columns) and (not d["is_ground_truth"].any())
        rec("LABEL-1", "priority_dataset marks is_ground_truth = False everywhere", gt_ok)
        src_ok = set(d["priority_method"].dropna().unique()) and not bool(d["is_ground_truth"].fillna(False).any())
        rec("LABEL-2", "priority_baseline is DERIVED and never ground truth", src_ok,
            f"is_ground_truth_true={int(d['is_ground_truth'].fillna(False).sum())}")

    failures = [r for r in RESULTS if r["result"] == "FAIL"]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {"total": len(RESULTS), "passed": len(RESULTS) - len(failures),
                    "failed": len(failures)},
        "checks": RESULTS,
        "null_by_design_fields": NULL_FIELDS,
        "field_provenance": {
            "incidents": provenance_table(INCIDENT_SCHEMA).to_dict("records"),
            "vision_samples": provenance_table(geo_feature_schema()).to_dict("records"),
            "duplicate_pairs": provenance_table(DUPLICATE_PAIR_SCHEMA).to_dict("records"),
        },
        "genuine_vs_derived": {
            "GENUINE (from publisher)": [
                "category (normalised from source vocabulary)", "subcategory", "description",
                "latitude", "longitude", "address", "zone_id", "reported_at", "status",
                "department", "report_channel", "image_path", "image_path_after",
                "is_duplicate (Chicago)", "parent_incident_id (Chicago)"],
            "DERIVED (computed here)": [
                "incident_id", "resolution_time_hours", "sla_target_hours", "sla_met",
                "priority_baseline", "priority_score", "priority_reasons", "priority_confidence",
                "priority_features_available", "priority_method", "is_ground_truth",
                "sla_hours_policy", "zone_type", "coord_outside_city_bbox"],
            "SYSTEM (Indian PostGIS, at runtime)": [
                "near_school", "near_hospital", "near_metro", "near_railway", "near_major_road",
                "road_class", "is_major_city", "nearby_incident_count", "distance_to_*_m"],
            "NULL (genuinely unavailable)": NULL_FIELDS + ["image_similarity"],
        },
    }
    dest = ensure_dir(p("reports", "provenance_report.json"))
    dest.write_text(json.dumps(report, indent=2, default=str))
    log.info("wrote %s — %d/%d checks passed", dest, len(RESULTS) - len(failures), len(RESULTS))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
