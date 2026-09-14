#!/usr/bin/env python3
"""
Build the consolidated ML-ready UrbanEye table.

    data/processed/incidents/all_incidents_prioritised.parquet   (identity, context, outcomes)
  + data/processed/features/incident_time_features.parquet       (local-time block)
  + data/processed/geo_features/incident_geo_features.parquet    (POI schema + density)
        -> data/processed/ml/urbaneye_ml.parquet
           data/processed/ml/urbaneye_ml.manifest.json
           data/processed/ml/urbaneye_ml_sample.csv

WHAT CHANGED AND WHY
--------------------
The previous version of this script read a hand-made
`data/processed/combined/urbaneye_incidents.parquet` that no script in the
repository produced, and then manufactured a target called `priority_target` by
counting keywords ("fire", "water", "hazard", ...) in `category` **and
`description`**. Three separate problems, each fatal on its own:

  1. `description` is AGENCY-written resolution text. It is written after the
     case is closed. A target derived from it cannot be predicted at report
     time, and a model trained on it scores well offline and is unusable live.
  2. It was a fabricated label presented as a target. The repository's whole
     provenance design exists to prevent exactly that, and `validate_provenance`
     actively tries the same trick and asserts it is rejected.
  3. It also dropped every priority/provenance column, so nothing downstream
     could tell that the target was invented.

That target is gone. This builder emits only:

  PREDICTORS      things knowable the moment the citizen presses submit
  TARGETS         only genuinely observed outcomes (resolution time, SLA breach)
  POLICY METADATA priority_baseline and friends — the rule engine's output,
                  carried for comparison and audit, never as a predictor and
                  never as a target
  PROVENANCE      what is missing and why, machine-readable

Every column's role is declared in the manifest, so "is this a feature?" has a
recorded answer rather than a convention.

    python scripts/features/build_ml_dataset.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.preprocess._ml_common import chronological_split
from scripts.utils.logging_setup import get_logger
from scripts.utils.mapping import CANONICAL
from scripts.utils.paths import ensure_dir, load_feature_config, load_priority_config, p

log = get_logger("features.ml_dataset")

IDENTIFIERS = ["incident_id", "source_dataset", "city"]

CATEGORICAL_PREDICTORS = ["category", "subcategory", "zone_id", "zone_type",
                          "report_channel", "department"]

# Known at report time: whether the submission carried usable coordinates.
# `coord_outside_city_bbox` is deliberately NOT here — it is False for all
# 1,900,000 rows, so as a predictor it is a constant. It stays in the table as a
# QA flag.
QUALITY_PREDICTORS = ["has_coordinates"]
QUALITY_METADATA = ["coord_outside_city_bbox"]

POLICY_METADATA = ["priority_baseline", "priority_score", "priority_reasons",
                   "priority_confidence", "priority_features_available",
                   "priority_feature_coverage", "priority_method", "sla_hours_policy"]

LABEL_PROVENANCE = ["priority_label", "priority_label_source", "is_ground_truth",
                    "target_status", "target_strategy"]

# Geometry and wall-clock are kept for auditing, joining and grouping. They are
# NOT predictors: a model that learns "latitude 41.88 is high priority" has
# learned Chicago, which is worse than useless in an Indian city.
CONTEXT_METADATA = ["latitude", "longitude", "reported_at", "reported_at_local",
                    "local_timezone", "date_local", "split"]

# Observed outcomes. Targets, never inputs.
TARGETS = ["resolution_time_hours", "sla_breach"]
TARGET_SUPPORT = ["sla_target_hours", "target_is_censored"]

# Known only after closure. Excluded from this table altogether (other than the
# declared targets above), and asserted absent by validate_leakage.py.
EXCLUDED_POST_RESOLUTION = ["closed_at", "status", "sla_met", "description",
                            "resolution_description", "status_notes", "image_url",
                            "image_url_after"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incidents", default=None)
    ap.add_argument("--in-scope-only", action="store_true",
                    help="drop UNMAPPED / REVIEW_REQUIRED / OUT_OF_SCOPE rows")
    ap.add_argument("--sample-rows", type=int, default=5000)
    args = ap.parse_args()

    src = Path(args.incidents) if args.incidents else \
        p("processed", "incidents", "all_incidents_prioritised.parquet")
    if not src.exists():
        src = p("processed", "incidents", "all_incidents.parquet")
    if not src.exists():
        log.error("no incidents table found. Run: python scripts/run_pipeline.py")
        return 2

    df = pd.read_parquet(src)
    log.info("loaded %d incidents from %s", len(df), src.name)

    time_fp = p("processed", "features", "incident_time_features.parquet")
    geo_fp = p("processed", "geo_features", "incident_geo_features.parquet")
    missing_inputs = [str(f) for f in (time_fp, geo_fp) if not f.exists()]
    if missing_inputs:
        log.error("missing upstream feature tables: %s\n"
                  "  Run build_time_features.py and build_geo_features.py first, "
                  "or simply: python scripts/run_pipeline.py", missing_inputs)
        return 2

    tf = pd.read_parquet(time_fp)
    geo = pd.read_parquet(geo_fp)
    df = df.merge(tf, on="incident_id", how="left", suffixes=("", "_time"))
    df = df.merge(geo, on="incident_id", how="left", suffixes=("", "_geo"))
    log.info("joined %d temporal and %d geospatial columns", tf.shape[1] - 1, geo.shape[1] - 1)

    if args.in_scope_only:
        before = len(df)
        df = df[df["category"].isin(CANONICAL)].copy()
        log.info("in-scope filter: kept %d of %d rows", len(df), before)

    fc = load_feature_config()
    temporal = list(fc.get("temporal_features", {}).keys())
    poi_cols, density_cols = [], []
    for block in ("poi_proximity_features", "road_network_features", "area_context_features"):
        poi_cols += list(fc.get(block, {}).keys())
    density_cols += list(fc.get("historical_density_features", {}).keys())
    density_cols += list(fc.get("historical_recency_features", {}).keys())

    out = pd.DataFrame(index=df.index)
    for c in IDENTIFIERS:
        out[c] = df[c] if c in df.columns else pd.NA

    out["has_coordinates"] = (df["latitude"].notna() & df["longitude"].notna()).astype("boolean")
    out["coord_outside_city_bbox"] = df.get(
        "coord_outside_city_bbox", pd.Series(pd.NA, index=df.index)).astype("boolean")

    for c in CATEGORICAL_PREDICTORS + temporal + poi_cols + density_cols:
        out[c] = df[c] if c in df.columns else pd.NA

    # ---- targets: observed outcomes only -----------------------------------
    res = pd.to_numeric(df.get("resolution_time_hours"), errors="coerce")
    sla_target = pd.to_numeric(df.get("sla_target_hours"), errors="coerce")
    out["resolution_time_hours"] = res.astype("float64")
    out["sla_breach"] = (res > sla_target).where(res.notna() & sla_target.notna()).astype("boolean")
    out["sla_target_hours"] = sla_target.astype("float64")
    # right-censoring is a property of the TARGET, not a feature
    out["target_is_censored"] = res.isna().astype("boolean")

    # ---- policy metadata ---------------------------------------------------
    for c in POLICY_METADATA:
        out[c] = df[c] if c in df.columns else pd.NA

    pri = load_priority_config()
    out["priority_label"] = pd.NA
    out["priority_label_source"] = pd.NA
    out["is_ground_truth"] = False
    out["target_status"] = "NO_PRIORITY_GROUND_TRUTH_AVAILABLE"
    out["target_strategy"] = "policy_baseline_only"

    for c in ("latitude", "longitude", "reported_at", "reported_at_local",
              "local_timezone", "date_local"):
        out[c] = df[c] if c in df.columns else pd.NA

    split, split_meta = chronological_split(df)
    out["split"] = split

    # ---- guards ------------------------------------------------------------
    leaked = [c for c in EXCLUDED_POST_RESOLUTION if c in out.columns]
    if leaked:
        raise RuntimeError(f"post-resolution columns reached the ML table: {leaked}")
    dupes = int(out["incident_id"].duplicated().sum())
    if dupes:
        raise RuntimeError(f"{dupes} duplicate incident_id values — refusing to write. "
                           "Deduplicating here would hide a defect upstream.")

    out = out.sort_values(["reported_at", "incident_id"], kind="stable").reset_index(drop=True)

    dest = ensure_dir(p("processed", "ml", "urbaneye_ml.parquet"))
    out.to_parquet(dest, index=False, compression="snappy")

    # ---- manifest: every column's role, declared --------------------------
    available = {c: int(out[c].notna().sum()) for c in out.columns}
    roles: dict[str, str] = {}
    for c in IDENTIFIERS:
        roles[c] = "identifier"
    for c in CATEGORICAL_PREDICTORS + temporal + QUALITY_PREDICTORS:
        roles[c] = "predictor"
    for c in QUALITY_METADATA:
        roles[c] = "context_metadata"
    # hour_utc is determined by (city, local hour) up to the DST shift, so as a
    # predictor it is a city indicator dressed as a clock. It is kept for audit.
    roles["hour_utc"] = "context_metadata"
    for c in poi_cols:
        roles[c] = "predictor_unavailable"      # correct name and dtype, no values
    for c in density_cols:
        roles[c] = "predictor"
    for c in TARGETS:
        roles[c] = "target"
    for c in TARGET_SUPPORT:
        roles[c] = "target_support"
    for c in POLICY_METADATA:
        roles[c] = "policy_metadata"
    for c in LABEL_PROVENANCE:
        roles[c] = "label_provenance"
    for c in CONTEXT_METADATA:
        roles[c] = "context_metadata"

    usable_predictors = [c for c, r in roles.items()
                         if r == "predictor" and available.get(c, 0) > 0]
    unavailable = [c for c, r in roles.items() if r == "predictor_unavailable"]

    manifest = {
        "rows": int(len(out)),
        "columns": int(out.shape[1]),
        "built_from": {"incidents": str(src), "time_features": str(time_fp),
                       "geo_features": str(geo_fp)},
        "column_roles": roles,
        "non_null_counts": available,
        "usable_predictors": usable_predictors,
        "declared_but_unavailable_predictors": {
            "columns": unavailable,
            "reason": ("POI, road-network and area-context features require an external "
                       "geospatial reference layer. The public US 311 datasets contain none, "
                       "and this repository ships no POI/road/boundary reference data — "
                       "data/external does not exist. The columns are emitted with the correct "
                       "name and dtype and left NULL so the schema is stable for the day an "
                       "Indian PostGIS provider is connected. They are NOT imputed."),
        },
        "targets": {
            "resolution_time_hours": {
                "kind": "regression", "rows_with_target": available.get("resolution_time_hours", 0),
                "note": "closed_at - reported_at. Right-censored: open cases are NULL, "
                        "flagged by target_is_censored, and are NOT dropped here."},
            "sla_breach": {
                "kind": "binary", "rows_with_target": available.get("sla_breach", 0),
                "note": "resolution_time_hours > sla_target_hours. Only where the publisher "
                        "ships a due/target timestamp: NYC (Due Date) and Boston (TARGET_DT)."},
            "priority_label": {
                "kind": "multiclass", "rows_with_target": 0, "available": False,
                "status": "NO_PRIORITY_GROUND_TRUTH_AVAILABLE",
                "note": ("No public 311 dataset records an operational priority, urgency or "
                         "severity. priority_baseline is a deterministic function of "
                         "config/priority_config.yaml and is carried as policy_metadata: "
                         "training on it would reproduce the YAML file, and quoting that "
                         "accuracy as model performance would be circular. Use it as the "
                         "BENCHMARK the eventual model must beat, not as its target.")},
        },
        "excluded_post_resolution_columns": {
            "columns": EXCLUDED_POST_RESOLUTION,
            "reason": ("None is knowable at report time. description in particular is "
                       "agency-written resolution text, not citizen text."),
        },
        "excluded_by_design": {
            "latitude/longitude as predictors": (
                "kept as context_metadata only. Raw coordinates teach US geography, which "
                "does not transfer to an Indian deployment; only semantic features derived "
                "from them may be used."),
            "hour_utc as a predictor": (
                "demoted to context_metadata: it is a deterministic function of the city and "
                "the local hour, so a model using it is partly keying on the city."),
            "coord_outside_city_bbox as a predictor": (
                "demoted to context_metadata: constant False across every row."),
            "category_frequency / geo_grid / description_length": (
                "removed. They were computed over the whole corpus (including validation and "
                "test) or over post-resolution text, so they leaked."),
            "priority_target": (
                "removed. It was a keyword count over category and agency-written resolution "
                "text, presented as a label. Fabricated targets are what this pipeline's "
                "provenance guard exists to prevent."),
        },
        "split": split_meta,
        "split_counts": {str(k): int(v) for k, v in out["split"].value_counts().items()},
        "policy_config_version": pri["config_version"],
        "policy_config_status": pri["status"],
    }
    ensure_dir(p("processed", "ml", "urbaneye_ml.manifest.json")).write_text(
        json.dumps(manifest, indent=2, default=str))
    ensure_dir(p("reports", "ml_dataset_summary.json")).write_text(
        json.dumps(manifest, indent=2, default=str))

    # small, stratified, human-readable sample
    n = min(args.sample_rows, len(out))
    parts = []
    for _, g in out.groupby("split", sort=True):
        take = max(1, int(round(n * len(g) / max(1, len(out)))))
        parts.append(g.sample(min(take, len(g)), random_state=7))
    sample = pd.concat(parts).sort_values(["reported_at", "incident_id"]) if parts else out
    sample_dest = ensure_dir(p("processed", "ml", "urbaneye_ml_sample.csv"))
    sample.to_csv(sample_dest, index=False)

    log.info("wrote %s (%d rows, %d cols)", dest, len(out), out.shape[1])
    log.info("  usable predictors     : %d", len(usable_predictors))
    log.info("  declared-unavailable  : %d (await an external geospatial layer)", len(unavailable))
    log.info("  targets present       : resolution_time_hours=%d  sla_breach=%d  priority_label=0",
             available.get("resolution_time_hours", 0), available.get("sla_breach", 0))
    log.info("  split                 : %s", manifest["split_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
