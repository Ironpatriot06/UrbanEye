#!/usr/bin/env python3
"""
The UrbanEye+ baseline priority engine, plus the ML-ready priority feature set.

WHAT THIS PRODUCES
------------------
  priority_baseline      P1-P4 from a configurable weighted score. DERIVED.
  priority_score         the normalised 0-1 score behind it
  priority_reasons       which features drove it, for the admin-facing explanation
  priority_confidence    HIGH/LOW based on how many inputs were actually available
  priority_label         NULL. A genuine operational priority. None exists publicly.
  priority_label_source  NULL.
  is_ground_truth        False, always, for the baseline.
  sla_hours_policy       the business-rule SLA for the assigned priority

WHY THE BASELINE IS NOT A TRAINING TARGET
-----------------------------------------
priority_baseline is a deterministic function of the weights in
config/priority_config.yaml. Training a model to predict it would produce a model
that has learned the YAML file — high accuracy, zero information. The ML priority
model needs priority_label, which requires real UrbanEye+ operational outcomes.
Until those exist, this script prepares the FEATURE side of that dataset and
leaves the target NULL. See PRIORITY_METHODOLOGY.md.

HOW MISSING FEATURES ARE HANDLED
--------------------------------
mode: strict_available_only. The score is the weighted sum over the features that
were actually non-null, divided by the sum of THOSE weights. Missing is never
silently treated as zero/false, because "we don't know whether a school is
nearby" and "there is definitely no school nearby" are different statements and
only one of them should lower a priority.

The practical consequence on US 311 data: every POI feature is NULL, so the score
reduces to the category term alone and every row is flagged
priority_confidence=LOW. That is the honest result and the script reports it
prominently rather than burying it.

  python scripts/preprocess/build_priority_features.py --accept-draft-config
  python scripts/preprocess/build_priority_features.py --explain POTHOLE \
      --near-school 50 --near-major-road 15 --nearby-30d 37
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
from scripts.utils.paths import (ensure_dir, load_config, load_feature_config,
                                 load_priority_config, p)

log = get_logger("preprocess.priority")

# feature name -> (kind, threshold_key)
PROXIMITY_FEATURES = {
    "near_school":              "school",
    "near_hospital":            "hospital",
    "near_metro":               "metro",
    "near_public_transport":    "public_transport",
    "near_major_road":          "major_road",
    "near_intersection":        "intersection",
    "near_fire_station":        "fire_station",
    "near_police_station":      "police_station",
    "near_government_building": "government_building",
}
DISTANCE_COL = {
    "near_school": "distance_to_nearest_school_m",
    "near_hospital": "distance_to_nearest_hospital_m",
    "near_metro": "distance_to_nearest_metro_m",
    "near_public_transport": "distance_to_nearest_public_transport_m",
    "near_major_road": "distance_to_major_road_m",
    "near_intersection": "distance_to_nearest_intersection_m",
    "near_fire_station": "distance_to_nearest_fire_station_m",
    "near_police_station": "distance_to_nearest_police_station_m",
    "near_government_building": "distance_to_nearest_government_building_m",
}
BOOLEAN_FEATURES = ["metro_city", "urban_area"]
COUNT_FEATURES = ["local_incident_density", "nearby_similar_incidents_30d"]

HUMAN = {
    "category": "incident category",
    "near_school": "near a school",
    "near_hospital": "near a hospital",
    "near_metro": "near a metro station",
    "near_public_transport": "near public transport",
    "near_major_road": "on or beside a major road",
    "near_intersection": "near an intersection",
    "near_fire_station": "near a fire station",
    "near_police_station": "near a police station",
    "near_government_building": "near a government building",
    "metro_city": "in a metropolitan city",
    "urban_area": "in an urban area",
    "local_incident_density": "high local incident density",
    "nearby_similar_incidents_30d": "many similar incidents reported nearby recently",
}


class PriorityEngine:
    """
    The configurable baseline priority engine.

    Two entry points that must agree exactly:
      evaluate(row)  one row, readable, used by --explain and by the tests
      apply(frame)   the same arithmetic vectorised over millions of rows

    tests/test_pipeline.py::test_vectorised_engine_matches_row_engine asserts
    they produce identical bands, scores and availability counts on randomised
    input, so the fast path can never quietly drift away from the readable one.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_priority_config()
        self.version = self.cfg["config_version"]
        self.status = self.cfg.get("status", "DRAFT")
        self.w = self.cfg["weights"]
        self.base = self.cfg["category_base_score"]
        self.absolute_p1 = set(self.cfg.get("category_absolute_p1", []))
        self.thresholds = self.cfg["proximity_thresholds_m"]
        self.saturation = self.cfg["count_saturation"]
        self.bands = self.cfg["score_bands"]
        self.nulls = self.cfg["null_handling"]
        self.expl = self.cfg["explanation"]
        self.sla = self.cfg["sla_hours"]
        self.tiers = self.cfg.get("confidence_tiers")
        self.total_weight = float(sum(self.w.values()))

    # -- signal extraction --------------------------------------------------
    def _proximity_signal(self, distance, threshold: float):
        """Graded: 1.0 at the POI, 0.0 at the threshold. NULL stays NULL."""
        if distance is None or (isinstance(distance, float) and np.isnan(distance)) or pd.isna(distance):
            return None
        return float(np.clip(1.0 - (float(distance) / float(threshold)), 0.0, 1.0))

    def _count_signal(self, value, saturation: float):
        if value is None or pd.isna(value):
            return None
        return float(np.clip(float(value) / float(saturation), 0.0, 1.0))

    def _bool_signal(self, value):
        if value is None or pd.isna(value):
            return None
        return 1.0 if bool(value) else 0.0

    def signals(self, row: dict) -> dict[str, float | None]:
        s: dict[str, float | None] = {}
        cat = row.get("category")
        s["category"] = float(self.base.get(cat, self.base.get("OTHER", 0.35))) if cat else None

        for feat, tkey in PROXIMITY_FEATURES.items():
            if feat not in self.w:
                continue
            s[feat] = self._proximity_signal(row.get(DISTANCE_COL[feat]),
                                             self.thresholds.get(tkey, 200))
        for feat in BOOLEAN_FEATURES:
            if feat in self.w:
                s[feat] = self._bool_signal(row.get(feat))
        for feat in COUNT_FEATURES:
            if feat in self.w:
                s[feat] = self._count_signal(row.get(feat), self.saturation.get(feat, 30))
        return s

    # -- confidence ---------------------------------------------------------
    def confidence(self, n_avail: int, coverage: float) -> str:
        """
        Deterministic tier from how much of the policy was actually evaluable.

        Nothing here is inferred from the incident itself: the same row with the
        same available inputs always lands in the same tier, and both inputs to
        the decision are published alongside it.
        """
        if self.tiers:
            for tier in self.tiers:
                if (n_avail >= int(tier.get("min_features", 0))
                        and coverage >= float(tier.get("min_weight_coverage", 0.0))):
                    return str(tier["name"])
            return "LOW"
        return "HIGH" if n_avail >= int(self.nulls["min_features_for_confident_score"]) else "LOW"

    def _band(self, score: float) -> str:
        for level in ("P1", "P2", "P3", "P4"):
            if score >= float(self.bands[level]["min"]):
                return level
        return "P4"

    # -- scoring (single row) ----------------------------------------------
    def evaluate(self, row: dict) -> dict:
        cat = row.get("category")

        if cat in self.absolute_p1:
            return {
                "priority_baseline": "P1", "priority_score": 1.0,
                "priority_reasons": f"category rule: {HUMAN['category']} {cat} is an "
                                    "unconditional immediate hazard",
                "priority_confidence": "HIGH",
                "priority_features_available": 1,
                "priority_feature_coverage": 1.0,
            }

        sig = self.signals(row)
        avail = {k: v for k, v in sig.items() if v is not None}
        wsum = sum(self.w[k] for k in avail if k in self.w)
        if wsum <= 0:
            return {
                "priority_baseline": None, "priority_score": None,
                "priority_reasons": "no scoring inputs available",
                "priority_confidence": "LOW", "priority_features_available": 0,
                "priority_feature_coverage": 0.0,
            }

        contrib = {k: self.w[k] * v for k, v in avail.items() if k in self.w}
        score = sum(contrib.values()) / wsum
        band = self._band(score)

        n_avail = len(avail)
        coverage = wsum / self.total_weight
        conf = self.confidence(n_avail, coverage)

        top = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
        floor = float(self.expl["min_contribution_to_report"])
        reasons = []
        for k, v in top[: int(self.expl["max_reasons"])]:
            if v < floor:
                continue
            label = HUMAN.get(k, k)
            # np.round, not the built-in, so this matches _reasons() bit for bit
            vr = float(np.round(v, 2))
            reasons.append(f"{label} ({vr:.2f})" if k != "category" else f"category {cat} ({vr:.2f})")
        if conf != "HIGH":
            reasons.append(f"{conf} CONFIDENCE: only {n_avail} of {len(self.w)} inputs available")

        return {
            "priority_baseline": band,
            "priority_score": float(np.round(score, 4)),
            "priority_reasons": " | ".join(reasons),
            "priority_confidence": conf,
            "priority_features_available": n_avail,
            "priority_feature_coverage": round(float(coverage), 4),
        }

    # -- scoring (vectorised) ----------------------------------------------
    def _signal_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        n x k float matrix of signals, NaN where the input was unavailable.

        NaN is the only "missing" marker used anywhere in the vectorised path;
        it is never coerced to 0.0, because that would turn "unknown" into
        "definitely not near anything" and systematically deflate priorities.
        """
        feats = list(self.w.keys())
        n = len(df)
        mat = np.full((n, len(feats)), np.nan, dtype=np.float64)
        for j, feat in enumerate(feats):
            if feat == "category":
                cat = df["category"].astype("string") if "category" in df.columns else pd.Series(
                    pd.NA, index=df.index, dtype="string")
                default = float(self.base.get("OTHER", 0.35))
                vals = cat.map(lambda c: float(self.base.get(c, default))
                               if isinstance(c, str) and c else np.nan)
                mat[:, j] = pd.to_numeric(vals, errors="coerce").to_numpy(np.float64)
            elif feat in PROXIMITY_FEATURES:
                col = DISTANCE_COL[feat]
                if col not in df.columns:
                    continue
                thr = float(self.thresholds.get(PROXIMITY_FEATURES[feat], 200))
                d = pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
                mat[:, j] = np.clip(1.0 - d / thr, 0.0, 1.0)
            elif feat in BOOLEAN_FEATURES:
                if feat not in df.columns:
                    continue
                v = df[feat]
                b = v.astype("boolean")
                mat[:, j] = np.where(b.isna().to_numpy(), np.nan,
                                     b.fillna(False).to_numpy(bool).astype(np.float64))
            elif feat in COUNT_FEATURES:
                if feat not in df.columns:
                    continue
                sat = float(self.saturation.get(feat, 30))
                c = pd.to_numeric(df[feat], errors="coerce").to_numpy(np.float64)
                mat[:, j] = np.clip(c / sat, 0.0, 1.0)
        return mat, feats

    def _reasons(self, contrib: np.ndarray, feats: list[str], cat: pd.Series,
                 conf: np.ndarray, n_avail: np.ndarray) -> pd.Series:
        """Top-k contribution explanation, built column-wise rather than row-wise."""
        max_reasons = int(self.expl["max_reasons"])
        floor = float(self.expl["min_contribution_to_report"])
        filled = np.where(np.isnan(contrib), -np.inf, contrib)
        order = np.argsort(-filled, axis=1, kind="stable")[:, :max_reasons]
        labels = np.array([HUMAN.get(f, f) for f in feats], dtype=object)
        cat_idx = feats.index("category") if "category" in feats else -1
        catstr = cat.astype("string").fillna("?")

        parts = []
        for r in range(order.shape[1]):
            idx = order[:, r]
            val = filled[np.arange(len(idx)), idx]
            keep = np.isfinite(val) & (val >= floor)
            label = pd.Series(labels[idx], index=cat.index)
            if cat_idx >= 0:
                label = label.where(idx != cat_idx, "category " + catstr)
            piece = label.astype("string") + " (" + pd.Series(
                np.round(val, 2), index=cat.index).map("{:.2f}".format) + ")"
            parts.append(piece.where(pd.Series(keep, index=cat.index), pd.NA))

        out = parts[0].fillna("")
        for piece in parts[1:]:
            out = np.where(piece.isna(), out, out + " | " + piece.fillna(""))
            out = pd.Series(out, index=cat.index, dtype="string")
        note = pd.Series(conf, index=cat.index).astype("string") + " CONFIDENCE: only " + \
            pd.Series(n_avail, index=cat.index).astype("string") + f" of {len(self.w)} inputs available"
        low = pd.Series(conf, index=cat.index) != "HIGH"
        out = out.where(~low, out.where(out.str.len() > 0, "").str.cat(note, sep=" | ").str.strip(" |"))
        return out.astype("string")

    def apply(self, df: pd.DataFrame, chunk_rows: int = 400_000) -> pd.DataFrame:
        """Score the whole frame. Chunked so peak memory stays flat on 1.9M rows."""
        pieces = []
        for s0 in range(0, len(df), chunk_rows):
            pieces.append(self._apply_block(df.iloc[s0:s0 + chunk_rows]))
        res = pd.concat(pieces) if pieces else self._apply_block(df)

        out = df.copy()
        for col in res.columns:
            out[col] = res[col]
        out["priority_method"] = f"priority_config.yaml@{self.version}"
        out["priority_label"] = pd.NA          # genuine label: none exists publicly
        out["priority_label_source"] = pd.NA
        out["is_ground_truth"] = False
        out["sla_hours_policy"] = out["priority_baseline"].map(self.sla).astype("float64")
        return out

    def _apply_block(self, df: pd.DataFrame) -> pd.DataFrame:
        mat, feats = self._signal_matrix(df)
        weights = np.array([float(self.w[f]) for f in feats], dtype=np.float64)
        avail = ~np.isnan(mat)
        n_avail = avail.sum(axis=1).astype(np.int64)
        wsum = (avail * weights).sum(axis=1)
        contrib = mat * weights
        total = np.nansum(np.where(avail, contrib, 0.0), axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(wsum > 0, total / np.where(wsum > 0, wsum, 1.0), np.nan)
        coverage = wsum / self.total_weight

        band = np.full(len(df), "P4", dtype=object)
        for level in ("P4", "P3", "P2", "P1"):
            band = np.where(score >= float(self.bands[level]["min"]), level, band)
        band = np.where(np.isnan(score), None, band)

        conf = np.full(len(df), "LOW", dtype=object)
        if self.tiers:
            for tier in reversed(self.tiers):
                hit = (n_avail >= int(tier.get("min_features", 0))) & \
                      (coverage >= float(tier.get("min_weight_coverage", 0.0)))
                conf = np.where(hit, str(tier["name"]), conf)
        else:
            conf = np.where(n_avail >= int(self.nulls["min_features_for_confident_score"]),
                            "HIGH", "LOW")

        cat = df["category"] if "category" in df.columns else pd.Series(
            pd.NA, index=df.index, dtype="string")
        reasons = self._reasons(contrib, feats, cat, conf, n_avail)
        reasons = reasons.where(wsum > 0, "no scoring inputs available")

        # absolute category rules bypass the score entirely
        is_abs = cat.isin(self.absolute_p1).fillna(False).to_numpy()
        if is_abs.any():
            band = np.where(is_abs, "P1", band)
            score = np.where(is_abs, 1.0, score)
            conf = np.where(is_abs, "HIGH", conf)
            n_avail = np.where(is_abs, 1, n_avail)
            coverage = np.where(is_abs, 1.0, coverage)
            abs_reason = pd.Series(
                [f"category rule: {HUMAN['category']} {c} is an unconditional immediate hazard"
                 if isinstance(c, str) else "" for c in cat], index=df.index, dtype="string")
            reasons = reasons.where(~is_abs, abs_reason)

        return pd.DataFrame({
            "priority_baseline": pd.Series(band, index=df.index, dtype="string"),
            "priority_score": pd.Series(np.round(score.astype(float), 4), index=df.index,
                                        dtype="float64"),
            "priority_reasons": reasons,
            "priority_confidence": pd.Series(conf, index=df.index, dtype="string"),
            "priority_features_available": pd.Series(n_avail, index=df.index, dtype="Int64"),
            "priority_feature_coverage": pd.Series(np.round(coverage, 4), index=df.index,
                                                   dtype="float64"),
        })


def _target_manifest(df: pd.DataFrame) -> dict:
    """
    State, in machine-readable form, what can and cannot be used as a target.

    This exists because "the target column is empty" is too easy to misread as
    "the target column has not been built yet". It has been built; there is
    nothing to put in it.
    """
    closed = int(pd.to_numeric(df.get("resolution_time_hours"), errors="coerce").notna().sum()) \
        if "resolution_time_hours" in df.columns else 0
    sla = int(pd.to_numeric(df.get("sla_target_hours"), errors="coerce").notna().sum()) \
        if "sla_target_hours" in df.columns else 0
    return {
        "priority": {
            "target_column": "priority_label",
            "available": False,
            "rows_with_target": 0,
            "status": "NO_PRIORITY_GROUND_TRUTH_AVAILABLE",
            "strategy": "policy_baseline_only",
            "explanation": (
                "No public 311 dataset records an operational priority, urgency, severity or "
                "risk grade, so there is nothing to learn from. priority_baseline is a "
                "deterministic function of config/priority_config.yaml: training on it would "
                "reproduce the YAML, and reporting that accuracy as model performance would be "
                "circular. priority_baseline is therefore published as POLICY METADATA, never "
                "as a label, and is_ground_truth is False on every row. What would unblock a "
                "real target is operator-assigned priorities from UrbanEye+'s own console — "
                "see PRIORITY_METHODOLOGY.md section 6."),
            "searched_for": ["status", "closed_at", "resolved_at", "completed_at",
                             "response_time", "resolution_time", "sla breach", "escalation",
                             "severity", "priority", "urgency", "risk"],
            "found": ("status / closed_at / resolution_time_hours exist but are POST-RESOLUTION "
                      "outcomes, not priorities; sla_target_hours exists for NYC (Due Date) and "
                      "Boston (TARGET_DT) only and encodes that city's own staffing policy."),
        },
        "observable_targets_that_do_exist": {
            "resolution_time_hours": {"available": closed > 0, "rows_with_target": closed,
                                      "built_by": "scripts/preprocess/build_resolution_dataset.py"},
            "sla_breach": {"available": sla > 0, "rows_with_target": sla,
                           "note": "requires sla_target_hours: NYC and Boston only",
                           "built_by": "scripts/preprocess/build_resolution_dataset.py"},
            "future_incident_count": {"available": True,
                                      "built_by": "scripts/preprocess/build_hotspot_dataset.py"},
            "same_incident": {"available": True, "note": "Chicago parent pointers only",
                              "built_by": "scripts/preprocess/build_duplicate_pairs.py"},
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incidents", default=None)
    ap.add_argument("--geo-features", default=None)
    ap.add_argument("--time-features", default=None)
    ap.add_argument("--accept-draft-config", action="store_true")
    ap.add_argument("--explain", default=None, help="category name for a single what-if")
    ap.add_argument("--near-school", type=float, default=None, help="metres")
    ap.add_argument("--near-hospital", type=float, default=None)
    ap.add_argument("--near-metro", type=float, default=None)
    ap.add_argument("--near-major-road", type=float, default=None)
    ap.add_argument("--near-intersection", type=float, default=None)
    ap.add_argument("--metro-city", action="store_true")
    ap.add_argument("--nearby-30d", type=float, default=None)
    args = ap.parse_args()

    engine = PriorityEngine()

    if args.explain:
        row = {"category": args.explain,
               "distance_to_nearest_school_m": args.near_school,
               "distance_to_nearest_hospital_m": args.near_hospital,
               "distance_to_nearest_metro_m": args.near_metro,
               "distance_to_major_road_m": args.near_major_road,
               "distance_to_nearest_intersection_m": args.near_intersection,
               "metro_city": True if args.metro_city else None,
               "nearby_similar_incidents_30d": args.nearby_30d}
        r = engine.evaluate(row)
        print(f"\n  category            : {args.explain}")
        print(f"  inputs supplied     : {[k for k, v in row.items() if k != 'category' and v is not None]}")
        print(f"  priority_baseline   : {r['priority_baseline']}")
        print(f"  priority_score      : {r['priority_score']}")
        print(f"  confidence          : {r['priority_confidence']} "
              f"({r['priority_features_available']} inputs available, "
              f"{100 * r['priority_feature_coverage']:.0f}% of policy weight)")
        print(f"  reasons             : {r['priority_reasons']}")
        print(f"  SLA (business rule) : {engine.sla.get(r['priority_baseline'])} hours")
        print(f"  method              : priority_config.yaml@{engine.version} (status={engine.status})")
        print("  NOTE                : DERIVED baseline. is_ground_truth=False.\n")
        return 0

    if engine.status != "APPROVED" and not args.accept_draft_config:
        print(
            f"\n  REFUSING TO RUN.\n"
            f"  config/priority_config.yaml has status={engine.status}, approved_by=None.\n"
            f"  Those weights and bands are a template, not UrbanEye+ policy.\n\n"
            f"  Set status: APPROVED (with approved_by/approved_at) after a product review,\n"
            f"  or re-run with --accept-draft-config to proceed knowingly.\n", file=sys.stderr)
        return 3

    src = Path(args.incidents) if args.incidents else p("processed", "incidents", "all_incidents.parquet")
    if not src.exists():
        log.error("not found: %s\n  Run the stages in order:\n"
                  "    python scripts/preprocess/preprocess_<source>.py\n"
                  "    python scripts/preprocess/build_incidents.py\n"
                  "    python scripts/preprocess/build_time_features.py\n"
                  "    python scripts/preprocess/build_geo_features.py\n"
                  "  or simply: python scripts/run_pipeline.py", src)
        return 2
    df = pd.read_parquet(src)

    # ---- time features (local city time) -----------------------------------
    time_fp = Path(args.time_features) if args.time_features else \
        p("processed", "features", "incident_time_features.parquet")
    if time_fp.exists():
        tf = pd.read_parquet(time_fp)
        df = df.merge(tf, on="incident_id", how="left", suffixes=("", "_time"))
        log.info("merged %d temporal columns from %s", tf.shape[1] - 1, time_fp.name)
        time_source = str(time_fp)
    else:
        log.warning("no time features at %s — deriving in process", time_fp)
        from scripts.utils.timefeatures import time_features as _tf
        for c, v in _tf(df).items():
            df[c] = v
        time_source = "computed in process (build_time_features.py had not been run)"

    # ---- geospatial + density features -------------------------------------
    geo_fp = Path(args.geo_features) if args.geo_features else \
        p("processed", "geo_features", "incident_geo_features.parquet")
    if geo_fp.exists():
        geo = pd.read_parquet(geo_fp)
        df = df.merge(geo, on="incident_id", how="left", suffixes=("", "_geo"))
        log.info("merged %d geospatial columns", geo.shape[1] - 1)
        geo_source = str(geo_fp)
    else:
        log.warning("no geo features at %s — every POI and density input will be NULL and "
                    "the score will collapse to the category term. Run build_geo_features.py.",
                    geo_fp)
        geo_source = "MISSING — build_geo_features.py was not run"

    log.info("applying priority config %s (status=%s) to %d rows", engine.version, engine.status, len(df))
    out = engine.apply(df)

    # --- write the enriched incidents table ---------------------------------
    from scripts.utils.schema import INCIDENT_SCHEMA, finalise_incidents
    inc_cols = [c for c in INCIDENT_SCHEMA if c in out.columns]
    inc = finalise_incidents(out[inc_cols])
    inc_dest = ensure_dir(p("processed", "incidents", "all_incidents_prioritised.parquet"))
    inc.to_parquet(inc_dest, index=False)

    # --- write the ML-ready priority feature set ----------------------------
    fc = load_feature_config()
    report_time = ["incident_id", "source_dataset", "city", "category", "subcategory",
                   "latitude", "longitude", "zone_id", "zone_type", "reported_at",
                   "report_channel"]
    feat = out[[c for c in report_time if c in out.columns]].copy()

    time_cols = list(fc.get("temporal_features", {}).keys()) + \
        list(fc.get("temporal_policy", {}).get("metadata_columns", []))
    for c in time_cols:
        feat[c] = out[c] if c in out.columns else pd.NA

    geo_cols = []
    for block in ("poi_proximity_features", "road_network_features",
                  "area_context_features", "historical_density_features",
                  "historical_recency_features"):
        geo_cols += list(fc.get(block, {}).keys())
    for c in geo_cols:
        feat[c] = out[c] if c in out.columns else pd.NA

    for c in ["priority_baseline", "priority_score", "priority_reasons",
              "priority_confidence", "priority_features_available",
              "priority_feature_coverage", "priority_method",
              "priority_label", "priority_label_source", "is_ground_truth",
              "sla_hours_policy"]:
        feat[c] = out[c]

    # The dataset states its own target situation, so a consumer cannot mistake
    # "empty by necessity" for "not built yet".
    manifest = _target_manifest(out)
    feat["target_status"] = manifest["priority"]["status"]
    feat["target_strategy"] = manifest["priority"]["strategy"]

    # leakage-resistant chronological split (per source; see dataset_config.yaml)
    split, split_meta = chronological_split(out)
    feat["split"] = split

    dest = ensure_dir(p("processed", "priority", "priority_dataset.parquet"))
    feat.to_parquet(dest, index=False)

    ts = pd.to_datetime(out["reported_at"], utc=True, errors="coerce")
    dist = out["priority_baseline"].value_counts().to_dict()
    conf = out["priority_confidence"].value_counts().to_dict()
    avail = {c: int(feat[c].notna().sum()) for c in geo_cols}
    time_avail = {c: int(feat[c].notna().sum()) for c in time_cols if c in feat.columns}
    split_profile = {}
    for s, g in feat.groupby("split"):
        gts = ts.loc[g.index]
        split_profile[str(s)] = {
            "rows": int(len(g)),
            "date_range": {"min": str(gts.min()), "max": str(gts.max())},
            "by_source": {str(k): int(v) for k, v in g["source_dataset"].value_counts().items()},
            "by_city": {str(k): int(v) for k, v in g["city"].value_counts().items()},
        }

    summary = {
        "config_version": engine.version,
        "config_status": engine.status,
        "rows": int(len(out)),
        "inputs": {"incidents": str(src), "time_features": time_source, "geo_features": geo_source},
        "priority_baseline_distribution": {str(k): int(v) for k, v in dist.items()},
        "priority_confidence_distribution": {str(k): int(v) for k, v in conf.items()},
        "priority_features_available_distribution": {
            str(k): int(v) for k, v in out["priority_features_available"].value_counts().items()},
        "priority_feature_coverage": {
            "min": float(out["priority_feature_coverage"].min()),
            "median": float(out["priority_feature_coverage"].median()),
            "max": float(out["priority_feature_coverage"].max()),
            "weighted_inputs_declared": len(engine.w),
        },
        "priority_score_distribution": {
            q: float(pd.to_numeric(out["priority_score"], errors="coerce").quantile(v))
            for q, v in [("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95)]},
        "priority_label_non_null": int(out["priority_label"].notna().sum()),
        "is_ground_truth": False,
        "target": manifest,
        "sla_hours_policy": engine.sla,
        "geo_feature_availability": avail,
        "time_feature_availability": time_avail,
        "split": split_meta,
        "split_counts": {str(k): int(v) for k, v in feat["split"].value_counts().items()},
        "split_profile": split_profile,
        "interpretation": (
            "priority_baseline is a deterministic function of config/priority_config.yaml. "
            "It is NOT ground truth and must not be used as an ML target — a model trained "
            "on it would simply re-learn the YAML. priority_label is NULL for every row "
            "because no public 311 dataset contains a genuine operational priority. "
            "POI, road-network and area-context inputs are NULL on US data, so the score "
            "rests on the category term plus the two backward-looking density terms; "
            "priority_confidence and priority_feature_coverage report exactly that."),
    }
    ensure_dir(p("reports", "priority_summary.json")).write_text(json.dumps(summary, indent=2))
    log.info("priority_baseline distribution: %s", dist)
    log.info("confidence distribution       : %s", conf)
    log.info("split counts                  : %s", summary["split_counts"])
    log.info("wrote %s and %s", inc_dest, dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
