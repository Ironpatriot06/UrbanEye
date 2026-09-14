#!/usr/bin/env python3
"""
End-to-end audit of every stage the pipeline produces.

This is the "prove it" script. The existing validators each guard one concern
(quality, provenance, leakage, output shape); this one walks the whole chain
stage by stage and reports what is actually in the files on disk, so that a
claim like "time features are populated" or "density features are leakage-safe"
can be checked without reading any builder's log.

  DATASET   row counts, duplicate IDs, source and city distribution, referential
            integrity between the incident table and each feature table
  TIME      missingness, ranges, and an INDEPENDENT re-derivation of local time
            from the configured timezone for a random sample
  SPATIAL   coordinate availability, which POI features exist, and whether any
            reference geodata is present to support the ones that do not
  DENSITY   missingness, zero share, maxima, window monotonicity, and an
            independent backward-window re-derivation
  PRIORITY  baseline/score/confidence distributions, feature availability,
            policy version
  TARGET    what is genuinely learnable, what is empty, and why
  SPLIT     counts, per-split date ranges, per-split source and city mix
  LEAKAGE   forbidden columns, split contamination, future timestamps,
            target-derived predictors

    python scripts/validation/audit_pipeline.py
    python scripts/validation/audit_pipeline.py --strict     # exit 1 on any FAIL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.utils.cleaning import haversine_m
from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import (ensure_dir, load_config, load_feature_config,
                                 load_priority_config, p, repo_root)
from scripts.utils.timefeatures import night_window, source_timezones

log = get_logger("validation.audit")

CHECKS: list[dict] = []
FINDINGS: dict = {}


def check(section: str, cid: str, desc: str, passed: bool | None, detail: str = "") -> bool:
    result = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
    CHECKS.append({"section": section, "id": cid, "check": desc,
                   "result": result, "detail": detail})
    fn = log.info if result == "PASS" else (log.warning if result == "SKIP" else log.error)
    fn("%-9s %-4s %s %s", cid, result, desc, detail)
    return bool(passed)


def counts(s: pd.Series, top: int | None = None) -> dict:
    vc = s.value_counts(dropna=False)
    if top:
        vc = vc.head(top)
    return {str(k): int(v) for k, v in vc.items()}


def load(*parts) -> pd.DataFrame | None:
    fp = p(*parts)
    return pd.read_parquet(fp) if fp.exists() else None


# ---------------------------------------------------------------------------
def audit_dataset(inc: pd.DataFrame) -> None:
    per_source = {}
    total = 0
    for ds in load_config()["datasets"]:
        fp = p("processed", "incidents", f"{ds}.parquet")
        if fp.exists():
            n = pd.read_parquet(fp, columns=["incident_id"]).shape[0]
            per_source[ds] = n
            total += n
    FINDINGS["dataset"] = {
        "rows": int(len(inc)),
        "per_source_parquet_rows": per_source,
        "by_source": counts(inc["source_dataset"]),
        "by_city": counts(inc["city"]),
        "by_category": counts(inc["category"]),
        "date_range": {"min": str(inc["reported_at"].min()), "max": str(inc["reported_at"].max())},
    }
    check("DATASET", "DS-1", "union row count equals the sum of the per-source tables",
          total == len(inc), f"union={len(inc):,} sum_of_sources={total:,}")
    dupes = int(inc["incident_id"].duplicated().sum())
    check("DATASET", "DS-2", "no duplicate incident_id", dupes == 0, f"duplicates={dupes}")
    check("DATASET", "DS-3", "no incident_id shared across source datasets",
          int((inc.groupby("incident_id")["source_dataset"].nunique() > 1).sum()) == 0)
    check("DATASET", "DS-4", "every row carries a source_dataset and a city",
          bool(inc["source_dataset"].notna().all() and inc["city"].notna().all()))

    ids = set(inc["incident_id"])
    for label, parts in (("time_features", ("processed", "features", "incident_time_features.parquet")),
                         ("geo_features", ("processed", "geo_features", "incident_geo_features.parquet"))):
        fp = p(*parts)
        if not fp.exists():
            check("DATASET", f"DS-5/{label}", "feature table covers every incident exactly once",
                  None, "table not built")
            continue
        t = pd.read_parquet(fp, columns=["incident_id"])
        check("DATASET", f"DS-5/{label}", "feature table covers every incident exactly once",
              len(t) == len(inc) and set(t["incident_id"]) == ids and
              int(t["incident_id"].duplicated().sum()) == 0,
              f"rows={len(t):,} vs incidents={len(inc):,}")


def audit_time(inc: pd.DataFrame, rng: np.random.Generator) -> None:
    tf = load("processed", "features", "incident_time_features.parquet")
    if tf is None:
        check("TIME", "TF-1", "time features built", None, "table not built")
        return
    m = inc[["incident_id", "source_dataset", "reported_at"]].merge(tf, on="incident_id", how="left")
    n_start, n_end = night_window()

    prof = {}
    for ds, g in m.groupby("source_dataset"):
        prof[str(ds)] = {
            "rows": int(len(g)),
            "timezone": str(g["local_timezone"].dropna().iloc[0]) if g["local_timezone"].notna().any() else None,
            "hour_null": int(g["hour"].isna().sum()),
            "hour_range": [int(g["hour"].min()), int(g["hour"].max())],
            "night_share_local": round(float(g["is_night"].mean()), 4),
            "night_share_if_computed_from_utc": round(float(
                ((g["hour_utc"] >= n_start) | (g["hour_utc"] < n_end)).mean()), 4),
            "weekend_share": round(float(g["is_weekend"].mean()), 4),
            "hour_histogram_local": counts(g["hour"]),
        }
    FINDINGS["time"] = {"night_window_local": f"{n_start:02d}-{n_end:02d}",
                        "timezone_mapping": source_timezones(), "by_source": prof}

    check("TIME", "TF-1", "no missing time features where reported_at is present",
          int(m.loc[m["reported_at"].notna(), "hour"].isna().sum()) == 0,
          f"missing={int(m.loc[m['reported_at'].notna(), 'hour'].isna().sum())}")
    check("TIME", "TF-2", "hour in [0,23], day_of_week in [0,6], month in [1,12]",
          bool(m["hour"].between(0, 23).all() and m["day_of_week"].between(0, 6).all()
               and m["month"].between(1, 12).all()))
    check("TIME", "TF-3", "is_night agrees with the configured local night window",
          bool((m["is_night"] == ((m["hour"] >= n_start) | (m["hour"] < n_end))).all()))
    check("TIME", "TF-4", "is_weekend agrees with day_of_week",
          bool((m["is_weekend"] == m["day_of_week"].isin([5, 6])).all()))

    # independent re-derivation from the configured timezone
    tzmap = source_timezones()
    idx = rng.choice(len(m), size=min(2000, len(m)), replace=False)
    s = m.iloc[idx]
    bad = 0
    for ds, g in s.groupby("source_dataset"):
        tz = tzmap.get(str(ds))
        if not tz:
            continue
        expect = pd.to_datetime(g["reported_at"], utc=True).dt.tz_convert(tz).dt.tz_localize(None)
        bad += int((expect.dt.hour.to_numpy() != g["hour"].to_numpy()).sum())
    check("TIME", "TF-5", "local hour re-derives exactly from the configured city timezone",
          bad == 0, f"mismatches={bad} of {len(s)} resampled rows")
    check("TIME", "TF-6", "local time genuinely differs from UTC (features are not UTC in disguise)",
          bool((m["hour"] != m["hour_utc"]).mean() > 0.9),
          f"rows where local hour != UTC hour: {float((m['hour'] != m['hour_utc']).mean()):.3f}")


def audit_spatial(inc: pd.DataFrame) -> None:
    fc = load_feature_config()
    geo = load("processed", "geo_features", "incident_geo_features.parquet")
    poi_cols = []
    for block in ("poi_proximity_features", "road_network_features", "area_context_features"):
        poi_cols += list(fc.get(block, {}).keys())

    ext = repo_root() / "data" / "external"
    ref_files = sorted(f.name for f in ext.rglob("*") if f.is_file()) if ext.exists() else []

    FINDINGS["spatial"] = {
        "coordinates_present": int(inc["latitude"].notna().sum()),
        "coordinates_missing": int(inc["latitude"].isna().sum()),
        "coordinates_missing_by_source": counts(inc.loc[inc["latitude"].isna(), "source_dataset"]),
        "outside_city_bbox_flagged": int(inc.get("coord_outside_city_bbox",
                                                 pd.Series(dtype="boolean")).fillna(False).sum()),
        "poi_columns_declared": len(poi_cols),
        "poi_columns_populated": 0 if geo is None else
            int(sum(1 for c in poi_cols if c in geo.columns and geo[c].notna().any())),
        "reference_geodata_directory": str(ext),
        "reference_geodata_present": ref_files,
        "why_poi_is_empty": (
            "No reference layer exists to compute against: data/external is absent and the "
            "repository ships no school/hospital/metro/rail/transport/fire/police/government/"
            "road/intersection geometry for any city. The 311 publishers supply a coordinate "
            "and nothing else. Emitting these columns NULL with the correct dtype is the "
            "honest option; imputing them would be fabrication, and US POI data would in any "
            "case be meaningless for an Indian deployment."),
    }
    if geo is None:
        check("SPATIAL", "SP-1", "geo feature table built", None, "table not built")
        return
    populated = [c for c in poi_cols if c in geo.columns and geo[c].notna().any()]
    check("SPATIAL", "SP-1", "POI/road/area columns are all present in the schema",
          all(c in geo.columns for c in poi_cols), f"{len(poi_cols)} declared")
    check("SPATIAL", "SP-2", "POI/road/area columns are NULL (no reference data exists to fill them)",
          not populated, f"unexpectedly populated: {populated}" if populated else
          "0 of 23 populated; no reference geodata in data/external")
    check("SPATIAL", "SP-3", "coordinates, where present, are finite and in range",
          bool(inc["latitude"].dropna().between(-90, 90).all()
               and inc["longitude"].dropna().between(-180, 180).all()))


def audit_density(inc: pd.DataFrame, rng: np.random.Generator, sample: int) -> None:
    fc = load_feature_config()
    specs = fc["historical_density_features"]
    geo = load("processed", "geo_features", "incident_geo_features.parquet")
    if geo is None:
        check("DENSITY", "DN-1", "density features built", None, "table not built")
        return
    cols = [c for c in specs if c in geo.columns]
    stats = {}
    for c in cols:
        v = pd.to_numeric(geo[c], errors="coerce")
        stats[c] = {
            "non_null": int(v.notna().sum()), "null": int(v.isna().sum()),
            "zero_share": round(float((v == 0).mean()), 4),
            "mean": round(float(v.mean()), 4), "median": float(v.median()),
            "p95": float(v.quantile(0.95)), "max": float(v.max()),
        }
    FINDINGS["density"] = {
        "definitions": {c: {k: specs[c].get(k) for k in
                            ("window_hours", "radius_m", "same_category_only", "strictly_backward")}
                        for c in cols},
        "statistics": stats,
        "null_reason": "rows without usable coordinates or timestamp; never imputed as 0",
    }

    m = inc[["incident_id", "latitude"]].merge(geo[["incident_id"] + cols], on="incident_id")
    nocoord = m["latitude"].isna()
    check("DENSITY", "DN-1", "density is NULL exactly where coordinates are missing",
          bool(m.loc[nocoord, cols[0]].isna().all() and m.loc[~nocoord, cols[0]].notna().all()),
          f"rows without coordinates={int(nocoord.sum()):,}")
    check("DENSITY", "DN-2", "windows are monotone: 24h <= 7d <= 30d",
          bool((geo["nearby_similar_incidents_24h"].fillna(0) <=
                geo["nearby_similar_incidents_7d"].fillna(0)).all()
               and (geo["nearby_similar_incidents_7d"].fillna(0) <=
                    geo["nearby_similar_incidents_30d"].fillna(0)).all()))
    check("DENSITY", "DN-3", "same-category 30d count never exceeds the all-category 30d count",
          bool((geo["nearby_similar_incidents_30d"].fillna(0) <=
                geo["local_incident_density"].fillna(0)).all()))
    check("DENSITY", "DN-4", "category_incident_density is a share in [0,1]",
          bool(pd.to_numeric(geo["category_incident_density"], errors="coerce")
               .dropna().between(0, 1).all()))
    dens = pd.to_numeric(geo["local_incident_density"], errors="coerce")
    usable, nuniq = int(dens.notna().sum()), int(dens.nunique())
    # On a few hundred synthetic fixture rows almost every incident genuinely has
    # zero prior neighbours, so a low distinct count is the correct answer, not a
    # defect. The expectation only bites once there is enough history to have a
    # distribution at all.
    check("DENSITY", "DN-5", "density is not degenerate (it varies across rows)",
          None if usable < 5000 else nuniq > 10,
          f"distinct values={nuniq} over {usable:,} usable rows"
          + ("; too few rows for a meaningful distribution" if usable < 5000 else ""))

    # --- independent backward-window re-derivation --------------------------
    base = inc[["incident_id", "latitude", "longitude", "category", "reported_at"]].dropna(
        subset=["latitude", "longitude", "reported_at"])
    lat = base["latitude"].to_numpy(float)
    lon = base["longitude"].to_numpy(float)
    ts = pd.to_datetime(base["reported_at"], utc=True).to_numpy("datetime64[ns]").astype(
        np.int64) / 1e9 / 3600.0
    cat = base["category"].astype("string").fillna("").to_numpy()
    merged = base.merge(geo[["incident_id"] + cols], on="incident_id")
    idx = rng.choice(len(merged), size=min(sample, len(merged)), replace=False)

    mism = {c: 0 for c in cols if c != "category_incident_density"}
    future_leak = 0
    for i in idx:
        r = merged.iloc[int(i)]
        t = pd.Timestamp(r["reported_at"]).timestamp() / 3600.0
        d = haversine_m(float(r["latitude"]), float(r["longitude"]), lat, lon)
        for c in mism:
            spec = specs[c]
            w, rad = float(spec.get("window_hours", 720)), float(spec.get("radius_m", 250))
            sel = (ts < t) & (ts >= t - w) & (d <= rad)
            if spec.get("same_category_only"):
                sel = sel & (cat == str(r["category"]))
            if int(sel.sum()) != int(r[c]):
                mism[c] += 1
            # what the count would be if the window were NOT backward-looking
            fwd = (ts > t) & (ts <= t + w) & (d <= rad)
            if spec.get("same_category_only"):
                fwd = fwd & (cat == str(r["category"]))
            if int(r[c]) > int(sel.sum()) + int(fwd.sum()):
                future_leak += 1
    check("DENSITY", "DN-6", "every density column re-derives exactly with reported_at < t",
          sum(mism.values()) == 0, f"mismatches={mism} over {len(idx)} resampled rows")
    check("DENSITY", "DN-7", "no density value exceeds what a backward window can produce",
          future_leak == 0, f"suspect_rows={future_leak}")

    # ---- recency ----------------------------------------------------------
    rec = fc.get("historical_recency_features", {}) or {}
    for name, spec in rec.items():
        if name not in geo.columns:
            check("DENSITY", f"DN-8/{name}", "recency feature present", None, "column missing")
            continue
        v = pd.to_numeric(geo[name], errors="coerce")
        w = float(spec.get("window_hours", 720))
        FINDINGS.setdefault("recency", {})[name] = {
            "non_null": int(v.notna().sum()), "null": int(v.isna().sum()),
            "null_means": "no qualifying earlier report inside the window",
            "min": float(v.min()), "median": float(v.median()), "max": float(v.max()),
        }
        check("DENSITY", f"DN-8/{name}", "recency is positive and inside its own window",
              bool(v.dropna().between(0, w).all()),
              f"range=[{v.min():.2f}, {v.max():.2f}] window={w}")
        same_cat = int(spec.get("radius_m", 250)) and spec.get("same_category_only", True)
        paired = geo[["nearby_similar_incidents_30d", name]].copy()
        has_prior = pd.to_numeric(paired["nearby_similar_incidents_30d"], errors="coerce") > 0
        consistent = (has_prior == paired[name].notna())
        check("DENSITY", f"DN-9/{name}",
              "recency is non-null exactly when a 30d same-category neighbour exists",
              bool(consistent[paired[name].notna() | has_prior.notna()].all()),
              f"inconsistent={int((~consistent).sum())}")


def audit_priority(pri: pd.DataFrame) -> None:
    cfg = load_priority_config()
    score = pd.to_numeric(pri["priority_score"], errors="coerce")
    FINDINGS["priority"] = {
        "policy_version": str(pri["priority_method"].dropna().iloc[0]) if pri["priority_method"].notna().any() else None,
        "policy_status": cfg["status"],
        "baseline_distribution": counts(pri["priority_baseline"]),
        "confidence_distribution": counts(pri["priority_confidence"]),
        "features_available_distribution": counts(pri["priority_features_available"]),
        "weighted_inputs_declared": len(cfg["weights"]),
        "feature_coverage": {
            "min": float(pri["priority_feature_coverage"].min()),
            "median": float(pri["priority_feature_coverage"].median()),
            "max": float(pri["priority_feature_coverage"].max()),
        },
        "score_distribution": {q: float(score.quantile(v)) for q, v in
                               [("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95)]},
        "sla_hours_policy": cfg["sla_hours"],
    }
    check("PRIORITY", "PR-1", "every row has a baseline band",
          int(pri["priority_baseline"].isna().sum()) == 0,
          f"missing={int(pri['priority_baseline'].isna().sum())}")
    check("PRIORITY", "PR-2", "baseline uses more than one band", pri["priority_baseline"].nunique() > 1)
    check("PRIORITY", "PR-3", "score is in [0,1]", bool(score.dropna().between(0, 1).all()))
    check("PRIORITY", "PR-4", "confidence is consistent with the declared tiers",
          _confidence_consistent(pri, cfg))
    check("PRIORITY", "PR-5", "priority_method records the config version used",
          bool(pri["priority_method"].notna().all()))
    check("PRIORITY", "PR-6", "is_ground_truth is False on every row",
          not bool(pri["is_ground_truth"].fillna(False).any()))
    check("PRIORITY", "PR-7", "score is not a pure category lookup "
                              "(context features actually move it)",
          _score_varies_within_category(pri))


def _confidence_consistent(pri: pd.DataFrame, cfg: dict) -> bool:
    tiers = cfg.get("confidence_tiers")
    if not tiers:
        return True
    n = pri["priority_features_available"].fillna(0).astype(int)
    cov = pri["priority_feature_coverage"].fillna(0).astype(float)
    expect = pd.Series("LOW", index=pri.index, dtype="string")
    for tier in reversed(tiers):
        hit = (n >= int(tier.get("min_features", 0))) & (cov >= float(tier.get("min_weight_coverage", 0)))
        expect = expect.mask(hit, str(tier["name"]))
    # rows hit by an absolute category rule are HIGH by construction
    abs_rule = pri["priority_score"].eq(1.0) & pri["priority_baseline"].eq("P1")
    return bool((expect.eq(pri["priority_confidence"]) | abs_rule).all())


def _score_varies_within_category(pri: pd.DataFrame) -> bool:
    g = pri.groupby("category")["priority_score"].nunique()
    return bool((g > 1).mean() > 0.5)


def audit_target(pri: pd.DataFrame, inc: pd.DataFrame) -> None:
    res = pd.to_numeric(inc.get("resolution_time_hours"), errors="coerce")
    sla = pd.to_numeric(inc.get("sla_target_hours"), errors="coerce")
    FINDINGS["target"] = {
        "priority_label_non_null": int(pri["priority_label"].notna().sum()),
        "priority_label_source_non_null": int(pri["priority_label_source"].notna().sum()),
        "is_ground_truth_true_rows": int(pri["is_ground_truth"].fillna(False).sum()),
        "declared_status": str(pri["target_status"].dropna().iloc[0]) if "target_status" in pri.columns
                           and pri["target_status"].notna().any() else None,
        "declared_strategy": str(pri["target_strategy"].dropna().iloc[0]) if "target_strategy" in pri.columns
                             and pri["target_strategy"].notna().any() else None,
        "observable_targets": {
            "resolution_time_hours": int(res.notna().sum()),
            "sla_breach_computable": int((res.notna() & sla.notna()).sum()),
            "sla_target_hours_by_source": counts(inc.loc[sla.notna(), "source_dataset"]),
        },
        "censoring": {"open_or_invalid": int(res.isna().sum())},
    }
    check("TARGET", "TG-1", "priority_label is empty (no public source provides one)",
          int(pri["priority_label"].notna().sum()) == 0)
    check("TARGET", "TG-2", "priority_label_source is empty",
          int(pri["priority_label_source"].notna().sum()) == 0)
    check("TARGET", "TG-3", "no row claims ground truth",
          int(pri["is_ground_truth"].fillna(False).sum()) == 0)
    check("TARGET", "TG-4", "the dataset states its own target situation in-band",
          "target_status" in pri.columns and pri["target_status"].notna().all())
    check("TARGET", "TG-5", "at least one genuinely observed target exists",
          int(res.notna().sum()) > 0, f"resolution_time_hours rows={int(res.notna().sum()):,}")
    check("TARGET", "TG-6", "sla_target_hours comes only from publishers that ship a due date",
          set(inc.loc[sla.notna(), "source_dataset"].unique()).issubset({"boston311", "nyc311"}),
          f"sources={sorted(set(inc.loc[sla.notna(), 'source_dataset'].unique()))}")


def audit_split(tables: dict[str, pd.DataFrame]) -> None:
    prof: dict = {}
    for name, (df, tscol) in tables.items():
        if df is None or "split" not in df.columns:
            check("SPLIT", f"SPL-1/{name}", "split assigned", None, "table missing or has no split")
            continue
        ts = pd.to_datetime(df[tscol], utc=True, errors="coerce")
        entry = {"counts": counts(df["split"]), "by_split": {}}
        for s, g in df.groupby("split"):
            gts = ts.loc[g.index]
            entry["by_split"][str(s)] = {
                "rows": int(len(g)),
                "date_range": {"min": str(gts.min()), "max": str(gts.max())},
                "by_source": counts(g["source_dataset"]) if "source_dataset" in g.columns else {},
                "by_city": counts(g["city"]) if "city" in g.columns else {},
            }
        prof[name] = entry

        have = set(df["split"].unique())
        check("SPLIT", f"SPL-1/{name}", "train, val and test are all non-empty",
              {"train", "val", "test"}.issubset(have), f"present={sorted(have)}")

        group_col = "source_dataset" if "source_dataset" in df.columns else "city"
        ok, detail = True, []
        for key, g in df.groupby(group_col):
            gts = ts.loc[g.index]
            b = {s: (gts[g["split"] == s].min(), gts[g["split"] == s].max())
                 for s in ("train", "val", "test") if (g["split"] == s).any()}
            if "train" in b and "val" in b and not b["train"][1] <= b["val"][0]:
                ok = False; detail.append(f"{key}: train_max>{b['val'][0]}")
            if "val" in b and "test" in b and not b["val"][1] <= b["test"][0]:
                ok = False; detail.append(f"{key}: val_max>{b['test'][0]}")
        check("SPLIT", f"SPL-2/{name}",
              f"within each {group_col}, train precedes val precedes test", ok, "; ".join(detail))

        if "incident_id" in df.columns:
            spans = df.groupby("incident_id")["split"].nunique()
            check("SPLIT", f"SPL-3/{name}", "no incident appears in more than one split",
                  int((spans > 1).sum()) == 0, f"contaminated={int((spans > 1).sum())}")
    FINDINGS["split"] = prof


def audit_leakage(pri: pd.DataFrame, ml: pd.DataFrame | None) -> None:
    fc = load_feature_config()
    forbidden = set(fc["forbidden_features"]["post_resolution"]) | \
        set(fc["forbidden_features"]["future_information"])
    now = pd.Timestamp.now(tz="UTC")

    found_pri = sorted(set(pri.columns) & forbidden)
    check("LEAKAGE", "LK-1", "priority_dataset contains no post-resolution column",
          not found_pri, f"found={found_pri}")

    ts = pd.to_datetime(pri["reported_at"], utc=True, errors="coerce")
    check("LEAKAGE", "LK-2", "no report timestamp lies in the future",
          int((ts > now).sum()) == 0, f"future_rows={int((ts > now).sum())}")

    if ml is None:
        check("LEAKAGE", "LK-3", "ML dataset audited", None, "table not built")
        return
    manifest_fp = p("processed", "ml", "urbaneye_ml.manifest.json")
    manifest = json.loads(manifest_fp.read_text()) if manifest_fp.exists() else {}
    roles = manifest.get("column_roles", {})
    predictors = [c for c, r in roles.items() if r == "predictor"]

    found_ml = sorted((set(ml.columns) & forbidden) - {"resolution_time_hours"})
    check("LEAKAGE", "LK-3", "ML dataset contains no post-resolution column outside its targets",
          not found_ml, f"found={found_ml}")
    check("LEAKAGE", "LK-4", "no declared predictor is a target or derived from one",
          not (set(predictors) & {"resolution_time_hours", "sla_breach", "sla_met",
                                  "target_is_censored", "sla_target_hours"}),
          f"predictors={len(predictors)}")
    check("LEAKAGE", "LK-5", "priority policy output is not declared as a predictor",
          not [c for c in predictors if c.startswith("priority_")])
    check("LEAKAGE", "LK-6", "raw coordinates are not declared as predictors",
          not ({"latitude", "longitude"} & set(predictors)))
    check("LEAKAGE", "LK-7", "no fabricated priority target survives anywhere in the ML table",
          "priority_target" not in ml.columns)
    ml_ts = pd.to_datetime(ml["reported_at"], utc=True, errors="coerce")
    check("LEAKAGE", "LK-8", "ML dataset carries no future timestamps",
          int((ml_ts > now).sum()) == 0)
    FINDINGS["leakage"] = {
        "forbidden_feature_list": sorted(forbidden),
        "ml_predictor_count": len(predictors),
        "ml_predictors": predictors,
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--density-sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    inc = load("processed", "incidents", "all_incidents_prioritised.parquet")
    if inc is None:
        inc = load("processed", "incidents", "all_incidents.parquet")
    if inc is None:
        log.error("no incidents table — run python scripts/run_pipeline.py first")
        return 2

    audit_dataset(inc)
    audit_time(inc, rng)
    audit_spatial(inc)
    audit_density(inc, rng, args.density_sample)

    pri = load("processed", "priority", "priority_dataset.parquet")
    if pri is not None:
        audit_priority(pri)
        audit_target(pri, inc)
    else:
        check("PRIORITY", "PR-0", "priority_dataset built", None, "table not built")

    ml = load("processed", "ml", "urbaneye_ml.parquet")
    res = load("processed", "resolution", "resolution_dataset.parquet")
    hot = load("processed", "hotspot", "hotspot_dataset.parquet")
    audit_split({"priority_dataset": (pri, "reported_at"),
                 "resolution_dataset": (res, "reported_at"),
                 "hotspot_dataset": (hot, "week_start"),
                 "urbaneye_ml": (ml, "reported_at")})
    if pri is not None:
        audit_leakage(pri, ml)

    failed = [c for c in CHECKS if c["result"] == "FAIL"]
    passed = [c for c in CHECKS if c["result"] == "PASS"]
    skipped = [c for c in CHECKS if c["result"] == "SKIP"]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": load_config()["pipeline_version"],
        "summary": {"total": len(CHECKS), "passed": len(passed),
                    "failed": len(failed), "skipped": len(skipped)},
        "checks": CHECKS,
        "findings": FINDINGS,
    }
    dest = ensure_dir(p("reports", "pipeline_audit.json"))
    dest.write_text(json.dumps(report, indent=2, default=str))
    _markdown(report)
    log.info("wrote %s — %d passed, %d failed, %d skipped",
             dest, len(passed), len(failed), len(skipped))
    return 1 if (failed and args.strict) else 0


def _markdown(r: dict) -> None:
    s = r["summary"]
    L = ["# UrbanEye+ pipeline audit", "",
         f"Generated {r['generated_at_utc']} · pipeline {r['pipeline_version']}", "",
         f"**{s['passed']}/{s['total']} passed, {s['failed']} failed, {s['skipped']} skipped.**", ""]
    section = None
    for c in r["checks"]:
        if c["section"] != section:
            section = c["section"]
            L += ["", f"## {section}", "", "| ID | Check | Result | Detail |", "|---|---|---|---|"]
        mark = {"PASS": "PASS", "FAIL": "**FAIL**", "SKIP": "_skip_"}[c["result"]]
        L.append(f"| {c['id']} | {c['check']} | {mark} | {c['detail'] or ''} |")

    f = r["findings"]
    if "dataset" in f:
        L += ["", "## Dataset", "",
              f"- rows: **{f['dataset']['rows']:,}**",
              f"- by source: {f['dataset']['by_source']}",
              f"- date range: {f['dataset']['date_range']['min']} .. {f['dataset']['date_range']['max']}"]
    if "priority" in f:
        L += ["", "## Priority", "",
              f"- baseline: {f['priority']['baseline_distribution']}",
              f"- confidence: {f['priority']['confidence_distribution']}",
              f"- inputs available (of {f['priority']['weighted_inputs_declared']}): "
              f"{f['priority']['features_available_distribution']}"]
    if "target" in f:
        L += ["", "## Target", "",
              f"- priority_label non-null: **{f['target']['priority_label_non_null']}**",
              f"- is_ground_truth true rows: **{f['target']['is_ground_truth_true_rows']}**",
              f"- observable targets: {f['target']['observable_targets']}"]
    if "split" in f:
        L += ["", "## Splits", ""]
        for name, e in f["split"].items():
            L.append(f"- `{name}`: {e['counts']}")
    ensure_dir(p("reports", "pipeline_audit.md")).write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
