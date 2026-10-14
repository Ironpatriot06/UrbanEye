#!/usr/bin/env python3
"""
Phase 2B — build the feature table and the v2 task tables.

Inputs (read-only, produced by the existing pipeline):
    data/processed/incidents/all_incidents.parquet
    data/processed/features/incident_time_features.parquet
    data/processed/geo_features/incident_geo_features.parquet
    config/urbaneye_category_mapping.csv

Outputs (gitignored, regenerable):
    data/processed/phase2b/incident_features_v2.parquet   every incident + features
    data/processed/phase2b/resolution_v2.parquet          resolution task table
    data/processed/phase2b/sla_v2.parquet                 UrbanEye-SLA task table
    data/processed/phase2b/*.manifest.json                hashes, filters, counts

Features are computed over ALL incidents (any status, any category), because
history is history: a cancelled or out-of-scope report still occupied the
queue. Targets are then defined on cleaned subsets.

    python scripts/phase2b/build_datasets.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.phase2b.common import (CATEGORY_DEFAULT_PRIORITY, DATA_DIR, MAX_RESOLUTION_HOURS,
                                    MIN_RESOLUTION_HOURS, UE_SUPPORTED, clean_period_mask,
                                    default_sla_hours, environment, map_to_urbaneye, now_utc,
                                    sha256_file, write_json)
from scripts.phase2b.features import (encode_groups, grid_cells, recent_breach_rate,
                                      spatial_counts, to_minutes, workload_features)
from scripts.utils.paths import p

INCIDENT_COLS = ["incident_id", "source_dataset", "city", "category", "subcategory",
                 "source_category_raw", "department", "report_channel", "zone_id", "zone_type",
                 "latitude", "longitude", "reported_at", "closed_at", "status",
                 "resolution_time_hours", "is_duplicate", "resolution_instant_closure"]
TIME_COLS = ["hour", "day_of_week", "month", "year", "is_weekend", "is_night"]
GEO250_COLS = ["nearby_similar_incidents_24h", "nearby_similar_incidents_7d",
               "nearby_similar_incidents_30d", "local_incident_density",
               "category_incident_density", "hours_since_previous_similar_incident"]
NOT_TARGETABLE = {"OUT_OF_SCOPE", "UNMAPPED", "REVIEW_REQUIRED"}
SLA_HORIZONS = [4, 24, 72, 168]
CELL_M = 250.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_base() -> tuple[pd.DataFrame, dict]:
    src = {
        "incidents": p("processed", "incidents", "all_incidents.parquet"),
        "time_features": p("processed", "features", "incident_time_features.parquet"),
        "geo_features": p("processed", "geo_features", "incident_geo_features.parquet"),
    }
    d = pd.read_parquet(src["incidents"], columns=INCIDENT_COLS)
    tf = pd.read_parquet(src["time_features"], columns=["incident_id"] + TIME_COLS)
    gf = pd.read_parquet(src["geo_features"], columns=["incident_id"] + GEO250_COLS)
    d = d.merge(tf, on="incident_id", how="left", validate="1:1")
    d = d.merge(gf, on="incident_id", how="left", validate="1:1")
    prov = {k: {"path": str(v.relative_to(p("processed").parent.parent)),
                "sha256": sha256_file(v)} for k, v in src.items()}
    return d, prov


def main() -> int:
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    d, prov = load_base()
    log(f"loaded {len(d):,} incidents")

    # ---- category alignment ------------------------------------------------
    ue = map_to_urbaneye(d)
    d["ue_category"] = ue["ue_category"]
    d["ue_confidence"] = ue["ue_confidence"]
    supported = d["ue_category"].isin(UE_SUPPORTED).fillna(False)
    d["is_ue_supported"] = supported
    # model_category: the UrbanEye category where one exists, otherwise the
    # canonical category (auxiliary training data the deployed model never
    # serves). Out-of-scope rows keep their label so they still count as history.
    d["model_category"] = d["ue_category"].where(supported, d["category"]).astype("string")
    d["zone_key"] = (d["city"].astype("string") + ":" + d["zone_id"].astype("string"))

    # ---- times -------------------------------------------------------------
    rep = to_minutes(d["reported_at"])
    clo = to_minutes(d["closed_at"])
    bad = ~np.isnan(clo) & (clo < rep)          # closed before reported: leave at once
    clo[bad] = rep[bad]
    dup = d["is_duplicate"].fillna(False).to_numpy(bool)
    res = d["resolution_time_hours"].to_numpy(float)
    valid_res = ((d["status"] == "closed").fillna(False).to_numpy(bool) & ~dup
                 & (res >= MIN_RESOLUTION_HOURS) & (res <= MAX_RESOLUTION_HOURS))
    d["valid_resolution"] = valid_res
    src_start = d.groupby("source_dataset")["reported_at"].transform("min")
    warm = to_minutes(src_start)

    # ---- workload: source x model_category ---------------------------------
    g_cat = encode_groups(d["source_dataset"], d["model_category"])
    wl = workload_features(g_cat, rep, clo, np.where(valid_res, res, np.nan),
                           warmup_start_min=warm)
    for c in wl.columns:
        d[c] = wl[c].to_numpy()
    # all-category workload of the whole city (crew contention across categories)
    g_city = encode_groups(d["source_dataset"])
    wl_city = workload_features(g_city, rep, clo, np.where(valid_res, res, np.nan),
                                warmup_start_min=warm)
    d["wl_city_backlog_open"] = wl_city["wl_backlog_open"].to_numpy()
    d["wl_city_intake_7d"] = wl_city["wl_intake_7d"].to_numpy()
    log("workload features done")

    # ---- recent SLA outcome rates (non-duplicate reference population) -----
    is_open = d["closed_at"].isna().to_numpy() & (d["status"] == "open").fillna(False).to_numpy()
    dur = np.where(np.isnan(clo), np.inf, (clo - rep) / 60.0)
    eligible = ~dup & (is_open | valid_res)
    for h in SLA_HORIZONS:
        d[f"hist_breach_rate_{h}h"] = recent_breach_rate(g_cat, rep, dur, is_open, h, eligible)
    log("recent breach rates done")

    # ---- zone-level frequency (research only: UrbanEye has no zones) -------
    g_zone = encode_groups(d["source_dataset"], d["zone_key"], d["model_category"])
    zf = workload_features(g_zone, rep, clo, np.full(len(d), np.nan), warmup_start_min=warm)
    d["zone_cat_intake_30d"] = zf["wl_intake_30d"].to_numpy()
    d["zone_cat_backlog_open"] = zf["wl_backlog_open"].to_numpy()

    # ---- spatial grid features (250 m cells, 3x3 block ≈ 750 m square) -----
    lat, lon = d["latitude"].to_numpy(float), d["longitude"].to_numpy(float)
    olat = d.groupby("source_dataset")["latitude"].transform("min").to_numpy(float)
    olon = d.groupby("source_dataset")["longitude"].transform("min").to_numpy(float)
    rlat = d.groupby("source_dataset")["latitude"].transform("median").to_numpy(float)
    ix, iy = grid_cells(lat, lon, olat, olon, rlat, CELL_M)
    exit_min = np.fmin(np.where(np.isnan(clo), np.inf, clo), rep + 90 * 1440.0)
    sc = spatial_counts(g_cat, ix, iy, rep, [7, 30, 90], closed_min=exit_min,
                        prefix="sp_samecat_box750")
    sa = spatial_counts(g_city, ix, iy, rep, [30], prefix="sp_all_box750")
    for frame in (sc, sa):
        for c in frame.columns:
            d[c] = frame[c].to_numpy()
    young = (rep - warm) < 90 * 1440.0
    d.loc[young, "sp_samecat_box750_90d"] = np.nan
    young30 = (rep - warm) < 30 * 1440.0
    for c in ["sp_samecat_box750_7d", "sp_samecat_box750_30d", "sp_samecat_box750_open_now",
              "sp_all_box750_30d"]:
        d.loc[young30, c] = np.nan
    log("spatial features done")

    feat_path = DATA_DIR / "incident_features_v2.parquet"
    d.to_parquet(feat_path, index=False)
    log(f"wrote {feat_path} ({len(d):,} rows)")

    # ---- resolution_v2 -----------------------------------------------------
    clean = clean_period_mask(d)
    r_mask = (d["valid_resolution"] & clean & ~d["model_category"].isin(NOT_TARGETABLE)
              & d["model_category"].notna())
    r = d[r_mask].copy()
    r["resolution_time_hours_log1p"] = np.log1p(r["resolution_time_hours"])
    r_path = DATA_DIR / "resolution_v2.parquet"
    r.to_parquet(r_path, index=False)
    log(f"resolution_v2: {len(r):,} rows")

    # ---- sla_v2: UrbanEye's SLA on UrbanEye-mapped rows ---------------------
    extract_end = d.groupby("source_dataset")["closed_at"].transform("max")
    s_base = supported & clean & ~d["is_duplicate"].fillna(False)
    closed_ok = d["valid_resolution"]
    open_ok = pd.Series(is_open, index=d.index)
    s = d[s_base & (closed_ok | open_ok)].copy()
    s["sla_hours"] = s["ue_category"].map(default_sla_hours).astype(float)
    s["priority_level"] = s["ue_category"].map(CATEGORY_DEFAULT_PRIORITY)
    deadline = s["reported_at"] + pd.to_timedelta(s["sla_hours"], unit="h")
    open_rows = open_ok[s.index]
    # an open case is a breach only once its deadline has passed by the extract end
    s = s[~open_rows | (deadline < extract_end[s.index])].copy()
    open_rows = open_ok[s.index]
    s["is_open_at_extract"] = open_rows.to_numpy()
    s["sla_breach_ue"] = (open_rows | (s["resolution_time_hours"] >= s["sla_hours"])).astype(bool)
    s_path = DATA_DIR / "sla_v2.parquet"
    s.to_parquet(s_path, index=False)
    log(f"sla_v2: {len(s):,} rows, breach rate {s['sla_breach_ue'].mean():.3f}")

    manifest = {
        "built_at_utc": now_utc(), "environment": environment(), "inputs": prov,
        "mapping": {"file": "config/urbaneye_category_mapping.csv",
                    "sha256": sha256_file(p("processed").parent.parent / "config" /
                                          "urbaneye_category_mapping.csv")},
        "filters": {
            "chicago_clean_from": "2019-03-01 (targets only; earlier rows remain history)",
            "valid_resolution": "status closed, not is_duplicate, 1 min <= duration <= 87,600 h",
            "resolution_categories": "model_category not in OUT_OF_SCOPE/UNMAPPED/REVIEW_REQUIRED",
            "sla_population": ("UrbanEye-mapped rows, not duplicates, clean period; closed with a "
                               "valid duration OR open with the deadline passed by the extract end"),
            "sla_target": "resolution_time_hours >= sla_hours(default priority of UE category), "
                          "or still open past the deadline (backend: breach at/after deadline)",
        },
        "rows": {"incident_features_v2": int(len(d)), "resolution_v2": int(len(r)),
                 "sla_v2": int(len(s))},
        "resolution_by_source": r["source_dataset"].value_counts().to_dict(),
        "resolution_ue_supported_rows": int(r["is_ue_supported"].sum()),
        "sla_by_category": s.groupby("ue_category")["sla_breach_ue"].agg(["size", "mean"])
                            .round(4).to_dict(orient="index"),
        "sla_by_source": s.groupby("source_dataset")["sla_breach_ue"].agg(["size", "mean"])
                          .round(4).to_dict(orient="index"),
        "outputs": {k: {"path": str(v.name), "sha256": sha256_file(v)}
                    for k, v in (("incident_features_v2", feat_path), ("resolution_v2", r_path),
                                 ("sla_v2", s_path))},
        "seconds": round(time.time() - t0, 1),
    }
    write_json(DATA_DIR / "phase2b_datasets.manifest.json", manifest)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
