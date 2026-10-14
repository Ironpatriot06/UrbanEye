#!/usr/bin/env python3
"""
Phase 2C — export the Phase 2B v2 artifacts as a self-contained serving bundle.

WHY A BUNDLE
------------
The backend cannot unpickle `models/<task>_v2/preprocessor.joblib`: the pickle
references `scripts.train._common.FeaturePreprocessor`, a module of this
pipeline (and `backend/scripts/` would shadow the `scripts` package). The
preprocessor is a plain ordinal encoder, so it is exported as JSON and the
backend re-implements its ~20-line transform. The boosters themselves are pure
scikit-learn objects and are copied BYTE FOR BYTE (their SHA-256 in the
manifest equals the Phase 2B artifact's), so nothing is retrained or re-fitted.

Phase 2B artifacts are read only. An existing bundle directory is never
overwritten.

CONTENTS  (<out>/<bundle_version>/)
    MANIFEST.json                 versions, SHA-256 of every file, provenance
    resolution/model.joblib       {point, q10, q90} HistGradientBoostingRegressor
    resolution/preprocessor.json  encoder (predictors, vocabularies)
    resolution/metadata.json      Phase 2B metadata, unchanged
    sla/model.joblib              HistGradientBoostingClassifier
    sla/preprocessor.json
    sla/metadata.json
    hotspot/metadata.json         EWMA settings (non-ML)
    baselines.json                training-data fallback baselines (US 311)
    feature_ranges.json           p01/p99 of every numeric input on training rows
    fixtures/golden_predictions.json  inputs + outputs of the ORIGINAL pipeline
    fixtures/feature_parity.json      synthetic incidents + features computed by
                                      scripts/phase2b/features.py

    python scripts/phase2c/export_serving_bundle.py [--out ../backend/ml_bundle]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.phase2b import features as F
from scripts.phase2b import resolution_experiments as RX
from scripts.phase2b import sla_experiments as SX
from scripts.phase2b.common import (ROOT, UE_SUPPORTED, environment, now_utc, sha256_file,
                                    write_json)
from scripts.phase2b.harness import ORIGINS

BUNDLE_VERSION = "urbaneye-ml-2026.10.13"
FINAL_CUT = ORIGINS[-1]
SEED = 20260915
DAY = 1440.0
PARITY_HORIZONS = [4, 12, 24, 72, 168]
# history features exactly as named in training
HISTORY_FEATURES = RX.WORKLOAD + RX.SPATIAL


# ---------------------------------------------------------------------------
def preprocessor_json(pre) -> dict:
    return {"class": "scripts.train._common.FeaturePreprocessor (exported)",
            "predictors": list(pre.predictors), "categorical": list(pre.categorical_),
            "numeric": list(pre.numeric_), "vocabulary": pre.vocabulary_,
            "max_levels": pre.max_levels, "missing_token": "__MISSING__",
            "other_token": "__OTHER__", "unknown_code": -1.0,
            "rules": ["numeric: float, NaN if missing or non-numeric",
                      "categorical: str(value) ('True'/'False' for booleans); missing -> "
                      "'__MISSING__'; not in vocabulary -> '__OTHER__'; code = index in "
                      "vocabulary, or -1 if the folded value is not in it"]}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# training-data baselines and input ranges (same rows the final models used)
# ---------------------------------------------------------------------------
def resolution_training_rows() -> pd.DataFrame:
    r = RX.load()
    return r[(r["reported_at"] < FINAL_CUT) & r["closed_at"].notna() & (r["closed_at"] < FINAL_CUT)]


def sla_training_rows() -> pd.DataFrame:
    s = SX.load()
    st = SX.stack_horizons(s[s["reported_at"] < FINAL_CUT])
    return st[((st["reported_at"] + pd.to_timedelta(st["sla_hours"], unit="h")) < FINAL_CUT).to_numpy()]


def baselines(rt: pd.DataFrame, st: pd.DataFrame) -> dict:
    ue = rt[rt["is_ue_supported"]]
    med = ue.groupby("ue_category")[RX.TARGET].agg(["median", "size"])
    rate = st.groupby(["ue_category", "sla_hours"])[SX.TARGET].agg(["mean", "size"])
    return {
        "source": "Phase 2B training rows known at 2020-09-01 (Chicago, San Francisco, New York "
                  "City 311). US data: these are NOT UrbanEye statistics.",
        "resolution_category_median_hours": {
            c: {"median_hours": float(v["median"]), "n": int(v["size"])}
            for c, v in med.iterrows()},
        "sla_category_breach_rate": {
            c: {str(int(h)): {"breach_rate": float(rate.loc[(c, h), "mean"]),
                              "n": int(rate.loc[(c, h), "size"])}
                for h in SX.HORIZONS if (c, h) in rate.index}
            for c in UE_SUPPORTED},
    }


def feature_ranges(rt: pd.DataFrame, st: pd.DataFrame, rmeta: dict, smeta: dict) -> dict:
    out = {}
    for task, rows, meta in (("resolution", rt[rt["is_ue_supported"]], rmeta), ("sla", st, smeta)):
        rng = {}
        for c in meta["numeric_features"]:
            v = pd.to_numeric(rows[c], errors="coerce").dropna()
            if len(v):
                rng[c] = {"p01": float(v.quantile(0.01)), "p99": float(v.quantile(0.99)),
                          "missing_share": float(rows[c].isna().mean())}
        out[task] = rng
    out["note"] = ("1st-99th percentile of each numeric input over the rows the final model was "
                   "trained on (UrbanEye categories). An input outside this range is outside the "
                   "evaluated regime; it does not by itself make a prediction wrong.")
    return out


# ---------------------------------------------------------------------------
# golden predictions: the ORIGINAL preprocessor + model, on real and edge inputs
# ---------------------------------------------------------------------------
def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def golden_predictions(rng: np.random.Generator) -> dict:
    import joblib
    out = {}
    # ---- resolution
    d = ROOT / "models" / "resolution_v2"
    models, pre = joblib.load(d / "model.joblib"), joblib.load(d / "preprocessor.joblib")
    feats = pre.predictors
    r = RX.load()
    rows = r[r["is_ue_supported"]].sample(150, random_state=SEED)[feats].copy()
    edge = rows.head(40).copy()
    mask = rng.random(edge.shape) < 0.4
    for j, c in enumerate(feats):
        if c not in ("model_category",):
            edge.loc[mask[:, j], c] = pd.NA if edge[c].dtype.name in ("boolean", "string", "Int64") else np.nan
    edge.iloc[:5, feats.index("model_category")] = "GRAFFITI_VISUAL_POLLUTION"
    edge.iloc[5:8, feats.index("model_category")] = "FIRE_HAZARD"          # unseen -> -1
    edge.iloc[8:10, feats.index("model_category")] = pd.NA                 # __MISSING__ -> -1
    allrows = pd.concat([rows, edge], ignore_index=True)
    X = pre.transform(allrows)
    cases = []
    for i in range(len(allrows)):
        cases.append({"features": {c: _clean(allrows.iloc[i][c]) for c in feats},
                      "encoded": [None if np.isnan(x) else float(x) for x in X.iloc[i].to_numpy()],
                      "raw_point": float(models["point"].predict(X.iloc[[i]])[0]),
                      "raw_q10": float(models["q10"].predict(X.iloc[[i]])[0]),
                      "raw_q90": float(models["q90"].predict(X.iloc[[i]])[0])})
    out["resolution"] = {"features": feats, "cases": cases}

    # ---- SLA
    d = ROOT / "models" / "sla_v2"
    model, pre = joblib.load(d / "model.joblib"), joblib.load(d / "preprocessor.joblib")
    feats = pre.predictors
    s = SX.load()
    rows = s.sample(150, random_state=SEED).copy()
    rows["sla_hours"] = rng.choice([4.0, 12.0, 24.0, 72.0, 168.0], len(rows))
    rows = rows[feats]
    edge = rows.head(40).copy()
    mask = rng.random(edge.shape) < 0.4
    for j, c in enumerate(feats):
        if c not in ("ue_category", "sla_hours"):
            edge.loc[mask[:, j], c] = pd.NA if edge[c].dtype.name in ("boolean", "string", "Int64") else np.nan
    edge.iloc[:3, feats.index("ue_category")] = "OTHER"
    allrows = pd.concat([rows, edge], ignore_index=True)
    X = pre.transform(allrows)
    prob = model.predict_proba(X)[:, 1]
    out["sla"] = {"features": feats, "cases": [
        {"features": {c: _clean(allrows.iloc[i][c]) for c in feats},
         "encoded": [None if np.isnan(x) else float(x) for x in X.iloc[i].to_numpy()],
         "probability": float(prob[i])} for i in range(len(allrows))]}
    return out


# ---------------------------------------------------------------------------
# feature parity: synthetic incidents through scripts/phase2b/features.py
# ---------------------------------------------------------------------------
def feature_parity(rng: np.random.Generator) -> dict:
    """
    One synthetic service area, ~600 incidents over 150 days, four categories,
    some never closed, some closed in < 1 minute, two in the same minute. The
    expected features are computed exactly as build_datasets.py does, from the
    FULL table (closures after t included): the backend must ignore anything not
    observable at t and still reproduce every value.
    """
    area = {"name": "PARITY_TEST", "lat_min": 12.90, "lat_max": 13.00, "lon_min": 77.55,
            "lon_max": 77.65, "ref_lat": 12.95}
    n = 600
    t0 = pd.Timestamp("2026-01-05T03:17:00Z")
    rep = np.sort(rng.uniform(0, 150 * DAY, n))
    rep[101] = rep[100]                                       # same-minute tie
    cats = rng.choice(["POTHOLE", "GARBAGE", "STREETLIGHT", "FIRE_HAZARD"], n, p=[.4, .3, .2, .1])
    lat = rng.uniform(12.93, 12.97, n)
    lon = rng.uniform(77.58, 77.62, n)
    dur_h = rng.lognormal(3.5, 1.6, n)
    clo = rep + dur_h * 60.0
    never = rng.random(n) < 0.08
    clo[never] = np.nan
    instant = rng.random(n) < 0.03
    clo[instant] = rep[instant] + 0.2                         # 12 s: not a valid resolution
    # microsecond precision: the backend parses these with datetime.fromisoformat
    created = [(t0 + pd.Timedelta(minutes=float(m))).round("us") for m in rep]
    closed = [None if np.isnan(c) else (t0 + pd.Timedelta(minutes=float(c))).round("us")
              for c in clo]

    # --- exactly build_datasets.py, with one source and model_category = category
    repm = F.to_minutes(pd.Series(created))
    clom = F.to_minutes(pd.Series([pd.NaT if c is None else c for c in closed]))
    res = (clom - repm) / 60.0
    valid = ~np.isnan(clom) & (res >= 1.0 / 60.0) & (res <= 87_600.0)
    warm = np.full(n, repm.min())
    g_cat = F.encode_groups(pd.Series(["A"] * n), pd.Series(cats))
    g_city = F.encode_groups(pd.Series(["A"] * n))
    wl = F.workload_features(g_cat, repm, clom, np.where(valid, res, np.nan), warmup_start_min=warm)
    wlc = F.workload_features(g_city, repm, clom, np.where(valid, res, np.nan), warmup_start_min=warm)
    wl["wl_city_backlog_open"] = wlc["wl_backlog_open"]
    wl["wl_city_intake_7d"] = wlc["wl_intake_7d"]
    is_open = np.isnan(clom)
    dur = np.where(np.isnan(clom), np.inf, (clom - repm) / 60.0)
    elig = is_open | valid
    rates = {h: F.recent_breach_rate(g_cat, repm, dur, is_open, h, elig) for h in PARITY_HORIZONS}
    ix, iy = F.grid_cells(lat, lon, np.full(n, area["lat_min"]), np.full(n, area["lon_min"]),
                          np.full(n, area["ref_lat"]), 250.0)
    exit_min = np.fmin(np.where(np.isnan(clom), np.inf, clom), repm + 90 * DAY)
    sc = F.spatial_counts(g_cat, ix, iy, repm, [7, 30, 90], closed_min=exit_min,
                          prefix="sp_samecat_box750")
    sa = F.spatial_counts(g_city, ix, iy, repm, [30], prefix="sp_all_box750")
    sp = pd.concat([sc, sa], axis=1)
    young90 = (repm - warm) < 90 * DAY
    young30 = (repm - warm) < 30 * DAY
    sp.loc[young90, "sp_samecat_box750_90d"] = np.nan
    for c in ["sp_samecat_box750_7d", "sp_samecat_box750_30d", "sp_samecat_box750_open_now",
              "sp_all_box750_30d"]:
        sp.loc[young30, c] = np.nan

    queries = sorted(set(rng.choice(n, 120, replace=False).tolist()) | {0, 1, 100, 101, n - 1})
    expected = []
    for i in queries:
        f = {c: _clean(float(wl[c].iloc[i])) for c in RX.WORKLOAD}
        f.update({c: _clean(float(sp[c].iloc[i])) for c in RX.SPATIAL})
        expected.append({"index": int(i), "features": f,
                         "hist_breach_rate": {str(h): _clean(float(rates[h][i]))
                                              for h in PARITY_HORIZONS}})
    return {"area": area, "cell_m": 250.0, "horizons": PARITY_HORIZONS,
            "incidents": [{"id": int(k + 1), "category": str(cats[k]),
                           "created_at": created[k].isoformat(),
                           "latitude": float(lat[k]), "longitude": float(lon[k]),
                           "first_closed_at": None if closed[k] is None else closed[k].isoformat()}
                          for k in range(n)],
            "expected": expected,
            "generated_by": "scripts/phase2b/features.py via scripts/phase2c/export_serving_bundle.py"}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "backend" / "ml_bundle"))
    a = ap.parse_args()
    out = os.path.join(a.out, BUNDLE_VERSION)
    if os.path.exists(out):
        print(f"refusing to overwrite existing bundle {out}")
        return 1
    warnings.filterwarnings("ignore", category=UserWarning)
    import joblib
    rng = np.random.default_rng(SEED)
    tmp = out + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    src = {}
    for task in ("resolution", "sla"):
        d = ROOT / "models" / f"{task}_v2"
        os.makedirs(os.path.join(tmp, task))
        shutil.copyfile(d / "model.joblib", os.path.join(tmp, task, "model.joblib"))
        shutil.copyfile(d / "metadata.json", os.path.join(tmp, task, "metadata.json"))
        pre = joblib.load(d / "preprocessor.joblib")
        write_json(os.path.join(tmp, task, "preprocessor.json"), preprocessor_json(pre))
        src[task] = {f: sha256_file(d / f) for f in ("model.joblib", "preprocessor.joblib",
                                                     "metadata.json")}
    os.makedirs(os.path.join(tmp, "hotspot"))
    shutil.copyfile(ROOT / "models" / "hotspot_v2" / "metadata.json",
                    os.path.join(tmp, "hotspot", "metadata.json"))
    src["hotspot"] = {"metadata.json": sha256_file(ROOT / "models" / "hotspot_v2" / "metadata.json")}

    rmeta = json.loads((ROOT / "models/resolution_v2/metadata.json").read_text())
    smeta = json.loads((ROOT / "models/sla_v2/metadata.json").read_text())
    rt, st = resolution_training_rows(), sla_training_rows()
    write_json(os.path.join(tmp, "baselines.json"), baselines(rt, st))
    write_json(os.path.join(tmp, "feature_ranges.json"), feature_ranges(rt, st, rmeta, smeta))
    os.makedirs(os.path.join(tmp, "fixtures"))
    write_json(os.path.join(tmp, "fixtures", "golden_predictions.json"), golden_predictions(rng))
    write_json(os.path.join(tmp, "fixtures", "feature_parity.json"), feature_parity(rng))

    files = {}
    for dirpath, _, names in os.walk(tmp):
        for nm in sorted(names):
            pth = os.path.join(dirpath, nm)
            files[os.path.relpath(pth, tmp)] = sha256_file(pth)
    env = environment()
    manifest = {
        "bundle_version": BUNDLE_VERSION, "created_at_utc": now_utc(),
        "models": {"resolution": rmeta["model_version"], "sla": smeta["model_version"],
                   "hotspot": json.loads((ROOT / "models/hotspot_v2/metadata.json")
                                         .read_text())["model_version"]},
        "training_data_through": "2020-08-28", "training_cities": ["Chicago", "San Francisco",
                                                                   "New York City"],
        "runtime_requirements": {"python": env["python"], "numpy": env["numpy"],
                                 "scikit_learn": env["scikit_learn"], "joblib": env["joblib"],
                                 "scipy": env["scipy"]},
        "source_artifacts_sha256": src, "files_sha256": files,
        "pipeline_git_commit": _git_commit(),
        "exported_by": "UrbanEye_ML_Data_Pipeline/scripts/phase2c/export_serving_bundle.py",
        "warning": "joblib files are pickles: load only after verifying files_sha256.",
    }
    write_json(os.path.join(tmp, "MANIFEST.json"), manifest)
    os.rename(tmp, out)
    print(f"wrote {out}: {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
