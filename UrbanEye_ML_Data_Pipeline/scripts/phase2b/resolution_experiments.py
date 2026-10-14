#!/usr/bin/env python3
"""
Phase 2B — resolution-time experiments.

Every experiment is (feature set, model, protocol). Each one is recorded under
experiments/phase2b/resolution/ with its per-fold metrics and predictions,
and is skipped on re-run unless --force is given, so the log only ever
contains fits that actually happened.

    python scripts/phase2b/resolution_experiments.py --stage baselines
    python scripts/phase2b/resolution_experiments.py --stage screen
    python scripts/phase2b/resolution_experiments.py --stage models
    python scripts/phase2b/resolution_experiments.py --stage tune
    python scripts/phase2b/resolution_experiments.py --stage label_policy
    python scripts/phase2b/resolution_experiments.py --stage loco
    python scripts/phase2b/resolution_experiments.py --stage track_a
    python scripts/phase2b/resolution_experiments.py --stage all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.phase2b.common import (DATA_DIR, EXPERIMENT_DIR, REPORT_DIR, SEED, ExperimentLog,
                                    paired_bootstrap_mae_gain, resolution_metrics, sha256_file,
                                    write_json)
from scripts.phase2b.harness import loco_folds, lookup_predict, rolling_folds
from scripts.train._common import FeaturePreprocessor

TASK = "resolution"
TARGET = "resolution_time_hours"
OUT = EXPERIMENT_DIR / TASK
PRED_DIR = OUT / "preds"

TIME = ["hour", "day_of_week", "month", "is_weekend", "is_night"]
GEO250 = ["nearby_similar_incidents_24h", "nearby_similar_incidents_7d",
          "nearby_similar_incidents_30d", "local_incident_density",
          "category_incident_density", "hours_since_previous_similar_incident"]
WORKLOAD = ["wl_backlog_open", "wl_intake_7d", "wl_intake_30d", "wl_closed_7d",
            "wl_backlog_days", "wl_recent_closures_30d", "wl_recent_mean_log_res",
            "wl_recent_median_res_h", "wl_city_backlog_open", "wl_city_intake_7d"]
SPATIAL = ["sp_samecat_box750_7d", "sp_samecat_box750_30d", "sp_samecat_box750_90d",
           "sp_samecat_box750_open_now", "sp_all_box750_30d"]
ZONE = ["zone_cat_intake_30d", "zone_cat_backlog_open"]
NONDEPLOY = ["city", "department", "report_channel", "zone_key"]
P2A_RECIPE = ["category", "subcategory", "city", "source_dataset", "department", "zone_key",
              "zone_type", "report_channel", "is_weekend", "is_night", "hour", "day_of_week",
              "month", "year"] + GEO250

FEATURE_SETS = {
    "F0_p2a_recipe": P2A_RECIPE,
    "F1_base": ["model_category"] + TIME,
    "F2_base_geo250": ["model_category"] + TIME + GEO250,
    "F3_base_workload": ["model_category"] + TIME + WORKLOAD,
    "F3b_workload_no_month": ["model_category"] + [t for t in TIME if t != "month"] + WORKLOAD,
    "F4_workload_spatial": ["model_category"] + TIME + WORKLOAD + SPATIAL,
    # best deployable combination from the screen: spatial helps log-error and
    # median AE; month cannot extrapolate with < 1 year of clean history
    "F4b_workload_spatial_no_month": ["model_category"] + [t for t in TIME if t != "month"]
    + WORKLOAD + SPATIAL,
    "F5_workload_spatial_zone": ["model_category"] + TIME + WORKLOAD + SPATIAL + ZONE,
    "F6_upper_bound_nondeploy": ["model_category"] + TIME + WORKLOAD + NONDEPLOY,
}
DEPLOYABLE = {"F1_base", "F2_base_geo250", "F3_base_workload", "F3b_workload_no_month",
              "F4_workload_spatial", "F4b_workload_spatial_no_month"}
BEST_FS = "F4b_workload_spatial_no_month"

_HGB_C = dict(max_iter=500, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=20,
              l2_regularization=1.0)
MODELS = {
    # the Phase 2A configuration exactly, including its implicit early stopping
    "M0_p2a_C_l2log_autoES": dict(kind="hgb", loss="squared_error", es="auto", **_HGB_C),
    "M1_l2log": dict(kind="hgb", loss="squared_error", es=False, **_HGB_C),
    "M2_l1log": dict(kind="hgb", loss="absolute_error", es=False, **_HGB_C),
    "M3_extratrees": dict(kind="et", n_estimators=150, min_samples_leaf=30, max_features=0.5,
                          max_samples=0.3),
    # tuning (small, each answering one question)
    "T1_l1log_small": dict(kind="hgb", loss="absolute_error", es=False, max_iter=300,
                           learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100,
                           l2_regularization=1.0),
    "T2_l1log_big": dict(kind="hgb", loss="absolute_error", es=False, max_iter=800,
                         learning_rate=0.05, max_leaf_nodes=127, min_samples_leaf=50,
                         l2_regularization=1.0),
    "T3_l1log_slowreg": dict(kind="hgb", loss="absolute_error", es=False, max_iter=600,
                             learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=200,
                             l2_regularization=5.0),
}


# ---------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "resolution_v2.parquet")
    return df.reset_index(drop=True)


def dataset_info() -> dict:
    f = DATA_DIR / "resolution_v2.parquet"
    m = json.loads((DATA_DIR / "phase2b_datasets.manifest.json").read_text())
    return {"table": "data/processed/phase2b/resolution_v2.parquet", "sha256": sha256_file(f),
            "built_at_utc": m["built_at_utc"], "inputs": m["inputs"], "filters": m["filters"]}


def make_model(spec: dict, cat_mask):
    if spec["kind"] == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        kw = {k: v for k, v in spec.items() if k not in ("kind", "es")}
        return HistGradientBoostingRegressor(categorical_features=cat_mask, random_state=SEED,
                                             early_stopping=spec["es"], **kw)
    if spec["kind"] == "et":
        from sklearn.ensemble import ExtraTreesRegressor
        kw = {k: v for k, v in spec.items() if k != "kind"}
        return ExtraTreesRegressor(random_state=SEED, n_jobs=-1, bootstrap=True, **kw)
    raise ValueError(spec["kind"])


def fit_predict(spec, features, tr: pd.DataFrame, te: pd.DataFrame,
                quantiles: tuple[float, float] | None = None):
    pre = FeaturePreprocessor(features).fit(tr)
    Xtr, Xte = pre.transform(tr), pre.transform(te)
    if spec["kind"] == "et":
        Xtr, Xte = Xtr.fillna(-2.0), Xte.fillna(-2.0)
    y = np.log1p(tr[TARGET].to_numpy(float))
    m = make_model(spec, pre.categorical_mask)
    t0 = time.time()
    m.fit(Xtr, y)
    secs = time.time() - t0
    pred = np.clip(np.expm1(m.predict(Xte)), 0, None)
    lo = hi = None
    qmodels = {}
    if quantiles:
        from sklearn.ensemble import HistGradientBoostingRegressor
        base = {k: v for k, v in spec.items() if k not in ("kind", "es", "loss")}
        for name, q in zip(("lo", "hi"), quantiles):
            qm = HistGradientBoostingRegressor(loss="quantile", quantile=q, early_stopping=False,
                                               categorical_features=pre.categorical_mask,
                                               random_state=SEED, **base)
            qm.fit(Xtr, y)
            qmodels[name] = qm
        lo = np.clip(np.expm1(qmodels["lo"].predict(Xte)), 0, None)
        hi = np.clip(np.expm1(qmodels["hi"].predict(Xte)), 0, None)
    n_iter = getattr(m, "n_iter_", None)
    return pred, lo, hi, {"fit_seconds": round(secs, 1), "n_iter": n_iter,
                          "model": m, "pre": pre, "qmodels": qmodels}


def baseline_preds(tr: pd.DataFrame, te: pd.DataFrame, cat_col: str = "model_category") -> dict:
    glob = float(tr[TARGET].median())
    cat = lookup_predict(tr, te, [cat_col], TARGET)
    out = {
        "B0_global_median": np.full(len(te), glob),
        "B1_category_median": cat,
        "B2_category_x_city_median": lookup_predict(tr, te, [cat_col, "city"], TARGET),
        "B3_category_x_department_median": lookup_predict(tr, te, [cat_col, "department"],
                                                          TARGET),
        # non-ML, leak-free: median duration of the same category's closures in
        # the 30 days before the report, in the same city; category median if
        # the window is too thin.
        "B4_recent_30d_category_median": te["wl_recent_median_res_h"].fillna(
            pd.Series(cat, index=te.index)).to_numpy(float),
    }
    return out


def score(te: pd.DataFrame, pred, lo=None, hi=None) -> dict:
    y = te[TARGET].to_numpy(float)
    ue = te["is_ue_supported"].to_numpy(bool)
    out = {"all": resolution_metrics(y, pred, lo, hi)}
    if ue.any():
        out["ue_supported"] = resolution_metrics(
            y[ue], np.asarray(pred)[ue], None if lo is None else lo[ue],
            None if hi is None else hi[ue])
    return out


def summarise(per_fold: dict, subset: str = "ue_supported") -> dict:
    keys = ["MAE", "median_AE", "MAE_log1p", "RMSE", "R2", "interval_coverage"]
    rows = [f[subset] for f in per_fold.values() if subset in f]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys if rows and k in rows[0]}


# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, force: bool = False):
        self.df = load()
        self.log = ExperimentLog(TASK, dataset_info())
        self.force = force
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        # reload previously recorded experiments so the flushed log is complete
        for f in sorted(OUT.glob("*.json")):
            self.log.records.append(json.loads(f.read_text()))

    def _done(self, exp_id):
        return (not self.force) and (OUT / f"{exp_id}.json").exists()

    def _drop(self, exp_id):
        self.log.records = [r for r in self.log.records if r["experiment_id"] != exp_id]

    def run(self, exp_id: str, fs: str, model: str, protocol: str = "rolling",
            label_policy: str = "known", quantiles=None, note: str = "") -> dict | None:
        if self._done(exp_id):
            print(f"skip {exp_id} (recorded)")
            return None
        self._drop(exp_id)
        features, spec = FEATURE_SETS[fs], MODELS[model]
        folds = (rolling_folds(self.df, label_policy) if protocol == "rolling"
                 else loco_folds(self.df))
        per_fold, preds, meta = {}, [], {}
        t0 = time.time()
        try:
            for name, trm, tem in folds:
                tr, te = self.df[trm], self.df[tem]
                pred, lo, hi, info = fit_predict(spec, features, tr, te, quantiles)
                per_fold[name] = score(te, pred, lo, hi)
                per_fold[name]["n_train"] = int(len(tr))
                per_fold[name]["train_period"] = [str(tr["reported_at"].min()),
                                                  str(tr["reported_at"].max())]
                per_fold[name]["test_period"] = [str(te["reported_at"].min()),
                                                 str(te["reported_at"].max())]
                per_fold[name]["train_cities"] = tr["source_dataset"].value_counts().to_dict()
                per_fold[name]["fit_seconds"] = info["fit_seconds"]
                per_fold[name]["n_iter"] = info["n_iter"]
                p = pd.DataFrame({"incident_id": te["incident_id"].to_numpy(), "fold": name,
                                  "pred": pred})
                if lo is not None:
                    p["lo"], p["hi"] = lo, hi
                preds.append(p)
                print(f"  {exp_id} {name}: MAE(ue)={per_fold[name].get('ue_supported', {}).get('MAE', float('nan')):.1f} "
                      f"MAE(all)={per_fold[name]['all']['MAE']:.1f} ({info['fit_seconds']}s)",
                      flush=True)
            pd.concat(preds).to_parquet(PRED_DIR / f"{exp_id}.parquet", index=False)
            status = "ok"
        except Exception as e:  # noqa: BLE001 — failures are recorded, not hidden
            status = f"failed: {e!r}"
            meta["traceback"] = traceback.format_exc()
            print(meta["traceback"])
        rec = self.log.record(
            exp_id, status=status, model_name=model, algorithm=spec["kind"],
            hyperparameters={k: v for k, v in spec.items() if k != "kind"},
            target=f"log1p({TARGET}); predictions expm1'd and clipped at 0",
            feature_set=fs, features=features, deployable=fs in DEPLOYABLE,
            protocol=protocol, label_policy=label_policy, quantiles=quantiles, note=note,
            per_fold=per_fold, summary_ue=summarise(per_fold, "ue_supported"),
            summary_all=summarise(per_fold, "all"), seconds=round(time.time() - t0, 1), **meta)
        self.log.flush()
        return rec

    def run_baselines(self, protocol: str = "rolling", label_policy: str = "known") -> None:
        exp_id = f"BASELINES_{protocol}_{label_policy}"
        if self._done(exp_id):
            print(f"skip {exp_id}")
            return
        self._drop(exp_id)
        folds = (rolling_folds(self.df, label_policy) if protocol == "rolling"
                 else loco_folds(self.df))
        results: dict = {}
        preds = []
        for name, trm, tem in folds:
            tr, te = self.df[trm], self.df[tem]
            bp = baseline_preds(tr, te)
            for b, pr in bp.items():
                results.setdefault(b, {})[name] = score(te, pr)
                preds.append(pd.DataFrame({"incident_id": te["incident_id"].to_numpy(),
                                           "fold": name, "baseline": b, "pred": pr}))
        pd.concat(preds).to_parquet(PRED_DIR / f"{exp_id}.parquet", index=False)
        self.log.record(exp_id, status="ok", algorithm="non-ML lookup", protocol=protocol,
                        label_policy=label_policy,
                        baselines={b: {"per_fold": v, "summary_ue": summarise(v, "ue_supported"),
                                       "summary_all": summarise(v, "all")}
                                   for b, v in results.items()})
        self.log.flush()
        for b, v in results.items():
            print(f"  {exp_id} {b}: MAE(ue)={summarise(v)['MAE']:.1f} "
                  f"medAE(ue)={summarise(v)['median_AE']:.1f}")


# ---------------------------------------------------------------------------
# Track A: the Phase 2A test rows, scored like-for-like
# ---------------------------------------------------------------------------
def track_a(force: bool = False) -> dict:
    import joblib
    from scripts.utils.paths import repo_root
    out_path = REPORT_DIR / "resolution_track_a.json"
    if out_path.exists() and not force:
        print("skip track_a")
        return json.loads(out_path.read_text())
    root = repo_root()
    old = pd.read_parquet(root / "data/processed/ml/resolution_ml.parquet")
    feats = pd.read_parquet(DATA_DIR / "incident_features_v2.parquet")
    v1_model = joblib.load(root / "models/resolution/model.joblib")
    v1_pre = joblib.load(root / "models/resolution/preprocessor.joblib")
    test = old[old["split_global"] == "test"].copy()
    val_start = old.loc[old["split_global"] == "val", "reported_at"].min()
    v1_pred = np.clip(np.expm1(v1_model.predict(v1_pre.transform(test))), 0, None)

    # the same test rows, enriched with Phase 2B features
    keep = ["incident_id", "model_category", "is_ue_supported", "is_duplicate",
            "valid_resolution"] + WORKLOAD + SPATIAL
    te = test[["incident_id", TARGET, "reported_at", "city", "department", "category"]].merge(
        feats[keep], on="incident_id", how="left", validate="1:1")
    te = te.merge(feats[["incident_id"] + TIME], on="incident_id", how="left")
    assert len(te) == len(test)

    # improved recipe, trained on the Phase 2A TRAIN period only, cleaned
    r = load()
    tr = r[r["reported_at"] < val_start]
    res = {"test_rows": int(len(te)), "train_rows_v2": int(len(tr)),
           "train_cutoff": str(val_start), "candidates": {}}
    subsets = {"all_phase2a_test_rows": np.ones(len(te), bool),
               "non_duplicate_rows": ~te["is_duplicate"].fillna(False).to_numpy(bool),
               "ue_supported_rows": te["is_ue_supported"].fillna(False).to_numpy(bool)}
    y = te[TARGET].to_numpy(float)

    preds = {"existing_v1_model": v1_pred}
    old_tr = old[old["split_global"] == "train"]
    preds["B0_global_median(p2a train)"] = np.full(len(te), float(old_tr[TARGET].median()))
    preds["B1_category_median(p2a train)"] = lookup_predict(old_tr, test, ["category"], TARGET)
    preds["B3_category_x_department_median(p2a train)"] = lookup_predict(
        old_tr, test, ["category", "department"], TARGET)
    preds["B4_recent_30d_category_median"] = te["wl_recent_median_res_h"].fillna(
        pd.Series(preds["B1_category_median(p2a train)"])).to_numpy(float)
    candidates = [("F3_base_workload", "M2_l1log"), ("F3_base_workload", "M1_l2log"),
                  ("F1_base", "M2_l1log"), (BEST_FS, "M2_l1log")]
    # the configuration train_final.py selected, if it has run
    gates = REPORT_DIR / "final_gates.json"
    if gates.exists():
        sel = json.loads(gates.read_text()).get("resolution", {}).get("candidate")
        if sel and "__" in sel:
            fs_sel, m_sel = sel.split("__")[:2]
            if (fs_sel, m_sel) not in candidates:
                candidates.append((fs_sel, m_sel))
    for fs, model in candidates:
        pred, _, _, info = fit_predict(MODELS[model], FEATURE_SETS[fs], tr, te)
        preds[f"v2_{fs}_{model}"] = pred
        print(f"  track_a {fs} {model}: {info['fit_seconds']}s")
    for name, pr in preds.items():
        res["candidates"][name] = {s: resolution_metrics(y[m], pr[m]) for s, m in subsets.items()}
    # paired bootstrap: each v2 candidate vs the existing model, all rows
    res["bootstrap_vs_existing"] = {
        name: paired_bootstrap_mae_gain(y, v1_pred, pr) for name, pr in preds.items()
        if name.startswith("v2_") or name.startswith("B4")}
    write_json(out_path, res)
    return res


# ---------------------------------------------------------------------------
STAGES = {
    "baselines": lambda R: (R.run_baselines("rolling", "known"),
                            R.run_baselines("rolling", "all"), R.run_baselines("loco", "all")),
    "screen": lambda R: [R.run(f"{fs}__M2_l1log", fs, "M2_l1log") for fs in FEATURE_SETS],
    "models": lambda R: [R.run(f"{BEST_FS}__{m}", BEST_FS, m)
                         for m in ("M1_l2log", "M0_p2a_C_l2log_autoES", "M3_extratrees")]
    + [R.run("F0_p2a_recipe__M0_p2a_C_l2log_autoES", "F0_p2a_recipe", "M0_p2a_C_l2log_autoES",
             note="the Phase 2A recipe (features + config) under the Phase 2B protocol"),
       R.run("F1_base__M1_l2log", "F1_base", "M1_l2log")],
    "tune": lambda R: [R.run(f"{BEST_FS}__{m}", BEST_FS, m)
                       for m in ("T1_l1log_small", "T2_l1log_big")],
    "label_policy": lambda R: [R.run(f"{BEST_FS}__M2_l1log__labels_all", BEST_FS, "M2_l1log",
                                     label_policy="all")],
    "loco": lambda R: [R.run(f"{fs}__M2_l1log__loco", fs, "M2_l1log", protocol="loco",
                             label_policy="all")
                       for fs in ("F1_base", BEST_FS, "F6_upper_bound_nondeploy")],
    "intervals": lambda R: R.run(f"{BEST_FS}__M2_l1log__q10q90", BEST_FS, "M2_l1log",
                                 quantiles=(0.1, 0.9)),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["track_a", "all"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.stage in ("track_a", "all"):
        track_a(args.force)
    if args.stage == "track_a":
        return 0
    R = Runner(args.force)
    for name, fn in STAGES.items():
        if args.stage in (name, "all"):
            print(f"== stage {name}")
            fn(R)
    R.log.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
