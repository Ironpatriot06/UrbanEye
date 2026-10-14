#!/usr/bin/env python3
"""
Phase 2B — SLA experiments on UrbanEye's own SLA definition.

TARGET (data/processed/phase2b/sla_v2.parquet, built by build_datasets.py)
    sla_breach_ue = resolution_time_hours >= sla_hours, or still open after the
    deadline, where sla_hours is UrbanEye's default for the mapped category's
    default priority (FLOOD P2 24 h; POTHOLE, STREETLIGHT P3 72 h; GARBAGE P4
    168 h). This mirrors backend/app/services/incident_service.py: a breach is
    resolution at or after created_at + sla_hours, and an open incident past
    its deadline is BREACHED. The Phase 2A target (NYC agency Due Date, open
    cases dropped) is a different question and is not reused.

PROTOCOL
    rolling origin (harness.ORIGINS), training rows restricted to those whose
    deadline had passed before the origin; leave-one-city-out for transfer.
    Threshold-free metrics (PR-AUC, ROC-AUC, Brier) are the selection
    criteria. Thresholded metrics are reported at 0.5 (equal-cost decision on a
    calibrated probability) and, for the final model only, at a threshold
    chosen on a validation window that precedes the test window.

    python scripts/phase2b/sla_experiments.py --stage all
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

from scripts.phase2b.common import (DATA_DIR, EXPERIMENT_DIR, SEED, ExperimentLog, sha256_file)
from scripts.phase2b.harness import loco_folds, lookup_predict, rolling_folds
from scripts.phase2b.resolution_experiments import GEO250, NONDEPLOY, SPATIAL, TIME, WORKLOAD
from scripts.train._common import FeaturePreprocessor, calibration_bins, classification_metrics

TASK = "sla"
TARGET = "sla_breach_ue"
OUT = EXPERIMENT_DIR / TASK
PRED_DIR = OUT / "preds"
HORIZONS = [4, 24, 72, 168]

HIST = ["hist_breach_rate_own"]
FEATURE_SETS = {
    # the Phase 2A SLA recipe, minus the NYC-only sla_target_hours (replaced
    # by UrbanEye's sla_hours) and with the canonical category
    "S0_p2a_recipe": ["category", "subcategory", "department", "zone_key", "report_channel",
                      "is_weekend", "is_night", "hour", "day_of_week", "month", "sla_hours"]
    + GEO250,
    "S1_base": ["ue_category"] + TIME,
    "S2_base_hist": ["ue_category"] + TIME + HIST,
    "S3_base_hist_workload": ["ue_category"] + TIME + HIST + WORKLOAD,
    "S4_hist_workload_spatial": ["ue_category"] + TIME + HIST + WORKLOAD + SPATIAL,
    "S5_upper_bound_nondeploy": ["ue_category"] + TIME + HIST + WORKLOAD + NONDEPLOY,
    # multi-horizon: sla_hours varies within category, so it can be learned
    "S6_multihorizon": ["ue_category", "sla_hours"] + TIME + HIST + WORKLOAD,
    # second screen: workload alone hurt (S3 < S2) while spatial helped (S4 > S3)
    "S4b_hist_spatial": ["ue_category"] + TIME + HIST + SPATIAL,
    "S7_multihorizon_spatial": ["ue_category", "sla_hours"] + TIME + HIST + WORKLOAD + SPATIAL,
}
DEPLOYABLE = {"S1_base", "S2_base_hist", "S3_base_hist_workload", "S4_hist_workload_spatial",
              "S6_multihorizon", "S4b_hist_spatial", "S7_multihorizon_spatial"}
MULTI = {"S6_multihorizon", "S7_multihorizon_spatial"}
MODELS = {
    "C0_p2a_A_default": dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                             min_samples_leaf=20, l2_regularization=0.0, early_stopping="auto"),
    "C1_slow": dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                    l2_regularization=1.0, early_stopping=False),
    "C2_slow_balanced": dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                             min_samples_leaf=100, l2_regularization=1.0, early_stopping=False,
                             class_weight="balanced"),
    "C3_small": dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
                     l2_regularization=1.0, early_stopping=False),
}


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "sla_v2.parquet").reset_index(drop=True)
    df["hist_breach_rate_own"] = own_rate(df, df["sla_hours"])
    return df


def own_rate(df: pd.DataFrame, hours: pd.Series) -> np.ndarray:
    out = np.full(len(df), np.nan)
    for h in HORIZONS:
        m = (hours == h).to_numpy()
        out[m] = df.loc[m, f"hist_breach_rate_{h}h"].to_numpy(float)
    return out


def stack_horizons(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (incident, horizon); label = not resolved within X hours."""
    parts = []
    dur = df["resolution_time_hours"].to_numpy(float)
    is_open = df["is_open_at_extract"].to_numpy(bool)
    for h in HORIZONS:
        x = df.copy()
        x["sla_hours"] = float(h)
        x[TARGET] = is_open | (dur >= h)
        x["hist_breach_rate_own"] = df[f"hist_breach_rate_{h}h"].to_numpy(float)
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def dataset_info() -> dict:
    f = DATA_DIR / "sla_v2.parquet"
    m = json.loads((DATA_DIR / "phase2b_datasets.manifest.json").read_text())
    return {"table": "data/processed/phase2b/sla_v2.parquet", "sha256": sha256_file(f),
            "built_at_utc": m["built_at_utc"], "filters": m["filters"]}


def fit_predict(params: dict, features, tr, te):
    """Fit on tr, return P(breach) for te. `sla_hours` (if present) is constrained
    so that a longer deadline can never raise the predicted breach risk."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    pre = FeaturePreprocessor(features).fit(tr)
    mono = [(-1 if c == "sla_hours" else 0) for c in features] if "sla_hours" in features \
        and tr["sla_hours"].nunique() > 1 else None
    m = HistGradientBoostingClassifier(categorical_features=pre.categorical_mask,
                                       random_state=SEED, monotonic_cst=mono, **params)
    t0 = time.time()
    m.fit(pre.transform(tr), tr[TARGET].astype(int).to_numpy())
    prob = m.predict_proba(pre.transform(te))[:, 1]
    return prob, {"fit_seconds": round(time.time() - t0, 1),
                  "n_iter": int(getattr(m, "n_iter_", -1)), "model": m, "pre": pre}


def score(te: pd.DataFrame, prob: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import log_loss
    y = te[TARGET].astype(int).to_numpy()
    m = classification_metrics(y, prob, threshold)
    m["log_loss"] = float(log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6), labels=[0, 1]))
    m["flagged_rate"] = float((prob >= threshold).mean())
    m["by_category"] = {}
    for c, g in te.groupby("ue_category", observed=True):
        idx = te.index.get_indexer(g.index)
        yy = y[idx]
        cm = {"n": int(len(idx)), "positive_rate": float(yy.mean()),
              "mean_prob": float(prob[idx].mean())}
        if 0 < yy.sum() < len(yy):
            from sklearn.metrics import average_precision_score, roc_auc_score
            cm["PR_AUC"] = float(average_precision_score(yy, prob[idx]))
            cm["ROC_AUC"] = float(roc_auc_score(yy, prob[idx]))
        m["by_category"][str(c)] = cm
    return m


def baseline_probs(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    rate = float(tr[TARGET].mean())
    trc = tr.assign(_y=tr[TARGET].astype(float))
    cat = lookup_predict(trc, te, ["ue_category"], "_y", agg="mean")
    return {
        "B0_base_rate": np.full(len(te), rate),
        "B1_category_rate": cat,     # == priority rule: priority is fixed per category
        "B2_category_x_city_rate": lookup_predict(trc, te, ["ue_category", "city"], "_y",
                                                  agg="mean"),
        # non-ML, leak-free: breach share of the same category in the same city
        # among incidents whose deadline expired in the 30 days before t
        "B3_recent_violation_rate": te["hist_breach_rate_own"].fillna(
            pd.Series(cat, index=te.index)).to_numpy(float),
    }


def summarise(per_fold: dict) -> dict:
    keys = ["PR_AUC", "ROC_AUC", "brier", "log_loss", "precision", "recall", "f1",
            "positive_rate"]
    rows = list(per_fold.values())
    return {k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys}


class Runner:
    def __init__(self, force=False):
        self.df = load()
        self.log = ExperimentLog(TASK, dataset_info())
        self.force = force
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        for f in sorted(OUT.glob("*.json")):
            self.log.records.append(json.loads(f.read_text()))

    def _skip(self, exp_id):
        if (OUT / f"{exp_id}.json").exists() and not self.force:
            print(f"skip {exp_id}")
            return True
        self.log.records = [r for r in self.log.records if r["experiment_id"] != exp_id]
        return False

    def folds(self, protocol):
        if protocol == "rolling":
            return rolling_folds(self.df, "known", sla_hours_col="sla_hours")
        return loco_folds(self.df)

    def run(self, exp_id, fs, model, protocol="rolling", note=""):
        if self._skip(exp_id):
            return
        feats, params = FEATURE_SETS[fs], MODELS[model]
        multi = fs in MULTI
        per_fold, preds, extra = {}, [], {}
        t0 = time.time()
        try:
            for name, trm, tem in self.folds(protocol):
                tr, te = self.df[trm], self.df[tem].reset_index(drop=True)
                if multi and protocol == "rolling":
                    # stacked rows must also satisfy the known-label rule at their horizon
                    origin = te["reported_at"].min().normalize()
                    st = stack_horizons(tr)
                    ok = (st["reported_at"] + pd.to_timedelta(st["sla_hours"], unit="h")) < origin
                    prob, info = fit_predict(params, feats, st[ok.to_numpy()], te)
                elif multi:
                    prob, info = fit_predict(params, feats, stack_horizons(tr), te)
                else:
                    prob, info = fit_predict(params, feats, tr, te)
                per_fold[name] = score(te, prob)
                per_fold[name].update(n_train=int(len(tr)), fit_seconds=info["fit_seconds"],
                                      n_iter=info["n_iter"],
                                      train_period=[str(tr["reported_at"].min()),
                                                    str(tr["reported_at"].max())],
                                      test_period=[str(te["reported_at"].min()),
                                                   str(te["reported_at"].max())],
                                      calibration=calibration_bins(te[TARGET].astype(int), prob))
                if multi:
                    per_fold[name]["by_horizon"] = self._score_horizons(info, feats, te)
                preds.append(pd.DataFrame({"incident_id": te["incident_id"], "fold": name,
                                           "prob": prob, "y": te[TARGET].astype(int)}))
                print(f"  {exp_id} {name}: PR-AUC={per_fold[name]['PR_AUC']:.4f} "
                      f"ROC={per_fold[name]['ROC_AUC']:.4f} brier={per_fold[name]['brier']:.4f} "
                      f"({info['fit_seconds']}s)", flush=True)
            pd.concat(preds).to_parquet(PRED_DIR / f"{exp_id}.parquet", index=False)
            status = "ok"
        except Exception as e:  # noqa: BLE001
            status = f"failed: {e!r}"
            extra["traceback"] = traceback.format_exc()
            print(extra["traceback"])
        self.log.record(exp_id, status=status, model_name=model,
                        algorithm="HistGradientBoostingClassifier", hyperparameters=params,
                        feature_set=fs, features=feats, deployable=fs in DEPLOYABLE,
                        protocol=protocol, note=note, threshold_policy="0.5 (fixed)",
                        per_fold=per_fold, summary=summarise(per_fold),
                        seconds=round(time.time() - t0, 1), **extra)
        self.log.flush()

    @staticmethod
    def _score_horizons(info, feats, te) -> dict:
        """Multi-horizon model scored at EVERY horizon (not only the default),
        against that horizon's own recent-violation-rate baseline."""
        st = stack_horizons(te)
        prob = info["model"].predict_proba(info["pre"].transform(st))[:, 1]
        out = {}
        for h in HORIZONS:
            m = (st["sla_hours"] == h).to_numpy()
            y = st.loc[m, TARGET].astype(int).to_numpy()
            if 0 < y.sum() < len(y):
                from sklearn.metrics import average_precision_score, brier_score_loss
                b = st.loc[m, "hist_breach_rate_own"].fillna(float(y.mean())).to_numpy()
                out[f"{h}h"] = {"positive_rate": float(y.mean()),
                                "PR_AUC": float(average_precision_score(y, prob[m])),
                                "brier": float(brier_score_loss(y, prob[m])),
                                "baseline_recent_rate_PR_AUC": float(average_precision_score(y, b)),
                                "baseline_recent_rate_brier": float(brier_score_loss(y, b))}
        return out

    def run_baselines(self, protocol):
        exp_id = f"BASELINES_{protocol}"
        if self._skip(exp_id):
            return
        res, preds = {}, []
        for name, trm, tem in self.folds(protocol):
            tr, te = self.df[trm], self.df[tem].reset_index(drop=True)
            for b, pr in baseline_probs(tr, te).items():
                res.setdefault(b, {})[name] = score(te, pr)
                preds.append(pd.DataFrame({"incident_id": te["incident_id"], "fold": name,
                                           "baseline": b, "prob": pr,
                                           "y": te[TARGET].astype(int)}))
        pd.concat(preds).to_parquet(PRED_DIR / f"{exp_id}.parquet", index=False)
        self.log.record(exp_id, status="ok", algorithm="non-ML lookup", protocol=protocol,
                        baselines={b: {"per_fold": v, "summary": summarise(v)}
                                   for b, v in res.items()})
        self.log.flush()
        for b, v in res.items():
            s = summarise(v)
            print(f"  {exp_id} {b}: PR-AUC={s['PR_AUC']:.4f} ROC={s['ROC_AUC']:.4f} "
                  f"brier={s['brier']:.4f}")


STAGES = {
    "baselines": lambda R: (R.run_baselines("rolling"), R.run_baselines("loco")),
    "screen": lambda R: [R.run(f"{fs}__C1_slow", fs, "C1_slow") for fs in FEATURE_SETS],
    "models": lambda R: [R.run(f"S4_hist_workload_spatial__{m}", "S4_hist_workload_spatial", m)
                         for m in ("C0_p2a_A_default", "C2_slow_balanced", "C3_small")]
    + [R.run("S0_p2a_recipe__C0_p2a_A_default", "S0_p2a_recipe", "C0_p2a_A_default",
             note="Phase 2A SLA recipe retrained on the UrbanEye-SLA target")],
    "loco": lambda R: [R.run(f"{fs}__C1_slow__loco", fs, "C1_slow", protocol="loco")
                       for fs in ("S1_base", "S4_hist_workload_spatial",
                                  "S7_multihorizon_spatial")],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    R = Runner(a.force)
    for name, fn in STAGES.items():
        if a.stage in (name, "all"):
            print(f"== stage {name}", flush=True)
            fn(R)
    R.log.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
