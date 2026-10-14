#!/usr/bin/env python3
"""
Phase 2B — select, gate and save the final v2 artifacts.

Nothing is trained or saved unless it passes explicit, pre-stated gates that
are evaluated on the out-of-sample predictions already recorded by the
experiment scripts (rolling origin, labels known at each origin). Every gate,
pass or fail, is written to reports/phase2b/final_gates.json.

RESOLUTION  candidate = deployable rolling-origin experiment with the lowest
            mean MAE on UrbanEye-supported rows. Gates, against the best non-ML
            baseline (lowest mean MAE):
              R1  mean MAE lower
              R2  lower MAE in >= 4 of 6 folds
              R3  mean median AE not worse
              R4  pooled paired bootstrap (clusters = report day) of the MAE
                  gain has a 95% CI that excludes 0
            Saves models/resolution_v2/ = {point, q10, q90} boosters + preprocessor.

SLA         candidates = deployable rolling-origin experiments. Gates, against the
            best non-ML baseline (the recent violation rate):
              S1  mean PR-AUC higher      S2  mean Brier lower
              S3  PR-AUC higher in >= 4 of 6 folds, Brier lower in >= 4 of 6
              S4  pooled day-cluster bootstrap CIs of the PR-AUC and Brier gains
                  both exclude 0
            Among passing candidates the best mean PR-AUC wins, except that a
            multi-horizon model (accepts any sla_hours, so it serves admin SLA
            overrides) is preferred when it is within 0.01 PR-AUC and 0.005
            Brier of the best. The decision threshold is chosen by F1 on the
            2020-05 fold and only then evaluated on the 2020-07 fold.

HOTSPOT     not ML (PHASE_2B_REPORT.md §8). models/hotspot_v2/metadata.json only:
            EWMA alpha chosen on panel C (UrbanEye categories x 1 km grid)
            training weeks, min_history_weeks = 4.

All final models are refit on every row whose outcome was known at
FINAL_CUT (= the last rolling origin, 2020-09-01).

    python scripts/phase2b/train_final.py [--task resolution|sla|hotspot|all]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.phase2b import resolution_experiments as RX
from scripts.phase2b import sla_experiments as SX
from scripts.phase2b.common import (DATA_DIR, EXPERIMENT_DIR, REPORT_DIR, ROOT, SEED,
                                    UE_SUPPORTED, UE_UNSUPPORTED, environment, now_utc,
                                    paired_bootstrap_mae_gain, sha256_file, write_json)
from scripts.phase2b.harness import ORIGINS
from scripts.train._common import best_threshold_by_f1, classification_metrics, save_model

FINAL_CUT = ORIGINS[-1]
GATES_PATH = REPORT_DIR / "final_gates.json"
MIN_FOLD_WINS = 4
N_BOOT = 300
THRESHOLD_VAL_FOLD, THRESHOLD_TEST_FOLD = "2020-05", "2020-07"
SLA_TIE_PR_AUC, SLA_TIE_BRIER = 0.01, 0.005
HOTSPOT_ALPHAS = {"ewm01": 0.1, "ewm02": 0.2, "ewm03": 0.3, "ewm05": 0.5}
HOTSPOT_MIN_WEEKS = 4


def _records(task: str) -> dict[str, dict]:
    return {json.loads(f.read_text())["experiment_id"]: json.loads(f.read_text())
            for f in sorted((EXPERIMENT_DIR / task).glob("*.json"))}


def _gate(name: str, passed: bool, **detail) -> dict:
    return {"gate": name, "passed": bool(passed), **detail}


def _day_groups(ts: pd.Series) -> np.ndarray:
    return ts.dt.strftime("%Y-%m-%d").to_numpy()


def _load_gates() -> dict:
    return json.loads(GATES_PATH.read_text()) if GATES_PATH.exists() else {}


def _save_gates(task: str, payload: dict) -> None:
    g = _load_gates()
    g[task] = payload
    g["environment"] = environment()
    write_json(GATES_PATH, g)


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
def resolution() -> dict:
    recs = _records("resolution")
    cands = {k: r for k, r in recs.items()
             if r.get("status") == "ok" and r.get("protocol") == "rolling"
             and r.get("label_policy") == "known" and r.get("deployable")
             and not r.get("quantiles")}
    cid = min(cands, key=lambda k: cands[k]["summary_ue"]["MAE"])
    cand = cands[cid]
    base_rec = recs["BASELINES_rolling_known"]["baselines"]
    bid = min(base_rec, key=lambda b: base_rec[b]["summary_ue"]["MAE"])
    base = base_rec[bid]
    folds = list(cand["per_fold"])
    c_mae = [cand["per_fold"][f]["ue_supported"]["MAE"] for f in folds]
    b_mae = [base["per_fold"][f]["ue_supported"]["MAE"] for f in folds]
    wins = int(sum(c < b for c, b in zip(c_mae, b_mae)))

    # pooled out-of-sample predictions, UrbanEye rows, paired by incident
    df = RX.load()[["incident_id", "reported_at", RX.TARGET, "is_ue_supported", "ue_category"]]
    pc = pd.read_parquet(RX.PRED_DIR / f"{cid}.parquet")
    pb = pd.read_parquet(RX.PRED_DIR / "BASELINES_rolling_known.parquet")
    pb = pb[pb["baseline"] == bid].rename(columns={"pred": "pred_base"})
    m = pc.merge(pb[["incident_id", "fold", "pred_base"]], on=["incident_id", "fold"],
                 validate="1:1").merge(df, on="incident_id", validate="m:1")
    m = m[m["is_ue_supported"]]
    boot = paired_bootstrap_mae_gain(m[RX.TARGET], m["pred_base"], m["pred"], n_boot=N_BOOT,
                                     groups=_day_groups(m["reported_at"]))
    gates = [
        _gate("R1_mean_MAE_beats_best_baseline", cand["summary_ue"]["MAE"] < base["summary_ue"]["MAE"],
              candidate=cand["summary_ue"]["MAE"], baseline=base["summary_ue"]["MAE"]),
        _gate("R2_fold_wins>=4of6", wins >= MIN_FOLD_WINS, wins=wins, folds=len(folds),
              candidate_by_fold=dict(zip(folds, c_mae)), baseline_by_fold=dict(zip(folds, b_mae))),
        _gate("R3_median_AE_not_worse",
              cand["summary_ue"]["median_AE"] <= base["summary_ue"]["median_AE"],
              candidate=cand["summary_ue"]["median_AE"], baseline=base["summary_ue"]["median_AE"]),
        _gate("R4_pooled_bootstrap_CI_excludes_0", boot["ci95"][0] > 0,
              rows=int(len(m)), cluster="report day (UTC)", **boot),
    ]
    passed = all(g["passed"] for g in gates)
    q_id = f"{cid}__q10q90" if f"{cid}__q10q90" in recs else \
        f"{RX.BEST_FS}__M2_l1log__q10q90"
    q_rec = recs.get(q_id)
    loco = recs.get(f"{cand['feature_set']}__M2_l1log__loco")
    out = {"candidate": cid, "candidates_considered": {k: r["summary_ue"] for k, r in cands.items()},
           "best_baseline": bid, "baseline_summary": base["summary_ue"],
           "candidate_summary": cand["summary_ue"], "gates": gates, "passed": passed,
           "interval_experiment": q_id if q_rec else None,
           "selection_note": ("candidate chosen as the lowest mean MAE among all recorded "
                              "deployable configurations on the same 6 folds the gates use; "
                              "this selection step is optimistic by construction, which is "
                              "why R2-R4 also require consistency and significance.")}
    print(f"resolution candidate {cid}: " + ", ".join(f"{g['gate']}={g['passed']}" for g in gates))
    if not passed:
        out["saved"] = None
        _save_gates("resolution", out)
        return out

    # ---- refit on everything known at FINAL_CUT --------------------------------
    full = RX.load()
    tr = full[(full["reported_at"] < FINAL_CUT) & full["closed_at"].notna()
              & (full["closed_at"] < FINAL_CUT)]
    spec = RX.MODELS[cand["model_name"]]
    feats = RX.FEATURE_SETS[cand["feature_set"]]
    q_base_spec = RX.MODELS[q_rec["model_name"]] if q_rec else spec
    t0 = time.time()
    _, _, _, info = RX.fit_predict(spec, feats, tr, tr.head(5))
    # quantile boosters reuse the hyperparameters of the interval experiment
    # whose coverage is reported, so the saved intervals are the evaluated ones
    _, _, _, qinfo = RX.fit_predict(q_base_spec, feats, tr, tr.head(5), quantiles=(0.1, 0.9))
    secs = round(time.time() - t0, 1)
    models = {"point": info["model"], "q10": qinfo["qmodels"]["lo"],
              "q90": qinfo["qmodels"]["hi"]}

    # reproducibility: refit the last rolling fold and compare with the record
    last = folds[-1]
    trm, tem = list(RX.rolling_folds(full, "known"))[-1][1:]
    pred_rep, _, _, _ = RX.fit_predict(spec, feats, full[trm], full[tem])
    rep_mae = RX.score(full[tem], pred_rep)["ue_supported"]["MAE"]
    repro = {"fold": last, "recorded_MAE_ue": c_mae[-1], "refit_MAE_ue": rep_mae,
             "match": bool(abs(rep_mae - c_mae[-1]) < 1e-6)}

    ds = RX.dataset_info()
    cov = ([v["ue_supported"]["interval_coverage"] for v in q_rec["per_fold"].values()]
           if q_rec else [])
    interval_cal = {"nominal_coverage": 0.8,
                    "empirical_coverage_mean": float(np.mean(cov)) if cov else None,
                    "empirical_coverage_range": [float(min(cov)), float(max(cov))] if cov else None,
                    # within 5 points of nominal on average AND in every fold
                    "calibrated": bool(bool(cov) and abs(np.mean(cov) - 0.8) <= 0.05
                                       and min(cov) >= 0.7)}
    meta = {
        "model_version": "resolution_v2.0.0", "task": "resolution_time",
        "created_at_utc": now_utc(),
        "artifact_layout": {"model.joblib": "dict {point, q10, q90} of "
                            "HistGradientBoostingRegressor, all on log1p(hours)",
                            "preprocessor.joblib": "scripts.train._common.FeaturePreprocessor"},
        "features": feats, "feature_set": cand["feature_set"],
        "categorical_features": info["pre"].categorical_, "numeric_features": info["pre"].numeric_,
        "category_column": "model_category",
        "supported_categories": UE_SUPPORTED, "unsupported_categories": UE_UNSUPPORTED,
        "target": "resolution_time_hours = closed_at - reported_at (hours); fit on log1p; "
                  "point = expm1(prediction) clipped at 0 (absolute-error loss on log1p => "
                  "a conditional MEDIAN estimate)",
        "target_transform": "log1p",
        "preprocessing": {"encoder": info["pre"].describe(),
                          "missing_numeric": "left as NaN (native HGB missing branch)",
                          "missing_categorical": "__MISSING__ level; unseen -> -1",
                          "calendar": "hour/day_of_week/is_weekend/is_night in local time "
                                      "(Asia/Kolkata at inference)"},
        "algorithm": "HistGradientBoostingRegressor", "model_name": cand["model_name"],
        "hyperparameters": {k: v for k, v in spec.items() if k != "kind"},
        "interval_calibration": interval_cal,
        "quantile_models": {"q10": 0.1, "q90": 0.9,
                            "hyperparameters": {k: v for k, v in q_base_spec.items()
                                                if k not in ("kind", "es", "loss")}},
        "random_seed": SEED,
        "training_data": {**ds, "rows": int(len(tr)),
                          "period": [str(tr["reported_at"].min()), str(tr["reported_at"].max())],
                          "rows_by_source": tr["source_dataset"].value_counts().to_dict(),
                          "ue_supported_rows": int(tr["is_ue_supported"].sum()),
                          "label_policy": f"closed_at < {FINAL_CUT.date()} (outcome known)",
                          "fit_seconds": secs},
        "evaluation_split": {"protocol": "rolling origin, 6 two-month Chicago test windows",
                             "origins": [str(o.date()) for o in ORIGINS],
                             "train": "all cleaned rows (any city) reported before the origin "
                                      "with closed_at before the origin",
                             "test": "Chicago rows in [origin, next origin); metrics on "
                                     "UrbanEye-supported categories"},
        "metrics": {"rolling_origin_summary_ue": cand["summary_ue"],
                    "rolling_origin_by_fold_ue": {f: cand["per_fold"][f]["ue_supported"]
                                                  for f in folds},
                    "pooled_bootstrap_vs_baseline": boot,
                    "interval_p10_p90": ({f: {"coverage": v["ue_supported"].get("interval_coverage"),
                                              "median_width_h": v["ue_supported"].get(
                                                  "interval_median_width_h")}
                                          for f, v in q_rec["per_fold"].items()} if q_rec else None),
                    "leave_one_city_out_all_rows": ({f: v["all"] for f, v in loco["per_fold"].items()}
                                                    if loco else None)},
        "baseline_metrics": {"name": bid, "summary_ue": base["summary_ue"],
                             "all_baselines_summary_ue": {b: v["summary_ue"]
                                                          for b, v in base_rec.items()}},
        "gates": gates, "reproducibility_check": repro,
        "software_versions": environment(),
        "limitations": ["trained on Chicago/SF/NYC 311 data; no Indian incidents",
                        "LOCO shows category-level knowledge transfers poorly between cities",
                        "p10-p90 interval is under-covered (see interval_calibration); do not "
                        "present it as an 80% interval",
                        "history features must be computed by the caller (see inference.py)"],
    }
    paths = save_model("resolution_v2", models, info["pre"], meta)
    out["saved"] = paths
    out["reproducibility_check"] = repro
    _save_gates("resolution", out)
    print(f"saved resolution_v2 ({len(tr):,} rows, {secs}s); reproducibility {repro['match']}")
    return out


# ---------------------------------------------------------------------------
# SLA
# ---------------------------------------------------------------------------
def _boot_cls(y, p_base, p_cand, groups, n_boot=N_BOOT) -> dict:
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(SEED)
    y = np.asarray(y, int)
    codes, uniq = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(order))
    pr_d, br_d = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[i]:ends[i]] for i in pick])
        yy = y[idx]
        pr_d.append(average_precision_score(yy, p_cand[idx]) - average_precision_score(yy, p_base[idx]))
        br_d.append(np.mean((p_base[idx] - yy) ** 2) - np.mean((p_cand[idx] - yy) ** 2))
    f = lambda v: [float(x) for x in np.quantile(v, [0.025, 0.975])]  # noqa: E731
    return {"PR_AUC_gain_ci95": f(pr_d), "brier_gain_ci95": f(br_d),
            "PR_AUC_gain": float(average_precision_score(y, p_cand) - average_precision_score(y, p_base)),
            "brier_gain": float(np.mean((p_base - y) ** 2) - np.mean((p_cand - y) ** 2))}


def _risk_bands(y, prob, threshold) -> dict:
    """HIGH = smallest cut whose validation precision is >= 0.90 (a flag that is
    right 9 times in 10); MEDIUM = the decision threshold."""
    from sklearn.metrics import precision_recall_curve
    pr, _, th = precision_recall_curve(y, prob)
    ok = np.where(pr[:-1] >= 0.90)[0]
    high = float(th[ok[0]]) if len(ok) else 0.8
    return {"medium": float(threshold), "high": float(max(high, threshold)),
            "rule": "HIGH if p >= high; MEDIUM if p >= medium (decision threshold); else LOW. "
                    "high = smallest probability with validation precision >= 0.90"}


def sla() -> dict:
    recs = _records("sla")
    base_rec = recs["BASELINES_rolling"]["baselines"]
    bid = max(base_rec, key=lambda b: base_rec[b]["summary"]["PR_AUC"])
    base = base_rec[bid]
    df = SX.load()[["incident_id", "reported_at", "ue_category"]]
    pb = pd.read_parquet(SX.PRED_DIR / "BASELINES_rolling.parquet")
    pb = pb[pb["baseline"] == bid].rename(columns={"prob": "p_base"})

    cands = {k: r for k, r in recs.items() if r.get("status") == "ok"
             and r.get("protocol") == "rolling" and r.get("deployable")}
    evals = {}
    for cid, r in cands.items():
        folds = list(r["per_fold"])
        prw = sum(r["per_fold"][f]["PR_AUC"] > base["per_fold"][f]["PR_AUC"] for f in folds)
        brw = sum(r["per_fold"][f]["brier"] < base["per_fold"][f]["brier"] for f in folds)
        pc = pd.read_parquet(SX.PRED_DIR / f"{cid}.parquet")
        m = pc.merge(pb[["incident_id", "fold", "p_base"]], on=["incident_id", "fold"],
                     validate="1:1").merge(df, on="incident_id", validate="m:1")
        boot = _boot_cls(m["y"].to_numpy(), m["p_base"].to_numpy(), m["prob"].to_numpy(),
                         _day_groups(m["reported_at"]))
        gates = [
            _gate("S1_mean_PR_AUC_beats_baseline", r["summary"]["PR_AUC"] > base["summary"]["PR_AUC"],
                  candidate=r["summary"]["PR_AUC"], baseline=base["summary"]["PR_AUC"]),
            _gate("S2_mean_Brier_beats_baseline", r["summary"]["brier"] < base["summary"]["brier"],
                  candidate=r["summary"]["brier"], baseline=base["summary"]["brier"]),
            _gate("S3_fold_wins>=4of6_on_both", prw >= MIN_FOLD_WINS and brw >= MIN_FOLD_WINS,
                  pr_auc_wins=int(prw), brier_wins=int(brw), folds=len(folds)),
            _gate("S4_pooled_bootstrap_CIs_exclude_0",
                  boot["PR_AUC_gain_ci95"][0] > 0 and boot["brier_gain_ci95"][0] > 0,
                  rows=int(len(m)), cluster="report day (UTC)", **boot),
        ]
        evals[cid] = {"summary": r["summary"], "feature_set": r["feature_set"],
                      "multi_horizon": r["feature_set"] in SX.MULTI, "gates": gates,
                      "passed": all(g["passed"] for g in gates)}
        print(f"  sla {cid}: " + ", ".join(f"{g['gate'][:2]}={g['passed']}" for g in gates))
    passing = [k for k, v in evals.items() if v["passed"]]
    out = {"best_baseline": bid, "baseline_summary": base["summary"], "candidates": evals,
           "tie_rule": {"PR_AUC": SLA_TIE_PR_AUC, "brier": SLA_TIE_BRIER,
                        "note": "multi-horizon preferred within this tolerance of the best "
                                "passing candidate; tolerance fixed in the session that ran "
                                "train_final.py, after the screen had been seen"}}
    if not passing:
        out.update(passed=False, saved=None)
        _save_gates("sla", out)
        print("sla: no candidate passes; nothing saved")
        return out
    best = max(passing, key=lambda k: evals[k]["summary"]["PR_AUC"])
    multi = [k for k in passing if evals[k]["multi_horizon"]
             and evals[best]["summary"]["PR_AUC"] - evals[k]["summary"]["PR_AUC"] <= SLA_TIE_PR_AUC
             and evals[k]["summary"]["brier"] - evals[best]["summary"]["brier"] <= SLA_TIE_BRIER]
    cid = max(multi, key=lambda k: evals[k]["summary"]["PR_AUC"]) if multi else best
    r = recs[cid]
    out.update(passed=True, best_by_pr_auc=best, candidate=cid)

    # ---- threshold: chosen on 2020-05, evaluated on 2020-07 -------------------------
    pc = pd.read_parquet(SX.PRED_DIR / f"{cid}.parquet")
    pbb = pd.read_parquet(SX.PRED_DIR / "BASELINES_rolling.parquet")
    pbb = pbb[pbb["baseline"] == bid]
    v, t = pc[pc["fold"] == THRESHOLD_VAL_FOLD], pc[pc["fold"] == THRESHOLD_TEST_FOLD]
    vb, tb = pbb[pbb["fold"] == THRESHOLD_VAL_FOLD], pbb[pbb["fold"] == THRESHOLD_TEST_FOLD]
    thr = best_threshold_by_f1(v["y"], v["prob"])
    thr_b = best_threshold_by_f1(vb["y"], vb["prob"])
    bands = _risk_bands(v["y"].to_numpy(), v["prob"].to_numpy(), thr)
    thresh = {"chosen_on": THRESHOLD_VAL_FOLD, "evaluated_on": THRESHOLD_TEST_FOLD,
              "rule": "max F1 on the validation fold (cost of a missed breach is undefined; "
                      "provisional until UrbanEye defines it)",
              "threshold": thr, "test_at_threshold": classification_metrics(t["y"], t["prob"], thr),
              "test_at_0.5": classification_metrics(t["y"], t["prob"], 0.5),
              "baseline_threshold": thr_b,
              "baseline_test_at_its_threshold": classification_metrics(tb["y"], tb["prob"], thr_b),
              "test_flag_rate": float((t["prob"] >= thr).mean()),
              "risk_bands": bands,
              "test_band_rates": {
                  "HIGH": {"share": float((t["prob"] >= bands["high"]).mean()),
                           "breach_rate": float(t.loc[t["prob"] >= bands["high"], "y"].mean())},
                  "MEDIUM": {"share": float(((t["prob"] >= thr) & (t["prob"] < bands["high"])).mean()),
                             "breach_rate": float(t.loc[(t["prob"] >= thr) & (t["prob"] < bands["high"]), "y"].mean())},
                  "LOW": {"share": float((t["prob"] < thr).mean()),
                          "breach_rate": float(t.loc[t["prob"] < thr, "y"].mean())}}}
    out["threshold"] = thresh

    # ---- refit on everything whose deadline passed before FINAL_CUT ------------------
    full = SX.load()
    feats, params = SX.FEATURE_SETS[r["feature_set"]], SX.MODELS[r["model_name"]]
    base_rows = full[full["reported_at"] < FINAL_CUT]
    if r["feature_set"] in SX.MULTI:
        st = SX.stack_horizons(base_rows)
        tr = st[((st["reported_at"] + pd.to_timedelta(st["sla_hours"], unit="h")) < FINAL_CUT).to_numpy()]
    else:
        tr = base_rows[(base_rows["reported_at"]
                        + pd.to_timedelta(base_rows["sla_hours"], unit="h")) < FINAL_CUT]
    t0 = time.time()
    _, info = SX.fit_predict(params, feats, tr, full.head(5))
    secs = round(time.time() - t0, 1)

    # reproducibility: refit the last rolling fold
    last_name, trm, tem = list(SX.rolling_folds(full, "known", sla_hours_col="sla_hours"))[-1]
    trf, tef = full[trm], full[tem].reset_index(drop=True)
    if r["feature_set"] in SX.MULTI:
        origin = tef["reported_at"].min().normalize()
        st = SX.stack_horizons(trf)
        trf = st[((st["reported_at"] + pd.to_timedelta(st["sla_hours"], unit="h")) < origin).to_numpy()]
    prob_rep, _ = SX.fit_predict(params, feats, trf, tef)
    rep = SX.score(tef, prob_rep)["PR_AUC"]
    repro = {"fold": last_name, "recorded_PR_AUC": r["per_fold"][last_name]["PR_AUC"],
             "refit_PR_AUC": rep, "match": bool(abs(rep - r["per_fold"][last_name]["PR_AUC"]) < 1e-9)}

    loco = recs.get(f"{r['feature_set']}__{r['model_name']}__loco")
    loco_b = recs.get("BASELINES_loco", {}).get("baselines", {}).get(bid)
    meta = {
        "model_version": "sla_v2.0.0", "task": "sla_breach_risk", "created_at_utc": now_utc(),
        "artifact_layout": {"model.joblib": "HistGradientBoostingClassifier",
                            "preprocessor.joblib": "scripts.train._common.FeaturePreprocessor"},
        "features": feats, "feature_set": r["feature_set"],
        "categorical_features": info["pre"].categorical_, "numeric_features": info["pre"].numeric_,
        "category_column": "ue_category", "multi_horizon": r["feature_set"] in SX.MULTI,
        "sla_hours_trained": SX.HORIZONS if r["feature_set"] in SX.MULTI else "category default",
        "sla_hours_range": [min(SX.HORIZONS), max(SX.HORIZONS)],
        "monotonic_constraint": "P(breach) non-increasing in sla_hours",
        "supported_categories": UE_SUPPORTED, "unsupported_categories": UE_UNSUPPORTED,
        "target": "breach = resolution_time_hours >= sla_hours, or still open past the deadline "
                  "(mirrors backend incident_service: breach at/after created_at + sla_hours)",
        "preprocessing": {"encoder": info["pre"].describe(),
                          "missing_numeric": "NaN (native HGB missing branch)",
                          "calendar": "local time (Asia/Kolkata at inference)"},
        "algorithm": "HistGradientBoostingClassifier", "model_name": r["model_name"],
        "hyperparameters": params, "random_seed": SEED,
        "decision_threshold": thr, "risk_bands": bands, "threshold_selection": thresh,
        "training_data": {**SX.dataset_info(), "rows": int(len(tr)),
                          "incidents": int(tr["incident_id"].nunique()),
                          "period": [str(tr["reported_at"].min()), str(tr["reported_at"].max())],
                          "rows_by_source": tr["source_dataset"].value_counts().to_dict(),
                          "label_policy": f"deadline (reported_at + sla_hours) < {FINAL_CUT.date()}",
                          "fit_seconds": secs},
        "evaluation_split": {"protocol": "rolling origin, 6 two-month Chicago test windows, "
                                         "scored at each incident's default SLA",
                             "origins": [str(o.date()) for o in ORIGINS]},
        "metrics": {"rolling_origin_summary": r["summary"],
                    "rolling_origin_by_fold": {f: {k: v[k] for k in ("PR_AUC", "ROC_AUC", "brier",
                                                                      "log_loss", "positive_rate")}
                                               for f, v in r["per_fold"].items()},
                    "by_horizon": {f: v.get("by_horizon") for f, v in r["per_fold"].items()},
                    "pooled_bootstrap_vs_baseline": evals[cid]["gates"][3],
                    "leave_one_city_out": ({f: {k: v[k] for k in ("PR_AUC", "brier", "positive_rate")}
                                            for f, v in loco["per_fold"].items()} if loco else None),
                    "leave_one_city_out_baseline": ({f: {k: v[k] for k in ("PR_AUC", "brier")}
                                                     for f, v in loco_b["per_fold"].items()}
                                                    if loco_b else None)},
        "baseline_metrics": {"name": bid, "summary": base["summary"],
                             "all_baselines_summary": {b: v["summary"] for b, v in base_rec.items()}},
        "gates": evals[cid]["gates"], "reproducibility_check": repro,
        "software_versions": environment(),
        "limitations": ["breach rates are Chicago's (62% at default SLAs); UrbanEye's are unknown",
                        "threshold is provisional: the cost of a missed breach is undefined; at "
                        "a ~65% base rate the max-F1 threshold flags most incidents, so the "
                        "risk bands carry more information than the binary flag",
                        "LOCO: beats the recent-violation rate on SF and NYC, not on Chicago"],
    }
    paths = save_model("sla_v2", info["model"], info["pre"], meta)
    out.update(saved=paths, reproducibility_check=repro)
    _save_gates("sla", out)
    print(f"saved sla_v2 = {cid} ({len(tr):,} rows); threshold {thr:.3f}; repro {repro['match']}")
    return out


# ---------------------------------------------------------------------------
# hotspot (non-ML)
# ---------------------------------------------------------------------------
def hotspot() -> dict:
    from scripts.phase2b.hotspot_experiments import HOTSPOT_CLEAN_FROM, ORIGINS as H_ORIGINS
    from scripts.phase2b.hotspot_experiments import TEST_WEEKS, WEEK, metrics
    pc = pd.read_parquet(DATA_DIR / "hotspot_panel_C.parquet")
    pc = pc[(pc["week_start"] >= HOTSPOT_CLEAN_FROM) & (pc["n_hist_weeks"] >= HOTSPOT_MIN_WEEKS)]
    last = pc["week_start"].max()
    mae = {a: float(np.mean(np.abs(pc["target"] - pc[c]))) for c, a in HOTSPOT_ALPHAS.items()}
    best_col = min(HOTSPOT_ALPHAS, key=lambda c: mae[HOTSPOT_ALPHAS[c]])
    alpha = HOTSPOT_ALPHAS[best_col]
    # how each fixed alpha does on each rolling test window (diagnostic, not selection)
    per_origin = {}
    for o in H_ORIGINS:
        te = pc[(pc["week_start"] >= o - WEEK) & (pc["week_start"] < o + (TEST_WEEKS - 1) * WEEK)]
        if len(te):
            per_origin[str(o.date())] = {
                **{f"ewma_{a}": metrics(te["target"], te[c])["MAE"] for c, a in HOTSPOT_ALPHAS.items()},
                "rolling_4w_mean": metrics(te["target"], te["roll4"])["MAE"]}
    pa = pd.read_parquet(DATA_DIR / "hotspot_panel_A.parquet")
    sf = pa[(pa["city"] == "San Francisco") & (pa["n_hist_weeks"] >= HOTSPOT_MIN_WEEKS)]
    sf_mae = {a: float(np.mean(np.abs(sf["target"] - sf[c]))) for c, a in HOTSPOT_ALPHAS.items()}
    hx = _records("hotspot")
    evidence = {k: hx[k]["summary"] for k in ("C_grid1km_rolling", "A_rolling_full",
                                              "A_transfer_chicago_to_sf",
                                              "A_transfer_chicago_to_sf_H2_H3",
                                              "A_ward_holdout") if k in hx}
    meta = {
        "model_version": "hotspot_v2.0.0", "task": "hotspot_next_week_count",
        "created_at_utc": now_utc(), "method": "ewma", "is_ml": False,
        "ewma_alpha": alpha, "min_history_weeks": HOTSPOT_MIN_WEEKS,
        "formula": "s_0 = y_0; s_t = alpha * y_t + (1 - alpha) * s_{t-1}; forecast(t+1) = s_t "
                   "(pandas ewm(alpha, adjust=False)), over complete weeks, oldest first",
        "unit": "UrbanEye category x 1 km grid cell x ISO week (Monday start)",
        "grid": json.loads((DATA_DIR / "hotspot_panel_C.grid.json").read_text()),
        "grid_note": "the Chicago origin is illustrative; UrbanEye must anchor its own grid "
                     "(e.g. ST_SnapToGrid on a local UTM projection) per city",
        "supported_categories": UE_SUPPORTED,
        "alpha_selection": {"panel": "C (Chicago, UrbanEye categories, 1 km cells)",
                            "rows": int(len(pc)), "weeks": [str(pc["week_start"].min().date()),
                                                            str(last.date())],
                            "criterion": "MAE of the one-week-ahead forecast",
                            "mae_by_alpha": mae, "san_francisco_mae_by_alpha": sf_mae,
                            "per_origin_test_mae": per_origin},
        "why_not_ml": ("Within Chicago the gradient-boosted forecaster beats the trained EWMA "
                       "by ~4% MAE on panel C (2-10% by fold), but trained on Chicago and applied to San "
                       "Francisco it is 37-52% worse than SF's own EWMA. UrbanEye would apply "
                       "a US-trained model to an unseen city, which is the transfer case."),
        "evidence_summaries": evidence, "software_versions": environment(),
    }
    d = ROOT / "models" / "hotspot_v2"
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / "metadata.json", meta)
    out = {"saved": str(d / "metadata.json"), "ewma_alpha": alpha, "mae_by_alpha": mae,
           "sf_mae_by_alpha": sf_mae, "is_ml": False}
    _save_gates("hotspot", out)
    print(f"saved hotspot_v2 metadata: alpha={alpha} (panel C MAE {mae})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["resolution", "sla", "hotspot", "all"])
    a = ap.parse_args()
    for name, fn in (("resolution", resolution), ("sla", sla), ("hotspot", hotspot)):
        if a.task in (name, "all"):
            fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
