#!/usr/bin/env python3
"""
Phase 2B — error analysis and permutation importance for the final choices.

Uses only out-of-sample predictions already recorded by the rolling-origin
experiments (each prediction came from a model trained before its test
window), so every number here is a held-out number. Permutation importance
refits the final configuration on the last rolling fold's training rows and
permutes features on that fold's test rows.

Reads reports/phase2b/final_gates.json (run train_final.py first) and writes
reports/phase2b/error_analysis.{json,md}.

    python scripts/phase2b/error_analysis.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.phase2b import resolution_experiments as RX
from scripts.phase2b import sla_experiments as SX
from scripts.phase2b.common import DATA_DIR, REPORT_DIR, SEED, environment, now_utc, write_json
from scripts.train._common import calibration_bins

GATES = REPORT_DIR / "final_gates.json"
DUR_BINS = [0, 1, 24, 72, 168, 720, 2160, 8760, np.inf]
DUR_LABELS = ["<1h", "1-24h", "1-3d", "3-7d", "7-30d", "30-90d", "90d-1y", ">1y"]
PI_ROWS, PI_REPEATS = 20_000, 5


def _res_table(d: pd.DataFrame, by: str) -> pd.DataFrame:
    tot = np.abs(d["y"] - d["pred"]).sum()
    rows = []
    for k, g in d.groupby(by, observed=True):
        e, eb = np.abs(g["y"] - g["pred"]), np.abs(g["y"] - g["pred_base"])
        r = {by: str(k), "n": len(g), "share_rows": len(g) / len(d),
             "true_median_h": g["y"].median(), "pred_median_h": g["pred"].median(),
             "MAE": e.mean(), "MAE_baseline": eb.mean(), "median_AE": e.median(),
             "median_AE_baseline": eb.median(), "share_of_total_abs_error": e.sum() / tot,
             "median_signed_error_h": float(np.median(g["pred"] - g["y"]))}
        if "lo" in g:
            r["p10_p90_coverage"] = float(((g["y"] >= g["lo"]) & (g["y"] <= g["hi"])).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def _perm_importance(model, X, scorer, rng_seed=SEED) -> dict:
    """Mean drop in score when one column is shuffled (higher = more important)."""
    rng = np.random.default_rng(rng_seed)
    base = scorer(model, X)
    out = {}
    for c in X.columns:
        drops = []
        for _ in range(PI_REPEATS):
            Xp = X.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            drops.append(base - scorer(model, Xp))
        out[c] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["mean"]))


# ---------------------------------------------------------------------------
def resolution(g: dict) -> dict:
    cid, bid = g["candidate"], g["best_baseline"]
    rec = json.loads((RX.OUT / f"{cid}.json").read_text())
    df = RX.load()
    pc = pd.read_parquet(RX.PRED_DIR / f"{cid}.parquet")
    pb = pd.read_parquet(RX.PRED_DIR / "BASELINES_rolling_known.parquet")
    pb = pb[pb["baseline"] == bid].rename(columns={"pred": "pred_base"})
    d = pc.merge(pb[["incident_id", "fold", "pred_base"]], on=["incident_id", "fold"])
    q_id = g.get("interval_experiment")
    if q_id:
        pq = pd.read_parquet(RX.PRED_DIR / f"{q_id}.parquet")[["incident_id", "fold", "lo", "hi"]]
        d = d.merge(pq, on=["incident_id", "fold"], how="left")
    d = d.merge(df[["incident_id", RX.TARGET, "is_ue_supported", "ue_category"]],
                on="incident_id").rename(columns={RX.TARGET: "y"})
    d = d[d["is_ue_supported"]].copy()
    d["duration_bin"] = pd.cut(d["y"], DUR_BINS, labels=DUR_LABELS, right=False)
    out = {"candidate": cid, "baseline": bid, "rows": int(len(d)),
           "by_duration_bin": _res_table(d, "duration_bin").to_dict(orient="records"),
           "by_category": _res_table(d, "ue_category").to_dict(orient="records"),
           "by_fold": _res_table(d, "fold").to_dict(orient="records")}
    # where the error mass lives
    e = np.abs(d["y"] - d["pred"])
    out["error_mass"] = {"rows_over_90d_share": float((d["y"] >= 2160).mean()),
                         "abs_error_share_from_rows_over_90d": float(e[d["y"] >= 2160].sum() / e.sum()),
                         "pred_p90_h": float(d["pred"].quantile(0.9)),
                         "true_p90_h": float(d["y"].quantile(0.9))}

    # permutation importance on the last fold
    feats, spec = rec["features"], RX.MODELS[rec["model_name"]]
    name, trm, tem = list(RX.rolling_folds(df, "known"))[-1]
    tr, te = df[trm], df[tem]
    te = te[te["is_ue_supported"]].sample(min(PI_ROWS, int(te["is_ue_supported"].sum())),
                                          random_state=SEED)
    _, _, _, info = RX.fit_predict(spec, feats, tr, te)
    X = info["pre"].transform(te)
    y = te[RX.TARGET].to_numpy(float)
    neg_mae_log = lambda m, X_: -np.mean(np.abs(np.log1p(y) - m.predict(X_)))  # noqa: E731
    neg_mae_h = lambda m, X_: -np.mean(np.abs(y - np.clip(np.expm1(m.predict(X_)), 0, None)))  # noqa: E731
    out["permutation_importance"] = {
        "fold": name, "rows": int(len(te)), "repeats": PI_REPEATS,
        "metric_log1p_MAE": _perm_importance(info["model"], X, neg_mae_log),
        "metric_MAE_hours": _perm_importance(info["model"], X, neg_mae_h)}
    return out


# ---------------------------------------------------------------------------
def sla(g: dict) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss
    cid, bid = g["candidate"], g["best_baseline"]
    rec = json.loads((SX.OUT / f"{cid}.json").read_text())
    df = SX.load()
    pc = pd.read_parquet(SX.PRED_DIR / f"{cid}.parquet")
    pb = pd.read_parquet(SX.PRED_DIR / "BASELINES_rolling.parquet")
    pb = pb[pb["baseline"] == bid].rename(columns={"prob": "p_base"})
    d = pc.merge(pb[["incident_id", "fold", "p_base"]], on=["incident_id", "fold"]) \
        .merge(df[["incident_id", "ue_category", "is_open_at_extract"]], on="incident_id")
    thr = g["threshold"]["threshold"]

    def tbl(by):
        rows = []
        for k, x in d.groupby(by, observed=True):
            r = {by: str(k), "n": len(x), "positive_rate": x["y"].mean(),
                 "mean_prob": x["prob"].mean(), "brier": brier_score_loss(x["y"], x["prob"]),
                 "brier_baseline": brier_score_loss(x["y"], x["p_base"]),
                 "flagged_at_threshold": (x["prob"] >= thr).mean()}
            if 0 < x["y"].sum() < len(x):
                r["PR_AUC"] = average_precision_score(x["y"], x["prob"])
                r["PR_AUC_baseline"] = average_precision_score(x["y"], x["p_base"])
            rows.append(r)
        return pd.DataFrame(rows).to_dict(orient="records")

    t = d[d["fold"] == g["threshold"]["evaluated_on"]]
    conf = {}
    for k, x in t.groupby("ue_category", observed=True):
        p = x["prob"] >= thr
        conf[str(k)] = {"tp": int((p & (x["y"] == 1)).sum()), "fp": int((p & (x["y"] == 0)).sum()),
                        "fn": int((~p & (x["y"] == 1)).sum()), "tn": int((~p & (x["y"] == 0)).sum())}
    out = {"candidate": cid, "baseline": bid, "rows": int(len(d)),
           "by_category": tbl("ue_category"), "by_fold": tbl("fold"),
           "calibration_pooled": calibration_bins(d["y"], d["prob"]),
           "calibration_pooled_baseline": calibration_bins(d["y"], d["p_base"]),
           "confusion_by_category_test_fold": conf,
           "open_at_extract_share_of_positives": float(d.loc[d["y"] == 1, "is_open_at_extract"].mean())}

    feats, params = rec["features"], SX.MODELS[rec["model_name"]]
    name, trm, tem = list(SX.rolling_folds(df, "known", sla_hours_col="sla_hours"))[-1]
    tr, te = df[trm], df[tem].reset_index(drop=True)
    if rec["feature_set"] in SX.MULTI:
        origin = te["reported_at"].min().normalize()
        st = SX.stack_horizons(tr)
        tr = st[((st["reported_at"] + pd.to_timedelta(st["sla_hours"], unit="h")) < origin).to_numpy()]
    te = te.sample(min(PI_ROWS, len(te)), random_state=SEED)
    _, info = SX.fit_predict(params, feats, tr, te)
    X = info["pre"].transform(te)
    y = te[SX.TARGET].astype(int).to_numpy()
    ap = lambda m, X_: average_precision_score(y, m.predict_proba(X_)[:, 1])  # noqa: E731
    nb = lambda m, X_: -brier_score_loss(y, m.predict_proba(X_)[:, 1])  # noqa: E731
    imp = {"fold": name, "rows": int(len(te)), "repeats": PI_REPEATS,
           "metric_PR_AUC": _perm_importance(info["model"], X, ap),
           "metric_neg_brier": _perm_importance(info["model"], X, nb)}
    if "sla_hours" in feats:
        imp["note"] = ("sla_hours is constant (the category default) in the test rows, so "
                       "permuting it within them only reshuffles category defaults")
    out["permutation_importance"] = imp
    return out


# ---------------------------------------------------------------------------
def hotspot(g: dict) -> dict:
    from scripts.phase2b.hotspot_experiments import (HOTSPOT_CLEAN_FROM, ORIGINS, TEST_WEEKS,
                                                     WEEK)
    alpha = g["ewma_alpha"]
    col = f"ewm{str(alpha).replace('.', '')}"
    pc = pd.read_parquet(DATA_DIR / "hotspot_panel_C.parquet")
    pc = pc[(pc["week_start"] >= HOTSPOT_CLEAN_FROM) & (pc["n_hist_weeks"] >= 4)]
    parts = []
    for o in ORIGINS:
        te = pc[(pc["week_start"] >= o - WEEK) & (pc["week_start"] < o + (TEST_WEEKS - 1) * WEEK)]
        parts.append(te.assign(origin=str(o.date())))
    d = pd.concat(parts)
    d["count_bin"] = pd.cut(d["target"], [0, 1, 3, 6, 11, np.inf],
                            labels=["0", "1-2", "3-5", "6-10", ">10"], right=False)

    def tbl(by):
        rows = []
        for k, x in d.groupby(by, observed=True):
            e = np.abs(x["target"] - x[col])
            rows.append({by: str(k), "n": len(x), "mean_count": x["target"].mean(),
                         "MAE_ewma": e.mean(), "MAE_roll4": np.abs(x["target"] - x["roll4"]).mean(),
                         "bias_ewma": float((x[col] - x["target"]).mean())})
        return pd.DataFrame(rows).to_dict(orient="records")

    return {"alpha": alpha, "rows": int(len(d)), "by_category": tbl("category"),
            "by_count_bin": tbl("count_bin"), "by_origin": tbl("origin"),
            "note": "panel C rolling test windows; alpha fixed (chosen on all clean weeks), so "
                    "this is a diagnostic of error structure, not an out-of-sample score"}


# ---------------------------------------------------------------------------
def _md(res: dict) -> str:
    def t(rows, cols, fmt=None):
        fmt = fmt or {}
        head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        body = ""
        for r in rows:
            cells = []
            for c in cols:
                v = r.get(c)
                cells.append(fmt.get(c, "{:.3f}").format(v) if isinstance(v, (float, np.floating))
                             else str(v))
            body += "| " + " | ".join(cells) + " |\n"
        return head + body
    s = ["# Phase 2B — error analysis\n", f"Generated {res['generated_at_utc']}. Held-out "
         "rolling-origin predictions only.\n"]
    if "resolution" in res:
        r = res["resolution"]
        f1 = {c: "{:.1f}" for c in ("true_median_h", "pred_median_h", "MAE", "MAE_baseline",
                                     "median_AE", "median_AE_baseline", "median_signed_error_h")}
        cols = ["n", "true_median_h", "pred_median_h", "MAE", "MAE_baseline", "median_AE",
                "median_AE_baseline", "share_of_total_abs_error", "median_signed_error_h",
                "p10_p90_coverage"]
        s += [f"## Resolution — {r['candidate']} vs {r['baseline']}\n", "### By true duration\n",
              t(r["by_duration_bin"], ["duration_bin"] + cols, f1), "\n### By category\n",
              t(r["by_category"], ["ue_category"] + cols, f1), "\n### By fold\n",
              t(r["by_fold"], ["fold"] + cols, f1),
              "\n### Permutation importance (Δ log1p-MAE, last fold)\n",
              t([{"feature": k, **v} for k, v in
                 r["permutation_importance"]["metric_log1p_MAE"].items()],
                ["feature", "mean", "std"], {"mean": "{:.4f}", "std": "{:.4f}"})]
    if "sla" in res:
        r = res["sla"]
        cols = ["n", "positive_rate", "mean_prob", "PR_AUC", "PR_AUC_baseline", "brier",
                "brier_baseline", "flagged_at_threshold"]
        s += [f"\n## SLA — {r['candidate']} vs {r['baseline']}\n", "### By category\n",
              t(r["by_category"], ["ue_category"] + cols), "\n### By fold\n",
              t(r["by_fold"], ["fold"] + cols), "\n### Calibration (pooled)\n",
              t(r["calibration_pooled"], ["bin", "n", "mean_predicted", "observed_rate"]),
              "\n### Permutation importance (Δ PR-AUC, last fold)\n",
              t([{"feature": k, **v} for k, v in r["permutation_importance"]["metric_PR_AUC"].items()],
                ["feature", "mean", "std"], {"mean": "{:.4f}", "std": "{:.4f}"})]
    if "hotspot" in res:
        r = res["hotspot"]
        cols = ["n", "mean_count", "MAE_ewma", "MAE_roll4", "bias_ewma"]
        s += [f"\n## Hotspot — EWMA alpha={r['alpha']}\n", "### By category\n",
              t(r["by_category"], ["category"] + cols), "\n### By next-week count\n",
              t(r["by_count_bin"], ["count_bin"] + cols), "\n### By origin\n",
              t(r["by_origin"], ["origin"] + cols)]
    return "\n".join(s)


def main() -> int:
    g = json.loads(GATES.read_text())
    res = {"generated_at_utc": now_utc(), "environment": environment()}
    if g.get("resolution", {}).get("passed"):
        res["resolution"] = resolution(g["resolution"])
        print("resolution error analysis done", flush=True)
    if g.get("sla", {}).get("passed"):
        res["sla"] = sla(g["sla"])
        print("sla error analysis done", flush=True)
    if "hotspot" in g:
        res["hotspot"] = hotspot(g["hotspot"])
    write_json(REPORT_DIR / "error_analysis.json", res)
    (REPORT_DIR / "error_analysis.md").write_text(_md(json.loads(
        (REPORT_DIR / "error_analysis.json").read_text())))
    print("wrote reports/phase2b/error_analysis.{json,md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
