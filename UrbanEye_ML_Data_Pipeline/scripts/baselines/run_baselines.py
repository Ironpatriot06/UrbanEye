#!/usr/bin/env python3
"""
Train a deliberately simple baseline for each SUPERVISED task and report honest
metrics.

This is NOT part of the data pipeline and `run_pipeline.py` does not call it.
Its purpose is evidence: a manifest can claim a task is model-ready, and the
only way to find out is to fit something and look at the numbers. The models
here are throwaway. The numbers are the deliverable, and the most useful of them
is the comparison against the naive baseline — a task where a gradient-boosted
tree cannot beat "predict last week's count" is a task that does not yet need a
model.

RULES OBSERVED
--------------
  * predictors come from the task MANIFEST, never from "every column except the
    target". If a column is not declared, it is not used.
  * every encoder is fitted on TRAIN ONLY and applied to val/test unchanged.
    Unseen categories map to a reserved code rather than causing a refit.
  * val and test are scored separately and reported separately. Where the split
    crosses a known regime change, the report says so.
  * PRIORITY IS NOT TRAINED. There is no priority ground truth, so a supervised
    model would be fitting the policy YAML. The script refuses.

    python scripts/baselines/run_baselines.py
    python scripts/baselines/run_baselines.py --task resolution --max-train-rows 200000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import ensure_dir, p

log = get_logger("baselines")

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 mean_absolute_error, mean_poisson_deviance,
                                 mean_squared_error, precision_recall_fscore_support,
                                 roc_auc_score)
    from sklearn.preprocessing import OrdinalEncoder
    HAVE_SKLEARN = True
except ImportError:                                          # pragma: no cover
    HAVE_SKLEARN = False


def load_task(name: str):
    fp = p("processed", "ml", f"{name}.parquet")
    mf = p("processed", "ml", f"{name}.manifest.json")
    if not fp.exists() or not mf.exists():
        return None, None
    return pd.read_parquet(fp), json.loads(mf.read_text())


def declared_predictors(m: dict) -> list[str]:
    block = m.get("predictors") or m.get("features") or {}
    return [c for c, v in block.items() if v.get("present")]


SPLIT_COL = "split_global"     # overridden by --split-column
CHOSEN_SPLIT: dict[str, str] = {}


def choose_split_column(df: pd.DataFrame) -> str:
    """
    Pick the split column that matches the table.

    `split_global` is the right answer for a POOLED model: one wall clock, so
    no evaluation row predates a training row. But it only means anything when
    the table actually spans several sources. On a SINGLE-source table it
    degenerates — sla_ml is NYC-only and NYC is the earliest data in the corpus,
    so a global cut puts every NYC row in train and leaves validation and test
    empty. There the per-source column is already chronological on one clock and
    is the correct choice, not a compromise.
    """
    if SPLIT_COL not in df.columns:
        return "split"
    # the grouping column is source_dataset on incident tables and city on the
    # hotspot panel; both mean "which publisher's timeline is this row on"
    grp = "source_dataset" if "source_dataset" in df.columns else (
        "city" if "city" in df.columns else None)
    n_src = df[grp].nunique() if grp else 1
    if n_src <= 1:
        return "split"
    if not (df[SPLIT_COL] == "val").any() or not (df[SPLIT_COL] == "test").any():
        return "split"
    return SPLIT_COL


def prepare(df: pd.DataFrame, cols: list[str]):
    """
    Split into (X_train, X_val, X_test) with encoders fitted on train alone.

    Uses SPLIT_COL. These baselines pool every city into one model, so the
    default is `split_global` — the column whose train/val/test are ordered on a
    single wall clock. Scoring a pooled model on the per-source `split` column
    means evaluating partly on the past: 19.5% of resolution test rows fall
    before the last training row there.
    """
    col = choose_split_column(df)
    tr = df[df[col] == "train"]
    va = df[df[col] == "val"]
    te = df[df[col] == "test"]
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])
                or pd.api.types.is_bool_dtype(df[c])]
    num_cols = [c for c in cols if c not in cat_cols]

    # Rare-level folding, fitted on TRAIN ONLY. The booster caps categorical
    # cardinality at 255, and `zone_key` / `subcategory` / `department` exceed
    # that; the tail is folded into an explicit __OTHER__ level rather than the
    # column being dropped. The vocabulary comes from train frequencies, so no
    # val/test level can influence the encoding.
    MAX_LEVELS = 250
    keep_levels: dict[str, set] = {}
    for c in cat_cols:
        vc = tr[c].astype("string").fillna("__MISSING__").value_counts()
        keep_levels[c] = set(vc.head(MAX_LEVELS).index)

    def as_str(part: pd.DataFrame) -> pd.DataFrame:
        # NULL becomes an explicit level rather than being dropped or imputed:
        # "no subcategory recorded" is information, and for several columns it is
        # the most informative thing about the row.
        out = {}
        for c in cat_cols:
            col = part[c].astype("string").fillna("__MISSING__")
            out[c] = col.where(col.isin(keep_levels[c]), "__OTHER__").astype("object")
        return pd.DataFrame(out, index=part.index)

    enc = None
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(as_str(tr))

    def build(part: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=part.index)
        for c in num_cols:
            out[c] = pd.to_numeric(part[c], errors="coerce").astype("float64")
        if cat_cols:
            coded = enc.transform(as_str(part))
            for i, c in enumerate(cat_cols):
                out[c] = coded[:, i]
        return out[cols]

    CHOSEN_SPLIT["last"] = col
    mask = [c in cat_cols for c in cols]
    return (tr, va, te), (build(tr), build(va), build(te)), mask


def _reg_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "median_AE": float(np.median(np.abs(y_true - y_pred))),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "RMSE_log1p": float(np.sqrt(mean_squared_error(
            np.log1p(np.clip(y_true, 0, None)), np.log1p(np.clip(y_pred, 0, None))))),
    }


def _clf_metrics(y_true, prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary",
                                                    zero_division=0)
    out = {"n": int(len(y_true)), "positive_rate": float(y_true.mean()),
           "PR_AUC": float(average_precision_score(y_true, prob)),
           "precision@0.5": float(pr), "recall@0.5": float(rc), "f1@0.5": float(f1),
           "brier": float(brier_score_loss(y_true, prob))}
    try:
        out["ROC_AUC"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        out["ROC_AUC"] = None
    return out


# ---------------------------------------------------------------------------
def run_resolution(max_rows: int) -> dict:
    df, m = load_task("resolution_ml")
    if df is None:
        return {}
    cols = declared_predictors(m)
    (tr, va, te), (Xtr, Xva, Xte), cat_mask = prepare(df, cols)
    if max_rows and len(tr) > max_rows:
        idx = tr.sample(max_rows, random_state=0).index
        tr, Xtr = tr.loc[idx], Xtr.loc[idx]

    y_tr = np.log1p(pd.to_numeric(tr["resolution_time_hours"], errors="coerce").to_numpy(float))
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                          categorical_features=cat_mask, random_state=0)
    t0 = time.time()
    model.fit(Xtr, y_tr)
    fit_s = round(time.time() - t0, 1)

    out = {"rows_trained_on": int(len(tr)), "fit_seconds": fit_s,
           "split_column": CHOSEN_SPLIT.get("last"),
           "target": "log1p(resolution_time_hours), metrics reported in hours",
           "naive_baseline": "median resolution time of the TRAIN split"}
    naive = float(np.median(pd.to_numeric(tr["resolution_time_hours"], errors="coerce")))
    for label, part, X in (("val", va, Xva), ("test", te, Xte)):
        if not len(part):
            continue
        y = pd.to_numeric(part["resolution_time_hours"], errors="coerce").to_numpy(float)
        pred = np.expm1(model.predict(X))
        out[label] = {"model": _reg_metrics(y, pred),
                      "naive": _reg_metrics(y, np.full(len(y), naive))}
        for city, g in part.groupby("city", observed=True):
            yg = pd.to_numeric(g["resolution_time_hours"], errors="coerce").to_numpy(float)
            out[label].setdefault("by_city", {})[str(city)] = _reg_metrics(
                yg, np.expm1(model.predict(X.loc[g.index])))
    out["interpretation"] = (
        "Compare model MAE against naive MAE. The target spans four orders of magnitude, so "
        "median absolute error is the more readable number and MAE is dominated by the tail. "
        "Per-city numbers matter more than the pooled one: the three cities have different "
        "operating regimes, and Chicago's validation window is the first COVID wave.")
    return out


def run_sla(max_rows: int) -> dict:
    df, m = load_task("sla_ml")
    if df is None:
        return {}
    cols = declared_predictors(m)
    (tr, va, te), (Xtr, Xva, Xte), cat_mask = prepare(df, cols)
    y_tr = tr["sla_breach"].astype("boolean").fillna(False).to_numpy(bool).astype(int)
    if len(np.unique(y_tr)) < 2:
        return {"skipped": "training split has only one class"}
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                           categorical_features=cat_mask, random_state=0)
    t0 = time.time()
    model.fit(Xtr, y_tr)
    out = {"rows_trained_on": int(len(tr)), "fit_seconds": round(time.time() - t0, 1),
           "split_column": CHOSEN_SPLIT.get("last"),
           "naive_baseline": "predict the TRAIN positive rate for every row"}
    base_rate = float(y_tr.mean())
    for label, part, X in (("val", va, Xva), ("test", te, Xte)):
        if not len(part):
            continue
        y = part["sla_breach"].astype("boolean").fillna(False).to_numpy(bool).astype(int)
        if len(np.unique(y)) < 2:
            out[label] = {"skipped": "single class in this split"}
            continue
        prob = model.predict_proba(X)[:, 1]
        out[label] = {"model": _clf_metrics(y, prob),
                      "naive": _clf_metrics(y, np.full(len(y), base_rate))}
    out["interpretation"] = (
        "PR-AUC against the positive rate is the number that matters; ROC-AUC flatters an "
        "imbalanced problem. Remember the population: NYC, Jan-Mar 2010, only the complaint "
        "types that carry a Due Date. Nothing here transfers to another city's SLA policy.")
    return out


def rolling_mean_baseline(panel: pd.DataFrame, weeks: int = 4,
                          lagged: bool = False) -> pd.Series:
    """
    Mean weekly count over the last `weeks` COMPLETED weeks, per series.

    Two variants, and the difference matters for how the comparison reads:

      lagged=False (default)  weeks t-3 .. t inclusive.
          The hotspot manifest states the deployment assumption explicitly: the
          forecast for week t+1 is produced at the END of week t, when week t is
          complete. Under that assumption week t is available, so excluding it
          would handicap the baseline and flatter the model — the exact error
          this baseline exists to correct.

      lagged=True             weeks t-4 .. t-1.
          The strictly-lagged variant already shipped as the `rolling_4w_mean`
          FEATURE, kept here so the model's inputs and this baseline can be
          compared on the same footing.

    Leakage: the window is trailing and ends at or before week t, while the
    target is week t+1, so no future or current-target information can enter.
    The window is computed on the full sorted panel, not within a split, because
    truncating history at a fold boundary would silently weaken the baseline.
    """
    keys = ["city", "zone_type", "zone_id", "category"]
    d = panel.sort_values(keys + ["week_start"], kind="stable")
    counts = pd.to_numeric(d["incident_count"], errors="coerce")
    g = counts.groupby([d[k] for k in keys], observed=True)
    if lagged:
        roll = g.transform(lambda x: x.shift(1).rolling(weeks, min_periods=1).mean())
    else:
        roll = g.transform(lambda x: x.rolling(weeks, min_periods=1).mean())
    return roll.reindex(panel.index)


def run_hotspot(max_rows: int) -> dict:
    df, m = load_task("hotspot_ml")
    if df is None:
        return {}
    cols = declared_predictors(m)
    df = df.copy()
    df["_roll4"] = rolling_mean_baseline(df, 4, lagged=False)
    df["_roll4_lagged"] = rolling_mean_baseline(df, 4, lagged=True)

    (tr, va, te), (Xtr, Xva, Xte), cat_mask = prepare(df, cols)
    y_tr = pd.to_numeric(tr["future_incident_count"], errors="coerce").to_numpy(float)
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.1,
                                          categorical_features=cat_mask, random_state=0)
    t0 = time.time()
    model.fit(Xtr, np.clip(y_tr, 0, None))
    out = {
        "rows_trained_on": int(len(tr)), "fit_seconds": round(time.time() - t0, 1),
        "split_column": CHOSEN_SPLIT.get("last"),
        "baselines": {
            "persistence": "future_incident_count = this week's incident_count",
            "rolling_4w_mean": "mean weekly count over weeks t-3..t (PRIMARY benchmark)",
            "rolling_4w_mean_lagged": "mean weekly count over weeks t-4..t-1",
        },
    }
    eps = 1e-9
    for label, part in (("val", va), ("test", te)):
        if not len(part):
            continue
        X = {"val": Xva, "test": Xte}[label]
        y = pd.to_numeric(part["future_incident_count"], errors="coerce").to_numpy(float)
        preds = {
            "model": np.clip(model.predict(X), 0, None),
            "persistence": pd.to_numeric(part["incident_count"], errors="coerce").to_numpy(float),
            "rolling_4w_mean": part["_roll4"].fillna(
                pd.to_numeric(part["incident_count"], errors="coerce")).to_numpy(float),
            "rolling_4w_mean_lagged": part["_roll4_lagged"].fillna(
                pd.to_numeric(part["incident_count"], errors="coerce")).to_numpy(float),
        }
        entry = {}
        for name, pr in preds.items():
            entry[name] = _reg_metrics(y, pr)
            entry[name]["poisson_deviance"] = float(
                mean_poisson_deviance(np.clip(y, eps, None), np.clip(pr, eps, None)))
        base = entry["rolling_4w_mean"]["MAE"]
        entry["model_vs_rolling_4w_mean_MAE_improvement"] = round(
            1 - entry["model"]["MAE"] / base, 4) if base else None
        entry["model_vs_persistence_MAE_improvement"] = round(
            1 - entry["model"]["MAE"] / entry["persistence"]["MAE"], 4) \
            if entry["persistence"]["MAE"] else None
        out[label] = entry
    out["interpretation"] = (
        "The ROLLING 4-WEEK MEAN is the primary benchmark. Persistence is retained because it "
        "is the simplest thing an operator would try, but it is the weaker of the two and "
        "quoting only the improvement over it overstates the model. A smoothed trailing mean "
        "is what any competent forecaster would reach for first on a weekly count panel, so "
        "that is the bar. Poisson deviance is the loss-aligned metric for a count target; MAE "
        "is included because it is the one an operator can read.")
    return out


def run_duplicate(max_rows: int) -> dict:
    df, m = load_task("duplicate_ml")
    if df is None:
        return {}
    cols = declared_predictors(m)
    (tr, va, te), (Xtr, Xva, Xte), cat_mask = prepare(df, cols)
    y_tr = tr["same_incident"].astype("boolean").fillna(False).to_numpy(bool).astype(int)
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                           categorical_features=cat_mask, random_state=0)
    t0 = time.time()
    model.fit(Xtr, y_tr)
    out = {"rows_trained_on": int(len(tr)), "fit_seconds": round(time.time() - t0, 1),
           "split_column": CHOSEN_SPLIT.get("last"),
           "naive_baseline": "predict the TRAIN positive rate for every pair"}
    base = float(y_tr.mean())
    for label, part, X in (("val", va, Xva), ("test", te, Xte)):
        if not len(part):
            continue
        y = part["same_incident"].astype("boolean").fillna(False).to_numpy(bool).astype(int)
        prob = model.predict_proba(X)[:, 1]
        entry = {"model": _clf_metrics(y, prob), "naive": _clf_metrics(y, np.full(len(y), base))}
        # per negative-sampling strategy: a pooled number hides which negatives are easy
        by = {}
        for strat, g in part.groupby(part["negative_strategy"].fillna("POSITIVE"), observed=True):
            if strat == "POSITIVE":
                continue
            sub = part[(part["negative_strategy"] == strat) | part["same_incident"].fillna(False)]
            ys = sub["same_incident"].astype("boolean").fillna(False).to_numpy(bool).astype(int)
            if len(np.unique(ys)) < 2:
                continue
            by[str(strat)] = _clf_metrics(ys, model.predict_proba(X.loc[sub.index])[:, 1])
        entry["positives_vs_each_negative_strategy"] = by
        out[label] = entry
    out["interpretation"] = (
        "TREAT THESE NUMBERS AS AN UPPER BOUND. The negatives are constructed, and two of the "
        "four rules define the negative by distance or time gap — the same quantities the model "
        "is given as features. The per-strategy breakdown shows how much of the pooled score "
        "comes from the easy negatives. The 1:1 class balance is also a sampling choice; the "
        "deployment prevalence of duplicate candidate pairs is far lower, so the operating "
        "threshold must be re-derived against it.")
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None,
                    choices=["resolution", "sla", "hotspot", "duplicate"])
    ap.add_argument("--max-train-rows", type=int, default=400_000,
                    help="cap on training rows, for runtime only")
    ap.add_argument("--split-column", default="split_global",
                    choices=["split_global", "split"],
                    help="split_global = one wall clock (correct for these POOLED baselines); "
                         "split = per-source chronological (correct for per-city models)")
    args = ap.parse_args()
    global SPLIT_COL
    SPLIT_COL = args.split_column

    if not HAVE_SKLEARN:
        log.error("scikit-learn is not installed. It is NOT a pipeline dependency — the "
                  "pipeline trains nothing. Install it only to run these baselines:\n"
                  "    pip install scikit-learn")
        return 2

    runners = {"resolution": run_resolution, "sla": run_sla,
               "hotspot": run_hotspot, "duplicate": run_duplicate}
    results = {}
    for name, fn in runners.items():
        if args.task and name != args.task:
            continue
        log.info("baseline: %s", name)
        try:
            results[name] = fn(args.max_train_rows)
        except Exception as exc:                              # pragma: no cover
            log.error("  %s failed: %s", name, exc)
            results[name] = {"error": str(exc)}

    results["priority"] = {
        "trained": False,
        "reason": ("REFUSED BY DESIGN. There is no priority ground truth in any public 311 "
                   "dataset. priority_baseline is a deterministic function of "
                   "config/priority_config.yaml, so a supervised model fitted to it would "
                   "reproduce the YAML and any accuracy quoted from it would be circular. "
                   "The policy score stands as the benchmark a future model must beat, once "
                   "operator-assigned labels exist."),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": ("evidence of dataset readiness, not production models. Every model here "
                    "is an out-of-the-box gradient-boosted tree with default-ish settings."),
        "protocol": {
            "split_column_requested": SPLIT_COL,
            "split_rationale": ("these baselines pool every city into one model, so they are "
                                "scored on split_global, whose train/val/test are ordered on a "
                                "single wall clock. On the per-source `split` column a pooled "
                                "model is evaluated partly on the past."),
            "predictors": "taken from each task manifest — never 'all columns except the target'",
            "preprocessing": ("ordinal encoding fitted on TRAIN only; the 250 most frequent "
                              "train levels are kept and the tail folded into __OTHER__; "
                              "unseen levels map to -1; NULL is an explicit __MISSING__ level"),
            "evaluation": "val and test scored separately, never merged",
        },
        "results": results,
    }
    dest = ensure_dir(p("reports", "baseline_models.json"))
    dest.write_text(json.dumps(report, indent=2, default=str))
    _markdown(report)
    log.info("wrote %s", dest)
    return 0


def _markdown(r: dict) -> None:
    L = ["# Baseline models", "", f"Generated {r['generated_at_utc']}", "",
         r["purpose"], "",
         "| Protocol | |", "|---|---|"]
    for k, v in r["protocol"].items():
        L.append(f"| {k} | {v} |")
    for task, res in r["results"].items():
        L += ["", f"## {task}", ""]
        if res.get("trained") is False:
            L += [f"**Not trained.** {res['reason']}", ""]
            continue
        if not res or "error" in res or "skipped" in res:
            L += [f"_{res.get('error') or res.get('skipped') or 'not built'}_", ""]
            continue
        L.append(f"Trained on {res.get('rows_trained_on'):,} rows in {res.get('fit_seconds')}s. "
                 f"Naive baseline: {res.get('naive_baseline')}.")
        for split in ("val", "test"):
            if split not in res or "skipped" in res[split]:
                continue
            L += ["", f"**{split}**", "", "| Metric | Model | Naive |", "|---|---|---|"]
            model = res[split].get("model", {})
            naive = res[split].get("naive") or res[split].get("naive_persistence") or {}
            for k in model:
                if isinstance(model[k], (int, float)):
                    nv = naive.get(k)
                    L.append(f"| {k} | {model[k]:,.4f} | "
                             f"{nv:,.4f} |" if isinstance(nv, (int, float))
                             else f"| {k} | {model[k]:,.4f} | — |")
        if res.get("interpretation"):
            L += ["", f"> {res['interpretation']}", ""]
    ensure_dir(p("reports", "baseline_models.md")).write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
