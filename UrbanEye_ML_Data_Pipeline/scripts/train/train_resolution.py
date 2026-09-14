#!/usr/bin/env python3
"""
Train and evaluate the resolution-time model.

    target      resolution_time_hours  (hours, closed cases with a MEASURABLE duration)
    dataset     data/processed/ml/resolution_ml.parquet
    split       split_global — pooled across cities on one wall clock

PROTOCOL
--------
1. The baseline is established first: predict the TRAINING median for every row.
   Nothing is allowed to be called an improvement until it beats that.
2. Three configurations are tried, each answering a specific question rather
   than sampling a hyper-parameter space:

     A  squared_error on log1p(target)   the shape the target actually has
     B  absolute_error on raw hours      optimises the headline metric directly
     C  squared_error on log1p, more capacity   is A capacity-limited?

   A and B exist because this target has a known tension: the audit found a
   model that wins on MAE while LOSING on median absolute error. Fitting on
   log1p optimises relative error (good for the middle of the distribution);
   fitting absolute_error on raw hours optimises MAE (good for the tail). The
   two configurations make that trade-off measurable instead of accidental.
3. Selection is on VALIDATION MAE. Test is scored once, at the end, with the
   selected configuration, and is never used to choose anything.

    python scripts/train/train_resolution.py
    python scripts/train/train_resolution.py --max-train-rows 200000
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

from scripts.train._common import (SEED, FeaturePreprocessor, environment, fmt_row, load_task,
                                   log, now_utc, regression_metrics, save_model, write_report)

TASK = "resolution"
DATASET = "resolution_ml"
PRIMARY_METRIC = "MAE"

CONFIGS = {
    "A_log1p_squared": {
        "question": "does the natural shape of the target (log) model it best?",
        "target_transform": "log1p",
        "params": dict(loss="squared_error", max_iter=250, learning_rate=0.1,
                       max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.0),
    },
    "B_raw_absolute": {
        "question": "does optimising MAE directly beat optimising it indirectly?",
        "target_transform": "none",
        "params": dict(loss="absolute_error", max_iter=250, learning_rate=0.1,
                       max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.0),
    },
    "C_log1p_deeper": {
        "question": "is configuration A limited by capacity?",
        "target_transform": "log1p",
        "params": dict(loss="squared_error", max_iter=500, learning_rate=0.06,
                       max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=1.0),
    },
}


def _fit_predict(cfg: dict, Xtr, ytr_raw, Xs: dict, cat_mask):
    from sklearn.ensemble import HistGradientBoostingRegressor
    y = np.log1p(ytr_raw) if cfg["target_transform"] == "log1p" else ytr_raw
    model = HistGradientBoostingRegressor(categorical_features=cat_mask,
                                          random_state=SEED, **cfg["params"])
    t0 = time.time()
    model.fit(Xtr, y)
    secs = round(time.time() - t0, 1)
    preds = {}
    for k, X in Xs.items():
        raw = model.predict(X)
        preds[k] = np.clip(np.expm1(raw) if cfg["target_transform"] == "log1p" else raw, 0, None)
    return model, preds, secs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train-rows", type=int, default=None,
                    help="cap training rows (runtime only); default uses the whole fold")
    ap.add_argument("--split-column", default="split_global", choices=["split_global", "split"])
    args = ap.parse_args()

    data = load_task(TASK, DATASET, args.split_column)
    tr, va, te = data.fold("train"), data.fold("val"), data.fold("test")
    log.info("%s: train=%d val=%d test=%d (split column: %s)",
             TASK, len(tr), len(va), len(te), data.split_column)
    if not len(va) or not len(te):
        raise SystemExit("validation or test fold is empty — refusing to train")

    if args.max_train_rows and len(tr) > args.max_train_rows:
        tr = tr.sample(args.max_train_rows, random_state=SEED).sort_index()
        data.notes.append(f"training fold capped at {args.max_train_rows:,} rows for runtime")

    pre = FeaturePreprocessor(data.predictors).fit(tr)
    Xtr = pre.transform(tr)
    Xs = {"val": pre.transform(va), "test": pre.transform(te)}
    ytr = pd.to_numeric(tr[data.target], errors="coerce").to_numpy(float)
    ys = {"val": pd.to_numeric(va[data.target], errors="coerce").to_numpy(float),
          "test": pd.to_numeric(te[data.target], errors="coerce").to_numpy(float)}

    # ---- 1. baseline first -------------------------------------------------
    median = float(np.median(ytr))
    baseline = {k: regression_metrics(ys[k], np.full(len(ys[k]), median)) for k in ys}
    log.info("baseline (train median = %.1f h): val MAE=%.1f  test MAE=%.1f",
             median, baseline["val"]["MAE"], baseline["test"]["MAE"])

    # ---- 2. candidates, scored on validation only -------------------------
    experiments = {}
    for name, cfg in CONFIGS.items():
        model, preds, secs = _fit_predict(cfg, Xtr, ytr, Xs, pre.categorical_mask)
        experiments[name] = {
            "question": cfg["question"], "target_transform": cfg["target_transform"],
            "params": cfg["params"], "fit_seconds": secs,
            "val": regression_metrics(ys["val"], preds["val"]),
            "_model": model, "_preds": preds,
        }
        log.info("  %-18s val MAE=%8.1f  medAE=%7.1f  R2=%.4f  (%ss)",
                 name, experiments[name]["val"]["MAE"],
                 experiments[name]["val"]["median_AE"], experiments[name]["val"]["R2"], secs)

    selected = min(experiments, key=lambda k: experiments[k]["val"][PRIMARY_METRIC])
    log.info("selected on validation %s: %s", PRIMARY_METRIC, selected)

    # ---- 3. test, scored once --------------------------------------------
    chosen = experiments[selected]
    test_metrics = regression_metrics(ys["test"], chosen["_preds"]["test"])
    val_metrics = chosen["val"]

    # per-city, reported but never used to select
    by_city = {}
    for city, g in te.groupby("city", observed=True):
        idx = te.index.get_indexer(g.index)
        by_city[str(city)] = {
            "model": regression_metrics(ys["test"][idx], chosen["_preds"]["test"][idx]),
            "baseline_train_median": regression_metrics(
                ys["test"][idx], np.full(len(idx), median)),
        }

    beats_mae = test_metrics["MAE"] < baseline["test"]["MAE"]
    beats_med = test_metrics["median_AE"] < baseline["test"]["median_AE"]
    verdict = ("beats the median baseline on MAE but LOSES on median absolute error — it wins "
               "on the tail and loses in the middle"
               if beats_mae and not beats_med else
               "beats the median baseline on both MAE and median absolute error"
               if beats_mae and beats_med else
               "does NOT beat the median baseline on MAE")

    meta = {
        "task": TASK, "dataset": DATASET, "target": data.target,
        "trained_at_utc": now_utc(), "environment": environment(),
        "split_column": data.split_column, "split_counts": data.counts(),
        "rows": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "rows_dropped_missing_target": data.rows_dropped_missing_target,
        "n_features": len(data.predictors), "features": data.predictors,
        "preprocessing": pre.describe(),
        "selected_config": selected, "selection_metric": f"validation {PRIMARY_METRIC}",
        "model_params": CONFIGS[selected]["params"],
        "target_transform": CONFIGS[selected]["target_transform"],
        "notes": data.notes,
    }
    paths = save_model(TASK, chosen["_model"], pre, meta)

    payload = {
        **meta,
        "artifacts": paths,
        "baseline": {"definition": "predict the TRAINING median for every row",
                     "train_median_hours": median, **baseline},
        "metrics": {"val": val_metrics, "test": test_metrics},
        "per_city_test": by_city,
        "experiments": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                        for k, v in experiments.items()},
        "verdict": verdict,
        "honesty_note": ("Selection used validation only. The test fold was scored once, with "
                         "the already-chosen configuration. Per-city numbers are reported, not "
                         "used to select."),
    }
    write_report(TASK, payload, _markdown(payload))
    log.info("test: MAE=%.1f (baseline %.1f)  medAE=%.1f (baseline %.1f)  R2=%.4f",
             test_metrics["MAE"], baseline["test"]["MAE"],
             test_metrics["median_AE"], baseline["test"]["median_AE"], test_metrics["R2"])
    log.info("verdict: %s", verdict)
    return 0


def _markdown(r: dict) -> str:
    K = ["MAE", "RMSE", "median_AE", "R2", "p90_AE", "p99_AE"]
    L = [f"# Resolution model — {r['target']}", "",
         f"Trained {r['trained_at_utc']} · seed {r['environment']['seed']} · "
         f"scikit-learn {r['environment']['scikit_learn']}", "",
         f"**Dataset** `{r['dataset']}` · **split column** `{r['split_column']}` "
         f"({r['split_counts']}) · **features** {r['n_features']}", "",
         f"**Selected configuration:** `{r['selected_config']}` "
         f"(chosen on {r['selection_metric']})", "",
         "## Metrics", "",
         "| Fold | " + " | ".join(K) + " |", "|---" * (len(K) + 1) + "|",
         fmt_row("validation — model", r["metrics"]["val"], K),
         fmt_row("validation — baseline", r["baseline"]["val"], K),
         fmt_row("**test — model**", r["metrics"]["test"], K),
         fmt_row("**test — baseline**", r["baseline"]["test"], K), "",
         f"Baseline: {r['baseline']['definition']} "
         f"({r['baseline']['train_median_hours']:.1f} h).", "",
         f"> **Verdict:** the model {r['verdict']}.", "",
         "## Configurations tried", "",
         "| Config | Question | val MAE | val median AE | val R2 |", "|---|---|---|---|---|"]
    for name, e in r["experiments"].items():
        mark = "**" if name == r["selected_config"] else ""
        L.append(f"| {mark}{name}{mark} | {e['question']} | {e['val']['MAE']:,.1f} | "
                 f"{e['val']['median_AE']:,.1f} | {e['val']['R2']:.4f} |")
    L += ["", "## Test metrics by city", "",
          "| City | model MAE | baseline MAE | model median AE | baseline median AE | n |",
          "|---|---|---|---|---|---|"]
    for city, m in r["per_city_test"].items():
        L.append(f"| {city} | {m['model']['MAE']:,.1f} | {m['baseline_train_median']['MAE']:,.1f} "
                 f"| {m['model']['median_AE']:,.1f} | "
                 f"{m['baseline_train_median']['median_AE']:,.1f} | {m['model']['n']:,} |")
    L += ["", "## Features", "", ", ".join(f"`{c}`" for c in r["features"]), "",
          "## Preprocessing", "",
          f"- fitted on: {r['preprocessing']['fitted_on']}",
          f"- {r['preprocessing']['missing_policy']}",
          f"- {r['preprocessing']['rare_level_policy']}",
          f"- {r['preprocessing']['unknown_policy']}", "",
          "## Notes", ""] + [f"- {n}" for n in r["notes"]] + \
         ["", f"- {r['honesty_note']}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
