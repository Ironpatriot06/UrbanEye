#!/usr/bin/env python3
"""
Train and evaluate the hotspot model.

    target      future_incident_count  (count in week t+1, same zone x category)
    dataset     data/processed/ml/hotspot_ml.parquet
    split       split_global — pooled across cities on one wall clock

THE BASELINES ARE THE POINT
---------------------------
The rolling 4-week mean currently beats the gradient-boosted model on
validation. This script is written so that fact cannot be hidden: all three
baselines are computed on the same rows as the model, the verdict is derived
mechanically from the numbers rather than written by hand, and a model that
loses is reported as losing.

`rolling_mean_baseline` is imported from scripts/baselines/run_baselines.py
rather than reimplemented. One definition, one place: if the baseline changes,
this comparison changes with it and cannot silently drift.

LEAKAGE
-------
Every predictor is a trailing quantity ending at or before week t while the
target is week t+1. `incident_count` — the current week's own count — is a
legitimate predictor only under the deployment assumption the manifest states:
the forecast for week t+1 is produced at the END of week t, when week t is
complete. The rolling baselines are computed on the full sorted panel, never
within a fold, because truncating history at a fold boundary would weaken the
baseline and flatter the model.

    python scripts/train/train_hotspot.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.baselines.run_baselines import rolling_mean_baseline
from scripts.train._common import (SEED, FeaturePreprocessor, environment, fmt_row, load_task,
                                   log, now_utc, regression_metrics, save_model, write_report)

TASK = "hotspot"
DATASET = "hotspot_ml"
PRIMARY_METRIC = "MAE"

CONFIGS = {
    "A_poisson": {
        "question": "the loss that matches a count target",
        "target_transform": "none",
        "params": dict(loss="poisson", max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                       min_samples_leaf=20, l2_regularization=0.0),
    },
    "B_poisson_regularised": {
        "question": ("the model loses to a 4-week mean despite having that mean as a "
                     "feature — is it over-fitting the panel?"),
        "target_transform": "none",
        "params": dict(loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                       min_samples_leaf=100, l2_regularization=5.0),
    },
    "C_log1p_squared": {
        "question": "does modelling log1p(count) behave better than a Poisson loss here?",
        "target_transform": "log1p",
        "params": dict(loss="squared_error", max_iter=300, learning_rate=0.1,
                       max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0),
    },
}


def poisson_deviance(y_true, y_pred) -> float:
    from sklearn.metrics import mean_poisson_deviance
    eps = 1e-9
    return float(mean_poisson_deviance(np.clip(y_true, eps, None), np.clip(y_pred, eps, None)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-column", default="split_global", choices=["split_global", "split"])
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor

    data = load_task(TASK, DATASET, args.split_column)
    panel = data.frame

    # baselines first, on the FULL sorted panel — never per fold
    panel = panel.copy()
    panel["_roll4"] = rolling_mean_baseline(panel, 4, lagged=False)
    panel["_roll4_lagged"] = rolling_mean_baseline(panel, 4, lagged=True)
    data.frame = panel

    tr, va, te = data.fold("train"), data.fold("val"), data.fold("test")
    log.info("%s: train=%d val=%d test=%d (split column: %s)",
             TASK, len(tr), len(va), len(te), data.split_column)
    if not len(va) or not len(te):
        raise SystemExit("validation or test fold is empty — refusing to train")

    pre = FeaturePreprocessor(data.predictors).fit(tr)
    Xtr = pre.transform(tr)
    Xs = {"val": pre.transform(va), "test": pre.transform(te)}
    ytr = pd.to_numeric(tr[data.target], errors="coerce").to_numpy(float)
    ys = {k: pd.to_numeric(v[data.target], errors="coerce").to_numpy(float)
          for k, v in (("val", va), ("test", te))}

    # ---- 1. baselines ------------------------------------------------------
    def base_preds(part: pd.DataFrame) -> dict:
        cur = pd.to_numeric(part["incident_count"], errors="coerce")
        return {
            "persistence": cur.to_numpy(float),
            "rolling_4w_mean": part["_roll4"].fillna(cur).to_numpy(float),
            "rolling_4w_mean_lagged": part["_roll4_lagged"].fillna(cur).to_numpy(float),
        }

    baselines = {}
    for fold, part in (("val", va), ("test", te)):
        baselines[fold] = {}
        for name, pr in base_preds(part).items():
            m = regression_metrics(ys[fold], pr)
            m["poisson_deviance"] = poisson_deviance(ys[fold], pr)
            baselines[fold][name] = m
    log.info("baselines  val: persistence MAE=%.3f  rolling4w MAE=%.3f | "
             "test: persistence MAE=%.3f  rolling4w MAE=%.3f",
             baselines["val"]["persistence"]["MAE"], baselines["val"]["rolling_4w_mean"]["MAE"],
             baselines["test"]["persistence"]["MAE"], baselines["test"]["rolling_4w_mean"]["MAE"])

    # ---- 2. candidates -----------------------------------------------------
    experiments = {}
    for name, cfg in CONFIGS.items():
        y = np.log1p(ytr) if cfg["target_transform"] == "log1p" else ytr
        model = HistGradientBoostingRegressor(categorical_features=pre.categorical_mask,
                                              random_state=SEED, **cfg["params"])
        t0 = time.time()
        model.fit(Xtr, np.clip(y, 0, None))
        secs = round(time.time() - t0, 1)
        preds = {}
        for k, X in Xs.items():
            raw = model.predict(X)
            preds[k] = np.clip(np.expm1(raw) if cfg["target_transform"] == "log1p" else raw,
                               0, None)
        mv = regression_metrics(ys["val"], preds["val"])
        mv["poisson_deviance"] = poisson_deviance(ys["val"], preds["val"])
        experiments[name] = {"question": cfg["question"], "params": cfg["params"],
                             "target_transform": cfg["target_transform"],
                             "fit_seconds": secs, "val": mv, "_model": model, "_preds": preds}
        log.info("  %-22s val MAE=%7.3f  RMSE=%8.3f  poisson_dev=%7.3f  (%ss)",
                 name, mv["MAE"], mv["RMSE"], mv["poisson_deviance"], secs)

    selected = min(experiments, key=lambda k: experiments[k]["val"][PRIMARY_METRIC])
    chosen = experiments[selected]
    log.info("selected on validation %s: %s", PRIMARY_METRIC, selected)

    # ---- 3. test, scored once ---------------------------------------------
    test_metrics = regression_metrics(ys["test"], chosen["_preds"]["test"])
    test_metrics["poisson_deviance"] = poisson_deviance(ys["test"], chosen["_preds"]["test"])

    comparison = {}
    for fold, model_m in (("val", chosen["val"]), ("test", test_metrics)):
        comparison[fold] = {}
        for bname, bm in baselines[fold].items():
            comparison[fold][bname] = {
                "model_MAE": model_m["MAE"], "baseline_MAE": bm["MAE"],
                "mae_improvement": round(1 - model_m["MAE"] / bm["MAE"], 4) if bm["MAE"] else None,
                "model_wins": bool(model_m["MAE"] < bm["MAE"]),
            }

    # the verdict is DERIVED, not written: a losing model reports as losing
    primary = "rolling_4w_mean"
    wins_val = comparison["val"][primary]["model_wins"]
    wins_test = comparison["test"][primary]["model_wins"]
    if wins_val and wins_test:
        verdict = ("beats the rolling 4-week mean on both validation and test — the primary "
                   "benchmark is cleared")
        recommend = "the model is a candidate; confirm on a wider evaluation window"
    elif wins_test and not wins_val:
        verdict = (f"LOSES to the rolling 4-week mean on validation "
                   f"({comparison['val'][primary]['mae_improvement']:+.2%}) and beats it only on "
                   f"test ({comparison['test'][primary]['mae_improvement']:+.2%}) — not a stable "
                   f"improvement")
        recommend = "ship the rolling 4-week mean; the model has not demonstrated a real edge"
    elif wins_val and not wins_test:
        verdict = "beats the rolling 4-week mean on validation but loses on test"
        recommend = "ship the rolling 4-week mean; the validation win did not transfer"
    else:
        verdict = "LOSES to the rolling 4-week mean on both validation and test"
        recommend = "ship the rolling 4-week mean; a gradient-boosted model adds nothing here"

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
        "primary_benchmark": primary,
        "notes": data.notes,
    }
    paths = save_model(TASK, chosen["_model"], pre, meta)

    payload = {
        **meta, "artifacts": paths,
        "baseline_definitions": {
            "persistence": "future_incident_count = this week's incident_count",
            "rolling_4w_mean": "mean weekly count over weeks t-3..t (PRIMARY benchmark)",
            "rolling_4w_mean_lagged": "mean weekly count over weeks t-4..t-1",
        },
        "baselines": baselines,
        "metrics": {"val": chosen["val"], "test": test_metrics},
        "comparison": comparison,
        "verdict": verdict, "recommendation": recommend,
        "experiments": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                        for k, v in experiments.items()},
        "leakage_note": ("Every predictor is a trailing quantity ending at or before week t, "
                         "while the target is week t+1. The rolling baselines are computed on "
                         "the full sorted panel so that history is not truncated at a fold "
                         "boundary. tests/test_pipeline.py asserts the rolling baseline does "
                         "not move when every future value is rewritten."),
    }
    write_report(TASK, payload, _markdown(payload))
    log.info("test: model MAE=%.3f | rolling4w=%.3f | persistence=%.3f",
             test_metrics["MAE"], baselines["test"]["rolling_4w_mean"]["MAE"],
             baselines["test"]["persistence"]["MAE"])
    log.info("VERDICT: the model %s", verdict)
    log.info("RECOMMENDATION: %s", recommend)
    return 0


def _markdown(r: dict) -> str:
    K = ["MAE", "RMSE", "median_AE", "poisson_deviance", "R2"]
    L = [f"# Hotspot model — {r['target']}", "",
         f"Trained {r['trained_at_utc']} · seed {r['environment']['seed']} · "
         f"scikit-learn {r['environment']['scikit_learn']}", "",
         f"**Dataset** `{r['dataset']}` · **split column** `{r['split_column']}` "
         f"({r['split_counts']}) · **features** {r['n_features']}", "",
         f"**Selected configuration:** `{r['selected_config']}` "
         f"(chosen on {r['selection_metric']})", "",
         f"> **VERDICT: the model {r['verdict']}.**", "",
         f"> **Recommendation:** {r['recommendation']}.", ""]
    for fold in ("val", "test"):
        L += [f"## {'Validation' if fold == 'val' else 'Test'}", "",
              "| Predictor | " + " | ".join(K) + " |", "|---" * (len(K) + 1) + "|",
              fmt_row("**model**", r["metrics"][fold], K)]
        for b in ("rolling_4w_mean", "rolling_4w_mean_lagged", "persistence"):
            label = f"**{b}**" if b == r["primary_benchmark"] else b
            L.append(fmt_row(label, r["baselines"][fold][b], K))
        L += ["", "| Compared with | model MAE | baseline MAE | improvement | model wins |",
              "|---|---|---|---|---|"]
        for b, c in r["comparison"][fold].items():
            imp = "—" if c["mae_improvement"] is None else f"{c['mae_improvement']:+.2%}"
            L.append(f"| {b} | {c['model_MAE']:.3f} | {c['baseline_MAE']:.3f} | {imp} | "
                     f"{'yes' if c['model_wins'] else '**no**'} |")
        L.append("")
    L += ["## Baseline definitions", ""]
    for k, v in r["baseline_definitions"].items():
        L.append(f"- `{k}` — {v}")
    L += ["", "## Configurations tried", "",
          "| Config | Question | val MAE | val RMSE | val Poisson dev. |", "|---|---|---|---|---|"]
    for name, e in r["experiments"].items():
        mark = "**" if name == r["selected_config"] else ""
        L.append(f"| {mark}{name}{mark} | {e['question']} | {e['val']['MAE']:.3f} | "
                 f"{e['val']['RMSE']:.3f} | {e['val']['poisson_deviance']:.3f} |")
    L += ["", "## Leakage", "", f"> {r['leakage_note']}", "",
          "## Features", "", ", ".join(f"`{c}`" for c in r["features"]), "",
          "## Notes", ""] + [f"- {n}" for n in r["notes"]] + [""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
