#!/usr/bin/env python3
"""
Train and evaluate the SLA-breach model.

    target      sla_breach  (bool: resolution_time_hours > sla_target_hours)
    dataset     data/processed/ml/sla_ml.parquet
    split       `split` — this table is NYC-only, so `split_global` would put
                every row in train. See scripts/train/_common.choose_split_column.

The target is NOT invented here. It comes from the pipeline: NYC publishes a
`Due Date` set at intake, the pipeline derives `sla_target_hours` from it, and
`sla_breach` is the observed comparison against the realised resolution time.
The deadline itself is a legitimate predictor (known at intake); the realised
resolution time is not, and is excluded.

PROTOCOL
--------
1. Baseline first: predict the training positive rate for every row. Its PR-AUC
   is the base rate, and nothing below that is a model.
2. Three configurations, each answering one question:
     A  default                    is there signal at all?
     B  class_weight='balanced'    does correcting the 10% imbalance help?
     C  slower/longer              is A under-fitted?
   Imbalance is handled inside the fit via class weights — never by resampling
   the data, which would change the validation and test distributions away from
   the prevalence any deployment would actually see.
3. Selection on VALIDATION PR-AUC. Accuracy is deliberately not reported as a
   headline: predicting "never breaches" scores ~90% accuracy and is useless.
4. The decision threshold is chosen on VALIDATION (max F1) and frozen for test.

    python scripts/train/train_sla.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.train._common import (SEED, FeaturePreprocessor, best_threshold_by_f1,
                                   calibration_bins, classification_metrics, environment,
                                   load_task, log, now_utc, save_model, write_report)

TASK = "sla"
DATASET = "sla_ml"
PRIMARY_METRIC = "PR_AUC"

CONFIGS = {
    "A_default": {
        "question": "is there any signal beyond the base rate?",
        "params": dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                       min_samples_leaf=20, l2_regularization=0.0),
    },
    "B_balanced": {
        "question": "does correcting the ~10% class imbalance help?",
        "params": dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                       min_samples_leaf=20, l2_regularization=0.0,
                       class_weight="balanced"),
    },
    "C_slow_regularised": {
        "question": "does a slower, more regularised fit generalise better on 13k rows?",
        "params": dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                       min_samples_leaf=40, l2_regularization=1.0),
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-column", default="split_global", choices=["split_global", "split"])
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier

    data = load_task(TASK, DATASET, args.split_column)
    tr, va, te = data.fold("train"), data.fold("val"), data.fold("test")
    log.info("%s: train=%d val=%d test=%d (split column: %s)",
             TASK, len(tr), len(va), len(te), data.split_column)
    if not len(va) or not len(te):
        raise SystemExit("validation or test fold is empty — refusing to train")

    pre = FeaturePreprocessor(data.predictors).fit(tr)
    Xtr, Xva, Xte = pre.transform(tr), pre.transform(va), pre.transform(te)

    def y_of(part):
        return part[data.target].astype("boolean").fillna(False).to_numpy(bool).astype(int)
    ytr, yva, yte = y_of(tr), y_of(va), y_of(te)
    if len(np.unique(ytr)) < 2:
        raise SystemExit("training fold has a single class — refusing to train")

    # ---- 1. baseline -------------------------------------------------------
    base_rate = float(ytr.mean())
    baseline = {
        "definition": "predict the TRAINING positive rate for every row",
        "train_positive_rate": base_rate,
        "val": classification_metrics(yva, np.full(len(yva), base_rate), 0.5),
        "test": classification_metrics(yte, np.full(len(yte), base_rate), 0.5),
    }
    log.info("baseline (train positive rate %.4f): val PR-AUC=%.4f  test PR-AUC=%.4f",
             base_rate, baseline["val"]["PR_AUC"], baseline["test"]["PR_AUC"])

    # ---- 2. candidates -----------------------------------------------------
    experiments = {}
    for name, cfg in CONFIGS.items():
        model = HistGradientBoostingClassifier(
            categorical_features=pre.categorical_mask, random_state=SEED, **cfg["params"])
        t0 = time.time()
        model.fit(Xtr, ytr)
        secs = round(time.time() - t0, 1)
        pv = model.predict_proba(Xva)[:, 1]
        thr = best_threshold_by_f1(yva, pv)
        experiments[name] = {
            "question": cfg["question"], "params": cfg["params"], "fit_seconds": secs,
            "val": classification_metrics(yva, pv, thr),
            "threshold_from_validation": thr,
            "_model": model, "_val_prob": pv,
        }
        log.info("  %-20s val PR-AUC=%.4f  ROC-AUC=%.4f  F1@%.3f=%.4f  (%ss)",
                 name, experiments[name]["val"]["PR_AUC"],
                 experiments[name]["val"]["ROC_AUC"] or float("nan"),
                 thr, experiments[name]["val"]["f1"], secs)

    selected = max(experiments, key=lambda k: experiments[k]["val"][PRIMARY_METRIC])
    chosen = experiments[selected]
    thr = chosen["threshold_from_validation"]
    log.info("selected on validation %s: %s (threshold %.4f frozen from validation)",
             PRIMARY_METRIC, selected, thr)

    # ---- 3. test, scored once ---------------------------------------------
    prob_te = chosen["_model"].predict_proba(Xte)[:, 1]
    test_metrics = classification_metrics(yte, prob_te, thr)
    lift = (test_metrics["PR_AUC"] / baseline["test"]["PR_AUC"]
            if baseline["test"]["PR_AUC"] else None)

    meta = {
        "task": TASK, "dataset": DATASET, "target": data.target,
        "trained_at_utc": now_utc(), "environment": environment(),
        "split_column": data.split_column, "split_counts": data.counts(),
        "rows": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "rows_dropped_missing_target": data.rows_dropped_missing_target,
        "class_balance": {"train_positive_rate": base_rate,
                          "val_positive_rate": float(yva.mean()),
                          "test_positive_rate": float(yte.mean())},
        "n_features": len(data.predictors), "features": data.predictors,
        "preprocessing": pre.describe(),
        "selected_config": selected, "selection_metric": f"validation {PRIMARY_METRIC}",
        "model_params": CONFIGS[selected]["params"],
        "decision_threshold": thr,
        "threshold_policy": "max F1 on VALIDATION, frozen before test is scored",
        "imbalance_policy": ("handled by class_weight inside the fit where selected; the data "
                             "is never resampled, so val and test keep the prevalence a "
                             "deployment would actually see"),
        "notes": data.notes,
    }
    paths = save_model(TASK, chosen["_model"], pre, meta)

    payload = {
        **meta, "artifacts": paths, "baseline": baseline,
        "metrics": {"val": chosen["val"], "test": test_metrics},
        "calibration_test": calibration_bins(yte, prob_te),
        "pr_auc_lift_over_base_rate_test": lift,
        "experiments": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                        for k, v in experiments.items()},
        "headline_metric_note": ("Accuracy is NOT reported as a headline. With a ~10% positive "
                                 "rate, always predicting 'no breach' scores ~90% accuracy and "
                                 "catches nothing. PR-AUC against the base rate is the number "
                                 "that means something."),
        "population_caveat": ("NYC only, January-March 2010, and only the complaint types that "
                              "carry a Due Date. This is a research model for that slice, not a "
                              "general SLA-risk model, and must never be presented as one."),
    }
    write_report(TASK, payload, _markdown(payload))
    log.info("test: PR-AUC=%.4f (base rate %.4f, lift x%.2f)  ROC-AUC=%.4f  P=%.3f R=%.3f F1=%.3f",
             test_metrics["PR_AUC"], test_metrics["positive_rate"], lift or float("nan"),
             test_metrics["ROC_AUC"] or float("nan"), test_metrics["precision"],
             test_metrics["recall"], test_metrics["f1"])
    return 0


def _markdown(r: dict) -> str:
    def row(label, m):
        return (f"| {label} | {m['PR_AUC']:.4f} | "
                f"{m['ROC_AUC']:.4f} | " if m["ROC_AUC"] is not None else
                f"| {label} | {m['PR_AUC']:.4f} | — | ") + \
               (f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                f"{m['brier']:.4f} | {m['positive_rate']:.4f} |")
    cm = r["metrics"]["test"]["confusion_matrix"]
    L = [f"# SLA breach model — {r['target']}", "",
         f"Trained {r['trained_at_utc']} · seed {r['environment']['seed']} · "
         f"scikit-learn {r['environment']['scikit_learn']}", "",
         f"**Dataset** `{r['dataset']}` · **split column** `{r['split_column']}` "
         f"({r['split_counts']}) · **features** {r['n_features']}", "",
         f"**Selected configuration:** `{r['selected_config']}` "
         f"(chosen on {r['selection_metric']}) · "
         f"**threshold {r['decision_threshold']:.4f}** ({r['threshold_policy']})", "",
         f"> {r['headline_metric_note']}", "",
         "## Metrics", "",
         "| Fold | PR-AUC | ROC-AUC | precision | recall | F1 | Brier | positive rate |",
         "|---|---|---|---|---|---|---|---|",
         row("validation — model", r["metrics"]["val"]),
         row("validation — baseline", r["baseline"]["val"]),
         row("**test — model**", r["metrics"]["test"]),
         row("**test — baseline**", r["baseline"]["test"]), "",
         f"Baseline: {r['baseline']['definition']} "
         f"({r['baseline']['train_positive_rate']:.4f}).", "",
         f"PR-AUC lift over the base rate on test: **x{r['pr_auc_lift_over_base_rate_test']:.2f}**",
         "", "## Confusion matrix (test, at the frozen threshold)", "",
         "| | predicted no breach | predicted breach |", "|---|---|---|",
         f"| **actual no breach** | {cm['tn']:,} | {cm['fp']:,} |",
         f"| **actual breach** | {cm['fn']:,} | {cm['tp']:,} |", "",
         "## Calibration (test)", "",
         "| Predicted probability | n | mean predicted | observed rate |", "|---|---|---|---|"]
    for b in r["calibration_test"]:
        L.append(f"| {b['bin']} | {b['n']:,} | {b['mean_predicted']:.4f} | "
                 f"{b['observed_rate']:.4f} |")
    L += ["", "## Configurations tried", "",
          "| Config | Question | val PR-AUC | val ROC-AUC | val F1 |", "|---|---|---|---|---|"]
    for name, e in r["experiments"].items():
        mark = "**" if name == r["selected_config"] else ""
        roc = f"{e['val']['ROC_AUC']:.4f}" if e["val"]["ROC_AUC"] is not None else "—"
        L.append(f"| {mark}{name}{mark} | {e['question']} | {e['val']['PR_AUC']:.4f} | "
                 f"{roc} | {e['val']['f1']:.4f} |")
    L += ["", "## Class balance", "",
          f"- train {r['class_balance']['train_positive_rate']:.4f} · "
          f"val {r['class_balance']['val_positive_rate']:.4f} · "
          f"test {r['class_balance']['test_positive_rate']:.4f}",
          f"- {r['imbalance_policy']}", "",
          "## Features", "", ", ".join(f"`{c}`" for c in r["features"]), "",
          "## Caveat", "", f"> {r['population_caveat']}", "",
          "## Notes", ""] + [f"- {n}" for n in r["notes"]] + [""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
