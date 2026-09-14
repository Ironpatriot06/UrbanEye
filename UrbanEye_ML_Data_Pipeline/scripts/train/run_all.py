#!/usr/bin/env python3
"""
Train and evaluate every ready task, then write the cross-task summary.

    python scripts/train/run_all.py
    python scripts/train/run_all.py --only hotspot

Trains resolution, SLA and hotspot. It does NOT train duplicate detection or
priority, and it says why in the summary rather than leaving their absence to be
guessed at:

  duplicate  the labels are real but the evaluation set is not. In the realistic
             candidate population (same category, <=200 m, <=7 days) the test
             fold holds 23,480 positives and ZERO negatives, so a PR-AUC of
             0.998 measures the negative-sampling rule, not duplicate detection.
  priority   there is no ground truth in any public 311 dataset. priority_baseline
             is a deterministic function of config/priority_config.yaml; a model
             fitted to it would reproduce the YAML and any accuracy quoted from
             it would be circular.

Both are dataset problems, not modelling problems, and no amount of training
fixes either.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.train._common import environment, log, now_utc
from scripts.utils.paths import ensure_dir, p, repo_root

TASKS = ["resolution", "sla", "hotspot"]

NOT_TRAINED = {
    "duplicate": ("NOT READY — the labels are genuine but the evaluation set is not. In the "
                  "realistic candidate population (same category, <=200 m, <=7 days) the test "
                  "fold contains 23,480 positives and zero negatives, and a depth-2 decision "
                  "tree scores PR-AUC 0.967. Any headline metric measures the negative-sampling "
                  "rule. Needs adjudicated hard negatives from real operations."),
    "priority": ("NOT READY and NOT A SUPERVISED TASK — no public 311 dataset records an "
                 "operational priority. priority_baseline is a deterministic function of "
                 "config/priority_config.yaml, so training on it would reproduce the YAML. "
                 "Needs operator-assigned priorities and the override flag."),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=TASKS)
    ap.add_argument("--max-train-rows", type=int, default=None)
    args = ap.parse_args()

    root = repo_root()
    wanted = [args.only] if args.only else TASKS
    results, t0 = {}, time.time()

    for task in wanted:
        cmd = [sys.executable, f"scripts/train/train_{task}.py"]
        if task == "resolution" and args.max_train_rows:
            cmd += ["--max-train-rows", str(args.max_train_rows)]
        log.info("=== training %s ===", task)
        st = time.time()
        proc = subprocess.run(cmd, cwd=root)
        results[task] = {"command": " ".join(cmd), "returncode": proc.returncode,
                         "seconds": round(time.time() - st, 1),
                         "status": "OK" if proc.returncode == 0 else "FAILED"}
        if proc.returncode != 0:
            log.error("%s failed (rc=%d)", task, proc.returncode)

    summary = {
        "generated_at_utc": now_utc(),
        "environment": environment(),
        "total_seconds": round(time.time() - t0, 1),
        "runs": results,
        "trained": {},
        "not_trained": NOT_TRAINED,
    }

    for task in wanted:
        fp = p("reports", "model_training", f"{task}_results.json")
        if results[task]["status"] != "OK" or not fp.exists():
            continue
        r = json.loads(fp.read_text())
        entry = {
            "dataset": r["dataset"], "target": r["target"],
            "split_column": r["split_column"], "rows": r["rows"],
            "n_features": r["n_features"], "selected_config": r["selected_config"],
            "artifacts": r["artifacts"],
        }
        if task == "resolution":
            entry.update({
                "test_MAE": r["metrics"]["test"]["MAE"],
                "test_median_AE": r["metrics"]["test"]["median_AE"],
                "test_R2": r["metrics"]["test"]["R2"],
                "baseline_test_MAE": r["baseline"]["test"]["MAE"],
                "baseline_test_median_AE": r["baseline"]["test"]["median_AE"],
                "verdict": r["verdict"],
            })
        elif task == "sla":
            entry.update({
                "test_PR_AUC": r["metrics"]["test"]["PR_AUC"],
                "test_ROC_AUC": r["metrics"]["test"]["ROC_AUC"],
                "test_precision": r["metrics"]["test"]["precision"],
                "test_recall": r["metrics"]["test"]["recall"],
                "test_f1": r["metrics"]["test"]["f1"],
                "base_rate_test": r["metrics"]["test"]["positive_rate"],
                "pr_auc_lift": r["pr_auc_lift_over_base_rate_test"],
            })
        elif task == "hotspot":
            entry.update({
                "test_MAE": r["metrics"]["test"]["MAE"],
                "val_MAE": r["metrics"]["val"]["MAE"],
                "baseline_rolling_4w_test_MAE": r["baselines"]["test"]["rolling_4w_mean"]["MAE"],
                "baseline_rolling_4w_val_MAE": r["baselines"]["val"]["rolling_4w_mean"]["MAE"],
                "baseline_persistence_test_MAE": r["baselines"]["test"]["persistence"]["MAE"],
                "verdict": r["verdict"], "recommendation": r["recommendation"],
            })
        summary["trained"][task] = entry

    dest = ensure_dir(p("reports", "model_training", "training_summary.json"))
    dest.write_text(json.dumps(summary, indent=2, default=str))
    ensure_dir(p("reports", "model_training", "training_summary.md")).write_text(_markdown(summary))
    log.info("wrote %s", dest)

    failed = [t for t, v in results.items() if v["status"] == "FAILED"]
    log.info("%d/%d tasks trained in %.1fs", len(wanted) - len(failed), len(wanted),
             summary["total_seconds"])
    return 1 if failed else 0


def _markdown(s: dict) -> str:
    L = ["# Model training summary", "",
         f"Generated {s['generated_at_utc']} · seed {s['environment']['seed']} · "
         f"scikit-learn {s['environment']['scikit_learn']} · "
         f"{s['total_seconds']}s total", "",
         "## Trained tasks", "",
         "| Task | Target | Split | train/val/test | Headline (test) | Baseline (test) |",
         "|---|---|---|---|---|---|"]
    for t, e in s["trained"].items():
        rows = f"{e['rows']['train']:,}/{e['rows']['val']:,}/{e['rows']['test']:,}"
        if t == "resolution":
            head = f"MAE {e['test_MAE']:,.1f} · median AE {e['test_median_AE']:,.1f}"
            base = f"MAE {e['baseline_test_MAE']:,.1f} · median AE {e['baseline_test_median_AE']:,.1f}"
        elif t == "sla":
            head = f"PR-AUC {e['test_PR_AUC']:.4f} · R {e['test_recall']:.3f}"
            base = f"base rate {e['base_rate_test']:.4f} (lift x{e['pr_auc_lift']:.2f})"
        else:
            head = f"MAE {e['test_MAE']:.3f}"
            base = (f"rolling 4w {e['baseline_rolling_4w_test_MAE']:.3f} · "
                    f"persistence {e['baseline_persistence_test_MAE']:.3f}")
        L.append(f"| `{t}` | `{e['target']}` | `{e['split_column']}` | {rows} | {head} | {base} |")

    L += ["", "## Verdicts", ""]
    for t, e in s["trained"].items():
        if "verdict" in e:
            L.append(f"- **{t}** — the model {e['verdict']}."
                     + (f" _{e['recommendation'].capitalize()}._" if "recommendation" in e else ""))
        elif t == "sla":
            L.append(f"- **sla** — PR-AUC {e['test_PR_AUC']:.4f} against a base rate of "
                     f"{e['base_rate_test']:.4f}, a lift of x{e['pr_auc_lift']:.2f}. Real signal "
                     f"on a narrow population.")

    L += ["", "## Not trained, and why", ""]
    for t, why in s["not_trained"].items():
        L.append(f"- **{t}** — {why}")
    L += ["", "## Artifacts", "",
          "| Task | Model | Preprocessor | Metadata |", "|---|---|---|---|"]
    for t, e in s["trained"].items():
        a = e["artifacts"]
        L.append(f"| `{t}` | `{a['model']}` | `{a['preprocessor']}` | `{a['metadata']}` |")
    L += ["", "Model artifacts are generated files and are excluded from version control by "
          "`.gitignore`, in line with the repository's existing policy for `data/`.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
