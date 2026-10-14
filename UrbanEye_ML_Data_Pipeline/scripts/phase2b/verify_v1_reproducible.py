#!/usr/bin/env python3
"""
Phase 2B — prove the ORIGINAL artifacts were left intact and still reproduce.

Loads models/{resolution,sla,hotspot}/ exactly as Phase 2A did, scores them on
their own saved test folds, and compares with the tracked
reports/model_training/<task>_results.json. Also records SHA-256 of every v1
artifact. Read-only with respect to everything except
reports/phase2b/v1_reproduction.json.

    python scripts/phase2b/verify_v1_reproducible.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from scripts.phase2b.common import REPORT_DIR, environment, now_utc, sha256_file, write_json
from scripts.train._common import (classification_metrics, load_model, load_task,
                                   regression_metrics)
from scripts.utils.paths import repo_root

TOL = 1e-9


def main() -> int:
    root = repo_root()
    out = {"generated_at_utc": now_utc(), "environment": environment(), "tasks": {}}
    ok_all = True
    for task, dataset in (("resolution", "resolution_ml"), ("sla", "sla_ml"),
                          ("hotspot", "hotspot_ml")):
        model, pre, meta = load_model(task)
        data = load_task(task, dataset, meta["split_column"])
        te = data.fold("test")
        y = te[data.target].astype(float).to_numpy()
        reported = json.loads((root / "reports/model_training" / f"{task}_results.json")
                              .read_text())["metrics"]["test"]
        X = pre.transform(te)
        if task == "sla":
            prob = model.predict_proba(X)[:, 1]
            got = classification_metrics(y, prob, reported["threshold"])
            keys = ["PR_AUC", "ROC_AUC", "precision", "recall", "f1", "brier"]
        else:
            raw = model.predict(X)
            pred = np.clip(np.expm1(raw) if meta.get("target_transform") == "log1p" else raw,
                           0, None)
            got = regression_metrics(y, pred)
            keys = ["MAE", "RMSE", "median_AE", "R2"]
        cmp = {k: {"reported": reported[k], "reproduced": got[k],
                   "match": abs(reported[k] - got[k]) <= TOL * max(1, abs(reported[k]))}
               for k in keys}
        ok = all(v["match"] for v in cmp.values())
        ok_all &= ok
        d = root / "models" / task
        out["tasks"][task] = {
            "match": ok, "test_rows": int(len(te)), "metrics": cmp,
            "artifact_sha256": {f.name: sha256_file(f) for f in sorted(d.iterdir()) if f.is_file()},
        }
        print(f"{task}: {'REPRODUCED' if ok else 'MISMATCH'} "
              + ", ".join(f"{k}={got[k]:.4f}" for k in keys))
    out["all_reproduced"] = ok_all
    write_json(REPORT_DIR / "v1_reproduction.json", out)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
