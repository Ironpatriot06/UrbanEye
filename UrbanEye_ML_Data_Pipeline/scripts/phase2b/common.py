"""
Phase 2B — shared machinery for the improvement experiments.

Nothing in this package writes to the Phase 2A artifacts. The original
`models/{resolution,sla,hotspot}/` and `reports/model_training/` are inputs
only; Phase 2B writes to

    data/processed/phase2b/      rebuilt task tables (gitignored, regenerable)
    experiments/phase2b/<task>/  one JSON record per experiment (gitignored)
    reports/phase2b/             tracked summaries, experiment log, final report
    models/<task>_v2/            new artifacts, only after they pass evaluation

`_common.load_model("<task>_v2")` loads a v2 artifact exactly as it loads v1.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.train._common import SEED, regression_metrics  # noqa: F401  (re-exported)
from scripts.utils.paths import p, repo_root

ROOT = repo_root()
REPORT_DIR = ROOT / "reports" / "phase2b"
EXPERIMENT_DIR = ROOT / "experiments" / "phase2b"
DATA_DIR = p("processed", "phase2b")
MAPPING_CSV = ROOT / "config" / "urbaneye_category_mapping.csv"

# ---------------------------------------------------------------------------
# UrbanEye policy, mirrored from backend/app/models/incident.py.
# tests/test_phase2b.py parses the backend source and fails if these drift.
# ---------------------------------------------------------------------------
UE_CATEGORIES = ["POTHOLE", "FLOOD", "FIRE_HAZARD", "GARBAGE", "STREETLIGHT", "OTHER"]
UE_SUPPORTED = ["POTHOLE", "FLOOD", "GARBAGE", "STREETLIGHT"]  # have mapped training data
UE_UNSUPPORTED = ["FIRE_HAZARD", "OTHER"]
DEFAULT_SLA_HOURS = {"P1": 4, "P2": 24, "P3": 72, "P4": 168}
CATEGORY_DEFAULT_PRIORITY = {"FIRE_HAZARD": "P1", "FLOOD": "P2", "POTHOLE": "P3",
                             "STREETLIGHT": "P3", "GARBAGE": "P4", "OTHER": "P3"}


def default_sla_hours(ue_category: str) -> int:
    return DEFAULT_SLA_HOURS[CATEGORY_DEFAULT_PRIORITY[ue_category]]


# ---------------------------------------------------------------------------
# data cleaning decisions (each one documented in PHASE_2B_REPORT.md §4)
# ---------------------------------------------------------------------------
# Chicago before this date is an extraction artifact: 4–38k rows/month against
# 55–119k afterwards, with median resolution 845–9,071 h against 80–270 h. The
# transition month is February 2019 (median 845 h), so the clean period starts
# on 1 March 2019, one month later than the Phase 2A recommendation.
CHICAGO_CLEAN_FROM = pd.Timestamp("2019-03-01", tz="UTC")
MIN_RESOLUTION_HOURS = 1.0 / 60.0   # same as the pipeline's instant-closure rule
MAX_RESOLUTION_HOURS = 87_600.0     # same 10-year cap as the pipeline


def clean_period_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows outside the Chicago pre-March-2019 artifact."""
    return ~((df["source_dataset"] == "chicago311") & (df["reported_at"] < CHICAGO_CLEAN_FROM))


# ---------------------------------------------------------------------------
# category mapping
# ---------------------------------------------------------------------------
def load_ue_mapping(path: Path = MAPPING_CSV) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    for r in csv.DictReader(lines):
        rows.append({k: (v or "").strip() for k, v in r.items()})
    return rows


def map_to_urbaneye(df: pd.DataFrame, rules: list[dict] | None = None) -> pd.DataFrame:
    """
    Return (ue_category, ue_confidence) for each row. First matching rule wins.
    Rows matched by no rule get <NA> — "never considered", distinct from an
    explicit UNSUPPORTED decision.
    """
    rules = rules if rules is not None else load_ue_mapping()
    out_cat = pd.Series(pd.NA, index=df.index, dtype="string")
    out_conf = pd.Series(pd.NA, index=df.index, dtype="string")
    src = df["source_dataset"].astype("string")
    raw = df["source_category_raw"].astype("string")
    sub = df["subcategory"].astype("string")
    unassigned = pd.Series(True, index=df.index)
    for r in rules:
        m = unassigned & (src == r["source_dataset"]) & (raw == r["source_category_raw"])
        if r["subcategory"] != "*":
            m &= sub == r["subcategory"]
        m = m.fillna(False)
        out_cat[m] = r["urbaneye_category"]
        out_conf[m] = r["confidence"] or pd.NA
        unassigned &= ~m
    return pd.DataFrame({"ue_category": out_cat, "ue_confidence": out_conf})


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def environment() -> dict:
    import joblib
    import pyarrow
    import scipy
    import sklearn
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__, "joblib": joblib.__version__, "pyarrow": pyarrow.__version__,
        "seed": SEED,
    }


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# experiment registry
# ---------------------------------------------------------------------------
class ExperimentLog:
    """
    Every fit is recorded — failures included — as one JSON file under
    experiments/phase2b/<task>/ and one line in the task's registry. The
    tracked summary in reports/phase2b/ is regenerated from the registry, so
    the report can only show experiments that actually ran.
    """

    def __init__(self, task: str, dataset: dict):
        self.task = task
        self.dataset = dataset
        self.dir = EXPERIMENT_DIR / task
        self.dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    def record(self, exp_id: str, **fields) -> dict:
        rec = {"experiment_id": exp_id, "task": self.task, "timestamp_utc": now_utc(),
               "dataset": self.dataset, "environment": environment(), **fields}
        write_json(self.dir / f"{exp_id}.json", rec)
        self.records.append(rec)
        return rec

    def flush(self) -> Path:
        out = write_json(REPORT_DIR / f"experiment_log_{self.task}.json",
                         {"task": self.task, "n_experiments": len(self.records),
                          "experiments": self.records})
        return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def resolution_metrics(y, pred, lo=None, hi=None) -> dict:
    y = np.asarray(y, float)
    pred = np.clip(np.asarray(pred, float), 0, None)
    m = regression_metrics(y, pred)
    m["MAE_log1p"] = float(np.mean(np.abs(np.log1p(y) - np.log1p(pred))))
    if lo is not None and hi is not None:
        m["interval_coverage"] = float(np.mean((y >= lo) & (y <= hi)))
        m["interval_median_width_h"] = float(np.median(np.asarray(hi) - np.asarray(lo)))
    return m


def paired_bootstrap_mae_gain(y, pred_a, pred_b, n_boot: int = 300, seed: int = SEED,
                              groups=None) -> dict:
    """
    95% CI for MAE(a) − MAE(b) (positive = b is better). With `groups`, whole
    groups are resampled (cluster bootstrap) so correlated rows do not shrink
    the interval.
    """
    rng = np.random.default_rng(seed)
    ea = np.abs(np.asarray(y, float) - np.asarray(pred_a, float))
    eb = np.abs(np.asarray(y, float) - np.asarray(pred_b, float))
    d = ea - eb
    if groups is None:
        n = len(d)
        stats = [d[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    else:
        g = pd.Series(d).groupby(np.asarray(groups))
        s, c = g.sum().to_numpy(), g.size().to_numpy()
        k = len(s)
        stats = []
        for _ in range(n_boot):
            i = rng.integers(0, k, k)
            stats.append(s[i].sum() / c[i].sum())
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return {"gain_mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "significant": bool(lo > 0 or hi < 0)}


def classification_summary(y, prob, threshold: float) -> dict:
    from scripts.train._common import classification_metrics
    return classification_metrics(y, prob, threshold)
