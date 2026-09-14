"""
Shared machinery for the UrbanEye+ training framework.

The data pipeline decides what may be learned from; this package decides
nothing. Every predictor comes from the task manifest, every split column comes
from the documented policy, and every transform is fitted on the training fold
alone. If a column is not declared in the manifest it does not reach a model,
even if it is sitting in the parquet.

WHY A CLASS AND NOT THE CLOSURE IN scripts/baselines/
-----------------------------------------------------
`run_baselines.py` builds its encoder inside a closure. That is fine for a
throwaway score, but a closure cannot be saved, so the exact transform used at
training time cannot be replayed at inference time. `FeaturePreprocessor` holds
the same logic in a picklable object: the fitted vocabulary, the rare-level
folding, the column order and the unknown-category policy all travel with the
model. A model without its preprocessor is not a deployable artifact.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import ensure_dir, p, repo_root

log = get_logger("train")

SEED = 20260915

# Columns that may never be handed to a model, whatever a manifest says. This is
# a belt-and-braces guard: the manifests are already correct, and a future edit
# to one of them should not be able to quietly turn an outcome into an input.
FORBIDDEN_PREDICTORS = {
    "resolution_time_hours", "resolution_time_hours_log1p", "resolution_instant_closure",
    "target_is_censored", "sla_breach", "sla_met", "closed_at", "status", "description",
    "resolution_description", "status_notes", "future_incident_count",
    "future_incident_flag", "priority_baseline", "priority_score", "priority_reasons",
    "priority_confidence", "priority_features_available", "priority_feature_coverage",
    "priority_method", "sla_hours_policy", "priority_label", "priority_label_source",
    "is_ground_truth", "split", "split_global", "incident_id", "incident_a", "incident_b",
}

# `sla_target_hours` is the one apparent exception and it is not one: the due
# date is set by the publisher at intake, so it is known at prediction time. It
# is a legitimate predictor for the SLA task and forbidden everywhere else,
# which is exactly how the manifests already declare it.
TASK_ALLOWED_EXCEPTIONS = {"sla": {"sla_target_hours"}}


# ---------------------------------------------------------------------------
# environment / provenance
# ---------------------------------------------------------------------------
def environment() -> dict:
    """Versions that actually affect the numbers, for reproducibility."""
    import sklearn
    import joblib
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "seed": SEED,
    }


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
@dataclass
class TaskData:
    name: str
    frame: pd.DataFrame
    manifest: dict
    predictors: list[str]
    target: str
    split_column: str
    rows_dropped_missing_target: int = 0
    notes: list[str] = field(default_factory=list)

    def fold(self, which: str) -> pd.DataFrame:
        return self.frame[self.frame[self.split_column] == which]

    def counts(self) -> dict:
        return {k: int(v) for k, v in self.frame[self.split_column].value_counts().items()}


def choose_split_column(df: pd.DataFrame, requested: str = "split_global") -> tuple[str, str]:
    """
    Pick the split column that matches the table, and say why.

    This mirrors scripts/baselines/run_baselines.py exactly — the training
    framework must not evaluate on a different footing from the baselines it is
    compared against.

    `split_global` is the right answer for a POOLED model: one wall clock, so no
    evaluation row predates a training row. It only means anything when the
    table actually spans several publishers. On a single-source table it
    degenerates — sla_ml is NYC-only and NYC is the earliest data in the corpus,
    so a global cut puts every row in train and leaves val and test empty. There
    the per-source column is already chronological on one clock.
    """
    if requested not in df.columns:
        return "split", f"'{requested}' not present in this table"
    grp = "source_dataset" if "source_dataset" in df.columns else (
        "city" if "city" in df.columns else None)
    n_src = int(df[grp].nunique()) if grp else 1
    if n_src <= 1:
        return "split", (f"single-source table ({grp}={df[grp].iloc[0] if grp else 'n/a'}); "
                         f"'{requested}' degenerates to an empty val/test")
    if not (df[requested] == "val").any() or not (df[requested] == "test").any():
        return "split", f"'{requested}' has an empty val or test fold on this table"
    return requested, f"multi-source table ({n_src} sources) — pooled evaluation on one wall clock"


def load_task(task: str, dataset: str, split_request: str = "split_global") -> TaskData:
    """Load a task table and its manifest, and resolve the declared predictors."""
    fp = p("processed", "ml", f"{dataset}.parquet")
    mf = p("processed", "ml", f"{dataset}.manifest.json")
    if not fp.exists() or not mf.exists():
        raise SystemExit(
            f"{dataset} is not built. Run the pipeline first:\n"
            f"    python scripts/run_pipeline.py")
    df = pd.read_parquet(fp)
    manifest = json.loads(mf.read_text())

    block = manifest.get("predictors") or manifest.get("features") or {}
    predictors = [c for c, v in block.items() if v.get("present")]
    target = (manifest.get("target") or {}).get("column")
    if not target:
        raise SystemExit(f"{dataset} declares no target — it is not a supervised task")

    allowed = TASK_ALLOWED_EXCEPTIONS.get(task, set())
    illegal = sorted((set(predictors) & FORBIDDEN_PREDICTORS) - allowed)
    if illegal:
        raise SystemExit(f"{dataset} manifest declares forbidden predictors: {illegal}")
    missing = [c for c in predictors + [target] if c not in df.columns]
    if missing:
        raise SystemExit(f"{dataset}: declared columns absent from the table: {missing}")

    split_col, why = choose_split_column(df, split_request)

    before = len(df)
    keep = df[target].notna()
    dropped = int((~keep).sum())
    df = df[keep].copy()

    notes = [f"split column: {split_col} — {why}"]
    if dropped:
        notes.append(f"{dropped:,} of {before:,} rows have no target and are excluded from "
                     f"training and evaluation alike (never silently imputed)")
    return TaskData(name=task, frame=df, manifest=manifest, predictors=predictors,
                    target=target, split_column=split_col,
                    rows_dropped_missing_target=dropped, notes=notes)


# ---------------------------------------------------------------------------
# preprocessing — fitted on TRAIN ONLY, and saveable
# ---------------------------------------------------------------------------
class FeaturePreprocessor:
    """
    Ordinal-encode categoricals, pass numerics through, in a fixed column order.

    Three deliberate choices, all of which must survive to inference:

    NULL IS A LEVEL. "no subcategory recorded" is information — for Chicago rows
    it is the single most informative thing about the column — so missing
    categorical values become an explicit `__MISSING__` level rather than being
    imputed or dropped.

    RARE LEVELS ARE FOLDED, NOT DROPPED. The booster caps categorical cardinality
    at 255 and `zone_key` / `subcategory` / `department` exceed it. The tail goes
    to `__OTHER__`. The vocabulary is built from TRAIN frequencies only, so no
    validation or test level can influence the encoding.

    UNKNOWN LEVELS ARE SAFE. Anything unseen at fit time — a new ward, a new
    department, a level that only exists in test — encodes to -1 and the model
    handles it as its own category. Inference never crashes and never refits.
    """

    def __init__(self, predictors: list[str], max_levels: int = 250):
        self.predictors = list(predictors)
        self.max_levels = int(max_levels)
        self.categorical_: list[str] = []
        self.numeric_: list[str] = []
        self.vocabulary_: dict[str, list[str]] = {}
        self.fitted_ = False

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _as_str(s: pd.Series) -> pd.Series:
        return s.astype("string").fillna("__MISSING__")

    def _fold(self, part: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.categorical_:
            col = self._as_str(part[c])
            allowed = self.vocabulary_[c]
            out[c] = col.where(col.isin(allowed), "__OTHER__")
        return pd.DataFrame(out, index=part.index)

    # -- api --------------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "FeaturePreprocessor":
        self.categorical_ = [c for c in self.predictors
                             if not pd.api.types.is_numeric_dtype(train[c])
                             or pd.api.types.is_bool_dtype(train[c])]
        self.numeric_ = [c for c in self.predictors if c not in self.categorical_]
        self.vocabulary_ = {}
        for c in self.categorical_:
            vc = self._as_str(train[c]).value_counts()
            levels = list(vc.head(self.max_levels).index)
            if len(vc) > self.max_levels:
                levels.append("__OTHER__")
            self.vocabulary_[c] = levels
        self.fitted_ = True
        return self

    def transform(self, part: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FeaturePreprocessor.transform called before fit")
        out = pd.DataFrame(index=part.index)
        for c in self.numeric_:
            out[c] = pd.to_numeric(part[c], errors="coerce").astype("float64")
        if self.categorical_:
            folded = self._fold(part)
            for c in self.categorical_:
                codes = {lvl: i for i, lvl in enumerate(self.vocabulary_[c])}
                out[c] = folded[c].map(codes).astype("float64").fillna(-1.0)
        return out[self.predictors]

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    @property
    def categorical_mask(self) -> list[bool]:
        """Boolean mask in predictor order, for HistGradientBoosting."""
        return [c in self.categorical_ for c in self.predictors]

    def describe(self) -> dict:
        return {
            "n_predictors": len(self.predictors),
            "categorical": self.categorical_,
            "numeric": self.numeric_,
            "max_levels_per_categorical": self.max_levels,
            "vocabulary_sizes": {c: len(v) for c, v in self.vocabulary_.items()},
            "missing_policy": "categorical NULL -> explicit __MISSING__ level",
            "rare_level_policy": f"levels outside the train top-{self.max_levels} -> __OTHER__",
            "unknown_policy": "levels unseen at fit time -> -1",
            "fitted_on": "training fold only",
        }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "median_AE": float(np.median(err)),
        "R2": float(r2_score(y_true, y_pred)),
        # tail behaviour: where a skewed target actually hurts
        "p90_AE": float(np.quantile(err, 0.90)),
        "p99_AE": float(np.quantile(err, 0.99)),
        "RMSE_log1p": float(np.sqrt(mean_squared_error(
            np.log1p(np.clip(y_true, 0, None)), np.log1p(np.clip(y_pred, 0, None))))),
    }


def classification_metrics(y_true, prob, threshold: float) -> dict:
    from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "PR_AUC": float(average_precision_score(y_true, prob)),
        "precision": float(pr), "recall": float(rc), "f1": float(f1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "brier": float(brier_score_loss(y_true, prob)),
    }
    try:
        out["ROC_AUC"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        out["ROC_AUC"] = None
    return out


def calibration_bins(y_true, prob, bins: int = 10) -> list[dict]:
    """Reliability table: does a predicted 0.3 actually happen 30% of the time?"""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        out.append({"bin": f"[{edges[b]:.1f},{edges[b + 1]:.1f})",
                    "n": int(m.sum()),
                    "mean_predicted": round(float(prob[m].mean()), 4),
                    "observed_rate": round(float(y_true[m].mean()), 4)})
    return out


def best_threshold_by_f1(y_true, prob) -> float:
    """
    Operating point chosen on VALIDATION only, then frozen for test.

    Picking the threshold on the test fold would be a quiet form of fitting to
    it — the metric would report an operating point no deployment could have
    known in advance.
    """
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1[:-1]))])


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------
def model_dir(task: str) -> Path:
    d = repo_root() / "models" / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def experiment_dir(task: str) -> Path:
    d = repo_root() / "experiments" / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_model(task: str, model, preprocessor: FeaturePreprocessor, metadata: dict) -> dict:
    """
    Persist the model, the preprocessor and everything needed to replay both.

    The preprocessor is saved alongside the model on purpose: a serialised
    booster on its own cannot be used, because it has no record of the column
    order, the folded vocabulary or the unknown-category code it was trained
    with.
    """
    import joblib
    d = model_dir(task)
    paths = {
        "model": str(d / "model.joblib"),
        "preprocessor": str(d / "preprocessor.joblib"),
        "metadata": str(d / "metadata.json"),
    }
    joblib.dump(model, paths["model"])
    joblib.dump(preprocessor, paths["preprocessor"])
    Path(paths["metadata"]).write_text(json.dumps(metadata, indent=2, default=str))
    log.info("saved %s model -> %s", task, d)
    return paths


def load_model(task: str):
    """Load (model, preprocessor, metadata) for inference."""
    import joblib
    d = repo_root() / "models" / task
    model = joblib.load(d / "model.joblib")
    pre = joblib.load(d / "preprocessor.joblib")
    meta = json.loads((d / "metadata.json").read_text())
    return model, pre, meta


def write_report(task: str, payload: dict, markdown: str) -> dict:
    out_json = ensure_dir(p("reports", "model_training", f"{task}_results.json"))
    out_md = ensure_dir(p("reports", "model_training", f"{task}_results.md"))
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    out_md.write_text(markdown)
    log.info("wrote %s and %s", out_json, out_md)
    return {"json": str(out_json), "markdown": str(out_md)}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def fmt_row(label: str, m: dict, keys: list[str]) -> str:
    cells = []
    for k in keys:
        v = m.get(k)
        cells.append("—" if v is None else (f"{v:,.4f}" if isinstance(v, float) else f"{v:,}"))
    return f"| {label} | " + " | ".join(cells) + " |"
