"""
Tests for the training framework.

These check the properties that a passing metric cannot: that the evaluation
folds are actually disjoint, that nothing fitted has seen validation or test,
that no outcome column can reach a model, that training is deterministic, and
that a saved model is genuinely loadable and usable for inference.

    python -m pytest tests/test_training.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train._common import (FORBIDDEN_PREDICTORS, SEED, FeaturePreprocessor,
                                   best_threshold_by_f1, calibration_bins,
                                   choose_split_column, classification_metrics,
                                   regression_metrics)

sk = pytest.importorskip("sklearn", reason="training framework needs scikit-learn")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def toy_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "source_dataset": rng.choice(["a311", "b311"], n),
        "city": rng.choice(["Alpha", "Beta"], n),
        "cat": rng.choice(list("abcde"), n),
        "num": rng.normal(size=n),
        "split": ["train"] * (n - 100) + ["val"] * 50 + ["test"] * 50,
        "split_global": ["train"] * (n - 100) + ["val"] * 50 + ["test"] * 50,
        "y": rng.gamma(2.0, 10.0, n),
    })


def task_table(name: str):
    fp = os.path.join("data", "processed", "ml", f"{name}.parquet")
    mf = os.path.join("data", "processed", "ml", f"{name}.manifest.json")
    if not (os.path.exists(fp) and os.path.exists(mf)):
        pytest.skip(f"{name} not built")
    return pd.read_parquet(fp), json.load(open(mf))


TASKS = [("resolution", "resolution_ml"), ("sla", "sla_ml"), ("hotspot", "hotspot_ml")]


# ---------------------------------------------------------------------------
# split separation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task,dataset", TASKS)
def test_folds_are_disjoint_and_non_empty(task, dataset):
    df, m = task_table(dataset)
    col, _ = choose_split_column(df)
    counts = df[col].value_counts()
    for fold in ("train", "val", "test"):
        assert counts.get(fold, 0) > 0, f"{dataset}: {fold} fold is empty on {col}"
    # every row belongs to exactly one fold
    assert set(df[col].dropna().unique()) <= {"train", "val", "test", "unassigned"}


@pytest.mark.parametrize("task,dataset", TASKS)
def test_split_choice_matches_the_documented_policy(task, dataset):
    """Single-source tables must fall back to `split`; pooled ones use split_global."""
    df, _ = task_table(dataset)
    col, why = choose_split_column(df)
    grp = "source_dataset" if "source_dataset" in df.columns else "city"
    n_src = df[grp].nunique()
    assert col == ("split" if n_src <= 1 else "split_global"), why


@pytest.mark.parametrize("task,dataset", TASKS)
def test_evaluation_folds_do_not_precede_training(task, dataset):
    """No test row may sit before the last training row on the chosen column."""
    df, _ = task_table(dataset)
    col, _ = choose_split_column(df)
    ts_col = "reported_at" if "reported_at" in df.columns else "week_start"
    if ts_col not in df.columns:
        pytest.skip("no timestamp column")
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if col == "split_global":
        assert ts[df[col] == "train"].max() <= ts[df[col] == "test"].min()
        assert ts[df[col] == "train"].max() <= ts[df[col] == "val"].min()
    else:
        grp = "source_dataset" if "source_dataset" in df.columns else "city"
        for key, g in df.groupby(grp, observed=True):
            t = ts.loc[g.index]
            if (g[col] == "val").any() and (g[col] == "train").any():
                assert t[g[col] == "train"].max() <= t[g[col] == "val"].min(), key


# ---------------------------------------------------------------------------
# leakage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task,dataset", TASKS)
def test_no_declared_predictor_is_an_outcome(task, dataset):
    _, m = task_table(dataset)
    block = m.get("predictors") or m.get("features") or {}
    preds = {c for c, v in block.items() if v.get("present")}
    allowed = {"sla_target_hours"} if task == "sla" else set()
    assert not (preds & FORBIDDEN_PREDICTORS) - allowed


@pytest.mark.parametrize("task,dataset", TASKS)
def test_target_is_never_a_predictor(task, dataset):
    _, m = task_table(dataset)
    block = m.get("predictors") or m.get("features") or {}
    target = m["target"]["column"]
    assert target not in block


def test_loader_rejects_a_manifest_that_declares_an_outcome(tmp_path, monkeypatch):
    """A future manifest edit must not be able to smuggle an outcome into a model."""
    from scripts.train import _common

    df = toy_frame()
    df["closed_at"] = pd.Timestamp("2020-01-01", tz="UTC")
    d = tmp_path / "ml"
    d.mkdir(parents=True)
    df.to_parquet(d / "evil_ml.parquet")
    (d / "evil_ml.manifest.json").write_text(json.dumps({
        "target": {"column": "y", "kind": "regression"},
        "predictors": {"num": {"present": True}, "closed_at": {"present": True}},
    }))
    monkeypatch.setattr(_common, "p", lambda *parts: d / parts[-1])
    with pytest.raises(SystemExit, match="forbidden predictors"):
        _common.load_task("resolution", "evil_ml")


# ---------------------------------------------------------------------------
# preprocessing fitted on train only
# ---------------------------------------------------------------------------
def test_preprocessor_vocabulary_comes_from_train_only():
    df = toy_frame()
    df.loc[df.split == "test", "cat"] = "ONLY_IN_TEST"
    tr = df[df.split == "train"]
    pre = FeaturePreprocessor(["cat", "num"]).fit(tr)
    assert "ONLY_IN_TEST" not in pre.vocabulary_["cat"]


def test_unseen_category_encodes_to_the_reserved_code_and_does_not_crash():
    df = toy_frame()
    tr = df[df.split == "train"]
    pre = FeaturePreprocessor(["cat", "num"]).fit(tr)
    unseen = pd.DataFrame({"cat": ["NEVER_SEEN", "a"], "num": [0.5, 0.5]})
    out = pre.transform(unseen)
    assert out["cat"].iloc[0] == -1.0
    assert out["cat"].iloc[1] != -1.0
    assert out.notna().all().all()


def test_null_categorical_becomes_an_explicit_level_not_a_drop():
    df = toy_frame()
    df.loc[df.index[:20], "cat"] = None
    tr = df[df.split == "train"]
    pre = FeaturePreprocessor(["cat", "num"]).fit(tr)
    assert "__MISSING__" in pre.vocabulary_["cat"]
    out = pre.transform(df[df.split == "test"].assign(cat=None))
    assert (out["cat"] != -1.0).all(), "a NULL seen in training must not encode as unknown"


def test_rare_levels_fold_rather_than_explode_cardinality():
    rng = np.random.default_rng(1)
    n = 2000
    df = pd.DataFrame({"cat": [f"z{i}" for i in range(n)], "num": rng.normal(size=n)})
    pre = FeaturePreprocessor(["cat", "num"], max_levels=50).fit(df)
    assert len(pre.vocabulary_["cat"]) <= 51          # 50 kept + __OTHER__
    out = pre.transform(df)
    assert out["cat"].nunique() <= 51


def test_transform_preserves_the_declared_column_order():
    df = toy_frame()
    cols = ["num", "cat", "city"]
    pre = FeaturePreprocessor(cols).fit(df)
    assert list(pre.transform(df).columns) == cols


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        FeaturePreprocessor(["num"]).transform(toy_frame())


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_training_is_deterministic_for_a_fixed_seed():
    from sklearn.ensemble import HistGradientBoostingRegressor

    df = toy_frame()
    tr = df[df.split == "train"]
    pre = FeaturePreprocessor(["cat", "num", "city"]).fit(tr)
    X, y = pre.transform(tr), tr["y"].to_numpy(float)
    a = HistGradientBoostingRegressor(max_iter=30, random_state=SEED,
                                      categorical_features=pre.categorical_mask).fit(X, y)
    b = HistGradientBoostingRegressor(max_iter=30, random_state=SEED,
                                      categorical_features=pre.categorical_mask).fit(X, y)
    assert np.allclose(a.predict(X), b.predict(X))


def test_preprocessor_is_deterministic():
    df = toy_frame()
    tr = df[df.split == "train"]
    a = FeaturePreprocessor(["cat", "num"]).fit(tr).transform(df)
    b = FeaturePreprocessor(["cat", "num"]).fit(tr).transform(df)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_regression_metrics_are_correct_on_a_hand_computed_case():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.0, 2.0, 3.0, 8.0])           # errors 0,0,0,4
    m = regression_metrics(y, pred)
    assert m["MAE"] == pytest.approx(1.0)
    assert m["median_AE"] == pytest.approx(0.0)
    assert m["RMSE"] == pytest.approx(2.0)
    assert m["n"] == 4


def test_classification_metrics_and_confusion_matrix_are_correct():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.6, 0.7, 0.2])           # at 0.5: pred 0,1,1,0
    m = classification_metrics(y, prob, 0.5)
    cm = m["confusion_matrix"]
    assert (cm["tn"], cm["fp"], cm["fn"], cm["tp"]) == (1, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["positive_rate"] == pytest.approx(0.5)


def test_threshold_is_selected_from_validation_scores_only():
    y = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.05, 0.1, 0.2, 0.8, 0.85, 0.9])
    thr = best_threshold_by_f1(y, prob)
    assert 0.2 < thr <= 0.8
    assert classification_metrics(y, prob, thr)["f1"] == pytest.approx(1.0)


def test_calibration_bins_report_observed_versus_predicted():
    y = np.array([0] * 50 + [1] * 50)
    prob = np.array([0.05] * 50 + [0.95] * 50)
    bins = calibration_bins(y, prob, bins=10)
    assert bins[0]["observed_rate"] == pytest.approx(0.0)
    assert bins[-1]["observed_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# hotspot: baselines and temporal safety
# ---------------------------------------------------------------------------
def _panel():
    weeks = pd.date_range("2020-01-06", periods=8, freq="7D")
    counts = [10, 20, 30, 40, 50, 60, 70, 80]
    return pd.DataFrame({
        "city": "Chicago", "zone_type": "ward", "zone_id": "1", "category": "POTHOLE",
        "week_start": weeks, "incident_count": counts,
        "future_incident_count": counts[1:] + [np.nan],
    })


def test_hotspot_rolling_baseline_ignores_the_target_entirely():
    from scripts.baselines.run_baselines import rolling_mean_baseline

    p = _panel()
    before = rolling_mean_baseline(p, 4, lagged=False).to_numpy(float)
    tampered = p.copy()
    tampered["future_incident_count"] = -999.0
    after = rolling_mean_baseline(tampered, 4, lagged=False).to_numpy(float)
    assert np.allclose(before, after, equal_nan=True)


def test_hotspot_rolling_baseline_cannot_see_a_later_week():
    from scripts.baselines.run_baselines import rolling_mean_baseline

    p = _panel()
    full = rolling_mean_baseline(p, 4, lagged=False).to_numpy(float)
    for t in range(1, len(p)):
        cut = rolling_mean_baseline(p.iloc[: t + 1].copy(), 4, lagged=False).to_numpy(float)
        assert cut[t] == pytest.approx(full[t])


def test_hotspot_predictors_are_all_trailing_quantities():
    """No predictor may be the target or a forward-looking aggregate."""
    _, m = task_table("hotspot_ml")
    preds = {c for c, v in m["predictors"].items() if v.get("present")}
    assert "future_incident_count" not in preds
    assert "future_incident_flag" not in preds
    assert not any(c.startswith("future_") for c in preds)


# ---------------------------------------------------------------------------
# artifacts: save / load / infer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task", ["resolution", "sla", "hotspot"])
def test_saved_model_loads_and_predicts_the_expected_shape(task):
    from scripts.train._common import load_model

    d = os.path.join("models", task)
    if not os.path.exists(os.path.join(d, "model.joblib")):
        pytest.skip(f"{task} model not trained yet")
    model, pre, meta = load_model(task)
    assert meta["task"] == task
    assert pre.fitted_ and pre.predictors == meta["features"]

    dataset = {"resolution": "resolution_ml", "sla": "sla_ml", "hotspot": "hotspot_ml"}[task]
    df, _ = task_table(dataset)
    sample = df.head(25)
    X = pre.transform(sample)
    assert list(X.columns) == meta["features"]
    pred = model.predict(X)
    assert pred.shape == (25,)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("task", ["resolution", "sla", "hotspot"])
def test_saved_metadata_records_what_is_needed_to_reproduce(task):
    path = os.path.join("models", task, "metadata.json")
    if not os.path.exists(path):
        pytest.skip(f"{task} model not trained yet")
    meta = json.load(open(path))
    for key in ("task", "dataset", "target", "split_column", "rows", "n_features",
                "features", "preprocessing", "selected_config", "model_params",
                "environment", "trained_at_utc"):
        assert key in meta, f"{task} metadata is missing {key}"
    assert meta["environment"]["seed"] == SEED
    assert meta["preprocessing"]["fitted_on"] == "training fold only"


def test_inference_on_an_unseen_category_does_not_crash():
    """The whole point of the reserved unknown code."""
    from scripts.train._common import load_model

    if not os.path.exists(os.path.join("models", "hotspot", "model.joblib")):
        pytest.skip("hotspot model not trained yet")
    model, pre, meta = load_model("hotspot")
    df, _ = task_table("hotspot_ml")
    row = df.head(1).copy()
    for c in pre.categorical_:
        row[c] = "A_ZONE_THAT_HAS_NEVER_EXISTED"
    pred = model.predict(pre.transform(row))
    assert pred.shape == (1,) and np.isfinite(pred).all()


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task", ["resolution", "sla", "hotspot"])
def test_report_records_baseline_comparison(task):
    path = os.path.join("reports", "model_training", f"{task}_results.json")
    if not os.path.exists(path):
        pytest.skip(f"{task} not trained yet")
    r = json.load(open(path))
    assert "metrics" in r and "test" in r["metrics"]
    if task == "hotspot":
        for b in ("rolling_4w_mean", "rolling_4w_mean_lagged", "persistence"):
            assert b in r["baselines"]["test"], f"missing baseline {b}"
        assert "verdict" in r and "recommendation" in r
    else:
        assert "baseline" in r and "test" in r["baseline"]


def test_hotspot_report_does_not_claim_a_win_it_did_not_earn():
    """
    The verdict must be derived from the numbers. If the rolling mean wins on a
    fold, the report has to say so.
    """
    path = os.path.join("reports", "model_training", "hotspot_results.json")
    if not os.path.exists(path):
        pytest.skip("hotspot not trained yet")
    r = json.load(open(path))
    cmp_ = r["comparison"]
    model_wins_val = cmp_["val"]["rolling_4w_mean"]["model_wins"]
    model_wins_test = cmp_["test"]["rolling_4w_mean"]["model_wins"]
    verdict = r["verdict"].lower()
    if not (model_wins_val and model_wins_test):
        assert "lose" in verdict or "not a stable" in verdict, \
            f"model did not clear the benchmark but the verdict reads: {r['verdict']}"
        assert "rolling" in r["recommendation"].lower()
