"""
Phase 2B tests: leak-free feature generation, category alignment, evaluation
folds, UrbanEye policy drift, and (when the v2 artifacts exist) the inference
contracts.

Feature functions are checked against brute-force reference implementations on
small random data, and against the leakage property directly: changing
anything that happens at or after an incident's report time must not change
that incident's features.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.phase2b import features as F  # noqa: E402
from scripts.phase2b.common import (CATEGORY_DEFAULT_PRIORITY, DEFAULT_SLA_HOURS, ROOT,  # noqa: E402
                                    UE_CATEGORIES, clean_period_mask, load_ue_mapping,
                                    map_to_urbaneye, paired_bootstrap_mae_gain,
                                    resolution_metrics)
from scripts.phase2b.harness import lookup_predict, rolling_folds  # noqa: E402

DAY = 1440.0


def _events(n=400, seed=0, groups=3, span_days=120):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, groups, n)
    rep = np.sort(rng.uniform(0, span_days * DAY, n))
    dur_h = rng.lognormal(3, 1.5, n)
    clo = rep + dur_h * 60
    open_ = rng.random(n) < 0.05
    clo[open_] = np.nan
    return g, rep, clo, dur_h, open_


# ---------------------------------------------------------------------------
# workload features
# ---------------------------------------------------------------------------
def _brute_workload(g, rep, clo, res, i, cap_days=90):
    t, gg = rep[i], g[i]
    same = g == gg
    t_floor = np.floor(t)
    before = same & (np.floor(rep) < t_floor)
    exit_ = np.fmin(np.where(np.isnan(clo), np.inf, clo), rep + cap_days * DAY)
    backlog = before.sum() - (same & (np.floor(exit_) < t_floor)).sum()
    intake7 = (same & (np.floor(rep) < t_floor) & (np.floor(rep) >= np.floor(t - 7 * DAY))).sum()
    closed_ok = same & ~np.isnan(clo) & ~np.isnan(res)
    win = closed_ok & (np.floor(clo) < t_floor) & (np.floor(clo) >= np.floor(t - 30 * DAY))
    return backlog, intake7, win.sum(), (np.log1p(res[win]).mean() if win.sum() >= 10 else np.nan)


def test_workload_matches_brute_force():
    g, rep, clo, dur, _ = _events()
    res = np.where(np.isnan(clo), np.nan, dur)
    out = F.workload_features(g, rep, clo, res)
    for i in range(0, len(g), 37):
        b, i7, nc, ml = _brute_workload(g, rep, clo, res, i)
        assert out["wl_backlog_open"].iloc[i] == b
        assert out["wl_intake_7d"].iloc[i] == i7
        assert out["wl_recent_closures_30d"].iloc[i] == nc
        if np.isnan(ml):
            assert np.isnan(out["wl_recent_mean_log_res"].iloc[i])
        else:
            assert out["wl_recent_mean_log_res"].iloc[i] == pytest.approx(ml)


def test_workload_is_unchanged_by_future_events():
    g, rep, clo, dur, _ = _events(seed=1)
    res = np.where(np.isnan(clo), np.nan, dur)
    base = F.workload_features(g, rep, clo, res)
    i = len(g) // 2
    t = rep[i]
    # rewrite the future: every closure at/after t and every report after t
    clo2, res2 = clo.copy(), res.copy()
    fut_close = ~np.isnan(clo2) & (clo2 >= t)
    clo2[fut_close] = clo2[fut_close] + 5000.0
    res2[fut_close] = res2[fut_close] * 7
    clo2[i] = np.nan      # the query incident's own outcome
    pert = F.workload_features(g, rep, clo2, res2)
    pd.testing.assert_series_equal(base.iloc[i], pert.iloc[i], check_names=False)


def test_backlog_age_cap_stops_zombie_accumulation():
    # 100 reports one day apart that never close: an uncapped backlog would reach 99
    rep = np.arange(100) * DAY
    out = F.workload_features(np.zeros(100, int), rep, np.full(100, np.nan),
                              np.full(100, np.nan), backlog_cap_days=10)
    assert out["wl_backlog_open"].max() == 10


def test_warmup_rows_are_blank():
    g, rep, clo, dur, _ = _events(seed=2)
    out = F.workload_features(g, rep, clo, dur, warmup_start_min=np.zeros(len(g)))
    young = rep < 30 * DAY
    assert out.loc[young].isna().all().all()
    assert out.loc[~young, "wl_intake_7d"].notna().all()


# ---------------------------------------------------------------------------
# recent breach rate
# ---------------------------------------------------------------------------
def test_recent_breach_rate_brute_force_and_leak_free():
    g, rep, clo, dur, open_ = _events(n=3000, seed=3, groups=2, span_days=200)
    dur_obs = np.where(open_, np.inf, dur)
    elig = np.ones(len(g), bool)
    X = 24.0
    rate = F.recent_breach_rate(g, rep, dur_obs, open_, X, elig, min_events=5)
    for i in range(500, len(g), 211):
        t = rep[i]
        w = (g == g[i]) & (np.floor(rep) < np.floor(t - X * 60)) & \
            (np.floor(rep) >= np.floor(t - X * 60 - 30 * DAY))
        exp = (open_[w] | (dur_obs[w] >= X)).mean() if w.sum() >= 5 else np.nan
        assert (np.isnan(exp) and np.isnan(rate[i])) or rate[i] == pytest.approx(exp)
        # every incident it uses had its deadline expire strictly before t
        assert np.all(rep[w] + X * 60 < t + 1)


# ---------------------------------------------------------------------------
# spatial counts
# ---------------------------------------------------------------------------
def test_spatial_counts_brute_force():
    rng = np.random.default_rng(4)
    n = 600
    ix = rng.integers(5, 12, n).astype(float)
    iy = rng.integers(5, 12, n).astype(float)
    rep = np.sort(rng.uniform(0, 60 * DAY, n))
    grp = rng.integers(0, 2, n)
    out = F.spatial_counts(grp, ix, iy, rep, [7, 30])
    for i in range(0, n, 29):
        nb = (grp == grp[i]) & (np.abs(ix - ix[i]) <= 1) & (np.abs(iy - iy[i]) <= 1)
        for w in (7, 30):
            exp = (nb & (np.floor(rep) < np.floor(rep[i]))
                   & (np.floor(rep) >= np.floor(rep[i] - w * DAY))).sum()
            assert out[f"sp_{w}d"].iloc[i] == exp


def test_spatial_open_now_brute_force_and_leak_free():
    # open_now = block events reported before t whose exit (closure, or report + cap,
    # supplied by the caller) is not before t. It is the top resolution feature, so
    # it is checked directly, including that rewriting the future cannot change it.
    rng = np.random.default_rng(5)
    n = 500
    ix = rng.integers(5, 9, n).astype(float)
    iy = rng.integers(5, 9, n).astype(float)
    rep = np.sort(rng.uniform(0, 60 * DAY, n))
    exit_ = rep + rng.uniform(0, 20 * DAY, n)
    grp = np.zeros(n, int)
    out = F.spatial_counts(grp, ix, iy, rep, [7], closed_min=exit_)
    for i in range(0, n, 23):
        nb = (np.abs(ix - ix[i]) <= 1) & (np.abs(iy - iy[i]) <= 1)
        t = np.floor(rep[i])
        exp = (nb & (np.floor(rep) < t) & ~(np.floor(exit_) < t)).sum()
        assert out["sp_open_now"].iloc[i] == exp
    i = n // 2
    exit2 = exit_.copy()
    fut = exit2 >= rep[i]
    exit2[fut] += 30 * DAY                      # every closure at/after t moves later
    exit2[i] = rep[i] + 1.0                     # the query incident's own closure
    pert = F.spatial_counts(grp, ix, iy, rep, [7], closed_min=exit2)
    assert pert["sp_open_now"].iloc[i] == out["sp_open_now"].iloc[i]


def test_spatial_counts_missing_coordinates_are_nan():
    out = F.spatial_counts(np.zeros(3, int), np.array([1.0, np.nan, 3.0]),
                           np.array([1.0, 2.0, 3.0]), np.array([0.0, 10.0, 20.0]), [7])
    assert np.isnan(out["sp_7d"].iloc[1]) and not np.isnan(out["sp_7d"].iloc[0])


# ---------------------------------------------------------------------------
# category alignment
# ---------------------------------------------------------------------------
def _frame(rows):
    return pd.DataFrame(rows, columns=["source_dataset", "source_category_raw", "subcategory"])


def test_mapping_rules_and_order():
    df = _frame([
        ("chicago311", "Pothole in Street Complaint", None),
        ("chicago311", "Garbage Cart Maintenance", None),
        ("chicago311", "Fly Dumping Complaint", None),
        ("sf311", "Street and Sidewalk Cleaning", "Bulky Items"),
        ("sf311", "Street and Sidewalk Cleaning", "Human or Animal Waste"),
        ("nyc311", "Dirty Conditions", "E8 Canine Violation"),
        ("nyc311", "Dirty Conditions", "E5 Loose Rubbish"),
        ("nyc311", "Sewer", "Street Flooding (SJ)"),
        ("nyc311", "Sewer", "Sewer Odor (SA2)"),
        ("chicago311", "Graffiti Removal Request", None),
    ])
    got = [x if isinstance(x, str) else None for x in map_to_urbaneye(df)["ue_category"]]
    # sewer odour and graffiti match no rule: "never considered", not UNSUPPORTED
    assert got == ["POTHOLE", "UNSUPPORTED", "GARBAGE", "GARBAGE", "UNSUPPORTED",
                   "UNSUPPORTED", "GARBAGE", "FLOOD", None, None]


def test_mapping_never_invents_fire_hazard_or_other():
    targets = {r["urbaneye_category"] for r in load_ue_mapping()}
    assert "FIRE_HAZARD" not in targets and "OTHER" not in targets
    assert targets <= set(UE_CATEGORIES) | {"UNSUPPORTED"}
    for r in load_ue_mapping():
        if r["urbaneye_category"] != "UNSUPPORTED":
            assert r["confidence"] in {"EXACT", "STRONG", "PARTIAL"}


def test_clean_period_mask():
    df = pd.DataFrame({"source_dataset": ["chicago311", "chicago311", "sf311"],
                       "reported_at": pd.to_datetime(["2019-02-28", "2019-03-01", "2018-01-01"],
                                                     utc=True)})
    assert clean_period_mask(df).tolist() == [False, True, True]


# ---------------------------------------------------------------------------
# UrbanEye policy must match the backend source
# ---------------------------------------------------------------------------
BACKEND_MODEL = ROOT.parent / "backend" / "app" / "models" / "incident.py"


@pytest.mark.skipif(not BACKEND_MODEL.exists(), reason="backend source not present")
def test_policy_mirrors_backend():
    src = BACKEND_MODEL.read_text()
    sla = dict(re.findall(r"IncidentPriority\.(P\d):\s*(\d+)", src.split("DEFAULT_SLA_HOURS")[1]
                          .split("}")[0]))
    assert {k: int(v) for k, v in sla.items()} == DEFAULT_SLA_HOURS
    block = src.split("CATEGORY_DEFAULT_PRIORITY: dict")[1].split("}")[0]
    pri = dict(re.findall(r"IncidentCategory\.(\w+):\s*IncidentPriority\.(P\d)", block))
    assert pri == CATEGORY_DEFAULT_PRIORITY
    cats = re.findall(r'^\s+(\w+) = "\1"', src.split("class IncidentCategory")[1]
                      .split("class ")[0], flags=re.M)
    assert cats == UE_CATEGORIES


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------
def test_rolling_folds_are_chronological_and_label_known():
    rep = pd.date_range("2019-01-01", "2020-08-31", freq="6h", tz="UTC")
    df = pd.DataFrame({"reported_at": rep, "source_dataset": "chicago311"})
    df["closed_at"] = df["reported_at"] + pd.to_timedelta(np.arange(len(df)) % 2000, unit="h")
    n = 0
    for _, tr, te in rolling_folds(df, "known"):
        n += 1
        assert df.loc[tr, "reported_at"].max() < df.loc[te, "reported_at"].min()
        assert df.loc[tr, "closed_at"].max() < df.loc[te, "reported_at"].min()
    assert n == 6
    df["sla_hours"] = 72.0
    for _, tr, te in rolling_folds(df, "known", sla_hours_col="sla_hours"):
        dl = df.loc[tr, "reported_at"] + pd.Timedelta(hours=72)
        assert dl.max() < df.loc[te, "reported_at"].min()


def test_lookup_predict_falls_back_to_global():
    tr = pd.DataFrame({"k": ["a", "a", "b"], "y": [1.0, 3.0, 10.0]})
    te = pd.DataFrame({"k": ["a", "b", "zzz"]})
    assert lookup_predict(tr, te, ["k"], "y").tolist() == [2.0, 10.0, 3.0]


def test_resolution_metrics_and_bootstrap():
    y = np.array([1.0, 10, 100, 1000])
    m = resolution_metrics(y, y * 1.1, lo=y * 0.5, hi=y * 2)
    assert m["interval_coverage"] == 1.0 and m["MAE"] > 0
    rng = np.random.default_rng(0)
    yy = rng.lognormal(3, 1, 2000)
    good, bad = yy * 1.01, yy * 2
    assert paired_bootstrap_mae_gain(yy, bad, good)["ci95"][0] > 0


# ---------------------------------------------------------------------------
# inference contracts (skipped until train_final.py has saved the artifacts)
# ---------------------------------------------------------------------------
from scripts.phase2b import inference as INF  # noqa: E402

REQUIRED_META = {"model_version", "task", "features", "target", "preprocessing", "training_data",
                 "evaluation_split", "metrics", "baseline_metrics", "software_versions",
                 "random_seed", "gates"}
needs_res = pytest.mark.skipif(not INF.available("resolution"), reason="models/resolution_v2 absent")
needs_sla = pytest.mark.skipif(not INF.available("sla"), reason="models/sla_v2 absent")
needs_hot = pytest.mark.skipif(not INF.available("hotspot"), reason="models/hotspot_v2 absent")
HISTORY = {"wl_backlog_open": 120.0, "wl_intake_7d": 40.0, "wl_intake_30d": 160.0,
           "wl_closed_7d": 35.0, "wl_backlog_days": 22.5, "wl_recent_closures_30d": 150.0,
           "wl_recent_mean_log_res": 4.2, "wl_recent_median_res_h": 70.0,
           "wl_city_backlog_open": 900.0, "wl_city_intake_7d": 400.0,
           "sp_samecat_box750_7d": 2.0, "sp_samecat_box750_30d": 6.0,
           "sp_samecat_box750_90d": 15.0, "sp_samecat_box750_open_now": 1.0,
           "sp_all_box750_30d": 30.0, "hist_breach_rate_own": 0.6}


def test_calendar_features_use_india_local_time():
    # 20:00 UTC on Thursday 1 Jan 2026 is 01:30 IST on Friday 2 Jan
    f = INF.calendar_features("2026-01-01T20:00:00Z")
    assert f == {"hour": 1, "day_of_week": 4, "month": 1, "is_weekend": False, "is_night": True}
    # naive timestamps are treated as UTC
    assert INF.calendar_features("2026-01-03T06:00:00")["is_weekend"] is True


@needs_res
def test_resolution_contract():
    P = INF.ResolutionPredictor()
    assert REQUIRED_META <= set(P.meta)
    assert P.meta["features"] == P.pre.predictors
    assert all(g["passed"] for g in P.meta["gates"])
    rows = [{"category": c, "created_at": "2026-03-10T05:00:00Z", "history": HISTORY}
            for c in ("POTHOLE", "FLOOD", "GARBAGE", "STREETLIGHT")]
    rows.append({"category": "POTHOLE", "created_at": "2026-03-10T05:00:00Z"})  # no history
    rows.append({"category": "FIRE_HAZARD", "created_at": "2026-03-10T05:00:00Z"})
    rows.append({"category": "OTHER", "created_at": "2026-03-10T05:00:00Z"})
    out = P.predict(rows)
    assert len(out) == len(rows)
    for o in out[:5]:
        assert o["supported"] is True and o["reason"] is None
        assert np.isfinite(o["predicted_resolution_hours"]) and o["predicted_resolution_hours"] >= 0
        assert 0 <= o["interval_p10_hours"] <= o["predicted_resolution_hours"] \
            <= o["interval_p90_hours"]
        assert isinstance(o["interval_calibrated"], bool)
    for o in out[5:]:
        assert o["supported"] is False and o["predicted_resolution_hours"] is None
    assert P.predict(rows) == out                      # deterministic
    assert P.predict([]) == []
    # categories differ, so the model is not a constant
    assert len({round(o["predicted_resolution_hours"], 6) for o in out[:4]}) > 1


@needs_res
def test_resolution_contract_matches_direct_model_call():
    P = INF.ResolutionPredictor()
    row = {"category": "STREETLIGHT", "created_at": "2026-05-01T12:00:00Z", "history": HISTORY}
    X = P.pre.transform(INF._frame([row], P.features, "model_category"))
    assert not (X["model_category"] == -1).any()      # category is in the vocabulary
    direct = float(np.clip(np.expm1(P.models["point"].predict(X)), 0, None)[0])
    assert P.predict([row])[0]["predicted_resolution_hours"] == pytest.approx(direct)


@needs_sla
def test_sla_contract():
    P = INF.SLAPredictor()
    assert REQUIRED_META <= set(P.meta)
    assert P.meta["features"] == P.pre.predictors
    assert all(g["passed"] for g in P.meta["gates"])
    assert 0 < P.threshold < 1 and P.bands["medium"] <= P.bands["high"]
    rows = [{"category": c, "created_at": "2026-03-10T05:00:00Z", "history": HISTORY}
            for c in ("POTHOLE", "FLOOD", "GARBAGE", "STREETLIGHT")]
    rows.append({"category": "GARBAGE", "created_at": "2026-03-10T05:00:00Z"})  # no history
    rows.append({"category": "FIRE_HAZARD", "created_at": "2026-03-10T05:00:00Z"})
    out = P.predict(rows)
    assert len(out) == len(rows)
    for r, o in zip(rows[:5], out[:5]):
        assert o["supported"] is True
        assert 0.0 <= o["breach_probability"] <= 1.0
        assert o["sla_hours"] == INF.default_sla_hours(r["category"])
        p = o["breach_probability"]
        exp = "HIGH" if p >= P.bands["high"] else "MEDIUM" if p >= P.bands["medium"] else "LOW"
        assert o["risk_band"] == exp
    assert out[5]["supported"] is False and out[5]["breach_probability"] is None
    assert P.predict(rows) == out


@needs_sla
def test_sla_override_semantics():
    P = INF.SLAPredictor()
    base = {"category": "POTHOLE", "created_at": "2026-03-10T05:00:00Z", "history": HISTORY}
    if "sla_hours" in P.features:
        probs = [P.predict([{**base, "sla_hours": h}])[0]["breach_probability"]
                 for h in (4, 12, 24, 72, 168)]
        # a longer deadline can never raise the breach risk (monotonic constraint)
        assert all(a >= b - 1e-12 for a, b in zip(probs, probs[1:]))
        for h in (1, 500):
            o = P.predict([{**base, "sla_hours": h}])[0]
            assert o["supported"] is False and "range" in o["reason"]
    else:
        o = P.predict([{**base, "sla_hours": 12}])[0]
        assert o["supported"] is False


@needs_hot
def test_hotspot_contract():
    P = INF.HotspotPredictor()
    assert P.meta["is_ml"] is False and P.meta["method"] == "ewma"
    assert 0 < P.alpha <= 1 and P.min_weeks == 4
    counts = [3, 0, 5, 2, 7, 4]
    out = P.predict([{"category": "POTHOLE", "weekly_counts": counts},
                     {"category": "POTHOLE", "weekly_counts": counts[:3]},
                     {"category": "FIRE_HAZARD", "weekly_counts": counts},
                     {"category": "FLOOD", "weekly_counts": [-1, 2, 2, 2]}])
    s = float(counts[0])
    for c in counts[1:]:
        s = P.alpha * c + (1 - P.alpha) * s
    assert out[0]["supported"] and out[0]["predicted_next_week_count"] == pytest.approx(s)
    assert not out[1]["supported"] and not out[2]["supported"]
    assert out[3]["predicted_next_week_count"] >= 0     # negative counts are clipped
