"""
Phase 2C ML layer — unit tests.  No database is needed.

Correctness is anchored to the ML pipeline, not to expectations written here:
  * fixtures/golden_predictions.json — inputs, encodings and outputs produced by
    the ORIGINAL Phase 2B preprocessor and models;
  * fixtures/feature_parity.json — synthetic incidents and the features
    scripts/phase2b/features.py computes for them.
Both ship inside the SHA-256-pinned bundle.  The remaining tests pin the
decision logic: statuses, fallbacks, SLA override semantics, failure handling.
"""

import copy
import dataclasses
import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from app.core.config import get_settings
from app.ml import registry as R
from app.ml.areas import ServiceArea, parse_areas
from app.ml.encoding import Encoder
from app.ml.features import (HISTORY_FEATURES, IncidentRecord, calendar_features,
                             history_features, local_closed_durations)
from app.ml.hotspot import ewma, forecast, week_start
from app.ml.predictors import (AVAILABLE, EXPERIMENTAL, FALLBACK, UNAVAILABLE, Context,
                               predict_resolution, predict_sla)

S = get_settings()
BUNDLE = Path(S.ML_BUNDLE_DIR)
UTC = timezone.utc


def _load(rel: str) -> dict:
    return json.loads((BUNDLE / rel).read_text())


def _nan_eq(expected, got, tol=1e-9) -> bool:
    if expected is None:
        return isinstance(got, float) and math.isnan(got)
    return not math.isnan(got) and abs(expected - got) <= tol * max(1.0, abs(expected))


@pytest.fixture(scope="module")
def reg() -> R.Registry:
    r = R.build_registry(True, S.ML_BUNDLE_DIR, S.ML_BUNDLE_MANIFEST_SHA256)
    assert r.error is None, r.public_status()
    return r


# ---------------------------------------------------------------------------
# artifacts: verification and readiness
# ---------------------------------------------------------------------------
def test_bundle_loads_and_reports_versions(reg):
    st = reg.public_status()
    assert st["bundle_version"] == "urbaneye-ml-2026.10.13"
    assert st["training_data_through"] == "2020-08-28"
    for name in ("resolution", "sla", "hotspot", "baselines", "feature_ranges"):
        assert st["components"][name]["ready"], name
    assert st["components"]["resolution"]["model_version"] == "resolution_v2.0.0"
    assert st["components"]["sla"]["model_version"] == "sla_v2.0.0"
    assert st["components"]["resolution"]["range_calibrated"] is False
    assert st["components"]["hotspot"]["method"].startswith("EWMA")
    assert "/" not in json.dumps(st["components"])          # no file system paths leak


def test_bundle_models_are_the_phase2b_artifacts(reg):
    m = reg.manifest
    for task in ("resolution", "sla"):
        assert m["files_sha256"][f"{task}/model.joblib"] == \
            m["source_artifacts_sha256"][task]["model.joblib"]


def test_missing_bundle_is_reported_not_raised(tmp_path):
    r = R.build_registry(True, str(tmp_path / "nope"), S.ML_BUNDLE_MANIFEST_SHA256)
    assert r.error == "BUNDLE_NOT_FOUND"
    assert not any(c.ready for c in r.components.values())


def test_manifest_hash_mismatch_refuses_everything():
    r = R.build_registry(True, S.ML_BUNDLE_DIR, "0" * 64)
    assert r.error == "MANIFEST_HASH_MISMATCH" and r.resolution_models is None


def test_tampered_file_disables_only_that_component(tmp_path):
    dst = tmp_path / "bundle"
    shutil.copytree(BUNDLE, dst)
    (dst / "baselines.json").write_text("{}")
    r = R.build_registry(True, str(dst), S.ML_BUNDLE_MANIFEST_SHA256)
    assert r.error == "FILE_HASH_MISMATCH"
    assert not r.ready("baselines") and r.ready("resolution") and r.ready("sla")


def test_runtime_version_mismatch_refuses_pickles(monkeypatch):
    monkeypatch.setattr(R, "_runtime_versions",
                        lambda: {"numpy": "2.2.6", "scikit_learn": "9.9.9", "joblib": "1.5.1"})
    r = R.build_registry(True, S.ML_BUNDLE_DIR, S.ML_BUNDLE_MANIFEST_SHA256)
    assert r.components["sla"].error == "RUNTIME_VERSION_MISMATCH"
    assert not r.ready("sla") and not r.ready("resolution") and r.ready("hotspot")


def test_disabled_registry_opens_nothing():
    r = R.build_registry(False, "/does/not/matter", "x")
    assert r.error == "ML_DISABLED" and r.resolution_models is None


# ---------------------------------------------------------------------------
# encoder + models reproduce the ORIGINAL pipeline exactly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("task", ["resolution", "sla"])
def test_encoder_matches_original_preprocessor(reg, task):
    g = _load("fixtures/golden_predictions.json")[task]
    enc = reg.resolution_encoder if task == "resolution" else reg.sla_encoder
    assert enc.predictors == g["features"]
    X = enc.transform([c["features"] for c in g["cases"]])
    exp = np.array([[np.nan if v is None else v for v in c["encoded"]] for c in g["cases"]])
    np.testing.assert_array_equal(np.isnan(X), np.isnan(exp))
    np.testing.assert_allclose(np.nan_to_num(X, nan=0), np.nan_to_num(exp, nan=0), rtol=0, atol=0)


def test_models_reproduce_original_outputs(reg):
    g = _load("fixtures/golden_predictions.json")
    res = g["resolution"]["cases"]
    X = reg.resolution_encoder.transform([c["features"] for c in res])
    for key, name in (("raw_point", "point"), ("raw_q10", "q10"), ("raw_q90", "q90")):
        np.testing.assert_allclose(reg.resolution_models[name].predict(X),
                                   [c[key] for c in res], rtol=0, atol=1e-9)
    sla = g["sla"]["cases"]
    X = reg.sla_encoder.transform([c["features"] for c in sla])
    np.testing.assert_allclose(reg.sla_model.predict_proba(X)[:, 1],
                               [c["probability"] for c in sla], rtol=0, atol=1e-9)


def test_encoder_handles_unknown_and_missing_values():
    enc = Encoder({"predictors": ["c", "n"], "categorical": ["c"], "numeric": ["n"],
                   "vocabulary": {"c": ["A", "B"]}})
    X = enc.transform([{"c": "A", "n": "3"}, {"c": "Z", "n": "x"}, {}])
    assert X[0].tolist() == [0.0, 3.0]
    assert X[1][0] == -1.0 and math.isnan(X[1][1])          # unseen level, non-numeric
    assert X[2][0] == -1.0 and math.isnan(X[2][1])          # missing level not in vocab


# ---------------------------------------------------------------------------
# features: parity with scripts/phase2b/features.py, and leak-freeness
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def parity():
    p = _load("fixtures/feature_parity.json")
    a = p["area"]
    area = ServiceArea(a["name"], a["lat_min"], a["lat_max"], a["lon_min"], a["lon_max"])
    assert area.ref_lat == pytest.approx(a["ref_lat"])
    recs = [IncidentRecord(r["id"], r["category"], datetime.fromisoformat(r["created_at"]),
                           r["latitude"], r["longitude"],
                           None if r["first_closed_at"] is None
                           else datetime.fromisoformat(r["first_closed_at"]))
            for r in p["incidents"]]
    return p, area, recs


def test_history_features_match_pipeline(parity):
    p, area, recs = parity
    checked = 0
    for e in p["expected"]:
        got = history_features(recs[e["index"]], recs, area, p["horizons"])
        for f in HISTORY_FEATURES:
            assert _nan_eq(e["features"][f], got[f]), (e["index"], f, e["features"][f], got[f])
            checked += 1
        for h, x in e["hist_breach_rate"].items():
            assert _nan_eq(x, got["hist_breach_rate"][int(h)][0], tol=1e-12), (e["index"], h)
    assert checked == len(p["expected"]) * len(HISTORY_FEATURES)
    # the fixture exercises both the warm-up and the fully-populated regime
    assert any(e["features"]["wl_backlog_open"] is None for e in p["expected"])
    assert any(e["features"]["sp_samecat_box750_90d"] is not None for e in p["expected"])


def test_history_features_ignore_the_future(parity):
    p, area, recs = parity
    q = recs[400]
    t = q.created_at
    base = history_features(q, recs, area, [24.0])
    future = []
    for r in recs:
        if r.id == q.id:
            future.append(dataclasses.replace(r, first_closed_at=t + timedelta(minutes=5)))
        elif r.first_closed_at is not None and r.first_closed_at >= t:
            future.append(dataclasses.replace(r, first_closed_at=r.first_closed_at + timedelta(days=40)))
        else:
            future.append(r)
    future += [IncidentRecord(10_000 + k, q.category, t + timedelta(hours=k + 1),
                              q.latitude, q.longitude, None) for k in range(50)]
    pert = history_features(q, future, area, [24.0])
    for f in HISTORY_FEATURES:
        assert _nan_eq(base[f], pert[f]) if not math.isnan(base[f]) else math.isnan(pert[f])
    assert _nan_eq(base["hist_breach_rate"][24.0][0], pert["hist_breach_rate"][24.0][0]) \
        or math.isnan(base["hist_breach_rate"][24.0][0])


def test_outside_service_area_has_no_history():
    q = IncidentRecord(1, "POTHOLE", datetime(2026, 5, 1, tzinfo=UTC), 0.0, 0.0, None)
    f = history_features(q, [], None, [72.0])
    assert all(math.isnan(f[k]) for k in HISTORY_FEATURES)
    assert math.isnan(f["hist_breach_rate"][72.0][0])


def test_calendar_features_use_india_local_time():
    f = calendar_features(datetime(2026, 1, 1, 20, 0, tzinfo=UTC))    # Fri 01:30 IST
    assert f == {"hour": 1, "day_of_week": 4, "month": 1, "is_weekend": False, "is_night": True}


def test_service_area_config():
    a = parse_areas("")
    assert [x.name for x in a] == ["Bengaluru", "Chennai"]
    assert a[0].contains(12.9716, 77.5946) and not a[0].contains(13.08, 80.27)
    with pytest.raises(ValueError):
        parse_areas('[{"name": "x", "lat_min": 2, "lat_max": 1, "lon_min": 0, "lon_max": 1}]')


# ---------------------------------------------------------------------------
# predictions: statuses, fallbacks, SLA semantics, failure handling
# ---------------------------------------------------------------------------
AREA = ServiceArea("Bengaluru", 12.70, 13.25, 77.35, 77.90)
T = datetime(2026, 6, 10, 6, 30, tzinfo=UTC)


def _q(cat="POTHOLE") -> IncidentRecord:
    return IncidentRecord(1, cat, T, 12.9716, 77.5946, None)


def _in_range_features(reg, task="sla") -> dict:
    rng = reg.feature_ranges[task]
    f = {k: (rng[k]["p01"] + rng[k]["p99"]) / 2 for k in HISTORY_FEATURES}
    f["hist_breach_rate"] = {72.0: ((rng["hist_breach_rate_own"]["p01"]
                                     + rng["hist_breach_rate_own"]["p99"]) / 2, 50)}
    return f


def _nan_features(horizon=72.0) -> dict:
    f = {k: float("nan") for k in HISTORY_FEATURES}
    f["hist_breach_rate"] = {horizon: (float("nan"), 0)}
    return f


def _closed(n, cat, hours, start=T - timedelta(days=60)) -> list:
    """n incidents of `cat` closed after `hours`, all closed before T."""
    return [IncidentRecord(100 + i, cat, start + timedelta(hours=i), 12.97, 77.59,
                           start + timedelta(hours=i + hours)) for i in range(n)]


def _ctx(features, cat="POTHOLE", sla=72.0, area=AREA, records=()) -> Context:
    return Context(query=_q(cat), category=cat, sla_hours=sla, area=area, features=features,
                   baseline_records=list(records))


def test_sufficient_history_is_available(reg):
    feats = _in_range_features(reg)
    r = predict_resolution(reg, _ctx(feats))
    s = predict_sla(reg, _ctx(feats))
    # resolution's in-range check uses its own training ranges
    assert r["method"] == "ML_MODEL" and r["status"] in (AVAILABLE, EXPERIMENTAL)
    assert s["status"] == AVAILABLE and s["method"] == "ML_MODEL" and s["reasons"] == []
    assert 0.0 <= s["breach_probability"] <= 1.0
    assert s["risk_band"] in ("LOW", "MEDIUM", "HIGH")
    assert s["model_version"] == "sla_v2.0.0" and s["local_reliability_established"] is False
    assert any("not been measured" in w for w in s["warnings"])
    rf = {k: (reg.feature_ranges["resolution"][k]["p01"]
              + reg.feature_ranges["resolution"][k]["p99"]) / 2 for k in HISTORY_FEATURES}
    r = predict_resolution(reg, _ctx(rf))
    assert r["status"] == AVAILABLE and r["reasons"] == []
    assert 0 <= r["range_p10_hours"] <= r["estimate_hours"] <= r["range_p90_hours"]
    assert r["range_calibrated"] is False and r["range_empirical_coverage"] == pytest.approx(0.5965, abs=1e-3)
    assert any("not a calibrated 80% interval" in w for w in r["warnings"])


def test_insufficient_history_still_predicts_as_experimental(reg):
    r = predict_resolution(reg, _ctx(_nan_features()))
    s = predict_sla(reg, _ctx(_nan_features()))
    for out in (r, s):
        assert out["status"] == EXPERIMENTAL and out["method"] == "ML_MODEL"
        codes = [x["code"] for x in out["reasons"]]
        assert "INSUFFICIENT_LOCAL_HISTORY" in codes
        assert out["fallback_reason"] is None
    assert r["estimate_hours"] is not None and s["breach_probability"] is not None
    missing = next(x for x in s["reasons"] if x["code"] == "INSUFFICIENT_LOCAL_HISTORY")
    assert "hist_breach_rate_own" in missing["detail"]["missing_features"]


def test_partially_missing_features_are_named(reg):
    f = _in_range_features(reg)
    f["sp_samecat_box750_90d"] = float("nan")
    s = predict_sla(reg, _ctx(f))
    assert s["status"] == EXPERIMENTAL
    miss = next(x for x in s["reasons"] if x["code"] == "INSUFFICIENT_LOCAL_HISTORY")
    assert miss["detail"]["missing_features"] == ["sp_samecat_box750_90d"]


def test_out_of_range_and_outside_area_are_experimental(reg):
    f = _in_range_features(reg)
    f["wl_intake_7d"] = 1.0                                  # UrbanEye-scale count
    s = predict_sla(reg, _ctx(f))
    assert s["status"] == EXPERIMENTAL
    assert any(x["code"] == "FEATURES_OUTSIDE_TRAINING_RANGE" for x in s["reasons"])
    s = predict_sla(reg, _ctx(_in_range_features(reg), area=None))
    assert any(x["code"] == "OUTSIDE_SERVICE_AREA" for x in s["reasons"])


def test_missing_artifacts_fall_back_to_documented_baselines(reg):
    broken = dataclasses.replace(reg, components=copy.deepcopy(reg.components))
    broken.components["resolution"].ready = False
    broken.components["sla"].ready = False
    # no local data -> training-data baseline, clearly labelled
    r = predict_resolution(broken, _ctx(_nan_features()))
    assert r["status"] == FALLBACK and r["method"] == "BASELINE_TRAINING_CATEGORY_MEDIAN"
    assert r["fallback_reason"] == "ARTIFACT_UNAVAILABLE"
    assert r["estimate_hours"] == pytest.approx(reg.baselines["resolution_category_median_hours"]["POTHOLE"]["median_hours"])
    assert r["range_p10_hours"] is None and any("US training data" in w for w in r["warnings"])
    s = predict_sla(broken, _ctx(_nan_features()))
    assert s["status"] == FALLBACK and s["method"] == "BASELINE_TRAINING_CATEGORY_BREACH_RATE"
    assert s["breach_probability"] is None and s["risk_band"] is None
    # enough local closures -> local baseline wins over the training data
    recs = _closed(12, "POTHOLE", 30.0)
    r = predict_resolution(broken, _ctx(_nan_features(), records=recs))
    assert r["method"] == "BASELINE_LOCAL_CATEGORY_MEDIAN" and r["estimate_hours"] == pytest.approx(30.0)
    assert r["baseline_sample_size"] == 12


def test_local_baseline_uses_only_closures_before_report_time():
    recs = _closed(12, "POTHOLE", 30.0) + [
        IncidentRecord(900, "POTHOLE", T - timedelta(days=1), 12.97, 77.59, T + timedelta(hours=1))]
    d = local_closed_durations(_q(), recs, "POTHOLE")
    assert len(d) == 12                                     # the one closed after T is invisible


def test_invalid_model_output_falls_back(reg):
    class NaNModel:
        def predict(self, X):
            return np.full(len(X), np.nan)

    class BadProb:
        def predict_proba(self, X):
            return np.tile([-0.5, 1.5], (len(X), 1))

    bad = dataclasses.replace(reg, resolution_models={k: NaNModel() for k in ("point", "q10", "q90")},
                              sla_model=BadProb())
    r = predict_resolution(bad, _ctx(_nan_features()))
    s = predict_sla(bad, _ctx(_nan_features()))
    assert r["status"] == s["status"] == FALLBACK
    assert r["fallback_reason"] == s["fallback_reason"] == "INVALID_MODEL_OUTPUT"


def test_unsupported_category_uses_local_baseline_or_is_unavailable(reg):
    r = predict_resolution(reg, _ctx(_nan_features(), cat="FIRE_HAZARD"))
    assert r["status"] == UNAVAILABLE and r["fallback_reason"] == "UNSUPPORTED_CATEGORY"
    assert [x["code"] for x in r["reasons"]] == ["UNSUPPORTED_CATEGORY", "NO_APPLICABLE_BASELINE"]
    recs = _closed(10, "FIRE_HAZARD", 3.0)
    r = predict_resolution(reg, _ctx(_nan_features(), cat="FIRE_HAZARD", records=recs))
    assert r["status"] == FALLBACK and r["method"] == "BASELINE_LOCAL_CATEGORY_MEDIAN"
    s = predict_sla(reg, _ctx(_nan_features(4.0), cat="FIRE_HAZARD", sla=4.0))
    assert s["status"] == UNAVAILABLE                       # no training rate exists for it


def test_sla_override_semantics(reg):
    # no SLA -> nothing to breach, no baseline attempted
    s = predict_sla(reg, _ctx(_nan_features(), sla=None))
    assert s["status"] == UNAVAILABLE and [x["code"] for x in s["reasons"]] == ["NO_SLA"]
    # an admin override inside the trained range is predicted, flagged as untrained value
    s12 = predict_sla(reg, _ctx(_nan_features(12.0), sla=12.0))
    assert s12["method"] == "ML_MODEL" and s12["sla_hours"] == 12.0
    assert any(x["code"] == "FEATURES_OUTSIDE_TRAINING_RANGE" and "sla_hours" in x["detail"]
               for x in s12["reasons"])
    # outside [4, 168]: never extrapolated; local baseline if possible, else unavailable
    for h in (2.0, 200.0):
        s = predict_sla(reg, _ctx(_nan_features(h), sla=h))
        assert s["fallback_reason"] == "SLA_OUT_OF_RANGE" and s["status"] == UNAVAILABLE
    recs = _closed(25, "POTHOLE", 300.0, start=T - timedelta(days=50))
    s = predict_sla(reg, _ctx(_nan_features(200.0), sla=200.0, records=recs))
    assert s["status"] == FALLBACK and s["historical_breach_rate"] == 1.0
    assert s["risk_band"] is None and s["breach_probability"] is None


def test_longer_sla_never_raises_risk(reg):
    probs = [predict_sla(reg, _ctx(_in_range_features(reg), sla=h))["breach_probability"]
             for h in (4.0, 12.0, 24.0, 72.0, 168.0)]
    assert all(a >= b - 1e-12 for a, b in zip(probs, probs[1:]))


def test_insufficient_history_does_not_force_high_risk(reg):
    s = predict_sla(reg, _ctx(_nan_features(168.0), cat="GARBAGE", sla=168.0))
    assert s["status"] == EXPERIMENTAL and s["risk_band"] != "HIGH"


def test_ml_disabled_returns_unavailable_without_fallback():
    off = R.build_registry(False, "", "")
    s = predict_sla(off, _ctx(_nan_features()))
    r = predict_resolution(off, _ctx(_nan_features(), records=_closed(20, "POTHOLE", 5.0)))
    assert s["status"] == r["status"] == UNAVAILABLE
    assert r["fallback_reason"] == "ML_DISABLED" and r["method"] is None


# ---------------------------------------------------------------------------
# hotspot EWMA
# ---------------------------------------------------------------------------
def _reports(weeks: dict[int, int], cat="POTHOLE", lat=12.9716, lon=77.5946, start=None):
    start = start or datetime(2026, 8, 3, tzinfo=UTC)               # a Monday
    out, i = [], 0
    for w, n in weeks.items():
        for k in range(n):
            out.append(IncidentRecord(i, cat, start + timedelta(weeks=w, hours=10 + k), lat, lon, None))
            i += 1
    return out


def test_hotspot_ewma_matches_definition():
    # week 0 is the area's first (partial by definition) and is dropped
    recs = _reports({0: 9, 1: 3, 2: 0, 3: 5, 4: 2, 5: 7, 6: 4})
    now = datetime(2026, 9, 16, 12, tzinfo=UTC)                    # week 6 is current
    out = forecast(recs, AREA, "POTHOLE", now, alpha=0.3, min_weeks=4)
    assert out["status"] == "AVAILABLE" and out["weeks_of_history"] == 5
    cell = out["cells"][0]
    assert cell["predicted_next_week_count"] == pytest.approx(ewma([3, 0, 5, 2, 7], 0.3))
    assert cell["last_week_count"] == 7 and cell["reports_in_history"] == 17
    b = cell["bounds"]
    assert b["south"] <= 12.9716 <= b["north"] and b["west"] <= 77.5946 <= b["east"]


def test_hotspot_short_history_and_unevaluated_category():
    recs = _reports({0: 2, 1: 1, 2: 4})
    now = datetime(2026, 8, 19, tzinfo=UTC)                        # weeks 1 complete, 2 current
    out = forecast(recs, AREA, "POTHOLE", now, alpha=0.3, min_weeks=4)
    assert out["status"] == "EXPERIMENTAL" and out["weeks_of_history"] == 1
    assert out["reasons"][0]["code"] == "INSUFFICIENT_HISTORY_WEEKS"
    out = forecast(_reports({0: 1}), AREA, "POTHOLE", datetime(2026, 8, 5, tzinfo=UTC), 0.3, 4)
    assert out["status"] == "UNAVAILABLE" and out["reasons"][0]["code"] == "NO_COMPLETE_WEEKS"
    out = forecast([], AREA, "POTHOLE", datetime(2026, 8, 5, tzinfo=UTC), 0.3, 4)
    assert out["status"] == "UNAVAILABLE" and out["reasons"][0]["code"] == "NO_DATA"
    recs = _reports({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, cat="FIRE_HAZARD")
    out = forecast(recs, AREA, "FIRE_HAZARD", datetime(2026, 9, 9, tzinfo=UTC), 0.3, 4)
    assert out["status"] == "EXPERIMENTAL"
    assert {r["code"] for r in out["reasons"]} == {"CATEGORY_NOT_EVALUATED"}


def test_week_start_is_monday_utc():
    assert week_start(datetime(2026, 10, 18, 23, 59, tzinfo=UTC)) == datetime(2026, 10, 12, tzinfo=UTC)


# ---------------------------------------------------------------------------
# shadow evaluation: outcome rules and right-censoring
# ---------------------------------------------------------------------------
from app.ml.evaluation import outcome, summarise  # noqa: E402


def test_outcome_rules():
    c = datetime(2026, 10, 1, tzinfo=UTC)
    now = c + timedelta(hours=100)
    assert outcome("resolution", c, c + timedelta(hours=5), None, now) == \
        {"state": "OBSERVED", "observed_hours": 5.0}
    assert outcome("resolution", c, None, None, now) == {"state": "CENSORED", "elapsed_hours": 100.0}
    assert outcome("sla", c, c + timedelta(hours=71), 72, now)["state"] == "MET"
    assert outcome("sla", c, c + timedelta(hours=72), 72, now)["state"] == "BREACHED"  # at deadline
    assert outcome("sla", c, None, 72, now)["state"] == "BREACHED"                   # open, past it
    assert outcome("sla", c, None, 168, now)["state"] == "PENDING"
    assert outcome("sla", c, None, None, now)["state"] == "NO_SLA"


def test_summary_never_scores_censored_rows_as_closed():
    c = datetime(2026, 10, 1, tzinfo=UTC)
    now = c + timedelta(hours=100)
    rows = [
        {"task": "resolution", "status": "EXPERIMENTAL", "method": "ML_MODEL",
         "output": {"estimate_hours": 10.0}, "sla_hours": None, "created_at": c,
         "first_closed_at": c + timedelta(hours=14)},                    # error 4
        {"task": "resolution", "status": "EXPERIMENTAL", "method": "ML_MODEL",
         "output": {"estimate_hours": 50.0}, "sla_hours": None, "created_at": c,
         "first_closed_at": None},                                       # open 100 h > 50 h
        {"task": "resolution", "status": "EXPERIMENTAL", "method": "ML_MODEL",
         "output": {"estimate_hours": 500.0}, "sla_hours": None, "created_at": c,
         "first_closed_at": None},                                       # open, not yet wrong
        {"task": "sla", "status": "EXPERIMENTAL", "method": "ML_MODEL",
         "output": {"breach_probability": 0.8}, "sla_hours": 24, "created_at": c,
         "first_closed_at": None},                                       # breached
        {"task": "sla", "status": "FALLBACK", "method": "BASELINE_LOCAL_RECENT_BREACH_RATE",
         "output": {"historical_breach_rate": 0.3}, "sla_hours": 168, "created_at": c,
         "first_closed_at": None},                                       # pending
    ]
    s = summarise(rows, now)
    res = next(g for g in s["groups"] if g["task"] == "resolution")
    assert res["observed_n"] == 1 and res["mae_hours_observed_only"] == pytest.approx(4.0)
    assert res["outcomes"] == {"OBSERVED": 1, "CENSORED": 2}
    assert res["censored_already_underestimated"] == 1
    model = next(g for g in s["groups"] if g["task"] == "sla" and g["method"] == "ML_MODEL")
    assert model["brier_resolved_only"] == pytest.approx(0.04)
    base = next(g for g in s["groups"] if g["method"].startswith("BASELINE"))
    assert base["resolved_n"] == 0 and base["brier_resolved_only"] is None
    assert not res["enough_for_evaluation"]
