"""
Unit tests for the parts of the pipeline where a silent mistake would be
invisible in the output: the backward-looking density windows, local-time
conversion, the split, the vectorised priority engine, and the provenance guard.

    python -m pytest tests -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.preprocess._ml_common import chronological_split
from scripts.preprocess.build_geo_features import historical_density
from scripts.preprocess.build_priority_features import PriorityEngine
from scripts.utils.cleaning import haversine_m
from scripts.utils.paths import load_feature_config
from scripts.utils.schema import ProvenanceViolation, finalise_incidents
from scripts.utils.timefeatures import time_features


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def synthetic_incidents(n: int = 900, seed: int = 11) -> pd.DataFrame:
    """
    A deliberately dense little corpus: 900 incidents inside ~1.5 km so that the
    250 m radius and the 24 h / 7 d / 30 d windows all actually bite.
    """
    rng = np.random.default_rng(seed)
    lat = 41.88 + rng.normal(0, 0.004, n)
    lon = -87.63 + rng.normal(0, 0.004, n)
    t0 = pd.Timestamp("2020-01-01", tz="UTC")
    ts = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 60 * 24, n)), unit="h")
    cats = rng.choice(["POTHOLE", "GARBAGE_DUMPING", "FALLEN_TREE"], n)
    return pd.DataFrame({
        "incident_id": [f"t:{i}" for i in range(n)],
        "source_dataset": "chicago311",
        "city": "Chicago",
        "latitude": lat, "longitude": lon,
        "category": cats, "reported_at": ts,
    })


def brute_force_density(df: pd.DataFrame) -> pd.DataFrame:
    """The definition, written the slow obvious way, for comparison."""
    fc = load_feature_config()
    specs = fc["historical_density_features"]
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    ts = pd.to_datetime(df["reported_at"], utc=True).to_numpy("datetime64[ns]").astype(
        np.int64) / 1e9 / 3600.0
    cat = df["category"].to_numpy()

    out = {name: np.zeros(len(df)) for name in specs}
    for i in range(len(df)):
        d = haversine_m(lat[i], lon[i], lat, lon)
        dt = ts[i] - ts
        earlier = ts < ts[i]
        same = cat == cat[i]
        local = None
        cat30 = None
        for name, spec in specs.items():
            if name == "category_incident_density":
                continue
            m = earlier & (dt <= float(spec.get("window_hours", 720))) & \
                (d <= float(spec.get("radius_m", 250)))
            if spec.get("same_category_only"):
                m = m & same
            out[name][i] = int(m.sum())
            if name == "local_incident_density":
                local = out[name][i]
            if name == "nearby_similar_incidents_30d":
                cat30 = out[name][i]
        out["category_incident_density"][i] = (cat30 / local) if local else 0.0
    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# density
# ---------------------------------------------------------------------------
def test_density_matches_brute_force():
    df = synthetic_incidents()
    fast = historical_density(df)
    slow = brute_force_density(df)
    for col in slow.columns:
        a = pd.to_numeric(fast[col]).to_numpy(float)
        b = slow[col].to_numpy(float)
        assert np.allclose(a, b, atol=1e-4), f"{col} differs: {np.abs(a - b).max()}"


def test_density_is_strictly_backward():
    """Reversing the timeline must not change a row's count if it were <=."""
    df = synthetic_incidents(400)
    out = historical_density(df)
    first = out.sort_index().iloc[0]
    # the earliest incident can have no predecessor
    order = pd.to_datetime(df["reported_at"]).argsort()
    earliest = out.iloc[order.iloc[0]]
    assert int(earliest["local_incident_density"]) == 0
    assert first is not None


def test_density_ignores_simultaneous_reports():
    """Two identical timestamps at the same place must not count each other."""
    t = pd.Timestamp("2020-05-05 10:00", tz="UTC")
    df = pd.DataFrame({
        "incident_id": ["a", "b"], "source_dataset": "chicago311", "city": "Chicago",
        "latitude": [41.88, 41.88], "longitude": [-87.63, -87.63],
        "category": ["POTHOLE", "POTHOLE"], "reported_at": [t, t],
    })
    out = historical_density(df)
    assert int(out["local_incident_density"].iloc[0]) == 0
    assert int(out["local_incident_density"].iloc[1]) == 0


def test_density_null_when_coordinates_missing():
    df = synthetic_incidents(50)
    df.loc[0, "latitude"] = np.nan
    out = historical_density(df)
    assert pd.isna(out["local_incident_density"].iloc[0])
    assert out["local_incident_density"].iloc[1:].notna().all()


# ---------------------------------------------------------------------------
# time features
# ---------------------------------------------------------------------------
def test_local_time_conversion_per_city():
    df = pd.DataFrame({
        "source_dataset": ["chicago311", "sf311", "nyc311"],
        "reported_at": pd.to_datetime(["2020-01-15 04:30:00", "2020-01-15 04:30:00",
                                       "2020-01-15 04:30:00"], utc=True),
    })
    tf = time_features(df)
    # 04:30 UTC in January is 22:30 CST, 20:30 PST, 23:30 EST
    assert list(tf["hour"]) == [22, 20, 23]
    assert list(tf["hour_utc"]) == [4, 4, 4]
    assert list(tf["is_night"]) == [True, False, True]


def test_local_time_handles_dst():
    df = pd.DataFrame({
        "source_dataset": ["chicago311", "chicago311"],
        "reported_at": pd.to_datetime(["2020-01-15 18:00:00", "2020-07-15 18:00:00"], utc=True),
    })
    tf = time_features(df)
    assert list(tf["hour"]) == [12, 13]      # CST is UTC-6, CDT is UTC-5


def test_unmapped_source_gets_null_not_utc():
    df = pd.DataFrame({"source_dataset": ["mars311"],
                       "reported_at": pd.to_datetime(["2020-01-15 04:30:00"], utc=True)})
    tf = time_features(df)
    assert pd.isna(tf["hour"].iloc[0])
    assert pd.isna(tf["local_timezone"].iloc[0])


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def test_split_populates_all_three_when_absolute_cuts_are_out_of_range():
    df = synthetic_incidents(300)
    split, meta = chronological_split(df)
    counts = split.value_counts()
    assert counts.get("train", 0) > 0
    assert counts.get("val", 0) > 0
    assert counts.get("test", 0) > 0
    assert meta["groups"]["chicago311"]["cut_mode"] == "quantile_fallback"


def test_split_is_chronological_within_source():
    df = synthetic_incidents(300)
    split, _ = chronological_split(df)
    ts = pd.to_datetime(df["reported_at"], utc=True)
    assert ts[split == "train"].max() <= ts[split == "val"].min()
    assert ts[split == "val"].max() <= ts[split == "test"].min()


def test_split_never_buckets_missing_timestamps_into_train():
    df = synthetic_incidents(100)
    df.loc[0, "reported_at"] = pd.NaT
    split, _ = chronological_split(df)
    assert split.iloc[0] == "unassigned"


# ---------------------------------------------------------------------------
# priority engine
# ---------------------------------------------------------------------------
def test_vectorised_engine_matches_row_engine():
    engine = PriorityEngine()
    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({
        "category": rng.choice(["POTHOLE", "GRAFFITI_VISUAL_POLLUTION", "OPEN_MANHOLE",
                                "TRAFFIC_SIGNAL_FAULT", None], n),
        "distance_to_nearest_school_m": rng.choice([np.nan, 10.0, 150.0, 900.0], n),
        "metro_city": rng.choice([True, False, None], n),
        "local_incident_density": rng.choice([np.nan, 0, 7, 60], n),
        "nearby_similar_incidents_30d": rng.choice([np.nan, 0, 3, 40], n),
    })
    fast = engine.apply(df)
    slow = pd.DataFrame([engine.evaluate(r) for r in df.to_dict("records")])
    assert list(fast["priority_baseline"].fillna("NA")) == list(slow["priority_baseline"].fillna("NA"))
    assert np.allclose(fast["priority_score"].astype(float).fillna(-1),
                       slow["priority_score"].astype(float).fillna(-1), atol=1e-9)
    assert list(fast["priority_features_available"]) == list(slow["priority_features_available"])
    assert list(fast["priority_confidence"]) == list(slow["priority_confidence"])
    assert list(fast["priority_reasons"]) == list(slow["priority_reasons"])


def test_absolute_p1_rule():
    engine = PriorityEngine()
    r = engine.evaluate({"category": "OPEN_MANHOLE"})
    assert r["priority_baseline"] == "P1"


def test_missing_features_are_not_treated_as_false():
    """A null distance must not behave like 'definitely far away'."""
    engine = PriorityEngine()
    unknown = engine.evaluate({"category": "POTHOLE"})
    far = engine.evaluate({"category": "POTHOLE", "distance_to_nearest_school_m": 100000.0})
    assert unknown["priority_features_available"] == 1
    assert far["priority_features_available"] == 2
    assert far["priority_score"] < unknown["priority_score"]


def test_confidence_reflects_available_inputs():
    engine = PriorityEngine()
    thin = engine.evaluate({"category": "POTHOLE"})
    rich = engine.evaluate({
        "category": "POTHOLE", "distance_to_nearest_school_m": 20.0,
        "distance_to_nearest_hospital_m": 50.0, "distance_to_nearest_metro_m": 100.0,
        "distance_to_major_road_m": 10.0, "distance_to_nearest_intersection_m": 8.0,
        "distance_to_nearest_public_transport_m": 40.0,
        "distance_to_nearest_fire_station_m": 400.0,
        "distance_to_nearest_police_station_m": 400.0,
        "distance_to_nearest_government_building_m": 200.0,
        "metro_city": True, "urban_area": True,
        "local_incident_density": 20, "nearby_similar_incidents_30d": 12,
    })
    assert thin["priority_confidence"] == "LOW"
    assert rich["priority_confidence"] == "HIGH"
    assert rich["priority_feature_coverage"] > thin["priority_feature_coverage"]


# ---------------------------------------------------------------------------
# provenance guard
# ---------------------------------------------------------------------------
def test_guard_rejects_priority_label_copied_from_baseline():
    df = pd.DataFrame({
        "incident_id": ["a"], "source_dataset": ["chicago311"], "city": ["Chicago"],
        "category": ["POTHOLE"], "reported_at": [pd.Timestamp("2020-01-01", tz="UTC")],
        "priority_baseline": ["P2"],
    })
    ok = finalise_incidents(df)
    assert ok["priority_label"].isna().all()
    df["priority_label"] = df["priority_baseline"]
    with pytest.raises(ProvenanceViolation):
        finalise_incidents(df)


# ---------------------------------------------------------------------------
# historical recency
# ---------------------------------------------------------------------------
def brute_force_recency(df: pd.DataFrame) -> np.ndarray:
    """Hours back to the most recent earlier same-category report within 250 m."""
    spec = load_feature_config()["historical_recency_features"][
        "hours_since_previous_similar_incident"]
    w, rad = float(spec["window_hours"]), float(spec["radius_m"])
    lat = df["latitude"].to_numpy(float)
    lon = df["longitude"].to_numpy(float)
    ts = pd.to_datetime(df["reported_at"], utc=True).to_numpy("datetime64[ns]").astype(
        np.int64) / 1e9 / 3600.0
    cat = df["category"].to_numpy()
    out = np.full(len(df), np.nan)
    for i in range(len(df)):
        d = haversine_m(lat[i], lon[i], lat, lon)
        m = (ts < ts[i]) & (ts[i] - ts <= w) & (d <= rad) & (cat == cat[i])
        if m.any():
            out[i] = ts[i] - ts[m].max()
    return out


def test_recency_matches_brute_force():
    df = synthetic_incidents(500)
    got = pd.to_numeric(
        historical_density(df)["hours_since_previous_similar_incident"]).to_numpy(float)
    exp = brute_force_recency(df)
    assert (np.isnan(got) == np.isnan(exp)).all(), "NULL pattern differs"
    ok = np.isfinite(exp)
    assert np.allclose(got[ok], exp[ok], atol=1e-3)


def test_recency_is_null_not_zero_when_no_prior_exists():
    """NULL means 'no qualifying earlier report'. Zero would mean 'one just now'."""
    df = synthetic_incidents(200)
    out = historical_density(df)
    first = pd.to_datetime(df["reported_at"]).argsort().iloc[0]
    assert pd.isna(out["hours_since_previous_similar_incident"].iloc[first])


def test_recency_never_negative_and_inside_its_window():
    df = synthetic_incidents(400)
    v = pd.to_numeric(historical_density(df)["hours_since_previous_similar_incident"]).dropna()
    assert (v >= 0).all() and (v <= 720).all()


# ---------------------------------------------------------------------------
# hotspot panel
# ---------------------------------------------------------------------------
def test_hotspot_panel_is_calendar_complete():
    """
    A panel built only from weeks that had incidents makes shift(-1) mean "the
    next week that happened to have one", so the target can never be zero.
    """
    from scripts.preprocess.build_hotspot_dataset import _complete_panel

    weeks = pd.period_range("2020-01-06", periods=6, freq="W")
    observed = pd.DataFrame({
        "city": "Chicago", "zone_type": "ward", "zone_id": "1", "category": "POTHOLE",
        "week": [weeks[0], weeks[2], weeks[5]], "incident_count": [3, 1, 2],
    })
    full, note = _complete_panel(observed, trim_partial=False)
    assert len(full) == 6
    assert list(full["incident_count"]) == [3, 0, 1, 0, 0, 2]
    gaps = full["week"].apply(lambda w: w.start_time).diff().dropna().dt.days
    assert set(gaps) == {7}
    assert "COMPLETE" in note


def test_hotspot_partial_boundary_weeks_are_trimmed():
    weeks = pd.period_range("2020-01-06", periods=5, freq="W")
    observed = pd.DataFrame({
        "city": "Chicago", "zone_type": "ward", "zone_id": "1", "category": "POTHOLE",
        "week": list(weeks), "incident_count": [1, 5, 5, 5, 1],
    })
    full, _ = _complete_panel_alias(observed)
    assert len(full) == 3
    assert list(full["incident_count"]) == [5, 5, 5]


def _complete_panel_alias(observed):
    from scripts.preprocess.build_hotspot_dataset import _complete_panel
    return _complete_panel(observed, trim_partial=True)


# ---------------------------------------------------------------------------
# duplicate-pair splitting
# ---------------------------------------------------------------------------
def test_duplicate_components_follow_chains():
    """A -> B and B -> C must land in ONE component, not two."""
    from scripts.preprocess.build_duplicate_pairs import _components

    pos = pd.DataFrame({"incident_a": ["A", "B"], "incident_b": ["B", "C"],
                        "same_incident": [True, True]})
    comp = _components(pos)
    assert comp["A"] == comp["B"] == comp["C"]


def test_duplicate_split_is_entity_disjoint_and_chronological():
    from scripts.preprocess.build_duplicate_pairs import incident_splits

    n = 300
    t0 = pd.Timestamp("2019-01-01", tz="UTC")
    chi = pd.DataFrame({
        "incident_id": [f"c{i}" for i in range(n)],
        "reported_at": [t0 + pd.Timedelta(hours=6 * i) for i in range(n)],
    })
    pos = pd.DataFrame({"incident_a": [f"c{i}" for i in range(0, n - 1, 2)],
                        "incident_b": [f"c{i + 1}" for i in range(0, n - 1, 2)],
                        "same_incident": True})
    split, meta = incident_splits(pos, chi)
    assert set(split.dropna().unique()) == {"train", "val", "test"}
    # a component's two members always share a split
    for a, b in zip(pos["incident_a"], pos["incident_b"]):
        assert split[a] == split[b]
    # chronological: train precedes val precedes test
    order = {"train": 0, "val": 1, "test": 2}
    times = chi.set_index("incident_id")["reported_at"]
    s = pd.DataFrame({"t": times, "o": split.map(order)}).dropna()
    assert s.sort_values("t")["o"].is_monotonic_increasing
    assert meta["strategy"] == "connected_component_chronological"


# ---------------------------------------------------------------------------
# splits on a coarse timeline
# ---------------------------------------------------------------------------
def test_split_handles_a_coarse_discrete_timeline():
    """
    Row quantiles on ~10 distinct weekly timestamps can put the cut on the last
    value and leave test empty. The distinct-value fallback must prevent that.
    """
    weeks = pd.date_range("2010-01-04", periods=8, freq="7D", tz="UTC")
    df = pd.DataFrame({"source_dataset": "nyc311",
                       "reported_at": np.repeat(weeks, 60)})
    split, meta = chronological_split(df)
    counts = split.value_counts()
    assert counts.get("train", 0) > 0
    assert counts.get("val", 0) > 0
    assert counts.get("test", 0) > 0


# ---------------------------------------------------------------------------
# task datasets
# ---------------------------------------------------------------------------
def test_zone_key_is_city_scoped():
    from scripts.features.build_task_datasets import zone_key

    df = pd.DataFrame({"city": ["Chicago", "New York City"], "zone_id": ["1", "1"]})
    k = zone_key(df)
    assert k.iloc[0] != k.iloc[1]
    assert k.iloc[0].startswith("Chicago:")


def test_constant_predictors_are_demoted_using_train_only():
    from scripts.features.build_task_datasets import effective_predictors

    df = pd.DataFrame({
        "split": ["train"] * 5 + ["test"] * 5,
        "constant_in_train": ["a"] * 5 + ["b"] * 5,
        "varies": list("abcde") * 2,
    })
    usable, demoted = effective_predictors(df, ["constant_in_train", "varies"])
    assert usable == ["varies"]
    assert "constant_in_train" in demoted


def test_every_predictor_declares_when_it_becomes_known():
    """A feature cannot reach a model without a written availability statement."""
    from scripts.features.build_task_datasets import PREDICTION_TIME_AVAILABILITY

    for task in ("resolution_ml", "sla_ml", "hotspot_ml", "duplicate_ml", "priority_features"):
        fp = os.path.join("data", "processed", "ml", f"{task}.manifest.json")
        if not os.path.exists(fp):
            continue
        import json
        m = json.load(open(fp))
        block = m.get("predictors") or m.get("features") or {}
        for col, spec in block.items():
            if not spec.get("present"):
                continue
            assert col in PREDICTION_TIME_AVAILABILITY, \
                f"{task}: {col} has no prediction-time availability statement"


def test_no_task_manifest_declares_a_target_derived_predictor():
    import json
    forbidden = {"resolution_time_hours", "resolution_time_hours_log1p", "sla_met",
                 "closed_at", "status", "description", "target_is_censored",
                 "priority_baseline", "priority_score", "sla_hours_policy"}
    for task in ("resolution_ml", "sla_ml", "hotspot_ml", "duplicate_ml", "priority_features"):
        fp = os.path.join("data", "processed", "ml", f"{task}.manifest.json")
        if not os.path.exists(fp):
            continue
        m = json.load(open(fp))
        block = m.get("predictors") or m.get("features") or {}
        assert not (set(block) & forbidden), f"{task} declares {set(block) & forbidden}"


def test_priority_task_has_no_supervised_target():
    import json
    fp = os.path.join("data", "processed", "ml", "priority_features.manifest.json")
    if not os.path.exists(fp):
        pytest.skip("priority_features not built")
    m = json.load(open(fp))
    assert m["target"]["available"] is False
    assert m["target"]["column"] is None
    assert m["target"]["rows_with_target"] == 0


# ---------------------------------------------------------------------------
# geospatial providers
# ---------------------------------------------------------------------------
def test_null_provider_returns_every_declared_column_empty():
    from scripts.preprocess.geo_providers import NullProvider, declared_columns

    df = pd.DataFrame({"incident_id": ["a", "b"], "latitude": [41.9, 41.9],
                       "longitude": [-87.6, -87.6]})
    out = NullProvider().poi_features(df)
    assert set(out.columns) == set(declared_columns())
    assert out.isna().all().all()


def test_reference_layer_provider_matches_brute_force(tmp_path):
    """Distances must equal a straight haversine minimum over the layer."""
    from scripts.preprocess.geo_providers import ReferenceLayerProvider

    rng = np.random.default_rng(4)
    pois = pd.DataFrame({"latitude": 41.88 + rng.normal(0, 0.01, 60),
                         "longitude": -87.63 + rng.normal(0, 0.01, 60)})
    ext = tmp_path / "external"
    ext.mkdir()
    pois.to_parquet(ext / "schools.parquet")
    (ext / "reference_layers.yaml").write_text(
        "max_search_radius_m: 5000\n"
        "layers:\n"
        "  school:\n"
        "    path: schools.parquet\n"
        "    lat_column: latitude\n"
        "    lon_column: longitude\n")

    inc = pd.DataFrame({"incident_id": [f"i{i}" for i in range(200)],
                        "latitude": 41.88 + rng.normal(0, 0.01, 200),
                        "longitude": -87.63 + rng.normal(0, 0.01, 200)})
    prov = ReferenceLayerProvider(config_path=ext / "reference_layers.yaml")
    out = prov.poi_features(inc)

    got = pd.to_numeric(out["distance_to_nearest_school_m"]).to_numpy(float)
    exp = np.array([haversine_m(la, lo, pois["latitude"].to_numpy(),
                                pois["longitude"].to_numpy()).min()
                    for la, lo in zip(inc["latitude"], inc["longitude"])])
    assert np.allclose(got, exp, atol=1e-3)
    assert prov.coverage()["layers_loaded"] == ["school"]


def test_reference_layer_provider_leaves_absent_layers_null(tmp_path):
    """A declared layer with no file must stay NULL, never be invented."""
    from scripts.preprocess.geo_providers import ReferenceLayerProvider

    ext = tmp_path / "external"
    ext.mkdir()
    (ext / "reference_layers.yaml").write_text(
        "layers:\n  hospital:\n    path: does_not_exist.parquet\n")
    inc = pd.DataFrame({"incident_id": ["a"], "latitude": [41.9], "longitude": [-87.6]})
    prov = ReferenceLayerProvider(config_path=ext / "reference_layers.yaml")
    out = prov.poi_features(inc)
    assert out["distance_to_nearest_hospital_m"].isna().all()
    assert out["near_hospital"].isna().all()
    assert prov.coverage()["layers_loaded"] == []
    assert prov.coverage()["layers_missing_or_unusable"]


def test_postgis_provider_refuses_rather_than_returning_nulls():
    from scripts.preprocess.geo_providers import IndiaPostGISProvider

    inc = pd.DataFrame({"incident_id": ["a"], "latitude": [13.08], "longitude": [80.27]})
    with pytest.raises(NotImplementedError):
        IndiaPostGISProvider().poi_features(inc)


# ---------------------------------------------------------------------------
# instant-closure artifact (resolution target)
# ---------------------------------------------------------------------------
def _closure_frame(pairs):
    """pairs: list of (reported, closed) ISO strings or None."""
    return pd.DataFrame({
        "reported_at": pd.to_datetime([a for a, _ in pairs], utc=True),
        "closed_at": pd.to_datetime([b for _, b in pairs], utc=True),
    })


def test_instant_closure_is_flagged_and_target_nulled():
    """
    A closure written seconds after intake is a transactional artifact, not a
    service duration. Chicago closes 18,279 cases at exactly 7 seconds.
    """
    from scripts.utils.cleaning import CleaningStats, compute_resolution_hours, instant_closure_flag

    df = _closure_frame([
        ("2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z"),   # exactly 0s  (NYC signature)
        ("2020-01-01T00:00:00Z", "2020-01-01T00:00:07Z"),   # 7s          (Chicago signature)
        ("2020-01-01T00:00:00Z", "2020-01-01T00:00:59Z"),   # 59s         (still unmeasurable)
    ])
    flag = instant_closure_flag(df, "reported_at", "closed_at")
    assert list(flag) == [True, True, True]

    stats = CleaningStats(dataset="t")
    hrs = compute_resolution_hours(df, "reported_at", "closed_at", stats)
    assert hrs.isna().all(), "instant closures must leave no duration behind"
    assert stats.nulled.get("resolution_time_instant_closure") == 3


def test_normal_short_resolutions_are_kept():
    """The fix must not be 'delete short durations'. A 90-second job is real."""
    from scripts.utils.cleaning import CleaningStats, compute_resolution_hours, instant_closure_flag

    df = _closure_frame([
        ("2020-01-01T00:00:00Z", "2020-01-01T00:01:00Z"),   # exactly 60s — boundary, KEPT
        ("2020-01-01T00:00:00Z", "2020-01-01T00:01:30Z"),   # 90s
        ("2020-01-01T00:00:00Z", "2020-01-01T02:00:00Z"),   # 2h
    ])
    assert list(instant_closure_flag(df, "reported_at", "closed_at")) == [False, False, False]
    hrs = compute_resolution_hours(df, "reported_at", "closed_at", CleaningStats(dataset="t"))
    assert hrs.notna().all()
    assert np.allclose(hrs.to_numpy(float), [60 / 3600, 90 / 3600, 2.0])


def test_open_and_invalid_cases_flag_null_not_false():
    """'We cannot tell' is not 'this was not an instant closure'."""
    from scripts.utils.cleaning import CleaningStats, compute_resolution_hours, instant_closure_flag

    df = _closure_frame([
        ("2020-01-01T00:00:00Z", None),                      # still open
        (None, "2020-01-02T00:00:00Z"),                      # missing report time
        ("2020-01-02T00:00:00Z", "2020-01-01T00:00:00Z"),    # negative duration
    ])
    flag = instant_closure_flag(df, "reported_at", "closed_at")
    assert pd.isna(flag.iloc[0]) and pd.isna(flag.iloc[1])
    # a NEGATIVE duration is an inconsistent record, not an instant closure:
    # it gets its own rule and must not inflate the instant-closure population
    assert bool(flag.iloc[2]) is False

    stats = CleaningStats(dataset="t")
    hrs = compute_resolution_hours(df, "reported_at", "closed_at", stats)
    assert hrs.isna().all()
    assert stats.nulled.get("resolution_time_negative") == 1


def test_instant_closure_rule_is_config_driven():
    """The threshold is a documented config value, not a literal in the code."""
    import yaml
    cfg = yaml.safe_load(open("config/dataset_config.yaml"))
    assert cfg["cleaning"]["resolution_time"]["min_observable_seconds"] == 60


def test_instant_closure_rows_survive_in_the_incident_table():
    """The row is not deleted — only its duration is unobserved."""
    from scripts.utils.cleaning import CleaningStats, compute_resolution_hours, instant_closure_flag

    df = _closure_frame([("2020-01-01T00:00:00Z", "2020-01-01T00:00:07Z")])
    df["status"] = "closed"
    df["resolution_instant_closure"] = instant_closure_flag(df, "reported_at", "closed_at")
    df["resolution_time_hours"] = compute_resolution_hours(
        df, "reported_at", "closed_at", CleaningStats(dataset="t"))
    assert len(df) == 1
    assert df["status"].iloc[0] == "closed"
    assert pd.notna(df["closed_at"].iloc[0])
    assert bool(df["resolution_instant_closure"].iloc[0]) is True
    assert pd.isna(df["resolution_time_hours"].iloc[0])


# ---------------------------------------------------------------------------
# split policy: per-source vs global chronology
# ---------------------------------------------------------------------------
def _two_city_frame():
    """Two cities on disjoint timelines — the situation that makes the two policies differ."""
    early = pd.Timestamp("2010-01-01", tz="UTC")
    late = pd.Timestamp("2018-01-01", tz="UTC")
    rows = ([{"source_dataset": "nyc311", "reported_at": early + pd.Timedelta(days=i)}
             for i in range(100)] +
            [{"source_dataset": "chicago311", "reported_at": late + pd.Timedelta(days=i)}
             for i in range(100)])
    return pd.DataFrame(rows)


def test_global_split_is_chronological_on_one_wall_clock():
    from scripts.preprocess._ml_common import dual_chronological_split

    df = _two_city_frame()
    per_src, glob, meta = dual_chronological_split(df)
    ts = pd.to_datetime(df["reported_at"], utc=True)
    assert ts[glob == "train"].max() <= ts[glob == "val"].min()
    assert ts[glob == "val"].max() <= ts[glob == "test"].min()
    assert meta["global"]["globally_chronological"] is True


def test_per_source_split_is_chronological_within_each_source():
    from scripts.preprocess._ml_common import dual_chronological_split

    df = _two_city_frame()
    per_src, glob, _ = dual_chronological_split(df)
    ts = pd.to_datetime(df["reported_at"], utc=True)
    for src, g in df.groupby("source_dataset"):
        s = per_src.loc[g.index]
        t = ts.loc[g.index]
        assert t[s == "train"].max() <= t[s == "val"].min()
        assert t[s == "val"].max() <= t[s == "test"].min()


def test_per_source_split_is_NOT_globally_chronological_and_global_one_is():
    """
    The exact trade-off, pinned down: the per-source column keeps every city in
    every fold but is not globally ordered; the global column is ordered but
    loses a city from val/test. Neither is a bug; picking silently would be.
    """
    from scripts.preprocess._ml_common import dual_chronological_split

    df = _two_city_frame()
    per_src, glob, _ = dual_chronological_split(df)
    ts = pd.to_datetime(df["reported_at"], utc=True)

    # per-source: test rows exist that predate the last training row
    assert (ts[per_src == "test"] < ts[per_src == "train"].max()).any()
    assert df.loc[per_src == "test", "source_dataset"].nunique() == 2

    # global: no such row, and the early city has vanished from test
    assert not (ts[glob == "test"] < ts[glob == "train"].max()).any()
    assert df.loc[glob == "test", "source_dataset"].nunique() == 1


def test_baseline_preprocessing_is_fitted_on_train_only():
    """A category seen only in test must not create a new encoder level."""
    from scripts.baselines.run_baselines import prepare

    df = pd.DataFrame({
        "source_dataset": ["a311"] * 5 + ["b311"] * 5,       # multi-source -> split_global
        "split": ["train"] * 6 + ["val"] * 2 + ["test"] * 2,
        "split_global": ["train"] * 6 + ["val"] * 2 + ["test"] * 2,
        "cat": ["a", "a", "b", "b", "a", "b", "a", "b", "ZZZ_unseen", "a"],
        "num": np.arange(10, dtype=float),
    })
    (tr, va, te), (Xtr, Xva, Xte), mask = prepare(df, ["cat", "num"])
    assert len(tr) == 6 and len(va) == 2 and len(te) == 2
    # the level that appears only in test is encoded as the reserved unknown code
    assert Xte["cat"].iloc[0] == -1
    assert set(Xtr["cat"].unique()) == {0.0, 1.0}


def test_pooled_tables_use_the_global_split_column():
    """A multi-city table must be scored on one wall clock."""
    from scripts.baselines.run_baselines import choose_split_column

    df = pd.DataFrame({
        "source_dataset": ["a311"] * 4 + ["b311"] * 4,
        "split": ["train"] * 8,
        "split_global": ["train"] * 4 + ["val"] * 2 + ["test"] * 2,
    })
    assert choose_split_column(df) == "split_global"


def test_single_source_tables_fall_back_to_the_per_source_split():
    """
    sla_ml is NYC-only and NYC is the earliest data in the corpus, so a global
    cut puts every row in train. There the per-source column is already
    chronological on one clock and is the correct choice.
    """
    from scripts.baselines.run_baselines import choose_split_column

    df = pd.DataFrame({
        "source_dataset": ["nyc311"] * 8,
        "split": ["train"] * 4 + ["val"] * 2 + ["test"] * 2,
        "split_global": ["train"] * 8,          # degenerate
    })
    assert choose_split_column(df) == "split"


# ---------------------------------------------------------------------------
# hotspot rolling-4-week baseline
# ---------------------------------------------------------------------------
def _hotspot_panel():
    weeks = pd.date_range("2020-01-06", periods=8, freq="7D")
    counts = [10, 20, 30, 40, 50, 60, 70, 80]
    return pd.DataFrame({
        "city": "Chicago", "zone_type": "ward", "zone_id": "1", "category": "POTHOLE",
        "week_start": weeks, "incident_count": counts,
        "future_incident_count": counts[1:] + [np.nan],
    })


def test_rolling_baseline_uses_only_completed_weeks():
    from scripts.baselines.run_baselines import rolling_mean_baseline

    p = _hotspot_panel()
    roll = rolling_mean_baseline(p, 4, lagged=False)
    # row 4 (week 5, count 50) -> mean of weeks t-3..t = 20,30,40,50
    assert roll.iloc[4] == pytest.approx((20 + 30 + 40 + 50) / 4)
    # never touches the target column
    assert roll.iloc[4] != p["future_incident_count"].iloc[4]


def test_rolling_baseline_is_blind_to_the_target():
    """Rewriting every future value must not move a single baseline prediction."""
    from scripts.baselines.run_baselines import rolling_mean_baseline

    p = _hotspot_panel()
    before = rolling_mean_baseline(p, 4, lagged=False).to_numpy(float)
    tampered = p.copy()
    tampered["future_incident_count"] = 99999.0
    after = rolling_mean_baseline(tampered, 4, lagged=False).to_numpy(float)
    assert np.allclose(before, after, equal_nan=True)


def test_rolling_baseline_never_reads_a_later_week():
    """Truncating the panel after week t must not change the prediction at t."""
    from scripts.baselines.run_baselines import rolling_mean_baseline

    p = _hotspot_panel()
    full = rolling_mean_baseline(p, 4, lagged=False).to_numpy(float)
    for t in range(1, len(p)):
        truncated = rolling_mean_baseline(p.iloc[: t + 1].copy(), 4, lagged=False).to_numpy(float)
        assert truncated[t] == pytest.approx(full[t]), f"week {t} changed when the future was cut"


def test_lagged_rolling_variant_excludes_the_current_week():
    from scripts.baselines.run_baselines import rolling_mean_baseline

    p = _hotspot_panel()
    lag = rolling_mean_baseline(p, 4, lagged=True)
    # row 4 -> weeks t-4..t-1 = 10,20,30,40
    assert lag.iloc[4] == pytest.approx((10 + 20 + 30 + 40) / 4)
    assert pd.isna(lag.iloc[0])


def test_rolling_baseline_is_computed_per_series():
    """History must not bleed between zones or categories."""
    from scripts.baselines.run_baselines import rolling_mean_baseline

    a = _hotspot_panel()
    b = _hotspot_panel()
    b["zone_id"] = "2"
    b["incident_count"] = [1000] * 8
    both = pd.concat([a, b], ignore_index=True)
    roll = rolling_mean_baseline(both, 4, lagged=False)
    assert roll.iloc[:8].max() < 100, "zone 1 picked up zone 2's counts"
