#!/usr/bin/env python3
"""
Phase 2B — hotspot (next-week count) experiments.

PANELS
  A  city x ward/neighbourhood/board x canonical category x UTC ISO week —
     the Phase 2A definition, rebuilt from incidents so the Chicago weeks
     before March 2019 can be dropped. Counts are checked against
     hotspot_ml.parquet row by row.
  C  UrbanEye category x 1 km grid cell x week, Chicago only. UrbanEye has no
     ward column but can compute a grid cell from `location`, so this is the
     panel a deployment could actually build.

TIMING CONVENTION (same as Phase 2A)
  A row is (series, week t). Features use weeks <= t; the target is week t+1.
  The forecast for t+1 is issued when week t has finished.

PROTOCOL
  Rolling origin O (forecast weeks O .. O+9): train rows have t <= O − 2 weeks
  (their target week t+1 <= O − 1 week was observed before the forecast); test
  rows have t in [O − 1 week, O + 9 weeks). The model is frozen at O; its lag
  features keep updating weekly, as they would in production.

    python scripts/phase2b/hotspot_experiments.py --stage all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.phase2b.common import (CHICAGO_CLEAN_FROM, DATA_DIR, EXPERIMENT_DIR, REPORT_DIR, SEED,
                                    UE_SUPPORTED, ExperimentLog, environment, sha256_file,
                                    write_json)
from scripts.train._common import FeaturePreprocessor
from scripts.utils.paths import repo_root

TASK = "hotspot"
OUT = EXPERIMENT_DIR / TASK
KEYS = ["city", "zone_id", "category"]
ORIGINS = [pd.Timestamp(x) for x in ("2019-09-02", "2019-11-11", "2020-01-20", "2020-03-30",
                                     "2020-06-08")]
TEST_WEEKS = 10
WEEK = pd.Timedelta(days=7)
# first clean week whose 12-week window no longer reaches into the artifact
HOTSPOT_CLEAN_FROM = pd.Timestamp("2019-05-27")

FEATURES = ["category", "y_t", "lag1", "lag2", "lag3", "roll4", "roll8", "roll12", "ewm03",
            "trend_4_12", "zero_share_12", "cc_total_t", "cc_shock", "zone_share_12"]
FEATURES_NO_CC = [f for f in FEATURES if not f.startswith("cc_") and f != "zone_share_12"]
MODELS = {
    "H1_poisson": dict(loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                       min_samples_leaf=50, l2_regularization=1.0),
    "H2_absolute": dict(loss="absolute_error", max_iter=300, learning_rate=0.05,
                        max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0),
    "H3_poisson_small": dict(loss="poisson", max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                             min_samples_leaf=200, l2_regularization=5.0),
}


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def _complete(obs: pd.DataFrame, weeks_by_city: dict) -> pd.DataFrame:
    frames = []
    for city, g in obs.groupby("city", observed=True):
        full = weeks_by_city[city]
        series = g[KEYS].drop_duplicates()
        grid = series.merge(pd.DataFrame({"week_start": full}), how="cross")
        f = grid.merge(g, on=KEYS + ["week_start"], how="left")
        f["y_t"] = f["y_t"].fillna(0).astype(float)
        frames.append(f)
    return pd.concat(frames, ignore_index=True).sort_values(KEYS + ["week_start"])


def build_panel_a() -> pd.DataFrame:
    old = pd.read_parquet(repo_root() / "data/processed/ml/hotspot_ml.parquet")
    inc = pd.read_parquet(DATA_DIR / "incident_features_v2.parquet",
                          columns=["city", "zone_id", "category", "reported_at"])
    inc = inc[inc["zone_id"].notna()]
    inc["week_start"] = inc["reported_at"].dt.tz_localize(None).dt.to_period("W").dt.start_time
    series = old[KEYS].drop_duplicates()
    inc = inc.merge(series, on=KEYS, how="inner")
    obs = inc.groupby(KEYS + ["week_start"], observed=True).size().rename("y_t").reset_index()
    # same week grid as the original builder (partial first/last weeks trimmed)
    weeks = {c: pd.date_range(g["week_start"].min(), g["week_start"].max() + WEEK, freq="7D")
             for c, g in old.groupby("city")}
    panel = _complete(obs, weeks)
    return _features(panel)


def build_panel_c(cell_m: float = 1000.0) -> pd.DataFrame:
    inc = pd.read_parquet(DATA_DIR / "incident_features_v2.parquet",
                          columns=["source_dataset", "city", "ue_category", "is_ue_supported",
                                   "latitude", "longitude", "reported_at"])
    inc = inc[(inc["source_dataset"] == "chicago311") & inc["is_ue_supported"]
              & inc["latitude"].notna()].copy()
    lat0, lon0 = inc["latitude"].min(), inc["longitude"].min()
    ref = inc["latitude"].median()
    ix = np.floor((inc["longitude"] - lon0) * 111_320 * np.cos(np.radians(ref)) / cell_m)
    iy = np.floor((inc["latitude"] - lat0) * 110_574 / cell_m)
    inc["zone_id"] = ix.astype(int).astype(str) + "_" + iy.astype(int).astype(str)
    inc["category"] = inc["ue_category"].astype("string")
    inc["week_start"] = inc["reported_at"].dt.tz_localize(None).dt.to_period("W").dt.start_time
    # keep cells with at least one report per ~10 weeks on average in the clean
    # period; the rest are structural zeros that only flatter every predictor
    clean = inc[inc["reported_at"] >= CHICAGO_CLEAN_FROM]
    active = clean.groupby(["zone_id", "category"]).size()
    active = active[active >= 8].reset_index()[["zone_id", "category"]]
    inc = inc.merge(active, on=["zone_id", "category"])
    obs = inc.groupby(KEYS + ["week_start"], observed=True).size().rename("y_t").reset_index()
    w0 = inc["week_start"].min() + WEEK
    w1 = inc["week_start"].max() - WEEK
    panel = _complete(obs[(obs.week_start >= w0) & (obs.week_start <= w1)],
                      {"Chicago": pd.date_range(w0, w1, freq="7D")})
    panel = _features(panel)
    panel.attrs["grid"] = {"cell_m": cell_m, "origin_lat": float(lat0),
                           "origin_lon": float(lon0), "ref_lat": float(ref)}
    return panel


def _features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(KEYS + ["week_start"]).reset_index(drop=True)
    g = panel.groupby(KEYS, observed=True)["y_t"]
    panel["lag1"] = g.shift(1)
    panel["lag2"] = g.shift(2)
    panel["lag3"] = g.shift(3)
    roll = lambda w: g.transform(lambda s: s.rolling(w, min_periods=1).mean())  # noqa: E731
    panel["roll4"] = roll(4)          # weeks t-3..t — the Phase 2A primary baseline
    panel["roll8"] = roll(8)
    panel["roll12"] = roll(12)
    for a in (0.1, 0.2, 0.3, 0.5):
        panel[f"ewm{str(a).replace('.', '')}"] = g.transform(
            lambda s, a=a: s.ewm(alpha=a, adjust=False).mean())
    panel["trend_4_12"] = panel["roll4"] - panel["roll12"]
    panel["zero_share_12"] = g.transform(lambda s: (s == 0).rolling(12, min_periods=1).mean())
    cc = panel.groupby(["city", "category", "week_start"], observed=True)["y_t"].transform("sum")
    panel["cc_total_t"] = cc
    cc_tbl = panel[["city", "category", "week_start"]].assign(cc=cc).drop_duplicates()
    cc_tbl = cc_tbl.sort_values(["city", "category", "week_start"])
    cc_tbl["cc_roll4_prev"] = cc_tbl.groupby(["city", "category"])["cc"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    panel = panel.merge(cc_tbl[["city", "category", "week_start", "cc_roll4_prev"]],
                        on=["city", "category", "week_start"], how="left")
    panel["cc_shock"] = panel["cc_total_t"] / panel["cc_roll4_prev"].replace(0, np.nan)
    cc12 = panel.groupby(["city", "category", "week_start"], observed=True)["roll12"] \
        .transform("sum")
    panel["zone_share_12"] = panel["roll12"] / cc12.replace(0, np.nan)
    panel["n_hist_weeks"] = panel.groupby(KEYS, observed=True).cumcount() + 1
    panel["target"] = panel.groupby(KEYS, observed=True)["y_t"].shift(-1)
    panel = panel[panel["target"].notna()].reset_index(drop=True)
    panel["series_id"] = panel.groupby(KEYS, observed=True).ngroup()
    return panel


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def metrics(y, pred) -> dict:
    from sklearn.metrics import mean_poisson_deviance
    y, pred = np.asarray(y, float), np.clip(np.asarray(pred, float), 0, None)
    err = np.abs(y - pred)
    return {"n": int(len(y)), "MAE": float(err.mean()),
            "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))), "median_AE": float(np.median(err)),
            "poisson_deviance": float(mean_poisson_deviance(np.clip(y, 1e-9, None),
                                                           np.clip(pred, 1e-9, None)))}


def topdecile_hit(te: pd.DataFrame, pred) -> float:
    """Share of actual top-10% series (per city x category x week) that the
    predictor also ranks in its top 10%."""
    d = te[["city", "category", "week_start", "target"]].copy()
    d["pred"] = pred
    hits = tot = 0
    for _, g in d.groupby(["city", "category", "week_start"], observed=True):
        k = max(1, int(round(0.1 * len(g))))
        if len(g) < 10:
            continue
        actual = set(g.nlargest(k, "target").index)
        hits += len(actual & set(g.nlargest(k, "pred").index))
        tot += k
    return float(hits / tot) if tot else float("nan")


def series_bootstrap(te, y, base, model, n=500) -> dict:
    rng = np.random.default_rng(SEED)
    d = pd.DataFrame({"s": te["series_id"].to_numpy(),
                      "gain": np.abs(y - base) - np.abs(y - model)})
    g = d.groupby("s")["gain"]
    s, c = g.sum().to_numpy(), g.size().to_numpy()
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(s), len(s))
        stats.append(s[i].sum() / c[i].sum())
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return {"gain_mean": float(d["gain"].mean()), "ci95": [float(lo), float(hi)],
            "significant": bool(lo > 0 or hi < 0)}


def weekly_win_rate(te, y, base, model) -> dict:
    d = pd.DataFrame({"w": te["week_start"].to_numpy(), "b": np.abs(y - base),
                      "m": np.abs(y - model)}).groupby("w").mean()
    return {"weeks": int(len(d)), "model_wins": int((d["m"] < d["b"]).sum())}


def baselines(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    out = {"B0_persistence": te["y_t"].to_numpy(float),
           "B1_rolling_4w_mean": te["roll4"].to_numpy(float),
           "B2_rolling_8w_mean": te["roll8"].to_numpy(float),
           "B3_rolling_12w_mean": te["roll12"].to_numpy(float)}
    # EWMA with alpha chosen on the TRAINING rows only
    best = min(("ewm01", "ewm02", "ewm03", "ewm05"),
               key=lambda c: np.mean(np.abs(tr["target"] - tr[c])))
    out[f"B4_ewma_trained({best})"] = te[best].to_numpy(float)
    return out


# ---------------------------------------------------------------------------
def fit_predict(params, feats, tr, te):
    from sklearn.ensemble import HistGradientBoostingRegressor
    pre = FeaturePreprocessor(feats).fit(tr)
    m = HistGradientBoostingRegressor(categorical_features=pre.categorical_mask,
                                      random_state=SEED, early_stopping=False, **params)
    t0 = time.time()
    m.fit(pre.transform(tr), tr["target"].to_numpy(float))
    return np.clip(m.predict(pre.transform(te)), 0, None), round(time.time() - t0, 1), m, pre


def folds(panel: pd.DataFrame, city: str = "Chicago", min_hist: int = 4):
    p_ = panel[(panel["city"] == city) & (panel["n_hist_weeks"] >= min_hist)]
    clean = p_["week_start"] >= HOTSPOT_CLEAN_FROM if city == "Chicago" \
        else pd.Series(True, index=p_.index)
    p_ = p_[clean]
    for o in ORIGINS:
        tr = p_[p_["week_start"] <= o - 2 * WEEK]
        te = p_[(p_["week_start"] >= o - WEEK) & (p_["week_start"] < o + (TEST_WEEKS - 1) * WEEK)]
        if len(te):
            yield f"{o:%Y-%m-%d}", tr, te


def evaluate(name, tr, te, feats, model_names) -> dict:
    y = te["target"].to_numpy(float)
    res = {"n_train": int(len(tr)), "n_test": int(len(te)),
           "train_weeks": [str(tr["week_start"].min().date()), str(tr["week_start"].max().date())],
           "test_weeks": [str(te["week_start"].min().date()), str(te["week_start"].max().date())],
           "series": int(te["series_id"].nunique()), "predictors": {}}
    preds = baselines(tr, te)
    for mn in model_names:
        try:
            pr, secs, _, _ = fit_predict(MODELS[mn], feats, tr, te)
            preds[f"ML_{mn}"] = pr
        except Exception as e:  # noqa: BLE001
            res["predictors"][f"ML_{mn}"] = {"status": f"failed: {e!r}",
                                             "traceback": traceback.format_exc()}
    b = preds["B1_rolling_4w_mean"]
    ewma = next(v for k, v in preds.items() if k.startswith("B4_ewma"))
    for k, pr in preds.items():
        m = metrics(y, pr)
        m["topdecile_hit"] = topdecile_hit(te, pr)
        if k.startswith("ML_"):
            m["vs_roll4_bootstrap"] = series_bootstrap(te, y, b, pr)
            m["vs_roll4_weekly"] = weekly_win_rate(te, y, b, pr)
            # the strongest non-ML baseline in every fold measured so far
            m["vs_ewma_bootstrap"] = series_bootstrap(te, y, ewma, pr)
            m["vs_ewma_weekly"] = weekly_win_rate(te, y, ewma, pr)
        res["predictors"][k] = m
    return res, preds


def summarise(per_fold: dict) -> dict:
    names = set().union(*[f["predictors"].keys() for f in per_fold.values()])
    out = {}
    for n in sorted(names):
        rows = [f["predictors"][n] for f in per_fold.values() if "MAE" in f["predictors"].get(n, {})]
        if rows:
            out[n] = {k: float(np.mean([r[k] for r in rows]))
                      for k in ("MAE", "RMSE", "poisson_deviance", "topdecile_hit")}
            if n.startswith("ML_"):
                out[n]["folds_beating_roll4_sig"] = sum(
                    r["vs_roll4_bootstrap"]["ci95"][0] > 0 for r in rows)
                out[n]["folds_losing_to_roll4_sig"] = sum(
                    r["vs_roll4_bootstrap"]["ci95"][1] < 0 for r in rows)
                out[n]["folds_beating_ewma_sig"] = sum(
                    r["vs_ewma_bootstrap"]["ci95"][0] > 0 for r in rows if "vs_ewma_bootstrap" in r)
                out[n]["folds_losing_to_ewma_sig"] = sum(
                    r["vs_ewma_bootstrap"]["ci95"][1] < 0 for r in rows if "vs_ewma_bootstrap" in r)
                out[n]["folds"] = len(rows)
    return out


class Runner:
    def __init__(self, force=False):
        self.force = force
        OUT.mkdir(parents=True, exist_ok=True)
        self.log = ExperimentLog(TASK, {})
        for f in sorted(OUT.glob("*.json")):
            self.log.records.append(json.loads(f.read_text()))
        self.panels = {}

    def panel(self, which):
        if which not in self.panels:
            path = DATA_DIR / f"hotspot_panel_{which}.parquet"
            if path.exists() and not self.force:
                self.panels[which] = pd.read_parquet(path)
            else:
                pnl = build_panel_a() if which == "A" else build_panel_c()
                grid = pnl.attrs.get("grid")
                pnl.to_parquet(path, index=False)
                if grid:
                    write_json(DATA_DIR / f"hotspot_panel_{which}.grid.json", grid)
                self.panels[which] = pnl
        return self.panels[which]

    def _skip(self, exp_id):
        if (OUT / f"{exp_id}.json").exists() and not self.force:
            print(f"skip {exp_id}")
            return True
        self.log.records = [r for r in self.log.records if r["experiment_id"] != exp_id]
        return False

    def rolling(self, exp_id, which, feats, model_names, note=""):
        if self._skip(exp_id):
            return
        pnl = self.panel(which)
        per_fold = {}
        for name, tr, te in folds(pnl):
            per_fold[name], _ = evaluate(name, tr, te, feats, model_names)
            s = per_fold[name]["predictors"]
            print(f"  {exp_id} {name}: roll4={s['B1_rolling_4w_mean']['MAE']:.3f} "
                  + " ".join(f"{k}={v['MAE']:.3f}" for k, v in s.items()
                             if k.startswith("ML_") and "MAE" in v), flush=True)
        self._record(exp_id, which, feats, model_names, "rolling_origin", per_fold, note)

    def ward_holdout(self, exp_id, feats, model_names, n_groups=5):
        """Train on 80% of zones, test on the held-out 20%, at each origin."""
        if self._skip(exp_id):
            return
        pnl = self.panel("A")
        per_fold = {}
        for name, tr, te in folds(pnl):
            zones = np.array(sorted(te["zone_id"].unique()))
            rng = np.random.default_rng(SEED)
            grp = dict(zip(zones, rng.permutation(len(zones)) % n_groups))
            agg = []
            for k in range(n_groups):
                trk = tr[tr["zone_id"].map(grp) != k]
                tek = te[te["zone_id"].map(grp) == k]
                _, preds = evaluate(name, trk, tek, feats, model_names)
                agg.append((tek, preds))
            te_all = pd.concat([a for a, _ in agg])
            preds_all = {p: np.concatenate([b[p] for _, b in agg]) for p in agg[0][1]}
            y = te_all["target"].to_numpy(float)
            res = {"predictors": {}}
            for p_, pr in preds_all.items():
                m = metrics(y, pr)
                m["topdecile_hit"] = topdecile_hit(te_all, pr)
                if p_.startswith("ML_"):
                    ew = next(v for k, v in preds_all.items() if k.startswith("B4_ewma"))
                    m["vs_ewma_bootstrap"] = series_bootstrap(te_all, y, ew, pr)
                    m["vs_roll4_bootstrap"] = series_bootstrap(te_all, y,
                                                               preds_all["B1_rolling_4w_mean"], pr)
                    m["vs_roll4_weekly"] = weekly_win_rate(te_all, y,
                                                           preds_all["B1_rolling_4w_mean"], pr)
                res["predictors"][p_] = m
            per_fold[name] = res
            print(f"  {exp_id} {name}: roll4={res['predictors']['B1_rolling_4w_mean']['MAE']:.3f} "
                  + " ".join(f"{k}={v['MAE']:.3f}" for k, v in res["predictors"].items()
                             if k.startswith("ML_")), flush=True)
        self._record(exp_id, "A", feats, model_names, f"ward holdout ({n_groups} groups)",
                     per_fold, "")

    def transfer_sf(self, exp_id, feats, model_names):
        """Train on clean Chicago weeks, test on San Francisco (never seen)."""
        if self._skip(exp_id):
            return
        pnl = self.panel("A")
        tr = pnl[(pnl["city"] == "Chicago") & (pnl["week_start"] >= HOTSPOT_CLEAN_FROM)
                 & (pnl["n_hist_weeks"] >= 4)]
        te = pnl[(pnl["city"] == "San Francisco") & (pnl["n_hist_weeks"] >= 4)]
        res, _ = evaluate("chicago_to_sf", tr, te, feats, model_names)
        self._record(exp_id, "A", feats, model_names, "train Chicago -> test San Francisco",
                     {"chicago_to_sf": res}, "")

    def _record(self, exp_id, which, feats, model_names, protocol, per_fold, note):
        self.log.record(exp_id, status="ok", panel=which, features=feats,
                        models={m: MODELS[m] for m in model_names}, protocol=protocol,
                        per_fold=per_fold, summary=summarise(per_fold), note=note,
                        algorithm="HistGradientBoostingRegressor + non-ML baselines")
        self.log.flush()


# ---------------------------------------------------------------------------
def track_a(force=False) -> dict:
    """Phase 2A hotspot test rows: existing model vs roll4 vs v2, like-for-like."""
    import joblib
    out = REPORT_DIR / "hotspot_track_a.json"
    if out.exists() and not force:
        print("skip track_a")
        return json.loads(out.read_text())
    root = repo_root()
    old = pd.read_parquet(root / "data/processed/ml/hotspot_ml.parquet")
    pa = pd.read_parquet(DATA_DIR / "hotspot_panel_A.parquet")
    # 1. the rebuilt panel must reproduce the original counts exactly
    chk = old.merge(pa[KEYS + ["week_start", "y_t", "target"]], on=KEYS + ["week_start"],
                    how="left")
    count_match = float((chk["incident_count"] == chk["y_t"]).mean())
    target_match = float((chk["future_incident_count"] == chk["target"]).mean())

    test = old[old["split_global"] == "test"].copy()
    m1 = joblib.load(root / "models/hotspot/model.joblib")
    p1 = joblib.load(root / "models/hotspot/preprocessor.joblib")
    meta = json.loads((root / "models/hotspot/metadata.json").read_text())
    v1 = np.clip(m1.predict(p1.transform(test)), 0, None)
    if meta.get("target_transform") == "log1p":
        v1 = np.clip(np.expm1(m1.predict(p1.transform(test))), 0, None)
    te = test[KEYS + ["week_start", "future_incident_count"]].merge(
        pa, on=KEYS + ["week_start"], how="left", validate="1:1")
    y = te["future_incident_count"].to_numpy(float)
    te["series_id"] = te.groupby(KEYS).ngroup()
    preds = {"existing_v1_model": v1, "B1_rolling_4w_mean": te["roll4"].to_numpy(float),
             "B0_persistence": te["y_t"].to_numpy(float),
             "B3_rolling_12w_mean": te["roll12"].to_numpy(float)}
    clean = (pa["city"] != "Chicago") | (pa["week_start"] >= HOTSPOT_CLEAN_FROM)
    for cut_name, cut in (("trained_to_2019-11-04(p2a_train_cut)", pd.Timestamp("2019-11-04")),
                          ("trained_to_2020-03-16(latest_known)", pd.Timestamp("2020-03-16"))):
        tr = pa[clean & (pa["week_start"] <= cut) & (pa["n_hist_weeks"] >= 4)]
        pr, _, _, _ = fit_predict(MODELS["H1_poisson"], FEATURES, tr, te)
        preds[f"v2_H1_poisson_{cut_name}"] = pr
    tr0 = pa[clean & (pa["week_start"] <= pd.Timestamp("2019-11-04")) & (pa["n_hist_weeks"] >= 4)]
    best = min(("ewm01", "ewm02", "ewm03", "ewm05"),
               key=lambda c: np.mean(np.abs(tr0["target"] - tr0[c])))
    preds[f"B4_ewma_trained({best})"] = te[best].to_numpy(float)
    res = {"rebuilt_panel_matches_original": {"incident_count": count_match,
                                              "future_incident_count": target_match},
           "test_rows": int(len(te)), "predictors": {}}
    for k, pr in preds.items():
        m = metrics(y, pr)
        m["topdecile_hit"] = topdecile_hit(te.assign(target=y), pr)
        if k != "B1_rolling_4w_mean":
            m["vs_roll4_bootstrap"] = series_bootstrap(te, y, preds["B1_rolling_4w_mean"], pr)
            m["vs_roll4_weekly"] = weekly_win_rate(te, y, preds["B1_rolling_4w_mean"], pr)
            m["vs_ewma_bootstrap"] = series_bootstrap(te, y, preds[f"B4_ewma_trained({best})"], pr)
        res["predictors"][k] = m
    res["environment"] = environment()
    write_json(out, res)
    return res


STAGES = {
    "rolling_A": lambda R: (R.rolling("A_rolling_full", "A", FEATURES,
                                      ["H1_poisson", "H2_absolute", "H3_poisson_small"]),
                            R.rolling("A_rolling_no_citywide", "A", FEATURES_NO_CC,
                                      ["H1_poisson"])),
    "rolling_C": lambda R: R.rolling("C_grid1km_rolling", "C", FEATURES,
                                     ["H1_poisson", "H3_poisson_small"]),
    "geo": lambda R: (R.ward_holdout("A_ward_holdout", FEATURES, ["H1_poisson"]),
                      R.transfer_sf("A_transfer_chicago_to_sf", FEATURES, ["H1_poisson"])),
    # the best rolling-origin candidate must pass the same geographic tests
    "geo_candidates": lambda R: (
        R.ward_holdout("A_ward_holdout_H2_H3", FEATURES, ["H2_absolute", "H3_poisson_small"]),
        R.transfer_sf("A_transfer_chicago_to_sf_H2_H3", FEATURES,
                      ["H2_absolute", "H3_poisson_small"])),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["track_a", "all"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    R = Runner(a.force)
    R.panel("A")
    if a.stage in ("track_a", "all"):
        r = track_a(a.force)
        print(json.dumps({k: v.get("MAE") for k, v in r["predictors"].items()}, indent=1))
        print(r["rebuilt_panel_matches_original"])
    for name, fn in STAGES.items():
        if a.stage in (name, "all"):
            print(f"== stage {name}", flush=True)
            fn(R)
    R.log.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
