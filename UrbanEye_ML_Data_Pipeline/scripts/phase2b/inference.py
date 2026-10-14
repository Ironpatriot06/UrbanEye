"""
Phase 2B — inference contracts for the v2 artifacts.

This is NOT a service and is not wired into FastAPI. It defines, in code, what
each v2 model needs as input and what it returns, so Phase 2C can integrate
against a tested contract instead of a description. Every predictor:

  * loads `models/<task>_v2/{model,preprocessor}.joblib` + `metadata.json`;
  * takes plain dicts (one per incident / series) and returns plain dicts;
  * computes calendar features itself from `created_at` in Asia/Kolkata;
  * expects history features (workload / recent outcomes) to be supplied by
    the caller — in Phase 2C a PostGIS/SQL layer (PHASE_2B_REPORT.md §13);
    missing ones are allowed and handled as missing, exactly as in training;
  * refuses unsupported categories (FIRE_HAZARD, OTHER) instead of guessing:
    `supported: false` with a reason, and no number.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.phase2b.common import UE_SUPPORTED, default_sla_hours
from scripts.utils.paths import repo_root

LOCAL_TZ = "Asia/Kolkata"
NIGHT_START, NIGHT_END = 22, 6  # same window as config/feature_config.yaml


def _load(task: str):
    import joblib
    d = repo_root() / "models" / f"{task}_v2"
    meta = json.loads((d / "metadata.json").read_text())
    return joblib.load(d / "model.joblib"), joblib.load(d / "preprocessor.joblib"), meta, d


def calendar_features(created_at) -> dict:
    ts = pd.Timestamp(created_at)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    loc = ts.tz_convert(LOCAL_TZ)
    h = int(loc.hour)
    return {"hour": h, "day_of_week": int(loc.dayofweek), "month": int(loc.month),
            "is_weekend": bool(loc.dayofweek >= 5),
            "is_night": bool(h >= NIGHT_START or h < NIGHT_END)}


def _frame(rows: list[dict], features: list[str], category_col: str) -> pd.DataFrame:
    recs = []
    for r in rows:
        rec = {**calendar_features(r["created_at"]), **(r.get("history") or {})}
        rec[category_col] = r["category"]
        if "sla_hours" in r:
            rec["sla_hours"] = r["sla_hours"]
        recs.append(rec)
    df = pd.DataFrame(recs)
    for c in features:                       # absent history -> missing, as in training
        if c not in df.columns:
            df[c] = np.nan
    df[category_col] = df[category_col].astype("string")
    for c in ("is_weekend", "is_night"):
        df[c] = df[c].astype("boolean")
    return df


class ResolutionPredictor:
    """
    INPUT  {category, created_at, history: {wl_*, sp_*: ...}}  (the caller turns
           latitude/longitude into the sp_* grid counts; raw coordinates are unused)
    OUTPUT {supported, predicted_resolution_hours, interval_p10_hours,
            interval_p90_hours, model_version, reason}
    """

    def __init__(self):
        self.models, self.pre, self.meta, _ = _load("resolution")
        self.features = self.meta["features"]

    def predict(self, rows: list[dict]) -> list[dict]:
        out = [None] * len(rows)
        ok = [i for i, r in enumerate(rows) if r.get("category") in UE_SUPPORTED]
        for i, r in enumerate(rows):
            if i not in ok:
                out[i] = {"supported": False, "predicted_resolution_hours": None,
                          "interval_p10_hours": None, "interval_p90_hours": None,
                          "model_version": self.meta["model_version"],
                          "reason": f"category {r.get('category')!r} has no training data"}
        if ok:
            X = self.pre.transform(_frame([rows[i] for i in ok], self.features, "model_category"))
            pt = np.clip(np.expm1(self.models["point"].predict(X)), 0, None)
            lo = np.clip(np.expm1(self.models["q10"].predict(X)), 0, None)
            hi = np.clip(np.expm1(self.models["q90"].predict(X)), 0, None)
            lo, hi = np.minimum(lo, pt), np.maximum(hi, pt)   # quantile crossing guard
            cal = self.meta.get("interval_calibration", {})
            for j, i in enumerate(ok):
                out[i] = {"supported": True, "predicted_resolution_hours": float(pt[j]),
                          "interval_p10_hours": float(lo[j]), "interval_p90_hours": float(hi[j]),
                          "interval_empirical_coverage": cal.get("empirical_coverage_mean"),
                          "interval_calibrated": cal.get("calibrated"),
                          "model_version": self.meta["model_version"], "reason": None}
        return out


class SLAPredictor:
    """
    INPUT  {category, created_at, sla_hours (optional; default from priority),
            history: {hist_breach_rate_own, wl_*, sp_*: ...}}
           hist_breach_rate_own is the recent breach share AT THIS sla_hours
    OUTPUT {supported, breach_probability, risk_band, threshold, sla_hours,
            model_version, reason}
    """

    def __init__(self):
        self.model, self.pre, self.meta, _ = _load("sla")
        self.features = self.meta["features"]
        self.threshold = float(self.meta["decision_threshold"])
        self.bands = self.meta["risk_bands"]

    def predict(self, rows: list[dict]) -> list[dict]:
        out = [None] * len(rows)
        ok = []
        for i, r in enumerate(rows):
            if r.get("category") not in UE_SUPPORTED:
                out[i] = {"supported": False, "breach_probability": None, "risk_band": None,
                          "threshold": self.threshold, "sla_hours": r.get("sla_hours"),
                          "model_version": self.meta["model_version"],
                          "reason": f"category {r.get('category')!r} has no training data"}
                continue
            sla = r.get("sla_hours") or default_sla_hours(r["category"])
            if "sla_hours" not in self.features and sla != default_sla_hours(r["category"]):
                out[i] = {"supported": False, "breach_probability": None, "risk_band": None,
                          "threshold": self.threshold, "sla_hours": sla,
                          "model_version": self.meta["model_version"],
                          "reason": "model only supports the category's default SLA"}
                continue
            lo, hi = self.meta.get("sla_hours_range", [sla, sla])
            if not lo <= float(sla) <= hi:
                out[i] = {"supported": False, "breach_probability": None, "risk_band": None,
                          "threshold": self.threshold, "sla_hours": sla,
                          "model_version": self.meta["model_version"],
                          "reason": f"sla_hours outside the trained range [{lo}, {hi}]"}
                continue
            ok.append((i, {**r, "sla_hours": float(sla)}))
        if ok:
            X = self.pre.transform(_frame([r for _, r in ok], self.features, "ue_category"))
            prob = self.model.predict_proba(X)[:, 1]
            for (i, r), p_ in zip(ok, prob):
                band = ("HIGH" if p_ >= self.bands["high"] else
                        "MEDIUM" if p_ >= self.bands["medium"] else "LOW")
                out[i] = {"supported": True, "breach_probability": float(p_), "risk_band": band,
                          "threshold": self.threshold, "sla_hours": r["sla_hours"],
                          "model_version": self.meta["model_version"], "reason": None}
        return out


class HotspotPredictor:
    """
    The hotspot v2 method is NOT a learned model. Phase 2B found that the
    gradient-boosted forecaster beats a trained EWMA by ~4% MAE inside Chicago
    but is 37-52% worse when moved to an unseen city, while the EWMA
    re-calibrates itself to any city's own counts. models/hotspot_v2/ therefore
    holds only metadata.json (the method, alpha and its evaluation).

    INPUT  per series (grid cell x category): {category, weekly_counts:
           [oldest .. latest COMPLETE week]}
    OUTPUT {supported, predicted_next_week_count, method, alpha, model_version,
            reason}
    """

    def __init__(self):
        d = repo_root() / "models" / "hotspot_v2"
        self.meta = json.loads((d / "metadata.json").read_text())
        self.alpha = float(self.meta["ewma_alpha"])
        self.min_weeks = int(self.meta["min_history_weeks"])

    def predict(self, rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            counts = r.get("weekly_counts") or []
            base = {"method": "ewma", "alpha": self.alpha,
                    "model_version": self.meta["model_version"]}
            if r.get("category") not in UE_SUPPORTED:
                out.append({**base, "supported": False, "predicted_next_week_count": None,
                            "reason": f"category {r.get('category')!r} has no evaluation data"})
            elif len(counts) < self.min_weeks:
                out.append({**base, "supported": False, "predicted_next_week_count": None,
                            "reason": f"needs >= {self.min_weeks} complete weeks of history"})
            else:
                y = pd.Series(counts, dtype=float).clip(lower=0)
                pred = float(y.ewm(alpha=self.alpha, adjust=False).mean().iloc[-1])
                out.append({**base, "supported": True, "predicted_next_week_count": pred,
                            "reason": None})
        return out


def available(task: str) -> bool:
    return (Path(repo_root()) / "models" / f"{task}_v2" / "metadata.json").exists()
