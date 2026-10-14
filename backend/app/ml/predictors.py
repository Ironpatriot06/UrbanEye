"""
Resolution-time and SLA-breach predictions with explicit, documented fallbacks.

Decision procedure (both tasks)
-------------------------------
1. Try the ML model.  It needs: a loaded, verified artifact; a category the
   model was trained on (POTHOLE, FLOOD, GARBAGE, STREETLIGHT); for SLA, an
   SLA window inside the trained range [4, 168] h.  Missing history does NOT
   stop inference — the model was trained with missing values and handles them.
2. If the model ran and produced a finite, in-range output:
     AVAILABLE     every history feature present and inside the training p01–p99 range
     EXPERIMENTAL  otherwise; `reasons` says which inputs (INSUFFICIENT_LOCAL_HISTORY,
                   FEATURES_OUTSIDE_TRAINING_RANGE, OUTSIDE_SERVICE_AREA)
3. Otherwise use the first applicable baseline, status FALLBACK, with the reason
   the model was not used in `fallback_reason`:
     resolution  local category median (>= 10 local closures)
                 -> local all-category median (>= 10)
                 -> training-data category median (US 311; trained categories only)
     SLA         local recent breach rate at this SLA (>= 20 incidents, Phase 2B's
                 strongest baseline) -> local category breach rate (>= 20, any age)
                 -> training-data category breach rate (US 311; 4/24/72/168 h only)
   A baseline never gets a risk band or a range: those are properties of the model.
4. Otherwise UNAVAILABLE.  An incident with no SLA has nothing to breach, so its
   SLA result is UNAVAILABLE (NO_SLA) without trying a baseline.

Nothing here claims local reliability: `local_reliability_established` is
always False until predictions have been evaluated against UrbanEye outcomes.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.ml.areas import ServiceArea
from app.ml.features import (HISTORY_FEATURES, IncidentRecord, calendar_features,
                             local_breach_outcomes, local_closed_durations)
from app.ml.registry import Registry

# The boosters were fitted on pandas frames and warn when given a numpy array.
# Column order is what matters, and registry.py refuses any model whose
# feature_names_in_ differ from the encoder's predictor order, so this one
# message carries no information here.
warnings.filterwarnings("ignore", message="X does not have valid feature names",
                        category=UserWarning)

SUPPORTED = ("POTHOLE", "FLOOD", "GARBAGE", "STREETLIGHT")
MIN_LOCAL_DURATIONS = 10
MIN_LOCAL_OUTCOMES = 20

AVAILABLE, EXPERIMENTAL, FALLBACK, UNAVAILABLE = "AVAILABLE", "EXPERIMENTAL", "FALLBACK", "UNAVAILABLE"
ML_MODEL = "ML_MODEL"

REASON_TEXT = {
    "ML_DISABLED": "Machine-learning predictions are switched off on this server.",
    "ARTIFACT_UNAVAILABLE": "The model files are missing or failed verification.",
    "UNSUPPORTED_CATEGORY": "The model has no training data for this category.",
    "NO_SLA": "This incident has no SLA window, so there is no deadline to breach.",
    "SLA_OUT_OF_RANGE": "The model was trained for SLA windows of 4 to 168 hours only.",
    "INVALID_MODEL_OUTPUT": "The model returned a value that is not a valid prediction.",
    "INSUFFICIENT_LOCAL_HISTORY": "Some history inputs could not be computed from local data.",
    "FEATURES_OUTSIDE_TRAINING_RANGE": "Some inputs are outside the range seen in training.",
    "OUTSIDE_SERVICE_AREA": "The incident is outside every configured service area.",
    "NO_APPLICABLE_BASELINE": "There is not enough local or training data for a baseline.",
    "CATEGORY_NOT_EVALUATED": "This category was not part of the offline evaluation.",
}
METHOD_TEXT = {
    ML_MODEL: "ML model",
    "BASELINE_LOCAL_CATEGORY_MEDIAN": "median resolution time of earlier UrbanEye incidents in this category",
    "BASELINE_LOCAL_GLOBAL_MEDIAN": "median resolution time of earlier UrbanEye incidents (all categories)",
    "BASELINE_TRAINING_CATEGORY_MEDIAN": "median resolution time of this category in the US training data",
    "BASELINE_LOCAL_RECENT_BREACH_RATE": "share of recent UrbanEye incidents in this category that missed this SLA",
    "BASELINE_LOCAL_CATEGORY_BREACH_RATE": "share of earlier UrbanEye incidents in this category that missed this SLA",
    "BASELINE_TRAINING_CATEGORY_BREACH_RATE": "share of this category that missed this SLA in the US training data",
}
GENERAL_WARNING = ("Experimental: this model was trained on 2010–2020 US 311 data (Chicago, San "
                   "Francisco, New York City). Its accuracy for UrbanEye incidents has not been "
                   "measured.")
RANGE_WARNING = ("The p10–p90 range is not a calibrated 80% interval: in offline tests it "
                 "contained about {cov:.0%} of outcomes.")


@dataclass
class Context:
    """Everything a prediction needs, already read from the database."""
    query: IncidentRecord
    category: str
    sla_hours: Optional[float]
    area: Optional[ServiceArea]
    features: dict                    # history_features() output
    baseline_records: list            # area records, or every incident if outside an area


def _reason(code: str, detail=None) -> dict:
    r = {"code": code, "message": REASON_TEXT.get(code, code)}
    if detail is not None:
        r["detail"] = detail
    return r


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _regime_reasons(ctx: Context, reg: Registry, task: str, inputs: dict) -> list[dict]:
    reasons = []
    if ctx.area is None:
        reasons.append(_reason("OUTSIDE_SERVICE_AREA"))
    missing = sorted(k for k, v in inputs.items() if _missing(v))
    if missing:
        reasons.append(_reason("INSUFFICIENT_LOCAL_HISTORY", {"missing_features": missing}))
    ranges = (reg.feature_ranges or {}).get(task, {})
    outside = {}
    for k, v in inputs.items():
        rg = ranges.get(k)
        if rg and not _missing(v) and not (rg["p01"] <= v <= rg["p99"]):
            outside[k] = {"value": v, "training_p01": rg["p01"], "training_p99": rg["p99"]}
    if outside:
        reasons.append(_reason("FEATURES_OUTSIDE_TRAINING_RANGE", outside))
    return reasons


def empty_result(task: str, bundle_version: Optional[str]) -> dict:
    """The result shape shared by every outcome, all values unset (UNAVAILABLE)."""
    out = {"task": task, "status": UNAVAILABLE, "method": None, "method_description": None,
           "reasons": [], "fallback_reason": None, "model_version": None,
           "bundle_version": bundle_version, "local_reliability_established": False,
           "warnings": []}
    if task == "resolution":
        out.update(estimate_hours=None, range_p10_hours=None, range_p90_hours=None,
                   range_nominal_coverage=None, range_empirical_coverage=None,
                   range_calibrated=None, baseline_sample_size=None)
    else:
        out.update(sla_hours=None, breach_probability=None, risk_band=None,
                   decision_threshold=None, historical_breach_rate=None,
                   baseline_sample_size=None)
    return out


def _model_blocker(reg: Registry, task: str, category: str) -> Optional[str]:
    if not reg.enabled:
        return "ML_DISABLED"
    if not reg.ready(task):
        return "ARTIFACT_UNAVAILABLE"
    if category not in SUPPORTED:
        return "UNSUPPORTED_CATEGORY"
    return None


def _history_inputs(ctx: Context) -> dict:
    return {k: ctx.features.get(k, float("nan")) for k in HISTORY_FEATURES}


def _finish(out: dict, method: str, status: str) -> dict:
    out["method"], out["status"] = method, status
    out["method_description"] = METHOD_TEXT.get(method)
    if status == FALLBACK:
        out["warnings"].append(f"Not an ML prediction: based on the {METHOD_TEXT[method]}.")
        if "TRAINING" in method:
            out["warnings"].append("US training data, not UrbanEye data.")
    else:
        out["warnings"].append(GENERAL_WARNING)
        if status == EXPERIMENTAL:
            out["warnings"].append("Local history is missing or outside the training range, "
                                   "so this output is less reliable than offline tests suggest.")
    return out


# ---------------------------------------------------------------------------
# resolution time
# ---------------------------------------------------------------------------
def predict_resolution(reg: Registry, ctx: Context) -> dict:
    out = empty_result("resolution", reg.bundle_version)
    blocker = _model_blocker(reg, "resolution", ctx.category)
    if blocker is None:
        hist = _history_inputs(ctx)
        row = {"model_category": ctx.category, **calendar_features(ctx.query.created_at), **hist}
        X = reg.resolution_encoder.transform([row])
        raw = {k: float(m.predict(X)[0]) for k, m in reg.resolution_models.items()}
        if all(math.isfinite(v) for v in raw.values()):
            pt, lo, hi = (max(float(np.expm1(raw[k])), 0.0) for k in ("point", "q10", "q90"))
            if all(math.isfinite(v) for v in (pt, lo, hi)):
                cal = reg.resolution_meta.get("interval_calibration") or {}
                out.update(estimate_hours=pt, range_p10_hours=min(lo, pt),
                           range_p90_hours=max(hi, pt),
                           range_nominal_coverage=cal.get("nominal_coverage"),
                           range_empirical_coverage=cal.get("empirical_coverage_mean"),
                           range_calibrated=bool(cal.get("calibrated")),
                           model_version=reg.resolution_meta.get("model_version"))
                out["reasons"] = _regime_reasons(ctx, reg, "resolution", hist)
                if not out["range_calibrated"]:
                    out["warnings"].append(RANGE_WARNING.format(
                        cov=out["range_empirical_coverage"] or 0.0))
                return _finish(out, ML_MODEL, EXPERIMENTAL if out["reasons"] else AVAILABLE)
        blocker = "INVALID_MODEL_OUTPUT"

    out["fallback_reason"] = blocker
    out["reasons"] = [_reason(blocker)]
    if blocker == "ML_DISABLED":
        return out
    for method, cat in (("BASELINE_LOCAL_CATEGORY_MEDIAN", ctx.category),
                        ("BASELINE_LOCAL_GLOBAL_MEDIAN", None)):
        d = local_closed_durations(ctx.query, ctx.baseline_records, cat)
        if len(d) >= MIN_LOCAL_DURATIONS:
            out.update(estimate_hours=float(np.median(d)), baseline_sample_size=int(len(d)))
            return _finish(out, method, FALLBACK)
    tb = ((reg.baselines or {}).get("resolution_category_median_hours") or {}).get(ctx.category)
    if tb and reg.ready("baselines"):
        out.update(estimate_hours=float(tb["median_hours"]), baseline_sample_size=int(tb["n"]))
        return _finish(out, "BASELINE_TRAINING_CATEGORY_MEDIAN", FALLBACK)
    out["reasons"].append(_reason("NO_APPLICABLE_BASELINE"))
    return out


# ---------------------------------------------------------------------------
# SLA breach
# ---------------------------------------------------------------------------
def _band(p: float, bands: dict) -> str:
    return "HIGH" if p >= bands["high"] else "MEDIUM" if p >= bands["medium"] else "LOW"


def predict_sla(reg: Registry, ctx: Context) -> dict:
    out = empty_result("sla", reg.bundle_version)
    out["sla_hours"] = ctx.sla_hours
    if ctx.sla_hours is None:
        out["reasons"] = [_reason("NO_SLA")]
        return out
    x = float(ctx.sla_hours)
    blocker = _model_blocker(reg, "sla", ctx.category)
    if blocker is None:
        lo, hi = reg.sla_meta.get("sla_hours_range", [4, 168])
        if not lo <= x <= hi:
            blocker = "SLA_OUT_OF_RANGE"
    if blocker is None:
        hist = _history_inputs(ctx)
        rate, _n = ctx.features.get("hist_breach_rate", {}).get(x, (float("nan"), 0))
        hist["hist_breach_rate_own"] = rate
        row = {"ue_category": ctx.category, "sla_hours": x,
               **calendar_features(ctx.query.created_at), **hist}
        X = reg.sla_encoder.transform([row])
        p = float(reg.sla_model.predict_proba(X)[0, 1])
        if math.isfinite(p) and 0.0 <= p <= 1.0:
            bands = reg.sla_meta["risk_bands"]
            out.update(breach_probability=p, risk_band=_band(p, bands),
                       decision_threshold=float(reg.sla_meta["decision_threshold"]),
                       model_version=reg.sla_meta.get("model_version"))
            out["reasons"] = _regime_reasons(ctx, reg, "sla", hist)
            if x not in (4.0, 24.0, 72.0, 168.0):
                out["reasons"].append(_reason(
                    "FEATURES_OUTSIDE_TRAINING_RANGE",
                    {"sla_hours": {"value": x, "trained_values": [4, 24, 72, 168]}}))
            return _finish(out, ML_MODEL, EXPERIMENTAL if out["reasons"] else AVAILABLE)
        blocker = "INVALID_MODEL_OUTPUT"

    out["fallback_reason"] = blocker
    out["reasons"] = [_reason(blocker)]
    if blocker == "ML_DISABLED":
        return out
    for method, window in (("BASELINE_LOCAL_RECENT_BREACH_RATE", 30.0),
                           ("BASELINE_LOCAL_CATEGORY_BREACH_RATE", None)):
        b = local_breach_outcomes(ctx.query, ctx.baseline_records, ctx.category, x, window)
        if len(b) >= MIN_LOCAL_OUTCOMES:
            out.update(historical_breach_rate=float(b.mean()), baseline_sample_size=int(len(b)))
            return _finish(out, method, FALLBACK)
    tb = (((reg.baselines or {}).get("sla_category_breach_rate") or {})
          .get(ctx.category) or {}).get(str(int(x))) if x.is_integer() else None
    if tb and reg.ready("baselines"):
        out.update(historical_breach_rate=float(tb["breach_rate"]),
                   baseline_sample_size=int(tb["n"]))
        return _finish(out, "BASELINE_TRAINING_CATEGORY_BREACH_RATE", FALLBACK)
    out["reasons"].append(_reason("NO_APPLICABLE_BASELINE"))
    return out
