#!/usr/bin/env python3
"""
Build one clean dataset per ML task, each with a manifest that declares every
column's role.

    data/processed/ml/resolution_ml.parquet     target: resolution_time_hours
    data/processed/ml/sla_ml.parquet            target: sla_breach
    data/processed/ml/hotspot_ml.parquet        target: future_incident_count
    data/processed/ml/duplicate_ml.parquet      target: same_incident
    data/processed/ml/priority_features.parquet target: NONE — policy only

WHY NOT ONE TABLE
-----------------
`urbaneye_ml.parquet` is a useful single view of an incident, but it is the
wrong unit for three of the five tasks and the wrong population for a fourth:

  * hotspot is a city x zone x category x WEEK panel, not an incident;
  * duplicate detection is a PAIR of incidents, not an incident;
  * SLA breach only exists where the publisher ships a due date — 2% of rows,
    one city — so a model trained on the incident table would silently be
    trained on NYC 2010 alone;
  * resolution time is only observed for closed cases, so the population is
    conditioned on an outcome;
  * priority has no target at all, and anything shaped like one here would be
    the policy engine's own output.

Keeping them in one frame invites a consumer to point a model at whatever column
looks like a label. Each task therefore gets its own file, its own row
population, its own split, and a manifest that says which columns are inputs and
which are not. `urbaneye_ml.parquet` is retained as the incident-level view.

EVERY predictor in every manifest is knowable at prediction time. That is the
only rule that cannot be traded away here.

    python scripts/features/build_task_datasets.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.utils.logging_setup import get_logger
from scripts.utils.mapping import CANONICAL
from scripts.utils.paths import ensure_dir, load_feature_config, load_priority_config, p

log = get_logger("features.task_datasets")


# ---------------------------------------------------------------------------
SPLIT_POLICY = {
    "strategy": "chronological",
    "columns": {
        "split": ("per-source chronological — each city cut on its own timeline. Every city "
                  "appears in every fold, so `city` stays evaluable. Use this for a PER-CITY "
                  "model. It is NOT globally chronological: measured on this corpus, 19.5% of "
                  "resolution test rows (every SF and NYC test row) fall before the last "
                  "Chicago training row on one wall clock."),
        "split_global": ("one global chronological cut — every evaluation row is strictly after "
                         "every training row on a single wall clock. Use this for a POOLED "
                         "model. Cost, stated plainly: SF ends May 2018 and NYC ends March 2010, "
                         "so validation and test contain Chicago and nothing else."),
    },
    "default_column": "split",
    "pooled_column": "split_global",
    "rule": ("Pick the column that matches how you model. Pooling cities on `split` means "
             "being scored partly on the past; using `split_global` means you cannot evaluate "
             "SF or NYC at all. There is no option that gives both, and inventing one would "
             "mean manufacturing data."),
}


def zone_key(df: pd.DataFrame) -> pd.Series:
    """
    city-scoped zone identifier.

    `zone_id` means a different administrative unit in every city — ward in
    Chicago, analysis neighbourhood in San Francisco, community board in New
    York. The raw values do not currently collide (50 + 77 + 41 distinct values,
    168 in total), but that is luck rather than design: any encoder that treats
    `zone_id` as one vocabulary is one new source away from silently merging two
    unrelated places. Prefixing with the city makes the scoping explicit.
    """
    return (df["city"].astype("string") + ":" + df["zone_id"].astype("string")).astype("string")


def effective_predictors(df: pd.DataFrame, predictors: list[str]) -> tuple[list[str], dict]:
    """
    Split a declared predictor list into ones that can inform a model and ones
    that cannot, judged on the TRAIN split only.

    A column with a single distinct value carries no information, and leaving it
    in the declared predictor set makes a manifest claim something false. This
    is not "dropping inconvenient data": the column stays in the table, it just
    stops being advertised as an input. The reason is recorded per column.

    The judgement is made on train alone, never on val or test — deciding what to
    feed a model by looking at the evaluation folds is exactly the habit these
    manifests exist to prevent.
    """
    scope = df[df["split"] == "train"] if "split" in df.columns and (df["split"] == "train").any() else df
    usable, demoted = [], {}
    for c in predictors:
        if c not in df.columns:
            demoted[c] = "column not present in this table"
            continue
        n = int(scope[c].nunique(dropna=True))
        if n == 0:
            demoted[c] = "entirely null on the training split"
        elif n == 1:
            demoted[c] = (f"constant on the training split "
                          f"(single value: {scope[c].dropna().iloc[0]!r})")
        else:
            usable.append(c)
    return usable, demoted


# ---------------------------------------------------------------------------
# Every predictor any task may declare has to appear here with a justification
# of WHY it is knowable at prediction time. audit_task_datasets.py fails if a
# manifest declares a predictor that is not in this table, so a new feature
# cannot reach a model without someone writing down when it becomes known.
# ---------------------------------------------------------------------------
PREDICTION_TIME_AVAILABILITY: dict[str, str] = {
    # citizen-supplied or assigned at intake
    "category": "citizen selects it in the app at submission",
    "subcategory": "publisher's second-level label, assigned at intake",
    "city": "known from the coordinate / the deployment",
    "source_dataset": "known: which system the report arrived through",
    "department": ("assigned by the 311 routing rules at intake. If a municipality assigns it "
                   "at closure instead, it must be dropped for that source."),
    "report_channel": "the channel the citizen used, known at submission",
    "zone_key": "administrative unit of the report location, resolved at intake",
    "zone_type": "names which administrative unit zone_key refers to",
    "zone_id": "raw administrative unit; prefer zone_key",
    "has_coordinates": "whether the submission carried a usable coordinate",
    # derived purely from the report's own timestamp
    "hour": "function of reported_at alone", "day_of_week": "function of reported_at alone",
    "month": "function of reported_at alone", "year": "function of reported_at alone",
    "week_of_year": "function of the period being forecast",
    "is_weekend": "function of reported_at alone", "is_night": "function of reported_at alone",
    # strictly backward-looking history
    "nearby_similar_incidents_24h": "counts only reports strictly earlier than this one",
    "nearby_similar_incidents_7d": "counts only reports strictly earlier than this one",
    "nearby_similar_incidents_30d": "counts only reports strictly earlier than this one",
    "local_incident_density": "counts only reports strictly earlier than this one",
    "category_incident_density": "ratio of two strictly-backward counts",
    "hours_since_previous_similar_incident": "gap to the most recent strictly earlier report",
    # SLA
    "sla_target_hours": ("the deadline, set by the publisher at intake. Only legitimate for the "
                         "SLA task, where the question is whether this known deadline will be "
                         "missed."),
    # hotspot panel
    "incident_count": ("the CURRENT week's own count. Legitimate ONLY under the stated "
                       "deployment assumption that the forecast for week t+1 is produced at "
                       "the end of week t, when week t is complete."),
    "previous_period_count": "count in week t-1; shift(1) before any rolling",
    "rolling_4w_count": "weeks t-4..t-1", "rolling_12w_count": "weeks t-12..t-1",
    "rolling_4w_mean": "weeks t-4..t-1", "trend_4w": "difference of two backward windows",
    # duplicate pairs — both incidents are in hand when the comparison is made
    "distance_meters": "computed from the two reports being compared",
    "time_difference_hours": "computed from the two reports being compared",
    "category_match": "computed from the two reports being compared",
}


def availability_block(predictors: list[str]) -> dict:
    """Per-predictor statement of when the value becomes known."""
    out = {}
    for c in predictors:
        why = PREDICTION_TIME_AVAILABILITY.get(c)
        out[c] = {"available_at_prediction_time": why is not None,
                  "why": why or "UNDECLARED — add it to PREDICTION_TIME_AVAILABILITY"}
    return out


SOURCE_SPECIFIC_SEMANTICS = {
    "zone_id / zone_key": ("ward (Chicago), analysis neighbourhood (San Francisco), community "
                           "board (New York). Not comparable units. Always city-scoped."),
    "department": ("owner department (Chicago, 3 values), agency (NYC, 5), responsible agency "
                   "(SF, 113). The same column name, three different vocabularies and three "
                   "different granularities."),
    "report_channel": ("normalised from Chicago `origin`, NYC `open_data_channel_type` and SF "
                       "`source`. The normalisation is lossy and 'other' means different things "
                       "in each city."),
    "subcategory": "100% NULL for Chicago, populated elsewhere — its missingness is a city flag.",
    "sla_target_hours": ("NYC Due Date and Boston TARGET_DT encode each city's own staffing "
                         "policy, not a shared notion of urgency."),
}


def _profile(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        if c not in df.columns:
            out[c] = {"present": False}
            continue
        s = df[c]
        entry = {
            "present": True,
            "dtype": str(s.dtype),
            "null_pct": round(float(s.isna().mean() * 100), 3),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            v = pd.to_numeric(s, errors="coerce").dropna()
            if len(v):
                entry["stats"] = {"min": float(v.min()), "p50": float(v.median()),
                                  "p99": float(v.quantile(0.99)), "max": float(v.max())}
        if entry["n_unique"] <= 1:
            entry["warning"] = "CONSTANT — carries no information"
        elif entry["null_pct"] > 95:
            entry["warning"] = "almost entirely missing"
        elif entry["n_unique"] > 1000:
            entry["warning"] = "high cardinality — needs target-free encoding"
        return_ = out[c] = entry
    return out


def _split_profile(df: pd.DataFrame, ts_col: str, by: list[str],
                   split_col: str = "split") -> dict:
    if split_col not in df.columns:
        return {}
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce") if ts_col in df.columns else None
    prof = {}
    for s, g in df.groupby(split_col, observed=True):
        entry = {"rows": int(len(g))}
        if ts is not None:
            gts = ts.loc[g.index]
            entry["date_range"] = {"min": str(gts.min()), "max": str(gts.max())}
        for b in by:
            if b in g.columns:
                entry[f"by_{b}"] = {str(k): int(v) for k, v in g[b].value_counts().items()}
        prof[str(s)] = entry
    return prof


def write_task(name: str, df: pd.DataFrame, manifest: dict) -> None:
    dest = ensure_dir(p("processed", "ml", f"{name}.parquet"))
    df.to_parquet(dest, index=False, compression="snappy")
    manifest["rows"] = int(len(df))
    manifest["columns"] = int(df.shape[1])
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["path"] = str(dest)
    ensure_dir(p("processed", "ml", f"{name}.manifest.json")).write_text(
        json.dumps(manifest, indent=2, default=str))
    log.info("%-20s %8d rows x %2d cols -> %s", name, len(df), df.shape[1], dest.name)


# ---------------------------------------------------------------------------
# 1. RESOLUTION TIME
# ---------------------------------------------------------------------------
def build_resolution(fc: dict) -> dict:
    src = p("processed", "resolution", "resolution_dataset.parquet")
    if not src.exists():
        log.warning("resolution_dataset missing — run build_resolution_dataset.py")
        return {}
    d = pd.read_parquet(src)
    d["zone_key"] = zone_key(d)

    density = list(fc["historical_density_features"]) + list(fc.get("historical_recency_features", {}))
    predictors = ["category", "subcategory", "city", "source_dataset", "department",
                  "zone_key", "zone_type", "report_channel",
                  "hour", "day_of_week", "month", "year", "is_weekend", "is_night"] + density
    target = "resolution_time_hours"
    keep = ["incident_id", "reported_at"] + predictors + [target, "sla_target_hours",
                                                     "split", "split_global"]
    out = d[[c for c in keep if c in d.columns]].copy()
    out["resolution_time_hours_log1p"] = np.log1p(
        pd.to_numeric(out[target], errors="coerce")).astype("float64")

    t = pd.to_numeric(out[target], errors="coerce")
    predictors, demoted = effective_predictors(out, predictors)
    manifest = {
        "task": "resolution_time",
        "unit": "one incident",
        "target": {
            "column": target, "kind": "regression", "available": True,
            "rows_with_target": int(t.notna().sum()),
            "distribution_hours": {q: float(t.quantile(v)) for q, v in
                                   [("p01", .01), ("p25", .25), ("p50", .5),
                                    ("p75", .75), ("p95", .95), ("p99", .99)]},
            "max_hours": float(t.max()),
            "alternative_column": "resolution_time_hours_log1p",
            "note": ("closed_at - reported_at. The distribution is extremely right-skewed "
                     "(p50 around 4 days, max over 8 years), so fit on log1p and report "
                     "errors in hours, or use a model with an appropriate loss."),
        },
        "population": {
            "rows": int(len(out)),
            "filter": "in-scope categories AND a closed case with a valid duration",
            "censoring": ("RIGHT-CENSORED BY CONSTRUCTION. Cases still open when the extract "
                          "was taken are excluded, which biases the sample toward faster "
                          "resolutions. Anything long-running is systematically under-"
                          "represented. Survival analysis is the correct treatment if the "
                          "tail matters; this table is for the regression framing."),
        },
        "predictors": _profile(out, predictors),
        "prediction_time_availability": availability_block(predictors),
        "source_specific_semantics": SOURCE_SPECIFIC_SEMANTICS,
        "predictors_demoted_no_information": demoted,
        "predictor_notes": {
            "department": ("assigned by the 311 routing rules at intake, so it is known at "
                           "report time — but its cardinality is 3 in Chicago and 113 in San "
                           "Francisco, so it is NOT one vocabulary. Encode per city or drop it "
                           "for a pooled model."),
            "subcategory": ("100% NULL for Chicago and 0% NULL elsewhere, so its missingness "
                            "is a perfect city indicator. In a pooled model it leaks the city "
                            "through the back door; that is not target leakage, but it will "
                            "make importance plots lie."),
            "zone_key": "city-scoped; never pool raw zone_id across cities.",
            "density/recency": ("strictly backward-looking, NULL where the incident has no "
                                "usable coordinate (1.7% of rows). NULL is not zero."),
        },
        "excluded_and_why": {
            "closed_at / status / sla_met / description": "post-resolution; not knowable at report time",
            "latitude / longitude": "raw geometry memorises US locations; use the derived density features",
            "hour_utc": ("dropped: it is determined by (city, local hour) up to DST, so it is a "
                         "city indicator wearing a clock's clothes"),
            "coord_outside_city_bbox": "constant False across all 1.9M rows — no information",
            "priority_*": "policy-engine output, not an observation",
        },
        "split": {**SPLIT_POLICY,
                  "profile": _split_profile(out, "reported_at", ["source_dataset", "split"]),
                  "profile_global": _split_profile(out, "reported_at", ["source_dataset"],
                                                   split_col="split_global")},
        "recommended_preprocessing": [
            "fit every encoder and imputer on TRAIN ONLY",
            "log1p the target; report MAE/median-AE back in hours",
            "ordinal/one-hot for low-cardinality categoricals; hashing or per-city "
            "target encoding fitted on train for zone_key and department",
            "keep NULL as its own category for density/recency rather than imputing 0",
        ],
        "recommended_metrics": ["MAE", "median absolute error", "RMSE on log1p",
                                "per-city and per-category breakdowns (pooled numbers hide "
                                "that the three cities have different operating regimes)"],
        "known_risks": [
            "Chicago's validation window (Mar-Jun 2020) is the first COVID wave: median "
            "resolution time falls from 137h in train to 51h in val and back to 144h in test. "
            "Any model scored across that boundary is being scored on a regime change.",
            "The three cities differ by an order of magnitude in resolution time; a pooled "
            "model mostly learns which city a row is from.",
        ],
    }
    write_task("resolution_ml", out, manifest)
    return manifest


# ---------------------------------------------------------------------------
# 2. SLA BREACH
# ---------------------------------------------------------------------------
def build_sla(fc: dict) -> dict:
    src = p("processed", "resolution", "resolution_dataset.parquet")
    if not src.exists():
        return {}
    d = pd.read_parquet(src)
    d = d[d["sla_breach"].notna()].copy()
    if d.empty:
        log.warning("no rows with an SLA target — skipping sla_ml")
        return {}
    d["zone_key"] = zone_key(d)

    density = list(fc["historical_density_features"]) + list(fc.get("historical_recency_features", {}))
    # sla_target_hours IS a predictor here: the deadline is set at intake and the
    # question is whether this case will miss it. It is NOT a predictor for the
    # resolution-time model, where it would simply be policy noise.
    predictors = ["category", "subcategory", "city", "source_dataset", "department",
                  "zone_key", "zone_type", "report_channel",
                  "hour", "day_of_week", "month", "year", "is_weekend", "is_night",
                  "sla_target_hours"] + density
    keep = ["incident_id", "reported_at"] + predictors + ["sla_breach", "split", "split_global"]
    out = d[[c for c in keep if c in d.columns]].copy()

    y = out["sla_breach"].astype("boolean")
    pos = int(y.fillna(False).sum())
    predictors, demoted = effective_predictors(out, predictors)
    manifest = {
        "task": "sla_breach",
        "unit": "one incident",
        "target": {
            "column": "sla_breach", "kind": "binary_classification", "available": True,
            "rows_with_target": int(y.notna().sum()),
            "positives": pos, "negatives": int(len(y) - pos),
            "positive_rate": round(float(pos / max(1, len(y))), 4),
            "definition": "resolution_time_hours > sla_target_hours",
        },
        "population": {
            "rows": int(len(out)),
            "sources": {str(k): int(v) for k, v in out["source_dataset"].value_counts().items()},
            "date_range": {"min": str(out["reported_at"].min()), "max": str(out["reported_at"].max())},
            "selection_bias": (
                "THREE stacked selections, all of which must be stated with any metric: "
                "(1) only publishers that ship a due date have a target at all — in this "
                "extract that is NYC alone, so the model is a NYC model; (2) NYC's extract "
                "covers Jan-Mar 2010 only, so there is no seasonal or multi-year coverage; "
                "(3) within NYC only some complaint types carry a Due Date, so the "
                "population is a subset of complaint types, not a random sample. Boston "
                "(TARGET_DT) would be the second source and is not downloaded."),
        },
        "predictors": _profile(out, predictors),
        "prediction_time_availability": availability_block(predictors),
        "source_specific_semantics": SOURCE_SPECIFIC_SEMANTICS,
        "predictors_demoted_no_information": demoted,
        "predictor_notes": {
            "sla_target_hours": ("the deadline itself, set at intake, so it is legitimately "
                                 "known at prediction time and is the single most informative "
                                 "input. It is the denominator of the target's definition, not "
                                 "a function of the outcome."),
        },
        "excluded_and_why": {
            "resolution_time_hours": "the outcome the target is computed FROM — including it "
                                     "would make the task trivial and meaningless",
            "sla_met": "the same quantity as the target, inverted",
        },
        "split": {**SPLIT_POLICY,
                  "profile": _split_profile(out, "reported_at", ["source_dataset"]),
                  "profile_global": _split_profile(out, "reported_at", ["source_dataset"],
                                                   split_col="split_global")},
        "recommended_preprocessing": [
            "fit on TRAIN ONLY", "keep the natural class ratio; do not resample before the "
            "metric is chosen", "treat NULL density as its own category",
        ],
        "recommended_metrics": ["PR-AUC (the positive class is the minority and the one that "
                                "matters)", "ROC-AUC", "precision/recall at an operating "
                                "threshold chosen from the cost of a missed breach",
                                "calibration curve / Brier score"],
        "known_risks": [
            "One city, one quarter, 2010. Nothing here generalises to another city's SLA "
            "policy, and nothing here should be presented as a general SLA-risk model.",
            "The whole table is about 2% of closed cases.",
        ],
    }
    write_task("sla_ml", out, manifest)
    return manifest


# ---------------------------------------------------------------------------
# 3. HOTSPOT
# ---------------------------------------------------------------------------
def build_hotspot() -> dict:
    src = p("processed", "hotspot", "hotspot_dataset.parquet")
    if not src.exists():
        return {}
    d = pd.read_parquet(src)
    d["zone_key"] = (d["city"].astype("string") + ":" + d["zone_id"].astype("string")).astype("string")
    d["week_start"] = pd.to_datetime(d["week_start"])
    d["week_of_year"] = d["week_start"].dt.isocalendar().week.astype("Int64")

    predictors = ["city", "zone_key", "zone_type", "category", "month", "year", "week_of_year",
                  "incident_count", "previous_period_count", "rolling_4w_count",
                  "rolling_12w_count", "rolling_4w_mean", "trend_4w"]
    keep = ["city", "zone_type", "zone_id", "category", "week", "week_start"] + predictors + \
           ["future_incident_count", "future_incident_flag", "split", "split_global"]
    out = d[[c for c in dict.fromkeys(keep) if c in d.columns]].copy()

    y = pd.to_numeric(out["future_incident_count"], errors="coerce")
    predictors, demoted = effective_predictors(out, predictors)
    manifest = {
        "task": "hotspot",
        "unit": "city x zone x category x ISO week",
        "target": {
            "column": "future_incident_count", "kind": "count_regression", "available": True,
            "secondary_column": "future_incident_flag", "secondary_kind": "binary",
            "rows_with_target": int(y.notna().sum()),
            "zero_share": round(float((y == 0).mean()), 4),
            "distribution": {q: float(y.quantile(v)) for q, v in
                             [("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95)]},
            "max": float(y.max()),
            "definition": "count in week t+1 for the same zone x category",
        },
        "population": {
            "rows": int(len(out)),
            "panel": ("COMPLETE calendar-week grid per city. Before this was fixed the panel "
                      "held only weeks that had an incident, so shift(-1) meant 'the next week "
                      "that happened to have one': the target could never be zero, "
                      "future_incident_flag was True for 100% of rows, and 6.2% of lag values "
                      "referred to a week that was not t-1 at all."),
            "series": int(out.groupby(["city", "zone_type", "zone_id", "category"],
                                      observed=True).ngroups),
        },
        "predictors": _profile(out, predictors),
        "prediction_time_availability": availability_block(predictors),
        "source_specific_semantics": SOURCE_SPECIFIC_SEMANTICS,
        "predictors_demoted_no_information": demoted,
        "predictor_notes": {
            "incident_count": ("the CURRENT week's own count. This is a legitimate predictor "
                               "only under the stated deployment assumption: the forecast for "
                               "week t+1 is made at the END of week t, when week t is complete. "
                               "Forecasting mid-week requires dropping it."),
            "previous_period_count / rolling_*": ("all shift(1) before rolling, so week t never "
                                                  "sees its own count. Re-derived independently "
                                                  "by validate_leakage LEAK-2."),
            "zone_key": "city-scoped. Chicago wards, SF neighbourhoods and NYC community boards "
                        "are not comparable units; never pool them as one vocabulary.",
        },
        "split": {**SPLIT_POLICY,
                  "strategy": "per-city chronological on week_start",
                  "profile": _split_profile(out, "week_start", ["city"]),
                  "profile_global": _split_profile(out, "week_start", ["city"],
                                                   split_col="split_global")},
        "recommended_preprocessing": [
            "fit on TRAIN ONLY",
            "counts are over-dispersed: use a Poisson/Tweedie objective or model log1p(count)",
            "keep the zeros — they are 20% of the panel and they are the quiet weeks a "
            "hotspot model exists to identify",
        ],
        "recommended_metrics": ["MAE", "RMSE", "Poisson deviance",
                                "compare against the naive baseline future = incident_count, "
                                "which is strong for a weekly panel and is the number any "
                                "model must beat"],
        "known_risks": [
            "NYC contributes 10 weeks and SF 19, so their val/test folds are one or two weeks "
            "wide. Per-city metrics on those two cities are not meaningful.",
            "Zone definitions differ per city; pooling without the city feature is invalid.",
        ],
    }
    write_task("hotspot_ml", out, manifest)
    return manifest


# ---------------------------------------------------------------------------
# 4. DUPLICATE DETECTION
# ---------------------------------------------------------------------------
def build_duplicate() -> dict:
    src = p("processed", "duplicates", "duplicate_pairs.parquet")
    if not src.exists():
        return {}
    d = pd.read_parquet(src)
    predictors = ["distance_meters", "time_difference_hours", "category_match"]
    unavailable = ["text_similarity", "image_similarity"]
    keep = ["incident_a", "incident_b"] + predictors + unavailable + \
           ["same_incident", "negative_strategy", "source_dataset", "split"]
    out = d[[c for c in keep if c in d.columns]].copy()

    y = out["same_incident"].astype("boolean")
    predictors, demoted = effective_predictors(out, predictors)
    per_split = {str(s): {"positive": int(g["same_incident"].fillna(False).sum()),
                          "negative": int((~g["same_incident"].fillna(False)).sum())}
                 for s, g in out.groupby("split", observed=True)}
    manifest = {
        "task": "duplicate_detection",
        "unit": "one ordered pair of incidents",
        "target": {
            "column": "same_incident", "kind": "binary_classification", "available": True,
            "positives": int(y.fillna(False).sum()),
            "negatives": int((~y.fillna(False)).sum()),
            "positive_source": "Chicago 311 PARENT_SR_NUMBER — a municipal duplicate "
                               "determination, the only such labelling in open civic data",
            "negative_source": "CONSTRUCTED by four documented sampling rules",
        },
        "population": {
            "rows": int(len(out)),
            "city": "Chicago only — no other publisher links duplicates",
            "class_balance_per_split": per_split,
            "negative_strategy_mix": {str(k): int(v) for k, v in
                                      out["negative_strategy"].fillna("POSITIVE")
                                      .value_counts().items()},
        },
        "predictors": _profile(out, predictors),
        "prediction_time_availability": availability_block(predictors),
        "predictors_demoted_no_information": demoted,
        "declared_but_unavailable_predictors": {
            "columns": unavailable,
            "reason": ("Chicago publishes neither a free-text description nor photographs, so "
                       "no lexical or visual similarity can be computed for the only pairs that "
                       "carry a label. Emitted NULL, never imputed as 0.0."),
        },
        "split": {"strategy": "connected-component chronological",
                  "column": "split",
                  "detail": ("Incidents are grouped into connected components of the positive-"
                             "pair graph, components are cut chronologically, and negatives are "
                             "sampled INSIDE each split. Verified: 0 incidents appear in more "
                             "than one split. The previous order-of-appearance split put all "
                             "231,955 positives in train and left validation and test 100% "
                             "negative."),
                  "profile": _split_profile(out, None, ["split"])},
        "recommended_preprocessing": ["fit on TRAIN ONLY",
                                      "no imputation of the two unavailable similarity columns"],
        "recommended_metrics": ["PR-AUC", "ROC-AUC", "precision/recall at threshold",
                                "METRICS PER negative_strategy — a pooled number is dominated "
                                "by whichever strategy is most common"],
        "known_risks": [
            "THE NEGATIVES ARE CONSTRUCTED AND THE CONSTRUCTION CORRELATES WITH THE LABEL. "
            "N1 negatives are >2 km apart and N2 negatives are >90 days apart BY DEFINITION, "
            "so distance and time difference separate the classes partly by fiat. Any pooled "
            "PR-AUC here is an upper bound on real performance, not an estimate of it.",
            "The 1:1 class balance is a sampling choice. In production the overwhelming "
            "majority of candidate pairs are not duplicates, so the operating threshold must "
            "be re-derived against the real prevalence.",
            "N2 (same category, >90 days apart) is thin in test because the test window is "
            "only a few months wide — a >90-day gap barely fits inside it.",
        ],
    }
    write_task("duplicate_ml", out, manifest)
    return manifest


# ---------------------------------------------------------------------------
# 5. PRIORITY — features and policy output only, NO target
# ---------------------------------------------------------------------------
def build_priority(fc: dict) -> dict:
    src = p("processed", "priority", "priority_dataset.parquet")
    if not src.exists():
        return {}
    d = pd.read_parquet(src)
    d["zone_key"] = zone_key(d)
    d["in_scope"] = d["category"].isin(CANONICAL).astype("boolean")

    # hour_utc is an audit column, not a feature: it is determined by (city,
    # local hour) up to the DST shift, so a model using it is partly keying on
    # the city. It stays in the table; it is not advertised as an input.
    temporal = [c for c in fc.get("temporal_features", {}) if c != "hour_utc"]
    density = list(fc["historical_density_features"]) + list(fc.get("historical_recency_features", {}))
    poi = []
    for block in ("poi_proximity_features", "road_network_features", "area_context_features"):
        poi += list(fc.get(block, {}))
    features = ["category", "subcategory", "city", "source_dataset", "zone_key", "zone_type",
                "report_channel"] + temporal + density
    policy = ["priority_baseline", "priority_score", "priority_reasons", "priority_confidence",
              "priority_features_available", "priority_feature_coverage", "priority_method",
              "sla_hours_policy"]
    keep = ["incident_id", "reported_at", "in_scope"] + features + poi + policy + \
           ["priority_label", "priority_label_source", "is_ground_truth",
            "target_status", "target_strategy", "split", "split_global"]
    out = d[[c for c in dict.fromkeys(keep) if c in d.columns]].copy()

    features, demoted = effective_predictors(out, features)
    strata = out.groupby("priority_features_available")["priority_score"].agg(
        ["count", "min", "median", "max"])
    manifest = {
        "task": "priority",
        "unit": "one incident",
        "target": {
            "column": None, "kind": None, "available": False,
            "status": "NO_PRIORITY_GROUND_TRUTH_AVAILABLE",
            "rows_with_target": 0,
            "why": ("No public 311 dataset records an operational priority, urgency, severity "
                    "or risk grade. Resolution time is an outcome shaped by departmental "
                    "capacity, not a statement of urgency, and a due date encodes one city's "
                    "staffing policy. There is nothing here to supervise against."),
            "what_would_unblock_it": ("operator-assigned priorities from UrbanEye+'s own "
                                      "console, plus the override flag — see "
                                      "PRIORITY_METHODOLOGY.md section 6"),
        },
        "framing": {
            "today": "deterministic policy score and RANKING, config-driven and explainable",
            "valid_uses": [
                "rank a queue of open incidents that share the same feature availability",
                "serve as the benchmark a future learned model must beat",
                "unsupervised analysis: drift in the score distribution, disagreement "
                "between the score and observed resolution behaviour",
            ],
            "invalid_uses": [
                "training a supervised classifier on priority_baseline — it is a deterministic "
                "function of config/priority_config.yaml, so the model would learn the YAML "
                "and any reported accuracy would be circular",
                "presenting priority_score as a probability or a calibrated risk",
                "comparing scores across rows with different feature availability (see below)",
            ],
        },
        "ranking_comparability": {
            "by_priority_features_available": {
                str(k): {"rows": int(v["count"]), "min": float(v["min"]),
                         "median": float(v["median"]), "max": float(v["max"])}
                for k, v in strata.iterrows()},
            "warning": ("The score is normalised over the weights of the features that were "
                        "actually available (null_handling: strict_available_only). Two rows "
                        "scored on different input sets are therefore NOT on the same scale: "
                        "the 29,606 rows with no usable coordinate are scored on the category "
                        "term alone. Rank within a stratum, or surface "
                        "priority_feature_coverage next to the score."),
        },
        "population": {
            "rows": int(len(out)),
            "in_scope_rows": int(out["in_scope"].fillna(False).sum()),
            "out_of_scope_rows": int((~out["in_scope"].fillna(False)).sum()),
            "note": ("UNMAPPED / OUT_OF_SCOPE / REVIEW_REQUIRED rows are scored with the "
                     "OTHER base score and retained for auditability, but they are not "
                     "UrbanEye incidents. Filter on in_scope before using this table for "
                     "anything operational."),
        },
        "features": _profile(out, features),
        "prediction_time_availability": availability_block(features),
        "source_specific_semantics": SOURCE_SPECIFIC_SEMANTICS,
        "features_demoted_no_information": demoted,
        "declared_but_unavailable_features": {
            "columns": poi,
            "reason": "no external geospatial reference layer is connected; see GEOSPATIAL_FEATURES.md",
        },
        "policy_columns": {
            "columns": policy,
            "role": "POLICY METADATA — outputs of the rule engine. Never a predictor, never a label.",
            "config": load_priority_config()["config_version"],
            "config_status": load_priority_config()["status"],
        },
        "split": {**SPLIT_POLICY,
                  "note": "provided for consistency; there is no supervised task to split for",
                  "profile": _split_profile(out, "reported_at", ["source_dataset"])},
        "recommended_metrics": [
            "NONE of the supervised kind. Monitor the score distribution over time, the "
            "share of rows in each confidence tier, and agreement between the ranking and "
            "whatever operators actually do once that is logged.",
        ],
    }
    write_task("priority_features", out, manifest)
    return manifest


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    choices=["resolution", "sla", "hotspot", "duplicate", "priority"])
    args = ap.parse_args()
    fc = load_feature_config()

    builders = {"resolution": lambda: build_resolution(fc), "sla": lambda: build_sla(fc),
                "hotspot": build_hotspot, "duplicate": build_duplicate,
                "priority": lambda: build_priority(fc)}
    index = {}
    for name, fn in builders.items():
        if args.only and name != args.only:
            continue
        m = fn()
        if m:
            index[name] = {
                "path": m.get("path"), "rows": m.get("rows"), "unit": m.get("unit"),
                "target": m["target"].get("column"),
                "target_available": m["target"].get("available"),
                "split_strategy": m["split"]["strategy"],
            }
    if index:
        ensure_dir(p("reports", "task_datasets_index.json")).write_text(
            json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "tasks": index}, indent=2, default=str))
        log.info("task index -> %s", p("reports", "task_datasets_index.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
