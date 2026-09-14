#!/usr/bin/env python3
"""
Build hotspot_dataset — spatial-temporal incident risk.

Aggregation unit: city x zone x category x ISO week.

FORMULATION AND WHY
-------------------
The target is `future_incident_count`: the count in week t+1 for that
zone/category, produced by shift(-1). The features describe weeks <= t only.
A binary `future_incident_flag` is also emitted for a classification framing.

Choosing next-week count over "risk score" keeps the target observable and the
evaluation honest — you can score it with MAE against reality. A "risk" target
would need a definition we would have to invent.

LEAKAGE
-------
Every lag and rolling feature is shift(1) before rolling, so the feature row for
week t never contains week t's own count. The chronological split then holds out
whole later periods. Both are asserted by validate_leakage.py.

ZONES ARE NOT TRANSFERABLE
--------------------------
zone_id here is a US ward / community area / analysis neighbourhood / community
board. These are NOT Indian deployment zones. This dataset teaches temporal
dynamics (how incident counts evolve given their own history), not geography.
Include `city` as a feature or model per city; never pool zones across cities as
if they were comparable units.

    python scripts/preprocess/build_hotspot_dataset.py
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from scripts.preprocess._ml_common import chronological_split, in_scope, load_incidents, write
from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import load_config

log = get_logger("preprocess.hotspot")


def _complete_panel(observed: pd.DataFrame, trim_partial: bool) -> tuple[pd.DataFrame, str]:
    """
    Reindex the weekly panel onto a COMPLETE calendar-week grid.

    Why this is not optional. Aggregating incidents by week produces rows only
    for weeks in which something was reported. `shift(1)` on that frame is then
    "the previous week that happened to have an incident", which may be five
    weeks earlier, and `shift(-1)` — the target — is "the next week that had an
    incident", which can never be zero. Measured on the observed-only panel:
    6.2% of rows had a non-adjacent neighbour, `future_incident_count` had a
    minimum of 1, and `future_incident_flag` was True for 100% of rows. The
    classification target was constant and the regression target could not
    express a quiet week.

    A week with no report genuinely has a count of zero, so the zeros are
    observed facts, not imputation: each city's extract is contiguous in the
    publisher's own id order, so within its date window it is complete for the
    complaint types it covers.

    The grid spans each CITY's observed weeks (so every series in a city shares
    one week index) and covers only zone x category series that occur at least
    once in that city — inventing cells for combinations that never occur would
    fill the panel with structural zeros that say nothing.

    The first and last week of each city are dropped when `trim_partial` is set:
    the extract starts and ends mid-week, so those two counts are truncated and
    would read as a spurious dip.
    """
    frames, notes = [], []
    for city, g in observed.groupby("city", observed=True):
        weeks = pd.PeriodIndex(sorted(g["week"].unique()), freq="W")
        full = pd.period_range(weeks.min(), weeks.max(), freq="W")
        if trim_partial and len(full) > 2:
            full = full[1:-1]
        series = g[["city", "zone_type", "zone_id", "category"]].drop_duplicates()
        grid = series.merge(pd.DataFrame({"week": full}), how="cross")
        filled = grid.merge(g, on=["city", "zone_type", "zone_id", "category", "week"], how="left")
        filled["incident_count"] = filled["incident_count"].fillna(0).astype("int64")
        frames.append(filled)
        notes.append(f"{city}: {len(series)} series x {len(full)} weeks "
                     f"({full.min()} .. {full.max()})")
    out = pd.concat(frames, ignore_index=True).sort_values(
        ["city", "zone_type", "zone_id", "category", "week"])
    note = ("COMPLETE calendar-week grid per city; weeks with no report are real zeros. "
            + "; ".join(notes))
    return out, note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incidents", default=None)
    args = ap.parse_args()

    df = in_scope(load_incidents(args.incidents))
    df = df[df["zone_id"].notna() & df["reported_at"].notna()].copy()
    if df.empty:
        log.error("no rows with both a zone and a timestamp")
        return 1

    ts = pd.to_datetime(df["reported_at"], utc=True, errors="coerce")
    # tz-naive before to_period: pandas drops tz on PeriodArray and warns otherwise
    df["week"] = ts.dt.tz_localize(None).dt.to_period("W")

    observed = (df.groupby(["city", "zone_type", "zone_id", "category", "week"], observed=True)
                  .size().reset_index(name="incident_count"))

    cfg = load_config().get("hotspot", {}) or {}
    complete_grid = str(cfg.get("panel", "complete_grid")) == "complete_grid"
    trim_partial = bool(cfg.get("drop_partial_boundary_weeks", True))

    if complete_grid:
        agg, panel_note = _complete_panel(observed, trim_partial)
    else:
        agg = observed.sort_values(["city", "zone_type", "zone_id", "category", "week"])
        panel_note = ("OBSERVED-ONLY panel: rows exist only for weeks that had at least one "
                      "incident, so shift(1)/shift(-1) are the previous/next OBSERVED week, "
                      "not the previous/next calendar week.")
    agg = agg.reset_index(drop=True)

    g = agg.groupby(["city", "zone_type", "zone_id", "category"], observed=True)["incident_count"]
    # shift(1) FIRST, then roll: week t never sees its own count
    agg["previous_period_count"] = g.shift(1)
    agg["rolling_4w_count"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=1).sum())
    agg["rolling_12w_count"] = g.transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum())
    agg["rolling_4w_mean"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    agg["trend_4w"] = agg["previous_period_count"] - agg["rolling_4w_mean"]

    # Target: next period
    agg["future_incident_count"] = g.shift(-1)
    agg["future_incident_flag"] = (agg["future_incident_count"] > 0).astype("boolean")

    agg["week_start"] = agg["week"].dt.start_time
    agg["week"] = agg["week"].astype("string")
    agg["month"] = agg["week_start"].dt.month.astype("Int64")
    agg["year"] = agg["week_start"].dt.year.astype("Int64")

    # The final week of each series has no observable next period. Drop those
    # rows BEFORE splitting, not after: the last week is exactly what a
    # chronological cut assigns to test, so dropping it afterwards emptied the
    # test split of any city whose panel was only a few weeks long (measured:
    # NYC got 5,061 train / 723 val / 0 test).
    unobservable = int(agg["future_incident_count"].isna().sum())
    agg = agg[agg["future_incident_count"].notna()].copy()

    # Same corpus-aware chronological rule as every other table, on week_start
    # and grouped by city (a weekly panel row has no source_dataset column).
    split, split_meta = chronological_split(agg, ts_col="week_start", source_col="city")
    agg["split"] = split

    write(agg, "hotspot", "hotspot_dataset", {
        "model": "Hotspot / incident risk",
        "unit": "city x zone_type x zone_id x category x ISO week",
        "targets": {"regression": "future_incident_count (t+1)",
                    "classification": "future_incident_flag"},
        "formulation_rationale": (
            "Next-period count is directly observable, so the model can be scored against "
            "reality with MAE. A 'risk score' target would require inventing a definition."),
        "panel": panel_note,
        "target_zero_share": float((agg["future_incident_count"] == 0).mean()),
        "leakage_guards": [
            "All lag/rolling features use shift(1) before rolling — week t never sees itself.",
            "The panel is a complete calendar-week grid, so shift(1) is genuinely week t-1 "
            "and shift(-1) is genuinely week t+1. On an observed-only panel both silently "
            "skipped quiet weeks and the target could never be zero.",
            "Chronological split by week_start.",
            "Rows with no observable t+1 are dropped, not imputed.",
        ],
        "rows_dropped_no_future_period": unobservable,
        "split": split_meta,
        "zone_warning": (
            "zone_id is a US administrative unit (ward / community area / analysis "
            "neighbourhood / community board). NOT an Indian deployment zone. Model per "
            "city or include city as a feature; never pool zones across cities."),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
