#!/usr/bin/env python3
"""
Independent data-leakage validation.

The builders claim to be leak-free. This script does not take their word for it:
where it can, it RE-DERIVES the value from first principles and compares.

Checks
------
LEAK-1  No ML-ready feature table contains a forbidden post-resolution column
        (closed_at, status, sla_met, description, resolution_description), other
        than as a declared target.
LEAK-2  Hotspot lag features are strictly backward. Re-computed independently:
        previous_period_count for week t must equal the observed count at t-1,
        and must never equal the count at t when they differ.
LEAK-3  Hotspot target is genuinely the NEXT period, not the current one.
LEAK-4  Historical density features are strictly backward. Rows are resampled and
        the neighbour count is recomputed from the incidents table with an
        explicit reported_at < t filter.
LEAK-5  Chronological splits do not overlap in time (max(train) < min(val) etc.).
LEAK-6  Duplicate groups do not span splits.
LEAK-7  resolution_dataset does not contain sla_met (derived from the target).
LEAK-8  priority_dataset contains no post-resolution column.

Exit code 1 under --strict if any check fails.

    python scripts/validation/validate_leakage.py
    python scripts/validation/validate_leakage.py --strict
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from scripts.utils.cleaning import haversine_m
from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import ensure_dir, load_config, load_feature_config, p

log = get_logger("validation.leakage")
CHECKS: list[dict] = []


def check(cid: str, desc: str, passed: bool, detail: str = "") -> bool:
    CHECKS.append({"id": cid, "description": desc,
                   "result": "PASS" if passed else "FAIL", "detail": detail})
    (log.info if passed else log.error)("%-8s %-4s %s %s", cid,
                                        "PASS" if passed else "FAIL", desc, detail)
    return passed


def skip(cid: str, desc: str, why: str) -> None:
    CHECKS.append({"id": cid, "description": desc, "result": "SKIP", "detail": why})
    log.warning("%-8s SKIP %s (%s)", cid, desc, why)


FORBIDDEN = ["closed_at", "status", "sla_met", "description", "resolution_description",
             "closure_reason", "status_notes"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--density-sample", type=int, default=150)
    args = ap.parse_args()

    fc = load_feature_config()

    # ---- LEAK-1 / 7 / 8 : forbidden columns --------------------------------
    tables = {
        "priority_dataset": p("processed", "priority", "priority_dataset.parquet"),
        "resolution_dataset": p("processed", "resolution", "resolution_dataset.parquet"),
        "hotspot_dataset": p("processed", "hotspot", "hotspot_dataset.parquet"),
    }
    for name, fp in tables.items():
        if not fp.exists():
            skip(f"LEAK-1/{name}", "no forbidden post-resolution columns", "table not built")
            continue
        cols = set(pd.read_parquet(fp).columns)
        # resolution_time_hours is the declared TARGET of resolution_dataset
        allowed_target = {"resolution_time_hours", "sla_breach"} if name == "resolution_dataset" else set()
        bad = sorted((cols & set(FORBIDDEN)) - allowed_target)
        check(f"LEAK-1/{name}", "no forbidden post-resolution columns", not bad,
              f"found={bad}" if bad else "")

    if tables["resolution_dataset"].exists():
        cols = set(pd.read_parquet(tables["resolution_dataset"]).columns)
        check("LEAK-7", "resolution_dataset excludes sla_met (derived from the target)",
              "sla_met" not in cols)

    # ---- LEAK-2 / 3 : hotspot lag + target ---------------------------------
    hp = tables["hotspot_dataset"]
    if hp.exists():
        h = pd.read_parquet(hp).copy()
        h["_ws"] = pd.to_datetime(h["week_start"])
        h = h.sort_values(["city", "zone_type", "zone_id", "category", "_ws"])
        g = h.groupby(["city", "zone_type", "zone_id", "category"], observed=True)
        recomputed_prev = g["incident_count"].shift(1)
        comparable = recomputed_prev.notna() & h["previous_period_count"].notna()
        mismatch = int((recomputed_prev[comparable] != h["previous_period_count"][comparable]).sum())
        check("LEAK-2", "hotspot previous_period_count equals the independently re-derived t-1 count",
              mismatch == 0, f"mismatches={mismatch} of {int(comparable.sum())} comparable")

        # It must NOT equal the current week's own count wherever they differ.
        self_leak = int((h["previous_period_count"].notna()
                         & (h["previous_period_count"] == h["incident_count"])
                         & (recomputed_prev != h["incident_count"])).sum())
        check("LEAK-2b", "previous_period_count never equals the current week where t-1 differs",
              self_leak == 0, f"suspicious_rows={self_leak}")

        recomputed_next = g["incident_count"].shift(-1)
        cmp2 = recomputed_next.notna() & h["future_incident_count"].notna()
        bad_target = int((recomputed_next[cmp2] != h["future_incident_count"][cmp2]).sum())
        check("LEAK-3", "hotspot target is the next period, independently re-derived",
              bad_target == 0, f"mismatches={bad_target}")
    else:
        skip("LEAK-2", "hotspot lag features strictly backward", "hotspot table not built")
        skip("LEAK-3", "hotspot target is next period", "hotspot table not built")

    # ---- LEAK-4 : historical density strictly backward ---------------------
    gp = p("processed", "geo_features", "incident_geo_features.parquet")
    ip = p("processed", "incidents", "all_incidents.parquet")
    if gp.exists() and ip.exists():
        geo = pd.read_parquet(gp)
        inc = pd.read_parquet(ip)
        spec = fc["historical_density_features"]["nearby_similar_incidents_30d"]
        radius, window = float(spec["radius_m"]), float(spec["window_hours"])

        base = inc[["incident_id", "latitude", "longitude", "category", "reported_at"]].dropna(
            subset=["latitude", "longitude", "reported_at"]).copy()
        base["_ts"] = pd.to_datetime(base["reported_at"], utc=True)
        merged = base.merge(geo[["incident_id", "nearby_similar_incidents_30d"]],
                            on="incident_id", how="inner")
        merged = merged[merged["nearby_similar_incidents_30d"].notna()]

        if merged.empty:
            skip("LEAK-4", "density features strictly backward", "no comparable rows")
        else:
            sample = merged.sample(min(args.density_sample, len(merged)), random_state=7)
            lat_all = base["latitude"].to_numpy(float)
            lon_all = base["longitude"].to_numpy(float)
            ts_all = base["_ts"].to_numpy("datetime64[ns]").astype("int64") / 1e9 / 3600.0
            cat_all = base["category"].astype("string").fillna("").to_numpy()

            mism, checked = 0, 0
            for _, r in sample.iterrows():
                t = pd.Timestamp(r["_ts"]).value / 1e9 / 3600.0
                d = haversine_m(float(r["latitude"]), float(r["longitude"]), lat_all, lon_all)
                # the definition under test: STRICTLY earlier only
                m = (ts_all < t) & (ts_all >= t - window) & (d <= radius) & \
                    (cat_all == str(r["category"]))
                expected = int(m.sum())
                if expected != int(r["nearby_similar_incidents_30d"]):
                    mism += 1
                checked += 1
            check("LEAK-4", "nearby_similar_incidents_30d re-derives exactly with reported_at < t",
                  mism == 0, f"mismatches={mism} of {checked} resampled rows")
    else:
        skip("LEAK-4", "density features strictly backward", "geo_features or incidents missing")

    # ---- LEAK-5 : chronological split boundaries ---------------------------
    # splits.tabular.per_source splits each city on its own timeline, because the
    # three cities in this corpus occupy disjoint date ranges and one global cut
    # would hand validation and test to Chicago alone. The ordering guarantee is
    # therefore WITHIN a source; asserting it globally would assert something the
    # configuration deliberately does not claim.
    sp_cfg = load_config()["splits"]["tabular"]
    per_source = bool(sp_cfg.get("per_source", False))
    for name, fp, tscol, gcol in [("priority_dataset", tables["priority_dataset"],
                                   "reported_at", "source_dataset"),
                                  ("hotspot_dataset", hp, "week_start", "city")]:
        if not fp.exists():
            skip(f"LEAK-5/{name}", "chronological splits do not overlap in time", "table not built")
            continue
        d = pd.read_parquet(fp)
        if "split" not in d.columns or tscol not in d.columns:
            skip(f"LEAK-5/{name}", "chronological splits do not overlap in time", "no split/ts column")
            continue
        ts = pd.to_datetime(d[tscol], utc=True, errors="coerce")
        groups = d.groupby(gcol) if (per_source and gcol in d.columns) else [("ALL", d)]
        ok, detail = True, []
        for key, g in groups:
            gts = ts.loc[g.index]
            bounds = {}
            for s in ("train", "val", "test"):
                sel = gts[g["split"] == s].dropna()
                if len(sel):
                    bounds[s] = (sel.min(), sel.max())
            if "train" in bounds and "val" in bounds and not bounds["train"][1] <= bounds["val"][0]:
                ok = False
                detail.append(f"{key}: train_max={bounds['train'][1]} > val_min={bounds['val'][0]}")
            if "val" in bounds and "test" in bounds and not bounds["val"][1] <= bounds["test"][0]:
                ok = False
                detail.append(f"{key}: val_max={bounds['val'][1]} > test_min={bounds['test'][0]}")
        scope = f"within each {gcol}" if per_source else "globally"
        check(f"LEAK-5/{name}", f"chronological splits do not overlap in time ({scope})",
              ok, "; ".join(detail))

    # ---- LEAK-6 : duplicate groups do not span splits ----------------------
    dp = p("processed", "duplicates", "duplicate_pairs.parquet")
    if dp.exists():
        d = pd.read_parquet(dp)
        if "split" in d.columns:
            spans = d.groupby("incident_a")["split"].nunique()
            check("LEAK-6", "no duplicate group spans multiple splits",
                  int((spans > 1).sum()) == 0, f"groups_spanning={int((spans > 1).sum())}")
        else:
            skip("LEAK-6", "no duplicate group spans multiple splits", "no split column")
    else:
        skip("LEAK-6", "no duplicate group spans multiple splits", "duplicate_pairs not built")

    failed = [c for c in CHECKS if c["result"] == "FAIL"]
    passed = [c for c in CHECKS if c["result"] == "PASS"]
    report = {
        "checks": CHECKS,
        "summary": {"total": len(CHECKS), "passed": len(passed),
                    "failed": len(failed), "skipped": len(CHECKS) - len(passed) - len(failed)},
        "forbidden_feature_list": FORBIDDEN,
        "note": ("LEAK-2, LEAK-3 and LEAK-4 independently re-derive the value rather than "
                 "trusting the builder that produced it."),
    }
    dest = ensure_dir(p("reports", "leakage_report.json"))
    dest.write_text(json.dumps(report, indent=2, default=str))
    log.info("wrote %s — %d passed, %d failed, %d skipped",
             dest, len(passed), len(failed), report["summary"]["skipped"])
    return 1 if (failed and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
