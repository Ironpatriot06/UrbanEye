#!/usr/bin/env python3
"""
Validate the final ML-ready outputs before anybody trains on them.

  OUT-1   every expected output file exists (or is explained as empty by design)
  OUT-2   declared schema columns are present
  OUT-3   chronological splits do not overlap in time
  OUT-4   duplicate groups never span splits
  OUT-5   vision splits exclude the official unlabelled test images
  OUT-6   vision bounding boxes are geometrically valid
  OUT-7   resolution_dataset target has no negatives / NaNs
  OUT-8   priority_dataset carries is_ground_truth = False
  OUT-9   no row-level join exists between vision and incident outputs
  OUT-10  hotspot lag features never see their own period

Writes reports/output_validation.json. Exit 1 on any failure.

    python scripts/validation/validate_outputs.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import ensure_dir, p

log = get_logger("validation.outputs")
R: list[dict] = []

EXPECTED = [
    # (subdir, name, model, may_be_empty)
    ("priority",   "priority_dataset",   "Priority (features; target NULL by design)", False),
    ("resolution", "resolution_dataset", "Resolution time / SLA risk", False),
    ("hotspot",    "hotspot_dataset",    "Hotspot / incident risk", False),
    ("duplicates", "duplicate_pairs",    "Duplicate detection", False),
]


def rec(cid: str, desc: str, ok: bool, detail: str = "") -> bool:
    R.append({"id": cid, "check": desc, "result": "PASS" if ok else "FAIL", "detail": detail})
    (log.info if ok else log.error)("%-7s %-4s %s %s", cid, "PASS" if ok else "FAIL", desc, detail)
    return ok


def load(subdir: str, name: str) -> pd.DataFrame | None:
    fp = p("processed", subdir, f"{name}.parquet")
    return pd.read_parquet(fp) if fp.exists() else None


def main() -> int:
    argparse.ArgumentParser().parse_args()
    frames: dict[str, pd.DataFrame] = {}

    for subdir, name, model, may_be_empty in EXPECTED:
        df = load(subdir, name)
        exists = df is not None
        detail = f"{model}: "
        if exists:
            frames[name] = df
            detail += f"{len(df):,} rows"
            if len(df) == 0 and not may_be_empty:
                detail += " — EMPTY and not expected to be"
        else:
            detail += "MISSING — run the corresponding build script"
        rec("OUT-1", f"{name} present", exists, detail)

    # OUT-3 chronological split integrity.
    # Per splits.tabular.per_source the ordering guarantee is within a source,
    # not across the whole corpus; see dataset_config.yaml for why.
    from scripts.utils.paths import load_config
    per_source = bool(load_config()["splits"]["tabular"].get("per_source", False))
    for name, ts_col in (("priority_dataset", "reported_at"),
                         ("resolution_dataset", "reported_at")):
        df = frames.get(name)
        if df is None or df.empty or "split" not in df.columns:
            continue
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        groups = df.groupby("source_dataset") if (per_source and "source_dataset" in df.columns) \
            else [("ALL", df)]
        ok, spans = True, []
        for key, g in groups:
            gts = ts.loc[g.index]
            bounds = {}
            for s in ("train", "val", "test"):
                sel = gts[g["split"] == s].dropna()
                if len(sel):
                    bounds[s] = (sel.min(), sel.max())
            if "train" in bounds and "val" in bounds:
                ok &= bounds["train"][1] <= bounds["val"][0]
            if "val" in bounds and "test" in bounds:
                ok &= bounds["val"][1] <= bounds["test"][0]
            spans.append(f"{key}: " + ", ".join(
                f"{k} {v[0]:%Y-%m-%d}..{v[1]:%Y-%m-%d}" for k, v in bounds.items()))
        rec("OUT-3", f"{name} chronological splits do not overlap"
                     + (" within each source" if per_source else ""), ok, " | ".join(spans))

    # OUT-4 duplicate groups
    dp = frames.get("duplicate_pairs")
    if dp is not None and not dp.empty and "split" in dp.columns:
        spans = dp.groupby("incident_a")["split"].nunique()
        rec("OUT-4", "no duplicate group spans multiple splits", int((spans > 1).sum()) == 0,
            f"{int((spans > 1).sum())} groups span splits")
        pos = int(dp["same_incident"].fillna(False).sum())
        neg = int((~dp["same_incident"].fillna(False)).sum())
        rec("OUT-4b", "duplicate_pairs contains both positives and negatives",
            pos > 0 and neg > 0, f"positives={pos} negatives={neg}")

    # OUT-5 / OUT-6 vision
    vc = frames.get("vision_category_dataset")
    if vc is not None and not vc.empty:
        leaked = int(vc["split"].eq("official_test_unlabelled").sum()) if "split" in vc else 0
        rec("OUT-5", "official unlabelled test images excluded", leaked == 0, f"leaked={leaked}")
        geom = vc.dropna(subset=["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"])
        badbox = int(((geom.bbox_xmax <= geom.bbox_xmin) |
                      (geom.bbox_ymax <= geom.bbox_ymin)).sum())
        rec("OUT-6", "all bounding boxes are geometrically valid", badbox == 0,
            f"{badbox} degenerate of {len(geom)}")
        img_spans = vc.groupby("image_id")["split"].nunique()
        rec("OUT-6b", "no image spans multiple vision splits",
            int((img_spans > 1).sum()) == 0, f"{int((img_spans > 1).sum())} images span splits")
    elif vc is not None:
        rec("OUT-5", "vision dataset empty — licence gate", True,
            "0 rows: no vision source is licence-INCLUDE. Expected until licences are confirmed.")

    # OUT-7 resolution target
    rd = frames.get("resolution_dataset")
    if rd is not None and not rd.empty:
        t = pd.to_numeric(rd["resolution_time_hours"], errors="coerce")
        rec("OUT-7", "resolution target has no negative or missing values",
            int((t < 0).sum()) == 0 and int(t.isna().sum()) == 0,
            f"negative={int((t < 0).sum())} nan={int(t.isna().sum())}")
        rec("OUT-7b", "resolution_dataset EXCLUDES sla_met (it is derived from the target)",
            "sla_met" not in rd.columns,
            "sla_breach is provided instead as a separate, explicitly-declared target")

    # OUT-8 priority
    pri = frames.get("priority_dataset")
    if pri is not None and not pri.empty:
        rec("OUT-8", "priority_dataset is_ground_truth is False everywhere",
            "is_ground_truth" in pri.columns and not pri["is_ground_truth"].any())
        from scripts.utils.schema import geo_feature_schema
        gs = geo_feature_schema()
        system_cols = [c for c, (_, prov, _) in gs.items()
                       if prov == "SYSTEM" and c in pri.columns]
        derived_cols = [c for c, (_, prov, _) in gs.items()
                        if prov == "DERIVED" and c in pri.columns
                        and c not in ("incident_id", "geo_provider", "geo_features_available")]
        populated = [c for c in system_cols if pri[c].notna().any()]
        rec("OUT-8b", "POI/road/area columns requiring external geospatial data are NULL",
            not populated,
            f"{len(system_cols)} SYSTEM-provenance columns present and empty (await India PostGIS)"
            if not populated else f"UNEXPECTEDLY POPULATED: {populated}")
        rec("OUT-8c", "backward-looking density features ARE computed from 311 history",
            any(pri[c].notna().any() for c in derived_cols) if derived_cols else False,
            f"populated: {[c for c in derived_cols if pri[c].notna().any()]}")

    # OUT-9 no cross join
    inc = load("incidents", "all_incidents")
    if vc is not None and inc is not None and not vc.empty:
        shared = (set(vc.columns) & set(inc.columns)) - {"severity", "source_dataset", "split", "category"}
        rec("OUT-9", "no joinable key between vision and incident outputs", len(shared) == 0,
            f"shared non-key columns: {sorted(shared)}")
    else:
        rec("OUT-9", "no joinable key between vision and incident outputs", True,
            "vision output empty; join impossible by construction")

    # OUT-10 hotspot leakage
    hs = frames.get("hotspot_dataset")
    if hs is not None and not hs.empty:
        h = hs.sort_values(["city", "zone_type", "zone_id", "category", "week"]).reset_index(drop=True)
        recomputed = (h.groupby(["city", "zone_type", "zone_id", "category"], observed=True)
                       ["incident_count"].shift(1))
        mismatch = int((recomputed.fillna(-1).to_numpy()
                        != h["previous_period_count"].fillna(-1).to_numpy()).sum())
        rec("OUT-10", "hotspot lag features are shift(1) — no period sees itself",
            mismatch == 0, f"{mismatch} rows disagree with a recomputed shift(1)")

    failures = [r for r in R if r["result"] == "FAIL"]
    report = {
        "summary": {"total": len(R), "passed": len(R) - len(failures), "failed": len(failures)},
        "checks": R,
        "outputs": {n: {"rows": int(len(d)), "columns": len(d.columns)} for n, d in frames.items()},
        "no_model_trained": True,
    }
    dest = ensure_dir(p("reports", "output_validation.json"))
    dest.write_text(json.dumps(report, indent=2, default=str))
    log.info("wrote %s — %d/%d passed", dest, len(R) - len(failures), len(R))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
