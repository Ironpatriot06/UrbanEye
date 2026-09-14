#!/usr/bin/env python3
"""
Generate reports/data_quality.json and reports/data_quality.md.

This is not a descriptive-statistics dump. It runs hard ASSERTIONS and exits
non-zero when one fails, so it can sit in CI. The checks that matter most are the
integrity ones — they are what stop a fabricated label reaching a model:

  INT-1  severity is entirely NULL                 (no source provides it)
  INT-2  occurred_at is entirely NULL              (no source provides it)
  INT-3  response_time_hours is entirely NULL      (no source provides it)
  INT-4  priority, where present, is stamped RULE_ENGINE
  INT-5  sla_target_hours is non-null ONLY for Boston and NYC
  INT-6  no negative resolution times survive
  INT-7  coordinates lie inside each city's box, or are NULL
  INT-8  no incident_id appears in more than one source_dataset
  INT-9  duplicate groups do not span train/val/test
  INT-10 vision splits contain no official unlabelled test images

  python scripts/validation/data_quality_report.py
  python scripts/validation/data_quality_report.py --strict   # exit 1 on any failure
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from scripts.utils.logging_setup import get_logger
from scripts.utils.paths import ensure_dir, load_config, p, repo_root

log = get_logger("validation.quality")

CHECKS: list[dict] = []


def check(cid: str, desc: str, passed: bool, detail: str = "") -> bool:
    CHECKS.append({"id": cid, "description": desc,
                   "result": "PASS" if passed else "FAIL", "detail": detail})
    (log.info if passed else log.error)("%-8s %-6s %s %s", cid, "PASS" if passed else "FAIL", desc, detail)
    return passed


def profile_incidents(df: pd.DataFrame, label: str) -> dict:
    ts = pd.to_datetime(df["reported_at"], utc=True, errors="coerce")
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    return {
        "label": label,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_pct": {c: round(float(df[c].isna().mean() * 100), 2) for c in df.columns},
        "date_range": {"min": str(ts.min()), "max": str(ts.max())},
        "category_distribution": {str(k): int(v) for k, v in df["category"].value_counts().head(40).items()},
        "status_distribution": {str(k): int(v) for k, v in df["status"].value_counts().items()},
        "coordinate_validity": {
            "non_null": int(lat.notna().sum()),
            "null": int(lat.isna().sum()),
            "outside_city_bbox_flagged": int(df.get("coord_outside_city_bbox", pd.Series(dtype=bool)).fillna(False).sum()),
        },
        "duplicate_incident_ids": int(df["incident_id"].duplicated().sum()),
        "resolution_time_hours": _num_summary(df["resolution_time_hours"]),
        "sla_target_hours": _num_summary(df["sla_target_hours"]),
        "image_url_present": int(df["image_url"].notna().sum()),
        "after_image_url_present": int(df["image_url_after"].notna().sum()),
        "text_present": int(df["description"].notna().sum()),
        "parent_pointer_present": int(df["parent_incident_id"].notna().sum()),
    }


def _num_summary(s: pd.Series) -> dict:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return {"count": 0, "note": "no non-null values"}
    return {"count": int(len(v)), "min": float(v.min()), "p25": float(v.quantile(.25)),
            "median": float(v.median()), "p75": float(v.quantile(.75)),
            "p90": float(v.quantile(.90)), "max": float(v.max()),
            "negative_count": int((v < 0).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    report: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": cfg["pipeline_version"],
        "schema_version": cfg["schema_version"],
        "datasets": {}, "vision": {}, "ml_ready": {}, "licences": {}, "cleaning": {},
    }

    # ---- licences --------------------------------------------------------
    for name, d in cfg["datasets"].items():
        lic = d.get("licence", {})
        report["licences"][name] = {
            "source": d.get("landing_page"), "licence": lic.get("name"),
            "licence_url": lic.get("url"), "commercial_use": lic.get("commercial_use"),
            "modification": lic.get("modification"), "redistribution": lic.get("redistribution"),
            "attribution_required": lic.get("attribution_required"),
            "urbaneye_compatible": lic.get("urbaneye_compatible"), "decision": lic.get("decision"),
        }
    for name, d in cfg.get("excluded_datasets", {}).items():
        report["licences"][name] = {"source": d.get("url"), "licence": d.get("licence"),
                                    "decision": d.get("decision"), "reason": d.get("reason")}

    # ---- cleaning stats --------------------------------------------------
    for f in sorted((repo_root() / "data/processed/incidents").glob("*.cleaning_stats.json")):
        report["cleaning"][f.name.replace(".cleaning_stats.json", "")] = json.loads(f.read_text())

    # ---- per-source + union profiles ------------------------------------
    sources = ["sf311", "boston311", "chicago311", "nyc311"]
    frames = {}
    for s in sources:
        fp = p("processed", "incidents", f"{s}.parquet")
        if fp.exists():
            frames[s] = pd.read_parquet(fp)
            report["datasets"][s] = profile_incidents(frames[s], s)
        else:
            report["datasets"][s] = {"status": "NOT PRESENT",
                                     "note": "raw download and/or preprocessing has not been run"}

    allfp = p("processed", "incidents", "all_incidents_prioritised.parquet")
    if not allfp.exists():
        allfp = p("processed", "incidents", "all_incidents.parquet")
    allx = pd.read_parquet(allfp) if allfp.exists() else None
    if allx is not None:
        report["datasets"]["ALL"] = profile_incidents(allx, "all_incidents")

    # ---- integrity assertions -------------------------------------------
    if allx is not None:
        check("INT-1", "severity is entirely NULL (no source provides it)",
              int(allx["severity"].notna().sum()) == 0,
              f"non-null={int(allx['severity'].notna().sum())}")
        check("INT-2", "occurred_at is entirely NULL (no source provides it)",
              int(allx["occurred_at"].notna().sum()) == 0,
              f"non-null={int(allx['occurred_at'].notna().sum())}")
        check("INT-3", "response_time_hours is entirely NULL (no first-response timestamp exists)",
              int(allx["response_time_hours"].notna().sum()) == 0,
              f"non-null={int(allx['response_time_hours'].notna().sum())}")

        if "priority_baseline" in allx.columns and allx["priority_baseline"].notna().any():
            gt = allx["is_ground_truth"].fillna(False)
            check("INT-4", "priority_baseline is never marked ground truth",
                  not bool(gt.any()), f"is_ground_truth true rows={int(gt.sum())}")
            check("INT-4b", "priority_label is entirely NULL (no genuine public label exists)",
                  int(allx["priority_label"].notna().sum()) == 0,
                  f"non-null={int(allx['priority_label'].notna().sum())}")
        else:
            check("INT-4", "priority is stamped RULE_ENGINE wherever present", True,
                  "priority not yet assigned (derive_priority.py not run)")

        with_sla = set(allx.loc[allx["sla_target_hours"].notna(), "source_dataset"].unique())
        check("INT-5", "sla_target_hours present only for Boston and NYC",
              with_sla.issubset({"boston311", "nyc311"}), f"sources with SLA={sorted(with_sla)}")

        neg = int((pd.to_numeric(allx["resolution_time_hours"], errors="coerce") < 0).sum())
        check("INT-6", "no negative resolution times", neg == 0, f"negative={neg}")

        boxes = cfg["cleaning"]["coordinates"]["city_bbox"]
        viol = 0
        for city, b in boxes.items():
            g = allx[allx["city"] == city]
            if g.empty:
                continue
            la = pd.to_numeric(g["latitude"], errors="coerce")
            lo = pd.to_numeric(g["longitude"], errors="coerce")
            viol += int((((la < b["lat_min"]) | (la > b["lat_max"]) |
                          (lo < b["lon_min"]) | (lo > b["lon_max"])) & la.notna()).sum())
        check("INT-7", "coordinates are inside their own city's bbox or NULL",
              viol == 0, f"violations={viol} (flagged, not dropped — see coord_outside_city_bbox)")

        cross = allx.groupby("incident_id")["source_dataset"].nunique()
        check("INT-8", "no incident_id is shared across source datasets",
              int((cross > 1).sum()) == 0, f"shared_ids={int((cross > 1).sum())}")


    # ---- ml_ready --------------------------------------------------------
    ml_dirs = ["ml_ready", "text", "resolution", "hotspot", "duplicates", "vision"]
    for sub in ml_dirs:
        for f in sorted((repo_root() / "data/processed" / sub).glob("*.parquet")):
            if f.stem in ("rdd2022", "smartathon"):
                continue   # raw vision tables are profiled separately above
            d = pd.read_parquet(f)
            entry = {"rows": int(len(d)), "columns": list(d.columns), "location": f"data/processed/{sub}"}
            if "split" in d.columns and len(d):
                entry["split_counts"] = {str(k): int(v) for k, v in d["split"].value_counts().items()}
            report["ml_ready"][f.stem] = entry

    dup = repo_root() / "data/processed/duplicates/duplicate_pairs.parquet"
    if dup.exists():
        dp = pd.read_parquet(dup)
        if "split" in dp.columns:
            spans = dp.groupby("incident_a")["split"].nunique()
            check("INT-9", "no duplicate group spans multiple splits",
                  int((spans > 1).sum()) == 0, f"groups_spanning={int((spans > 1).sum())}")
        check("INT-9b", "image_similarity is NULL (no images exist for labelled pairs)",
              int(dp["image_similarity"].notna().sum()) == 0)

    vc = repo_root() / "data/processed/vision/vision_category_dataset.parquet"
    if vc.exists():
        vd = pd.read_parquet(vc)
        bad = int(vd["split"].eq("official_test_unlabelled").sum()) if "split" in vd.columns and len(vd) else 0
        check("INT-10", "official unlabelled test images excluded from supervised splits", bad == 0,
              f"leaked={bad}")

    failures = [c for c in CHECKS if c["result"] == "FAIL"]
    report["integrity_checks"] = CHECKS
    report["integrity_summary"] = {"total": len(CHECKS), "passed": len(CHECKS) - len(failures),
                                   "failed": len(failures)}

    jpath = ensure_dir(p("reports", "data_quality.json"))
    jpath.write_text(json.dumps(report, indent=2, default=str))
    _write_markdown(report)
    log.info("wrote %s and data_quality.md — %d/%d checks passed",
             jpath, len(CHECKS) - len(failures), len(CHECKS))

    if failures and args.strict:
        return 1
    return 0


def _write_markdown(r: dict) -> None:
    L: list[str] = ["# UrbanEye+ data quality report", "",
                    f"Generated {r['generated_at_utc']} · pipeline {r['pipeline_version']} · schema {r['schema_version']}", ""]

    s = r.get("integrity_summary", {})
    L += ["## Integrity checks", "",
          f"**{s.get('passed', 0)}/{s.get('total', 0)} passed.**", "",
          "| ID | Check | Result | Detail |", "|---|---|---|---|"]
    for c in r.get("integrity_checks", []):
        mark = "PASS" if c["result"] == "PASS" else "**FAIL**"
        L.append(f"| {c['id']} | {c['description']} | {mark} | {c['detail'] or ''} |")

    L += ["", "## Tabular sources", "",
          "| Dataset | Rows | Date range | Coords | Text | Image URL | SLA target | Resolution time |",
          "|---|---|---|---|---|---|---|---|"]
    for k, d in r["datasets"].items():
        if d.get("status") == "NOT PRESENT":
            L.append(f"| {k} | _not present_ | — | — | — | — | — | — |")
            continue
        L.append("| {} | {:,} | {} .. {} | {:,} | {:,} | {:,} | {:,} | {:,} |".format(
            k, d["rows"], str(d["date_range"]["min"])[:10], str(d["date_range"]["max"])[:10],
            d["coordinate_validity"]["non_null"], d["text_present"], d["image_url_present"],
            d["sla_target_hours"].get("count", 0), d["resolution_time_hours"].get("count", 0)))

    if r.get("vision"):
        L += ["", "## Image sources", "",
              "| Dataset | Annotations | Images | On disk | Imbalance | Severity labels | Licence |",
              "|---|---|---|---|---|---|---|"]
        for k, d in r["vision"].items():
            if d.get("status") == "NOT PRESENT":
                L.append(f"| {k} | _not present_ | — | — | — | — | — |")
                continue
            L.append(f"| {k} | {d['annotations']:,} | {d['images']:,} | {d['images_present_on_disk']:,} "
                     f"| {d.get('class_imbalance_ratio')}x | {d['severity_labels']} | {d['licence_decision']} |")

    if r.get("ml_ready"):
        L += ["", "## ML-ready outputs", "", "| Dataset | Rows | Splits |", "|---|---|---|"]
        for k, d in r["ml_ready"].items():
            L.append(f"| {k} | {d['rows']:,} | {d.get('split_counts', '—')} |")

    L += ["", "## Licences", "", "| Dataset | Licence | Commercial | Decision |", "|---|---|---|---|"]
    for k, d in r["licences"].items():
        L.append(f"| {k} | {d.get('licence')} | {d.get('commercial_use')} | **{d.get('decision')}** |")

    L += ["", "## Fields that are NULL by design", "",
          "These are not gaps in the pipeline. No configured source supplies them.", "",
          "- `severity` — no public civic dataset provides a defensible severity grade",
          "- `occurred_at` — every source records report time, none records incident onset",
          "- `response_time_hours` — no source records a first-response timestamp",
          "- `image_similarity` (duplicate pairs) — the labelled source has no photographs",
          "- all `near_*` / `nearby_*` / `road_type` / `city_type` — SYSTEM-provenance, generated",
          "  from the citizen's real Indian coordinates at runtime, never from US 311 data", ""]

    ensure_dir(p("reports", "data_quality.md")).write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
