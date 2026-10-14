#!/usr/bin/env python3
"""
Phase 2B — US 311 training data vs UrbanEye: explicit, measured alignment.

Writes reports/phase2b/domain_alignment.{json,md}. Every number is computed
from data/processed/phase2b/incident_features_v2.parquet, the mapping file and
the backend policy constants mirrored (and drift-tested) in common.py.

    python scripts/phase2b/domain_alignment.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json

import pandas as pd

from scripts.phase2b.common import (CATEGORY_DEFAULT_PRIORITY, DATA_DIR, DEFAULT_SLA_HOURS,
                                    REPORT_DIR, UE_CATEGORIES, default_sla_hours, load_ue_mapping,
                                    now_utc, write_json)

BENGALURU_BBOX = (12.70, 13.25, 77.35, 77.90)


def main() -> int:
    d = pd.read_parquet(DATA_DIR / "incident_features_v2.parquet",
                        columns=["source_dataset", "category", "source_category_raw", "subcategory",
                                 "ue_category", "ue_confidence", "latitude", "longitude",
                                 "report_channel", "department", "zone_type", "valid_resolution",
                                 "resolution_time_hours", "reported_at"])
    rules = load_ue_mapping()

    # rows per rule, so every mapping line carries its evidence
    rule_rows = []
    for r in rules:
        m = (d["source_dataset"] == r["source_dataset"]) & \
            (d["source_category_raw"] == r["source_category_raw"])
        if r["subcategory"] != "*":
            m &= d["subcategory"] == r["subcategory"]
        m &= d["ue_category"].fillna("") == r["urbaneye_category"]
        x = d[m.fillna(False)]
        rule_rows.append({**r, "rows": int(len(x)),
                          "canonical_categories": sorted(x["category"].dropna().unique().tolist()),
                          "median_resolution_h": (None if not x["valid_resolution"].any() else
                                                  round(float(x.loc[x["valid_resolution"],
                                                        "resolution_time_hours"].median()), 1))})

    per_ue = {}
    for c in UE_CATEGORIES:
        x = d[d["ue_category"] == c]
        v = x[x["valid_resolution"]]
        per_ue[c] = {
            "status": "SUPPORTED" if len(x) else "UNSUPPORTED (no training data)",
            "rows": int(len(x)),
            "by_source": x["source_dataset"].value_counts().to_dict(),
            "by_confidence": x["ue_confidence"].value_counts().to_dict(),
            "median_resolution_h_by_source": v.groupby("source_dataset")["resolution_time_hours"]
                                              .median().round(1).to_dict(),
            "default_priority": CATEGORY_DEFAULT_PRIORITY[c],
            "default_sla_hours": default_sla_hours(c),
            "share_resolved_within_sla_by_source": (
                (v["resolution_time_hours"] < default_sla_hours(c))
                .groupby(v["source_dataset"]).mean().round(3).to_dict()),
        }

    # canonical categories with NO UrbanEye equivalent (would be OTHER in UrbanEye)
    aux = d[d["ue_category"].isna() | (d["ue_category"] == "UNSUPPORTED")]
    aux = aux[~aux["category"].isin(["OUT_OF_SCOPE", "UNMAPPED"])]
    no_equiv = aux["category"].value_counts().to_dict()

    lat0, lat1, lon0, lon1 = BENGALURU_BBOX
    geo = {}
    for s, g in d.groupby("source_dataset"):
        geo[s] = {"lat_range": [round(float(g["latitude"].min()), 3),
                                round(float(g["latitude"].max()), 3)],
                  "lon_range": [round(float(g["longitude"].min()), 3),
                                round(float(g["longitude"].max()), 3)],
                  "missing_coordinates": round(float(g["latitude"].isna().mean()), 4),
                  "rows_inside_bengaluru_bbox": int(((g["latitude"].between(lat0, lat1))
                                                     & (g["longitude"].between(lon0, lon1))).sum()),
                  "zone_type": g["zone_type"].dropna().unique().tolist(),
                  "period": [str(g["reported_at"].min().date()), str(g["reported_at"].max().date())]}

    db_audit_path = REPORT_DIR / "urbaneye_db_audit.json"
    db = json.loads(db_audit_path.read_text()) if db_audit_path.exists() else None

    out = {
        "generated_at_utc": now_utc(),
        "mapping_file": "config/urbaneye_category_mapping.csv",
        "mapping_rules": rule_rows,
        "per_urbaneye_category": per_ue,
        "canonical_categories_without_urbaneye_equivalent": no_equiv,
        "geography": {"training": geo, "bengaluru_bbox": BENGALURU_BBOX,
                      "urbaneye_db_by_city_bbox": db and db["incidents"]["by_city_bbox"]},
        "source_channel": {
            "training_report_channel": d["report_channel"].value_counts().to_dict(),
            "urbaneye_source_enum": ["CITIZEN", "CAMERA", "IOT"],
            "urbaneye_db_sources": db and db["incidents"]["by_source"],
            "verdict": "no defensible mapping; report_channel excluded from deployable models",
        },
        "department": {
            "training_distinct": int(d["department"].nunique()),
            "urbaneye": "no department column",
            "verdict": "excluded from deployable models",
        },
        "zones": {"training": "Chicago ward / SF analysis neighbourhood / NYC community board",
                  "urbaneye": "no zone/ward/city column; PostGIS location only (GiST index exists)",
                  "verdict": "deployable models use no zone; hotspot v2 uses a 1 km grid computed "
                             "from location"},
        "sla": {
            "urbaneye_default_sla_hours": DEFAULT_SLA_HOURS,
            "urbaneye_category_default_priority": CATEGORY_DEFAULT_PRIORITY,
            "urbaneye_db_sla_values_seen": db and db["incidents"]["sla_hours_values"],
            "phase2a_target": "NYC agency Due Date vs closed_date (median deadline 240 h), "
                              "open cases dropped",
            "phase2b_target": "UrbanEye default deadline of the mapped category; open past "
                              "deadline = breach",
        },
    }
    write_json(REPORT_DIR / "domain_alignment.json", out)
    (REPORT_DIR / "domain_alignment.md").write_text(_md(out))
    print("wrote domain_alignment.{json,md}")
    return 0


def _md(o: dict) -> str:
    L = ["# Phase 2B — Domain alignment (US 311 → UrbanEye)", "",
         f"Generated {o['generated_at_utc']} by `scripts/phase2b/domain_alignment.py`.", "",
         "## UrbanEye categories", "",
         "| UrbanEye | Status | Rows | Sources | Default SLA | Median resolution by source (h) "
         "| Resolved within SLA by source |", "|---|---|---|---|---|---|---|"]
    for c, v in o["per_urbaneye_category"].items():
        L.append(f"| {c} | {v['status']} | {v['rows']:,} | {v['by_source']} | "
                 f"{v['default_priority']} / {v['default_sla_hours']} h | "
                 f"{v['median_resolution_h_by_source']} | "
                 f"{v['share_resolved_within_sla_by_source']} |")
    L += ["", "## Mapping rules (raw publisher category → UrbanEye)", "",
          "| Source | Raw category | Subcategory | → UrbanEye | Confidence | Rows | Notes |",
          "|---|---|---|---|---|---|---|"]
    for r in o["mapping_rules"]:
        L.append(f"| {r['source_dataset']} | {r['source_category_raw']} | {r['subcategory']} | "
                 f"{r['urbaneye_category']} | {r['confidence'] or '—'} | {r['rows']:,} | "
                 f"{r['notes']} |")
    L += ["", "## Canonical categories with no UrbanEye equivalent", "",
          "Used only as auxiliary training rows; in UrbanEye these would be filed as OTHER, "
          "whose composition in Bengaluru is unknown.", ""]
    L += [f"- {k}: {v:,}" for k, v in o["canonical_categories_without_urbaneye_equivalent"].items()]
    L += ["", "## Geography", ""]
    for s, g in o["geography"]["training"].items():
        L.append(f"- **{s}**: lat {g['lat_range']}, lon {g['lon_range']}, period {g['period']}, "
                 f"zones {g['zone_type']}, rows inside Bengaluru bbox: "
                 f"{g['rows_inside_bengaluru_bbox']}")
    L += [f"- UrbanEye DB incidents by bbox: {o['geography']['urbaneye_db_by_city_bbox']}", "",
          "## Source / department / zones / SLA", "",
          f"- Channel: {o['source_channel']['verdict']}",
          f"- Department: {o['department']['verdict']}",
          f"- Zones: {o['zones']['verdict']}",
          f"- SLA (UrbanEye): {o['sla']['urbaneye_default_sla_hours']}; values in DB: "
          f"{o['sla']['urbaneye_db_sla_values_seen']}",
          f"- Phase 2A SLA target: {o['sla']['phase2a_target']}",
          f"- Phase 2B SLA target: {o['sla']['phase2b_target']}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
