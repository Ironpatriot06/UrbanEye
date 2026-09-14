#!/usr/bin/env python3
"""
Generate DATA_SCHEMA.md from the code, so the document can never drift from the
actual schema. Re-run after any schema change.

    python scripts/validation/generate_schema_doc.py
"""
from __future__ import annotations
import os, sys
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.utils.paths import repo_root, load_feature_config
from scripts.utils.schema import (INCIDENT_SCHEMA, DUPLICATE_PAIR_SCHEMA,
                                  geo_feature_schema, time_feature_schema,
                                  POST_RESOLUTION_FIELDS)

LEGEND = """| Provenance | Meaning |
|---|---|
| `SOURCE` | Taken verbatim (or normalised) from the publisher |
| `DERIVED` | Computed by this pipeline from SOURCE fields |
| `SYSTEM` | Produced at runtime by the UrbanEye+ backend (India PostGIS). **NULL here.** |
| `NULL` | Genuinely unavailable in every configured source. **Enforced empty.** |
"""


def table(schema: dict) -> str:
    rows = ["| Field | Type | Provenance | Notes |", "|---|---|---|---|"]
    for k, (dt, prov, note) in schema.items():
        mark = f"**{prov}**" if prov in ("NULL", "SYSTEM") else prov
        rows.append(f"| `{k}` | `{dt}` | {mark} | {note} |")
    return "\n".join(rows)


def main() -> int:
    fc = load_feature_config()
    geo = geo_feature_schema()
    tim = time_feature_schema()
    nulls = [k for k, v in INCIDENT_SCHEMA.items() if v[1] == "NULL"]
    sysf = [k for k, v in geo.items() if v[1] == "SYSTEM"]
    derf = [k for k, v in geo.items() if v[1] == "DERIVED" and k != "incident_id"]

    doc = f"""# UrbanEye+ — Data Schema

*Generated from code by `scripts/validation/generate_schema_doc.py` on
{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Do not edit by hand.*

{LEGEND}

The `NULL` rows are not oversights. `scripts/utils/schema.py::assert_null_fields_empty`
raises `ProvenanceViolation` if any of them is ever populated, and
`scripts/validation/validate_provenance.py` actively attempts to populate them to
prove the guard works.

---

## 1. `incidents`

`data/processed/incidents/all_incidents.parquet`
(and `all_incidents_prioritised.parquet` after the priority step)

{table(INCIDENT_SCHEMA)}

### Fields that are always NULL, and why

{chr(10).join(f"- **`{k}`** — {INCIDENT_SCHEMA[k][2]}" for k in nulls)}

### Fields that are POST-RESOLUTION (never valid as model features)

{", ".join(f"`{c}`" for c in POST_RESOLUTION_FIELDS)}

These are known only after the case closes. `validate_leakage.py` asserts they do
not appear in any feature table.

---

## 2. `incident_time_features`

`data/processed/features/incident_time_features.parquet` — joins to `incidents` on `incident_id`.

{table(tim)}

`reported_at` is stored in UTC so four cities share one axis. Every feature above
is derived AFTER converting it to the source city's own timezone, using the single
mapping in `dataset_config.yaml -> cleaning.timestamps.source_timezones`. Taking
`hour` or `is_night` off the UTC timestamp would encode the city's longitude as
if it were a time-of-day effect; `hour_utc` is kept so the conversion stays
auditable.

---

## 3. `incident_geo_features`

`data/processed/geo_features/incident_geo_features.parquet` — joins to `incidents` on `incident_id`.

{table(geo)}

**{len(sysf)} of these columns require an external geospatial layer** and are NULL
for every US 311 row. **{len(derf)} are computed now** from 311 history alone.

- Requires PostGIS: {", ".join(f"`{c}`" for c in sorted(sysf))}
- Computed now: {", ".join(f"`{c}`" for c in sorted(derf))}

---

## 4. `duplicate_pairs`

`data/processed/duplicates/duplicate_pairs.parquet`

{table(DUPLICATE_PAIR_SCHEMA)}

---

## 5. ML-ready tables

| Table | Path | Target | Target available today? |
|---|---|---|---|
| `priority_dataset` | `data/processed/priority/` | `priority_label` | **No** — NULL everywhere. `priority_baseline` is provided but is DERIVED, not a target. |
| `resolution_dataset` | `data/processed/resolution/` | `resolution_time_hours`, `sla_breach` | Yes |
| `hotspot_dataset` | `data/processed/hotspot/` | `future_incident_count`, `future_incident_flag` | Yes |
| `duplicate_pairs` | `data/processed/duplicates/` | `same_incident` | Yes (Chicago only) |
| `urbaneye_ml` | `data/processed/ml/` | `resolution_time_hours`, `sla_breach` | Yes. `priority_label` is present and empty; see below. |

### `urbaneye_ml` column roles

Every column in the consolidated table carries a declared role in
`data/processed/ml/urbaneye_ml.manifest.json`:

| Role | Meaning |
|---|---|
| `identifier` | keys, not features |
| `predictor` | knowable at report time, safe to train on |
| `predictor_unavailable` | declared with the right name and dtype, NULL because no reference data exists |
| `target` | a genuinely observed outcome |
| `target_support` | describes the target (censoring flag, SLA window), never an input |
| `policy_metadata` | the rule engine's own output — a benchmark to beat, never a feature and never a label |
| `label_provenance` | what is missing and why, machine-readable |
| `context_metadata` | coordinates, timestamps, split — for auditing and joining, not for training |

Raw `latitude`/`longitude` are context, not predictors, on purpose: a model that
learns "latitude 41.88 is high priority" has learned Chicago.

---

## 6. Category taxonomy

{", ".join(f"`{c}`" for c in [
 "ROAD_DAMAGE","POTHOLE","GARBAGE_DUMPING","STREETLIGHT_FAULT","TRAFFIC_SIGNAL_FAULT",
 "FOOTPATH_DAMAGE","OPEN_MANHOLE","DRAINAGE_SEWER","WATER_LEAKAGE_WATERLOGGING","FALLEN_TREE",
 "PUBLIC_INFRASTRUCTURE_DAMAGE","SIGNAGE_DAMAGE","GRAFFITI_VISUAL_POLLUTION",
 "CONSTRUCTION_OBSTRUCTION","OTHER"])}

Plus three sentinels that stay in `incidents` for auditability but never reach a
model-ready table: `REVIEW_REQUIRED`, `OUT_OF_SCOPE`, `UNMAPPED`.

Mapping lives in `config/category_mapping.csv`. Unmapped source values are never
guessed — they become `UNMAPPED` and are reported with row counts in
`reports/unmapped_categories.csv`.
"""
    out = repo_root() / "DATA_SCHEMA.md"
    out.write_text(doc)
    print(f"wrote {out} ({len(INCIDENT_SCHEMA)} incident fields, {len(geo)} geo fields)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
