# UrbanEye+ data quality report

Generated 2026-09-14T18:11:06.245530+00:00 · pipeline 2.0.0 · schema 2.0.0

## Integrity checks

**11/11 passed.**

| ID | Check | Result | Detail |
|---|---|---|---|
| INT-1 | severity is entirely NULL (no source provides it) | PASS | non-null=0 |
| INT-2 | occurred_at is entirely NULL (no source provides it) | PASS | non-null=0 |
| INT-3 | response_time_hours is entirely NULL (no first-response timestamp exists) | PASS | non-null=0 |
| INT-4 | priority_baseline is never marked ground truth | PASS | is_ground_truth true rows=0 |
| INT-4b | priority_label is entirely NULL (no genuine public label exists) | PASS | non-null=0 |
| INT-5 | sla_target_hours present only for Boston and NYC | PASS | sources with SLA=['nyc311'] |
| INT-6 | no negative resolution times | PASS | negative=0 |
| INT-7 | coordinates are inside their own city's bbox or NULL | PASS | violations=0 (flagged, not dropped — see coord_outside_city_bbox) |
| INT-8 | no incident_id is shared across source datasets | PASS | shared_ids=0 |
| INT-9 | no duplicate group spans multiple splits | PASS | groups_spanning=0 |
| INT-9b | image_similarity is NULL (no images exist for labelled pairs) | PASS |  |

## Tabular sources

| Dataset | Rows | Date range | Coords | Text | Image URL | SLA target | Resolution time |
|---|---|---|---|---|---|---|---|
| sf311 | 200,000 | 2018-01-01 .. 2018-05-10 | 193,724 | 199,967 | 84,309 | 0 | 195,698 |
| boston311 | _not present_ | — | — | — | — | — | — |
| chicago311 | 1,600,000 | 2018-07-01 .. 2020-08-28 | 1,593,148 | 0 | 0 | 0 | 1,400,289 |
| nyc311 | 100,000 | 2010-01-01 .. 2010-03-09 | 83,522 | 99,896 | 0 | 22,458 | 84,495 |
| ALL | 1,900,000 | 2010-01-01 .. 2020-08-28 | 1,870,394 | 299,863 | 84,309 | 22,458 | 1,680,482 |

## ML-ready outputs

| Dataset | Rows | Splits |
|---|---|---|
| resolution_dataset | 942,325 | {'train': 659627, 'test': 141350, 'val': 141348} |
| hotspot_dataset | 67,092 | {'train': 47694, 'val': 9949, 'test': 9449} |
| duplicate_pairs | 455,460 | {'train': 324501, 'val': 69276, 'test': 61683} |

## Licences

| Dataset | Licence | Commercial | Decision |
|---|---|---|---|
| sf311 | Open Data Commons Public Domain Dedication and Licence (PDDL) 1.0 | True | **INCLUDE** |
| boston311 | Open Data Commons Public Domain Dedication and Licence (PDDL) | True | **INCLUDE** |
| chicago311 | City of Chicago Open Data Terms of Use | True | **INCLUDE** |
| nyc311 | NYC Open Data Terms of Use | True | **INCLUDE** |

## Fields that are NULL by design

These are not gaps in the pipeline. No configured source supplies them.

- `severity` — no public civic dataset provides a defensible severity grade
- `occurred_at` — every source records report time, none records incident onset
- `response_time_hours` — no source records a first-response timestamp
- `image_similarity` (duplicate pairs) — the labelled source has no photographs
- all `near_*` / `nearby_*` / `road_type` / `city_type` — SYSTEM-provenance, generated
  from the citizen's real Indian coordinates at runtime, never from US 311 data
