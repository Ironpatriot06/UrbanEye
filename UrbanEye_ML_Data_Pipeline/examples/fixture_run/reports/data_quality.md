# UrbanEye+ data quality report

Generated 2026-09-13T17:29:50.066425+00:00 · pipeline 2.0.0 · schema 2.0.0

## Integrity checks

**11/11 passed.**

| ID | Check | Result | Detail |
|---|---|---|---|
| INT-1 | severity is entirely NULL (no source provides it) | PASS | non-null=0 |
| INT-2 | occurred_at is entirely NULL (no source provides it) | PASS | non-null=0 |
| INT-3 | response_time_hours is entirely NULL (no first-response timestamp exists) | PASS | non-null=0 |
| INT-4 | priority_baseline is never marked ground truth | PASS | is_ground_truth true rows=0 |
| INT-4b | priority_label is entirely NULL (no genuine public label exists) | PASS | non-null=0 |
| INT-5 | sla_target_hours present only for Boston and NYC | PASS | sources with SLA=['boston311', 'nyc311'] |
| INT-6 | no negative resolution times | PASS | negative=0 |
| INT-7 | coordinates are inside their own city's bbox or NULL | PASS | violations=0 (flagged, not dropped — see coord_outside_city_bbox) |
| INT-8 | no incident_id is shared across source datasets | PASS | shared_ids=0 |
| INT-9 | no duplicate group spans multiple splits | PASS | groups_spanning=0 |
| INT-9b | image_similarity is NULL (no images exist for labelled pairs) | PASS |  |

## Tabular sources

| Dataset | Rows | Date range | Coords | Text | Image URL | SLA target | Resolution time |
|---|---|---|---|---|---|---|---|
| sf311 | 400 | 2021-03-02 .. 2024-12-27 | 376 | 400 | 100 | 0 | 363 |
| boston311 | 300 | 2021-03-09 .. 2024-12-17 | 300 | 300 | 60 | 300 | 262 |
| chicago311 | 345 | 2021-03-01 .. 2024-12-03 | 345 | 0 | 0 | 0 | 303 |
| nyc311 | 400 | 2021-03-11 .. 2024-12-21 | 400 | 400 | 0 | 266 | 367 |
| ALL | 1,445 | 2021-03-01 .. 2024-12-27 | 1,421 | 1,100 | 160 | 566 | 1,295 |

## ML-ready outputs

| Dataset | Rows | Splits |
|---|---|---|
| resolution_dataset | 1,115 | {'train': 520, 'val': 304, 'test': 291} |
| hotspot_dataset | 783 | {'train': 462, 'val': 217, 'test': 104} |
| duplicate_pairs | 73 | {'train': 54, 'test': 10, 'val': 9} |

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
