# UrbanEye+ pipeline audit

Generated 2026-09-14T12:38:26.774324+00:00 · pipeline 2.0.0

**56/56 passed, 0 failed, 0 skipped.**


## DATASET

| ID | Check | Result | Detail |
|---|---|---|---|
| DS-1 | union row count equals the sum of the per-source tables | PASS | union=1,900,000 sum_of_sources=1,900,000 |
| DS-2 | no duplicate incident_id | PASS | duplicates=0 |
| DS-3 | no incident_id shared across source datasets | PASS |  |
| DS-4 | every row carries a source_dataset and a city | PASS |  |
| DS-5/time_features | feature table covers every incident exactly once | PASS | rows=1,900,000 vs incidents=1,900,000 |
| DS-5/geo_features | feature table covers every incident exactly once | PASS | rows=1,900,000 vs incidents=1,900,000 |

## TIME

| ID | Check | Result | Detail |
|---|---|---|---|
| TF-1 | no missing time features where reported_at is present | PASS | missing=0 |
| TF-2 | hour in [0,23], day_of_week in [0,6], month in [1,12] | PASS |  |
| TF-3 | is_night agrees with the configured local night window | PASS |  |
| TF-4 | is_weekend agrees with day_of_week | PASS |  |
| TF-5 | local hour re-derives exactly from the configured city timezone | PASS | mismatches=0 of 2000 resampled rows |
| TF-6 | local time genuinely differs from UTC (features are not UTC in disguise) | PASS | rows where local hour != UTC hour: 1.000 |

## SPATIAL

| ID | Check | Result | Detail |
|---|---|---|---|
| SP-1 | POI/road/area columns are all present in the schema | PASS | 23 declared |
| SP-2 | POI/road/area columns are NULL (no reference data exists to fill them) | PASS | 0 of 23 populated; no reference geodata in data/external |
| SP-3 | coordinates, where present, are finite and in range | PASS |  |

## DENSITY

| ID | Check | Result | Detail |
|---|---|---|---|
| DN-1 | density is NULL exactly where coordinates are missing | PASS | rows without coordinates=29,606 |
| DN-2 | windows are monotone: 24h <= 7d <= 30d | PASS |  |
| DN-3 | same-category 30d count never exceeds the all-category 30d count | PASS |  |
| DN-4 | category_incident_density is a share in [0,1] | PASS |  |
| DN-5 | density is not degenerate (it varies across rows) | PASS | distinct values=1555 over 1,870,394 usable rows |
| DN-6 | every density column re-derives exactly with reported_at < t | PASS | mismatches={'nearby_similar_incidents_24h': 0, 'nearby_similar_incidents_7d': 0, 'nearby_similar_incidents_30d': 0, 'local_incident_density': 0} over 200 resampled rows |
| DN-7 | no density value exceeds what a backward window can produce | PASS | suspect_rows=0 |
| DN-8/hours_since_previous_similar_incident | recency is positive and inside its own window | PASS | range=[0.00, 720.00] window=720.0 |
| DN-9/hours_since_previous_similar_incident | recency is non-null exactly when a 30d same-category neighbour exists | PASS | inconsistent=0 |

## PRIORITY

| ID | Check | Result | Detail |
|---|---|---|---|
| PR-1 | every row has a baseline band | PASS | missing=0 |
| PR-2 | baseline uses more than one band | PASS |  |
| PR-3 | score is in [0,1] | PASS |  |
| PR-4 | confidence is consistent with the declared tiers | PASS |  |
| PR-5 | priority_method records the config version used | PASS |  |
| PR-6 | is_ground_truth is False on every row | PASS |  |
| PR-7 | score is not a pure category lookup (context features actually move it) | PASS |  |

## TARGET

| ID | Check | Result | Detail |
|---|---|---|---|
| TG-1 | priority_label is empty (no public source provides one) | PASS |  |
| TG-2 | priority_label_source is empty | PASS |  |
| TG-3 | no row claims ground truth | PASS |  |
| TG-4 | the dataset states its own target situation in-band | PASS |  |
| TG-5 | at least one genuinely observed target exists | PASS | resolution_time_hours rows=1,853,828 |
| TG-6 | sla_target_hours comes only from publishers that ship a due date | PASS | sources=['nyc311'] |

## SPLIT

| ID | Check | Result | Detail |
|---|---|---|---|
| SPL-1/priority_dataset | train, val and test are all non-empty | PASS | present=['test', 'train', 'val'] |
| SPL-2/priority_dataset | within each source_dataset, train precedes val precedes test | PASS |  |
| SPL-3/priority_dataset | no incident appears in more than one split | PASS | contaminated=0 |
| SPL-1/resolution_dataset | train, val and test are all non-empty | PASS | present=['test', 'train', 'val'] |
| SPL-2/resolution_dataset | within each source_dataset, train precedes val precedes test | PASS |  |
| SPL-3/resolution_dataset | no incident appears in more than one split | PASS | contaminated=0 |
| SPL-1/hotspot_dataset | train, val and test are all non-empty | PASS | present=['test', 'train', 'val'] |
| SPL-2/hotspot_dataset | within each city, train precedes val precedes test | PASS |  |
| SPL-1/urbaneye_ml | train, val and test are all non-empty | PASS | present=['test', 'train', 'val'] |
| SPL-2/urbaneye_ml | within each source_dataset, train precedes val precedes test | PASS |  |
| SPL-3/urbaneye_ml | no incident appears in more than one split | PASS | contaminated=0 |

## LEAKAGE

| ID | Check | Result | Detail |
|---|---|---|---|
| LK-1 | priority_dataset contains no post-resolution column | PASS | found=[] |
| LK-2 | no report timestamp lies in the future | PASS | future_rows=0 |
| LK-3 | ML dataset contains no post-resolution column outside its targets | PASS | found=[] |
| LK-4 | no declared predictor is a target or derived from one | PASS | predictors=19 |
| LK-5 | priority policy output is not declared as a predictor | PASS |  |
| LK-6 | raw coordinates are not declared as predictors | PASS |  |
| LK-7 | no fabricated priority target survives anywhere in the ML table | PASS |  |
| LK-8 | ML dataset carries no future timestamps | PASS |  |

## Dataset

- rows: **1,900,000**
- by source: {'chicago311': 1600000, 'sf311': 200000, 'nyc311': 100000}
- date range: 2010-01-01 05:38:00+00:00 .. 2020-08-28 12:52:21+00:00

## Priority

- baseline: {'P3': 1341213, 'P2': 261410, 'P4': 252963, 'P1': 44414}
- confidence: {'MEDIUM': 1870394, 'LOW': 29606}
- inputs available (of 14): {'3': 1870394, '1': 29606}

## Target

- priority_label non-null: **0**
- is_ground_truth true rows: **0**
- observable targets: {'resolution_time_hours': 1853828, 'sla_breach_computable': 20822, 'sla_target_hours_by_source': {'nyc311': 22458}}

## Splits

- `priority_dataset`: {'train': 1330004, 'test': 284999, 'val': 284997}
- `resolution_dataset`: {'train': 718627, 'test': 153989, 'val': 153980}
- `hotspot_dataset`: {'train': 47694, 'val': 9949, 'test': 9449}
- `urbaneye_ml`: {'train': 1330004, 'test': 284999, 'val': 284997}