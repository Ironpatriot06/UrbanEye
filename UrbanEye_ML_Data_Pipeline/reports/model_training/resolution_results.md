# Resolution model — resolution_time_hours

Trained 2026-09-14T19:53:56.446214+00:00 · seed 20260915 · scikit-learn 1.6.1

**Dataset** `resolution_ml` · **split column** `split_global` ({'train': 659627, 'val': 141349, 'test': 141349}) · **features** 20

**Selected configuration:** `C_log1p_deeper` (chosen on validation MAE)

## Metrics

| Fold | MAE | RMSE | median_AE | R2 | p90_AE | p99_AE |
|---|---|---|---|---|---|---|
| validation — model | 734.4972 | 3,101.8649 | 84.2157 | 0.2587 | 1,151.0777 | 11,939.4248 |
| validation — baseline | 866.6051 | 3,684.0064 | 108.7511 | -0.0457 | 1,288.6085 | 14,117.1095 |
| **test — model** | 896.5125 | 3,153.7599 | 133.3216 | 0.2139 | 1,575.8869 | 17,380.6022 |
| **test — baseline** | 1,034.8018 | 3,684.1564 | 115.4539 | -0.0728 | 1,687.3052 | 18,044.7799 |

Baseline: predict the TRAINING median for every row (115.5 h).

> **Verdict:** the model beats the median baseline on MAE but LOSES on median absolute error — it wins on the tail and loses in the middle.

## Configurations tried

| Config | Question | val MAE | val median AE | val R2 |
|---|---|---|---|---|
| A_log1p_squared | does the natural shape of the target (log) model it best? | 743.9 | 83.1 | 0.2151 |
| B_raw_absolute | does optimising MAE directly beat optimising it indirectly? | 794.3 | 132.8 | 0.1987 |
| **C_log1p_deeper** | is configuration A limited by capacity? | 734.5 | 84.2 | 0.2587 |

## Test metrics by city

| City | model MAE | baseline MAE | model median AE | baseline median AE | n |
|---|---|---|---|---|---|
| Chicago | 896.5 | 1,034.8 | 133.3 | 115.5 | 141,349 |

## Features

`category`, `subcategory`, `city`, `source_dataset`, `department`, `zone_key`, `zone_type`, `report_channel`, `hour`, `day_of_week`, `month`, `year`, `is_weekend`, `is_night`, `nearby_similar_incidents_24h`, `nearby_similar_incidents_7d`, `nearby_similar_incidents_30d`, `local_incident_density`, `category_incident_density`, `hours_since_previous_similar_incident`

## Preprocessing

- fitted on: training fold only
- categorical NULL -> explicit __MISSING__ level
- levels outside the train top-250 -> __OTHER__
- levels unseen at fit time -> -1

## Notes

- split column: split_global — multi-source table (3 sources) — pooled evaluation on one wall clock

- Selection used validation only. The test fold was scored once, with the already-chosen configuration. Per-city numbers are reported, not used to select.
