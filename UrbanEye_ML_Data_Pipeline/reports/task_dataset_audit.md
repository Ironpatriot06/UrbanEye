# UrbanEye+ task dataset & leakage audit

Generated 2026-09-14T12:38:52.167417+00:00

**78/80 checks passed, 0 failed, 2 skipped.**

## Tasks

| Task | Rows | Target | Predictors | Split |
|---|---|---|---|---|
| `resolution_ml` | 1,026,596 | resolution_time_hours | 20 | per-source chronological |
| `sla_ml` | 20,224 | sla_breach | 17 | per-source chronological |
| `hotspot_ml` | 67,092 | future_incident_count | 13 | per-city chronological on week_start |
| `duplicate_ml` | 455,460 | same_incident | 3 | connected-component chronological |
| `priority_features` | 1,900,000 | **none** | 19 | per-source chronological |

## Checks

| ID | Task | Check | Result | Detail |
|---|---|---|---|---|
| TASK-1 | resolution_ml | every declared predictor exists in the table | PASS | missing=[] |
| TASK-2 | resolution_ml | no declared predictor is constant | PASS | constant=[] |
| TASK-3 | resolution_ml | the target is declared | PASS | target=resolution_time_hours |
| TASK-4 | resolution_ml | the target column exists and has values | PASS | rows_with_target=1026596 |
| TL-6 | resolution_ml | every predictor states when its value becomes known | PASS | undeclared=[] |
| TL-1 | resolution_ml | no predictor is the target or a derivative of it | PASS | found=[] |
| TL-2 | resolution_ml | no post-resolution column is a predictor | PASS | found=[] |
| TL-12 | resolution_ml | target-support columns are not offered to the wrong task | PASS | found=[] |
| TL-3 | resolution_ml | the priority policy output is not a predictor | PASS | found=[] |
| TL-4 | resolution_ml | raw coordinates are not predictors | PASS | found=[] |
| TL-5 | resolution_ml | no predictor is a global frequency/target encoding | PASS | named=[] detected=[]; structural co-linearity (not leakage): ['year is constant within subcategory (4 values / 248 levels)'] |
| TL-7 | resolution_ml | joined history features match the feature table exactly | PASS | mismatched=[] over 50,000 sampled rows |
| TL-11 | resolution_ml | train, val and test are all non-empty | PASS | present=['test', 'train', 'val'] |
| TL-8 | resolution_ml | no entity appears in more than one split | PASS | entities_in_two_splits=0 |
| TL-11b | resolution_ml | splits are chronological within each source_dataset | PASS |  |
| TL-10 | resolution_ml | source-specific columns carry a documented caveat | PASS | undocumented=[] |
| TASK-1 | sla_ml | every declared predictor exists in the table | PASS | missing=[] |
| TASK-2 | sla_ml | no declared predictor is constant | PASS | constant=[] |
| TASK-3 | sla_ml | the target is declared | PASS | target=sla_breach |
| TASK-4 | sla_ml | the target column exists and has values | PASS | rows_with_target=20224 |
| TL-6 | sla_ml | every predictor states when its value becomes known | PASS | undeclared=[] |
| TL-1 | sla_ml | no predictor is the target or a derivative of it | PASS | found=[] |
| TL-2 | sla_ml | no post-resolution column is a predictor | PASS | found=[] |
| TL-12 | sla_ml | target-support columns are not offered to the wrong task | PASS | found=[] |
| TL-3 | sla_ml | the priority policy output is not a predictor | PASS | found=[] |
| TL-4 | sla_ml | raw coordinates are not predictors | PASS | found=[] |
| TL-5 | sla_ml | no predictor is a global frequency/target encoding | PASS | named=[] detected=[] |
| TL-7 | sla_ml | joined history features match the feature table exactly | PASS | mismatched=[] over 20,224 sampled rows |
| TL-11 | sla_ml | train, val and test are all non-empty | PASS | present=['test', 'train', 'val'] |
| TL-8 | sla_ml | no entity appears in more than one split | PASS | entities_in_two_splits=0 |
| TL-11b | sla_ml | splits are chronological within each source_dataset | PASS |  |
| TL-10 | sla_ml | source-specific columns carry a documented caveat | PASS | undocumented=[] |
| TASK-1 | hotspot_ml | every declared predictor exists in the table | PASS | missing=[] |
| TASK-2 | hotspot_ml | no declared predictor is constant | PASS | constant=[] |
| TASK-3 | hotspot_ml | the target is declared | PASS | target=future_incident_count |
| TASK-4 | hotspot_ml | the target column exists and has values | PASS | rows_with_target=67092 |
| TL-6 | hotspot_ml | every predictor states when its value becomes known | PASS | undeclared=[] |
| TL-1 | hotspot_ml | no predictor is the target or a derivative of it | PASS | found=[] |
| TL-2 | hotspot_ml | no post-resolution column is a predictor | PASS | found=[] |
| TL-12 | hotspot_ml | target-support columns are not offered to the wrong task | PASS | found=[] |
| TL-3 | hotspot_ml | the priority policy output is not a predictor | PASS | found=[] |
| TL-4 | hotspot_ml | raw coordinates are not predictors | PASS | found=[] |
| TL-5 | hotspot_ml | no predictor is a global frequency/target encoding | PASS | named=[] detected=[] |
| TL-7 | hotspot_ml | joined history features match the feature table exactly | _skip_ | no joinable history columns |
| TL-11 | hotspot_ml | train, val and test are all non-empty | PASS | present=['test', 'train', 'val'] |
| TL-11b | hotspot_ml | splits are chronological within each city | PASS |  |
| TL-10 | hotspot_ml | source-specific columns carry a documented caveat | PASS | undocumented=[] |
| TASK-1 | duplicate_ml | every declared predictor exists in the table | PASS | missing=[] |
| TASK-2 | duplicate_ml | no declared predictor is constant | PASS | constant=[] |
| TASK-3 | duplicate_ml | the target is declared | PASS | target=same_incident |
| TASK-4 | duplicate_ml | the target column exists and has values | PASS | rows_with_target=455460 |
| TL-6 | duplicate_ml | every predictor states when its value becomes known | PASS | undeclared=[] |
| TL-1 | duplicate_ml | no predictor is the target or a derivative of it | PASS | found=[] |
| TL-2 | duplicate_ml | no post-resolution column is a predictor | PASS | found=[] |
| TL-12 | duplicate_ml | target-support columns are not offered to the wrong task | PASS | found=[] |
| TL-3 | duplicate_ml | the priority policy output is not a predictor | PASS | found=[] |
| TL-4 | duplicate_ml | raw coordinates are not predictors | PASS | found=[] |
| TL-5 | duplicate_ml | no predictor is a global frequency/target encoding | PASS | named=[] detected=[] |
| TL-7 | duplicate_ml | joined history features match the feature table exactly | _skip_ | no joinable history columns |
| TL-11 | duplicate_ml | train, val and test are all non-empty | PASS | present=['test', 'train', 'val'] |
| TL-8 | duplicate_ml | no entity appears in more than one split | PASS | entities_in_two_splits=0 |
| TASK-1 | priority_features | every declared predictor exists in the table | PASS | missing=[] |
| TASK-2 | priority_features | no declared predictor is constant | PASS | constant=[] |
| TASK-3 | priority_features | the target is declared | PASS | target=None |
| TL-6 | priority_features | every predictor states when its value becomes known | PASS | undeclared=[] |
| TL-1 | priority_features | no predictor is the target or a derivative of it | PASS | found=[] |
| TL-2 | priority_features | no post-resolution column is a predictor | PASS | found=[] |
| TL-12 | priority_features | target-support columns are not offered to the wrong task | PASS | found=[] |
| TL-3 | priority_features | the priority policy output is not a predictor | PASS | found=[] |
| TL-4 | priority_features | raw coordinates are not predictors | PASS | found=[] |
| TL-5 | priority_features | no predictor is a global frequency/target encoding | PASS | named=[] detected=[] |
| TL-7 | priority_features | joined history features match the feature table exactly | PASS | mismatched=[] over 50,000 sampled rows |
| TL-11 | priority_features | train, val and test are all non-empty | PASS | present=['test', 'train', 'val'] |
| TL-8 | priority_features | no entity appears in more than one split | PASS | entities_in_two_splits=0 |
| TL-11b | priority_features | splits are chronological within each source_dataset | PASS |  |
| TL-10 | priority_features | source-specific columns carry a documented caveat | PASS | undocumented=[] |
| TL-9 | hotspot_ml | every consecutive panel row is exactly one calendar week apart | PASS | non_adjacent_rows=0 of 65,506 |
| TL-9b | hotspot_ml | the target can express a quiet week (it contains zeros) | PASS | zero_share=0.1997 |
| TL-9c | hotspot_ml | the binary target is not constant | PASS | distinct=2 |
| TL-13 | priority_features | priority has no supervised target | PASS |  |

## Feature / target audit

| Task | Column | Role | Null % | Distinct | Known at prediction time | When |
|---|---|---|---|---|---|---|
| resolution_ml | `category` | predictor | 0.0 | 12 | True | citizen selects it in the app at submission |
| resolution_ml | `subcategory` | predictor | 80.48 | 344 | True | publisher's second-level label, assigned at intake |
| resolution_ml | `city` | predictor | 0.0 | 3 | True | known from the coordinate / the deployment |
| resolution_ml | `source_dataset` | predictor | 0.0 | 3 | True | known: which system the report arrived through |
| resolution_ml | `department` | predictor | 0.0 | 121 | True | assigned by the 311 routing rules at intake. If a municipality assigns it at closure instead, it must be dropped for tha |
| resolution_ml | `zone_key` | predictor | 0.544 | 168 | True | administrative unit of the report location, resolved at intake |
| resolution_ml | `zone_type` | predictor | 0.0 | 3 | True | names which administrative unit zone_key refers to |
| resolution_ml | `report_channel` | predictor | 0.0 | 7 | True | the channel the citizen used, known at submission |
| resolution_ml | `hour` | predictor | 0.0 | 24 | True | function of reported_at alone |
| resolution_ml | `day_of_week` | predictor | 0.0 | 7 | True | function of reported_at alone |
| resolution_ml | `month` | predictor | 0.0 | 12 | True | function of reported_at alone |
| resolution_ml | `year` | predictor | 0.0 | 4 | True | function of reported_at alone |
| resolution_ml | `is_weekend` | predictor | 0.0 | 2 | True | function of reported_at alone |
| resolution_ml | `is_night` | predictor | 0.0 | 2 | True | function of reported_at alone |
| resolution_ml | `nearby_similar_incidents_24h` | predictor | 1.719 | 111 | True | counts only reports strictly earlier than this one |
| resolution_ml | `nearby_similar_incidents_7d` | predictor | 1.719 | 319 | True | counts only reports strictly earlier than this one |
| resolution_ml | `nearby_similar_incidents_30d` | predictor | 1.719 | 626 | True | counts only reports strictly earlier than this one |
| resolution_ml | `local_incident_density` | predictor | 1.719 | 1317 | True | counts only reports strictly earlier than this one |
| resolution_ml | `category_incident_density` | predictor | 1.719 | 8409 | True | ratio of two strictly-backward counts |
| resolution_ml | `hours_since_previous_similar_incident` | predictor | 15.542 | 420424 | True | gap to the most recent strictly earlier report |
| resolution_ml | `resolution_time_hours` | target | None | None | False | observed only after the fact — this is the label |
| sla_ml | `category` | predictor | 0.0 | 6 | True | citizen selects it in the app at submission |
| sla_ml | `subcategory` | predictor | 0.0 | 90 | True | publisher's second-level label, assigned at intake |
| sla_ml | `department` | predictor | 0.0 | 4 | True | assigned by the 311 routing rules at intake. If a municipality assigns it at closure instead, it must be dropped for tha |
| sla_ml | `zone_key` | predictor | 0.0 | 75 | True | administrative unit of the report location, resolved at intake |
| sla_ml | `report_channel` | predictor | 0.0 | 3 | True | the channel the citizen used, known at submission |
| sla_ml | `hour` | predictor | 0.0 | 24 | True | function of reported_at alone |
| sla_ml | `day_of_week` | predictor | 0.0 | 7 | True | function of reported_at alone |
| sla_ml | `month` | predictor | 0.0 | 3 | True | function of reported_at alone |
| sla_ml | `is_weekend` | predictor | 0.0 | 2 | True | function of reported_at alone |
| sla_ml | `is_night` | predictor | 0.0 | 2 | True | function of reported_at alone |
| sla_ml | `sla_target_hours` | predictor | 0.0 | 4318 | True | the deadline, set by the publisher at intake. Only legitimate for the SLA task, where the question is whether this known |
| sla_ml | `nearby_similar_incidents_24h` | predictor | 4.193 | 22 | True | counts only reports strictly earlier than this one |
| sla_ml | `nearby_similar_incidents_7d` | predictor | 4.193 | 34 | True | counts only reports strictly earlier than this one |
| sla_ml | `nearby_similar_incidents_30d` | predictor | 4.193 | 46 | True | counts only reports strictly earlier than this one |
| sla_ml | `local_incident_density` | predictor | 4.193 | 173 | True | counts only reports strictly earlier than this one |
| sla_ml | `category_incident_density` | predictor | 4.193 | 890 | True | ratio of two strictly-backward counts |
| sla_ml | `hours_since_previous_similar_incident` | predictor | 37.816 | 10992 | True | gap to the most recent strictly earlier report |
| sla_ml | `sla_breach` | target | None | None | False | observed only after the fact — this is the label |
| hotspot_ml | `city` | predictor | 0.0 | 3 | True | known from the coordinate / the deployment |
| hotspot_ml | `zone_key` | predictor | 0.0 | 168 | True | administrative unit of the report location, resolved at intake |
| hotspot_ml | `zone_type` | predictor | 0.0 | 3 | True | names which administrative unit zone_key refers to |
| hotspot_ml | `category` | predictor | 0.0 | 12 | True | citizen selects it in the app at submission |
| hotspot_ml | `month` | predictor | 0.0 | 12 | True | function of reported_at alone |
| hotspot_ml | `year` | predictor | 0.0 | 4 | True | function of reported_at alone |
| hotspot_ml | `week_of_year` | predictor | 0.0 | 52 | True | function of the period being forecast |
| hotspot_ml | `incident_count` | predictor | 0.0 | 331 | True | the CURRENT week's own count. Legitimate ONLY under the stated deployment assumption that the forecast for week t+1 is p |
| hotspot_ml | `previous_period_count` | predictor | 2.364 | 300 | True | count in week t-1; shift(1) before any rolling |
| hotspot_ml | `rolling_4w_count` | predictor | 2.364 | 720 | True | weeks t-4..t-1 |
| hotspot_ml | `rolling_12w_count` | predictor | 2.364 | 1475 | True | weeks t-12..t-1 |
| hotspot_ml | `rolling_4w_mean` | predictor | 2.364 | 852 | True | weeks t-4..t-1 |
| hotspot_ml | `trend_4w` | predictor | 2.364 | 841 | True | difference of two backward windows |
| hotspot_ml | `future_incident_count` | target | None | None | False | observed only after the fact — this is the label |
| duplicate_ml | `distance_meters` | predictor | 0.233 | 295057 | True | computed from the two reports being compared |
| duplicate_ml | `time_difference_hours` | predictor | 0.0 | 385513 | True | computed from the two reports being compared |
| duplicate_ml | `category_match` | predictor | 0.0 | 2 | True | computed from the two reports being compared |
| duplicate_ml | `same_incident` | target | None | None | False | observed only after the fact — this is the label |
| priority_features | `category` | predictor | 0.0 | 15 | True | citizen selects it in the app at submission |
| priority_features | `subcategory` | predictor | 84.211 | 584 | True | publisher's second-level label, assigned at intake |
| priority_features | `city` | predictor | 0.0 | 3 | True | known from the coordinate / the deployment |
| priority_features | `source_dataset` | predictor | 0.0 | 3 | True | known: which system the report arrived through |
| priority_features | `zone_key` | predictor | 0.766 | 169 | True | administrative unit of the report location, resolved at intake |
| priority_features | `zone_type` | predictor | 0.0 | 3 | True | names which administrative unit zone_key refers to |
| priority_features | `report_channel` | predictor | 0.0 | 7 | True | the channel the citizen used, known at submission |
| priority_features | `hour` | predictor | 0.0 | 24 | True | function of reported_at alone |
| priority_features | `day_of_week` | predictor | 0.0 | 7 | True | function of reported_at alone |
| priority_features | `month` | predictor | 0.0 | 12 | True | function of reported_at alone |
| priority_features | `year` | predictor | 0.0 | 4 | True | function of reported_at alone |
| priority_features | `is_weekend` | predictor | 0.0 | 2 | True | function of reported_at alone |
| priority_features | `is_night` | predictor | 0.0 | 2 | True | function of reported_at alone |
| priority_features | `nearby_similar_incidents_24h` | predictor | 1.558 | 190 | True | counts only reports strictly earlier than this one |
| priority_features | `nearby_similar_incidents_7d` | predictor | 1.558 | 524 | True | counts only reports strictly earlier than this one |
| priority_features | `nearby_similar_incidents_30d` | predictor | 1.558 | 824 | True | counts only reports strictly earlier than this one |
| priority_features | `local_incident_density` | predictor | 1.558 | 1555 | True | counts only reports strictly earlier than this one |
| priority_features | `category_incident_density` | predictor | 1.558 | 9046 | True | ratio of two strictly-backward counts |
| priority_features | `hours_since_previous_similar_incident` | predictor | 10.394 | 653464 | True | gap to the most recent strictly earlier report |