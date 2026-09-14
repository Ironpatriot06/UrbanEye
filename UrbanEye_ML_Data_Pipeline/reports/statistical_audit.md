# UrbanEye+ statistical audit

Generated 2026-09-14T19:51:51.791413+00:00

| Severity | Meaning | Count |
|---|---|---|
| `safe` | no action needed | 0 |
| `needs_preprocessing` | usable, with the documented treatment | 40 |
| `dangerous` | will produce a misleading model if used as-is | 1 |
| `policy_decision` | a human must decide; the pipeline must not decide for them | 2 |

Nothing is dropped automatically. A `dangerous` finding is a statement about how results must be reported, not an instruction to delete a column.

## dangerous (1)

| Table | Column | Issue | Detail | Action |
|---|---|---|---|---|
| corpus | `resolution_time_hours` | regime change across splits | median resolution hours by split: {'test': 198.1, 'train': 187.4, 'val': 82.5} — the validation window covers the first COVID wave | do not present a single val/test score as an estimate of steady-state performance; report per-period metrics and say which period they cover |

## policy_decision (2)

| Table | Column | Issue | Detail | Action |
|---|---|---|---|---|
| corpus | `category` | large unmapped / out-of-scope population | 45.5% of rows are UNMAPPED, OUT_OF_SCOPE or REVIEW_REQUIRED | extend config/category_mapping.csv — see reports/category_mapping_candidates.csv. This is a taxonomy decision, not one the pipeline may take on its own |
| corpus | `reported_at` | sources cover wildly different time spans | span in days: {'chicago311': 789, 'nyc311': 67, 'sf311': 129} | the cities are not exchangeable; model per city or accept that `city` carries most of the signal |

## needs_preprocessing (40)

| Table | Column | Issue | Detail | Action |
|---|---|---|---|---|
| resolution_ml | `local_incident_density` | high cardinality | 1,317 distinct values | hashing or per-group target encoding FITTED ON TRAIN ONLY; never plain label-encode into a numeric feature |
| resolution_ml | `category_incident_density` | high cardinality | 8,317 distinct values | hashing or per-group target encoding FITTED ON TRAIN ONLY; never plain label-encode into a numeric feature |
| resolution_ml | `hours_since_previous_similar_incident` | high cardinality | 402,776 distinct values | hashing or per-group target encoding FITTED ON TRAIN ONLY; never plain label-encode into a numeric feature |
| resolution_ml | `subcategory` | missingness is a perfect source_dataset indicator | missing share by source_dataset: {'chicago311': 1.0, 'nyc311': 0.0, 'sf311': 0.0} | do not read importance for this column as anything but 'source_dataset'; add an explicit is_missing flag and consider per-source_dataset models |
| resolution_ml | `category` | distribution drift across splits | total-variation 0.224 (threshold 0.2) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| resolution_ml | `department` | distribution drift across splits | total-variation 0.210 (threshold 0.2) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| resolution_ml | `month` | distribution drift across splits | PSI 7.084 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| resolution_ml | `local_incident_density` | distribution drift across splits | PSI 0.278 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| sla_ml | `sla_target_hours` | high cardinality | 4,307 distinct values | hashing or per-group target encoding FITTED ON TRAIN ONLY; never plain label-encode into a numeric feature |
| sla_ml | `hours_since_previous_similar_incident` | high cardinality | 10,991 distinct values | hashing or per-group target encoding FITTED ON TRAIN ONLY; never plain label-encode into a numeric feature |
| sla_ml | `subcategory` | distribution drift across splits | total-variation 0.219 (threshold 0.2) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| sla_ml | `sla_target_hours` | distribution drift across splits | PSI 0.851 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| sla_ml | `local_incident_density` | distribution drift across splits | PSI 0.289 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| hotspot_ml | `rolling_12w_count` | high cardinality | 1,475 distinct values | hashing or per-group target encoding FITTED ON TRAIN ONLY; never plain label-encode into a numeric feature |
| hotspot_ml | `month` | distribution drift across splits | PSI 6.655 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| hotspot_ml | `year` | distribution drift across splits | PSI 2.059 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| hotspot_ml | `week_of_year` | distribution drift across splits | PSI 7.684 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| hotspot_ml | `incident_count` | distribution drift across splits | PSI 0.541 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| hotspot_ml | `previous_period_count` | distribution drift across splits | PSI 0.504 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| hotspot_ml | `rolling_4w_count` | distribution drift across splits | PSI 0.810 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| hotspot_ml | `rolling_12w_count` | distribution drift across splits | PSI 1.007 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| hotspot_ml | `rolling_4w_mean` | distribution drift across splits | PSI 0.769 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| hotspot_ml | `trend_4w` | distribution drift across splits | PSI 0.352 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| duplicate_ml | `distance_meters` | high cardinality | 295,057 distinct values | hashing or per-group target encoding FITTED ON TRAIN ONLY; never plain label-encode into a numeric feature |
| duplicate_ml | `time_difference_hours` | high cardinality | 385,513 distinct values | hashing or per-group target encoding FITTED ON TRAIN ONLY; never plain label-encode into a numeric feature |
| duplicate_ml | `time_difference_hours` | distribution drift across splits | PSI 2.746 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| priority_features | `local_incident_density` | high cardinality | 1,555 distinct values | hashing or per-group target encoding FITTED ON TRAIN ONLY; never plain label-encode into a numeric feature |
| priority_features | `category_incident_density` | high cardinality | 9,046 distinct values | hashing or per-group target encoding FITTED ON TRAIN ONLY; never plain label-encode into a numeric feature |
| priority_features | `hours_since_previous_similar_incident` | high cardinality | 653,464 distinct values | hashing or per-group target encoding FITTED ON TRAIN ONLY; never plain label-encode into a numeric feature |
| priority_features | `subcategory` | missingness is a perfect source_dataset indicator | missing share by source_dataset: {'chicago311': 1.0, 'nyc311': 0.0, 'sf311': 0.0} | do not read importance for this column as anything but 'source_dataset'; add an explicit is_missing flag and consider per-source_dataset models |
| priority_features | `month` | distribution drift across splits | PSI 7.714 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| priority_features | `local_incident_density` | distribution drift across splits | PSI 0.292 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| urbaneye_ml | `local_incident_density` | high cardinality | 1,555 distinct values | hashing or per-group target encoding FITTED ON TRAIN ONLY; never plain label-encode into a numeric feature |
| urbaneye_ml | `category_incident_density` | high cardinality | 9,046 distinct values | hashing or per-group target encoding FITTED ON TRAIN ONLY; never plain label-encode into a numeric feature |
| urbaneye_ml | `hours_since_previous_similar_incident` | high cardinality | 653,464 distinct values | hashing or per-group target encoding FITTED ON TRAIN ONLY; never plain label-encode into a numeric feature |
| urbaneye_ml | `subcategory` | missingness is a perfect source_dataset indicator | missing share by source_dataset: {'chicago311': 1.0, 'nyc311': 0.0, 'sf311': 0.0} | do not read importance for this column as anything but 'source_dataset'; add an explicit is_missing flag and consider per-source_dataset models |
| urbaneye_ml | `department` | distribution drift across splits | total-variation 0.205 (threshold 0.2) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| urbaneye_ml | `month` | distribution drift across splits | PSI 7.714 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| urbaneye_ml | `local_incident_density` | distribution drift across splits | PSI 0.292 (threshold 0.25) | expected under a chronological split — report per-split metrics and do not treat a single pooled score as an estimate of future performance |
| corpus | `source_dataset` | one source dominates the corpus | chicago311 is 84.2% of all rows | report per-source metrics; a pooled score is a Chicago score |

## Corpus

- source balance: {'chicago311': 1600000, 'sf311': 200000, 'nyc311': 100000}
- temporal coverage: {'chicago311': ('2018-07-01', '2020-08-28'), 'nyc311': ('2010-01-01', '2010-03-09'), 'sf311': ('2018-01-01', '2018-05-10')}
- median resolution hours by split (dominant source): {'test': 198.09, 'train': 187.42, 'val': 82.51}
