# UrbanEye+ Phase 2A — ML Baseline Evaluation

All numbers below come from code run during this phase against the data and model files on disk. Nothing was retrained. Paths are relative to `UrbanEye_ML_Data_Pipeline/` unless stated.

## 1. Executive Summary

- **Every reported metric reproduced exactly**, to 4 decimals. I loaded the saved `models/<task>/*.joblib` files and scored them on the saved `data/processed/ml/*.parquet` files. The environment is identical to the one used for training: Python 3.11.8, numpy 2.2.6, pandas 2.2.3, sklearn 1.6.1.
- **The existing ML tests pass: 90/90.** SHA-1 checksums of `models/`, `data/processed/`, `reports/`, `config/` and `scripts/` (147 files) were identical before and after.
- **Resolution model.** It beats the training-median baseline on MAE by 13.4% (bootstrap 95% CI for the gain: +134 to +143 h). It loses on median absolute error by 17.9 h (CI: −19.5 to −15.8 h). Both effects are real, not noise. However, **a simple lookup of the training median per category × department, with no ML, gets within 2.6% of the model's MAE** (920.8 h vs 896.5 h). Category alone accounts for most of what the model uses.
- **SLA model.** Test PR-AUC is 0.249 against a base rate of 0.104. **A lookup table of training breach rate per subcategory scores PR-AUC 0.246**, so the model adds roughly nothing over that table. The target also systematically leaves out the cases that were still open (likely breaches): including them would move the breach rate from 10.5% to ≤16.5%.
- **Hotspot model.** It loses to the rolling 4-week mean on validation and wins on test. Both differences are statistically clear in a bootstrap over series, so the effect **flips sign between periods**. That is evidence of regime dependence, not of a better model. In addition, 44% of the Chicago training panel comes from a period whose weekly volume is about 17× lower than normal. In that period 61.5% of targets are zero, versus 4.5% afterwards.
- **Compatibility with the backend.** No training row is Indian. Of the 6 backend categories, 2 map cleanly (POTHOLE, STREETLIGHT), 2 map partially (GARBAGE, FLOOD) and 2 have no training data at all (FIRE_HAZARD, OTHER). Of the 20 resolution features, the backend can supply 6 today. SLA means something different in the training data and in the backend.
- **New issues the previous audit did not report:**
  - Chicago records from before February 2019 look like an extraction artifact.
  - 23% of Chicago resolution targets are duplicate service requests.
  - Open cases are excluded from the SLA target.
  - Early stopping is silently active and uses a *random* 10% hold-out.
  - The 250 m density features add **no** measurable value to any model: their permutation importance is ≤ 0.

## 2. Existing ML Architecture

Every stage below was checked against the code, not just the docs.

| Stage | Exact file(s) | Output |
|---|---|---|
| **Data** | `scripts/download/download_{chicago,sf,nyc,boston}311.py` (Socrata; Chicago `$order=sr_number`) | `data/raw/chicago311/chicago311.jsonl` (1,600,000 rows), `nyc311/nyc311_2010_2019.jsonl` (100k), `sf311/sf311_cases.jsonl` (200k). Boston: `data/manifests/boston311.FAILED.json`. **No Chicago manifest exists** (only `sf311.json` and `nyc311.json`). |
| **Preprocessing** | `scripts/preprocess/preprocess_{chicago,sf,nyc}311.py`, `_tabular.py`, `utils/cleaning.py`, `utils/mapping.py` + `config/category_mapping.csv`, `utils/schema.py` (provenance guard) → `build_incidents.py` | `data/processed/incidents/{city}.parquet` → `all_incidents.parquet` (1.9M rows, 41 columns) |
| **Feature engineering** | `build_time_features.py` + `utils/timefeatures.py` (local-time features)<br>`build_geo_features.py` + `geo_providers.py` (250 m backward density; 23 POI columns all NULL)<br>`build_resolution_dataset.py` (target, `sla_breach`, merges time + density)<br>`build_hotspot_dataset.py` (weekly panel, lags)<br>`features/build_task_datasets.py` (per-task tables + manifests; adds `zone_key`) | `features/`, `geo_features/`, `resolution/`, `hotspot/`, `ml/{resolution,sla,hotspot}_ml.parquet` + `.manifest.json` |
| **Train/val/test split** | `scripts/preprocess/_ml_common.py::chronological_split / dual_chronological_split` writes both `split` (per source) and `split_global`<br>`scripts/train/_common.py::choose_split_column` picks one at training time | Columns in each task table |
| **Model training** | `scripts/train/train_{resolution,sla,hotspot}.py` + `_common.py::FeaturePreprocessor`; orchestrated by `run_all.py` | — |
| **Evaluation** | Inside each `train_*.py` (via `_common.regression_metrics / classification_metrics`)<br>Separately `scripts/baselines/run_baselines.py` → `reports/baseline_models.*`<br>`scripts/validation/*` (leakage, provenance, audits) | `reports/model_training/*.{json,md}` (**tracked in git**) |
| **Model artifact** | `_common.save_model` | `models/<task>/{model,preprocessor}.joblib`, `metadata.json` (gitignored by root `.gitignore:43`; `data/processed/**` by `:31`) |

**Operational warning:** `train_*.py` and `run_all.py` call `save_model()` and `write_report()` unconditionally. Running any of them overwrites the artifacts and the tracked reports. That is why this phase did **not** execute them.

**Inference:** there is no `predict()` layer. `_common.load_model()` exists; nothing calls it outside the tests.

## 3. Resolution-Time Model

### Dataset
- `data/processed/ml/resolution_ml.parquet` has 942,325 rows, 27 columns, and 0 duplicate `incident_id`s.
- Rows by city: Chicago 749,053 (79.5%), SF 114,469, NYC 78,803.
- Population: in-scope categories, closed, with a measurable duration.
- 173,346 instant closures (< 1 minute) and open cases are excluded.

### Features
There are 20 features, taken from the manifest and confirmed in `preprocessor.joblib`.

| Group | Features |
|---|---|
| Categorical (10) | `category`, `subcategory`, `city`, `source_dataset`, `department`, `zone_key`, `zone_type`, `report_channel`, `is_weekend`, `is_night` |
| Numeric (10) | `hour`, `day_of_week`, `month`, `year`, `nearby_similar_incidents_{24h,7d,30d}`, `local_incident_density`, `category_incident_density`, `hours_since_previous_similar_incident` |
| Geographic | `city`, `zone_key` / `zone_type` (Chicago ward, SF neighbourhood, NYC community board). Raw lat/lon is not used. |
| Temporal | `hour`, `day_of_week`, `month`, `year`, `is_weekend`, `is_night`, all in city-local time |
| Historical | The 5 density features plus recency. All are strictly backward (`reported_at < t`), within 250 m, over 24 h / 7 d / 30 d windows |

Preprocessing:
- Ordinal encoding, fitted on the training fold only.
- NULL becomes `__MISSING__`.
- Levels outside the top 250 become `__OTHER__`; unseen levels become −1.
- The booster's native categorical support is used.

### Target
- `resolution_time_hours = closed_at − reported_at`, in hours.
- Sub-minute durations, negative durations and durations over 87,600 h are set to NULL.
- The model is fit on `log1p(y)`; predictions are `clip(expm1(ŷ), 0)`.

Distribution:

| Statistic | Value |
|---|---|
| p25 | 20.2 h |
| p50 | 116 h |
| p75 | 707 h |
| p95 | 6,197 h |
| p99 | 18,609 h |
| Max | 71,015 h (8.1 years) |
| Share over 1 year | 3.06% |

Median by city: Chicago 166 h, NYC 38.6 h, SF 20.2 h.

Missing values:

| Column | Missing |
|---|---|
| `subcategory` | 79.5% (100% for Chicago) |
| `hours_since_previous_similar_incident` | 16.0% |
| Density features | 1.5% |
| `zone_key` | 0.48% |

### Training
- **Split:** `split_global`, a single chronological cut on one clock.

| Fold | Rows | Date range | Cities |
|---|---|---|---|
| Train | 659,627 | 2010-01-01 → 2020-01-17 | All three |
| Val | 141,349 | 2020-01-17 → 2020-06-04 | **100% Chicago** |
| Test | 141,349 | 2020-06-04 → 2020-08-28 | **100% Chicago** |

  ROAD_DAMAGE and PUBLIC_INFRASTRUCTURE_DAMAGE never appear in val or test.
- **Model:** `HistGradientBoostingRegressor`, configuration `C_log1p_deeper`:
  - `loss=squared_error`, `max_iter=500`, `learning_rate=0.06`
  - `max_leaf_nodes=63`, `min_samples_leaf=20`, `l2_regularization=1.0`
  - `random_state=20260915`
- **Selection:** the best of three configurations (A, B, C) on validation MAE.
- **Undocumented:** `early_stopping='auto'` is active and uses a **random** 10% of the training fold. It used all 500 iterations (`n_iter_=500`), so it never stopped and may be capacity-capped.

### Evaluation
Metrics are MAE, RMSE, median AE, R², p90 AE and p99 AE. The baseline predicts the training median, 115.49 h.

### Reproduced Results

| Fold | MAE | RMSE | Median AE | R² | p90 AE |
|---|---|---|---|---|---|
| Val, model | 734.50 | 3,101.86 | 84.22 | 0.2587 | 1,151.1 |
| Val, median baseline | 866.61 | 3,684.01 | 108.75 | −0.0457 | 1,288.6 |
| **Test, model** | **896.51** | **3,153.76** | **133.32** | **0.2139** | 1,575.9 |
| Test, median baseline | 1,034.80 | 3,684.16 | 115.45 | −0.0728 | 1,687.3 |

All values match `reports/model_training/resolution_results.md` exactly.

### Baseline Comparison: is the improvement meaningful?

| Test predictor | MAE | Median AE | R² |
|---|---|---|---|
| Training median (existing baseline) | 1,034.8 | **115.5** | −0.073 |
| Median per category (no ML; added in this audit) | 939.2 | 143.9 | 0.140 |
| Median per category × department (no ML; added) | 920.8 | 145.9 | 0.185 |
| **Model** | **896.5** | 133.3 | **0.214** |

- **Fact:** the model beats the global median by 138 h (CI +134 to +143 h). It beats the category × department lookup by only **24 h (2.6%)**.
- **Fact:** in log space, the scale the model was trained on, the gain is larger. Mean |Δlog1p| is 1.42 for the model versus 1.97 for the baseline (−28%).
- **Fact:** permutation importance on test (Δ log-MAE):

| Feature | Importance |
|---|---|
| `category` | 0.707 |
| `report_channel` | 0.166 |
| `hour` | 0.088 |
| `zone_key` | 0.047 |
| `hours_since_previous_similar_incident` | 0.020 |
| All 5 density/count features | ≤ 0 |
| `city`, `year`, `source_dataset`, `subcategory` | 0 (constant or NULL in the Chicago-only test fold) |

**Why median AE is worse even though MAE is better:**
1. **Most of the MAE gain is in the tails, not the middle.**
   - Errors on test rows with true duration over 90 days (8% of rows) make up **69%** of the model's total absolute error. Rows over 1 year (2.8%) make up 46%.
   - The model wins clearly on fast-closing categories: <1 h and 1–24 h rows (26% of test) have MAE 55 vs 115 and 53 vs 104. Examples: GRAFFITI (median 19.8 h; MAE 17.8 vs 95.6), SIGNAGE, TRAFFIC_SIGNAL, WATER.
2. **It loses in the 1–30 day middle.** Rows of 3–7 days: median AE 66.5 vs 24.5. Rows of 7–30 days: true median 336 h, but the model's median prediction is 151 h.
3. **The test distribution shifted upward.**
   - True medians: train 115.5 h, val 73.7 h, test **187.1 h**.
   - Chicago monthly medians rose from 62 h (April 2020) to 238 h (August 2020).
   - The model's median prediction is 112.6 h. It learned the faster-closing period that just preceded the test fold, and `year` is useless for this: in training, `year=2020` covers only 1–17 January.
   - The constant 115 h baseline happens to sit closer to the new middle.
4. **The model squeezes its predictions toward the middle.** Predicted p90 is 430 h versus a true p90 of 1,803 h. It almost never predicts the long tail.
5. Per category, the model is worse than the global median on median AE for **POTHOLE** (646 vs 394 h) and DRAINAGE_SEWER (521 vs 416 h).

### Problems
- **Chicago records before February 2019 (likely an artifact).**
  - **Fact:** Chicago from July 2018 to January 2019 has 4–20k rows/month, versus 55–119k afterwards. Those rows have median resolution of **3,745–9,071 h**, versus about 150–270 h afterwards.
  - In `resolution_ml` that is 23,702 training rows.
  - **Inference:** these are probably legacy or migrated service requests (the download is ordered by `sr_number` and capped at exactly 1.6M), not a representative sample. **Unknown:** the true cause.
- **Duplicates in the target.** 23.4% of Chicago target rows have `is_duplicate=True`: train 26.0%, val 19.9%, test 18.1%. Their "resolution" follows the parent request's closure. Median duration: 221 h for duplicates versus 147 h for the rest.
- **Late closures.** Closures run up to 2026-09 (Chicago), 2024 (SF) and 2016 (NYC). 8.2% of training labels resolve *after* validation starts. Long-tail labels include administrative closures made years later.
- **City and year are confounded.** NYC is only 2010, SF only 2018 and Chicago only 2018–2020.
- **Exact duplicates.** 286 rows are identical on every feature and the target. 1,383 share the same timestamp, category, zone and target.
- **Inconsistent docs.** `ML_READINESS.md` quotes two different test results (876.5 / 127.7 in §2 versus 896.5 / 133.3 in §8). The §2 numbers come from `run_baselines.py`, not the saved model.

## 4. SLA Model

### Dataset
- `sla_ml.parquet` has 20,213 rows, **all NYC**, all reported 2010-01-01 to 2010-03-09.
- Categories: ROAD_DAMAGE 9,673; FALLEN_TREE 2,858; SIGNAGE 2,754; FOOTPATH 2,554; PUBLIC_INFRA 2,318; GRAFFITI 56.
- Departments: DOT 17,299; Parks 2,858; NYPD 56.

### Features
There are 17 features: the resolution features minus `city`, `source_dataset`, `zone_type` and `year` (constant, so dropped), plus `sla_target_hours`.
- Categorical: `category`, `subcategory`, `department`, `zone_key`, `report_channel`, `is_weekend`, `is_night`.
- Numeric: the rest.
- Missing values: density 4.2%, recency 37.8%.

### Target
- `sla_breach = resolution_time_hours > sla_target_hours`, where `sla_target_hours = NYC Due Date − Created Date`.
- `sla_target_hours`: median 240 h, p10 192 h, p90 1,106 h, minimum 8 h, 4,307 distinct values.
- Positives are 10.5% overall: train 11.2%, val 7.8%, test 10.4%.
- **Open cases are excluded.** 1,437 in-scope NYC cases with a due date were dropped because they were still open (status `open`, `closed_at` NULL). Their due dates are in 2010, so by the dataset's own definition they missed the deadline.
  - Counting them as breaches would raise the rate from 10.5% to **≤ 16.5%**.
  - So roughly a third of the true breaches are missing from the target.

### Training
- **Split:** per-source chronological `split`.

| Fold | Rows | Dates (2010) |
|---|---|---|
| Train | 13,209 | 1 Jan – 21 Feb |
| Val | 3,131 | 21 Feb – 1 Mar (**8 days**) |
| Test | 3,873 | 1 Mar – 9 Mar (**8 days**) |

- **Model:** `HistGradientBoostingClassifier`, configuration `A_default`:
  - `max_iter=200`, `learning_rate=0.1`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=0`
  - Early stopping (random 10% of training) stopped it at **54 iterations**.
- **Selection:** best validation PR-AUC.

### Evaluation
- **Threshold 0.16663**: the threshold that maximises F1 on validation, then frozen.
- **Baseline:** the training positive rate (0.1120).

### Reproduced Results

| Fold | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|
| Val, model | 0.2786 | 0.7554 | 0.199 | 0.430 | 0.272 | 0.0661 |
| **Test, model** | **0.2489** | **0.6829** | **0.220** | **0.361** | **0.274** | 0.0903 |
| Test, base-rate baseline | 0.1038 | 0.500 | 0 | 0 | 0 | 0.0931 |

Test confusion matrix at threshold 0.1666: TN 2,958, FP 513, FN 257, TP 145. 17.0% of cases are flagged.

**Metrics at other test thresholds** (descriptive only; no threshold is being selected):

| Threshold | Precision | Recall | F1 | TP / FP / FN |
|---|---|---|---|---|
| 0.05 | 0.154 | 0.697 | 0.252 | 280 / 1,537 / 122 |
| 0.10 | 0.179 | 0.500 | 0.263 | 201 / 925 / 201 |
| 0.1666 | 0.220 | 0.361 | 0.274 | 145 / 513 / 257 |
| 0.20 | 0.249 | 0.323 | 0.281 | 130 / 392 / 272 |
| 0.30 | 0.314 | 0.219 | 0.258 | 88 / 192 / 314 |
| 0.50 | 0.474 | 0.092 | 0.154 | 37 / 41 / 365 |

**What 0.1666 means:** a case is flagged "will breach" when its predicted probability is at least 16.7%. That is about 1.5× the training base rate. At this threshold about 1 in 5 flagged cases actually breaches, and about 1 in 3 actual breaches is caught. Calibration is weak above 0.4: the [0.4, 0.5) bin has an observed rate of 0.18.

### Baseline Comparison

| Test predictor | PR-AUC | ROC-AUC |
|---|---|---|
| Base rate | 0.104 | 0.500 |
| Breach rate per category (no ML; added in this audit) | 0.122 | 0.548 |
| Breach rate per `sla_target_hours` value (no ML; added) | 0.139 | 0.594 |
| **Breach rate per subcategory (no ML; added)** | **0.246** | **0.675** |
| Model | 0.249 | 0.683 |

Permutation importance (Δ PR-AUC):

| Feature | Importance |
|---|---|
| `subcategory` | 0.131 |
| `zone_key` | 0.037 |
| `sla_target_hours` | 0.021 |
| `day_of_week` | 0.017 |
| Density features | ≈ 0 or negative |
| `hour` | −0.015 |

**Fact:** the model is essentially a learned lookup of NYC complaint type (subcategory).

### Problems
- One city, one quarter, and an 8-day test window.
- The target leaves out the open cases most likely to be breaches.
- The model's main signal is NYC's own complaint-type vocabulary, which UrbanEye does not have.
- `sla_target_hours` encodes NYC agency policy: the median deadline is 240 h. **UrbanEye's deadlines are 4, 24, 72 and 168 h**, so 4 h and 24 h are below anything in the training data except its minimum.
- **Not usable for the UrbanEye SLA definition.** The backend counts a breach when the resolve or close time is at or after `created_at + sla_hours(priority)`, with hours of 4, 24, 72 or 168 (`backend/app/models/incident.py:190-195`; `incident_service.py:119-148, 493-497`). The ML target is the NYC agency's Due Date measured against NYC `closed_date`. These are different deadlines, a different "closed" event, and different categories.

## 5. Hotspot Model

### Dataset
- `hotspot_ml.parquet` has 67,092 rows: Chicago 55,500, SF 5,808, NYC 5,784.
- Each row is one series (city × zone × category) for one ISO week, on a complete weekly grid.
- Series count: Chicago 500 (50 wards × 10 categories), SF 363, NYC 723.
- No duplicate keys.

### Features
13 features:
- `city`, `zone_key`, `zone_type`, `category`
- `month`, `year`, `week_of_year`
- `incident_count` (week t)
- `previous_period_count`, `rolling_4w_count`, `rolling_12w_count`, `rolling_4w_mean`, `trend_4w` (all built with `shift(1)`)

2.36% of rows have NULL lag features (the first week of each series).

### Target
- `future_incident_count`: the count in week t+1 for the same series.
- Mean 14.8, median 5, p99 116, max 1,211.
- 20.0% of targets are zero (train 26.3%, val 7.0%, test 3.3%).
- **Unit of prediction:** a zone × category × week count, not an incident.
- **Granularity:** US ward / neighbourhood / community board, weekly.

### Training
- **Split:** `split_global`.

| Fold | Rows | Weeks | Cities |
|---|---|---|---|
| Train | 47,092 | 2010-01-04 → 2019-11-04 | All three |
| Val | 10,000 | 2019-11-11 → 2020-03-23 (20 weeks) | Chicago only |
| Test | 10,000 | 2020-03-30 → 2020-08-10 (20 weeks; COVID onset) | Chicago only |

- **Model:** `HistGradientBoostingRegressor`, configuration `C_log1p_squared`:
  - `max_iter=300`, `learning_rate=0.1`, `max_leaf_nodes=31`, `min_samples_leaf=20`
  - Early stopping (random 10% of training rows from a time-series panel) stopped it at **143 iterations**.
- **Baselines:**
  - Persistence
  - Rolling 4-week mean over t−3..t (the primary baseline)
  - Lagged rolling mean over t−4..t−1

### Evaluation
Metrics are MAE, RMSE, median AE, R² and Poisson deviance.

### Reproduced Results

| Fold | Predictor | MAE | RMSE | Median AE | R² |
|---|---|---|---|---|---|
| Val | Model | 7.4571 | 14.220 | 3.52 | 0.550 |
| Val | **Rolling 4-week mean** | **6.9779** | **13.013** | 3.50 | **0.623** |
| Val | Persistence | 7.9257 | 15.129 | 4.00 | 0.491 |
| Test | **Model** | **10.3145** | 32.413 | **4.37** | 0.308 |
| Test | Rolling 4-week mean | 10.8850 | **31.649** | 5.00 | **0.340** |
| Test | Persistence | 12.6231 | 37.105 | 5.00 | 0.093 |

All values match `hotspot_results.md` exactly.

### Baseline Comparison
Bootstrap of the MAE gain over the rolling mean, resampling by series:
- **Validation:** CI −0.67 to −0.30. The model loses in 14 of 20 weeks.
- **Test:** CI +0.39 to +0.74. The model wins in 16 of 20 weeks.

Both results are statistically clear and point in **opposite directions**. On test, the model also has **worse RMSE and R²** than the rolling mean. **Conclusion from the evidence:** the model has not shown that it adds predictive value over the rolling 4-week mean. Which one wins depends on the period: stable winter versus the 2020 shock.

### Problems
- **The Chicago 2018 artifact contaminates the panel.**
  - 15,500 of the 35,500 Chicago training rows (44%) are weeks before February 2019.
  - In-scope volume then was about 602 per week, versus about 10,585 per week from March to October 2019.
  - Zero targets were 61.5% versus 4.5%.
  - The builder's docstring says the zeros "are observed facts". For Chicago before 2019 that claim is contradicted by the volume figures above.
- Val and test are Chicago only.
- Zones are US administrative units. NYC and SF each contribute only 8–16 training weeks.
- Early stopping takes a random hold-out from the training panel, so neighbouring weeks of the same series land on both sides.
- `year` cannot generalise forward in time.

## 6. Data Quality

| Check | Resolution | SLA | Hotspot |
|---|---|---|---|
| Rows | 942,325 | 20,213 | 67,092 |
| Duplicate keys | 0 ids; 286 identical feature + target rows | 0 | 0 |
| Target | Very right-skewed; 3.1% over 1 year; max 8.1 years | 10.5% positive (biased down by excluding open cases) | 20% zero; heavy tail (max 1,211) |
| Missing values | `subcategory` 79.5%; recency 16%; density 1.5% | Recency 37.8%; density 4.2% | Lag features 2.4% |
| Geography | Chicago 79.5% / SF 12.1% / NYC 8.4% | NYC 100% | Chicago 82.7% |
| Time coverage | NYC Jan–Mar 2010; SF Jan–May 2018; Chicago Jul 2018–Aug 2020 | 9 weeks in 2010 | Same as resolution, weekly |
| Suspicious data | Chicago before February 2019; duplicate SRs 23% of Chicago; closures years later | 1,437 open cases dropped | Chicago 2018 weeks about 17× too sparse |
| Leakage | No feature leakage found. Training labels closing after the cut: 8.2% | None in the features | Features end at or before t. The series list uses the whole period (0 series start after the training cut, so no effect here) |

**What the raw data does and does not contain:**
- **The raw downloads are capped subsets:** exactly 1.6M Chicago rows, 100k NYC and 200k SF.
- **Category mapping:** 18.9% of rows are UNMAPPED and 26% are OUT_OF_SCOPE (from the pipeline's own reports).
- **Synthetic negatives:** none in these three tasks. They exist only in the untrained duplicate task, where they make up its whole negative class.

## 7. UrbanEye ↔ ML Compatibility

The backend facts come from the working tree: `backend/app/models/incident.py`, `services/incident_service.py`, `api/incidents.py` and `frontend/src/app/dashboard/page.tsx`.

| Dimension | ML pipeline | UrbanEye backend | Status |
|---|---|---|---|
| Categories | 12 categories that actually have data (out of 15 canonical; 0 rows for OPEN_MANHOLE, CONSTRUCTION_OBSTRUCTION, OTHER) | POTHOLE, FLOOD, FIRE_HAZARD, GARBAGE, STREETLIGHT, OTHER | POTHOLE ✔︎<br>STREETLIGHT ≈ STREETLIGHT_FAULT ✔︎<br>GARBAGE ≈ GARBAGE_DUMPING (partial)<br>FLOOD ≈ WATER_LEAKAGE / DRAINAGE (partial: US complaints, not floods)<br>**FIRE_HAZARD: no data**<br>**OTHER: no data** |
| Geography | Chicago, SF, NYC | Bengaluru coordinates (tests and placeholders); no city field | ✘ No Indian rows |
| Incident source | `report_channel` phone / web / mobile / other / … | `source` CITIZEN / CAMERA / IOT; frontend always sends CITIZEN | ✘ Different vocabulary; lossy mapping |
| SLA definition | Agency Due Date (median 240 h) vs `closed_date`; open cases excluded | 4 / 24 / 72 / 168 h by priority; breach if resolved/closed ≥ deadline; open cases count as BREACHED once past the deadline | ✘ Different deadline, event and population |
| Priority | DRAFT config: SLA 30 / 50 / 72 / 90 h, 15 categories, density weights | Fixed per category: FIRE P1, FLOOD P2, POTHOLE / STREETLIGHT / OTHER P3, GARBAGE P4; admin override | ✘ Two inconsistent policies |
| Timestamps | `reported_at`, `closed_at` | `created_at` only. Resolved / closed times exist **only** as `incident_history` rows (`INCIDENT_RESOLVED` / `INCIDENT_CLOSED`) | ⚠︎ The target can be rebuilt from history; reopen semantics need defining |
| Spatial | 250 m radius counts | `location GEOGRAPHY(POINT,4326)` + lat/lon; GiST index probably exists (GeoAlchemy2 default; not checked in the live DB); **no ST_* queries anywhere** | ⚠︎ Possible with PostGIS; not built |
| Zones | Ward / neighbourhood / community board | No ward, zone, city or district column | ✘ |
| History needed | 30 days of nearby history (density); ≥ 12 weeks per zone × category (hotspot) | Demo script creates 0 incidents | ✘ No history yet |

**Model features the backend cannot currently produce from an incident:**

| Feature | What the model expects | Does UrbanEye store the data? | Could PostGIS / SQL compute it? | What it would eventually need |
|---|---|---|---|---|
| `subcategory` | Publisher's second-level label | No | No | A product decision; otherwise it is always `__MISSING__` (as for Chicago) |
| `city` / `source_dataset` | "Chicago", "chicago311", … | No | Partly (from coordinates) | Values would be unseen (encoded −1). The model has never seen India. |
| `department` | US agency name | No | No | A department/agency model in the backend |
| `zone_key`, `zone_type` | `City:ward` string | No | Yes, with point-in-polygon against BBMP ward polygons (not loaded) | Ward boundary layer; the values would still be unseen |
| `report_channel` | phone / web / mobile … | Only CITIZEN / CAMERA / IOT | No | Capture the channel in the frontend |
| `nearby_similar_incidents_{24h,7d,30d}` | Count of same-category reports within 250 m in [t−w, t) | Yes (`location`, `category`, `created_at`) | Yes: `ST_DWithin(location, :pt, 250)` plus time and category filters | A query or service, and real history. Measured value in current models: **≤ 0** |
| `local_incident_density`, `category_incident_density` | All-category count within 250 m over 30 days, and the same-category share | Yes | Yes | Same as above |
| `hours_since_previous_similar_incident` | Gap to the previous same-category report within 250 m / 30 days | Yes | Yes (ORDER BY created_at DESC LIMIT 1) | Same as above |
| `sla_target_hours` (SLA model) | NYC agency deadline | `sla_hours` exists, with different semantics | — | A different target; it cannot be swapped in |
| Hotspot lags (`incident_count`, rolling counts, trend) | Weekly counts per zone × category | Raw incidents yes; zones no | Yes, once zones exist | Zones plus ≥ 12 weeks of history |
| `hour`, `day_of_week`, `month`, `is_weekend`, `is_night` | City-local time | Yes (`created_at`) | Yes | Convert to Asia/Kolkata |
| `year` | 2010–2020 | Yes | Yes | 2026 is outside the training range; the trees treat it as the last bin |

The backend can supply 6 of the 20 resolution features today: `category` (after mapping) and the 5 time features. With a PostGIS query layer, the 6 history features become possible.

## 8. Generalization Analysis

**FACT**
- The resolution and hotspot validation and test folds are 100% Chicago, and their periods are later than training on a single clock. The SLA test fold is 8 days of NYC, March 2010.
- The test folds are **not geographically independent**. The same Chicago wards and categories appear in training, and the same NYC zones in the SLA data.
- They are temporally independent: chronological, with no feature leakage found. But 8.2% of resolution training labels became known only after validation starts.
- No Indian or Bengaluru incidents exist anywhere in the corpus. FIRE_HAZARD and OTHER have zero rows. FLOOD has no direct equivalent.
- The main signals are US-specific vocabularies:
  - resolution: `category` 0.71, `report_channel` 0.17
  - SLA: NYC `subcategory` 0.13
- `city` and `year` are confounded.

**INFERENCE**
- Test metrics estimate *within-Chicago, next-few-months* performance. They do not estimate performance in a new city.
- Most of what the resolution model knows is that certain US categories close fast or slow. Whether Bengaluru's categories show the same ordering is not measured.

**UNKNOWN**
- How these models would perform on any Indian incident. There is no data to test it.
- The real cause of the Chicago 2018 anomaly.
- Whether the backend database has the GiST index.

## 9. Integration Blockers

1. **No inference layer.** There is no `predict()`, no service and no API endpoint.
2. **Category mismatch.** FIRE_HAZARD and OTHER have no training data; FLOOD and GARBAGE are only partial matches.
3. **Features the backend cannot supply.** 14 of the 20 resolution features: `subcategory`, `department`, `city`, `source_dataset`, `zone_key`, `zone_type`, `report_channel`, `year`, plus the 6 history features, which have no query layer yet.
4. **Different SLA meaning.** The SLA model's target and its `sla_target_hours` input mean something different from the backend's SLA.
5. **No zones.** The hotspot model needs a zone layer and a weekly panel with ≥ 12 weeks of history. Neither exists.
6. **No `resolved_at` column.** It must be derived from `incident_history`, and the backend has no historical incidents.
7. **Artifacts are not versioned.** They are gitignored, and the training scripts overwrite them in place.
8. **Two priority/SLA policies disagree.** The ML config uses 30 / 50 / 72 / 90 h; the backend uses 4 / 24 / 72 / 168 h.

## 10. Phase 2B Retraining Plan

### Resolution
- **Is retraining justified?** Only after the data is fixed. Retraining with the same setup would reproduce a model that beats a lookup table by 2.6%.
- **Data:**
  - Investigate, and probably exclude, Chicago before February 2019.
  - Decide how to treat `is_duplicate` rows (exclude them, or model duplicate closure separately).
  - Decide how to handle administrative closures made years later. Options: a censoring horizon, or survival analysis.
- **Features:**
  - Restrict to features UrbanEye can supply: category (mapped), the time features, and optionally the PostGIS history features.
  - Drop `year`, `city`, `source_dataset`, `department`, `subcategory` and `report_channel`, or record each as an explicit, justified exception.
- **Evaluation:**
  - Rolling-origin (multiple-cut) temporal evaluation.
  - Leave-one-city-out, to measure transfer to an unseen city.
  - Report results per category.
  - Set `early_stopping` explicitly, using a temporal hold-out.
- **Baselines:** global median, category median, and category × department median.
- **Metrics:** MAE, median AE, MAE in log space, coverage of P50/P90 quantile intervals, and errors by duration bin.

### SLA
- **Is retraining justified?** Not on the current target. It does not answer UrbanEye's question.
- **Data:**
  - Count open cases that are past their deadline as breaches. This matches the backend's definition.
  - Add Boston (TARGET_DT) to get a second city.
  - Better option: reframe as P(resolution > X h) for X in {4, 24, 72, 168}, using the resolution data from every city.
- **Features:** the same restricted set as the resolution model. No `sla_target_hours` from NYC policy.
- **Evaluation:** rolling-origin, more than 8 days of test data, and leave-one-city-out.
- **Baseline:** breach rate per (sub)category.
- **Metrics:** PR-AUC against both the base rate and the lookup table, a calibration curve, Brier score, and precision/recall at several thresholds. No threshold should be chosen until the cost of a missed breach is defined.

### Hotspot
- **Is retraining justified?** No, not before the baseline has been adopted and the data fixed.
- **Data:**
  - Remove or flag the sparse Chicago weeks from 2018.
  - Do not build a pooled panel in which val and test are Chicago only.
- **Evaluation:** rolling-origin across several regimes; per-week win rate; a bootstrap over series.
- **Baselines:** rolling 4-week mean (adopt it as the shipped method now), rolling 8- and 12-week means, and a seasonal naive forecast.
- **Metrics:** MAE, RMSE, Poisson deviance, and top-k zone recall if the product question is "which zones next week".

## 11. Exact Recommended Order of Work

1. Agree on a single SLA/priority policy, backend versus `priority_config.yaml`. This is a product decision.
2. Define how the 6 backend categories map to the 15 ML categories. Decide FIRE_HAZARD and OTHER explicitly: no model, or a fallback.
3. Investigate the Chicago records before February 2019 (raw `sr_number` / `created_date`) and decide whether to exclude them. Do the same for duplicate SRs.
4. Fix the SLA target: open-and-past-deadline cases count as breaches; consider the X-hour reframing.
5. Add a non-overwriting evaluation harness: rolling-origin, leave-one-city-out, lookup-table baselines, explicit early stopping. Use it to evaluate the *existing* models first.
6. Restrict the feature sets to what UrbanEye can supply, and measure how much accuracy that costs.
7. Only then retrain resolution and SLA. Ship the rolling-mean hotspot method as a non-ML statistic.
8. Separately, decide on the backend data needed for future in-domain training. The backend already logs `resolved_at` through history; it would also need a ward/zone and the channel.

### Summary Table

| Model | Existing model | Existing metric (test) | Baseline | Main issue | Next action |
|---|---|---|---|---|---|
| Resolution | HGB regressor, log1p, `C_log1p_deeper` | MAE 896.5 h · median AE 133.3 h · R² 0.214 | Global median: 1,034.8 / 115.5 · **category × department median: 920.8 / 145.9** | Gain over a lookup is 2.6%; Chicago-only test; 2018 artifact; duplicates in the target | Clean the data, restrict features, rolling and cross-city evaluation |
| SLA | HGB classifier, threshold 0.1666 | PR-AUC 0.249 · ROC 0.683 · F1 0.274 | Base rate 0.104 · **subcategory rate 0.246** | NYC-only, 8-day test, open cases excluded, NYC SLA semantics | Redefine the target to match UrbanEye's SLA |
| Hotspot | HGB regressor, log1p, `C_log1p_squared` | MAE 10.31 · RMSE 32.41 | **Rolling 4-week mean: 10.89 / 31.65** (wins on val 6.98 vs 7.46) | Sign flips between periods; sparse Chicago 2018 weeks; US zones | Ship the rolling mean; clean the panel; rolling-origin evaluation |

---

**Explicit statements**

1. **Successfully reproduced:** every validation and test metric of all three saved models, exactly. Also the confusion matrix, the threshold (0.16663) and the existing baselines. The tests pass 90/90.
2. **Could not be reproduced:**
   - The **retraining** itself was not attempted, because `train_*.py` overwrite artifacts and tracked reports.
   - The §2–§4 numbers in `ML_READINESS.md` (e.g. resolution test MAE 876.5, SLA PR-AUC 0.260, hotspot GBM 10.54) do not match the saved models. They come from a separate `run_baselines.py` run, which was not executed here because it writes to `reports/`.
   - The cause of the Chicago anomaly cannot be established from the processed data.
3. **Data problems:**
   - Chicago records before February 2019 (sparse, extreme durations, 44% of the Chicago hotspot training panel).
   - 23% duplicate SRs in the Chicago target.
   - Open SLA cases excluded (breach rate biased from ≤ 16.5% down to 10.5%).
   - Closures years after the report.
   - City and year confounded.
   - Capped raw subsets; no Chicago manifest.
   - No Indian data; FIRE_HAZARD and OTHER have no data.
4. **Model problems:**
   - Resolution ≈ a category lookup, it squeezes the tail, and it is worse at the median.
   - SLA ≈ a subcategory lookup.
   - Hotspot does not beat the rolling mean consistently.
   - Early stopping is silently active with a random hold-out.
   - The density features carry no value.
5. **Must be fixed before retraining:**
   - Agree on SLA/priority semantics.
   - Define the category mapping.
   - Rule on the Chicago 2018 records and on duplicate SRs.
   - Fix the SLA target.
   - Build a non-overwriting evaluation harness with lookup baselines and cross-city / rolling evaluation.
   - Restrict to features UrbanEye can produce.
6. **Phase 2B:** steps 3–7 of §11, in that order.
