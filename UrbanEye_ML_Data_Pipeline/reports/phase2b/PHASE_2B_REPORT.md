# UrbanEye+ Phase 2B — ML Improvement & Retraining

All numbers below were produced by code in `scripts/phase2b/` against data on disk. Paths are relative to `UrbanEye_ML_Data_Pipeline/`.

- Every experiment, failures included, is recorded as one JSON file in `experiments/phase2b/<task>/`: 24 resolution, 18 SLA and 7 hotspot experiments. Each records:
  - dataset hash and filters
  - features and hyperparameters
  - seed and software versions
  - per-fold metrics
- Summaries are in `reports/phase2b/experiment_log_<task>.json`.
- Final gates are in `reports/phase2b/final_gates.json`.
- Error analysis is in `reports/phase2b/error_analysis.{json,md}`.

Environment: Python 3.11.8, numpy 2.2.6, pandas 2.2.3, scikit-learn 1.6.1, seed 20260915.

## 1. Objective

The goal was to improve the three UrbanEye ML tasks (resolution time, SLA breach, hotspot volume) without artificially maximising a metric. Phase 2B covered:
- domain alignment
- leak-free features
- a realistic validation design
- gated retraining

A model is saved only if it beats the strongest non-ML baseline under the same methodology, consistently and with a confidence interval that excludes zero.

Out of scope: no FastAPI, Next.js, DB schema or Edge AI change was made (verified in §14).

## 2. Phase 2A Baseline

Source: `reports/phase2a/PHASE_2A_REPORT.md`. All three v1 models still reproduce exactly; see `reports/phase2b/v1_reproduction.json` (re-run 2026-10-13).

| Model | v1 test metric (Phase 2A split) | Non-ML comparator | Phase 2A verdict |
|---|---|---|---|
| Resolution | MAE 896.5 h, median AE 133.3 h | category × department median: 920.8 h | Beats a lookup by only 2.6%. Its features are mostly US-specific. |
| SLA | PR-AUC 0.249 | subcategory rate: 0.246 | It is a lookup of NYC complaint type. Its target is NYC's agency due date, **not** UrbanEye's SLA. |
| Hotspot | MAE 10.31 | rolling 4-week mean: 10.89 (and it wins on validation) | The sign flips between periods; no consistent value. |

Weaknesses Phase 2A identified, all addressed below:
- Chicago records before February 2019 are an extraction artifact.
- 23% of Chicago targets are duplicate service requests.
- Open cases were dropped from the SLA target.
- Early stopping silently used a random hold-out.
- A single split tested on Chicago only.
- 14 of 20 resolution features cannot be produced by UrbanEye.

## 3. Data Analysis

Rebuilt tables (gitignored; regenerate with `python scripts/phase2b/build_datasets.py`, about 1 minute). The manifest with input and output SHA-256 hashes is `data/processed/phase2b/phase2b_datasets.manifest.json`.

| Table | Rows | Content |
|---|---|---|
| `incident_features_v2` | 1,900,000 | Every incident, with leak-free history features. History uses all statuses and categories, because any report occupies the queue. |
| `resolution_v2` | 725,624 | Chicago 541,440 / SF 105,381 / NYC 78,803. 271,979 rows are UrbanEye-supported categories. |
| `sla_v2` | 272,007 | UrbanEye categories only. Breach rate: Chicago 66.3% (185,988 rows), NYC 36.8% (28,958), SF 8.7% (57,061). |
| `hotspot_panel_A` | 67,092 | The Phase 2A panel rebuilt from incidents. It matches `hotspot_ml.parquet` 100% on counts and targets. |
| `hotspot_panel_C` | 247,086 | Chicago only: UrbanEye category × 1 km grid cell × week. This is the panel UrbanEye could actually build. |

**The same category behaves very differently by city** (`reports/phase2b/domain_alignment.md`):

| Category | Median resolution: Chicago / NYC / SF |
|---|---|
| STREETLIGHT | 554 h / 42 h / 69 h |
| POTHOLE | 332 h / 37 h / 115 h |
| GARBAGE | 93 h / 92 h / 2.6 h |

Category is the strongest single signal *within* a city and a weak one *across* cities.

**The UrbanEye database is not trainable** (`reports/phase2b/urbaneye_db_audit.json`, read-only session):
- 6 incidents spanning 2.6 weeks, 1 of them resolved (after 0.15 s, a test action).
- Coordinates are in Bengaluru (3) and Chennai (3).
- `sla_hours` values are 4, 12, 24 and 72. The value 12 shows admins override the default SLAs.
- There is no `resolved_at` column. Resolution exists only as `incident_history` events.
- There is no zone, ward or city column.
- A GiST index on `location` exists.

## 4. Data Quality Problems

Each fix is in code; the constants are in `scripts/phase2b/common.py`.

| Problem | Evidence | Fix |
|---|---|---|
| Chicago extraction artifact | Before 2019 there are 4–38k rows/month (55–119k afterwards), with median resolution of 845–9,071 h (80–270 h afterwards). The transition month is February 2019. | Chicago **targets** start on 2019-03-01. Earlier rows remain history for the features. Hotspot targets start 2019-05-27, so 12-week windows are clean. |
| Duplicate service requests | 23% of Chicago targets; they close when the parent closes | Excluded from targets (`valid_resolution`). They still count as queue load. |
| Instant and invalid durations | Closures under 1 minute, negative durations, over 10 years | Excluded from targets. Closures stamped *before* the report are treated as leaving the queue at the report time. |
| "Zombie" open records | 2–5% of Chicago reports are still open in the 2026 extract. An uncapped backlog grows forever and becomes a proxy for the calendar. | Backlog age is capped at 90 days (`features.workload_features`). Tested. |
| Labels unknown at training time | 8.2% of Phase 2A training labels closed after validation began | Label-availability policy (§10). A training row needs `closed_at` before the origin (resolution) or a passed deadline (SLA). |
| Open cases dropped from the SLA target | Phase 2A: about a third of breaches were missing | An open case is a breach once its deadline passed before the extract end, as in the backend. |
| Category taxonomy wrong for UrbanEye | Canonical GARBAGE_DUMPING includes 84,697 Chicago "Garbage Cart Maintenance" repairs. WATER_LEAKAGE is mostly hydrants and basements. | New raw-category mapping (§5) |
| Feature warm-up truncation | Counts near the start of a source's history are truncated | Workload features are blank for the first 30 days of each source; 90-day windows for the first 90 days |

## 5. Domain Alignment

The mapping file is `config/urbaneye_category_mapping.csv`. It maps the **raw publisher category** (and subcategory where needed), rule by rule, with a confidence of EXACT, STRONG or PARTIAL. The full table with row counts is in `reports/phase2b/domain_alignment.md`.

| UrbanEye | Status | Mapped rows | Notes |
|---|---|---|---|
| POTHOLE | Supported | 157,907 | Chicago, NYC and SF potholes: EXACT. SF pavement defects: PARTIAL. |
| STREETLIGHT | Supported | 162,330 | EXACT in all three cities |
| GARBAGE | Supported | 114,644 | Fly dumping: EXACT. Missed pick-up, bulky items, general cleaning: STRONG. Sanitation violations, NYC dirty conditions: PARTIAL. Cart maintenance, dog waste and posters are explicitly UNSUPPORTED. |
| FLOOD | Supported (with caveats) | 19,732 | NYC and SF street flooding: EXACT. Chicago "Water On Street": STRONG. Hydrants, basement water and sewer inspections are UNSUPPORTED. |
| FIRE_HAZARD | **Unsupported** | 0 | The only candidates are 710 commercial fire-*inspection* requests and 34 unknown SF requests. Neither is a citizen fire-hazard report. |
| OTHER | **Unsupported** | 0 | Its Bengaluru composition is unknown, so no US category set can stand in for it |

Other dimensions:
- **Geography:** 0 training rows fall inside any Indian bounding box.
- **Channel:** the backend's CITIZEN/CAMERA/IOT has no defensible mapping to phone/web/mobile, so it is excluded.
- **Department:** US agencies; excluded.
- **Zones:** no UrbanEye column exists, so zones are excluded. The hotspot method uses a 1 km grid computed from `location`.
- **SLA:** the UrbanEye defaults (P1 4 h, P2 24 h, P3 72 h, P4 168 h) are mirrored from `backend/app/models/incident.py`. `tests/test_phase2b.py::test_policy_mirrors_backend` fails if the backend policy drifts.

**Conclusion:** there is no evidence that any US-trained model is accurate for Indian incidents, and there is no data to test it. Every result below is a *Chicago temporal* result, plus a *cross-US-city* transfer test.

## 6. Resolution Model

### Existing model
v1: HGB regressor on log1p with the `C_log1p_deeper` configuration and 20 features, 14 of which UrbanEye cannot supply. Test MAE is 896.5 h on the Phase 2A split.

### Baseline
These are evaluated under the Phase 2B protocol (rolling origin, labels known at the origin, UrbanEye-supported rows, mean of 6 folds):

| Baseline | MAE (h) | Median AE (h) | MAE log1p |
|---|---|---|---|
| B0 global median | 626.2 | 195.1 | 1.862 |
| **B1 category median (best on MAE)** | **592.9** | 195.4 | 1.697 |
| B2/B3 category × city / department median | 595.9 | 242.1 | 1.487 |
| B4 recent 30-day category median (adaptive) | 679.1 | 313.3 | 1.461 |

### Experiments
All runs use the same 6 rolling folds and UrbanEye rows. Under the absolute-error loss on log1p, the point estimate is a conditional *median*.

| Experiment | Deployable | MAE | Median AE | MAE log1p |
|---|---|---|---|---|
| F0 Phase 2A recipe + Phase 2A config (`M0`, auto early stopping) | no | 592.5 | 169.7 | 1.501 |
| F0 Phase 2A recipe + absolute loss | no | 574.9 | 166.5 | 1.374 |
| F1 category + time | yes | 632.3 | 213.0 | 1.735 |
| F1 + squared loss | yes | 606.6 | 208.4 | 1.694 |
| F2 + 250 m density (pipeline) | yes | 628.2 | 214.2 | 1.608 |
| F3 + workload | yes | 595.1 | 181.6 | 1.616 |
| F3b workload, no month | yes | 590.1 | 181.3 | 1.605 |
| F4 workload + spatial | yes | 590.0 | 181.3 | 1.577 |
| **F4b workload + spatial, no month (absolute loss)** | **yes** | **581.7** | **172.0** | 1.508 |
| F4b, squared loss | yes | 591.2 | 173.8 | 1.624 |
| F4b, Phase 2A config | yes | 590.3 | 174.0 | 1.645 |
| F4b, ExtraTrees | yes | 586.6 | 166.2 | 1.478 |
| F4b, T1 (small) / T2 (large) tuning | yes | 583.7 / 584.9 | 175.3 / 176.5 | 1.531 / 1.565 |
| F5 + zone features | no (no zones) | 580.0 | 171.6 | 1.495 |
| F6 + city, department, channel, zone | no | 583.3 | 180.1 | 1.481 |

Findings:
- **Tuning does not help.** The best configuration is the untuned absolute-loss model, and T1/T2 are within noise.
- **ExtraTrees is close** on MAE and slightly better on median AE. It loses on mean MAE, the pre-stated criterion.
- **The non-deployable features add almost nothing.** F6 vs F4: 583 vs 590. The 2.6% Phase 2A gain was mostly category.

**Label-policy sensitivity.** F4b trained with *all* labels (including closures not yet known at the origin) scores MAE 538.2, but median AE 231.8. B1 under the same policy scores 584.0.
- The "known" policy is the realistic one: a deployed model only sees closed incidents.
- That policy biases training towards fast closures. This is a **censoring** problem, worth roughly 40 h of MAE.
- It is the largest single effect measured, larger than any feature set. Survival-style modelling is the right fix (§15).

**Leave-one-city-out** (train on two cities, test on the third, all rows):

| Predictor | MAE | Median AE | MAE log1p |
|---|---|---|---|
| Global median | 889.0 | 89.6 | 2.211 |
| Recent 30-day median | 889.4 | 72.4 | 1.675 |
| Category median | 1,023.1 | 167.8 | 2.426 |
| F1 | 1,066.9 | 131.2 | 2.303 |
| **F4b** | 899.9 | **76.6** | 1.876 |

- Category medians transfer **worse than one global median**.
- Under transfer, F4b does **not** beat the global median on MAE. It has a better median AE and log error.
- The adaptive recent-median lookup has the best log error.

### Feature engineering
Every new feature group follows one rule (`scripts/phase2b/features.py`): **a feature at report time t uses only reports before t, closures before t, and SLA outcomes whose deadline expired before t.**

| Feature (group) | Source | Why it should help | In training data | In future UrbanEye inference | Leakage control |
|---|---|---|---|---|---|
| `wl_backlog_open`, `wl_intake_7d/30d`, `wl_closed_7d`, `wl_backlog_days` (city × category) | `reported_at`, `closed_at` of other incidents | Queue length and arrival rate drive waiting time | yes | yes: `incidents.created_at` + resolve/close events in `incident_history` | Brute-force test + future-perturbation test. 90-day age cap. |
| `wl_recent_closures_30d`, `wl_recent_mean_log_res`, `wl_recent_median_res_h` | Durations of closures in [t−30d, t) | Current service speed adapts to regime shifts | yes | yes (needs history) | Only closures before t. The median is snapshotted at the UTC midnight before t. |
| `wl_city_backlog_open`, `wl_city_intake_7d` | All categories, whole city | Crew contention across categories | yes | yes, **if** UrbanEye defines a city per deployment | as above |
| `sp_samecat_box750_{7,30,90}d`, `sp_all_box750_30d` | Reports in the 3×3 block of 250 m cells (about 750 m) | Local clustering and repeat reports | yes (lat/lon) | yes: `location`, PostGIS grid (§13) | Brute-force test |
| `sp_samecat_box750_open_now` | Same-category reports in the block still open at t | Crews clear nearby clusters together; open neighbours mean a queue | yes | yes | Brute-force + future-perturbation test (added this session; it is the top feature) |
| `hist_breach_rate_{4,24,72,168}h` (SLA) | Same category, reported in [t−X−30d, t−X) | Recent breach share at this deadline | yes | yes | Uses only incidents whose deadline passed before t. Tested. |
| `zone_cat_*` | Ward × category workload | Local service capacity | yes | **no** (no zones) | Research only (F5) |
| 250 m density (pipeline) | Phase 2A geo features | — | yes | yes | Already backward-looking. Adds ~0 (F2 vs F1 on MAE). Dropped. |
| `month` | Calendar | Seasonality | yes | yes | Dropped from resolution: with under 18 months of clean Chicago data it cannot extrapolate (F3b vs F3, F4b vs F4) |

### Final model
`models/resolution_v2/`, configuration `F4b_workload_spatial_no_month__M2_l1log`:
- `HistGradientBoostingRegressor` with absolute-error loss on log1p, max_iter 500, learning rate 0.06, 63 leaves, min leaf 20, L2 1.0, early stopping off, seed 20260915.
- **Features (20):** `model_category`, `hour`, `day_of_week`, `is_weekend`, `is_night`, 10 workload features and 5 spatial features.
- **Refit data:** all 682,884 rows known at 2020-09-01, of which 262,660 are UrbanEye categories. The other categories are auxiliary rows and are never served.
- **Quantile models:** q10 and q90 boosters with the same features are saved alongside (`model.joblib` = `{point, q10, q90}`).
- **Gates** (`final_gates.json`), against B1 category median:

| Gate | Result |
|---|---|
| R1 mean MAE | 581.7 < 592.9 ✔ |
| R2 fold wins | 5 of 6 ✔ (loses 2020-01: 411.9 vs 394.1) |
| R3 median AE | 172.0 ≤ 195.4 ✔ |
| R4 pooled day-cluster bootstrap of the MAE gain | +13.7 h, 95% CI [+10.6, +17.3] ✔ |
| Reproducibility | Refitting fold 2020-07 reproduces the recorded MAE exactly |

- **Selection caveat:** the candidate was the best of 13 recorded deployable configurations on the same folds, so the margin is optimistic by construction. R2–R4 exist to require consistency on top of the mean.

### Metrics

| Fold | n | Category median MAE | v2 MAE | v2 median AE | p10–p90 coverage |
|---|---|---|---|---|---|
| 2019-09 | 17,898 | 1,146.6 | 1,145.5 | 395.1 | 0.28 |
| 2019-11 | 13,314 | 729.8 | 727.7 | 234.7 | 0.52 |
| 2020-01 | 15,159 | 394.1 | 411.9 | 85.6 | 0.58 |
| 2020-03 | 19,025 | 277.0 | 255.5 | 45.7 | 0.70 |
| 2020-05 | 22,225 | 585.5 | 557.3 | 126.7 | 0.67 |
| 2020-07 | 21,315 | 424.1 | 392.2 | 143.9 | 0.84 |
| **Mean** | | **592.9** | **581.7 (−1.9%)** | **172.0 (−12%)** | **0.60** |

**Track A** (`reports/phase2b/resolution_track_a.json`): the Phase 2A test rows (Chicago, June–August 2020), with v2 trained only on the Phase 2A training period and frozen.

| Predictor | All rows: MAE / median AE | UrbanEye rows: MAE / median AE |
|---|---|---|
| v1 model | 896.5 / 133.3 | 489.1 / 185.4 |
| v2 (F4b) | 891.3 / 136.3 | **506.6 / 256.3** |
| Category median | 939.2 / 143.9 | 485.5 / 178.8 |
| Recent 30-day median (non-ML) | **838.6 / 120.7** | 485.0 / 163.5 |

- Frozen for 4.5 months, v2 beats v1 on all rows by only 5.2 h (CI +2.2 to +8.1 h).
- On UrbanEye rows it is **worse** than v1 and than both lookups.
- **The rolling-origin gain depends on retraining every ~2 months.** A frozen v2 is not better than a lookup.

**Intervals are under-covered.** The nominal 80% p10–p90 interval covers 60% on average (28–84% by fold), and covers 0–5% for incidents taking more than 90 days. `metadata.json` records `interval_calibration.calibrated = false`, and the inference output carries `interval_calibrated: false`.

### Error analysis
Source: `error_analysis.md`, pooled held-out predictions, UrbanEye rows.

- **The error is in the tail, and neither ML nor the lookup can predict it.**
  - Rows over 30 days are 23% of rows but carry **84%** of all absolute error. Rows from 90 days to 1 year alone carry 47%.
  - In those bins the model and the category median have the same MAE (for example 3,320 vs 3,337 h). Both predict about 100 h for incidents that take months.
- **The gain is in the short and middle range.**

| True duration | v2 MAE | Baseline MAE |
|---|---|---|
| < 1 h | 91 | 147 |
| 1–3 d | 51 | 85 |
| 3–7 d | 81 | 109 |
| 1–24 h | 50 | 47 (v2 slightly worse) |

- **By category:**

| Category | v2 MAE | Baseline MAE | Note |
|---|---|---|---|
| GARBAGE | 153 | 186 | Clear win |
| POTHOLE | 631 | 651 | Win |
| STREETLIGHT | 705 | 692 | Slightly worse |
| FLOOD | 1,796 | 1,796 | Equal. Median 44 h, but an extreme tail. |

- **Systematic under-prediction.** The model's median prediction is below the true median in every fold (for example 74 vs 524 h in 2019-09). This is the censoring bias noted above.
- **Permutation importance** (last fold, Δ log1p-MAE):

| Feature | Importance |
|---|---|
| `sp_samecat_box750_open_now` | 0.190 |
| `model_category` | 0.169 |
| `wl_recent_median_res_h` | 0.056 |
| `wl_backlog_days` | 0.024 |
| `day_of_week` | 0.021 |
| `sp_samecat_box750_7d` | 0.020 |
| `wl_recent_mean_log_res` | 0.016 |
| All raw count features (`wl_intake_*`, `wl_backlog_open`, `wl_city_*`) | ≤ 0 |

  The useful signals are *local open load*, *category* and *recent service speed*.

## 7. SLA Model

### Existing model
The v1 SLA target is NYC's agency Due Date measured against `closed_date`, with open cases dropped. That is a different question from UrbanEye's SLA, and v1 needs NYC-only inputs (`sla_target_hours`, subcategory). **It cannot be scored on the UrbanEye target.** The closest like-for-like is its *recipe* retrained on the UrbanEye target (S0 below).

**The UrbanEye target (v2)** is `breach = resolution_time_hours ≥ sla_hours`, or still open past the deadline. Here `sla_hours` is the UrbanEye default for the category's default priority: FLOOD 24 h, POTHOLE and STREETLIGHT 72 h, GARBAGE 168 h. This mirrors `incident_service.py`. Default-SLA breach rates in the Chicago test windows run from 43% to 72% by fold.

### Baseline
Rolling origin, 6 Chicago folds:

| Baseline | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| Base rate | 0.622 | 0.500 | 0.259 |
| Category rate (= priority rule, since priority is fixed per category) | 0.815 | 0.791 | 0.185 |
| Category × city rate | 0.817 | 0.796 | 0.181 |
| **Recent violation rate** (same category, deadlines expired in the 30 days before t) | **0.849** | 0.801 | **0.161** |

### Experiments
All use `C1_slow` (HGB classifier, 400 iterations, learning rate 0.05, min leaf 100) unless noted.

| Experiment | Deployable | PR-AUC | Brier | ROC-AUC |
|---|---|---|---|---|
| S0 Phase 2A recipe (subcategory, department, zone, channel…), `C0` / `C1` | no | 0.889 / 0.891 | 0.152 / 0.152 | 0.861 / 0.862 |
| S1 category + time | yes | 0.844 | 0.182 | 0.802 |
| S2 + recent breach rate | yes | 0.872 | 0.160 | 0.821 |
| S3 + workload | yes | 0.866 | 0.167 | 0.812 |
| S4 + workload + spatial (`C1`; `C0` 0.877; class-balanced 0.878; small 0.880) | yes | 0.879 | 0.152 | 0.843 |
| **S4b** recent breach rate + spatial, no workload | yes | **0.884** | **0.145** | 0.850 |
| S5 + non-deployable | no | 0.892 | 0.141 | 0.864 |
| S6 multi-horizon, no spatial | yes | 0.865 | 0.161 | 0.816 |
| **S7 multi-horizon + spatial** | yes | 0.878 | 0.147 | 0.845 |

Findings:
- Workload features on their own *hurt* (S3 < S2), and spatial features help.
- **The non-deployable recipe (S0) has the best PR-AUC but worse calibration** (Brier 0.152). In fold 2020-03 its Brier is 0.250, worse than the baseline.
- **Multi-horizon (S6/S7):**
  - Each training incident is stacked at 4, 24, 72 and 168 h, with label = not resolved within X h, and `sla_hours` as a monotone-decreasing feature.
  - The training row at horizon X is used only if its deadline passed before the origin.
  - One model therefore serves *any* SLA in [4, 168] h, including admin overrides such as the 12 h seen in the DB.
  - Scored at every horizon against that horizon's own recent-violation baseline, S7 wins PR-AUC in 6 of 6 folds at all four horizons, and Brier in 5–6 of 6:

| Horizon | Breach rate | S7 PR-AUC | Baseline PR-AUC | S7 Brier | Baseline Brier |
|---|---|---|---|---|---|
| 4 h | 0.94 | 0.983 | 0.971 | 0.052 | 0.055 |
| 24 h | 0.85 | 0.947 | 0.923 | 0.112 | 0.119 |
| 72 h | 0.68 | 0.873 | 0.834 | 0.161 | 0.177 |
| 168 h | 0.51 | 0.749 | 0.719 | 0.175 | 0.188 |

**Leave-one-city-out:**

| Model | SF (PR-AUC / Brier) | NYC (PR-AUC / Brier) | Chicago (PR-AUC / Brier) | Mean (PR-AUC / Brier) |
|---|---|---|---|---|
| S7 | 0.465 / 0.059 | 0.456 / 0.233 | 0.859 / 0.188 | 0.593 / 0.160 |
| Recent violation rate | 0.433 / 0.061 | 0.427 / 0.238 | 0.870 / 0.167 | 0.577 / 0.155 |
| S4 | — | — | — | 0.533 / 0.228 |

- Under transfer the ML gain is **mixed**: S7 is better on SF and NYC, worse on Chicago, and worse on mean Brier.
- S4 transfers worse than the baseline.

### Final model
`models/sla_v2/`, configuration `S7_multihorizon_spatial__C1_slow`:
- HGB classifier with a monotonic constraint that P(breach) is non-increasing in `sla_hours`.
- **Features (23):** `ue_category`, `sla_hours`, 5 calendar features, `hist_breach_rate_own`, 10 workload features and 5 spatial features.
- **Refit data:** 1,086,703 stacked rows known at 2020-09-01.
- **Selection.** Every deployable candidate was gated against the recent violation rate:
  - S4 (all three configurations), S4b and S7 pass.
  - S4b has the best PR-AUC.
  - S7 is chosen under the tie rule: a multi-horizon model is preferred when within 0.01 PR-AUC and 0.005 Brier of the best. ΔPR-AUC is 0.006 and ΔBrier is 0.002.
  - The progress checkpoint stated the preference in advance, but the numeric tolerance was fixed in this session, after the screen.
  - S4b would also have passed. The cost of choosing S7 is 0.006 PR-AUC.
- **Gates for S7:**

| Gate | Result |
|---|---|
| S1 PR-AUC | 0.878 > 0.849 ✔ |
| S2 Brier | 0.147 < 0.161 ✔ |
| S3 fold wins | PR-AUC 6 of 6, Brier 6 of 6 ✔ |
| S4 pooled day-cluster bootstrap | ΔPR-AUC +0.020 [+0.017, +0.023]; ΔBrier +0.015 [+0.013, +0.018] ✔ |
| Reproducibility | Refitting fold 2020-07 reproduces PR-AUC exactly |

- **Threshold.** The 0.1666 from Phase 2A is not reused.
  - The new threshold is chosen by maximum F1 on the 2020-05 fold, giving **0.287**, and evaluated only on 2020-07. With no defined cost for a missed breach, this is provisional.
  - **Risk bands:** HIGH if p ≥ 0.841, the smallest cut with validation precision ≥ 0.90. MEDIUM if p ≥ 0.287. Otherwise LOW.

### Metrics
2020-07 test fold, threshold frozen beforehand:

| Predictor | PR-AUC | ROC-AUC | Brier | Precision | Recall | F1 | Flagged |
|---|---|---|---|---|---|---|---|
| Recent violation rate (its own threshold 0.340, chosen on 2020-05) | 0.900 | 0.817 | 0.151 | 0.682 | 1.000 | 0.811 | 100% |
| **S7 at 0.287** | **0.936** | **0.892** | **0.121** | 0.766 | 0.982 | 0.861 | 87% |
| S7 at 0.5 (for reference) | 0.936 | 0.892 | 0.121 | 0.835 | 0.959 | 0.893 | — |

- Confusion matrix at 0.287: TN 2,422, FP 4,363, FN 259, TP 14,272.
- **The binary flag carries little information at this base rate (68%).** The bands do:

| Band | Share of incidents | Actually breached |
|---|---|---|
| HIGH | 50% | 93% |
| MEDIUM | 38% | 55% |
| LOW | 13% | 10% |

- **Calibration** (pooled, `error_analysis.md`): predicted and observed rates agree within 0.06 in every decile. For example, 0.85 predicted vs 0.86 observed, and 0.24 vs 0.19.

### Error analysis
- **By category** (pooled):

| Category | S7 PR-AUC | Baseline PR-AUC | S7 Brier | Baseline Brier | Note |
|---|---|---|---|---|---|
| GARBAGE | 0.616 | 0.365 | 0.163 | 0.202 | Large gain |
| POTHOLE | 0.851 | 0.814 | 0.185 | 0.197 | Gain |
| STREETLIGHT | 0.955 | 0.942 | 0.061 | 0.062 | Breach rate 93% |
| FLOOD | **0.607** | **0.629** | **0.254** | **0.238** | **Worse than the baseline** |

  Where flagging is near-total (FLOOD, POTHOLE, STREETLIGHT ≥ 99% flagged at 0.287), the decision is effectively a category rule. Only GARBAGE (41% flagged) is genuinely discriminated.
- **By fold:** S7 has better PR-AUC in every fold. In 2020-03 the Brier is a tie (0.193 vs 0.193): the COVID onset, when the breach rate fell to 43% and the model over-predicted (mean predicted 0.51).
- **Permutation importance** (last fold, Δ PR-AUC):

| Feature | Importance |
|---|---|
| `hist_breach_rate_own` | 0.099 |
| `sp_samecat_box750_open_now` | 0.057 |
| `sp_samecat_box750_7d` | 0.008 |
| `sla_hours` | 0.007 (constant within the test rows, so understated) |
| `day_of_week` | 0.006 |
| All workload counts and `month` | ≈ 0 |

## 8. Hotspot Model

### Existing model
v1: HGB on log1p over the ward × canonical-category panel.

### Baseline
Persistence; rolling 4-, 8- and 12-week means; EWMA with α ∈ {0.1, 0.2, 0.3, 0.5}, with α chosen on each fold's training rows.

### Experiments
Rolling origin: 5 origins × 10 test weeks, with the model frozen at each origin and a series bootstrap.

| Experiment | Result |
|---|---|
| Track A (Phase 2A test rows) | v1 MAE 10.31 vs EWMA 10.46 vs rolling 4-week 10.89. v1's gain over EWMA has CI [−0.04, +0.33]: **not significant**. |
| Panel A, Chicago rolling (wards × canonical categories) | Poisson GBM is mixed. It loses significantly to EWMA at 2019-09 (8.25 vs 7.76). The absolute-loss GBM has lower MAE in 5 of 5 folds (0–4%; significant in 3), but worse Poisson deviance in 4 of 5. |
| Panel A, held-out wards | Same picture as the rolling test |
| Panel A, no city-wide features | Mixed; slightly worse on average (MAE 8.55 vs 8.48) |
| **Panel C, UrbanEye categories × 1 km grid, Chicago** | GBM (Poisson) MAE 1.418 vs trained EWMA 1.476 vs rolling 4-week 1.506. **GBM beats EWMA significantly in 5 of 5 folds (3.9% on average, 2–10% by fold, with better deviance).** |
| **Chicago → San Francisco transfer** | EWMA 5.11. GBM Poisson 7.79 (**+52%**). GBM absolute 6.99 (**+37%**). Both are significantly worse. |

### Final model
**Not ML.** `models/hotspot_v2/metadata.json` describes an EWMA with α = **0.3** and `min_history_weeks` = 4:
- α is chosen by one-week-ahead MAE on all clean panel C weeks: 0.1 → 1.525, 0.2 → 1.502, **0.3 → 1.499**, 0.5 → 1.524.
- On San Francisco's own counts, α = 0.3 is also best (5.11).
- **Why not ML:** within Chicago the GBM is about 4% better (2–10% by fold), which is real. But UrbanEye would apply a US-trained model to an unseen city, which is exactly the transfer case, and there it is 37–52% worse. The EWMA needs no training and recalibrates itself to each series. The earlier checkpoint described panel C as "ML ≈ EWMA"; it is a small, consistent ML win, and the decision still holds on the transfer evidence.

### Metrics
Panel C test windows with α = 0.3 fixed. This is slightly optimistic, because α was chosen on all weeks.

| Origin | EWMA MAE | Rolling 4-week MAE |
|---|---|---|
| 2019-09-02 | 1.632 | 1.693 |
| 2019-11-11 | 1.370 | 1.409 |
| 2020-01-20 | 1.373 | 1.422 |
| 2020-03-30 | 1.450 | 1.498 |
| 2020-06-08 | 1.439 | 1.505 |
| **Mean** | **1.453** | **1.506 (EWMA −3.5%)** |

### Error analysis
- The EWMA beats the rolling 4-week mean in every category and every origin.
- It over-predicts empty weeks (bias +0.83 when the next week is 0) and under-predicts bursts (bias −10.8 when the next week is over 10). A smoother cannot anticipate spikes.
- Top-decile series recall on panel C is about 0.31 for every method. Ranking next week's hottest cells is hard at this granularity.

## 9. Leakage Analysis

| Risk | Status |
|---|---|
| Query incident's own closure, status or duration | Never read by any feature. The future-perturbation tests rewrite the query's own outcome and every closure at or after t; the features do not change. |
| Other incidents' future closures | Only closures strictly before t (minute floor). Tested by brute force for workload, recent-duration stats, breach rates, spatial counts and **open_now**. |
| SLA outcome of recent incidents | Counted only once the deadline expired before t (window [t−X−30d, t−X)) |
| Backlog of never-closed records | Capped at 90 days of age; the exit time is known at t |
| Aggregates fitted on test data | Lookup baselines, encoders (`FeaturePreprocessor`), EWMA α and model fits all use training-fold rows only. The `recent_*` baselines are computed from history, not from folds. |
| Labels unknown at training time | Removed by the label-availability policy. Its effect is reported separately (§6). |
| Early stopping on a random hold-out | Disabled (`early_stopping=False`) in every v2 configuration. The Phase 2A configuration was run once for comparison. |
| Hotspot targets | Features use weeks ≤ t, and the target is t+1. Training rows need t+1 ≤ origin − 1 week. |
| Threshold and band selection | Chosen on 2020-05, evaluated only on 2020-07 |
| Model and config selection | Not leakage, but optimistic: the best of several configurations on the same folds. It is disclosed, and the gates require consistency and significance. |

No leakage was found in the deployed feature sets.

## 10. Validation Strategy

- **Resolution and SLA: rolling origin.**
  - Origins: 2019-09-01, 2019-11-01, 2020-01-01, 2020-03-01, 2020-05-01, 2020-07-01, plus the 2020-09-01 end.
  - Training data: every cleaned row of any city reported before the origin *and* whose outcome was known by then.
  - Test data: Chicago rows in the next two months, scored on UrbanEye-supported categories.
  - Training periods span NYC January–March 2010, SF January–May 2018, and Chicago March 2019 up to the origin.
  - Chicago is the only city with a clean timeline long enough to test forward in time. Test windows cover 13–22k incidents and include the COVID onset (2020-03).
- **Geographic generalisation:**
  - resolution and SLA: leave-one-city-out
  - hotspot: held-out wards (5 groups) and Chicago → San Francisco transfer
- **Like-for-like with Phase 2A (Track A):** the original test rows and the original single split, for resolution and hotspot.
- **Significance:**
  - resolution and SLA: paired bootstrap over pooled held-out predictions, clustered by report day
  - hotspot: series bootstrap
- **Thresholds:** chosen on an earlier fold than the one they are scored on.

## 11. Final Model Comparison

The same evaluation methodology is used within each row.

| Task | Baseline | Existing ML | Improved ML | Improvement | Validation |
|---|---|---|---|---|---|
| Resolution (UrbanEye rows) | Category median: MAE 592.9 h, median AE 195.4 h | Phase 2A recipe + configuration retrained per fold: 592.5 / 169.7 (non-deployable) | **v2 F4b: 581.7 / 172.0** | −11.2 h MAE (−1.9%) vs baseline; pooled +13.7 h [+10.6, +17.3]; 5 of 6 folds; median AE −12% | Rolling origin, 6 × 2 months, Chicago, labels known at the origin |
| Resolution (Track A, frozen) | Category median: 939.2 (all rows) / 485.5 (UrbanEye rows) | v1: 896.5 / 489.1 | v2: 891.3 / 506.6 | All rows +5.2 h [+2.2, +8.1]. **UrbanEye rows: worse.** The recent-median lookup (838.6) beats all of them. | Phase 2A single split, Chicago June–August 2020 |
| Resolution (cross-city) | Global median MAE 889.0 | — | v2 F4b: 899.9 | **None on MAE.** Better median AE (76.6 vs 89.6) and log error. | Leave-one-city-out |
| SLA (UrbanEye definition) | Recent violation rate: PR-AUC 0.849, Brier 0.161 | v1: not applicable (different target and inputs). Its recipe retrained: 0.889 / 0.152 (non-deployable) | **v2 S7: 0.878 / 0.147** | +0.029 PR-AUC, −0.014 Brier (−8.7%); pooled CIs exclude 0; 6 of 6 folds on both | Rolling origin, 6 × 2 months, Chicago |
| SLA (cross-city) | Recent violation rate 0.577 / 0.155 | — | S7: 0.593 / 0.160 | Mixed (2 of 3 cities on PR-AUC; worse Brier) | Leave-one-city-out |
| Hotspot (deployable grid) | Rolling 4-week mean: MAE 1.506. EWMA trained per fold: 1.476. | v2-era GBM: 1.418 | **Shipped: EWMA α 0.3 (non-ML), 1.453** | EWMA −3.5% vs rolling 4-week. ML not shipped (−3.9% in-city, but +37–52% when transferred). | Rolling origin on panel C; Chicago → SF transfer |
| Hotspot (Track A) | Rolling 4-week mean 10.89; EWMA 10.46 | v1: 10.31 | — | v1 vs EWMA not significant [−0.04, +0.33] | Phase 2A single split |

## 12. Final Feature Sets

| Model | Features | Supplied by |
|---|---|---|
| `resolution_v2` | `model_category`; `hour`, `day_of_week`, `is_weekend`, `is_night` (local time); `wl_backlog_open`, `wl_intake_7d`, `wl_intake_30d`, `wl_closed_7d`, `wl_backlog_days`, `wl_recent_closures_30d`, `wl_recent_mean_log_res`, `wl_recent_median_res_h`, `wl_city_backlog_open`, `wl_city_intake_7d`; `sp_samecat_box750_7d`, `_30d`, `_90d`, `_open_now`, `sp_all_box750_30d` | Category and `created_at` from the incident. Everything else comes from history queries (§13). Missing history becomes NaN, as in training. |
| `sla_v2` | The resolution set, plus `ue_category` (instead of `model_category`), `month`, `sla_hours` and `hist_breach_rate_own` | Same, plus the incident's `sla_hours` |
| `hotspot_v2` | That series' weekly counts (UrbanEye category × 1 km cell) | Weekly aggregation of `incidents` |

Removed relative to v1: `subcategory`, `city`, `source_dataset`, `department`, `zone_key`, `zone_type`, `report_channel`, `year` and the 250 m density features. They are either unavailable in UrbanEye or measured at about zero value.

## 13. Model Inference Contracts

The contracts are implemented and tested in `scripts/phase2b/inference.py` and `tests/test_phase2b.py`. They are not wired into FastAPI. Common rules:
- Plain dicts in and out.
- Calendar features are derived from `created_at` in **Asia/Kolkata**.
- FIRE_HAZARD and OTHER return `supported: false` with a reason and no number.

**Resolution:** `ResolutionPredictor().predict([...])`

```text
IN   {category, created_at, history: {wl_*, sp_*}}     # missing keys -> NaN
OUT  {supported, predicted_resolution_hours, interval_p10_hours, interval_p90_hours,
      interval_empirical_coverage (0.60), interval_calibrated (false), model_version, reason}
```

**SLA:** `SLAPredictor().predict([...])`

```text
IN   {category, created_at, sla_hours?=category default, history: {hist_breach_rate_own@sla_hours, wl_*, sp_*}}
OUT  {supported, breach_probability, risk_band LOW|MEDIUM|HIGH, threshold 0.287, sla_hours,
      model_version, reason}          # sla_hours outside [4,168] -> supported:false
```

**Hotspot:** `HotspotPredictor().predict([...])`

```text
IN   {category, weekly_counts: [oldest .. latest complete week]}   # >= 4 weeks
OUT  {supported, predicted_next_week_count, method "ewma", alpha 0.3, model_version, reason}
```

**History features as PostGIS/SQL.** Phase 2C must reproduce `features.py` exactly. `:t` is the incident's `created_at`, and "closed" means the first INCIDENT_RESOLVED or INCIDENT_CLOSED event in `incident_history`; that event still has to be agreed (§15). Counts use rows strictly before `:t`.

```sql
-- workload, same category (training group = city x category)
SELECT count(*) FILTER (WHERE i.created_at >= :t - interval '7 days')        AS wl_intake_7d,
       count(*) FILTER (WHERE i.created_at >= :t - interval '30 days')       AS wl_intake_30d,
       count(*) FILTER (WHERE i.created_at >= :t - interval '90 days'
                          AND (c.closed_at IS NULL OR c.closed_at >= :t))    AS wl_backlog_open
FROM incidents i LEFT JOIN incident_closure c ON c.incident_id = i.id   -- view over incident_history
WHERE i.category = :cat AND i.created_at < :t;
-- wl_closed_7d: closures with closed_at in [:t-7d, :t)
-- wl_backlog_days = wl_backlog_open / (wl_intake_30d / 30)
-- wl_recent_*: durations of closures in [:t-30d, :t), excluding duplicates and < 1 min;
--   median uses closures before the UTC midnight starting :t's day; NULL if fewer than 10
-- wl_city_*: the same without the category filter (needs a deployment/city boundary)

-- recent breach rate at horizon X (NULL if fewer than 20 incidents)
SELECT avg((c.closed_at IS NULL OR c.closed_at - i.created_at >= make_interval(hours => :X))::int)
FROM incidents i LEFT JOIN incident_closure c ON c.incident_id = i.id
WHERE i.category = :cat
  AND i.created_at >= :t - make_interval(hours => :X) - interval '30 days'
  AND i.created_at <  :t - make_interval(hours => :X);

-- spatial: 250 m cells on a fixed per-city origin (local UTM), 3x3 block around the incident's cell
--   cell = floor((ST_X(ST_Transform(location, :utm)) - :x0) / 250), same for Y
-- sp_samecat_box750_{7,30,90}d: same-category reports in the block within the window before :t
-- sp_samecat_box750_open_now:   same-category block reports before :t, not closed before :t,
--                               and less than 90 days old
-- sp_all_box750_30d:            all categories in the block, 30 days
```

Warm-up rules: workload and 7/30-day spatial features are NULL until the deployment has 30 days of history. `_90d` is NULL until it has 90 days.

## 14. Remaining Limitations

- **No Indian data.** All accuracy claims are Chicago-temporal. Cross-city tests in the US show that resolution gains vanish and SLA gains are mixed under transfer. Performance in Bengaluru is **unknown**.
- **Feature scale.** Workload and spatial counts are on a Chicago scale (about 10k reports per week). UrbanEye volumes will be orders of magnitude lower, which puts the inputs out of distribution. The most important features (`open_now`, recent service speed, backlog-days) are partly scale-free, but this is untested.
- **Resolution.** The gain is small (1.9% MAE, 12% median AE) and depends on retraining every ~2 months: frozen, it is worse than a lookup on UrbanEye rows. The long tail (84% of error) is unpredictable. Intervals are under-covered. Training is biased towards fast closures (censoring).
- **SLA.** It is clearly better than the baseline in Chicago, but:
  - FLOOD is worse than the baseline.
  - At a ~65% base rate the binary flag is close to "flag everything", so bands are the useful output.
  - The threshold is provisional until the cost of a missed breach is defined.
  - Breach rates differ hugely by city: SF 9%, NYC 37%, Chicago 66%.
- **Hotspot.** The EWMA cannot anticipate bursts. Top-decile recall is about 0.31.
- **Unsupported categories:** FIRE_HAZARD and OTHER have no model.
- **Selection optimism:** the final configurations were chosen on the same folds that gated them (disclosed in §6 and §7).
- **Artifacts are gitignored** (`.gitignore:43`, `models/**`). The v2 artifacts are not versioned in git.
- **Verification this session:**
  - Tests: 111/111 pass (90 original + 21 Phase 2B).
  - The v1 artifacts' SHA-256 hashes are identical to the earlier record, and v1 metrics reproduce exactly.
  - SHA-1 of the 118 pre-existing files under `models/`, `data/processed/`, `reports/model_training/`, `config/` and `scripts/` is identical at the start and end of this session. The earlier 147-file snapshot was in a temp directory that no longer exists.
  - The uncommitted changes outside `UrbanEye_ML_Data_Pipeline/` (`README.md`, `backend/…`, `frontend/…/ImageUploadField.tsx`) have mtimes of 2026-10-03 22:44–22:56 IST. That is before Phase 2A or 2B began (Phase 2B started 23:56 IST), and no Phase 2B edit touched them.

## 15. Phase 2C Integration Plan

1. **Settle the definitions first.**
   - Which history event closes an incident: RESOLVED or CLOSED, and how reopens count.
   - One SLA/priority policy.
   - The cost of a missed breach.
2. **Build a feature service** (SQL/PostGIS, read-only) that implements §13 exactly. Add a parity test that runs it and `features.py` on the same fixture and requires identical values.
3. **Build an inference module** in the backend that loads `models/{resolution,sla}_v2` and `hotspot_v2/metadata.json` read-only and calls the §13 contracts. Version-pin by `model_version`, and move the artifacts to versioned storage.
4. **Run resolution and SLA in shadow mode.** Log every prediction with its features and model version, show nothing to users, and evaluate against the category-median and recent-violation lookups once UrbanEye outcomes exist.
5. **Make hotspot user-facing** as a weekly EWMA job over a fixed per-city 1 km grid (≥ 4 weeks of history per cell).
6. **Promote SLA risk bands** to admin-facing advisory only after shadow evaluation shows they beat the recent-violation rate on UrbanEye data.
7. **Keep resolution internal** until (a) the shadow MAE beats the lookup, (b) a retraining cadence of ≤ 2 months is in place, and (c) intervals are recalibrated (conformal or similar).
8. **Fallbacks and monitoring.** Unsupported categories get no prediction. Monitor monthly against the lookup baselines.

### Final answers

1. **Did resolution prediction improve?** *Marginally, and only with frequent retraining.*
   - Under rolling-origin evaluation v2 beats the best non-ML baseline: MAE 581.7 vs 592.9 h (−1.9%), median AE −12%, 5 of 6 folds, pooled CI [+10.6, +17.3] h.
   - It uses only features UrbanEye can compute.
   - Frozen for 4.5 months (Track A), it is no better than a lookup on UrbanEye categories.
   - Across cities it does not beat a global median on MAE.
2. **Did SLA prediction improve?** *Yes, in Chicago, on UrbanEye's own SLA definition.*
   - PR-AUC 0.878 vs 0.849 and Brier 0.147 vs 0.161 against the recent violation rate.
   - It wins 6 of 6 folds on both metrics, at every horizon from 4 h to 168 h, and is well calibrated.
   - It is not better for FLOOD, and is mixed across cities.
   - v1 answered a different question (the NYC due date), so it is not comparable.
3. **Did hotspot prediction improve?** *Yes over the Phase 2A baseline, without ML.*
   - EWMA (α 0.3) beats the rolling 4-week mean by 3.5% on the deployable grid.
   - ML is about 4% better than EWMA within Chicago, but 37–52% worse in an unseen city, so no ML model is shipped.
4. **Which model should proceed to Phase 2C?**
   - Hotspot v2 (EWMA, non-ML) as a user-facing feature.
   - SLA v2 (S7, risk bands) through shadow mode, then admin-facing advisory.
   - Resolution v2 for shadow-mode integration only.
5. **Which model should NOT proceed yet?**
   - Resolution v2 as a user-facing estimate, and its p10–p90 interval (60% actual coverage).
   - Any hotspot ML model.
   - All v1 models: SLA v1 targets NYC's deadline, and resolution and hotspot v1 rely on features or zones UrbanEye does not have.
   - Any prediction for FIRE_HAZARD or OTHER.
6. **What data is still needed?**
   - UrbanEye incidents with outcomes: at least 500 resolved incidents per supported category and at least 26 weeks of history, with consistent resolve/close events.
   - Labelled FIRE_HAZARD and OTHER incidents.
   - A city/deployment boundary.
   - Report channel, if wanted.
   - Recorded SLA overrides.
   - The business cost of a missed breach.
   - Ideally, Indian municipal service records (e.g. BBMP grievances) to test transfer before relying on any US-trained model.
7. **What exactly must Phase 2C implement?**
   - The definitions in §15 step 1.
   - A PostGIS/SQL feature service reproducing §13, with parity tests.
   - A backend inference module that calls the tested `inference.py` contracts against the versioned v2 artifacts.
   - Prediction logging (shadow mode) with outcome joins.
   - A weekly EWMA hotspot job on a fixed 1 km grid.
   - Monitoring against lookup baselines.
   - A retraining pipeline (≤ 2-month cadence) that reuses `train_final.py`'s gates.
   - No Edge AI, and no user-facing resolution numbers, until the shadow results justify them.
