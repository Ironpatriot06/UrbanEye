# UrbanEye+ — ML readiness, task by task

What can be modelled with this corpus, what cannot, and what the numbers
actually say. Every figure here was produced by running the pipeline and the
baselines on the real data in `data/raw/`; nothing is projected.

Generated artefacts behind this document:

| Artefact | What it holds |
|---|---|
| `reports/task_dataset_audit.{json,md}` | per-task feature/target audit + 12 leakage routes |
| `reports/statistical_audit.{json,md}` | imbalance, drift, degenerate columns, classified |
| `reports/baseline_models.{json,md}` | baseline metrics vs naive baselines |
| `data/processed/ml/*.manifest.json` | per-task column roles, availability, caveats |

---

## 1. Verdicts

| Task | Dataset | Rows | Target | Verdict |
|---|---|---|---|---|
| Resolution time | `resolution_ml.parquet` | 942,325 | `resolution_time_hours` | **CONDITIONALLY READY** — trained; wins on MAE, loses on median AE |
| SLA breach | `sla_ml.parquet` | 20,213 | `sla_breach` | **CONDITIONALLY READY** — trained; 2.4x lift over the base rate |
| Hotspot | `hotspot_ml.parquet` | 67,092 | `future_incident_count` | **CONDITIONALLY READY** — trained; still does not stably beat a 4-week moving average |
| Duplicate detection | `duplicate_ml.parquet` | 455,460 | `same_incident` | **NOT READY for evaluation** |
| Priority | `priority_features.parquet` | 1,900,000 | none | **NOT READY — and not a supervised task** |

---

## 2. Resolution time — CONDITIONALLY READY

**Target** `resolution_time_hours`, **942,325** closed in-scope cases with a
*measurable* duration. Median 116 h, p99 ~18,600 h. Fit on `log1p`.

### The instant-closure artifact, and how it is handled

8.2% of the old target was not a service duration at all. Chicago closed 18,279
cases at exactly 7 seconds, 18,125 at 6 s and 14,578 at 5 s — a transactional
write, not field work — and NYC's entire sub-minute population sat at exactly
0 s (7,640 rows, with just 5 rows anywhere in the 1–60 s range).

These rows are now excluded from the target by a rule grounded in the data, not
a chosen threshold. One minute is the coarsest timestamp granularity in the
sources (90.8% of SF `requested_datetime` and 57.7% of NYC `created_date` land
exactly on :00 seconds), so a sub-minute duration is **below what the source can
express**. See `cleaning.resolution_time.min_observable_seconds`.

Nothing is deleted. All 1,900,000 incidents remain; 173,346 of them keep their
`status`, `closed_at` and a new `resolution_instant_closure = True` flag, and
only their duration is NULL — the same treatment negative and over-cap durations
already received.

| Target statistic | before | after |
|---|---|---|
| rows in `resolution_ml` | 1,026,596 | 942,325 |
| p10 | 0.03 h | 1.32 h |
| p25 | 8.87 h | 20.20 h |
| median | 82.99 h | 116.04 h |
| p75 | 598.8 h | 706.9 h |
| share under one minute | **8.21%** | **0.00%** |
| share under one hour | 16.43% | 8.95% |

### Baseline

GBM on `log1p`, evaluated on `split_global` (pooled, one wall clock):

| Split | Model MAE | Naive median MAE | Model median AE | Naive median AE |
|---|---|---|---|---|
| val | 734.1 | 866.6 | 84.7 | 109.0 |
| test | 876.5 | 1034.8 | **127.7** | **115.7** |

**Read that last row carefully.** On test the model beats the training-median
baseline on MAE (−15.3%) but is *worse* on median absolute error (127.7 h vs
115.7 h). It wins on the tail and loses in the middle. That is not a model ready
to be trusted for typical cases.

**Why still only conditional**

1. **A regime change sits inside the split.** Chicago's 2020 window is the first
   COVID wave; median resolution time moves sharply across the fold boundary.
2. **The cities are not exchangeable** — different department vocabularies (3
   values in Chicago, 113 in SF) and `subcategory` missing for 100% of Chicago.
3. **Censoring is mild, and was previously over-stated here.** Measured: 0.82%
   of in-scope incidents are still open (NYC 8.2%, Chicago and SF ≈0.1%). Right
   censoring is a real caveat but it is *not* the main target problem; the
   instant-closure artifact was.

---

## 3. SLA breach — CONDITIONALLY READY

**Target** `sla_breach`, 20,224 rows, 10.5% positive.

| Split | Model PR-AUC | Base rate | ROC-AUC |
|---|---|---|---|
| val | 0.349 | 0.084 | 0.766 |
| test | 0.260 | 0.103 | 0.679 |

Real signal — roughly 2.5x the base rate on test — but the population is the
constraint, not the model:

* **one city.** Only publishers that ship a due date have a target at all; in
  this extract that is NYC alone. Boston (`TARGET_DT`) is not downloaded.
* **one quarter.** NYC's extract covers January–March 2010. No seasonality, no
  multi-year behaviour.
* **a selected slice of complaint types.** Only some NYC complaint types carry a
  Due Date, so this is not a random sample of incidents.

This is a usable research dataset for a NYC-2010 SLA-risk model. It is not a
general SLA model and must never be presented as one.

---

## 4. Hotspot — CONDITIONALLY READY (and it does **not** beat a moving average)

**Target** `future_incident_count`, week t+1 per city × zone × category. 67,092
panel rows, 20.0% of them zero.

**Baselines matter more than the model here, so all three are reported.**
Evaluated on `split_global` (one wall clock — see §8), test MAE:

| Predictor | val MAE | test MAE | test Poisson dev. |
|---|---|---|---|
| GBM (Poisson) | 7.90 | **10.54** | 14.02 |
| **Rolling 4-week mean (t-3..t)** — primary benchmark | **6.98** | 10.89 | 14.48 |
| Rolling 4-week mean, lagged (t-4..t-1) | 7.32 | 11.76 | 16.56 |
| Persistence (t+1 = t) | 7.93 | 12.62 | 23.88 |

| Comparison | val | test |
|---|---|---|
| Model vs **persistence** | +0.4% | **+16.5%** |
| Model vs **rolling 4-week mean** | **−13.2%** | **+3.2%** |

**An earlier version of this document claimed "a 15% MAE improvement" and cited
only persistence. That was misleading and is withdrawn.** Persistence is the
weakest thing anyone would try on a weekly count panel. Against a four-week
moving average — the first tool any forecaster reaches for — the model is **3.2%
better on test and 13.2% WORSE on validation**. On Poisson deviance the two are
also within a few percent of each other on test.

The honest reading: the gradient-boosted model has not demonstrated that it adds
anything over a moving average. Persistence is kept in the table because it is
the naive floor and shows the panel has real autocorrelation, but it is no
longer presented as the benchmark.

**Why the rolling window is defined as t-3..t.** The manifest states the
deployment assumption: the forecast for week t+1 is produced at the END of week
t, when week t is complete. Week t is therefore available, and excluding it
would handicap the baseline and flatter the model — the exact error being
corrected. The strictly-lagged t-4..t-1 variant is reported alongside because it
is what already ships as the `rolling_4w_mean` feature. Both are trailing
windows that end at or before week t while the target is week t+1, so neither
can see the target; `tests/test_pipeline.py` proves it by rewriting every future
value and asserting no prediction moves.

**This task was previously unlearnable and nobody could see it.** The panel used
to contain only weeks that had at least one incident, so `shift(-1)` meant "the
next week that happened to have one": the target's minimum was 1, the binary
`future_incident_flag` was `True` for 100% of rows, and 6.2% of the lag features
referred to a week that was not t−1 at all. The panel is now a complete calendar
grid and zeros are real observations.

**Remaining caveat:** under `split_global`, validation and test are Chicago-only,
because SF ends May 2018 and NYC ends March 2010. Under the per-city `split`
column all three cities appear, and the conclusion is unchanged (model vs
rolling mean: −10.0% val, +1.3% test).

---

## 5. Duplicate detection — NOT READY for evaluation

**Target** `same_incident`, 455,460 pairs, 231,955 positives from Chicago's
`PARENT_SR_NUMBER`.

The baseline scores **PR-AUC 0.998 on test**. That number is not a measure of
anything useful, and the per-strategy breakdown proves it:

| Negative strategy | PR-AUC vs positives |
|---|---|
| N1 same category, >2 km apart | 0.9986 |
| N2 same category, <200 m, >90 days apart | 0.9998 |
| N3 different category, <200 m, <7 days | 0.9995 |
| N4 random unrelated | 0.9996 |

Every negative is separable from a positive using exactly the three features the
model is given. That is by construction: N1 is *defined* by distance, N2 by time
gap, N3 by category mismatch. The classifier is recovering the sampling rules,
not learning what a duplicate looks like.

**What is genuinely ready:** the positive labels (a real municipal
determination), the pair-feature computation, and the split — which is now
connected-component chronological with **0 incidents appearing in more than one
split**, replacing an order-of-appearance split that had put all 231,955
positives in train and left validation and test 100% negative.

**What is missing:** a realistic negative population. The deployment candidate
set is "pairs that are close in space and time and the same category, that are
nevertheless different incidents". Sampling those from this corpus would inject
false negatives, because Chicago's parent pointer is incomplete — two unlinked
reports 50 m and 2 hours apart probably *are* duplicates. Manufacturing that
label is exactly the thing this pipeline refuses to do. The honest path is to
evaluate against real candidate pairs once UrbanEye+ logs its own duplicate
decisions.

Until then: use this table to check that a model *runs*, never to estimate how
well it will work.

---

## 6. Priority — NOT READY, and not a supervised task

There is **no priority ground truth**, in this corpus or in any public 311
dataset. Searched and not found: priority, urgency, risk or severity fields.
Found but rejected as proxies: `status`, `closed_at` and `resolution_time_hours`
(outcomes shaped by departmental capacity, not statements of urgency) and due
dates (one city's staffing policy).

The pipeline therefore ships priority as a **deterministic policy score and
ranking**:

```
priority_baseline      P1-P4 from config/priority_config.yaml   DERIVED, auditable
priority_score         the normalised 0-1 score behind it        DERIVED
priority_label         NULL — guard-enforced, provenance NULL
is_ground_truth        False on every row
target_status          "NO_PRIORITY_GROUND_TRUTH_AVAILABLE"               in-band
```

`scripts/baselines/run_baselines.py` **refuses** to train a priority model and
records why.

### One thing to know before using the score as a ranking

The score is normalised over the weights of the features that were actually
available (`null_handling: strict_available_only`). Two rows scored on different
input sets are therefore **not on the same scale**:

| Inputs available | Rows | Score range |
|---|---|---|
| 3 of 14 (category + both density terms) | 1,870,394 | 0.083 – 0.875 |
| 1 of 14 (category only — no usable coordinate) | 29,606 | 0.100 – 0.850 |

Rank within a stratum, or show `priority_feature_coverage` next to the score.
Mixing the two in one ordered queue compares numbers that were computed
differently.

**What would unblock a learned model:** operator-assigned priorities from
UrbanEye+'s own console, and above all the *override* flag — the rows where a
human disagreed with the policy are the only rows that can teach a model
something the rules do not already contain. See `PRIORITY_METHODOLOGY.md` §6.

---

## 7. Split policy — two columns, stated explicitly

Every table carries **both** split columns, because there is no single cut that
answers both questions:

| Column | Guarantee | Cost | Use for |
|---|---|---|---|
| `split` | chronological **within each city**; every city in every fold | not globally ordered — 19.5% of resolution test rows fall before the last training row on one wall clock | per-city models |
| `split_global` | chronological on **one wall clock**; no evaluation row predates a training row | SF ends May 2018 and NYC ends March 2010, so val/test are Chicago-only | pooled models |

Concrete counts for `resolution_ml` (942,325 rows):

| Column | train | val | test | cities in val/test |
|---|---|---|---|---|
| `split` | 659,627 | 141,348 | 141,350 | all three |
| `split_global` | 659,627 | 141,349 | 141,349 | Chicago only |

The baseline script picks automatically and records which it used: `split_global`
for genuinely multi-city tables, `split` for single-source ones (sla_ml is
NYC-only, and a global cut would put every NYC row in train and leave validation
and test empty). Encoders are fitted on the training fold alone; unseen levels
map to a reserved code.

---

---

## 8. Trained models — status and results

`scripts/train/` fits one model per ready task. Everything below was produced by
`python scripts/train/run_all.py`; the full reports live in
`reports/model_training/`.

### Ready for training — and trained

| Task | Model | Test headline | Baseline (test) | Clears its baseline? |
|---|---|---|---|---|
| **Resolution** | HGB, squared_error on log1p | MAE **896.5** · median AE **133.3** · R² 0.214 | median 1034.8 / 115.5 | on MAE yes, **on median AE no** |
| **SLA** | HGB, default | PR-AUC **0.249** · ROC-AUC 0.683 · recall 0.361 | base rate 0.104 | **yes, x2.40 lift** |
| **Hotspot** | HGB, squared_error on log1p | MAE **10.31** | rolling 4-week mean 10.89 | **not stably — loses on validation** |

### NOT ready — not trained, and no amount of training fixes them

| Task | Why |
|---|---|
| **Duplicate detection** | The labels are genuine but the evaluation set is not. In the realistic candidate population (same category, ≤200 m, ≤7 days) the test fold holds **23,480 positives and zero negatives**, and a depth-2 decision tree scores PR-AUC 0.967. Any headline metric measures the negative-sampling rule. Needs adjudicated hard negatives. |
| **Priority** | No public 311 dataset records an operational priority. `priority_baseline` is a deterministic function of `config/priority_config.yaml`, so a model fitted to it would reproduce the YAML and any accuracy quoted from it would be circular. Needs operator-assigned priorities and the override flag. |

`scripts/train/run_all.py` refuses to train either and records the reason in
`training_summary.json`.

### Hotspot: the rolling 4-week mean is the benchmark, and it is strong

| Fold | Model | Rolling 4-week mean | Rolling 4-week, lagged | Persistence |
|---|---|---|---|---|
| validation | 7.457 | **6.978** | 7.323 | 7.926 |
| test | **10.314** | 10.885 | 11.762 | 12.623 |

The model **loses to the rolling mean on validation by 6.9%** and beats it on
test by 5.2%. A win on one fold and a loss on the other is not an improvement,
it is noise, and the verdict in `hotspot_results.json` is **derived from the
numbers in code** rather than written by hand so that it cannot drift into
optimism. `tests/test_training.py::test_hotspot_report_does_not_claim_a_win_it_did_not_earn`
asserts exactly that.

**Validation performance must never be hidden when the ML model loses to the
baseline.** If only the test number were quoted, this model would look like a
5% improvement. It is not one. The current recommendation is to **ship the
rolling 4-week mean** and treat the model as unproven.

### Resolution: a known median-error limitation

The model wins on MAE (−13.4% against the training-median baseline) and **loses
on median absolute error** (133.3 h vs 115.5 h). It is better on the long tail
and worse in the middle of the distribution, which is where most operational
decisions live. The same pattern does not appear on validation (model 84.2 h vs
baseline 108.8 h), so the reversal is a property of the test period — Chicago's
2020 window — not a stable model characteristic.

Anyone quoting a single resolution number must quote both, and must say which
period it covers.

Configurations tried, all scored on validation MAE:

| Config | Question | val MAE |
|---|---|---|
| A_log1p_squared | does the natural shape of the target model it best? | 743.9 |
| B_raw_absolute | does optimising MAE directly beat optimising it indirectly? | 794.3 |
| **C_log1p_deeper** | is A capacity-limited? | **734.5** |

B is the informative failure: fitting `absolute_error` on raw hours optimises
the headline metric directly and still does worse than fitting squared error on
`log1p`. The target's shape matters more than the loss function's name.

### SLA: real signal on a narrow population

PR-AUC 0.249 against a 0.104 base rate is a **2.4x lift**, with ROC-AUC 0.683.
Class weighting (`class_weight='balanced'`) made it *worse* (val PR-AUC 0.263 vs
0.279), so the selected model is the unweighted one; imbalance is never handled
by resampling, which would move validation and test away from the prevalence a
deployment would see. The decision threshold (0.167) is chosen by maximising F1
on **validation** and frozen before test is scored.

Accuracy is deliberately not a headline. At a ~10% positive rate, "never
breaches" scores ~90% accuracy and catches nothing.

The constraint is the population, not the model: NYC only, January–March 2010,
and only the complaint types that carry a Due Date.

### Split policy in training

The training framework uses the same `choose_split_column` rule as the
baselines, so a model is never scored on a different footing from the baseline
it is compared against:

- **`split_global`** for resolution and hotspot — multi-city tables, pooled
  models, one wall clock.
- **`split`** for SLA — the table is NYC-only, so a global cut would put every
  row in train and leave validation and test empty.

The two columns answer different questions and neither is a compromise. See §7.

---

## 9. Cross-cutting limits

1. **Chicago is 84% of the corpus.** Any pooled metric is a Chicago metric.
2. **The cities barely overlap in time** — NYC Jan–Mar 2010, SF Jan–May 2018,
   Chicago Jul 2018–Aug 2020. `city` and `year` are nearly interchangeable, so a
   pooled model that looks like it learned geography may have learned the
   calendar.
3. **18.9% of rows are UNMAPPED** and 26% are OUT_OF_SCOPE. They never reach a
   model-ready table. `reports/category_mapping_candidates.csv` ranks the
   candidates — 11 source categories covering 76,306 rows have high-confidence
   proposals backed by existing mappings — but nothing is applied automatically.
4. **23 POI / road / area features are NULL** because no geospatial reference
   layer exists. The provider architecture is ready for one
   (`scripts/preprocess/geo_providers.py`); the data is not.
5. **`priority_config.yaml` is still `status: DRAFT`.** The weights are a
   template, not approved policy.
