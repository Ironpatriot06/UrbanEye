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
| Resolution time | `resolution_ml.parquet` | 1,026,596 | `resolution_time_hours` | **CONDITIONALLY READY** |
| SLA breach | `sla_ml.parquet` | 20,224 | `sla_breach` | **CONDITIONALLY READY** |
| Hotspot | `hotspot_ml.parquet` | 67,092 | `future_incident_count` | **CONDITIONALLY READY** |
| Duplicate detection | `duplicate_ml.parquet` | 455,460 | `same_incident` | **NOT READY for evaluation** |
| Priority | `priority_features.parquet` | 1,900,000 | none | **NOT READY — and not a supervised task** |

---

## 2. Resolution time — CONDITIONALLY READY

**Target** `resolution_time_hours`, 1,026,596 closed in-scope cases. Median ~83 h,
p99 ~18,000 h, max 71,015 h. Fit on `log1p`.

**Baseline** (HistGradientBoosting, 400k train rows, 20 declared predictors):

| Split | Model median AE | Naive median AE | Model MAE | Naive MAE |
|---|---|---|---|---|
| val | 47.0 h | 91.8 h | 681.9 h | 856.2 h |
| test | 84.7 h | 92.3 h | 733.5 h | 896.9 h |

The model beats "predict the training median", but not by much on test: 84.7 h
against 92.3 h is an 8% improvement, and the typical error is roughly the size of
the typical target. Per-city test MAE: Chicago 774 h, NYC 275 h, SF 784 h.

**Why only conditional**

1. **A regime change sits inside the split.** Chicago's validation window is the
   first COVID wave; median resolution time falls from 137 h in train to 51 h in
   val and returns to 144 h in test. That is why val looks better than test. No
   single number from this split estimates steady-state performance.
2. **Right-censoring.** Open cases are excluded, so the long tail is
   systematically under-represented and the model is fitted on the cases that
   closed.
3. **The three cities are not exchangeable** — an order of magnitude apart in
   median resolution time, with different department vocabularies (3 values in
   Chicago, 113 in SF) and `subcategory` missing for 100% of Chicago rows.

**Before deploying anything:** model per city, report per-period metrics, and
decide whether the censored tail matters enough to need survival analysis.

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

## 4. Hotspot — CONDITIONALLY READY

**Target** `future_incident_count`, week t+1 per city × zone × category. 67,092
panel rows, 20.0% of them zero.

| Split | Model MAE | Persistence MAE | Model Poisson dev. | Persistence Poisson dev. |
|---|---|---|---|---|
| val | 7.62 | 7.81 | 6.42 | 15.91 |
| test | 10.92 | 12.82 | 14.85 | 23.39 |

The model beats persistence ("next week equals this week") on both splits, which
is the bar that matters for a weekly panel — a 15% MAE improvement and a 37%
reduction in Poisson deviance on test.

**This task was previously unlearnable and nobody could see it.** The panel used
to contain only weeks that had at least one incident, so `shift(-1)` meant "the
next week that happened to have one": the target's minimum was 1, the binary
`future_incident_flag` was `True` for 100% of rows, and 6.2% of the lag features
referred to a week that was not t−1 at all. The panel is now a complete calendar
grid and zeros are real observations.

**Remaining caveat:** NYC contributes 10 weeks and SF 19, so their validation and
test folds are one or two weeks wide. Per-city metrics for those two cities are
not meaningful. Chicago carries the result.

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

## 7. Cross-cutting limits

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
