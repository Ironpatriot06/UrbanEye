# What was broken, and what was done about it

A record of the state this pipeline was found in, the root cause of each defect,
and the fix. Written so the decisions can be argued with rather than taken on
trust.

---

## 1. Root cause, in one paragraph

The pipeline's design was sound and its documentation was honest. What had
happened is that **two of its stages never ran, and a third script was added
outside it**.

`build_geo_features.py` could not finish: its density loop iterated in Python
over every row and compared each one against every incident that had *ever*
occurred in its grid block, ignoring the time window when selecting candidates.
On 1.9M rows that does not complete, so `data/processed/geo_features/` was
empty. Everything downstream then behaved exactly as designed for a corpus with
no features: the priority score collapsed to the category term, confidence was
LOW for all 1.9M rows, and all 28 spatial/density columns were NULL. The output
looked like a policy statement; it was a missing file.

Separately, `scripts/features/build_ml_dataset.py` had been added outside
`run_pipeline.py`. It read a hand-made parquet no script produced and invented a
`priority_target` from keyword counts over agency-written resolution text. That
is the one thing the repository's provenance guard exists to prevent, and it was
happening in a directory the guard did not cover.

---

## 2. Defect by defect

| # | Defect | Root cause | Fix |
|---|---|---|---|
| 1 | 28 spatial/density columns 100% NULL | `build_geo_features.py` never completed; output file absent | Vectorised the density builder: **1.9M rows / 429M candidate pairs in 27s** |
| 2 | Density counts would have been **wrong** even if it had run | `density_grid.neighbour_ring: 1` is a 3x3 block; at 0.0025° a longitude cell is ~207 m at Chicago's latitude, so a true neighbour 240 m away sat two cells out and was never counted | Ring is now derived per axis from `radius_m` and the cell's own latitude (`required_rings()`); a brute-force test asserts exact equality |
| 3 | Time features missing from `all_incidents`, and derived from **UTC** | They were computed inline inside the priority builder, off the UTC timestamp. `hour` and `is_night` then encoded the city's longitude, not the time of day | New stage `build_time_features.py` derives them in each city's own timezone, from the mapping that already existed in `dataset_config.yaml`; `hour_utc` retained for audit |
| 4 | `split` = train for 100% of rows | Absolute cut dates (2022-12-31 / 2023-12-31) lie beyond the corpus, which ends Aug 2020 | Cuts fall back to quantiles of the data's own timeline when the absolute dates are out of range, and the chosen timestamps are recorded in every report |
| 5 | A global chronological split would have given val and test to Chicago alone | The three cities occupy disjoint date ranges (NYC Jan–Mar 2010, SF Jan–May 2018, Chicago Jul 2018–Aug 2020) | Split **per source**, so every city appears in every split while train still strictly precedes val strictly precedes test *within* a city |
| 6 | `urbaneye_ml.parquet` carried a fabricated target | `build_ml_dataset.py` counted keywords in `category` + `description`; `description` is post-resolution agency text | Rewritten. No invented target; every column carries a declared role in a manifest; post-resolution columns excluded |
| 7 | `priority_confidence` LOW everywhere, with no way to check it | Binary HIGH/LOW on a count of available inputs | Three deterministic tiers over **two published numbers** — `priority_features_available` and `priority_feature_coverage` — so the tier is auditable, not asserted |
| 8 | The priority engine took ~1 dict per row | `apply()` built 1.9M dicts and scored them one at a time | Vectorised, chunked; a test asserts the fast path and the readable path agree on band, score, confidence and reason string |
| 9 | `build_duplicate_pairs.py` took >10 minutes | Co-located negative sampling appended `.iloc[i]` Series in a Python loop | Vectorised cell sampling; 61s |
| 10 | The orchestrator refused to run at all if any city was missing | Boston was never downloaded | Missing sources are skipped with a warning; reports record which were included |

---

## 3. What is populated now, and what is not

| Feature family | Status |
|---|---|
| `hour`, `day_of_week`, `month`, `year`, `is_weekend`, `is_night`, `hour_utc` | **Populated**, in local city time, 0 missing |
| `nearby_similar_incidents_24h / 7d / 30d`, `local_incident_density`, `category_incident_density` | **Populated** for 1,870,394 rows — every row with a usable coordinate |
| `near_school`, `near_hospital`, `near_metro`, `near_rail_station`, `near_public_transport`, `near_fire_station`, `near_police_station`, `near_government_building` and their distances | **NULL, and correctly so** |
| `near_major_road`, `distance_to_major_road_m`, `road_class`, `near_intersection`, `distance_to_nearest_intersection_m` | **NULL** |
| `metro_city`, `urban_area` | **NULL** |

The 23 NULL columns were not skipped for convenience. There is **no reference
geodata in this repository** — `data/external/` does not exist, and no
school/hospital/metro/rail/transport/fire/police/government/road/intersection
geometry ships with it. The 311 publishers supply a coordinate and nothing else.
Computing these would mean acquiring and licensing a POI layer; inventing them
would be fabrication. They are emitted with the right name and dtype and left
empty, which is what keeps the schema stable for the day a provider is connected.

`audit_pipeline.py` check **SP-2** asserts they are still empty, and records in
`reports/pipeline_audit.json` that `data/external` was looked for and not found.

---

## 4. The target question

There is **no priority ground truth**, and none can be manufactured from this
data.

What was searched for, and what was found:

| Looked for | Result |
|---|---|
| `priority`, `urgency`, `risk`, `severity` fields | Not present in any of the four publishers' schemas |
| `status`, `closed_at`, `resolution_time` | Present — but these are **outcomes**, not priorities. How long a case took reflects departmental capacity, not how urgent it was |
| Due date / SLA target | NYC `Due Date` only in this extract (22,458 rows). It encodes that city's staffing policy, not a safety judgement |
| Escalation / override records | Do not exist in public 311 data |

So the pipeline keeps three things strictly apart:

```
priority_baseline      P1-P4 from config/priority_config.yaml    DERIVED, auditable
priority_score         the 0-1 score behind it                    DERIVED
priority_label         NULL — guard-enforced, provenance NULL
is_ground_truth        False on every row
target_status          "NO_PRIORITY_GROUND_TRUTH_AVAILABLE"       in-band, machine-readable
```

`priority_baseline` was **not** promoted to a pseudo-label. It is a deterministic
function of a YAML file: a model trained on it would reproduce the YAML, and any
accuracy quoted against it would be circular. It travels in the ML table as
`policy_metadata` — the benchmark a future model must beat, not its target.

The targets that *are* real, and are built: `resolution_time_hours` (1,853,828
rows, right-censoring flagged not dropped), `sla_breach` (20,822 rows, NYC only),
`future_incident_count` (hotspot panel), `same_incident` (231,955 Chicago-labelled
positive pairs).

---

## 5. Leakage: what was actually checked

Not "the builder says it is backward-looking" — each of these **re-derives the
value independently and compares**:

- `audit_pipeline.py` **DN-6** recomputes all four density counts from the
  incidents table with an explicit `reported_at < t` filter, for a random sample.
  0 mismatches.
- **DN-7** computes what each count would be under a *forward* window and
  asserts no observed value could only have come from the future. 0 suspect rows.
- **DN-2/DN-3** assert the definitional invariants: 24h ≤ 7d ≤ 30d, and
  same-category ≤ all-category.
- `tests/test_pipeline.py` compares the vectorised builder against a brute-force
  implementation of the definition on a dense synthetic corpus, and asserts that
  two incidents with **identical timestamps** do not count each other (the window
  is half-open: `< t`, never `<= t`).
- `validate_leakage.py` **LEAK-2/3/4** re-derive the hotspot lag, the hotspot
  target and the 30-day density independently.
- **SPL-3** asserts no incident appears in two splits; **LK-4/5/6** assert no
  declared predictor is a target, a target-derivative, the policy output, or a
  raw coordinate.

---

## 6. Judgement calls a human should review

Three decisions here are policy, not engineering. They are implemented the way
that seemed most defensible, and they are all in config so they can be changed:

1. **Per-source splitting** (`splits.tabular.per_source: true`). The alternative
   — one global chronological cut — is the stricter reading, but on this corpus
   it puts zero SF and zero NYC rows in validation or test, making `city`
   unevaluable. If you would rather have the stricter split and accept a
   Chicago-only test set, set it to `false`.
2. **The MEDIUM confidence tier.** Adding it means most rows no longer report
   LOW. That is a real change in what the column says, so the thresholds are
   explicit in `priority_config.yaml` and both underlying numbers travel with
   every row. If you prefer the old binary scheme, delete `confidence_tiers`;
   the engine falls back to `min_features_for_confident_score`.
3. **The category mapping was left alone.** 358,838 rows (18.9%) are `UNMAPPED`
   and there are obvious candidates in `reports/unmapped_categories.csv`
   ("Alley Light Out Complaint" → `STREETLIGHT_FAULT`, 42,315 rows). Extending
   the mapping is a taxonomy decision with real consequences for what the model
   sees, so the pipeline reports the gap rather than closing it unilaterally.

---

## 7. Files changed

**New**

```
scripts/utils/timefeatures.py            local-time derivation
scripts/preprocess/build_time_features.py   new pipeline stage
scripts/validation/audit_pipeline.py     54-check stage-by-stage audit
tests/test_pipeline.py                   15 unit tests
PIPELINE_FIXES.md                        this file
data/processed/combined/README.txt       marks the orphaned hand-made parquet
```

**Rewritten**

```
scripts/features/build_ml_dataset.py     fabricated target removed; role manifest added
```

**Modified**

```
config/dataset_config.yaml               split: quantile fallback, per_source
config/feature_config.yaml               temporal_policy; density_grid ring: auto
config/priority_config.yaml              confidence_tiers
scripts/preprocess/build_geo_features.py vectorised density; correct neighbour ring
scripts/preprocess/build_priority_features.py  vectorised engine; time join; split; target manifest
scripts/preprocess/_ml_common.py         corpus-aware split; local-time features
scripts/preprocess/build_resolution_dataset.py  split metadata
scripts/preprocess/build_hotspot_dataset.py     shared split
scripts/preprocess/build_duplicate_pairs.py     vectorised negative sampling
scripts/utils/schema.py                  priority_feature_coverage; time_feature_schema()
scripts/validation/validate_leakage.py   per-source split assertion
scripts/validation/validate_outputs.py   per-source split assertion
scripts/validation/generate_schema_doc.py  documents the new tables
scripts/run_pipeline.py                  new stages; tolerates missing sources
README.md  PRIORITY_METHODOLOGY.md  GEOSPATIAL_FEATURES.md  MODEL_SCOPE.md
DATA_SCHEMA.md (regenerated)  requirements.txt  reports/README.md
```

No raw file was modified. No row was dropped. No generated parquet was edited by
hand — every output in `data/processed/` was produced by re-running the scripts
from `data/raw/`.


---

# Round 2 — task design, and two defects that made whole tasks unlearnable

The first round made the pipeline run correctly. This round asked a different
question: *would a model trained on these tables mean anything?* Two of the five
tasks turned out to be broken in ways every existing validator passed.

## 8. The two that mattered

| # | Defect | How it hid | Fix |
|---|---|---|---|
| 11 | **The hotspot target could never be zero.** The weekly panel contained only weeks that had an incident, so `shift(-1)` meant "the next week that happened to have one". `future_incident_flag` was `True` for **100%** of rows — a constant classification target — and 6.2% of lag features referred to a week that was not t−1. | `LEAK-2` re-derived `shift(1)` and compared it to `shift(1)`: it validated the implementation against itself, never against the definition. | The panel is reindexed onto a complete calendar-week grid per city. Zeros are observed facts (each extract is contiguous in the publisher's own id order). Target zero share is now 19.97%, and the binary target has two classes. |
| 12 | **Validation and test contained zero positives.** The duplicate split assigned rows by the order `incident_a` first appeared, and positives were concatenated first: all 231,955 positives landed in train, leaving val and test 100% negative. The negative-strategy mix was skewed too — test was entirely N4 random-unrelated, the easiest negatives in the set. | `OUT-4b` checked that both classes existed **in the table**, not in each split. | Incidents are grouped into connected components of the positive-pair graph, components are cut chronologically, and negatives are now sampled *inside* each split. 0 incidents appear in more than one split, all three splits carry both classes, and no pair has to be discarded. |

## 9. Everything else in round 2

| # | Change | Why |
|---|---|---|
| 13 | Per-task datasets: `resolution_ml`, `sla_ml`, `hotspot_ml`, `duplicate_ml`, `priority_features`, each with a manifest | three of five tasks are not incident-shaped and a fourth has a different population; one table invites pointing a model at whatever column looks like a label |
| 14 | `PREDICTION_TIME_AVAILABILITY` — every predictor must carry a written statement of when its value becomes known, enforced by an audit check and a test | a feature should not be able to reach a model without someone writing down why it is knowable |
| 15 | Split fallback for coarse timelines (`_distinct_value_cuts`) | row quantiles on ~10 distinct weekly timestamps gave NYC's hotspot panel 5,061 train / 723 val / **0 test** |
| 16 | Hotspot: drop rows with no observable t+1 **before** splitting | the last week is exactly what a chronological cut assigns to test, so dropping it afterwards emptied the test fold |
| 17 | `hours_since_previous_similar_incident` — new backward-looking feature, free (reuses the same candidate pairs), verified against brute force | five reports over a month and five in the last hour give the same 30-day count |
| 18 | `hour_utc` and `coord_outside_city_bbox` demoted from predictors to metadata | `hour_utc` is a city indicator dressed as a clock; the bbox flag is constant `False` across all 1.9M rows |
| 19 | `zone_key` (city-scoped zone) | `zone_id` is a ward in Chicago, a neighbourhood in SF and a community board in NYC |
| 20 | Geo provider registry: `null_provider` / `reference_layer` / `india_postgis`, config-selected | the architecture is now pluggable and the PostGIS stub fails loudly rather than degrading into NULLs indistinguishable from the default |
| 21 | Three new audits — per-task + 12 leakage routes, statistical, category-mapping candidates | the existing validators check what the builders promise; these check what a consumer could still get wrong |
| 22 | Baseline models with train-only preprocessing | a manifest can claim a task is ready; only fitting something shows whether it is |

## 10. What the baselines actually showed

Running them was the point. Two results changed the verdicts:

* **Hotspot beats persistence** (test MAE 10.9 vs 12.8, Poisson deviance 14.8 vs
  23.4). It could not have, before the panel fix — the target had no zeros.
* **Duplicate detection scores PR-AUC 0.998, and that number is worthless.** The
  per-strategy breakdown shows ≥0.9986 against every negative strategy, because
  N1 is *defined* by distance and N2 by time gap — the two features the model is
  given. The classifier is recovering the sampling rules. The dataset is marked
  NOT READY for evaluation, and the manifest says why.

Resolution beats "predict the training median" by only 8% on median absolute
error, and the validation window sits inside the first COVID wave. That is
recorded as a `dangerous` finding in the statistical audit — meaning *do not
quote a single number from this split as steady-state performance*, not *delete
the column*.

## 11. What was deliberately NOT done

* **No harder duplicate negatives.** The obvious fix for the 0.998 artefact is to
  sample same-category pairs that are close in space and time. Those are
  precisely the pairs most likely to be *unlabelled duplicates* — Chicago's
  parent pointer is incomplete — so labelling them negative would fabricate
  labels. Left for real operational data.
* **No category mappings applied.** 11 source categories covering 76,306 rows
  have high-confidence proposals backed by existing mappings. They are ranked in
  `reports/category_mapping_candidates.csv` with their evidence, and applied to
  nothing: moving 76k rows into scope changes the category prior every model
  sees.
* **No new priority policy weights.** `hours_since_previous_similar_incident` is
  computed and is not wired into the engine. Whether repeat-report pressure
  should escalate a priority is open policy question 4, not an engineering call.
* **No POI features.** Still no reference geodata. The architecture is ready; the
  data is not.
