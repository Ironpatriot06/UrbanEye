# UrbanEye+ — ML Data Pipeline

A reproducible, auditable pipeline that turns four public 311 datasets into
ML-ready tables for UrbanEye+.

**This is a normal folder.** No git repository, no commits. Drop it anywhere
inside your UrbanEye+ project. It does not touch your frontend, backend, database
or API.

---

## 1. What this does

- Downloads (or accepts manually downloaded) **SF, Boston, Chicago and NYC 311** data
- Cleans it — coordinates, timestamps, timezones, impossible dates, negative durations
- Maps ~120 source categories onto the **15-category UrbanEye+ taxonomy**
- Normalises everything into one `incidents` table with **enforced provenance**
- Derives SLA signals from Boston `TARGET_DT` and NYC `Due Date`
- Builds a **pluggable geospatial feature layer** (`incident_geo_features`)
- Runs a **configurable, explainable priority baseline engine**
- Derives report-time temporal features **in each city's own local timezone**
- Computes **strictly backward-looking** incident-density features
- Emits **one clean dataset per ML task**, each with a manifest declaring every
  column's role and when its value becomes known
- Audits leakage, statistics and per-task readiness, and says which tasks are
  actually modellable (**[`ML_READINESS.md`](ML_READINESS.md)**)
- Validates data quality, provenance and **data leakage**

## 2. What this does NOT do

- **Does not train any model.** Not one. That is your next step.
- **Does not predict category** — the citizen selects it in the app.
- **Does not predict severity** — out of product scope, and no public labels exist.
- **Does not download images.** No image task exists, so none is needed.
- **Does not fabricate labels.** Missing stays missing, and a guard enforces it.
- **Does not pretend US geography is Indian geography.**
- **Does not modify your application.**

---

## 3. Install

Python **3.10+** required.

```bash
cd UrbanEye_ML_Data_Pipeline
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Five dependencies: pandas, pyarrow, PyYAML, requests, numpy. No ML libraries —
nothing is trained here. `pytest` is needed only to run `tests/`.

**Verify the install without any data:**

```bash
python tests/make_fixtures.py
python scripts/run_pipeline.py --fixtures
```

Expect `19/19 steps OK` (17 if you have no Boston fixture). Then delete the
synthetic output before using real data:

```bash
rm -rf data/processed/*
```

And run the unit tests, which check the things a summary report cannot show you —
that the density windows match a brute-force implementation of their own
definition, that local time converts correctly across DST, that the vectorised
priority engine agrees with the readable one row for row, and that the
provenance guard still fires:

```bash
python -m pytest tests -q          # 15 passed
```

---

## 4. Get the data

Full instructions, exact URLs, subset recommendations and disk estimates:
**[`DATASET_DOWNLOAD_GUIDE.md`](DATASET_DOWNLOAD_GUIDE.md)**

Short version:

| Dataset | Source | Download | Put it in |
|---|---|---|---|
| SF 311 | https://data.sf.gov/d/vw6y-z8j6 | **SUBSET** (2018+) | `data/raw/sf311/` |
| Boston 311 | https://data.boston.gov/dataset/311-service-requests | **SUBSET** (2011–2024 yearly CSVs) | `data/raw/boston311/` |
| Chicago 311 | https://data.cityofchicago.org/d/v6vf-nfxy | **SUBSET** (exclude 2 SR types) | `data/raw/chicago311/` |
| NYC 311 | https://data.cityofnewyork.us/d/76ig-c548 | **SUBSET** (filter complaint types) | `data/raw/nyc311/` |

**Disk: 15 GB minimum, 25 GB recommended.** Subset downloads are ~8.9 GB raw
versus ~25.7 GB for the full sets, and the difference is entirely rows you would
discard anyway.

Raw files are never modified. Everything in `data/processed/` is regenerable from
`data/raw/` by re-running the scripts.

---

## 5. Run it

```bash
python scripts/run_pipeline.py
```

Or step by step, in dependency order:

```bash
# Phase 2 — normalise each city (only the ones you have downloaded)
python scripts/preprocess/preprocess_sf311.py
python scripts/preprocess/preprocess_boston311.py
python scripts/preprocess/preprocess_chicago311.py
python scripts/preprocess/preprocess_nyc311.py

# Phase 3 — union
python scripts/preprocess/build_incidents.py

# Phase 4 — feature tables (both join back on incident_id)
python scripts/preprocess/build_time_features.py      # local-city-time block
python scripts/preprocess/build_geo_features.py       # POI schema + backward density

# Phase 5 — priority baseline + priority_dataset + split
python scripts/preprocess/build_priority_features.py --accept-draft-config

# Phase 6 — ML-ready tables
python scripts/preprocess/build_resolution_dataset.py
python scripts/preprocess/build_hotspot_dataset.py
python scripts/preprocess/build_duplicate_pairs.py
python scripts/features/build_ml_dataset.py           # consolidated urbaneye_ml
python scripts/features/build_task_datasets.py        # one dataset per ML task

# Phase 7 — validation
python scripts/validation/data_quality_report.py
python scripts/validation/validate_provenance.py
python scripts/validation/validate_leakage.py
python scripts/validation/validate_outputs.py
python scripts/validation/audit_pipeline.py           # full stage-by-stage audit
python scripts/validation/audit_task_datasets.py      # per-task + 12 leakage routes
python scripts/validation/audit_statistics.py         # imbalance, drift, degenerate columns
python scripts/validation/report_category_candidates.py
```

Optional, and **not** part of the pipeline — baseline models, for evidence that
a task is actually modellable (needs `pip install scikit-learn`):

```bash
python scripts/baselines/run_baselines.py
```

Useful flags: `--from 4-features`, `--only build_ml_dataset`, `--dry-run`,
`--fixtures`. Sources with no raw files are skipped with a warning instead of
aborting the run, and every report records which sources were included.

The order matters: priority scoring normalises over *available* features, so the
feature tables must exist first or every row scores on category alone. Each
script also refuses to run with an explicit "run X first" message when its input
is missing, so the dependency is enforced whichever way you invoke it.

---

## 6. Outputs

| Table | Path | Purpose |
|---|---|---|
| `all_incidents` | `data/processed/incidents/` | Normalised union, 40 columns |
| `all_incidents_prioritised` | `data/processed/incidents/` | The same table with the baseline policy columns filled |
| `incident_time_features` | `data/processed/features/` | 10 columns, derived in each city's **local** time |
| `incident_geo_features` | `data/processed/geo_features/` | 32 columns; 6 computed from 311 history, 23 await an external geospatial layer |
| `priority_dataset` | `data/processed/priority/` | Report-time features + policy baseline; **no target, by necessity** |
| `resolution_dataset` | `data/processed/resolution/` | Target: `resolution_time_hours`, `sla_breach` |
| `hotspot_dataset` | `data/processed/hotspot/` | Target: `future_incident_count`, on a complete weekly grid |
| `duplicate_pairs` | `data/processed/duplicates/` | Target: `same_incident` |
| `urbaneye_ml` | `data/processed/ml/` | Consolidated incident-level view |

**Per-task datasets** — one file per model, each with a `.manifest.json`:

| Task | File | Target | Unit |
|---|---|---|---|
| Resolution time | `ml/resolution_ml.parquet` | `resolution_time_hours` | incident |
| SLA breach | `ml/sla_ml.parquet` | `sla_breach` | incident |
| Hotspot | `ml/hotspot_ml.parquet` | `future_incident_count` | zone × category × week |
| Duplicates | `ml/duplicate_ml.parquet` | `same_incident` | pair of incidents |
| Priority | `ml/priority_features.parquet` | **none** | incident |

They exist because three of the five tasks are not incident-shaped and a fourth
has a different population. Read **[`ML_READINESS.md`](ML_READINESS.md)** before
training anything.

Reports land in `reports/`: `pipeline_audit.{json,md}`, `task_dataset_audit.{json,md}`,
`statistical_audit.{json,md}`, `category_mapping_candidates.{csv,md}`,
`baseline_models.{json,md}`, `data_quality.{json,md}`,
`provenance_report.json`, `leakage_report.json`, `output_validation.json`,
`dataset_status.md`, `priority_summary.json`, `time_features_summary.json`,
`geo_features_summary.json`, `ml_dataset_summary.json`, `unmapped_categories.csv`.

A record of what was broken in this pipeline, why, and what was changed:
**[`PIPELINE_FIXES.md`](PIPELINE_FIXES.md)**.

Full field-by-field documentation: **[`DATA_SCHEMA.md`](DATA_SCHEMA.md)**
(generated from code — regenerate with
`python scripts/validation/generate_schema_doc.py`).

---

## 7. Provenance rules

Every field is classified `SOURCE`, `DERIVED`, `SYSTEM` or `NULL`.

**Five fields are NULL and enforced empty:**

| Field | Why |
|---|---|
| `severity` | Out of product scope, and no public source provides a defensible grade |
| `occurred_at` | Every source records report time; none records incident onset |
| `response_time_hours` | No source records a first-response timestamp |
| `priority_label` | No public 311 dataset contains a genuine operational priority |
| `priority_label_source` | Same |

This is enforced in code, not documented as an aspiration.
`finalise_incidents()` raises `ProvenanceViolation` on any attempt to populate
them, and `validate_provenance.py` **actively tries all four fabrications** to
prove the guard works:

```
ACTIVE-1  PASS  guard REJECTS severity derived from RDD damage type
ACTIVE-2  PASS  guard REJECTS occurred_at derived from reported_at
ACTIVE-3  PASS  guard REJECTS response_time derived from resolution_time
ACTIVE-5  PASS  guard REJECTS priority_label copied from priority_baseline
```

---

## 8. Priority architecture

```
citizen (category, description, lat, lon, timestamp)
   -> time features          local city time, deterministic, not ML
   -> density features       backward-looking 311 history, deterministic, not ML
   -> PostGIS enrichment     deterministic, not ML  (NOT CONNECTED — all NULL)
   -> priority baseline      configurable, explainable, not ML
   -> SLA business rule      P1 30h | P2 50h | P3 72h | P4 90h
```

Priority is **not an ML target** and cannot be one yet. `priority_baseline` is a
deterministic function of `config/priority_config.yaml`; training a model on it
would teach the model the YAML file. A learned model needs `priority_label`,
which requires real UrbanEye+ operational outcomes.

The engine is not a category lookup. Verified by running it:

| Input | Output |
|---|---|
| `POTHOLE` + school 50 m + metro 300 m + major road 15 m + intersection 12 m + metro city + 37 nearby | **P2** (0.623) |
| `POTHOLE`, isolated residential street | **P4** (0.241) |
| `OPEN_MANHOLE`, no context | **P1** (absolute rule) |

On the real corpus the score also moves within a category, because the
backward-looking density features are genuinely available: identical categories
land in different bands depending on how much recent repeat-reporting surrounds
them. `audit_pipeline.py` check **PR-7** asserts exactly that.

### Confidence means feature coverage, nothing else

Each row publishes `priority_features_available` (how many of the 14 weighted
inputs were non-null) and `priority_feature_coverage` (the share of total policy
weight they carry). `priority_confidence` is a deterministic tier over those two
numbers, defined in `config/priority_config.yaml -> confidence_tiers`:

| Tier | Meaning on this corpus |
|---|---|
| HIGH | most of the policy evaluable, POI layer included — **not reachable on US 311 data** |
| MEDIUM | category + backward density available, POI/road/area layer absent |
| LOW | one or two inputs only — in practice, rows with no usable coordinate |

The config ships as `status: DRAFT`; the script **refuses to run** without
`--accept-draft-config`. Full rationale, including the five open policy questions
that need deciding: **[`PRIORITY_METHODOLOGY.md`](PRIORITY_METHODOLOGY.md)**.

### Splits

`splits.tabular` is chronological and **per source**. The absolute cut dates are
used only when they fall inside the corpus; otherwise the cuts fall back to
quantiles of the data's own timeline and the chosen timestamps are recorded in
every split report. Per-source rather than global because the three cities
occupy disjoint date ranges — one global cut would give validation and test to
Chicago and nothing else, leaving `city` unevaluable. Within each city, train
strictly precedes val strictly precedes test.

---

## 9. US data vs Indian deployment

The US 311 data supplies **behavioural** patterns — category/resolution
relationships, temporal dynamics, what a duplicate looks like.

It supplies **no Indian geographic context**. All 23 POI/road/area columns are
NULL and stay NULL until an Indian geospatial layer is plugged in. Raw
latitude/longitude is never a model feature — only semantic features derived from
it. A model that learns "latitude 37.77 is high priority" has learned San
Francisco, which is worse than useless in Chennai.

Zone IDs are US wards and community boards, **not** Indian zones. Model per city
or include `city` as a feature; never pool them.

See **[`GEOSPATIAL_FEATURES.md`](GEOSPATIAL_FEATURES.md)** for the provider
interface and how to add `IndiaPostGISProvider`.

---

## 10. Severity

Not predicted, not displayed, NULL everywhere. The citizen UI shows **one**
urgency concept: priority.

The column is retained for schema compatibility and is guard-enforced empty.
RDD2022 damage type is **not** severity ground truth and mapping it to one is
blocked by `ACTIVE-1`.

---

## 11. Limitations you should know before modelling

1. **`description` is agency-written**, not citizen-written. It is 311 resolution
   notes and case titles. It is also post-resolution, so it is excluded from every
   feature set.
2. **Chicago has no free-text field at all** — `description` is NULL for those rows.
3. **`OPEN_MANHOLE` has zero incidents** across all four cities. No US 311
   vocabulary has that category. It is in the taxonomy because UrbanEye+ needs it;
   training data must come from your own collection.
4. **Right-censoring** in `resolution_dataset`: open cases are excluded, biasing
   toward faster resolutions.
5. **`priority_confidence` never reaches HIGH** on this corpus. It is MEDIUM for
   every row with a usable coordinate (category + density available, 36% of
   policy weight) and LOW for the 29,606 rows without one. HIGH requires the POI
   layer, which no public US 311 dataset contains.
6. **`image_similarity` is NULL** in `duplicate_pairs` — Chicago, the only
   labelled source, publishes no photographs.
7. **18.9% of rows are `UNMAPPED`** and a further 26% are `OUT_OF_SCOPE`. They
   stay in `all_incidents` so the counts remain auditable but never reach a
   model-ready table. `reports/unmapped_categories.csv` is the work queue;
   extending `config/category_mapping.csv` is a taxonomy decision for a human,
   so the pipeline reports the gap rather than guessing.
8. **The corpus on disk is a subset**: Chicago Jul 2018 - Aug 2020, SF Jan - May
   2018, NYC Jan - Mar 2010, and Boston not downloaded at all. Because the split
   is per source, every split contains all three cities, but no city's test
   period overlaps another's in wall-clock time.

---

## 12. Verification status

**Everything reported below was executed on the real corpus currently in
`data/raw/` (Chicago 1.6M rows, SF 200k, NYC 100k; Boston not downloaded).**

Full rebuild from an empty `data/processed/`:

| Suite | Result |
|---|---|
| `run_pipeline.py` (real data, from scratch) | **22/22 steps OK in 522s** |
| `run_pipeline.py --fixtures` | **19/19 steps OK in 8s** |
| `pytest tests` | **32/32 passed** |
| `audit_pipeline.py` | **56/56 checks** |
| `audit_task_datasets.py` | **78/80 checks** (2 skips: no joinable history columns) |
| `audit_statistics.py` | 42 findings, classified; 1 `dangerous` (a documented regime change) |
| `data_quality_report.py` | **11/11 checks** |
| `validate_provenance.py` | **11/11 checks** (incl. 4 active fabrication attempts) |
| `validate_leakage.py` | **11/11 checks** |
| `validate_outputs.py` | **15/15 checks** |

A second run from the same raw files reproduced every table's row and column
count exactly.

Stage timings on the 1.9M-row corpus (8-core laptop): preprocess 55s, union 20s,
time features 3s, geo/density 27s, priority 55s, resolution 16s, hotspot 8s,
duplicates 61s, ML table 35s, validation ~120s.

> The synthetic fixtures remain useful as an offline smoke test and are still
> exercised by `--fixtures`, which writes a `SYNTHETIC_FIXTURE_OUTPUT.txt`
> marker so its output can never be mistaken for real data. Fixture results are
> not evidence about real data, and the table above does not use them except
> where it says so.

---

## 13. Folder layout

```
UrbanEye_ML_Data_Pipeline/
├── README.md                     DATASET_DOWNLOAD_GUIDE.md
├── PIPELINE_FIXES.md             what was broken and what changed
├── ML_READINESS.md               per-task verdicts, with measured baselines
├── DATASET_SOURCES.md            DATA_SCHEMA.md          (generated)
├── PRIORITY_METHODOLOGY.md       GEOSPATIAL_FEATURES.md
├── MODEL_SCOPE.md                requirements.txt
├── config/
│   ├── dataset_config.yaml       sources, licences, cleaning rules, splits
│   ├── category_mapping.csv      ~120 source categories -> 15 canonical
│   ├── priority_config.yaml      weights, thresholds, bands, SLA hours
│   └── feature_config.yaml       feature inventory + provider contract
├── scripts/
│   ├── download/                 4 optional download helpers
│   ├── preprocess/               4 city processors + 5 builders
│   ├── features/                 consolidated + per-task ML dataset builders
│   ├── baselines/                optional baseline models (not part of the pipeline)
│   ├── validation/               7 validators + schema-doc generator
│   ├── utils/                    paths, schema, cleaning, mapping, manifest
│   └── run_pipeline.py           orchestrator
├── data/raw/                     you put downloads here (never modified)
├── data/processed/               generated outputs
├── data/manifests/               checksums, source URLs, row counts
├── reports/                      quality, provenance, leakage, status
├── tests/                        synthetic fixture generator + unit tests
└── optional_future_vision/       image scripts, NOT part of this pipeline
```

---

## 14. Next steps

1. Download the four datasets (`DATASET_DOWNLOAD_GUIDE.md`).
2. Run the pipeline; read `reports/data_quality.md`.
3. Extend `config/category_mapping.csv` using `reports/unmapped_categories.csv` —
   expect many unmapped values on the first real run.
4. Review and approve `config/priority_config.yaml`.
5. Plug in an Indian geospatial provider (`GEOSPATIAL_FEATURES.md`).
6. Train resolution, hotspot and duplicate models. **Not priority** — see
   `PRIORITY_METHODOLOGY.md` §6 for what has to be logged first.
