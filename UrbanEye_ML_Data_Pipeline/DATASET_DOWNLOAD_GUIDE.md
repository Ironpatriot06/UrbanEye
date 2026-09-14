# UrbanEye+ — Dataset Download Guide

You download the data; the pipeline processes it. This file tells you exactly
what to fetch, where to put it, and what to run.

**Four datasets. All tabular. No images.** RDD2022, Smartathon, TACO and
Mapillary are **not required** — see [Why no image datasets](#why-no-image-datasets).

---

## 0. Disk space

| Item | Subset (recommended) | Full |
|---|---|---|
| SF 311 raw | ~1.4 GB | ~4.0 GB |
| Boston 311 raw (2011–2024) | ~2.5 GB | ~2.5 GB |
| Chicago 311 raw | ~2.4 GB | ~5.2 GB |
| NYC 311 raw | ~2.6 GB | ~14.0 GB |
| **Raw total** | **~8.9 GB** | **~25.7 GB** |
| Processed Parquet (all tables) | ~1.2 GB | ~3.0 GB |
| Temporary / working | ~2.0 GB | ~5.0 GB |
| Reports | < 10 MB | < 10 MB |
| **Peak total** | **~12 GB** | **~34 GB** |

**Minimum free space: 15 GB. Recommended: 25 GB.**

These are estimates from row counts × measured bytes-per-row on the projected
column sets, not measured downloads — I could not download the data from my
environment. Treat them as ±30%.

> **Why subset beats full every time here.** The full downloads are dominated by
> columns and rows you will immediately discard: NYC is ~70% housing/noise/parking
> complaints, Chicago has 7.3M rows of "information only call" and aircraft noise,
> and SF publishes ~30 legacy `DELETE - …` geography columns. Filtering at the API
> cuts the download roughly in half or better and changes nothing downstream.

---

## 1. SF 311 Cases

| | |
|---|---|
| **Source page** | https://data.sf.gov/d/vw6y-z8j6 |
| **Dataset ID** | `vw6y-z8j6` |
| **Direct CSV** | https://data.sf.gov/api/v3/views/vw6y-z8j6/export.csv?accessType=DOWNLOAD |
| **API** | https://data.sf.gov/resource/vw6y-z8j6.json |
| **Licence** | ODC-PDDL 1.0 (public domain) — commercial use OK, no attribution required |
| **Recommendation** | **SUBSET** (2018-01-01 onward, projected columns) |
| **Rows** | ~8.9M full; ~3.5M from 2018 |

**What to download:** either works.

*Option A — helper script (does the filtering for you):*
```bash
python scripts/download/download_sf311.py --estimate-only   # check size first
python scripts/download/download_sf311.py
```
Writes `data/raw/sf311/sf311_cases.jsonl`.

*Option B — manual bulk CSV.* Click the Direct CSV link above, then:
```bash
mv ~/Downloads/311_Cases*.csv data/raw/sf311/sf311_bulk_export.csv
```

**Do NOT download:** any image referenced by `Media URL`. The pipeline keeps the
URL as metadata only.

**Then run:**
```bash
python scripts/preprocess/preprocess_sf311.py
```
Output: `data/processed/incidents/sf311.parquet` + `sf311.cleaning_stats.json`

---

## 2. Boston 311

| | |
|---|---|
| **Source page** | https://data.boston.gov/dataset/311-service-requests |
| **Licence** | ODC-PDDL — commercial use OK |
| **Recommendation** | **SUBSET — years 2011–2024 only** |
| **Rows** | ~270k per year (~3.8M total) |

**Download the yearly CSVs 2011–2024.** Each year is a separate resource on the
page.

```bash
python scripts/download/download_boston311.py            # resolves URLs via the CKAN API
python scripts/download/download_boston311.py --years 2019 2020 2021 2022 2023 2024
```
Or download manually and name them `data/raw/boston311/boston311_<YEAR>.csv`.

> **Do NOT include 2025 or 2026.** Boston began migrating to a new 311 backend in
> October 2025 and the replacement resource has a *different schema*. Mixing them
> silently corrupts the SLA series, which is the main reason Boston is here at all.

**Then run:**
```bash
python scripts/preprocess/preprocess_boston311.py
```

---

## 3. Chicago 311

| | |
|---|---|
| **Source page** | https://data.cityofchicago.org/d/v6vf-nfxy |
| **Dataset ID** | `v6vf-nfxy` |
| **Direct CSV** | https://data.cityofchicago.org/api/v3/views/v6vf-nfxy/export.csv?accessType=DOWNLOAD |
| **API** | https://data.cityofchicago.org/resource/v6vf-nfxy.json |
| **Licence** | City of Chicago open terms — attribution required |
| **Recommendation** | **SUBSET — exclude two SR types server-side** |
| **Rows** | 13.8M full; ~6.5M after exclusions |

```bash
python scripts/download/download_chicago311.py --estimate-only
python scripts/download/download_chicago311.py
```
This puts the exclusion in the SoQL `$where`, so you never transfer the 7.3M rows
of `311 INFORMATION ONLY CALL` (4.92M) and `Aircraft Noise Complaint` (2.39M).

If you download the bulk CSV manually instead, save it as
`data/raw/chicago311/chicago311_bulk_export.csv` — the preprocessor applies the
same exclusion locally as a safety net.

**Then run:**
```bash
python scripts/preprocess/preprocess_chicago311.py
```

---

## 4. NYC 311 (2010–2019)

| | |
|---|---|
| **Source page** | https://data.cityofnewyork.us/d/76ig-c548 |
| **Dataset ID** | `76ig-c548` |
| **Direct CSV** | https://data.cityofnewyork.us/api/v3/views/76ig-c548/export.csv?accessType=DOWNLOAD |
| **API** | https://data.cityofnewyork.us/resource/76ig-c548.json |
| **Licence** | NYC Open Data terms — commercial use OK |
| **Recommendation** | **SUBSET — filter `complaint_type` server-side** |
| **Rows** | 22.6M full; ~4M after filtering |

The full CSV is ~14 GB and roughly 70% of it is housing, noise and parking
complaints that map to `OUT_OF_SCOPE` and are then discarded. The filter list is
in `config/dataset_config.yaml` under `include_complaint_types`.

```bash
python scripts/download/download_nyc311.py --estimate-only
python scripts/download/download_nyc311.py
```

**Then run:**
```bash
python scripts/preprocess/preprocess_nyc311.py
```

---

## 5. Full run order

Once all four raw sets are in place:

```bash
# one command for everything
python scripts/run_pipeline.py

# or step by step
python scripts/preprocess/preprocess_sf311.py
python scripts/preprocess/preprocess_boston311.py
python scripts/preprocess/preprocess_chicago311.py
python scripts/preprocess/preprocess_nyc311.py
python scripts/preprocess/build_incidents.py
python scripts/preprocess/build_geo_features.py
python scripts/preprocess/build_priority_features.py --accept-draft-config
python scripts/preprocess/build_resolution_dataset.py
python scripts/preprocess/build_hotspot_dataset.py
python scripts/preprocess/build_duplicate_pairs.py
python scripts/validation/data_quality_report.py
python scripts/validation/validate_provenance.py
python scripts/validation/validate_leakage.py
python scripts/validation/validate_outputs.py
```

Optional, network-dependent:
```bash
python scripts/validation/probe_sf_media_urls.py --sample 200
```

> `build_geo_features.py` is the slow step on full data — the backward-looking
> density features are O(rows × local neighbours). On ~10M rows expect tens of
> minutes. `--skip-density` produces the same schema with those five columns NULL
> if you want a fast first pass.

---

## Why no image datasets

The citizen selects the category in the UrbanEye+ app. There is therefore no
image→category task, and severity is out of the product's current UI scope, so
there is no image→severity task either. Downloading ~50,000 images to train
models nothing currently calls would cost tens of GB for no benefit.

The scripts are preserved under `optional_future_vision/scripts/` and the sources
are documented in `DATASET_SOURCES.md`. Two of them also have unresolved licence
questions (Smartathon's original terms could not be verified; the Civic Issue
Dataset states no licence at all), which is a second reason not to pull them in.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No raw files found for: …` | Raw files are missing or misnamed. Check `data/raw/<dataset>/` against the expected filenames in `config/dataset_config.yaml`. |
| `REFUSING TO RUN … status=DRAFT` | Intentional. The priority weights are unapproved. Pass `--accept-draft-config`, or approve them in `config/priority_config.yaml`. |
| `ProvenanceViolation` | Something tried to populate a NULL-provenance field (severity, occurred_at, response_time, priority_label). This is the guard working — do not disable it. |
| Many `UNMAPPED` categories | Expected on first real run; fixture vocabulary is small. Read `reports/unmapped_categories.csv` and extend `config/category_mapping.csv`. |
| `host_not_allowed` from a download script | Your network blocks the portal. Download via a browser and place the file manually. |
| Memory errors | Everything is chunked; if it still happens, lower `--chunk-size` (default 200,000). |
