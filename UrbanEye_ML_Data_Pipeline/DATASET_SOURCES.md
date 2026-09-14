# UrbanEye+ — Dataset Sources & Licences

Provenance and licence position for every dataset considered.

**Verification note.** Schemas, row counts and column statistics for the four
core datasets were read from the publishers' live metadata endpoints (Socrata
`columns.json` / `columns.xml`, CKAN `package_show`) on **2026-09-13**. Figures
below come from those reads, not from dataset descriptions. Where something could
not be verified, it says so.

---

## Core datasets — all INCLUDE

### 1. SF 311 Cases

| | |
|---|---|
| Source | https://data.sf.gov/d/vw6y-z8j6 |
| ID | `vw6y-z8j6` |
| Publisher | City and County of San Francisco (DataSF) |
| Licence | **ODC-PDDL 1.0** — http://opendatacommons.org/licenses/pddl/1.0/ |
| Commercial use | Yes | 
| Research use | Yes |
| Attribution | Not required |
| Redistribution | Yes |
| **Decision** | **INCLUDE** |

**Verified:** 8,909,991 rows, ~52 columns, 2008-07-01 → present, nightly updates.
Latitude present on 8,881,125 rows — **but 551,131 of those are exactly `0.0, 0.0`**
(the null-island sentinel), which the pipeline nulls.

**Role:** incident history, resolution time, temporal/geographic patterns, weak
duplicate signal via "Case is a Duplicate" in Status Notes.

**`Media URL` caveat:** this column carries **no cached-contents statistics** in
the portal metadata, so its fill rate cannot be determined without downloading.
Retained as metadata only; `probe_sf_media_urls.py` samples it. Photographs are
public submissions — treat faces, number plates and house numbers as a privacy
obligation even under a public-domain dedication.

### 2. Boston 311 Service Requests

| | |
|---|---|
| Source | https://data.boston.gov/dataset/311-service-requests |
| Publisher | City of Boston, Department of Innovation and Technology |
| Licence | **ODC-PDDL** — stated on the portal |
| Commercial / research / redistribution | Yes |
| **Decision** | **INCLUDE — years 2011–2024 only** |

**Verified:** per-year CSVs 2011–2026; the 2021 file at 273,951 rows / 29 columns.

**Role:** the only source with an explicit SLA target (`TARGET_DT`) and a
compliance flag (`OnTime_Status`). Also the only source with before/after photo
URLs, though no image is downloaded.

> **2025+ excluded.** Boston began migrating to a new 311 backend in October 2025
> and the replacement resource has a different schema. The portal also noted that
> during 2026 some case types were mistakenly dropped from the legacy dataset
> (reported fixed 2026-08-25). Mixing schemas would corrupt the SLA series.

### 3. Chicago 311 Service Requests

| | |
|---|---|
| Source | https://data.cityofchicago.org/d/v6vf-nfxy |
| ID | `v6vf-nfxy` |
| Publisher | City of Chicago |
| Licence | City of Chicago Open Data Terms of Use |
| Commercial / research | Yes |
| Attribution | **Required** |
| **Decision** | **INCLUDE** |

**Verified:** 13,796,455 rows, 48 columns, from 2018-12-18.
`DUPLICATE = true` on **707,269** rows; `PARENT_SR_NUMBER` non-null on **706,730**.

**Role:** the duplicate-detection backbone. That parent pointer is a genuine
municipal determination that two requests describe the same physical incident —
the best duplicate supervision in open civic data. Also the cleanest
spatio-temporal panel (ward, community area, police beat, precinct, plus
pre-computed hour/day/month).

**Excluded server-side:** `311 INFORMATION ONLY CALL` (4,921,249 rows) and
`Aircraft Noise Complaint` (2,389,257) — 7.3M rows of non-incidents.

**No free-text description field exists.** `description` is NULL for all Chicago
rows by design, not by omission.

### 4. NYC 311 Service Requests 2010–2019

| | |
|---|---|
| Source | https://data.cityofnewyork.us/d/76ig-c548 |
| ID | `76ig-c548` |
| Publisher | City of New York |
| Licence | NYC Open Data Terms of Use |
| Commercial / research / redistribution | Yes |
| **Decision** | **INCLUDE** |

**Verified:** 22,613,917 rows, 48 columns. `Due Date` non-null on **8,663,794**
rows, documented by the publisher as based on complaint type and internal SLAs.

**Role:** second independent SLA source, cross-validating Boston.

**Known data quality issues, handled by the cleaning rules:** `Closed Date` ranges
from `1899-12-31` to `3027-03-30`. Both are nulled by the valid-window rule.

---

## Not required for the current scope

The citizen selects the category in the app, so there is no image→category or
image→severity task. **Do not download these.** Scripts are preserved under
`optional_future_vision/scripts/`.

| Dataset | Source | Licence | Why not now |
|---|---|---|---|
| **RDD2022** | https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547 | **Unverified** — read from the Figshare record at download time | No image task. 47,420 images incl. an India subset; the best road-damage source if image detection is ever added. |
| **SDAIA Smartathon Theme 1** | https://smartathon.hackerearth.com/ | **Unverified at source**; Roboflow mirror declares CC BY 4.0 | No image task, **and** the original competition terms could not be read. A mirror cannot grant rights the original publisher did not. |
| **TACO** | http://tacodataset.org/ | Per-image CC BY 4.0 or ODbL | Item-level litter granularity (one bottle, one cigarette butt); no tabular value. |
| **Mapillary Vistas** | https://research.mapillary.com/publication/iccv17a/ | **CC BY-NC-SA** | NonCommercial clause is incompatible with a deployed product. Also labels object *presence*, not *fault state*. |
| **Civic Issue Dataset (IIT-K / IBM Research India)** | https://github.com/Sshanu/civic_issue_dataset | **None stated** | **Excluded because the licence could not be verified.** Notable as the only Indian civic-issue image dataset found and the only one covering manholes — worth contacting the authors for an explicit grant. |

### Also rejected

| Dataset | Why |
|---|---|
| Kaggle "Civic and municipality complaint system dataset" | No stated provenance or methodology; advertises ready-made priority labels. Small user-uploaded CSVs of that shape are commonly synthetic. Using it would mean the priority engine learns from an unknown generator. |
| CPGRAMS aggregates (data.gov.in) | State-level aggregate counts, not per-complaint records. No coordinates, no resolution times. |

---

## Indian data: the honest position

There is **no bulk-downloadable, record-level Indian municipal complaint
dataset**. The Swachhata / SBM platform collects exactly what UrbanEye+ needs —
geo-tagged photo, category, ward, resolution status — across 4,905 cities, but
publishes only dashboards, not open bulk data.

This is the single strongest argument for UrbanEye+ accumulating its own
operational data. It is also why `priority_label` is NULL: no Indian source of
genuine priorities exists to import.

If you identify a high-quality Indian civic dataset, report it rather than adding
it silently — source, URL, licence, coverage, fields, and whether it should be
included now.

---

## Licence summary

| Dataset | Licence | Commercial | Decision |
|---|---|---|---|
| SF 311 | ODC-PDDL 1.0 | Yes | **INCLUDE** |
| Boston 311 | ODC-PDDL | Yes | **INCLUDE** |
| Chicago 311 | Chicago Open Data Terms | Yes (attribution) | **INCLUDE** |
| NYC 311 | NYC Open Data Terms | Yes | **INCLUDE** |
| RDD2022 | Unverified | Unknown | Not required |
| Smartathon | Unverified | Unknown | Not required |
| TACO | CC BY 4.0 / ODbL | Mixed | Not required |
| Mapillary Vistas | CC BY-NC-SA | **No** | Excluded |
| Civic Issue Dataset | None stated | Unknown | Excluded |

All four core datasets are commercial-use compatible. Only Chicago requires
attribution. `reports/data_quality.json` reproduces this table from
`config/dataset_config.yaml` on every run.
