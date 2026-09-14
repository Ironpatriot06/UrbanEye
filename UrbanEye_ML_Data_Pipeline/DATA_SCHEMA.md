# UrbanEye+ — Data Schema

*Generated from code by `scripts/validation/generate_schema_doc.py` on
2026-09-14 18:00 UTC. Do not edit by hand.*

| Provenance | Meaning |
|---|---|
| `SOURCE` | Taken verbatim (or normalised) from the publisher |
| `DERIVED` | Computed by this pipeline from SOURCE fields |
| `SYSTEM` | Produced at runtime by the UrbanEye+ backend (India PostGIS). **NULL here.** |
| `NULL` | Genuinely unavailable in every configured source. **Enforced empty.** |


The `NULL` rows are not oversights. `scripts/utils/schema.py::assert_null_fields_empty`
raises `ProvenanceViolation` if any of them is ever populated, and
`scripts/validation/validate_provenance.py` actively attempts to populate them to
prove the guard works.

---

## 1. `incidents`

`data/processed/incidents/all_incidents.parquet`
(and `all_incidents_prioritised.parquet` after the priority step)

| Field | Type | Provenance | Notes |
|---|---|---|---|
| `incident_id` | `string` | DERIVED | f'{source_dataset}:{source_incident_id}' — globally unique |
| `source_dataset` | `string` | DERIVED | which publisher this row came from |
| `source_incident_id` | `string` | SOURCE | CaseID / CASE_ENQUIRY_ID / SR_NUMBER / Unique Key |
| `city` | `string` | SOURCE | publisher's city — never rewritten to an Indian city |
| `category` | `string` | DERIVED | source category mapped via config/category_mapping.csv |
| `subcategory` | `string` | SOURCE | publisher's second-level label; NULL for Chicago |
| `source_category_raw` | `string` | SOURCE | retained so mapping decisions stay auditable |
| `description` | `string` | SOURCE | AGENCY-written text (resolution notes / case titles). NOT citizen text. Forbidden as a report-time feature. |
| `latitude` | `float64` | SOURCE | publisher WGS84. US coordinates stay US coordinates. |
| `longitude` | `float64` | SOURCE |  |
| `address` | `string` | SOURCE |  |
| `zone_id` | `string` | SOURCE | ward / community area / analysis neighbourhood / community board. A US admin unit, NOT an Indian zone. |
| `zone_type` | `string` | DERIVED | names which administrative unit zone_id refers to |
| `reported_at` | `datetime64[ns, UTC]` | SOURCE | when the citizen reported it |
| `occurred_at` | `datetime64[ns, UTC]` | **NULL** | NO configured source records when the problem began. Always NULL. Never derived from reported_at. |
| `closed_at` | `datetime64[ns, UTC]` | SOURCE | POST-RESOLUTION. Never a feature. |
| `status` | `string` | SOURCE | normalised open/closed/cancelled/other. POST-RESOLUTION. |
| `department` | `string` | SOURCE | responsible agency / owner department |
| `priority_baseline` | `string` | DERIVED | P1-P4 from the weighted engine in config/priority_config.yaml. NOT ground truth. |
| `priority_score` | `float64` | DERIVED | normalised 0-1 score behind priority_baseline |
| `priority_reasons` | `string` | DERIVED | pipe-separated explanation of the top contributing features |
| `priority_confidence` | `string` | DERIVED | HIGH/MEDIUM/LOW — a deterministic tier over how much of the policy was evaluable, per priority_config.confidence_tiers. Not a probability. |
| `priority_features_available` | `Int64` | DERIVED | how many weighted inputs were non-null for this row |
| `priority_feature_coverage` | `float64` | DERIVED | share of TOTAL policy weight those inputs carry, 0-1. Published so priority_confidence can always be audited. |
| `priority_method` | `string` | DERIVED | priority_config.yaml@<config_version> |
| `priority_label` | `string` | **NULL** | GENUINE operational priority. No public 311 dataset provides one. Always NULL here. |
| `priority_label_source` | `string` | **NULL** | provenance of a genuine label. Always NULL until real labels exist. |
| `is_ground_truth` | `boolean` | DERIVED | always False for priority in this pipeline |
| `sla_target_hours` | `float64` | DERIVED | Boston TARGET_DT-OPEN_DT; NYC Due Date-Created Date; NULL for SF/Chicago |
| `sla_met` | `boolean` | DERIVED | resolution_time_hours <= sla_target_hours, only when both valid. POST-RESOLUTION. |
| `sla_hours_policy` | `float64` | DERIVED | UrbanEye+ business-rule SLA for priority_baseline, from priority_config.sla_hours. Product config, not evidence. |
| `response_time_hours` | `float64` | **NULL** | NO configured source records a first-response timestamp. Always NULL. |
| `resolution_time_hours` | `float64` | DERIVED | closed_at - reported_at. NULL when still open, when the duration is negative/absurd, or when it is below cleaning.resolution_time.min_observable_seconds. POST-RESOLUTION — target only, never a feature. |
| `resolution_instant_closure` | `boolean` | DERIVED | closure recorded sooner than the source timestamps can measure (see cleaning.resolution_time.min_observable_seconds). Such rows keep status/closed_at but have a NULL resolution_time_hours. NULL when the case is not closed. |
| `is_duplicate` | `boolean` | SOURCE | Chicago DUPLICATE flag; weak text signal from SF Status Notes; NULL elsewhere |
| `parent_incident_id` | `string` | SOURCE | Chicago PARENT_SR_NUMBER only |
| `report_channel` | `string` | SOURCE | phone / web / mobile / social / other |
| `image_url` | `string` | SOURCE | SF Media URL or Boston SubmittedPhoto. METADATA ONLY — no image is ever downloaded. |
| `image_url_after` | `string` | SOURCE | Boston ClosedPhoto only. Metadata only. |
| `severity` | `Int64` | **NULL** | Out of current product scope AND unavailable. No public source provides a defensible grade. Always NULL. |
| `coord_outside_city_bbox` | `boolean` | DERIVED | QA flag, not a filter |

### Fields that are always NULL, and why

- **`occurred_at`** — NO configured source records when the problem began. Always NULL. Never derived from reported_at.
- **`priority_label`** — GENUINE operational priority. No public 311 dataset provides one. Always NULL here.
- **`priority_label_source`** — provenance of a genuine label. Always NULL until real labels exist.
- **`response_time_hours`** — NO configured source records a first-response timestamp. Always NULL.
- **`severity`** — Out of current product scope AND unavailable. No public source provides a defensible grade. Always NULL.

### Fields that are POST-RESOLUTION (never valid as model features)

`closed_at`, `status`, `resolution_time_hours`, `sla_met`, `description`

These are known only after the case closes. `validate_leakage.py` asserts they do
not appear in any feature table.

---

## 2. `incident_time_features`

`data/processed/features/incident_time_features.parquet` — joins to `incidents` on `incident_id`.

| Field | Type | Provenance | Notes |
|---|---|---|---|
| `incident_id` | `string` | DERIVED | join key to incidents |
| `reported_at_local` | `datetime64[ns]` | DERIVED | reported_at converted to the source city's timezone (dataset_config.yaml: cleaning.timestamps.source_timezones). tz-naive wall clock, NOT UTC. |
| `local_timezone` | `string` | DERIVED | IANA timezone used for the conversion |
| `hour` | `Int64` | DERIVED | hour of day in the source city's LOCAL time, not UTC |
| `day_of_week` | `Int64` | DERIVED | Monday=0, local time |
| `month` | `Int64` | DERIVED | local time |
| `year` | `Int64` | DERIVED | local time |
| `is_weekend` | `boolean` | DERIVED | Saturday/Sunday in local time |
| `is_night` | `boolean` | DERIVED | temporal_policy night window, local time |
| `hour_utc` | `Int64` | DERIVED | audit column: the UTC hour the local hour was converted from |
| `date_local` | `string` | DERIVED | local calendar date, for grouping |

`reported_at` is stored in UTC so four cities share one axis. Every feature above
is derived AFTER converting it to the source city's own timezone, using the single
mapping in `dataset_config.yaml -> cleaning.timestamps.source_timezones`. Taking
`hour` or `is_night` off the UTC timestamp would encode the city's longitude as
if it were a time-of-day effect; `hour_utc` is kept so the conversion stays
auditable.

---

## 3. `incident_geo_features`

`data/processed/geo_features/incident_geo_features.parquet` — joins to `incidents` on `incident_id`.

| Field | Type | Provenance | Notes |
|---|---|---|---|
| `incident_id` | `string` | DERIVED | join key to incidents |
| `near_school` | `boolean` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `distance_to_nearest_school_m` | `float64` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `near_hospital` | `boolean` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `distance_to_nearest_hospital_m` | `float64` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `near_metro` | `boolean` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `distance_to_nearest_metro_m` | `float64` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `near_rail_station` | `boolean` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `distance_to_nearest_rail_station_m` | `float64` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `near_public_transport` | `boolean` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `distance_to_nearest_public_transport_m` | `float64` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `near_fire_station` | `boolean` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `distance_to_nearest_fire_station_m` | `float64` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `near_police_station` | `boolean` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `distance_to_nearest_police_station_m` | `float64` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `near_government_building` | `boolean` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `distance_to_nearest_government_building_m` | `float64` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `near_major_road` | `boolean` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `distance_to_major_road_m` | `float64` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `road_class` | `string` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `near_intersection` | `boolean` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `distance_to_nearest_intersection_m` | `float64` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `metro_city` | `boolean` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `urban_area` | `boolean` | **SYSTEM** | REQUIRES EXTERNAL GEOSPATIAL DATA. NULL for US 311 rows. Populated at runtime by the India PostGIS layer. |
| `nearby_similar_incidents_24h` | `Int64` | DERIVED | computed from 311 history (DERIVED) |
| `nearby_similar_incidents_7d` | `Int64` | DERIVED | computed from 311 history (DERIVED) |
| `nearby_similar_incidents_30d` | `Int64` | DERIVED | computed from 311 history (DERIVED) |
| `local_incident_density` | `Int64` | DERIVED | computed from 311 history (DERIVED) |
| `category_incident_density` | `float64` | DERIVED | local_incident_density restricted to this category, as a share |
| `hours_since_previous_similar_incident` | `float64` | DERIVED | Hours between this report and the most recent EARLIER report of the same category within radius_m. NULL when there was none inside window_hours. Small values mean the same problem is being re-reported at the same place right now, which the count features cannot express: 5 reports spread over a month and 5 reports in the last hour give the same 30-day count.
 |
| `geo_provider` | `string` | DERIVED | which provider produced this row, e.g. 'null_provider' or 'india_postgis' |
| `geo_features_available` | `Int64` | DERIVED | count of non-null geospatial features |

**23 of these columns require an external geospatial layer** and are NULL
for every US 311 row. **8 are computed now** from 311 history alone.

- Requires PostGIS: `distance_to_major_road_m`, `distance_to_nearest_fire_station_m`, `distance_to_nearest_government_building_m`, `distance_to_nearest_hospital_m`, `distance_to_nearest_intersection_m`, `distance_to_nearest_metro_m`, `distance_to_nearest_police_station_m`, `distance_to_nearest_public_transport_m`, `distance_to_nearest_rail_station_m`, `distance_to_nearest_school_m`, `metro_city`, `near_fire_station`, `near_government_building`, `near_hospital`, `near_intersection`, `near_major_road`, `near_metro`, `near_police_station`, `near_public_transport`, `near_rail_station`, `near_school`, `road_class`, `urban_area`
- Computed now: `category_incident_density`, `geo_features_available`, `geo_provider`, `hours_since_previous_similar_incident`, `local_incident_density`, `nearby_similar_incidents_24h`, `nearby_similar_incidents_30d`, `nearby_similar_incidents_7d`

---

## 4. `duplicate_pairs`

`data/processed/duplicates/duplicate_pairs.parquet`

| Field | Type | Provenance | Notes |
|---|---|---|---|
| `incident_a` | `string` | SOURCE | parent incident_id |
| `incident_b` | `string` | SOURCE | child incident_id |
| `same_incident` | `boolean` | SOURCE | TRUE from Chicago PARENT_SR_NUMBER; FALSE from documented negative sampling |
| `distance_meters` | `float64` | DERIVED | haversine; NULL if either coordinate is missing |
| `time_difference_hours` | `float64` | DERIVED |  |
| `category_match` | `boolean` | DERIVED |  |
| `text_similarity` | `float64` | DERIVED | NULL for Chicago — Chicago publishes no free-text description |
| `source_dataset` | `string` | DERIVED |  |
| `image_similarity` | `float64` | **NULL** | NULL. The labelled source (Chicago) has no photographs. |
| `negative_strategy` | `string` | DERIVED | which negative-sampling rule produced a FALSE pair; NULL for positives |
| `split` | `string` | DERIVED | group-aware; a duplicate group never spans splits |

---

## 5. ML-ready tables

| Table | Path | Target | Target available today? |
|---|---|---|---|
| `priority_dataset` | `data/processed/priority/` | `priority_label` | **No** — NULL everywhere. `priority_baseline` is provided but is DERIVED, not a target. |
| `resolution_dataset` | `data/processed/resolution/` | `resolution_time_hours`, `sla_breach` | Yes |
| `hotspot_dataset` | `data/processed/hotspot/` | `future_incident_count`, `future_incident_flag` | Yes |
| `duplicate_pairs` | `data/processed/duplicates/` | `same_incident` | Yes (Chicago only) |
| `urbaneye_ml` | `data/processed/ml/` | `resolution_time_hours`, `sla_breach` | Yes. `priority_label` is present and empty; see below. |

### `urbaneye_ml` column roles

Every column in the consolidated table carries a declared role in
`data/processed/ml/urbaneye_ml.manifest.json`:

| Role | Meaning |
|---|---|
| `identifier` | keys, not features |
| `predictor` | knowable at report time, safe to train on |
| `predictor_unavailable` | declared with the right name and dtype, NULL because no reference data exists |
| `target` | a genuinely observed outcome |
| `target_support` | describes the target (censoring flag, SLA window), never an input |
| `policy_metadata` | the rule engine's own output — a benchmark to beat, never a feature and never a label |
| `label_provenance` | what is missing and why, machine-readable |
| `context_metadata` | coordinates, timestamps, split — for auditing and joining, not for training |

Raw `latitude`/`longitude` are context, not predictors, on purpose: a model that
learns "latitude 41.88 is high priority" has learned Chicago.

---

## 6. Category taxonomy

`ROAD_DAMAGE`, `POTHOLE`, `GARBAGE_DUMPING`, `STREETLIGHT_FAULT`, `TRAFFIC_SIGNAL_FAULT`, `FOOTPATH_DAMAGE`, `OPEN_MANHOLE`, `DRAINAGE_SEWER`, `WATER_LEAKAGE_WATERLOGGING`, `FALLEN_TREE`, `PUBLIC_INFRASTRUCTURE_DAMAGE`, `SIGNAGE_DAMAGE`, `GRAFFITI_VISUAL_POLLUTION`, `CONSTRUCTION_OBSTRUCTION`, `OTHER`

Plus three sentinels that stay in `incidents` for auditability but never reach a
model-ready table: `REVIEW_REQUIRED`, `OUT_OF_SCOPE`, `UNMAPPED`.

Mapping lives in `config/category_mapping.csv`. Unmapped source values are never
guessed — they become `UNMAPPED` and are reported with row counts in
`reports/unmapped_categories.csv`.
