# UrbanEye+ — Geospatial Features

The contract between this pipeline and the future PostGIS enrichment layer.

---

## 1. The problem this solves

UrbanEye+ needs POI proximity features — school, hospital, metro, major road,
intersection — to assign priority sensibly. **The public US 311 datasets contain
none of them.** They give you a latitude and a longitude, nothing more.

There are three ways to handle that gap and only one is honest:

| Approach | Consequence |
|---|---|
| Omit the columns | The schema changes shape the day PostGIS arrives; every downstream consumer breaks. |
| Fill with plausible values | Fabrication. The model learns noise and you cannot tell. |
| **Emit every column, fill with NULL, record the provider** | Schema stable from day one; values stay honest. |

This pipeline does the third.

---

## 2. Feature inventory

31 columns in `incident_geo_features`. **23 require an external geospatial layer
and are NULL today. 8 are computed now.**

### Requires PostGIS — NULL until an Indian layer is connected

| Group | Columns |
|---|---|
| POI proximity | `near_school` / `distance_to_nearest_school_m`, `near_hospital` / `distance_to_nearest_hospital_m`, `near_metro` / `distance_to_nearest_metro_m`, `near_rail_station` / `distance_to_nearest_rail_station_m`, `near_public_transport` / `distance_to_nearest_public_transport_m`, `near_fire_station` / `distance_to_nearest_fire_station_m`, `near_police_station` / `distance_to_nearest_police_station_m`, `near_government_building` / `distance_to_nearest_government_building_m` |
| Road network | `near_major_road`, `distance_to_major_road_m`, `road_class`, `near_intersection`, `distance_to_nearest_intersection_m` |
| Area context | `metro_city`, `urban_area` |

### Computed now, from 311 history alone

| Column | Window | Radius | Scope |
|---|---|---|---|
| `nearby_similar_incidents_24h` | 24 h | 250 m | same category |
| `nearby_similar_incidents_7d` | 7 d | 250 m | same category |
| `nearby_similar_incidents_30d` | 30 d | 250 m | same category |
| `local_incident_density` | 30 d | 250 m | all categories |
| `category_incident_density` | 30 d | 250 m | share of local that is this category |

These need only past incidents, not a POI layer, so they work today.

Thresholds, windows and radii are all in `config/feature_config.yaml`.

---

## 3. Leakage: the one real trap here

A naive "count incidents within 250 m" counts incidents **that had not yet been
reported**. That is future information, and it inflates offline scores for a
feature that cannot exist at inference time.

Every window here is **strictly backward**: for a row reported at time `t`, only
incidents with `reported_at < t` are counted. Not `<=` — strictly less than, so a
row never counts itself.

This is asserted independently. `validate_leakage.py` check **LEAK-4** resamples
rows, recomputes the neighbour count from the incidents table with an explicit
`reported_at < t` filter, and compares. On the fixture run: **0 mismatches across
150 resampled rows**. The validator does not trust the builder that produced the
value — it re-derives it.

### Performance note

Exact pairwise distance over millions of rows is impossible (10^13 comparisons
for 10M rows). The builder blocks on a ~250 m grid: rows are sorted by time once,
each row is assigned a cell, and candidates are drawn from that cell plus the
surrounding ring. Within a cell the candidates are already time-ordered, so the
backward window is two `searchsorted` calls and the variable-length slices are
flattened into one pair array evaluated with vectorised numpy.

**Measured: 1.9M rows, 429M candidate pairs, 27 seconds.**
`--skip-density` emits the same schema with those five columns NULL if you want
a fast first pass anyway.

### A recency feature was added alongside the counts

`hours_since_previous_similar_incident` — hours back to the most recent EARLIER
report of the same category within 250 m, NULL when there was none inside 30
days. It costs nothing extra (it reuses the same candidate pairs) and it says
something the counts cannot: five reports spread over a month and five reports in
the last hour produce the same 30-day count. NULL here means "no qualifying
earlier report", never zero. Verified against a brute-force implementation in
`tests/test_pipeline.py::test_recency_matches_brute_force`.

### Two defects that were fixed here

1. **The ring was too small.** The config specified `neighbour_ring: 1`, i.e. a
   3x3 block. At 0.0025 degrees a cell is ~277 m tall but only ~207 m wide at
   Chicago's latitude, so a neighbour 240 m away in longitude sits TWO cells out
   and was silently never counted. The ring is now derived per axis from
   `radius_m` and the cell's own latitude (`required_rings()`), which makes the
   block a true superset again. `tests/test_pipeline.py` compares the result
   against a brute-force implementation of the definition and requires exact
   equality.
2. **The window was applied after candidate selection, not before.** Every row
   was compared against every incident that had ever occurred in its cell block,
   over the whole corpus, in a Python loop. That is what made the step
   unfinishable on real data — and an unrun step is why the density columns were
   100% NULL downstream.

---

## 4. The provider interface

```python
class GeoProvider:
    name = "abstract"

    def poi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df has incident_id, latitude, longitude.
        Return a frame, same index, with the POI/road/area columns populated.
        """
```

### `NullProvider` — the only one implemented

Returns the full column set, entirely NULL. Correct and honest for US
coordinates: no Indian POI layer is connected, and US POI data would be actively
misleading for a product deploying in India.

Every row records `geo_provider = "null_provider"` and
`geo_features_available`, so a consumer can always tell what it is looking at.

### Adding `IndiaPostGISProvider`

1. Implement `poi_features` against your PostGIS instance.
2. Register it in `PROVIDERS` in `scripts/preprocess/build_geo_features.py`.
3. Run `build_geo_features.py --provider india_postgis`.

Nothing else changes. The schema, the priority engine and every downstream table
already expect these columns; they simply stop being NULL.

**One rule for any implementation:** do not invent values for coordinates outside
your coverage. Return NULL and let `geo_features_available` reflect reality. The
priority engine's `strict_available_only` mode depends on NULL genuinely meaning
"unknown".

---

## 5. Candidate Indian data sources

Not evaluated in depth, and **licences must be verified before use**:

| Source | Provides | Licence note |
|---|---|---|
| OpenStreetMap (Overpass / Geofabrik India) | schools, hospitals, metro/rail stations, road network + classes, intersections | ODbL — **share-alike**; check compatibility with your product |
| Survey of India / state GIS portals | authoritative boundaries, roads | Varies; often restrictive |
| Municipal GIS (GCC, BBMP, MCGM…) | ward boundaries, civic assets | Usually per-city agreement |
| Census of India / MoHUA | urban/metro classification | Generally open |
| Bhuvan (ISRO) | imagery, some thematic layers | Registration required |

OSM is the pragmatic starting point: it covers every feature in the inventory
above. The ODbL share-alike clause is the thing to check with whoever handles
licensing, since it can attach obligations to derived databases.

### Suggested PostGIS sketch

```sql
-- Nearest school, with distance, using a geography index
SELECT i.incident_id,
       ST_Distance(i.geom::geography, s.geom::geography) AS distance_m
FROM incidents i
CROSS JOIN LATERAL (
    SELECT geom FROM poi_schools s
    ORDER BY s.geom <-> i.geom      -- KNN index scan
    LIMIT 1
) s;
```

Then `near_school = distance_m <= threshold`, with the threshold from
`config/feature_config.yaml` so the pipeline and the backend cannot disagree.

---

## 6. What is deliberately not built

- **Traffic / footfall prediction.** Explicitly out of scope. Basic contextual
  proximity is sufficient for v1 priority.
- **US POI enrichment.** Technically possible, deliberately not done. It would
  populate the columns with values that mean nothing for an Indian deployment and
  would make the NULLs — which are currently informative — invisible.
- **Raw coordinates as model features.** Latitude and longitude are inputs to
  feature *derivation*, never features themselves. A model that learns
  "latitude 37.77 is high priority" has learned San Francisco, and that knowledge
  is worse than useless in Chennai.


---

## 5. The provider architecture

`scripts/preprocess/geo_providers.py` defines the contract. A provider receives
`incident_id`, `latitude`, `longitude` and returns every declared POI / road /
area column, with the declared dtype, always.

Three rules, and the third is the one that matters:

1. return every declared column;
2. compute what you can from real reference data;
3. **return NULL for anything you cannot compute** — never a sentinel, never an
   imputed mean, never a plausible-looking number. A row outside the provider's
   coverage comes back NULL, and `coverage()` reports how many rows that was.

| Provider | Status | Behaviour |
|---|---|---|
| `null_provider` | **default** | full schema, every value NULL. Correct for US 311 data. |
| `reference_layer` | implemented | nearest-POI distances from local files; absent layers stay NULL |
| `india_postgis` | documented stub | raises `NotImplementedError` rather than degrading silently to NULLs |

Select with `build_geo_features.py --provider <name>`, or set
`geo_provider.name` in `config/feature_config.yaml`.

### What external data would actually be needed

To populate the 23 NULL columns, supply point or line geometry per layer, in
WGS84, for the deployment city:

| Layer | Geometry | Feature it fills | Typical source |
|---|---|---|---|
| `school` | points | `near_school`, `distance_to_nearest_school_m` | OSM `amenity=school`; state education registry |
| `hospital` | points | `near_hospital`, `distance_to_nearest_hospital_m` | OSM `amenity=hospital`; health department registry |
| `metro` | points | `near_metro`, `distance_to_nearest_metro_m` | metro rail operator station list |
| `rail_station` | points | `near_rail_station`, `distance_to_nearest_rail_station_m` | railway operator |
| `public_transport` | points | `near_public_transport`, `distance_to_nearest_public_transport_m` | GTFS `stops.txt` |
| `fire_station` | points | `near_fire_station`, `distance_to_nearest_fire_station_m` | OSM `amenity=fire_station` |
| `police_station` | points | `near_police_station`, `distance_to_nearest_police_station_m` | OSM `amenity=police` |
| `government_building` | points | `near_government_building`, `distance_to_nearest_government_building_m` | municipal asset register |
| `major_road` | lines | `near_major_road`, `distance_to_major_road_m`, `road_class` | OSM `highway=primary\|secondary\|trunk` |
| `intersection` | points | `near_intersection`, `distance_to_nearest_intersection_m` | derived from the road network |
| area classification | polygons | `metro_city`, `urban_area` | Census / MoHUA classification |

Declare them in `data/external/reference_layers.yaml`:

```yaml
crs: EPSG:4326
max_search_radius_m: 5000
layers:
  school:
    path: schools_chennai.parquet     # relative to data/external/
    lat_column: lat
    lon_column: lon
    licence: "ODbL — OpenStreetMap contributors"
```

Only the layers present are computed; everything else stays NULL and is named in
`reports/geo_features_summary.json` under `provider_coverage`. Beyond
`max_search_radius_m` the distance stays NULL while `near_*` is `False` — "there
is no school within 5 km" is knowledge and is recorded, but a precise distance to
something 12 km away is not, and inventing one would be fabrication.

**Two things to settle before switching a provider on:** the licence of every
layer (it propagates to anything trained on it), and the coverage polygon —
outside it, the provider must return NULL rather than the distance to whatever
happens to be nearest inside it.

### Why the current corpus stays NULL

This repository ships no reference geodata: `data/external/` does not exist. And
even with a US POI layer these columns would be the wrong ones — the product
deploys in India, where "200 m from a school" is a different exposure than it is
in suburban Boston. The columns are emitted with the right name and dtype so the
schema is stable for the day a real layer arrives.
