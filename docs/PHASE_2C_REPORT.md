# UrbanEye+ Phase 2C — ML Integration

Phase 2C integrates the Phase 2B results into the application. Every number in this report comes from code that was run on 2026-10-13. **None of the Phase 2B offline metrics establish production performance.** No prediction has been evaluated against UrbanEye outcomes, and every response says so.

## 1. Decisions taken at review (2C-A)

| Question | Decision | Note |
|---|---|---|
| Who sees incident predictions | All roles, each on the incidents they can already open | Phase 2B advised against showing resolution estimates to citizens. That choice was made explicitly, so every view carries the warnings. |
| Little local history (today's state) | Predict anyway as **EXPERIMENTAL** with reasons; use labelled baselines when the model cannot run; **UNAVAILABLE** only when neither works | See §4 |
| Artifact delivery | Commit a versioned, SHA-256-pinned serving bundle | `backend/ml_bundle/urbaneye-ml-2026.10.13/` (≈13 MB) |
| Prediction log | Yes, as an additive table `ml_predictions` | For the shadow evaluation Phase 2B §15 requires |

Hotspots aggregate every citizen's report locations, so they are restricted to admins and agents.

## 2. Architecture

```
Phase 2B artifacts (read-only)         backend/ml_bundle/<version>/   (committed)
models/{resolution,sla}_v2  ──export──▶  model.joblib (byte-identical copy)
models/hotspot_v2                        preprocessor.json, metadata.json, baselines.json,
                                         feature_ranges.json, fixtures/, MANIFEST.json
                                                   │  SHA-256 verified at startup
FastAPI ───────────────────────────────────────────▼──────────────────────────────
  app/ml/registry.py   load once, verify, report readiness, never raise
  app/ml/service.py    read incidents + first resolve/close event, as of created_at
  app/ml/features.py   history features (port of scripts/phase2b/features.py)
  app/ml/predictors.py model → EXPERIMENTAL/AVAILABLE, else labelled baseline, else UNAVAILABLE
  app/ml/hotspot.py    EWMA (alpha 0.3) weekly counts per 1 km cell — not ML
  app/api/predictions.py  GET /incidents/{id}/predictions · GET /ml/status · GET /hotspots
  ml_predictions table  one row per distinct prediction served
Next.js: IncidentPredictions panel (citizen, agent and admin detail views) · /hotspots page
```

Rules the code enforces:
- Nothing trains at request time.
- Predictions never change an incident's priority, SLA or status.
- Predictions live on their own endpoint, so a prediction failure cannot block incident creation or any workflow step.

## 3. Files

**New**
- **Backend:** `app/ml/{__init__,areas,encoding,registry,features,predictors,hotspot,service}.py`, `app/api/predictions.py`, `app/schemas/prediction.py`, `app/models/ml_prediction.py`, `migrations/add_ml_predictions.sql`, `ml_bundle/urbaneye-ml-2026.10.13/` (11 files), `tests/test_ml_unit.py`, `tests/test_ml_api.py`.
- **ML pipeline:** `scripts/phase2c/export_serving_bundle.py`, `tests/test_phase2c_export.py`.
- **Frontend:** `src/components/IncidentPredictions.tsx`, `src/app/hotspots/page.tsx`.

**Modified (additive edits only)**
- `backend/app/main.py`: model import, router, and `load_registry()` in the startup hook.
- `backend/app/core/config.py`: an ML settings block appended after the existing demo settings. Your uncommitted edits there are preserved.
- `backend/requirements.txt`: exact pins `numpy==2.2.6`, `scipy==1.17.1`, `scikit-learn==1.6.1`, `joblib==1.5.1`.
- `backend/pyproject.toml`: one pytest warning filter.
- `frontend/src/lib/api.ts`: types and API calls appended.
- `frontend/src/app/{admin,agent,dashboard}/page.tsx`: one import plus the panel under the SLA block.
- `frontend/src/components/Navbar.tsx`: a Hotspots link for admins and agents.
- `frontend/src/app/globals.css`: prediction styles appended.

**Untouched:** the files with your earlier uncommitted edits (`incidents.py`, `incident_service.py`, `image_service.py`, `create_demo_data.py`, `test_incidents.py`, `ImageUploadField.tsx`, `README.md`) still have their 2026-10-03 / 09-16 timestamps. Phase 2B experiments and artifacts are also untouched.

## 4. API contracts

### `GET /api/v1/incidents/{id}/predictions[?include_inputs=true]`

Access follows the existing `_assert_incident_access` rule:
- A citizen can read only their own incidents (404 otherwise).
- An agent can read only incidents assigned to them (403 otherwise).
- An admin can read any incident.
- Without a token the response is 401.
- `include_inputs` is honoured for admins only.

```text
{ incident_id, generated_at, features_as_of (= created_at), incident_status, sla_deadline,
  service_area, bundle_version, logged, inputs (admin only),
  resolution: { task, status, method, method_description, reasons[{code,message,detail}],
                fallback_reason, model_version, bundle_version, local_reliability_established:false,
                warnings[], estimate_hours, range_p10_hours, range_p90_hours,
                range_nominal_coverage, range_empirical_coverage (0.596), range_calibrated:false,
                baseline_sample_size },
  sla:        { ...same envelope..., sla_hours (the incident's current value),
                breach_probability + risk_band + decision_threshold   (ML_MODEL only),
                historical_breach_rate + baseline_sample_size          (FALLBACK only) } }
```

**Status values**

| Status | Meaning |
|---|---|
| `AVAILABLE` | The model ran, and every history input is present and inside its training 1st–99th percentile range |
| `EXPERIMENTAL` | The model ran, but at least one of: `INSUFFICIENT_LOCAL_HISTORY` (lists the missing inputs), `FEATURES_OUTSIDE_TRAINING_RANGE` (lists values and ranges), `OUTSIDE_SERVICE_AREA`, or an SLA override value the model never trained on (e.g. 12 h) |
| `FALLBACK` | The model could not run (`ARTIFACT_UNAVAILABLE`, `UNSUPPORTED_CATEGORY`, `SLA_OUT_OF_RANGE`, `INVALID_MODEL_OUTPUT`), so the first applicable baseline below was used. A fallback never has a risk band or a range. |
| `UNAVAILABLE` | `ML_DISABLED`, `NO_SLA`, `INTERNAL_ERROR`, or `NO_APPLICABLE_BASELINE` |

**Fallback baselines, in order of preference**

| Task | 1st choice | 2nd choice | 3rd choice |
|---|---|---|---|
| Resolution | Local category median (needs at least 10 earlier closures) | Local all-category median (needs at least 10) | Training-data category median (US 311 data; the 4 trained categories only) |
| SLA | Local recent breach rate at this SLA window (needs at least 20; Phase 2B's strongest baseline) | Local category breach rate, any age (needs at least 20) | Training-data rate (US data; only at 4/24/72/168 h) |

FIRE_HAZARD and OTHER never fall back to US data, because there is no training data for those categories.

**SLA overrides**
- The prediction always uses the incident's current `sla_hours`, which is the admin `PATCH /sla` value when one was set.
- An override inside 4–168 h is predicted by the multi-horizon model and flagged EXPERIMENTAL if it is not 4, 24, 72 or 168.
- An override outside that range is never extrapolated.
- A cleared SLA gives `NO_SLA`.
- Insufficient history never pushes the risk band up; a unit test covers this.

### `GET /api/v1/ml/status` (admin)
Readiness per component: `ready`, `error`, `model_version`, training period, decision threshold and risk bands, range calibration. Also the bundle version and creation time, the training-data end date (2020-08-28), runtime versions and service areas. File paths are never exposed.

### `GET /api/v1/hotspots?category=&area=&limit=` (admin and agent)
See §7. An invalid category returns 422; an unknown area returns `UNAVAILABLE` / `UNKNOWN_AREA` with the list of valid areas.

## 5. Model loading and artifact delivery

- **Exporting the bundle:** `UrbanEye_ML_Data_Pipeline/scripts/phase2c/export_serving_bundle.py`. It:
  - copies the Phase 2B `model.joblib` and `metadata.json` byte for byte (the manifest records equal hashes);
  - converts the preprocessor to JSON, because the original pickle needs the pipeline's `scripts.train._common` module;
  - writes baselines and feature ranges computed from the exact rows the final models were trained on;
  - writes reference fixtures produced by the original code;
  - refuses to overwrite an existing bundle.
- **Startup checks:** `registry.py` verifies, in this order: `ML_ENABLED`; the MANIFEST SHA-256 against the pinned `ML_BUNDLE_MANIFEST_SHA256`; every file's SHA-256; the numpy, scikit-learn and joblib versions against the manifest (pickles are version-bound, and version warnings are treated as failures); and that each model's `feature_names_in_` equals the encoder's order and the metadata's feature list.
- **When a check fails:** the failure is recorded per component. The app still starts, and the affected predictions fall back or are unavailable.
- **No hot reload.** A new bundle requires a restart.

## 6. Feature construction

`app/ml/features.py` ports the Phase 2B rules exactly. It reads only `incidents` (category, created_at, latitude, longitude) and the **first** `INCIDENT_RESOLVED` or `INCIDENT_CLOSED` event of each incident. As of the incident's `created_at`:

| Group | Definition |
|---|---|
| Workload | Per service area × category: backlog (open, with a 90-day age cap), intake over 7 and 30 days, closures over 7 days, backlog-days, and recent-closure count / mean log duration / median duration (needs at least 10). The same backlog and 7-day intake for all categories in the service area. |
| Spatial | 250 m cells anchored at the service area's south-west corner; counts over the 3×3 block of same-category reports in 7, 30 and 90 days, still-open reports, and all-category reports in 30 days |
| Recent breach rate | At the incident's own SLA window X: incidents reported in [t−X−30d, t−X) that were not resolved within X (needs at least 20) |
| Calendar | Asia/Kolkata local time |
| Warm-up | Workload and spatial counts stay empty for the first 30 days of a service area (90 days for the 90-day window) |

**Correctness checks.** The backend reproduces the pipeline's own feature output with **0 mismatches**: 124 query incidents × 15 features plus 5 horizons, over 600 synthetic incidents that include future closures, same-minute ties, sub-minute closures and the warm-up period. A separate test changes everything that happens after t and confirms no feature changes. Encodings and model outputs match the original pipeline to within 1e-9 on 380 reference cases.

**UrbanEye-specific choices, all documented in the code:**
- A city is a configured bounding box: Bengaluru and Chennai by default, overridable via `ML_SERVICE_AREAS`.
- UrbanEye has no duplicate flag, so no incident is excluded as a duplicate.
- A reopen after a closure is not undone, because the training data records a single closure per incident.

## 7. Hotspot baseline (2C-D)

**This is a weighted moving average, not an ML model.**

| Item | Definition |
|---|---|
| Input | Reports of one category inside one service area (`category`, `created_at`, `latitude`/`longitude`) |
| Geographic aggregation | 1 km square cells, equirectangular projection anchored at the area's south-west corner, x-scale taken at the area's mid-latitude. The same formula as Phase 2B panel C, with a configured origin instead of the data's minimum. |
| Time windows | ISO weeks, Monday 00:00 UTC, as in Phase 2B. The first (partial) week of the area is dropped, and the current week is never counted. |
| Missing data | A week with no report in a cell counts as 0. Series include every cell with at least one report; Phase 2B required at least 8 reports, which UrbanEye is too young for. |
| Method | s₀ = y₀, then s_w = 0.3·y_w + 0.7·s_(w−1); the forecast for the current week is the value after the last complete week. α = 0.3 comes from the bundle. |
| Status | AVAILABLE with at least 4 complete weeks; EXPERIMENTAL with 1–3 weeks or for FIRE_HAZARD/OTHER (`CATEGORY_NOT_EVALUATED`); UNAVAILABLE with none (`NO_DATA` / `NO_COMPLETE_WEEKS`) |
| Output | Area, category, method, α, status, reasons, weeks of history, first and last complete week, forecast week, and per cell: id, centre, bounds, predicted count, last week's count, 4-week mean, reports counted |
| Baseline comparison | Phase 2B, Chicago 1 km grid, offline: EWMA MAE 1.453 vs rolling 4-week mean 1.506 (−3.5%). Not evaluated on UrbanEye data. |
| Limitations | It lags bursts and over-predicts quiet weeks (Phase 2B error analysis). Top-decile recall is about 0.31. |

The cross-city hotspot ML model is not shipped.

**Live state:** Bengaluru has 1 complete week, so the page shows EXPERIMENTAL with no cells yet.

## 8. UI changes (2C-E)

- **`IncidentPredictions` panel** under the SLA block in the citizen, agent and admin detail views. It shows:
  - For each task: a status badge (Model / Experimental / Baseline, not ML / Unavailable).
  - Resolution: the estimate, and the model range explicitly labelled "not a reliable interval (it held about 60% of outcomes in offline tests)".
  - SLA: the risk band and probability for model output; for a fallback, the historical rate with an explicit "no risk band" note.
  - The reasons, the backend's warnings, the as-of time, the bundle version, and the statement that priority and SLA are never changed.
  - Admins also get a collapsible "Model inputs" list.
  - A load failure shows an "unavailable" message and leaves the rest of the view working.
- **`/hotspots`** (admins and agents, plus a Navbar link): area and category selectors, a status banner with reasons and the baseline comparison, and a ranked table of cells linking to OpenStreetMap.
- No value is computed or defaulted in the browser.

## 9. Tests executed (2C-F) — actual results

| Suite | Command | Result |
|---|---|---|
| Backend, full (existing + new) | `backend/.venv/bin/python -m pytest` with Postgres running | **234 passed, 1 skipped**. The skip is pre-existing (`test_auth_roles.py:518`, "Google OAuth is configured in this environment"). |
| of which ML unit | `tests/test_ml_unit.py` | 31 passed |
| of which ML API | `tests/test_ml_api.py` | 11 passed |
| ML pipeline | `python3 -m pytest tests` | **114 passed** (111 Phase 2B + 3 export) |
| v1 reproduction | `scripts/phase2b/verify_v1_reproducible.py` | All three reproduced; artifact SHA-256 unchanged |
| Frontend types | `npx tsc --noEmit` | exit 0 |
| Frontend lint | `npm run lint` | No ESLint warnings or errors |
| Frontend build | `npm run build` | Compiled; 12 static pages including `/hotspots` |
| Live smoke | `uvicorn` + curl | `/health` ok; `/ml/status` returns 401 without a token and shows all 5 components ready with an admin token; `/hotspots` EXPERIMENTAL |

**What these cover**
- Artifact availability, versions, tampering, missing bundle, version mismatch, ML disabled.
- Encoder and model parity, feature parity and leakage.
- Sufficient history, insufficient history, partly missing inputs, out-of-range inputs, outside a service area.
- Missing artifacts, invalid model output (NaN, probability outside [0, 1]), baseline fallback ordering, unsupported category.
- SLA overrides (12 / 200 h / cleared) and the monotonic constraint.
- Authentication and role checks for each role, plus admin-only inputs.
- Response schemas, logging deduplication, and that incidents are unchanged by a prediction.
- With artifacts missing: create, assign, status change and history still work.
- The existing suite still passes: incident creation, assignment, transitions, history, images and admin flows.

**Data safety**
- Row counts were identical before and after all test runs: incidents 6, incident_history 47, users 15, incident_images 4, user_audit_log 20.
- The only database change is the new empty `ml_predictions` table, created by `create_all`.
- Docker was started with `open -a Docker`; the existing container restarted on its own. No volume, reset or delete command was run.

**Not verified:** the UI was not exercised in a browser. Tests cover the API, and type checking and the build cover the UI code.

## 10. Remaining limitations

- **The models are not validated for UrbanEye.**
  - They were trained on 2010–2020 US 311 data, and the database holds 6 incidents.
  - Every real incident is currently EXPERIMENTAL (`INSUFFICIENT_LOCAL_HISTORY`).
  - Workload counts at UrbanEye's scale will sit far below the training range; for example, the training 1st percentile of `wl_intake_7d` is 102.
  - Under these conditions the outputs swing a lot: in the read-only probe, POTHOLE's estimate moves between 25 h and 364 h depending on history. That is why they are labelled experimental.
- **The resolution range is not a calibrated interval:** it held 60% of outcomes in offline tests, against a nominal 80%.
- **The SLA threshold is provisional** (the cost of a missed breach is undefined). FLOOD was weaker than the baseline offline.
- **Citizens see experimental estimates.** That was your decision; the warnings are always shown.
- **Closure and reopen semantics:** the first resolve or close counts as the closure, and reopens are ignored.
- **Service areas are bounding boxes.** Real city boundaries and a city column do not exist.
- **Predictions are computed on request** with one query per incident over its area. That is fine at today's volume; at scale, add a time-bounded query or precomputed features.
- ~~Logging happens on view only~~ — *resolved in the software release (2026-10-14)*: predictions are now also logged on incident creation and on SLA/priority changes (best effort, inside a SAVEPOINT), `backend/scripts/ml_log_predictions.py` backfills idempotently, and `GET /api/v1/ml/shadow-evaluation` joins them to outcomes with right-censoring. See `docs/SOFTWARE_RELEASE_READINESS.md`.
- `.env.example` is gitignored in this repo, so the new settings are documented in `config.py` and here.

## 11. Security considerations

- Every endpoint requires authentication. Incident predictions reuse the existing incident access rule, so citizens cannot probe others' incidents (404) and agents cannot read unassigned ones (403).
- Clients cannot submit their own feature values: every input is computed server-side from the database, so the model cannot be queried with arbitrary inputs.
- Model inputs summarise other incidents (nearby open reports, workload), so they are returned to admins only.
- Hotspots aggregate everyone's report locations, so citizens get a 403.
- `.joblib` files are pickles that can run code when loaded. A pickle is opened only after the pinned manifest hash and every file hash verify. The bundle path comes from server configuration, never from a request.
- Status and error responses contain codes, not file paths or stack traces.
- The log table is written with parameterised inserts. It stores the inputs served, which only admins can read through the API.

## 12. Deployment requirements

- Python 3.11 with `pip install -r backend/requirements.txt`. The exact pins are mandatory: a different scikit-learn version makes the registry refuse the models.
- `backend/ml_bundle/urbaneye-ml-2026.10.13/` must be present in the deployed tree; it is committed.
- Optional settings:

| Setting | Default |
|---|---|
| `ML_ENABLED` | true |
| `ML_BUNDLE_DIR` | the committed bundle |
| `ML_BUNDLE_MANIFEST_SHA256` | pinned to the committed bundle |
| `ML_SERVICE_AREAS` | Bengaluru / Chennai boxes (JSON list to override) |

- For a database not managed by `create_all`, run `backend/migrations/add_ml_predictions.sql` (additive, idempotent).
- After deploying, call `GET /api/v1/ml/status` (admin) and confirm every component shows `ready: true`.

## 13. Retraining and model versioning

- **Version identity:** each bundle is immutable and has a name such as `urbaneye-ml-YYYY.MM.DD`. Each model inside carries its own `model_version` (`resolution_v2.0.0`, `sla_v2.0.0`, `hotspot_v2.0.0`) and training period. Every response and every log row records the bundle and model versions.
- **To ship a new model:**
  1. Retrain offline with `scripts/phase2b/train_final.py`, which only saves models that pass its gates. Write a new `models/<task>_v3`; never overwrite the existing artifacts.
  2. Export a new bundle directory.
  3. Commit it.
  4. Update `ML_BUNDLE_DIR` and `ML_BUNDLE_MANIFEST_SHA256` in one reviewed change.
  5. Restart.
- **Rolling back** is the same config change in reverse, because old bundles are never overwritten.
- **Cadence:** Phase 2B showed the resolution model degrades without retraining roughly every 2 months.
- **What justifies retraining on UrbanEye data:** `ml_predictions` joined to resolve/close events in `incident_history` gives the first local evaluation. Retraining on UrbanEye data is only justified once the Phase 2B thresholds are met: at least 500 resolved incidents per category and at least 26 weeks of history. Until a local evaluation shows the model beats the local baselines, `local_reliability_established` stays false.

## 14. What remains before Edge AI integration

1. Accumulate real UrbanEye outcomes and run the first shadow evaluation from `ml_predictions`: model vs local baselines, per category.
2. Agree on closure and reopen semantics and on the cost of a missed breach, then set the SLA threshold from that cost.
3. ~~Write a backfill script~~ — done (`backend/scripts/ml_log_predictions.py`).
4. Define real service-area boundaries (and ideally a city/ward column), and recalibrate or replace the resolution range (for example with conformal intervals).
5. Gather labelled FIRE_HAZARD and OTHER examples.
6. Only then design Edge AI inputs (camera and IoT sources). Their outputs must enter through the same versioned, hash-verified, status-labelled path, and must not change priority automatically.
