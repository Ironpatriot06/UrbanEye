# UrbanEye+ — Software-Only Release Readiness

**Scope:** the software-only variant, which runs with no cameras, sensors, ESP32 or Raspberry Pi.

**Verified:** 2026-10-13 → 2026-10-14 on macOS (arm64): Python 3.11.8, Node 25.2.1 / npm 11.6.2, Docker with `postgis/postgis:16-3.4-alpine`, Google Chrome (driven by Playwright 1.48.2).

Every result below was produced by a command that was run. Nothing has been committed.

## 1. Summary

The application runs end to end without hardware. The pieces that were not yet verified are now verified:
- **Browser flows:** citizen, agent and admin journeys exercised in real Chrome.
- **Clean checkout:** a fresh install of what would be committed, on a brand-new database.
- **Prediction logging:** completed and tested.

The audit found **seven defects**, and all were fixed:

| # | Defect | Fix | Regression test |
|---|---|---|---|
| 1 | A prediction-logging failure called `db.rollback()` on the whole session, which could undo the caller's uncommitted work (caught by a new test) | Every ML read and insert now runs in its own SAVEPOINT. A failure rolls back ML work only. | `test_logging_failure_never_blocks_incident_writes`, `test_database_error_inside_prediction_is_contained` |
| 2 | Fresh databases: the README's migrations fail before tables exist, and without them the append-only triggers on the audit tables are never installed | `backend/scripts/init_db.py`: PostGIS, then tables, then every migration in order. Idempotent. | Run on a brand-new database and re-run on an initialised one (§8) |
| 3 | `.env.example` was git-ignored, so a clone had no configuration template | Narrow `.gitignore` change: real `.env` files stay ignored | `git check-ignore` (§9) |
| 4 | A fresh pipeline venv from its `requirements.txt` lacked scikit-learn, which the tests and Phase 2B scripts import | Added `scikit-learn==1.6.1` and documented the verified versions | Clean-checkout pipeline run (§8) |
| 5 | Under pandas 3 (allowed by the pipeline requirements), the row-wise priority engine scored a missing category as a category named "nan" | One-line fix in `build_priority_features.py` | Existing `test_vectorised_engine_matches_row_engine` now passes on pandas 3.0.6 and 2.2.3 |
| 6 | Hotspots page in the "no complete week" state said "History from Oct 19" (a future date) | Wording now distinguishes "History from" and "The first complete week starts" | Browser check (§7) |
| 7 | A RESOLVED or CLOSED incident showed an SLA countdown ("3d remaining") although its SLA clock had stopped | Shows "clock stopped" instead | Browser check (§7) |

**Also hardened:** the backend now logs a warning at startup while the published default `SECRET_KEY` is in use (`test_startup.py`).

## 2. Features verified

**COMPLETE AND VERIFIED**

| Area | How |
|---|---|
| Backend starts; `/health`, `/health/db` | Live server from the clean checkout (§8) |
| PostgreSQL + PostGIS, initialisation, required tables, append-only triggers, FKs, `ml_predictions` | `init_db.py` on a new database: 7 tables, 4 triggers; backend tests |
| Registration, login, bcrypt hashing, JWT issue and validation, inactive accounts, role changes effective on the next request, last-admin protection | Backend suite (`test_auth_roles.py`, 76+ tests); browser registration and login |
| RBAC: own / assigned / all incidents; 401 / 403 / 404; admin-only endpoints; direct API calls | Backend suite plus ML API tests; browser checks that a citizen is turned away from `/admin` and `/hotspots` |
| Incident creation, listing, detail, assignment, status transitions, history, priority, SLA override | Backend suite; browser: citizen report with geolocation → admin assign and SLA override → agent In Progress → Resolved; history visible to the citizen |
| Images: upload validation (type, 5 MB, magic bytes), authenticated serving, proof of work | Backend suite (`test_resolution_images.py`, `test_incidents.py`) |
| ML bundle: present in a clean checkout; manifest and 11 file hashes match; pinned manifest hash; runtime versions; feature-order checks; missing / tampered / version-mismatch behaviour | Clean-checkout hash check; `test_ml_unit.py` |
| Resolution predictions: features (0 mismatches against the pipeline), EXPERIMENTAL with reasons, local category and global fallbacks, training-data fallback, UNAVAILABLE, range warning | Unit and API tests; browser (citizen, agent and admin panels; FIRE_HAZARD shows "Baseline, not ML") |
| SLA predictions: current SLA used, overrides 12 h / 200 h / cleared, bands only for model output, never changes priority, SLA or assignment | Unit and API tests; browser: override to 12 h updates the panel, 200 h gives a non-ML state |
| Hotspots (EWMA α 0.3, not ML): weekly UTC windows, partial and current weeks excluded, insufficient data, unknown area, role checks, UI | Unit and API tests; browser in the UNAVAILABLE, NO_DATA and AVAILABLE (5 cells) states |
| ML status endpoint (admin only) | API tests; live 401 without a token |
| Prediction logging on create / SLA / priority change, deduplication, failure isolation, backfill, shadow evaluation with right-censoring | API and unit tests; backfill run (278 rows, then 0 on re-run) |
| Frontend: routes render, API routes correct, loading / error / empty states, labels and warnings, responsive at 1366×900 and 390×844 | Playwright run (§7) |
| Fresh-clone reproducibility | §8 |

**IMPLEMENTED BUT NOT FULLY VERIFIED**
- **Google sign-in.** Configured locally. Verified: `/auth/config` reports it enabled; `/auth/google/login` redirects 307 to accounts.google.com with a state token, the right redirect URI and a state cookie; a forged state is rejected. A real consent screen was **not** completed, because that needs a human Google login.
- **Browser coverage of some admin and agent controls.** Image upload in the browser (report photo, proof of work), the admin role-change and deactivation controls, and the agent availability toggle were not clicked in the browser. Their APIs are covered by the backend suite, and the pages rendered without errors.
- **Frontend dev mode.** `npm run dev` was not run this session. `npm run build && npm start` was, from the clean checkout.
- **Viewports.** Only 1366×900 and 390×844 were checked.

**INCOMPLETE:** none of the documented software-only features. See the known limitations in §10.

**BLOCKED BY EXTERNAL REQUIREMENTS**
- **ML validity for UrbanEye or India.** This needs real UrbanEye outcomes. The shadow log and evaluation that will measure it are in place.
- **Completing Google OAuth end to end.** This needs an interactive Google account.

## 3. Files created or modified (this release pass)

**Created**
- `backend/app/ml/evaluation.py`
- `backend/scripts/init_db.py`
- `backend/scripts/ml_log_predictions.py`
- `backend/tests/test_startup.py`
- `docs/SOFTWARE_RELEASE_READINESS.md`

**Modified**
- `backend/app/ml/service.py`: `record_predictions`, `backfill_predictions`, SAVEPOINT isolation.
- `backend/app/api/incidents.py`: three best-effort logging calls. Your earlier validate-before-write edit is preserved.
- `backend/app/api/predictions.py`: `GET /ml/shadow-evaluation`.
- `backend/app/core/config.py`: `ML_LOG_ON_WRITE`.
- `backend/app/main.py`: `SECRET_KEY` warning.
- `backend/tests/test_ml_api.py`, `backend/tests/test_ml_unit.py`: new tests.
- `frontend/src/app/hotspots/page.tsx`, `frontend/src/components/IncidentDetailParts.tsx`: defects 6 and 7.
- `UrbanEye_ML_Data_Pipeline/requirements.txt`, `UrbanEye_ML_Data_Pipeline/scripts/preprocess/build_priority_features.py`: defects 4 and 5.
- `.gitignore`, `.env.example`: defect 3, plus documentation of the ML variables.
- `README.md`: restructured for the release. Your uncommitted `.env` paragraph and the Google, API, example and domain sections are kept verbatim.
- `docs/architecture.md`: roadmap statuses.
- `docs/PHASE_2C_REPORT.md`: limitations that are now resolved are marked.

**From Phase 2C (still uncommitted):** `backend/app/ml/*`, `backend/app/api/predictions.py`, `backend/app/schemas/prediction.py`, `backend/app/models/ml_prediction.py`, `backend/migrations/add_ml_predictions.sql`, `backend/ml_bundle/`, the ML tests, the frontend prediction panel and Hotspots page, and the pipeline's `scripts/phase2b`, `scripts/phase2c`, `reports/phase2a`, `reports/phase2b` and the Phase 2B / 2C tests.

**Your pre-existing uncommitted edits, untouched by this pass:** `backend/app/services/image_service.py`, `backend/app/services/incident_service.py`, `backend/scripts/create_demo_data.py`, `backend/tests/test_incidents.py`, `frontend/src/components/ImageUploadField.tsx`, and the untracked `presentation/`.

## 4. Database changes

| Change | Kind |
|---|---|
| `ml_predictions` table (Phase 2C) | Additive; created by `create_all` and by `migrations/add_ml_predictions.sql` |
| No other schema change in this pass | — |

- **Your development database** (`urbaneye-postgres`, volume `postgres_data`): row counts were identical before and after every test run (incidents 6, incident_history 47, users 15, incident_images 4, user_audit_log 20, ml_predictions 0).
- **Browser and clean-checkout testing** used a throwaway container (`urbaneye-verify-pg`, port 55432) with its data on **tmpfs**. It created no Docker volume, and the container was removed afterwards.
- No `down -v`, no volume removal, no deletes.

## 5. ML model bundle

| Item | Value |
|---|---|
| Location | `backend/ml_bundle/urbaneye-ml-2026.10.13/` (13 MB, 11 files plus the manifest); not git-ignored |
| Pinned manifest hash | `ML_BUNDLE_MANIFEST_SHA256 = b89f0e32…7c7e` (`config.py`) |
| Models | `resolution_v2.0.0`, `sla_v2.0.0` (copied byte-for-byte from Phase 2B; hashes recorded in the manifest), `hotspot_v2.0.0` (EWMA settings, not ML) |
| Runtime | Python 3.11, scikit-learn 1.6.1, numpy 2.2.6, scipy 1.17.1, joblib 1.5.1. Exact pins; `pip check` is clean in a fresh venv. |
| Training data | US 311 (Chicago, San Francisco, New York City), 2010–2020-08-28. **Not validated for Indian cities.** |
| Phase 2B originals | Unchanged: `verify_v1_reproducible.py` reproduces all v1 metrics; `test_phase2c_export.py` matches the bundle against the `*_v2` artifacts |

## 6. Test commands and actual results

| Suite | Command | Result |
|---|---|---|
| Backend, local (development database, rollback-safe) | `cd backend && .venv/bin/python -m pytest -q -rs` | **244 passed, 1 skipped.** The skip is `test_auth_roles.py:518`, "Google OAuth is configured in this environment". |
| Backend, clean checkout (fresh venv, throwaway database configured via `cp .env.example .env`) | same | **245 passed, 0 skipped** |
| ML pipeline, local | `cd UrbanEye_ML_Data_Pipeline && python3 -m pytest tests -q` | **114 passed** |
| ML pipeline, clean checkout (fresh venv: pandas 3.0.6, numpy 2.4.6) | `pip install -r requirements.txt pytest && python -m pytest tests -q` | **82 passed, 32 skipped.** The skips need the git-ignored datasets and trained models. Before fixes 4 and 5 this run gave 3 failed. |
| v1 reproduction | `python3 scripts/phase2b/verify_v1_reproducible.py` | All 3 reproduced |
| Frontend, local and clean | `npx tsc --noEmit`; `npm run lint`; `npm run build` | exit 0; no ESLint warnings; compiled, 12 routes |

New tests in this pass:
- `test_ml_unit.py`: 33 tests (2 new: outcome rules, censoring summary).
- `test_ml_api.py`: 18 tests (7 new: logging on create, failure isolation, SQL-error containment, SLA-override history, switch-off, backfill idempotency, shadow evaluation and role checks).
- `test_startup.py`: 1 test.

## 7. Browser verification (Playwright + Google Chrome, headless)

Run against the clean checkout: frontend `next start` on :3001, backend on :8100, throwaway database. **20/20 checks passed**, run twice: before and after the UI fixes.

1. Unauthenticated `/dashboard` redirects to `/login`.
2. A new citizen registers.
3. The citizen reports an incident using browser geolocation.
4. The citizen sees the detail view, workflow, history and the prediction panel. It shows EXPERIMENTAL with reasons, the "not a reliable interval (about 60%)" text, and the not-validated warning. No admin inputs are shown.
5. The citizen is turned away from `/admin`.
6. The citizen is turned away from `/hotspots`.
7. Admin logs in and sees the incident list.
8. Admin detail shows the predictions with "Model inputs (admin)".
9. An SLA override to 12 h updates the panel to "12 h target".
10. An SLA override to 200 h gives a non-ML state (`SLA_OUT_OF_RANGE`; a labelled baseline or UNAVAILABLE) with no risk band.
11. The SLA is restored to 72 h.
12. Admin assigns the incident to an agent.
13. User Management renders and lists the new account.
14. Hotspots shows an honest status.
15. The agent sees the incident in the queue with predictions.
16. The agent moves it to In Progress, then Resolved.
17. The agent can open Hotspots.
18. Mobile 390×844: the citizen detail view has no horizontal overflow.
19. Mobile: Hotspots has no horizontal overflow.
20. With the predictions API forced to 500, the panel shows "Predictions are unavailable right now" and the rest of the detail view still works.

Extra checks on seeded history in the throwaway database:
- A FIRE_HAZARD report shows "Baseline, not ML" (local category median, 38 incidents; SLA as a historical rate with no band).
- Hotspots is AVAILABLE with 5 cells for Bengaluru potholes, and Chennai shows `NO_DATA`.
- A RESOLVED incident shows "clock stopped" with no countdown.

**Console errors:** only the deliberately injected 500 in check 20. No other console error or failed request.

## 8. Clean-checkout verification

Method: every file that would be committed (`git ls-files -co --exclude-standard`, minus your unrelated `presentation/`) was copied to an isolated directory. Nothing from the working tree's ignored files was carried over: no `.env`, venvs, `node_modules`, data or models.

1. Bundle present; 11 file hashes match the manifest; the manifest hash matches the pin. ✅
2. `python3.11 -m venv` + `pip install -r backend/requirements.txt`; `pip check` is clean. ✅
3. `npm ci` (385 packages, from the lockfile); type check, lint and build pass. ✅
4. **Database from scratch:** the old README order (migrations first) failed on a new database (`relation "incidents" does not exist`; each file rolled back cleanly). `scripts/init_db.py` succeeded on a brand-new database **without PostGIS enabled** and again on an initialised one. ✅
5. `create_demo_data.py` created the 6 demo accounts. ✅
6. `.env` copied from `.env.example` is picked up from the repository root. ✅
7. Backend starts; `/health` and `/health/db` are ok; the bundle loads (`error: None`). ✅
8. Frontend serves `/login` and `/hotspots` (200). ✅
9. Authentication, role checks, predictions and Hotspots were verified in the browser (§7) and by 245 backend tests. ✅
10. `ml_log_predictions.py`: backfilled 142 incidents (278 rows); re-run added 0. ✅
11. Pipeline tests in a fresh venv pass after fixes 4 and 5. ✅

Not verifiable in that run: anything needing the git-ignored 311 data (pipeline re-runs, retraining).

## 9. Security findings

| Area | Finding |
|---|---|
| Secrets | No real secret is tracked or in history. Every historical `.env.example` contained placeholders only; no real `.env` was ever committed; `.env`, `backend/.env`, `frontend/.env` and `frontend/.env.local` are ignored. One tracked file, `UrbanEye_ML_Data_Pipeline/data/manifests/boston311.FAILED.json`, contains an **expired pre-signed S3 URL** generated by Boston's open-data portal. It includes the portal's AWS *access key ID* but no secret, and it was already in history. Low risk; left as is. |
| Default `SECRET_KEY` | Insecure default, used if `.env` does not override it. Now warned about at startup. **Must** be set for any shared deployment. |
| Authorization | Roles are read from the database on every request; inactive accounts get 401. Prediction endpoints reuse the incident access rule. Model inputs are admin-only; Hotspots is admin or agent; ML status and shadow evaluation are admin-only. Shadow evaluation returns aggregates only. |
| Input validation | Pydantic schemas; client-supplied priority and severity stripped; SLA 1–8760 h; image type, size and magic bytes. |
| Database | ORM and parameterised inserts only; ML work isolated in SAVEPOINTs. |
| Model artifacts | Pickles are loaded only after the pinned manifest hash and every file hash verify; no request can choose the path; status responses carry error codes, not paths. |
| CORS | A fixed list of localhost origins in `app/main.py`. A deployment must edit it. |
| Remaining limitations | JWT kept in `localStorage` (exposed to any XSS); no login rate-limiting or lockout; no HTTPS or production server configuration (development setup only). |

## 10. Known limitations

- **The ML is experimental decision support.** It was trained on US data and is not validated for Indian cities; every prediction says so.
  - The resolution range covers about 60% of outcomes, not 80%.
  - FIRE_HAZARD and OTHER have no model.
  - The SLA threshold is provisional.
  - With today's data most predictions are EXPERIMENTAL.
- **Shadow evaluation** reports error only for closed incidents, and needs at least 30 outcomes per group before calling a metric usable.
- **Service areas** are bounding boxes (Bengaluru, Chennai).
- **No Alembic.** Schema setup is `init_db.py` plus idempotent SQL.
- **Your development database** prints `WARNING: database "urbaneye" has no actual collation version` on each `psql` call. This is environmental: an amd64 image on an arm64 host. It is harmless for the app and was not changed.
- **Hardware, Edge AI, camera and IoT ingestion, notifications, routing** are future work. They are not defects of this release.

## 11. Remaining blockers

**None for publishing the software-only release.** Two external items remain: completing a real Google consent screen, and local validation of the ML, which needs real outcomes.

## 12. Exact commands to start the application

```bash
cd UrbanEye
cp .env.example .env                       # first time; set SECRET_KEY for shared use
docker compose up -d                       # PostgreSQL + PostGIS

cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/init_db.py                  # idempotent
python scripts/create_demo_data.py         # demo accounts, idempotent
uvicorn app.main:app --reload              # http://localhost:8000

# second terminal
cd frontend
npm ci
cp .env.local.example .env.local
npm run dev                                # http://localhost:3000
```

Demo sign-ins: `admin@urbaneye.local` / `Admin@1234`, `agent@urbaneye.local` / `Agent@1234`, `user@urbaneye.local` / `User@1234`.

## 13. Release-readiness assessment

**Ready to publish on GitHub as a software-only release**, with the limitations in §10 stated in the README.

- Every core workflow was exercised in a real browser and by the automated suites.
- A fresh clone installs, initialises an empty database, starts and passes its tests.
- The ML layer is integrity-checked and honestly labelled.

**Not claimed:**
- ML accuracy for UrbanEye or Indian conditions.
- A completed Google consent flow.
- Production deployment hardening (HTTPS, rate-limiting, secret management beyond `.env`).
