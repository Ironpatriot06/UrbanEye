# UrbanEye+

## Smart City Incident Reporting & Intelligent Response System

UrbanEye+ is a civic incident response platform. Citizens report incidents, the system
prioritises them and assigns them to field agents, tracks them through a governed workflow to
resolution with full GIS/spatial awareness and SLA monitoring, and offers **experimental,
clearly-labelled ML decision support**: an estimated time to resolve, the risk of missing the
SLA, and next-week hotspots.

> **Software-only release.** Everything in this repository runs on a laptop with Docker,
> Python and Node — no cameras, sensors, ESP32 or Raspberry Pi. Incidents come from citizens
> through the web app. Camera / IoT ingestion and Edge AI are **future work** (see
> [Hardware and Edge AI](#hardware-and-edge-ai--future-work)).

---

## Current state

| Area | State |
|---|---|
| Backend — FastAPI, PostGIS, incident workflow, auth, roles, SLA, images, audit trails | **Implemented and tested** |
| Frontend — Next.js citizen / agent / admin apps, user management, hotspots | **Implemented; browser-verified** |
| ML pipeline — 1.9 M public 311 records, audited task datasets, training, Phase 2B evaluation | **Implemented** (data is not in Git; see below) |
| ML serving — resolution-time and SLA-breach predictions, EWMA hotspots, prediction log, shadow evaluation | **Implemented; experimental** — not validated on UrbanEye data |
| Camera / IoT ingestion, Edge AI, notifications, department routing | **Not implemented — future work** |

Release verification (tests, browser runs, clean-checkout install):
[`docs/SOFTWARE_RELEASE_READINESS.md`](docs/SOFTWARE_RELEASE_READINESS.md).

### Implemented

**Backend**
- FastAPI REST API with JWT authentication (bcrypt password hashing)
- Role-based access control — `USER` (citizen), `AGENT` (field agent), `ADMIN`
- **Self-service sign-up** — email/password registration and **Google Sign-In** (OAuth 2.0
  authorization-code flow, ID token verified server-side). New accounts are always `USER`;
  a client cannot choose its own role. Ordinary users no longer need a seeding script.
- **Account linking** — signing in with Google using an address that already has a password
  account links the two rather than creating a duplicate, and keeps the existing role
- **Admin user management** — list every account, promote `USER ⇄ AGENT ⇄ ADMIN`, deactivate
  and reactivate accounts. Guarded so an admin cannot demote themselves and the last active
  admin cannot be removed — the system always retains a usable administrator.
- **Authorization from the database, not the token** — a promotion or demotion takes effect
  on the user's next request; a deactivated account's outstanding tokens stop working at once
- **Account audit trail** (`user_audit_log`) — registrations, role changes, activations and
  Google links, append-only, kept separate from incident history
- PostgreSQL / PostGIS spatial storage (GEOGRAPHY POINT, SRID 4326)
- **System-assigned priority** — P1/P2/P3/P4, derived server-side from the incident
  category. Citizens cannot set it; client-supplied values are stripped.
- **SLA tracking** — `sla_hours`, `sla_deadline`, `sla_status` (ON_TRACK / AT_RISK / BREACHED).
  Visible to all roles, editable only by an admin. The clock freezes at RESOLVED/CLOSED.
- **Enforced status workflow** — REPORTED → TRIAGED → ASSIGNED → IN_PROGRESS → RESOLVED →
  CLOSED. Invalid jumps are rejected server-side (HTTP 422); agents are confined to
  ASSIGNED → IN_PROGRESS → RESOLVED (HTTP 403 otherwise).
- **Agent assignment queue** — new incidents auto-route to the longest-idle available agent
- **Persistent agent availability** — never reset by login; changed only by the agent or an admin
- **Admin availability override** — assigning an unavailable agent requires an explicit
  `override_availability` flag, validated on the server
- **Authenticated image upload/serving** — images stored as BYTEA, streamed only to the
  reporter, the assigned agent, or an admin
- **Proof of work** — after marking an incident Resolved, the assigned agent attaches a
  photo of the finished job. Stored and served identically to the citizen's report photo,
  tagged `RESOLUTION`, and shown to the reporter, the agent and admins alike.

**Frontend** (Next.js 14 App Router)
- Citizen dashboard — report incidents (no severity picker: the system decides), track progress
- Agent dashboard — work queue, ASSIGNED → IN_PROGRESS → RESOLVED, own availability toggle
- Admin dashboard — all incidents, full detail, status/assignment/SLA/priority control,
  agent availability management
- **Admin → User Management** — every account in one table (name, email, role, sign-in
  method, created, last login, status), role changes with confirmation before granting
  admin, and the account audit trail. Linked from the admin navbar; hidden from citizens
  and agents (and refused by the backend regardless)
- Login page with email/password, *Continue with Google*, and create-account
- Visual workflow timeline, SLA badges, authenticated image galleries with lightbox
- Responsive dark theme, WCAG-AA text contrast, keyboard-accessible controls

**Machine learning in the app** (`backend/app/ml/`, details in
[`docs/PHASE_2C_REPORT.md`](docs/PHASE_2C_REPORT.md))

- **Resolution-time estimate** and **SLA-breach risk** for every incident, shown to the roles
  that can open it. Computed as of the report time from earlier incidents only, so the
  incident's own outcome can never leak in.
- Every result states **how** it was produced and **how far to trust it**:

  | Status | Meaning |
  |---|---|
  | `AVAILABLE` | ML model output; every input inside the range seen in training |
  | `EXPERIMENTAL` | ML model output, but local history is missing or out of range (today's normal state) |
  | `FALLBACK` | Not ML: a named baseline (local category median, local recent breach rate, or the US training-data figure), with the reason the model was not used |
  | `UNAVAILABLE` | Neither a valid model result nor an applicable baseline |

- The resolution **p10–p90 range is not a calibrated 80% interval** — it contained about 60% of
  outcomes in offline tests — and the UI says so.
- SLA predictions use the incident's **current** SLA, including admin overrides; windows outside
  the trained 4–168 h are never extrapolated; a cleared SLA gives "no SLA".
- **Hotspots** — next-week report counts per 1 km cell by an exponentially weighted moving
  average (α = 0.3). This is a statistic, **not an ML model**; the cross-city hotspot ML model
  generalised badly and is not shipped.
- Predictions never change an incident's priority, SLA, assignment or status.
- **Prediction log** (`ml_predictions`) written automatically on incident creation and on SLA /
  priority changes (best effort — a failure can never block the write), plus an idempotent
  backfill command and an admin **shadow evaluation** that joins predictions to observed
  outcomes and treats still-open incidents as right-censored.
- Models ship as a **versioned, SHA-256-pinned bundle** (`backend/ml_bundle/`); the backend
  verifies every file and the scikit-learn version before loading anything, and keeps running
  (with fallbacks) if the bundle is missing or tampered with.

**Machine learning pipeline** (`UrbanEye_ML_Data_Pipeline/` — offline; it never touches the
database or the API)

- Reproducible pipeline over public 311 data — **1,900,000 normalised incidents** from SF
  (200 k), Chicago (1.6 M) and NYC (100 k)
- Category mapping to UrbanEye's six categories (`config/urbaneye_category_mapping.csv`) —
  FIRE_HAZARD and OTHER have **no** training data and get no model
- Leak-free history features, rolling-origin and leave-one-city-out evaluation, gated final
  training — [Phase 2B report](UrbanEye_ML_Data_Pipeline/reports/phase2b/PHASE_2B_REPORT.md)

### Not implemented (future work)

- 📷 Camera / IoT / ESP32 / Raspberry Pi ingestion and Edge AI — see below
- 🤖 AI classification of text or images (priority stays rule-based from the category)
- 🔁 Duplicate detection — blocked on evaluation data
- 🗺️ Department routing · 📲 Notifications · ⚡ async workers · 📈 production monitoring
- 🌏 Indian training data: the models are trained on **US** 311 data and have **not** been
  validated for Indian operating conditions

---

## Model results (offline, US 311 data)

Phase 2B rolling-origin evaluation (six two-month Chicago test windows, UrbanEye categories),
against the strongest simple baseline. These are offline US numbers — **not** UrbanEye
performance.

| Task | Baseline | Model in the app | Result |
|---|---|---|---|
| Resolution time (MAE, hours) | category median 592.9 | `resolution_v2` 581.7 | −1.9%, 5 of 6 windows; degrades without retraining; does not beat a global median in an unseen city |
| SLA breach (PR-AUC / Brier) | recent breach rate 0.849 / 0.161 | `sla_v2` 0.878 / 0.147 | better in 6 of 6 windows; weaker for FLOOD; mixed across cities |
| Hotspot (MAE per cell-week) | rolling 4-week mean 1.506 | EWMA α 0.3 — 1.453 | −3.5%; the ML model was not shipped |

Priority is **not** an ML task (no public dataset records operational priority), and duplicate
detection is not trained. Full reasoning:
[Phase 2B report](UrbanEye_ML_Data_Pipeline/reports/phase2b/PHASE_2B_REPORT.md) ·
[Phase 2A baseline](UrbanEye_ML_Data_Pipeline/reports/phase2a/PHASE_2A_REPORT.md) ·
[`MODEL_SCOPE.md`](UrbanEye_ML_Data_Pipeline/MODEL_SCOPE.md).

---

## Technology Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI 0.115 |
| Language | Python 3.11 (verified 3.11.8) |
| ORM | SQLAlchemy 2.0 |
| Spatial | PostGIS 3.4 (via GeoAlchemy2) |
| Database | PostgreSQL 16 |
| DB Driver | psycopg2-binary |
| Auth | python-jose (JWT) + passlib/bcrypt, Google OpenID Connect (optional) |
| Config | pydantic-settings |
| Server | Uvicorn |
| ML serving | scikit-learn 1.6.1 + numpy 2.2.6 (exact pins), joblib |
| Testing | pytest + httpx |
| Frontend | Next.js 14 (App Router), React 18, TypeScript (Node.js 18.17+; verified on 25.2) |
| Frontend styling | Vanilla CSS design system (`globals.css`) |
| Maps | Leaflet + OpenStreetMap |
| Container | Docker / Docker Compose (PostGIS only) |
| ML pipeline (offline) | pandas, pyarrow, numpy, PyYAML, scikit-learn 1.6.1 |

---

## Project Structure

```
UrbanEye/
├── docker-compose.yml              # PostgreSQL/PostGIS container
├── .env.example                    # Backend environment template (placeholders only)
├── .env                            # Your local environment (git-ignored)
├── docs/
│   ├── architecture.md             # Architecture documentation
│   ├── authentication.md           # Auth, Google setup, roles, security notes
│   ├── PHASE_2C_REPORT.md          # ML integration design and contracts
│   └── SOFTWARE_RELEASE_READINESS.md  # What was verified for this release, and how
├── backend/
│   ├── requirements.txt            # exact pins for the ML runtime
│   ├── pyproject.toml              # pytest configuration
│   ├── migrations/                 # Plain-SQL migrations (no Alembic), idempotent
│   ├── ml_bundle/urbaneye-ml-2026.10.13/   # versioned model bundle (committed, hash-pinned)
│   ├── scripts/
│   │   ├── init_db.py              # PostGIS + tables + every migration, idempotent
│   │   ├── create_demo_data.py     # Seeds the ADMIN / AGENT / USER accounts
│   │   └── ml_log_predictions.py   # Backfill the prediction log + shadow evaluation
│   ├── app/
│   │   ├── main.py                 # FastAPI app, CORS, health endpoints, startup checks
│   │   ├── core/                   # config, security (bcrypt + JWT), auth dependencies
│   │   ├── db/                     # Declarative base, engine, session
│   │   ├── models/                 # incident, user, image, history, user_audit, ml_prediction
│   │   ├── schemas/                # Pydantic request/response shapes
│   │   ├── api/                    # auth, incidents, agents, images, admin_users, predictions
│   │   ├── services/               # Business logic / DB operations
│   │   └── ml/                     # registry, features, predictors, hotspot, evaluation
│   └── tests/                      # API, workflow, auth, images, history, ML
├── frontend/
│   ├── .env.local.example          # NEXT_PUBLIC_API_URL template
│   └── src/
│       ├── app/                    # login, auth/callback, dashboard (citizen), agent,
│       │                           #   admin, admin/users, hotspots
│       ├── components/             # Timeline, images, badges, detail panels, predictions
│       └── lib/api.ts              # Typed API client + JWT handling
└── UrbanEye_ML_Data_Pipeline/      # Offline data pipeline, training and evaluation
    ├── config/  scripts/  tests/  reports/
    ├── data/  models/  experiments/   # git-ignored, regenerable
    └── MODEL_SCOPE.md · ML_READINESS.md · DATA_SCHEMA.md · DATASET_SOURCES.md
```

---

## Setup

Prerequisites: **Docker** (for PostgreSQL/PostGIS), **Python 3.11**, **Node.js 18.17+** with npm,
and Git. No hardware is needed.

### 1. Clone and configure

```bash
git clone <your-fork-url> UrbanEye && cd UrbanEye
cp .env.example .env                 # placeholders; works as-is for local development
```

Set a random `SECRET_KEY` in `.env` before exposing the server to anyone else
(`python3 -c "import secrets; print(secrets.token_hex(32))"`). The backend logs a warning at
startup while the published default is in use.

| Variable | Required | Default / note |
|---|---|---|
| `DATABASE_URL` | no | `postgresql+psycopg2://urbaneye:urbaneye_dev@localhost:5432/urbaneye` |
| `SECRET_KEY` | **yes, for any shared deployment** | `change-me-in-production` (insecure) |
| `JWT_ALGORITHM`, `JWT_EXPIRE_MINUTES` | no | `HS256`, `1440` |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | no | blank → Google sign-in hidden, endpoints return 503 |
| `GOOGLE_REDIRECT_URI`, `FRONTEND_URL`, `OAUTH_STATE_EXPIRE_SECONDS`, `COOKIE_SECURE` | no | local defaults |
| `DEMO_*` | no | accounts created by `create_demo_data.py` |
| `ML_ENABLED`, `ML_BUNDLE_DIR`, `ML_BUNDLE_MANIFEST_SHA256`, `ML_SERVICE_AREAS`, `ML_LOG_ON_WRITE` | no | defaults use the committed bundle (see `.env.example`) |
| `NEXT_PUBLIC_API_URL` (in `frontend/.env.local`) | no | `http://localhost:8000` |

### 2. Start PostgreSQL/PostGIS

```bash
docker compose up -d
docker ps --filter "name=urbaneye-postgres"
```

Connection details come from `.env` (defaults: `localhost:5432`, database `urbaneye`,
user `urbaneye`, password `urbaneye_dev`). The data lives in the `postgres_data` Docker volume;
`docker compose down` keeps it, **`docker compose down -v` deletes it**.

### 3. Install backend dependencies

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`scikit-learn==1.6.1`, `numpy==2.2.6`, `scipy` and `joblib` are pinned **exactly**: the bundled
models are scikit-learn pickles, and the backend refuses to load them under other versions
(predictions then fall back or report `UNAVAILABLE`; everything else keeps working).

### 4. Initialise the database

```bash
python scripts/init_db.py        # from backend/, venv active
```

Idempotent and non-destructive, on a brand-new or an existing database: it enables PostGIS,
creates any missing table, then applies every SQL migration in `backend/migrations/` in order.
The migrations also install what table creation alone cannot — the database triggers that
make `incident_history` and `user_audit_log` append-only — so run it on every new install.

On an existing database the migrations back-fill older rows as they document:

- `add_priority_sla_columns.sql` backfills `priority_level` from the legacy `severity`
  column and derives the SLA window from it, so incidents created before Phase 2 continue
  to load correctly.
- `add_incident_history.sql` creates the incident audit trail and gives pre-existing
  incidents an opening event.
- `add_auth_and_user_management.sql` adds `is_active`, `google_subject_id` and
  `last_login_at` to `users`, makes `password_hash` nullable (a Google-only account has no
  password), and creates the append-only `user_audit_log`. **Existing accounts keep their
  id, email, password and role** — `is_active` defaults to `TRUE`, so every current admin,
  agent and citizen continues to work unchanged.
- `add_resolution_proof_images.sql` adds `kind` (`REPORT` / `RESOLUTION`) to
  `incident_images` so an agent's proof-of-work photo can be told apart from the
  citizen's report photo. Every existing image is back-filled to `REPORT` by the
  column default — no image bytes are read, rewritten or deleted.

- `add_ml_predictions.sql` creates the prediction log (additive only).

To apply a single migration by hand instead:
`docker exec -i urbaneye-postgres psql -U urbaneye -d urbaneye < backend/migrations/<file>.sql`.

### 5. Seed the bootstrap admin and demo accounts

```bash
cd backend
source .venv/bin/activate
python scripts/create_demo_data.py
```

| Role | Email | Password |
|---|---|---|
| ADMIN | `admin@urbaneye.local` | `Admin@1234` |
| AGENT | `agent@urbaneye.local` | `Agent@1234` |
| USER | `user@urbaneye.local` | `User@1234` |

The script is idempotent — existing accounts are skipped, never overwritten. Credentials are
read from `DEMO_*` variables (see `.env.example`). The backend and this script always load
the repository-root `.env` (then `backend/.env`, if present, as an override) by absolute path,
so they pick up the same configuration whichever directory they are started from; exported
environment variables still take precedence over both files.

**This script is now only a bootstrap.** Ordinary users register in the app and appear
immediately in Admin → User Management, where an admin promotes them. The script still
exists because the *first* admin cannot promote themselves into existence; run it once, sign
in as that admin, and manage everyone else from the UI. Change `DEMO_ADMIN_PASSWORD` before
using it anywhere but a development machine.

### 5b. (Optional) Configure Google Sign-In

Google sign-in is optional — leave the variables blank and the app runs on email/password
alone, with the Google button hidden and the endpoints returning `503`.

Create an OAuth client at <https://console.cloud.google.com/apis/credentials>
(Web application), register the redirect URI

```
http://localhost:8000/api/v1/auth/google/callback
```

and set in `.env`:

```bash
GOOGLE_CLIENT_ID=<your-id>.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=<your-secret>
GOOGLE_REDIRECT_URI=http://localhost:8000/api/v1/auth/google/callback
FRONTEND_URL=http://localhost:3000
```

Full walkthrough, including the consent screen and test users:
**[docs/authentication.md](docs/authentication.md)**. Never commit real credentials —
`.env.example` holds placeholders only.

### 6. Start the backend

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```

Runs on **http://localhost:8000** (`/health`, `/health/db`, `/docs`). On startup it verifies the
model bundle; check it as an admin at `GET /api/v1/ml/status`.

### 7. Start the frontend

```bash
cd frontend
npm ci                               # exact versions from package-lock.json
cp .env.local.example .env.local     # set NEXT_PUBLIC_API_URL if not localhost:8000
npm run dev                          # or: npm run build && npm start
```

Runs on **http://localhost:3000**. The backend's CORS policy (`app/main.py`) allows
`localhost:3000`, `127.0.0.1:3000` and `localhost:3001` — add your origin there if you
serve the frontend elsewhere.

### 8. Try it without hardware

1. Sign in as `user@urbaneye.local` / `User@1234` (or create an account) → **Report incident**,
   use *Use my current location* or enter coordinates (Bengaluru or Chennai are inside the
   default ML service areas).
2. Open the report: workflow, history and the **Predictions (experimental)** panel.
3. Sign in as `admin@urbaneye.local` / `Admin@1234` → assign, override the SLA (watch the SLA
   prediction follow it), manage users, open **Hotspots**.
4. Sign in as `agent@urbaneye.local` / `Agent@1234` → move the incident to In Progress and
   Resolved, attach a proof-of-work photo.

With a new database every prediction is `EXPERIMENTAL` (or `FALLBACK` for FIRE_HAZARD / OTHER)
and Hotspots reports `UNAVAILABLE` until a full week of reports exists — that is the honest
state, not an error.

### 9. (Optional) Prediction log and shadow evaluation

```bash
cd backend && source .venv/bin/activate
python scripts/ml_log_predictions.py              # backfill missing predictions, then evaluate
python scripts/ml_log_predictions.py --evaluate-only
```

Or `GET /api/v1/ml/shadow-evaluation` as an admin. This is monitoring, **not validation**: it
reports per-method error only once enough incidents have closed, and never scores a still-open
incident as if it had closed.

### 10. (Optional) Run the ML pipeline and training

The pipeline is independent of the app — it reads nothing from the database and writes
nothing to it. It needs the raw 311 downloads (~9 GB subset; see
[`DATASET_DOWNLOAD_GUIDE.md`](UrbanEye_ML_Data_Pipeline/DATASET_DOWNLOAD_GUIDE.md)).
The app does **not** need it: the trained models it serves are in `backend/ml_bundle/`.

```bash
cd UrbanEye_ML_Data_Pipeline
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
python -m pytest tests -q        # data- and model-dependent tests skip on a fresh clone

# With real data in data/raw/
python scripts/run_pipeline.py                 # datasets + reports
python scripts/train/run_all.py                # Phase 2A models
python scripts/phase2b/build_datasets.py       # Phase 2B datasets, then the experiment scripts
python scripts/phase2b/train_final.py          # gated v2 models (never overwrites v1)
python scripts/phase2c/export_serving_bundle.py   # new bundle dir (refuses to overwrite)
```

`data/`, `models/` and `experiments/` are git-ignored by design. A new bundle is adopted by
pointing `ML_BUNDLE_DIR` and `ML_BUNDLE_MANIFEST_SHA256` at it in one reviewed change.

---

## API Documentation

| Interface | URL |
|---|---|
| Swagger UI (interactive) | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| OpenAPI JSON | http://localhost:8000/openapi.json |

---

## Running Tests

```bash
# Backend — needs the PostgreSQL container running and scripts/init_db.py applied
cd backend && source .venv/bin/activate
python -m pytest -q
```

They run against the configured database; every test that writes uses a SAVEPOINT transaction
rolled back afterwards, so no test data persists. Point `DATABASE_URL` at a separate database
to keep them entirely away from your development data. One test is skipped when Google OAuth
*is* configured (it checks the unconfigured behaviour).

Google's own servers are not contacted by the suite. What *is* tested is everything on this
side of that boundary: the OAuth `state`/CSRF checks, ID-token claim validation, the refusal
of an unverified email, and account creation, reuse and linking. Completing a real consent
screen requires an interactive login — see the manual step in
[docs/authentication.md](docs/authentication.md).

```bash
# Frontend
cd frontend
npx tsc --noEmit && npm run lint && npm run build

# ML pipeline (independent of the app and the database)
cd UrbanEye_ML_Data_Pipeline && source .venv/bin/activate
python -m pytest tests -q
```

Current counts and the browser verification are recorded in
[`docs/SOFTWARE_RELEASE_READINESS.md`](docs/SOFTWARE_RELEASE_READINESS.md).

---

## API Endpoints

### Authentication

| Method | Path | Access | Description |
|---|---|---|---|
| POST | `/api/v1/auth/register` | Public | Register an account (**always USER**), returns a JWT |
| POST | `/api/v1/auth/login` | Public | Log in, receive a JWT |
| GET | `/api/v1/auth/me` | Authenticated | Current user profile |
| GET | `/api/v1/auth/config` | Public | Whether Google sign-in is configured |
| GET | `/api/v1/auth/google/login` | Public | Start Google sign-in (redirects to Google) |
| GET | `/api/v1/auth/google/callback` | Public | Google's redirect back; issues the session |

> **A client can never choose its own role.** A `role` field in the request body — or in the
> query string — is ignored; registration and Google sign-in always produce `USER`. Only an
> authenticated `ADMIN` can change a role, through the endpoints below.

### Admin · User management

| Method | Path | Access | Description |
|---|---|---|---|
| GET | `/api/v1/admin/users` | ADMIN | Every account, newest first (`?role=`, `?q=`) |
| PATCH | `/api/v1/admin/users/{id}/role` | ADMIN | Grant `USER` / `AGENT` / `ADMIN` |
| PATCH | `/api/v1/admin/users/{id}/status` | ADMIN | Activate or deactivate an account |
| GET | `/api/v1/admin/audit` | ADMIN | Account & authorization audit trail |

Unauthenticated → `401`. A `USER` or `AGENT` → `403`, regardless of what the frontend shows.
Refused with `409` when the caller targets their own account, or when the change would leave
no active administrator.

Responses carry **no credential material** — no password hashes, no Google subject ids, no
tokens. `auth_methods` reports *how* an account signs in (`EMAIL`, `GOOGLE`, or both).

### Incidents

| Method | Path | Access | Description |
|---|---|---|---|
| POST | `/api/v1/incidents/` | Authenticated | Create an incident (multipart, optional images) |
| GET | `/api/v1/incidents/` | Authenticated | List — role-filtered (own / assigned / all) |
| GET | `/api/v1/incidents/{id}` | Authenticated | Get one incident — role-gated |
| PATCH | `/api/v1/incidents/{id}/status` | AGENT / ADMIN | Advance status through valid transitions |
| PUT | `/api/v1/incidents/{id}/assign` | ADMIN | Assign/unassign an agent (+ availability override) |
| PATCH | `/api/v1/incidents/{id}/sla` | ADMIN | Override the SLA window |
| PATCH | `/api/v1/incidents/{id}/priority` | ADMIN | Override priority *(AI integration seam)* |

### Images

| Method | Path | Access | Description |
|---|---|---|---|
| GET | `/api/v1/incidents/{id}/images` | Reporter / assigned agent / admin | List image metadata (both kinds) |
| GET | `/api/v1/incidents/{id}/images/{image_id}` | Reporter / assigned agent / admin | Stream image bytes |
| POST | `/api/v1/incidents/{id}/images/resolution` | Assigned agent / admin | Attach proof that the work was completed |

> Images are **not** public. There is no query-string token — requests must carry
> `Authorization: Bearer <token>`. The frontend fetches them with the header and renders
> from blob object URLs.

> **Proof of work.** Once the assigned agent marks an incident **Resolved**, they can
> attach a photo of the finished job. It is stored exactly like a citizen's report photo —
> same table, same JPEG/PNG/WEBP + 5 MB validation, same authenticated streaming endpoint —
> and tagged `RESOLUTION` so each dashboard can label it. It is then visible to the
> reporting citizen, the assigned agent and any admin, under the same access rule as every
> other image on that incident. A citizen gets `403` (they cannot attest that work was
> done); uploading before the incident is Resolved gets `422`.

### Agents

| Method | Path | Access | Description |
|---|---|---|---|
| GET | `/api/v1/agents/me` | AGENT | Own profile + persisted availability |
| POST | `/api/v1/agents/availability` | AGENT | Set own availability |
| GET | `/api/v1/agents/` | ADMIN | List agents with availability and workload |
| PATCH | `/api/v1/agents/{id}/availability` | ADMIN | Override an agent's availability |

### ML predictions

| Method | Path | Access | Description |
|---|---|---|---|
| GET | `/api/v1/incidents/{id}/predictions` | Same as reading the incident | Resolution-time and SLA-breach predictions with status, method and warnings. `?include_inputs=true` adds model inputs for admins only |
| GET | `/api/v1/hotspots?category=&area=` | AGENT / ADMIN | Next-week EWMA counts per 1 km cell (not ML) |
| GET | `/api/v1/ml/status` | ADMIN | Model bundle readiness, versions, service areas |
| GET | `/api/v1/ml/shadow-evaluation` | ADMIN | Logged predictions vs observed outcomes (aggregates only) |

### Health

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/health/db` | Database connectivity probe |

---

## Example: Report an Incident

Incident creation is authenticated and uses `multipart/form-data`: a `data` field holding the
JSON body, plus zero or more `images` files.

```bash
# 1. Log in
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@urbaneye.local","password":"User@1234"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')

# 2. Report an incident with a photo
curl -X POST http://localhost:8000/api/v1/incidents/ \
  -H "Authorization: Bearer $TOKEN" \
  -F 'data={
        "title": "Large pothole near City Bus Stop 14",
        "description": "Approximately 30 cm deep, causing vehicle damage.",
        "category": "POTHOLE",
        "source": "CITIZEN",
        "latitude": 12.9716,
        "longitude": 77.5946
      }' \
  -F 'images=@pothole.jpg;type=image/jpeg'
```

Response (HTTP 201):

```json
{
  "id": 42,
  "title": "Large pothole near City Bus Stop 14",
  "category": "POTHOLE",
  "source": "CITIZEN",
  "status": "ASSIGNED",
  "priority_level": "P3",
  "priority_label": "Medium",
  "severity": "MEDIUM",
  "priority": 3,
  "sla_hours": 72,
  "sla_deadline": "2026-09-14T10:17:08.126313Z",
  "sla_status": "ON_TRACK",
  "latitude": 12.9716,
  "longitude": 77.5946,
  "reported_by": 3,
  "reported_by_name": "Demo Citizen",
  "assigned_agent_id": 2,
  "assigned_agent_name": "Field Agent",
  "image_count": 1,
  "created_at": "2026-09-11T10:17:08.119444Z",
  "updated_at": "2026-09-11T10:17:08.119444Z"
}
```

Note there is **no `severity` or `priority_level` in the request**. The system derives them
from the category (POTHOLE → P3 → MEDIUM → 72h SLA) and strips any client-supplied values.
The status is `ASSIGNED` rather than `REPORTED` because an available agent was found and
auto-assigned.

---

## Domain Reference

### Priority (system-assigned)

| Level | Meaning | Default SLA | Legacy `severity` |
|---|---|---|---|
| P1 | Critical | 4 hours | CRITICAL |
| P2 | High | 24 hours | HIGH |
| P3 | Medium | 72 hours | MEDIUM |
| P4 | Low | 168 hours | LOW |

Default priority by category: `FIRE_HAZARD` → P1, `FLOOD` → P2, `POTHOLE`/`STREETLIGHT`/`OTHER`
→ P3, `GARBAGE` → P4. These mapping tables live in `app/models/incident.py` and are the
designed replacement point for an AI classifier — `_set_priority_fields()` in
`incident_service.py` keeps every derived field (severity, legacy integer, SLA) consistent
whenever priority changes.

### SLA status

`ON_TRACK` · `AT_RISK` (less than 20% of the window remains) · `BREACHED` (deadline passed).
The deadline is anchored to `created_at`, so an admin override never grants a fresh window.

### Status transitions

```
REPORTED ──▶ TRIAGED ──▶ ASSIGNED ──▶ IN_PROGRESS ──▶ RESOLVED ──▶ CLOSED
```

Enforced in `VALID_TRANSITIONS` / `AGENT_TRANSITIONS` (`app/models/incident.py`) and applied
by the service layer. Agents may only perform ASSIGNED → IN_PROGRESS → RESOLVED. `CLOSED` is
terminal. The frontend mirrors these rules for UX only — the backend is the authority.

---

## Notes

- **No Alembic.** Schema changes use plain, idempotent SQL files in `backend/migrations/`.
  They all expect the tables to exist (so `scripts/init_db.py` creates the tables first) and
  are otherwise independent; `init_db.py` lists the order it applies. They are not
  sequence-numbered, so a future migration with a real dependency should adopt a
  `NNN_description.sql` prefix and be appended to `MIGRATIONS` in `init_db.py`.
- **The `models/` gitignore trap.** The root `.gitignore` ML section anchors its rule as
  `/models/` specifically because an unanchored `models/` also matches
  `backend/app/models/` — the ORM package — and silently excludes it from version control.
- **Legacy data.** Incidents created before Phase 2 are backfilled by the migration, and the
  service layer additionally derives `priority_level` from `severity` at read time as a
  safety net if the migration has not been run.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/health/db` returns 503 | Postgres is not running: `docker compose up -d`; check `DATABASE_URL` |
| `relation "incidents" does not exist` when running a migration | Run `python scripts/init_db.py`, which creates the tables before applying migrations |
| Predictions show `ARTIFACT_UNAVAILABLE` | `GET /api/v1/ml/status` names the cause: `BUNDLE_NOT_FOUND`, `MANIFEST_HASH_MISMATCH`, `FILE_HASH_MISMATCH` or `RUNTIME_VERSION_MISMATCH` (reinstall `requirements.txt` exactly) |
| Every prediction is `EXPERIMENTAL` | Expected on a young database: local history is too short, or outside the US training range |
| Hotspots says `NO_COMPLETE_WEEKS` | Weeks run Monday–Sunday (UTC) and the first, partial week is dropped; wait for a full week of reports |
| Browser shows CORS errors | Serve the frontend on an allowed origin (see step 7) or add yours in `app/main.py` |
| Google button missing | `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` not set — optional |
| Startup warns about `SECRET_KEY` | Set a random value in `.env` |

---

## Known limitations

- **ML is decision support, not a guarantee.** The models were trained on 2010–2020 US 311 data
  (Chicago, San Francisco, New York City) and have **not** been validated for Indian cities;
  `local_reliability_established` is `false` everywhere until shadow evaluation says otherwise.
- The resolution range is under-covered (≈60% vs a nominal 80%); FIRE_HAZARD and OTHER have no
  model; the SLA decision threshold is provisional (the cost of a missed breach is undefined).
- ML service areas are bounding boxes (Bengaluru, Chennai by default; `ML_SERVICE_AREAS`).
- The JWT is kept in `localStorage`; there is no login rate-limiting; CORS origins are set in code.
- No Alembic: schema changes are plain idempotent SQL applied by `scripts/init_db.py`.

---

## Hardware and Edge AI — future work

This release is deliberately software-only. The incident model already records a `source`
(`CITIZEN` / `CAMERA` / `IOT`), but only `CITIZEN` reports are produced. Not implemented:
camera (CCTV/YOLO) and IoT sensor ingestion, ESP32 / Raspberry Pi devices, on-device (edge)
inference, and automatic priority changes from any model. Any of these would enter through the
same authenticated API and the same versioned, hash-verified, status-labelled ML path.
