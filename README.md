# UrbanEye+

## Smart City Incident Reporting & Intelligent Response System

UrbanEye+ is an end-to-end civic incident response platform. Citizens report incidents, the
system classifies and prioritises them, assigns them to field agents, and tracks them through
a governed workflow to resolution — with full GIS/spatial awareness and SLA monitoring.

---

## Current State — Phase 2 (Auth, Roles, Workflow, SLA, Web App)

Phase 1 delivered the backend foundation. **Phase 2 adds authentication, three role-based
dashboards, a governed incident lifecycle, SLA tracking, and a Next.js web application.**

### Implemented

**Backend**
- FastAPI REST API with JWT authentication (bcrypt password hashing)
- Role-based access control — `USER` (citizen), `AGENT` (field agent), `ADMIN`
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
- 61 passing tests

**Frontend** (Next.js 14 App Router)
- Citizen dashboard — report incidents (no severity picker: the system decides), track progress
- Agent dashboard — work queue, ASSIGNED → IN_PROGRESS → RESOLVED, own availability toggle
- Admin dashboard — all incidents, full detail, status/assignment/SLA/priority control,
  agent availability management
- Visual workflow timeline, SLA badges, authenticated image galleries with lightbox
- Responsive dark theme, WCAG-AA text contrast, keyboard-accessible controls

### Not yet implemented

- 🤖 AI classification (NLP, Computer Vision / YOLO) — priority is currently rule-based
- 🧠 ML-driven priority and SLA prediction — the `/priority` and `/sla` endpoints are the
  designed integration seams
- 🗺️ Department routing · 📲 Notifications · 🔁 Duplicate detection
- ⚡ Redis / Celery async tasks · 📷 Camera / IoT ingestion
- 📊 Analytics dashboard · 🚁 Kubernetes / production deployment

---

## Technology Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI 0.115 |
| Language | Python 3.11 |
| ORM | SQLAlchemy 2.0 |
| Spatial | PostGIS 3.4 (via GeoAlchemy2) |
| Database | PostgreSQL 16 |
| DB Driver | psycopg2-binary |
| Auth | python-jose (JWT) + passlib/bcrypt |
| Config | pydantic-settings |
| Server | Uvicorn |
| Testing | pytest + httpx |
| Frontend | Next.js 14 (App Router), React 18, TypeScript |
| Frontend styling | Vanilla CSS design system (`globals.css`) |
| Maps | Leaflet + OpenStreetMap |
| Container | Docker / Docker Compose |

---

## Project Structure

```
UrbanEye/
├── docker-compose.yml              # PostgreSQL/PostGIS container
├── .env.example                    # Backend environment template
├── .env                            # Local environment (git-ignored)
├── docs/
│   └── architecture.md             # Architecture documentation
├── backend/
│   ├── requirements.txt
│   ├── pyproject.toml              # pytest configuration
│   ├── migrations/                 # Plain-SQL migrations (no Alembic)
│   │   ├── add_user_columns.sql
│   │   └── add_priority_sla_columns.sql
│   ├── scripts/
│   │   └── create_demo_data.py     # Seeds the ADMIN / AGENT / USER accounts
│   ├── app/
│   │   ├── main.py                 # FastAPI app, CORS, health endpoints
│   │   ├── core/
│   │   │   ├── config.py           # Settings (reads from .env)
│   │   │   ├── security.py         # bcrypt hashing + JWT encode/decode
│   │   │   └── deps.py             # Auth dependencies, role guards
│   │   ├── db/                     # Declarative base, engine, session
│   │   ├── models/                 # incident.py, user.py, image.py (+ workflow rules)
│   │   ├── schemas/                # Pydantic request/response shapes
│   │   ├── api/                    # auth.py, incidents.py, agents.py, images.py
│   │   └── services/               # Business logic / DB operations
│   └── tests/
│       └── test_incidents.py       # 61 tests
├── frontend/
│   ├── .env.local.example          # NEXT_PUBLIC_API_URL template
│   └── src/
│       ├── app/                    # login, dashboard (citizen), agent, admin
│       ├── components/             # Timeline, images, badges, detail panels
│       └── lib/api.ts              # Typed API client + JWT handling
├── ml/                             # Placeholder — not yet implemented
└── ai/                             # Placeholder — not yet implemented
```

---

## Setup

### 1. Start PostgreSQL/PostGIS

```bash
docker compose up -d
docker ps --filter "name=urbaneye-postgres"
```

Connection details come from `.env` (defaults: `localhost:5432`, database `urbaneye`,
user `urbaneye`, password `urbaneye_dev`).

### 2. Install backend dependencies

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Apply database migrations

`Base.metadata.create_all()` creates *new* tables on startup but never alters existing ones.
On an existing database you must run the SQL migrations once, in order:

```bash
docker exec -i urbaneye-postgres psql -U urbaneye -d urbaneye \
  < backend/migrations/add_user_columns.sql

docker exec -i urbaneye-postgres psql -U urbaneye -d urbaneye \
  < backend/migrations/add_priority_sla_columns.sql
```

Both are idempotent and safe to re-run. `add_priority_sla_columns.sql` backfills
`priority_level` from the legacy `severity` column and derives the SLA window from it, so
incidents created before Phase 2 continue to load correctly. No existing row is modified
destructively.

### 4. Seed the demo accounts

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
read from `DEMO_*` variables (see `.env.example`); note that `pydantic-settings` resolves
`.env` relative to the **current working directory**, so run the script from a directory
containing your `.env` or export the variables to override the defaults.

### 5. Start the backend

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```

Runs on **http://localhost:8000**.

### 6. Start the frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local    # set NEXT_PUBLIC_API_URL if not localhost:8000
npm run dev
```

Runs on **http://localhost:3000**. The backend's CORS policy (`app/main.py`) allows
`localhost:3000`, `127.0.0.1:3000` and `localhost:3001` — add your origin there if you
serve the frontend elsewhere.

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
cd backend
source .venv/bin/activate
pytest -v
```

**61 tests.** They run against the real development PostgreSQL/PostGIS database; each test
that writes uses a SAVEPOINT transaction rolled back afterwards, so no test data persists.
The Docker container must be running first.

Frontend checks:

```bash
cd frontend
npx tsc --noEmit     # type check
npm run lint
npm run build
```

---

## API Endpoints

### Authentication

| Method | Path | Access | Description |
|---|---|---|---|
| POST | `/api/v1/auth/register` | Public | Register a citizen account (**USER role only**) |
| POST | `/api/v1/auth/login` | Public | Log in, receive a JWT |
| GET | `/api/v1/auth/me` | Authenticated | Current user profile |

> Admin and agent accounts cannot be created through the API. Use
> `scripts/create_demo_data.py` — the `role` field is ignored on registration and always
> forced to `USER`.

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
| GET | `/api/v1/incidents/{id}/images` | Reporter / assigned agent / admin | List image metadata |
| GET | `/api/v1/incidents/{id}/images/{image_id}` | Reporter / assigned agent / admin | Stream image bytes |

> Images are **not** public. There is no query-string token — requests must carry
> `Authorization: Bearer <token>`. The frontend fetches them with the header and renders
> from blob object URLs.

### Agents

| Method | Path | Access | Description |
|---|---|---|---|
| GET | `/api/v1/agents/me` | AGENT | Own profile + persisted availability |
| POST | `/api/v1/agents/availability` | AGENT | Set own availability |
| GET | `/api/v1/agents/` | ADMIN | List agents with availability and workload |
| PATCH | `/api/v1/agents/{id}/availability` | ADMIN | Override an agent's availability |

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
  The two existing files are independent of each other, so either order works — but they are
  not sequence-numbered, so a future migration with a real dependency should adopt a
  `NNN_description.sql` prefix to make the order explicit.
- **The `models/` gitignore trap.** The root `.gitignore` ML section anchors its rule as
  `/models/` specifically because an unanchored `models/` also matches
  `backend/app/models/` — the ORM package — and silently excludes it from version control.
- **Legacy data.** Incidents created before Phase 2 are backfilled by the migration, and the
  service layer additionally derives `priority_level` from `severity` at read time as a
  safety net if the migration has not been run.
