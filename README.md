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
- 161 passing tests

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
│   ├── architecture.md             # Architecture documentation
│   └── authentication.md           # Auth, Google setup, roles, security notes
├── backend/
│   ├── requirements.txt
│   ├── pyproject.toml              # pytest configuration
│   ├── migrations/                 # Plain-SQL migrations (no Alembic)
│   │   ├── add_user_columns.sql
│   │   ├── add_priority_sla_columns.sql
│   │   ├── add_incident_history.sql
│   │   └── add_auth_and_user_management.sql
│   ├── scripts/
│   │   └── create_demo_data.py     # Seeds the ADMIN / AGENT / USER accounts
│   ├── app/
│   │   ├── main.py                 # FastAPI app, CORS, health endpoints
│   │   ├── core/
│   │   │   ├── config.py           # Settings (reads from .env)
│   │   │   ├── security.py         # bcrypt hashing + JWT encode/decode
│   │   │   └── deps.py             # Auth dependencies, role guards
│   │   ├── db/                     # Declarative base, engine, session
│   │   ├── models/                 # incident.py, user.py, image.py, history.py,
│   │   │                           #   user_audit.py (+ workflow rules)
│   │   ├── schemas/                # Pydantic request/response shapes
│   │   ├── api/                    # auth.py, incidents.py, agents.py, images.py,
│   │   │                           #   admin_users.py
│   │   └── services/               # Business logic / DB operations
│   │                               #   (incl. google_oauth.py, user_audit_service.py)
│   └── tests/
│       ├── test_incidents.py       # 72 tests — API, workflow, SLA, authorization
│       ├── test_history.py         # 15 tests — incident audit trail
│       └── test_auth_roles.py      # 74 tests — auth, Google, roles, lockout
├── frontend/
│   ├── .env.local.example          # NEXT_PUBLIC_API_URL template
│   └── src/
│       ├── app/                    # login, auth/callback, dashboard (citizen),
│       │                           #   agent, admin, admin/users
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
cd /path/to/UrbanEye

docker exec -i urbaneye-postgres psql -U urbaneye -d urbaneye \
  < backend/migrations/add_user_columns.sql

docker exec -i urbaneye-postgres psql -U urbaneye -d urbaneye \
  < backend/migrations/add_priority_sla_columns.sql

docker exec -i urbaneye-postgres psql -U urbaneye -d urbaneye \
  < backend/migrations/add_incident_history.sql

docker exec -i urbaneye-postgres psql -U urbaneye -d urbaneye \
  < backend/migrations/add_auth_and_user_management.sql
```

All four are idempotent and safe to re-run; none deletes or overwrites an existing row.

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

Verify afterwards:

```bash
docker exec -i urbaneye-postgres psql -U urbaneye -d urbaneye \
  -c "SELECT role, count(*) FROM users GROUP BY role;"
```

### 4. Seed the bootstrap admin and demo accounts

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

**This script is now only a bootstrap.** Ordinary users register in the app and appear
immediately in Admin → User Management, where an admin promotes them. The script still
exists because the *first* admin cannot promote themselves into existence; run it once, sign
in as that admin, and manage everyone else from the UI. Change `DEMO_ADMIN_PASSWORD` before
using it anywhere but a development machine.

### 4b. (Optional) Configure Google Sign-In

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

**161 tests** — 72 API/workflow/SLA, 15 incident-history, 74 authentication and role
management. They run against the real development PostgreSQL/PostGIS database; each test
that writes uses a SAVEPOINT transaction rolled back afterwards, so no test data persists.
The Docker container must be running first, and the migrations above must have been applied.

Google's own servers are not contacted by the suite. What *is* tested is everything on this
side of that boundary: the OAuth `state`/CSRF checks, ID-token claim validation, the refusal
of an unverified email, and account creation, reuse and linking. Completing a real consent
screen requires an interactive login — see the manual step in
[docs/authentication.md](docs/authentication.md).

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
