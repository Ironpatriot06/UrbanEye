# UrbanEye+

## AI-Powered Smart City Incident Reporting & Intelligent Response System

UrbanEye+ is an end-to-end intelligent civic incident response platform.  
Citizens, cameras, and IoT sensors report incidents which are ingested, classified by AI, prioritised by ML, routed to the correct department, and tracked through to resolution — all with full GIS/spatial awareness.

---

## Current State — Phase 1 (Foundation + Incident Core)

Phase 1 delivers the **clean backend foundation** on which all future AI/ML features are built:

- FastAPI REST API
- PostgreSQL / PostGIS database
- Incident CRUD (Create, Read, Update status)
- PostGIS spatial storage (GEOGRAPHY POINT, SRID 4326)
- Pydantic input validation
- Health endpoints

---

## Technology Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI 0.115 |
| Language | Python 3.11+ |
| ORM | SQLAlchemy 2.0 |
| Spatial | PostGIS 3.4 (via GeoAlchemy2) |
| Database | PostgreSQL 16 |
| DB Driver | psycopg2-binary |
| Config | pydantic-settings |
| Server | Uvicorn |
| Testing | pytest + httpx |
| Container | Docker / Docker Compose |

---

## Project Structure

```
UrbanEye/
├── docker-compose.yml          # PostgreSQL/PostGIS container
├── .env.example                # Environment variable template
├── .env                        # Local environment (git-ignored)
├── README.md
├── docs/
│   └── architecture.md         # Architecture documentation
├── backend/
│   ├── requirements.txt
│   ├── pyproject.toml          # pytest configuration
│   ├── .venv/                  # Python virtual environment (git-ignored)
│   ├── app/
│   │   ├── main.py             # FastAPI app, startup, health endpoints
│   │   ├── core/
│   │   │   └── config.py       # Settings (reads from .env)
│   │   ├── db/
│   │   │   ├── base.py         # SQLAlchemy declarative base
│   │   │   └── database.py     # Engine, SessionLocal, get_db dependency
│   │   ├── models/
│   │   │   └── incident.py     # Incident ORM model + enums
│   │   ├── schemas/
│   │   │   └── incident.py     # Pydantic request/response schemas
│   │   ├── api/
│   │   │   └── incidents.py    # REST router (CRUD endpoints)
│   │   └── services/
│   │       └── incident_service.py  # Business logic / DB operations
│   └── tests/
│       └── test_incidents.py   # pytest test suite
├── frontend/                   # Not yet implemented
├── ml/                         # Not yet implemented
└── ai/                         # Not yet implemented
```

---

## How to Start PostgreSQL/PostGIS

The database runs in Docker.  From the repository root:

```bash
docker compose up -d
```

Verify it is running:

```bash
docker ps --filter "name=urbaneye-postgres"
```

Connection details (from `.env`):

```
Host:     localhost
Port:     5432
Database: urbaneye
User:     urbaneye
Password: urbaneye_dev
```

---

## How to Install Backend Dependencies

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Start the FastAPI Server

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```

The server starts on **http://localhost:8000**.

Database tables are created automatically on first startup.

---

## How to Access API Documentation

| Interface | URL |
|---|---|
| Swagger UI (interactive) | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| OpenAPI JSON | http://localhost:8000/openapi.json |

---

## How to Run Tests

```bash
cd backend
source .venv/bin/activate
pytest -v
```

**Note:** Tests run against the real development PostgreSQL/PostGIS database.  
Each test that writes data uses a SAVEPOINT transaction that is rolled back after the test, so no test data persists.  
The Docker container must be running before executing the tests.

---

## Example: Create an Incident

```bash
curl -X POST http://localhost:8000/api/v1/incidents/ \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Large pothole near City Bus Stop 14",
    "description": "Approximately 30 cm deep pothole causing vehicle damage.",
    "category": "POTHOLE",
    "source": "CITIZEN",
    "severity": "HIGH",
    "priority": 3,
    "latitude": 12.9716,
    "longitude": 77.5946
  }'
```

Expected response (HTTP 201):

```json
{
  "id": 1,
  "title": "Large pothole near City Bus Stop 14",
  "description": "Approximately 30 cm deep pothole causing vehicle damage.",
  "category": "POTHOLE",
  "source": "CITIZEN",
  "status": "REPORTED",
  "severity": "HIGH",
  "priority": 3,
  "latitude": 12.9716,
  "longitude": 77.5946,
  "created_at": "2026-09-10T16:10:00Z",
  "updated_at": "2026-09-10T16:10:00Z"
}
```

---

## API Endpoints — Phase 1

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/health/db` | Database connectivity probe |
| POST | `/api/v1/incidents/` | Create an incident |
| GET | `/api/v1/incidents/` | List incidents (paginated) |
| GET | `/api/v1/incidents/{id}` | Get incident by ID |
| PATCH | `/api/v1/incidents/{id}/status` | Update incident status |

---

## Not Yet Implemented

The following features are planned for future phases and are **not present** in Phase 1:

- 🔐 Authentication / JWT
- 👤 User accounts and officer management
- 🤖 AI classification (NLP, Computer Vision, YOLO)
- 🧠 ML priority prediction
- 🗺️ Department routing
- 📲 Notifications (email, SMS, push)
- 🔁 Duplicate detection
- ⚡ Redis / Celery async tasks
- 📷 Camera / IoT integration
- 🌐 Frontend web application
- 📊 Analytics dashboard
- 🚁 Kubernetes / production deployment
