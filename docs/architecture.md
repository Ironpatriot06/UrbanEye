# UrbanEye+ — System Architecture

## Overview

This document describes the current (Phase 1) architecture and the planned future architecture of the UrbanEye+ smart city incident response platform.

---

## Phase 1 — Currently Implemented

### Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│                   Client                        │
│  (curl / Swagger UI / future web frontend)      │
└───────────────────────┬─────────────────────────┘
                        │ HTTP/REST
                        ▼
┌─────────────────────────────────────────────────┐
│              FastAPI Application                │
│                                                 │
│  ┌──────────────┐    ┌───────────────────────┐  │
│  │   Routers    │───▶│   Service Layer       │  │
│  │  /incidents  │    │  incident_service.py  │  │
│  │  /health     │    └──────────┬────────────┘  │
│  └──────────────┘               │               │
│                                 ▼               │
│              ┌──────────────────────────┐       │
│              │    SQLAlchemy ORM        │       │
│              │    GeoAlchemy2           │       │
│              └──────────┬───────────────┘       │
└─────────────────────────┼───────────────────────┘
                          │ psycopg2
                          ▼
┌─────────────────────────────────────────────────┐
│          PostgreSQL 16 + PostGIS 3.4            │
│                                                 │
│  ┌─────────────────────────────────────────┐    │
│  │  incidents table                        │    │
│  │  - id, title, description               │    │
│  │  - category, source, status, severity   │    │
│  │  - latitude, longitude (FLOAT)          │    │
│  │  - location (GEOGRAPHY POINT, SRID 4326)│    │
│  │  - created_at, updated_at               │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | File | Responsibility |
|---|---|---|
| Settings | `app/core/config.py` | Read environment variables, provide typed config |
| ORM Base | `app/db/base.py` | Shared SQLAlchemy declarative base |
| Database | `app/db/database.py` | Engine, session factory, health check |
| Model | `app/models/incident.py` | Incident table definition + enums |
| Schemas | `app/schemas/incident.py` | Pydantic input validation + response shape |
| Service | `app/services/incident_service.py` | Business logic, DB operations |
| Router | `app/api/incidents.py` | HTTP endpoint definitions |
| Application | `app/main.py` | FastAPI instance, startup, health endpoints |

### Data Flow for POST /api/v1/incidents

```
HTTP POST (JSON body)
    │
    ▼
FastAPI router validates body against IncidentCreate schema (Pydantic)
    │  400/422 if invalid
    ▼
incident_service.create_incident(db, payload)
    │
    ▼
Constructs PostGIS WKT: "SRID=4326;POINT(<lon> <lat>)"
    │
    ▼
Inserts Incident row into PostgreSQL
    │
    ▼
Returns Incident ORM object → serialised as IncidentRead schema
    │
    ▼
HTTP 201 (JSON response)
```

### Spatial Storage

The `location` column uses PostGIS **GEOGRAPHY(POINT, 4326)**:

- **GEOGRAPHY** (not GEOMETRY) — uses the WGS-84 ellipsoid, so `ST_DWithin` returns distances in metres, not degrees.
- **SRID 4326** — standard GPS coordinate system (latitude/longitude).
- `latitude` and `longitude` are also stored as plain `FLOAT` columns for simple queries and JSON serialisation.

Future proximity queries will look like:

```sql
SELECT * FROM incidents
WHERE ST_DWithin(
    location,
    ST_GeogFromText('POINT(77.5946 12.9716)'),
    500  -- 500 metres
);
```

---

## Future Architecture — Planned Phases

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Incident Sources                                  │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │   Citizen    │  │  CCTV Camera │  │  IoT Sensors              │  │
│  │  Mobile App  │  │  (YOLO/CV)   │  │  (Smart bins, air, flood) │  │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬──────────────┘  │
└─────────┼─────────────────┼───────────────────────┼─────────────────┘
          │                 │                       │
          └────────────────▼────────────────────────┘
                    Ingestion Layer (FastAPI + Celery)
                           │
                           ▼
               ┌───────────────────────┐
               │   AI / NLP Engine     │
               │  - Text classification│
               │  - Image recognition  │
               │  - Deduplication      │
               └──────────┬────────────┘
                          │
                          ▼
               ┌───────────────────────┐
               │  Incident Intelligence│
               │  - Fusion             │
               │  - Enrichment         │
               │  - Severity scoring   │
               └──────────┬────────────┘
                          │
                          ▼
               ┌───────────────────────┐
               │  ML Priority Engine   │
               │  - Priority scoring   │
               │  - SLA calculation    │
               └──────────┬────────────┘
                          │
                          ▼
               ┌───────────────────────┐
               │  Department Router    │
               │  - Rule-based routing │
               │  - Assignment         │
               └──────────┬────────────┘
                          │
                          ▼
               ┌───────────────────────┐
               │  Workflow Automation  │
               │  - Notifications      │
               │  - Escalation         │
               │  - SLA monitoring     │
               └──────────┬────────────┘
                          │
                          ▼
               ┌───────────────────────┐
               │  Resolution & Audit   │
               │  - Resolution tracking│
               │  - Audit logs         │
               │  - Citizen feedback   │
               └──────────┬────────────┘
                          │
               ┌──────────▼────────────┐
               │  PostgreSQL/PostGIS   │
               │  + Redis cache        │
               └──────────┬────────────┘
                          │
               ┌──────────▼────────────┐
               │  GIS / Analytics      │
               │  - Heat maps          │
               │  - Trend analysis     │
               │  - Reporting          │
               └───────────────────────┘
```

### Implementation Roadmap

| Phase | Focus | Status |
|---|---|---|
| **Phase 1** | Foundation + Incident Core | ✅ **Implemented** |
| Phase 2 | Authentication + User Management | 🔲 Planned |
| Phase 3 | AI/NLP Classification | 🔲 Planned |
| Phase 4 | Computer Vision / Camera Integration | 🔲 Planned |
| Phase 5 | ML Priority Engine | 🔲 Planned |
| Phase 6 | Department Routing + SLA | 🔲 Planned |
| Phase 7 | Notifications (email, SMS, push) | 🔲 Planned |
| Phase 8 | IoT Integration | 🔲 Planned |
| Phase 9 | Frontend / Mobile App | 🔲 Planned |
| Phase 10 | Production Infrastructure (k8s) | 🔲 Planned |

---

## Technology Decisions

### Why FastAPI?
- Automatic OpenAPI docs generation from type hints
- Native Pydantic integration for validation
- High performance (comparable to Node.js/Go)
- Easy to add async later

### Why SQLAlchemy 2.0 + GeoAlchemy2?
- Mature, battle-tested ORM
- GeoAlchemy2 provides PostGIS column types and spatial functions
- Easy migration to async (with asyncpg) without changing the service layer

### Why PostGIS GEOGRAPHY vs GEOMETRY?
- GEOGRAPHY uses the ellipsoidal Earth model — distance functions return metres, not degrees
- Correct behaviour for features spanning large areas or near the poles
- Slight performance overhead vs GEOMETRY, but irrelevant at city scale
- Enables accurate `ST_DWithin(<metres>)` queries with no manual conversion

### Why synchronous SQLAlchemy (not async)?
- Simpler to reason about during the academic development phase
- psycopg2 is the most stable and well-documented PostgreSQL driver
- When concurrency becomes a concern, the engine can be replaced with asyncpg and `async/await` without changing the service layer interface

### Why no Alembic in Phase 1?
- `Base.metadata.create_all()` is sufficient for a single-developer dev database
- Alembic should be introduced at Phase 2 when schema stability and team collaboration require tracked migrations
