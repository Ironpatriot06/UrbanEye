"""
UrbanEye+ FastAPI application entry point.

Application layout
------------------
  /               — root redirect to /docs
  /health         — liveness probe
  /health/db      — database connectivity probe
  /api/v1/auth    — authentication (register, login, me)
  /api/v1/incidents   — incident CRUD (role-filtered, auth-required)
  /api/v1/agents  — agent management
  /api/v1/        — image endpoints (mounted under incidents)

Table creation
------------------
`Base.metadata.create_all(engine)` is called inside the lifespan startup so
the schema is created automatically in development.  When Alembic migrations
are introduced (Phase 2+), this call should be removed and replaced by
running `alembic upgrade head`.

CORS
----
The Next.js frontend runs on http://localhost:3000 in development.
We configure CORS to allow requests from that origin.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import get_settings
from app.db.base import Base
from app.db.database import check_db_connection, engine

# Import ALL models before create_all so SQLAlchemy discovers every table.
from app.models import incident as _incident_models  # noqa: F401
from app.models import user as _user_models  # noqa: F401
from app.models import image as _image_models  # noqa: F401

from app.api.auth import router as auth_router
from app.api.incidents import router as incidents_router
from app.api.images import router as images_router
from app.api.agents import router as agents_router

settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create database tables on startup (idempotent)."""
    Base.metadata.create_all(bind=engine)
    yield


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "**UrbanEye+** — AI-Powered Smart City Incident Reporting & Intelligent "
        "Response System.\n\n"
        "**Review III** implements: JWT authentication, role-based access (USER/ADMIN/AGENT), "
        "incident image uploads, automatic agent assignment queue, and three-role dashboards."
    ),
    contact={"name": "UrbanEye+ Team"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow Next.js dev server and common localhost ports
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the bare root to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    tags=["Health"],
    summary="Liveness probe",
    description="Returns a simple JSON object confirming the API process is running.",
)
def health_check() -> dict:
    return {"status": "ok", "service": settings.APP_NAME, "version": settings.APP_VERSION}


@app.get(
    "/health/db",
    tags=["Health"],
    summary="Database connectivity probe",
    description=(
        "Attempts to execute a trivial query (`SELECT 1`) against PostgreSQL. "
        "Returns `{\"status\": \"ok\"}` if the connection succeeds, or a 503 "
        "error with details if it fails."
    ),
)
def health_db() -> dict:
    from fastapi import HTTPException, status as http_status

    try:
        check_db_connection()
        return {"status": "ok", "database": "postgresql+postgis"}
    except Exception as exc:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {exc}",
        )


# ---------------------------------------------------------------------------
# API routers
# ---------------------------------------------------------------------------

app.include_router(auth_router, prefix="/api/v1")
app.include_router(incidents_router, prefix="/api/v1")
app.include_router(images_router, prefix="/api/v1")
app.include_router(agents_router, prefix="/api/v1")
