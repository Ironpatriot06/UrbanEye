"""
ML prediction endpoints (Phase 2C).

GET /api/v1/incidents/{incident_id}/predictions
    Resolution-time and SLA-breach predictions for one incident.
    Access: exactly the rule for reading the incident (incidents._assert_incident_access):
      USER  -> own incidents only (404 otherwise)
      AGENT -> incidents assigned to them (403 otherwise)
      ADMIN -> any incident
    `include_inputs=true` returns the model inputs, for admins only: they
    summarise OTHER incidents (nearby open reports, recent workload), which a
    citizen or agent must not be able to read through this endpoint.

GET /api/v1/ml/status            (admin)  model readiness, versions, service areas
GET /api/v1/ml/shadow-evaluation (admin)  logged predictions vs observed outcomes (aggregates)
GET /api/v1/hotspots             (admin, agent)  next-week EWMA counts per 1 km cell.
    Aggregates every citizen's report locations, so it is not exposed to USER.

Predictions are read-only with respect to incidents: nothing here changes an
incident's priority, SLA or status, and a prediction failure returns a typed
UNAVAILABLE result rather than an error.  Every call is computed on request
from the verified bundle; nothing is trained.
"""

from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.incidents import _assert_incident_access
from app.core.deps import get_current_user, require_admin, require_admin_or_agent
from app.db.database import get_db
from app.ml import service as ml_service
from app.ml.areas import get_areas
from app.ml.registry import get_registry
from app.models.incident import Incident
from app.models.user import UserRole
from app.schemas.prediction import HotspotForecast, IncidentPredictions

router = APIRouter(tags=["ML Predictions"])

Category = Literal["POTHOLE", "FLOOD", "FIRE_HAZARD", "GARBAGE", "STREETLIGHT", "OTHER"]


@router.get(
    "/incidents/{incident_id}/predictions",
    response_model=IncidentPredictions,
    summary="Resolution-time and SLA-breach predictions for an incident",
    description=(
        "Computed as of the incident's report time from earlier incidents only. "
        "Each result states its status (AVAILABLE / EXPERIMENTAL / FALLBACK / UNAVAILABLE), "
        "the method used (ML model or a named baseline), and warnings. "
        "No prediction has been validated on UrbanEye data."
    ),
)
def get_incident_predictions(
    incident_id: int,
    include_inputs: bool = Query(False, description="Admins only: return the model inputs"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> IncidentPredictions:
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    _assert_incident_access(incident, current_user)
    return ml_service.predict_incident(
        db, incident, include_inputs=include_inputs and current_user.role == UserRole.ADMIN)


@router.get("/ml/status", summary="ML artifact readiness (admin)")
def ml_status(_admin=Depends(require_admin)) -> dict:
    status = get_registry().public_status()
    status["service_areas"] = [
        {"name": a.name, "lat_min": a.lat_min, "lat_max": a.lat_max, "lon_min": a.lon_min,
         "lon_max": a.lon_max} for a in get_areas()]
    return status


@router.get("/ml/shadow-evaluation", summary="Logged predictions vs observed outcomes (admin)")
def ml_shadow_evaluation(db: Session = Depends(get_db), _admin=Depends(require_admin)) -> dict:
    """Aggregates only.  Monitoring of the shadow log — not a model validation."""
    from app.ml.evaluation import shadow_evaluation

    return shadow_evaluation(db)


@router.get(
    "/hotspots",
    response_model=HotspotForecast,
    summary="Next-week report counts per 1 km cell (EWMA, non-ML)",
)
def get_hotspots(
    category: Category = Query(..., description="UrbanEye category"),
    area: Optional[str] = Query(None, description="Service area name; default: the first"),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    _user=Depends(require_admin_or_agent),
) -> HotspotForecast:
    return ml_service.hotspots(db, area, category, limit)
