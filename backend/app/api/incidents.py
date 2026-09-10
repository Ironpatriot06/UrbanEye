"""
Incidents API router.

Endpoints
---------
POST   /api/v1/incidents                    Create a new incident
GET    /api/v1/incidents                    List incidents (paginated, filterable)
GET    /api/v1/incidents/{incident_id}      Get one incident by ID
PATCH  /api/v1/incidents/{incident_id}/status  Update incident status
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.incident import IncidentCategory, IncidentStatus
from app.schemas.incident import IncidentCreate, IncidentRead, IncidentStatusUpdate
from app.services import incident_service

router = APIRouter(prefix="/incidents", tags=["Incidents"])


@router.post(
    "/",
    response_model=IncidentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new incident",
    description=(
        "Report a new civic incident. The `location` field is automatically "
        "constructed from the supplied `latitude` and `longitude` values and "
        "stored as a PostGIS GEOGRAPHY(POINT, 4326) for future spatial queries."
    ),
)
def create_incident(
    payload: IncidentCreate,
    db: Session = Depends(get_db),
) -> IncidentRead:
    incident = incident_service.create_incident(db, payload)
    return incident


@router.get(
    "/",
    response_model=List[IncidentRead],
    summary="List incidents",
    description=(
        "Return a paginated list of incidents, newest first.  "
        "Optionally filter by `status` or `category`."
    ),
)
def list_incidents(
    skip: int = Query(0, ge=0, description="Number of records to skip (pagination)."),
    limit: int = Query(100, ge=1, le=500, description="Maximum records to return."),
    status: Optional[IncidentStatus] = Query(None, description="Filter by status."),
    category: Optional[IncidentCategory] = Query(None, description="Filter by category."),
    db: Session = Depends(get_db),
) -> List[IncidentRead]:
    category_str = category.value if category is not None else None
    incidents = incident_service.get_incidents(
        db, skip=skip, limit=limit, status=status, category=category_str
    )
    return incidents


@router.get(
    "/{incident_id}",
    response_model=IncidentRead,
    summary="Get an incident by ID",
    description="Return the full detail of a single incident identified by its integer ID.",
)
def get_incident(
    incident_id: int,
    db: Session = Depends(get_db),
) -> IncidentRead:
    incident = incident_service.get_incident_by_id(db, incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found.",
        )
    return incident


@router.patch(
    "/{incident_id}/status",
    response_model=IncidentRead,
    summary="Update incident status",
    description=(
        "Transition an incident to a new status. "
        "Valid statuses: REPORTED → TRIAGED → ASSIGNED → IN_PROGRESS → RESOLVED → CLOSED."
    ),
)
def update_status(
    incident_id: int,
    payload: IncidentStatusUpdate,
    db: Session = Depends(get_db),
) -> IncidentRead:
    incident = incident_service.get_incident_by_id(db, incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found.",
        )
    updated = incident_service.update_incident_status(db, incident, payload)
    return updated
