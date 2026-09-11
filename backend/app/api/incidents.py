"""
Incidents API router.

Endpoints
---------
POST   /api/v1/incidents/
    Create a new incident (authenticated; multipart/form-data with optional images)
    Citizens provide: title, description, category, source, lat/lon.
    System determines: priority_level, severity, SLA.

GET    /api/v1/incidents/
    List incidents — role-filtered:
    USER  -> own incidents only
    AGENT -> incidents assigned to them
    ADMIN -> all incidents

GET    /api/v1/incidents/{incident_id}
    Get one incident — role-gated

PATCH  /api/v1/incidents/{incident_id}/status
    Update status:
    USER  -> 403
    AGENT -> own assigned incidents only (ASSIGNED->IN_PROGRESS->RESOLVED)
    ADMIN -> any valid transition

PUT    /api/v1/incidents/{incident_id}/assign
    Admin: manually assign/reassign an agent
    Pass override_availability=true to assign an unavailable agent

PATCH  /api/v1/incidents/{incident_id}/sla
    Admin-only: override SLA hours

PATCH  /api/v1/incidents/{incident_id}/priority
    Admin-only: override priority level (AI hook)
"""

import json
from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, require_admin
from app.db.database import get_db
from app.models.incident import IncidentCategory, IncidentStatus
from app.models.user import UserRole
from app.schemas.incident import (
    AgentAssignUpdate,
    IncidentCreate,
    IncidentRead,
    IncidentStatusUpdate,
    PriorityUpdate,
    SLAUpdate,
)
from app.services import image_service, incident_service

router = APIRouter(prefix="/incidents", tags=["Incidents"])


def _assert_incident_access(incident, current_user):
    """Raise 404/403 if the current user should not see this incident."""
    if incident is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found.")
    role = current_user.role
    if role == UserRole.ADMIN:
        return
    if role == UserRole.USER and incident.reported_by != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found.")
    if role == UserRole.AGENT and incident.assigned_agent_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")


@router.post(
    "/",
    response_model=IncidentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new incident",
    description=(
        "Report a new civic incident with optional image uploads. "
        "Accepts multipart/form-data. "
        "The `data` field must be a JSON string containing: title, description, "
        "category, source, latitude, longitude. "
        "Priority and SLA are determined automatically by the system. "
        "Images are optional: provide one or more files in the `images` field."
    ),
)
async def create_incident(
    data: str = Form(..., description="JSON string of incident fields"),
    images: List[UploadFile] = File(default=[]),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> IncidentRead:
    try:
        payload_dict = json.loads(data)
        # Strip any client-supplied severity/priority to prevent user manipulation
        payload_dict.pop("severity", None)
        payload_dict.pop("priority", None)
        payload_dict.pop("priority_level", None)
        payload = IncidentCreate(**payload_dict)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid incident data: {exc}",
        )

    incident = incident_service.create_incident(
        db, payload, reported_by_id=current_user.id
    )

    # Persist any uploaded images
    for upload in images:
        if upload.filename:
            await image_service.save_image(db, incident.id, upload)

    # Re-fetch with updated image_count
    return incident_service.get_incident_by_id(db, incident.id)


@router.get(
    "/",
    response_model=List[IncidentRead],
    summary="List incidents",
    description=(
        "Return a role-filtered paginated list of incidents, newest first. "
        "USER -> own incidents. AGENT -> assigned incidents. ADMIN -> all incidents."
    ),
)
def list_incidents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    status_filter: Optional[IncidentStatus] = Query(None, alias="status"),
    category: Optional[IncidentCategory] = Query(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> List[IncidentRead]:
    role = current_user.role
    user_id = current_user.id if role == UserRole.USER else None
    agent_id = current_user.id if role == UserRole.AGENT else None

    category_str = category.value if category is not None else None
    return incident_service.get_incidents(
        db,
        skip=skip,
        limit=limit,
        status=status_filter,
        category=category_str,
        user_id=user_id,
        agent_id=agent_id,
    )


@router.get(
    "/{incident_id}",
    response_model=IncidentRead,
    summary="Get an incident by ID",
)
def get_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> IncidentRead:
    incident = incident_service.get_incident_by_id(db, incident_id)
    _assert_incident_access(incident, current_user)
    return incident


@router.patch(
    "/{incident_id}/status",
    response_model=IncidentRead,
    summary="Update incident status",
    description=(
        "AGENT: ASSIGNED->IN_PROGRESS->RESOLVED only. "
        "ADMIN: any valid transition. "
        "USER: not permitted (403)."
    ),
)
def update_status(
    incident_id: int,
    payload: IncidentStatusUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> IncidentRead:
    role = current_user.role

    if role == UserRole.USER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Citizens cannot update incident status.",
        )

    incident = incident_service.get_incident_by_id(db, incident_id)
    if incident is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found.")

    if role == UserRole.AGENT and incident.assigned_agent_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update incidents assigned to you.",
        )

    is_agent = (role == UserRole.AGENT)
    return incident_service.update_incident_status(db, incident, payload, is_agent=is_agent)


@router.put(
    "/{incident_id}/assign",
    response_model=IncidentRead,
    summary="Assign or reassign an agent (admin only)",
    description=(
        "Admin: manually assign an agent to an incident or remove assignment. "
        "By default, returns 422 if the agent is unavailable. "
        "Set override_availability=true to force-assign an unavailable agent."
    ),
)
def assign_agent(
    incident_id: int,
    payload: AgentAssignUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(require_admin),
) -> IncidentRead:
    incident = incident_service.get_incident_by_id(db, incident_id)
    if incident is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found.")
    return incident_service.assign_agent(db, incident, payload)


@router.patch(
    "/{incident_id}/sla",
    response_model=IncidentRead,
    summary="Override SLA hours (admin only)",
    description="Admin: manually set the SLA target hours for an incident.",
)
def update_sla(
    incident_id: int,
    payload: SLAUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(require_admin),
) -> IncidentRead:
    incident = incident_service.get_incident_by_id(db, incident_id)
    if incident is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found.")
    return incident_service.update_sla(db, incident, payload)


@router.patch(
    "/{incident_id}/priority",
    response_model=IncidentRead,
    summary="Override priority level (admin only / AI hook)",
    description=(
        "Admin: manually set the priority level. "
        "This endpoint is also the hook for the future AI analysis module."
    ),
)
def update_priority(
    incident_id: int,
    payload: PriorityUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(require_admin),
) -> IncidentRead:
    incident = incident_service.get_incident_by_id(db, incident_id)
    if incident is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found.")
    return incident_service.update_priority(db, incident, payload)
