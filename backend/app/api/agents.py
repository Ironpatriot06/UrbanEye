"""
Agents API router.

Endpoints
---------
GET   /api/v1/agents/me                        — AGENT: own profile + availability
POST  /api/v1/agents/availability              — AGENT: set own availability
GET   /api/v1/agents/                          — ADMIN: list all agents + workload
PATCH /api/v1/agents/{agent_id}/availability   — ADMIN: override an agent's availability

Availability semantics
----------------------
`is_available` is persistent database state.  Logging in never modifies it.
It changes only through the two availability endpoints above — the agent
setting their own status, or an admin overriding it.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.orm import Session

from app.core.deps import require_admin, require_agent
from app.db.database import get_db
from app.schemas.user import AgentAvailabilityUpdate, UserRead
from app.services import incident_service, user_service

router = APIRouter(prefix="/agents", tags=["Agents"])


@router.get(
    "/me",
    response_model=UserRead,
    summary="Get current agent profile",
    description=(
        "Returns the authenticated agent's profile and their persisted "
        "availability status.  Clients should call this on load rather than "
        "assuming a default — availability survives logout."
    ),
)
def agent_me(
    db: Session = Depends(get_db),
    current_user=Depends(require_agent),
) -> UserRead:
    current_user.active_incident_count = incident_service.count_open_incidents_for_agent(
        db, current_user.id
    )
    return current_user


@router.post(
    "/availability",
    response_model=UserRead,
    summary="Set own availability (agent only)",
    description=(
        "Toggle the authenticated agent's availability for new incident "
        "assignments.  The value is persisted and is not reset on login."
    ),
)
def set_availability(
    payload: AgentAvailabilityUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(require_agent),
) -> UserRead:
    return user_service.set_agent_availability(db, current_user, payload.is_available)


@router.get(
    "/",
    response_model=List[UserRead],
    summary="List all agents (admin only)",
    description=(
        "Returns every agent with their availability status, the time of their "
        "last assignment, and the number of incidents still open on their plate."
    ),
)
def list_agents(
    db: Session = Depends(get_db),
    current_user=Depends(require_admin),
) -> List[UserRead]:
    return user_service.get_all_agents(db)


@router.patch(
    "/{agent_id}/availability",
    response_model=UserRead,
    summary="Override an agent's availability (admin only)",
    description=(
        "Admin: mark another agent available or unavailable.  Use this to bring "
        "an agent back into the assignment queue, or to take them out of it."
    ),
)
def admin_set_availability(
    payload: AgentAvailabilityUpdate,
    agent_id: int = Path(..., description="ID of the agent to update."),
    db: Session = Depends(get_db),
    current_user=Depends(require_admin),
) -> UserRead:
    agent = user_service.get_agent_by_id(db, agent_id)
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent with id={agent_id} not found.",
        )
    updated = user_service.set_agent_availability(db, agent, payload.is_available)
    updated.active_incident_count = incident_service.count_open_incidents_for_agent(
        db, updated.id
    )
    return updated
