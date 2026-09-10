"""
Agents API router.

Endpoints
---------
GET  /api/v1/agents/me           — AGENT: get own info + assigned incidents
POST /api/v1/agents/availability — AGENT: toggle availability
GET  /api/v1/agents/             — ADMIN: list all agents
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, require_admin, require_agent
from app.db.database import get_db
from app.models.user import UserRole
from app.schemas.user import AgentAvailabilityUpdate, UserRead
from app.services import incident_service, user_service

router = APIRouter(prefix="/agents", tags=["Agents"])


@router.get(
    "/me",
    response_model=UserRead,
    summary="Get current agent profile",
    description="Returns the authenticated agent's profile and availability status.",
)
def agent_me(current_user=Depends(require_agent)) -> UserRead:
    return current_user


@router.post(
    "/availability",
    response_model=UserRead,
    summary="Set agent availability",
    description="Toggle the authenticated agent's availability for new incident assignments.",
)
def set_availability(
    payload: AgentAvailabilityUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(require_agent),
) -> UserRead:
    updated = user_service.set_agent_availability(db, current_user, payload.is_available)
    return updated


@router.get(
    "/",
    response_model=List[UserRead],
    summary="List all agents (admin only)",
    description="Returns all agents with their availability status.",
)
def list_agents(
    db: Session = Depends(get_db),
    current_user=Depends(require_admin),
) -> List[UserRead]:
    return user_service.get_all_agents(db)
