"""
Incident service layer.

All database interactions for incidents live here.  The API router calls
these functions rather than talking to the database directly.

PostGIS POINT construction
---------------------------
PostGIS expects coordinates in (longitude, latitude) order inside WKT.
We pass a WKT string with the SRID prefix so SQLAlchemy / GeoAlchemy2
stores a fully-typed GEOGRAPHY value rather than raw bytes.

    "SRID=4326;POINT(<longitude> <latitude>)"

Agent assignment queue
----------------------
When a new incident is created, the service searches for available agents
and selects the one with the oldest `last_assigned_at` timestamp (or NULL,
which is treated as highest priority — never been assigned).
"""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.incident import Incident, IncidentStatus
from app.models.user import User, UserRole
from app.schemas.incident import AgentAssignUpdate, IncidentCreate, IncidentStatusUpdate
from app.services.image_service import count_images_for_incident


def _make_point_wkt(latitude: float, longitude: float) -> str:
    """Return a WKT string for a PostGIS GEOGRAPHY POINT."""
    return f"SRID=4326;POINT({longitude} {latitude})"


def _find_best_available_agent(db: Session) -> Optional[User]:
    """
    Return the available agent who has been idle the longest.

    Selection rules:
    1. Agent must have is_available = True.
    2. Prefer agents with last_assigned_at = NULL (never assigned).
    3. Among assigned agents, choose the one with the oldest last_assigned_at.
    """
    agents = (
        db.query(User)
        .filter(User.role == UserRole.AGENT, User.is_available == True)
        .all()
    )
    if not agents:
        return None

    def sort_key(a: User):
        # None (never assigned) sorts first (lowest possible timestamp)
        if a.last_assigned_at is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        return a.last_assigned_at

    return min(agents, key=sort_key)


def _enrich_incident(db: Session, incident: Incident) -> Incident:
    """
    Attach computed fields to an incident object so IncidentRead schema
    can populate reported_by_name, assigned_agent_name, image_count.
    These are transient attributes — not ORM columns.
    """
    if incident.reported_by:
        reporter = db.query(User).filter(User.id == incident.reported_by).first()
        incident.reported_by_name = reporter.name if reporter else None
    else:
        incident.reported_by_name = None

    if incident.assigned_agent_id:
        agent = db.query(User).filter(User.id == incident.assigned_agent_id).first()
        incident.assigned_agent_name = agent.name if agent else None
    else:
        incident.assigned_agent_name = None

    incident.image_count = count_images_for_incident(db, incident.id)
    return incident


def create_incident(
    db: Session,
    payload: IncidentCreate,
    reported_by_id: Optional[int] = None,
) -> Incident:
    """
    Persist a new incident and return the created row.

    Automatically assigns the incident to the longest-idle available agent
    and sets status to ASSIGNED.  If no agent is available, the incident
    remains in REPORTED status with no assignment.
    """
    agent = _find_best_available_agent(db)

    status = IncidentStatus.REPORTED
    agent_id = None

    if agent is not None:
        agent_id = agent.id
        status = IncidentStatus.ASSIGNED
        agent.last_assigned_at = datetime.now(timezone.utc)

    incident = Incident(
        title=payload.title,
        description=payload.description,
        category=payload.category,
        source=payload.source,
        severity=payload.severity,
        priority=payload.priority,
        status=status,
        latitude=payload.latitude,
        longitude=payload.longitude,
        location=_make_point_wkt(payload.latitude, payload.longitude),
        reported_by=reported_by_id,
        assigned_agent_id=agent_id,
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return _enrich_incident(db, incident)


def get_incidents(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    status: Optional[IncidentStatus] = None,
    category: Optional[str] = None,
    user_id: Optional[int] = None,       # filter to incidents reported by this user
    agent_id: Optional[int] = None,      # filter to incidents assigned to this agent
) -> List[Incident]:
    """
    Return a paginated, optionally-filtered list of incidents.

    Role-based filtering:
    - Pass user_id to show only that user's incidents (USER role).
    - Pass agent_id to show only incidents assigned to that agent (AGENT role).
    - Pass neither to show all incidents (ADMIN role).
    """
    query = db.query(Incident)

    if user_id is not None:
        query = query.filter(Incident.reported_by == user_id)
    if agent_id is not None:
        query = query.filter(Incident.assigned_agent_id == agent_id)
    if status is not None:
        query = query.filter(Incident.status == status)
    if category is not None:
        query = query.filter(Incident.category == category)

    incidents = (
        query.order_by(Incident.created_at.desc()).offset(skip).limit(limit).all()
    )
    return [_enrich_incident(db, inc) for inc in incidents]


def get_incident_by_id(db: Session, incident_id: int) -> Optional[Incident]:
    """Return a single incident by primary key, or None if not found."""
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if incident is None:
        return None
    return _enrich_incident(db, incident)


def update_incident_status(
    db: Session,
    incident: Incident,
    payload: IncidentStatusUpdate,
) -> Incident:
    """Apply a status transition and persist."""
    incident.status = payload.status  # type: ignore[assignment]
    db.commit()
    db.refresh(incident)
    return _enrich_incident(db, incident)


def assign_agent(
    db: Session,
    incident: Incident,
    payload: AgentAssignUpdate,
) -> Incident:
    """
    Admin: manually assign or unassign an agent to an incident.

    When assigning, also moves status to ASSIGNED if currently REPORTED/TRIAGED.
    """
    incident.assigned_agent_id = payload.agent_id
    if payload.agent_id is not None:
        if incident.status in (IncidentStatus.REPORTED, IncidentStatus.TRIAGED):
            incident.status = IncidentStatus.ASSIGNED
        # Update agent's last_assigned_at
        agent = db.query(User).filter(User.id == payload.agent_id).first()
        if agent:
            agent.last_assigned_at = datetime.now(timezone.utc)
    else:
        # Unassigned
        if incident.status == IncidentStatus.ASSIGNED:
            incident.status = IncidentStatus.REPORTED
    db.commit()
    db.refresh(incident)
    return _enrich_incident(db, incident)
