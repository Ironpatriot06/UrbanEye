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

Priority and SLA
----------------
On creation, priority_level is determined from the incident category using
CATEGORY_DEFAULT_PRIORITY.  The severity column is derived automatically.
SLA hours are set from DEFAULT_SLA_HOURS[priority_level].
The sla_deadline is calculated as created_at + sla_hours.

This is the clean hook for future AI analysis — the AI module calls
update_priority() to override the system default before or after creation.

Status transitions
------------------
VALID_TRANSITIONS defines allowed next-states for each current state.
AGENT_TRANSITIONS is the subset agents may use.
The service enforces these rules and raises HTTP 422 on violations.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import HTTPException, status as http_status
from sqlalchemy.orm import Session

from app.models.incident import (
    AGENT_TRANSITIONS,
    CATEGORY_DEFAULT_PRIORITY,
    DEFAULT_SLA_HOURS,
    PRIORITY_LABELS,
    PRIORITY_TO_INT,
    PRIORITY_TO_SEVERITY,
    SEVERITY_TO_PRIORITY,
    SLA_AT_RISK_RATIO,
    SLA_FREEZE_STATUSES,
    VALID_TRANSITIONS,
    Incident,
    IncidentPriority,
    IncidentSeverity,
    IncidentStatus,
)
from app.models.user import User, UserRole
from app.schemas.incident import (
    AgentAssignUpdate,
    IncidentCreate,
    IncidentStatusUpdate,
    PriorityUpdate,
    SLAUpdate,
)
from app.services.image_service import count_images_for_incident


def _make_point_wkt(latitude: float, longitude: float) -> str:
    """Return a WKT string for a PostGIS GEOGRAPHY POINT."""
    return f"SRID=4326;POINT({longitude} {latitude})"


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Return a timezone-aware UTC datetime (naive values are assumed UTC)."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _compute_sla_status(
    sla_deadline: Optional[datetime],
    sla_hours: Optional[int] = None,
) -> Optional[str]:
    """
    Compute SLA status from the deadline and the current time.

    Returns None if no deadline is set.

    BREACHED — the deadline has passed.
    AT_RISK  — less than SLA_AT_RISK_RATIO (20%) of the SLA window remains.
    ON_TRACK — otherwise.

    When `sla_hours` is unknown we fall back to a fixed two-hour warning window
    so legacy rows without an SLA size still report something sensible.
    """
    sla_deadline = _as_utc(sla_deadline)
    if sla_deadline is None:
        return None
    now = datetime.now(timezone.utc)
    if now >= sla_deadline:
        return "BREACHED"
    remaining = (sla_deadline - now).total_seconds()
    if sla_hours:
        warn_window = sla_hours * 3600 * SLA_AT_RISK_RATIO
    else:
        warn_window = 7200
    if remaining < warn_window:
        return "AT_RISK"
    return "ON_TRACK"


def _set_priority_fields(incident: Incident, priority_level: IncidentPriority) -> None:
    """
    Apply a priority level to an incident, deriving severity, legacy priority, and SLA.

    This is the single seam a future AI classifier plugs into: it decides the
    priority level, then calls this to keep every derived field consistent.
    Nothing here guesses or pretends to be intelligent — the mapping tables in
    app/models/incident.py are the current, explicit policy.

    The SLA deadline is anchored to the incident's creation time (not "now"),
    so re-prioritising an old incident does not silently grant it a fresh
    window.
    """
    incident.priority_level = priority_level
    incident.severity = PRIORITY_TO_SEVERITY[priority_level]
    incident.priority = PRIORITY_TO_INT[priority_level]

    sla_hours = DEFAULT_SLA_HOURS[priority_level]
    incident.sla_hours = sla_hours
    anchor = _as_utc(incident.created_at) or datetime.now(timezone.utc)
    incident.sla_deadline = anchor + timedelta(hours=sla_hours)
    incident.sla_status = _compute_sla_status(incident.sla_deadline, sla_hours)


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
        if a.last_assigned_at is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        return a.last_assigned_at

    return min(agents, key=sort_key)


def _enrich_incident(db: Session, incident: Incident) -> Incident:
    """
    Attach computed fields to an incident object so IncidentRead schema
    can populate reported_by_name, assigned_agent_name, image_count, sla_status.
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

    # Backward compatibility: rows created before the priority_level column
    # existed may still be NULL if the SQL migration has not been run.  Derive
    # a priority from the legacy severity so the incident still serialises.
    if incident.priority_level is None and incident.severity is not None:
        incident.priority_level = SEVERITY_TO_PRIORITY.get(
            incident.severity, IncidentPriority.P3
        )

    incident.priority_label = PRIORITY_LABELS.get(incident.priority_level, "")

    # Refresh SLA status dynamically rather than trusting the cached column —
    # except once the incident is RESOLVED/CLOSED, where the clock has stopped
    # and the stored verdict is the historical truth.
    if incident.sla_deadline and incident.status not in SLA_FREEZE_STATUSES:
        incident.sla_status = _compute_sla_status(
            incident.sla_deadline, incident.sla_hours
        )

    return incident


def create_incident(
    db: Session,
    payload: IncidentCreate,
    reported_by_id: Optional[int] = None,
) -> Incident:
    """
    Persist a new incident and return the created row.

    - Priority is determined from the incident category (CATEGORY_DEFAULT_PRIORITY).
    - SLA is set from DEFAULT_SLA_HOURS[priority_level].
    - Automatically assigns the incident to the longest-idle available agent.
    - If no agent is available, the incident remains REPORTED.
    """
    # Determine system priority from category
    priority_level = CATEGORY_DEFAULT_PRIORITY.get(payload.category, IncidentPriority.P3)
    severity = PRIORITY_TO_SEVERITY[priority_level]
    priority_int = PRIORITY_TO_INT[priority_level]
    sla_hours = DEFAULT_SLA_HOURS[priority_level]

    # Find best available agent
    agent = _find_best_available_agent(db)
    incident_status = IncidentStatus.REPORTED
    agent_id = None

    if agent is not None:
        agent_id = agent.id
        incident_status = IncidentStatus.ASSIGNED
        agent.last_assigned_at = datetime.now(timezone.utc)

    now = datetime.now(timezone.utc)
    sla_deadline = now + timedelta(hours=sla_hours)

    incident = Incident(
        title=payload.title,
        description=payload.description,
        category=payload.category,
        source=payload.source,
        status=incident_status,
        priority_level=priority_level,
        severity=severity,
        priority=priority_int,
        sla_hours=sla_hours,
        sla_deadline=sla_deadline,
        sla_status=_compute_sla_status(sla_deadline, sla_hours),
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
    user_id: Optional[int] = None,
    agent_id: Optional[int] = None,
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
    is_agent: bool = False,
) -> Incident:
    """
    Apply a status transition and persist.

    Raises HTTP 422 for invalid transitions.
    Raises HTTP 403 if an agent attempts a transition outside AGENT_TRANSITIONS.
    """
    current = incident.status
    new_status = payload.status

    # Agents are restricted to a strict subset of the workflow
    # (ASSIGNED -> IN_PROGRESS -> RESOLVED).  Check this first so an agent
    # always gets a consistent 403 for anything outside their lane, rather
    # than a 422 that leaks which transitions admins can make.
    if is_agent:
        agent_allowed = AGENT_TRANSITIONS.get(current, set())
        if new_status not in agent_allowed:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail=(
                    "Agents may only move an incident ASSIGNED -> IN_PROGRESS -> RESOLVED. "
                    f"Allowed from {current.value}: "
                    f"{sorted(s.value for s in agent_allowed) or ['none']}. "
                    f"Attempted: {current.value} -> {new_status.value}."
                ),
            )

    # Validate the transition against the shared workflow rules
    allowed = VALID_TRANSITIONS.get(current, set())
    if new_status not in allowed:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Invalid status transition: {current.value} -> {new_status.value}. "
                f"Allowed next statuses: "
                f"{sorted(s.value for s in allowed) or ['none (terminal state)']}."
            ),
        )

    incident.status = new_status

    # Reaching RESOLVED/CLOSED stops the SLA clock — stamp the final verdict so
    # it is preserved instead of drifting to BREACHED as time passes.
    if new_status in SLA_FREEZE_STATUSES and incident.sla_deadline:
        incident.sla_status = _compute_sla_status(
            incident.sla_deadline, incident.sla_hours
        )

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

    By default, raises HTTP 422 if the target agent is unavailable.
    Pass override_availability=True to force the assignment (admin override).

    When assigning, also moves status to ASSIGNED if currently REPORTED/TRIAGED.
    """
    if payload.agent_id is not None:
        agent = db.query(User).filter(User.id == payload.agent_id).first()
        if agent is None:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Agent with id={payload.agent_id} not found.",
            )
        if not agent.is_available and not payload.override_availability:
            raise HTTPException(
                status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Agent '{agent.name}' is currently unavailable. "
                    "Set override_availability=true to assign anyway."
                ),
            )
        incident.assigned_agent_id = payload.agent_id
        if incident.status in (IncidentStatus.REPORTED, IncidentStatus.TRIAGED):
            incident.status = IncidentStatus.ASSIGNED
        agent.last_assigned_at = datetime.now(timezone.utc)
    else:
        # Unassign
        incident.assigned_agent_id = None
        if incident.status == IncidentStatus.ASSIGNED:
            incident.status = IncidentStatus.REPORTED

    db.commit()
    db.refresh(incident)
    return _enrich_incident(db, incident)


def update_sla(
    db: Session,
    incident: Incident,
    payload: SLAUpdate,
) -> Incident:
    """
    Admin-only: override the SLA hours and recalculate the deadline.

    Setting sla_hours=None clears the SLA.
    This is the hook for future AI modules to set custom SLAs.
    """
    if payload.sla_hours is None:
        incident.sla_hours = None
        incident.sla_deadline = None
        incident.sla_status = None
    else:
        incident.sla_hours = payload.sla_hours
        # Recalculate from creation time so the SLA always means "resolved
        # within N hours of being reported", not "N hours from this edit".
        anchor = _as_utc(incident.created_at) or datetime.now(timezone.utc)
        incident.sla_deadline = anchor + timedelta(hours=payload.sla_hours)
        incident.sla_status = _compute_sla_status(
            incident.sla_deadline, payload.sla_hours
        )
    db.commit()
    db.refresh(incident)
    return _enrich_incident(db, incident)


def update_priority(
    db: Session,
    incident: Incident,
    payload: PriorityUpdate,
) -> Incident:
    """
    Admin-only (also AI hook): update incident priority level.

    Recalculates severity, legacy priority integer, and SLA hours.
    """
    _set_priority_fields(incident, payload.priority_level)
    db.commit()
    db.refresh(incident)
    return _enrich_incident(db, incident)


def count_open_incidents_for_agent(db: Session, agent_id: int) -> int:
    """
    Return the number of incidents currently on an agent's plate.

    "Open" means assigned to the agent and not yet RESOLVED or CLOSED.  The
    admin dashboard uses this to distinguish an agent who is available but
    already busy from one who is genuinely free.
    """
    return (
        db.query(Incident)
        .filter(
            Incident.assigned_agent_id == agent_id,
            Incident.status.notin_(list(SLA_FREEZE_STATUSES)),
        )
        .count()
    )
