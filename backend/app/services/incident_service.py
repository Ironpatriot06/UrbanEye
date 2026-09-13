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

Audit trail
-----------
Every mutating function here also writes the matching IncidentHistory rows via
history_service, staged in the same transaction as the change itself and
committed with it — so an operation and its audit entry are never separated.

Two rules shape what gets written:

  * An event is recorded only when something actually changed.  Re-assigning an
    incident to the agent who already holds it, or saving an SLA that is
    already the current one, writes nothing, because nothing happened.
  * The actor is the authenticated `User` the router resolved from the JWT, or
    None for work the system did on its own (the assignment queue, the SLA
    stamped on a new incident).  It is never read from request input.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import HTTPException, status as http_status
from sqlalchemy.orm import Session

from app.models.history import STATUS_MILESTONE_ACTIONS, HistoryAction
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
from app.services import history_service
from app.services.image_service import count_images_for_incident


def _make_point_wkt(latitude: float, longitude: float) -> str:
    """Return a WKT string for a PostGIS GEOGRAPHY POINT."""
    return f"SRID=4326;POINT({longitude} {latitude})"


def _status_label(value: IncidentStatus) -> str:
    """'IN_PROGRESS' -> 'In Progress', for audit descriptions."""
    return value.value.replace("_", " ").title()


def _sla_label(hours: Optional[int]) -> str:
    """Render an SLA window the way the audit trail and UI display it."""
    if hours is None:
        return "No SLA"
    return f"{hours} hour" if hours == 1 else f"{hours} hours"


def _actor_label(actor: Optional[User]) -> str:
    """Name the actor for an audit description, or the system when nobody asked."""
    return actor.name if actor is not None else "the system"


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
    reporter: Optional[User] = None,
) -> Incident:
    """
    Persist a new incident and return the created row.

    - Priority is determined from the incident category (CATEGORY_DEFAULT_PRIORITY).
    - SLA is set from DEFAULT_SLA_HOURS[priority_level].
    - Automatically assigns the incident to the longest-idle available agent.
    - If no agent is available, the incident remains REPORTED.

    `reporter` is the authenticated citizen, used both as the incident's owner
    and as the actor on the opening audit event.  The automatic assignment and
    the default SLA are attributed to SYSTEM, because no person chose them.
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
        reported_by=reporter.id if reporter is not None else None,
        assigned_agent_id=agent_id,
    )
    db.add(incident)
    # Flush rather than commit: the incident needs an id so its audit entries
    # can reference it, but both must land in the same transaction.
    db.flush()

    history_service.record(
        db,
        incident.id,
        HistoryAction.INCIDENT_CREATED,
        actor=reporter,
        new_value=IncidentStatus.REPORTED.value,
        description=(
            f"Incident reported by {reporter.name}."
            if reporter is not None
            else "Incident reported."
        ),
    )

    # Recorded as SLA_CREATED, not SLA_UPDATED: this is the first window the
    # incident has ever had.  The description carries the reason, so a later
    # reader can see why the current SLA is what it is.
    history_service.record(
        db,
        incident.id,
        HistoryAction.SLA_CREATED,
        new_value=_sla_label(sla_hours),
        description=(
            f"Response target set to {_sla_label(sla_hours)} — the default for "
            f"{priority_level.value} ({PRIORITY_LABELS[priority_level]}) incidents."
        ),
    )

    if agent is not None:
        history_service.record(
            db,
            incident.id,
            HistoryAction.AGENT_ASSIGNED,
            new_value=agent.name,
            description=(
                f"Assigned automatically to {agent.name}, the available agent "
                "idle the longest."
            ),
        )
        history_service.record(
            db,
            incident.id,
            HistoryAction.STATUS_CHANGED,
            old_value=IncidentStatus.REPORTED.value,
            new_value=IncidentStatus.ASSIGNED.value,
            description="Status moved to Assigned when an agent was allocated.",
        )

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


def _record_status_change(
    db: Session,
    incident_id: int,
    old_status: IncidentStatus,
    new_status: IncidentStatus,
    actor: Optional[User],
    note: Optional[str] = None,
) -> None:
    """
    Write the audit entry for one status transition.

    Milestone targets (TRIAGED / RESOLVED / CLOSED) get their own action so the
    timeline reads as operations rather than a column of identical
    "status changed" lines; everything else is STATUS_CHANGED.  Moving back out
    of a finished state is a reopening, which is the one transition where the
    direction matters more than the destination.  Either way old_value and
    new_value carry the full transition, so no detail depends on the label.
    """
    if old_status in SLA_FREEZE_STATUSES and new_status not in SLA_FREEZE_STATUSES:
        action = HistoryAction.INCIDENT_REOPENED
    else:
        action = STATUS_MILESTONE_ACTIONS.get(
            new_status.value, HistoryAction.STATUS_CHANGED
        )

    description = (
        f"Status moved from {_status_label(old_status)} to "
        f"{_status_label(new_status)} by {_actor_label(actor)}."
    )
    if note:
        description = f"{description} {note}"

    history_service.record(
        db,
        incident_id,
        action,
        actor=actor,
        old_value=old_status.value,
        new_value=new_status.value,
        description=description,
    )


def update_incident_status(
    db: Session,
    incident: Incident,
    payload: IncidentStatusUpdate,
    is_agent: bool = False,
    actor: Optional[User] = None,
) -> Incident:
    """
    Apply a status transition and persist.

    Raises HTTP 422 for invalid transitions.
    Raises HTTP 403 if an agent attempts a transition outside AGENT_TRANSITIONS.

    A rejected transition writes no history: the audit trail records what
    happened, not what was attempted and refused.
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

    _record_status_change(db, incident.id, current, new_status, actor)

    db.commit()
    db.refresh(incident)
    return _enrich_incident(db, incident)


def assign_agent(
    db: Session,
    incident: Incident,
    payload: AgentAssignUpdate,
    actor: Optional[User] = None,
) -> Incident:
    """
    Admin: manually assign or unassign an agent to an incident.

    By default, raises HTTP 422 if the target agent is unavailable.
    Pass override_availability=True to force the assignment (admin override).

    When assigning, also moves status to ASSIGNED if currently REPORTED/TRIAGED.

    Audit entries written here:
      ADMIN_OVERRIDE   — only when the override flag actually took effect, i.e.
                         the target agent really was unavailable.
      AGENT_ASSIGNED / AGENT_REASSIGNED / AGENT_UNASSIGNED
      STATUS_CHANGED   — when the assignment moved the status as a side effect.
    """
    previous_agent = (
        db.query(User).filter(User.id == incident.assigned_agent_id).first()
        if incident.assigned_agent_id
        else None
    )
    previous_status = incident.status

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

        # The override is only real if the guard it bypasses would have fired.
        # Sending the flag for an available agent overrides nothing, so it
        # records nothing.
        overrode_availability = not agent.is_available

        incident.assigned_agent_id = payload.agent_id
        if incident.status in (IncidentStatus.REPORTED, IncidentStatus.TRIAGED):
            incident.status = IncidentStatus.ASSIGNED
        agent.last_assigned_at = datetime.now(timezone.utc)

        if overrode_availability:
            history_service.record(
                db,
                incident.id,
                HistoryAction.ADMIN_OVERRIDE,
                actor=actor,
                old_value="Unavailable",
                new_value="Assigned anyway",
                description=(
                    f"{_actor_label(actor)} overrode the availability check to assign "
                    f"{agent.name}, who is marked unavailable."
                ),
            )

        if previous_agent is None:
            history_service.record(
                db,
                incident.id,
                HistoryAction.AGENT_ASSIGNED,
                actor=actor,
                new_value=agent.name,
                description=f"Assigned to {agent.name} by {_actor_label(actor)}.",
            )
        elif previous_agent.id != agent.id:
            history_service.record(
                db,
                incident.id,
                HistoryAction.AGENT_REASSIGNED,
                actor=actor,
                old_value=previous_agent.name,
                new_value=agent.name,
                description=(
                    f"Reassigned from {previous_agent.name} to {agent.name} "
                    f"by {_actor_label(actor)}."
                ),
            )
        # Assigning the agent who already holds the incident changes nothing
        # about the assignment, so no assignment event is written.
    else:
        # Unassign
        incident.assigned_agent_id = None
        if incident.status == IncidentStatus.ASSIGNED:
            incident.status = IncidentStatus.REPORTED

        if previous_agent is not None:
            history_service.record(
                db,
                incident.id,
                HistoryAction.AGENT_UNASSIGNED,
                actor=actor,
                old_value=previous_agent.name,
                description=(
                    f"{previous_agent.name} was unassigned by {_actor_label(actor)}."
                ),
            )

    if incident.status != previous_status:
        _record_status_change(
            db,
            incident.id,
            previous_status,
            incident.status,
            actor,
            note="Followed from the change of assignment.",
        )

    db.commit()
    db.refresh(incident)
    return _enrich_incident(db, incident)


def update_sla(
    db: Session,
    incident: Incident,
    payload: SLAUpdate,
    actor: Optional[User] = None,
) -> Incident:
    """
    Admin-only: override the SLA hours and recalculate the deadline.

    Setting sla_hours=None clears the SLA.
    This is the hook for future AI modules to set custom SLAs.

    Records SLA_CREATED when the incident had no window before, SLA_UPDATED
    when one is being replaced or cleared, and nothing at all when the saved
    value matches the current one.
    """
    old_hours = incident.sla_hours

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

    if incident.sla_hours != old_hours:
        action = (
            HistoryAction.SLA_CREATED
            if old_hours is None
            else HistoryAction.SLA_UPDATED
        )
        if incident.sla_hours is None:
            note = f"{_actor_label(actor)} cleared the response target."
        else:
            note = (
                f"{_actor_label(actor)} set the response target to "
                f"{_sla_label(incident.sla_hours)} from the time of report."
            )
        history_service.record(
            db,
            incident.id,
            action,
            actor=actor,
            old_value=_sla_label(old_hours),
            new_value=_sla_label(incident.sla_hours),
            description=note,
        )

    db.commit()
    db.refresh(incident)
    return _enrich_incident(db, incident)


def update_priority(
    db: Session,
    incident: Incident,
    payload: PriorityUpdate,
    actor: Optional[User] = None,
) -> Incident:
    """
    Admin-only (also AI hook): update incident priority level.

    Recalculates severity, legacy priority integer, and SLA hours.

    Re-prioritising also resets the SLA window, so a priority change that moves
    the window writes a second SLA_UPDATED entry.  Without it the trail would
    show an SLA that changed with nothing to explain it.
    """
    old_priority = incident.priority_level
    old_sla_hours = incident.sla_hours

    _set_priority_fields(incident, payload.priority_level)

    if incident.priority_level != old_priority:
        history_service.record(
            db,
            incident.id,
            HistoryAction.PRIORITY_CHANGED,
            actor=actor,
            old_value=old_priority.value if old_priority is not None else None,
            new_value=incident.priority_level.value,
            description=(
                f"Priority changed to {incident.priority_level.value} "
                f"({PRIORITY_LABELS[incident.priority_level]}) by {_actor_label(actor)}."
            ),
        )

    if incident.sla_hours != old_sla_hours:
        history_service.record(
            db,
            incident.id,
            HistoryAction.SLA_UPDATED,
            actor=actor,
            old_value=_sla_label(old_sla_hours),
            new_value=_sla_label(incident.sla_hours),
            description=(
                f"Response target moved to {_sla_label(incident.sla_hours)} — the "
                f"default for {incident.priority_level.value} incidents — following "
                "the priority change."
            ),
        )

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
