"""
Incident history service — the only place audit rows are written.

Transaction contract
--------------------
`record()` adds the row and flushes it, but never commits.  The audit entry
therefore lives or dies with the operation that caused it: if the caller's
transaction rolls back, so does the history.  Callers commit once, after both
the change and its audit entry are staged.

Trust model
-----------
Actor identity comes from the authenticated `User` the router resolved from the
JWT, or from `SYSTEM` when no person requested the work.  Nothing here reads an
actor, a role or a timestamp from client input — there is no code path that
could, because `record()` takes an ORM User object, not a request payload.

Reading
-------
`get_history_for_incident()` returns events oldest -> newest, ordered by
(created_at, id).  The primary key is the tiebreaker so events written inside a
single transaction keep their true insertion order even if their clock readings
land on the same microsecond.
"""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.history import (
    ACTIONS_HIDDEN_FROM_REPORTER,
    ActorRole,
    HistoryAction,
    IncidentHistory,
)
from app.models.user import User, UserRole

#: Label shown for events no person performed.
SYSTEM_ACTOR_NAME = "System"


def _actor_fields(actor: Optional[User]) -> tuple[Optional[int], ActorRole, str]:
    """
    Resolve (actor_id, actor_role, actor_name) from an authenticated user.

    A missing actor is the system acting on its own — the assignment queue, the
    SLA stamped on a new incident — not an anonymous user.
    """
    if actor is None:
        return None, ActorRole.SYSTEM, SYSTEM_ACTOR_NAME

    # UserRole and ActorRole share member names; ActorRole additionally has
    # SYSTEM, which no User can ever hold.
    role_name = actor.role.value if isinstance(actor.role, UserRole) else str(actor.role)
    return actor.id, ActorRole(role_name), actor.name


def record(
    db: Session,
    incident_id: int,
    action: HistoryAction,
    *,
    actor: Optional[User] = None,
    old_value: Optional[str] = None,
    new_value: Optional[str] = None,
    description: Optional[str] = None,
) -> IncidentHistory:
    """
    Append one audit entry for an incident.

    Staged in the caller's transaction and flushed so it has an id, but not
    committed — see the module docstring.
    """
    actor_id, actor_role, actor_name = _actor_fields(actor)

    entry = IncidentHistory(
        incident_id=incident_id,
        actor_id=actor_id,
        actor_role=actor_role,
        actor_name=actor_name,
        action=action,
        old_value=old_value,
        new_value=new_value,
        description=description,
        # Server-derived wall clock.  Passed explicitly rather than left to the
        # column default so several events in one transaction are ordered by
        # when they happened, not by when the transaction opened.
        created_at=datetime.now(timezone.utc),
    )
    db.add(entry)
    db.flush()
    return entry


def get_history_for_incident(
    db: Session,
    incident_id: int,
    *,
    viewer_role: Optional[UserRole] = None,
) -> List[IncidentHistory]:
    """
    Return an incident's audit trail, oldest event first.

    `viewer_role` trims the trail for citizens: staffing events (an agent's
    availability changing, an admin overriding it) are internal operations that
    say nothing about the reporter's incident.  Agents and admins receive every
    event, which is what the agent dashboard and the admin audit view need.
    """
    query = db.query(IncidentHistory).filter(
        IncidentHistory.incident_id == incident_id
    )

    if viewer_role == UserRole.USER:
        query = query.filter(
            IncidentHistory.action.notin_(list(ACTIONS_HIDDEN_FROM_REPORTER))
        )

    return query.order_by(
        IncidentHistory.created_at.asc(), IncidentHistory.id.asc()
    ).all()


def has_history(db: Session, incident_id: int) -> bool:
    """Return True if any audit entry exists for the incident."""
    return (
        db.query(IncidentHistory.id)
        .filter(IncidentHistory.incident_id == incident_id)
        .first()
        is not None
    )


def backfill_initial_history(db: Session) -> int:
    """
    Give every incident that has no audit trail an opening INCIDENT_CREATED event.

    Incidents reported before this table existed cannot have their real history
    reconstructed, and inventing one would make the trail a liar.  The single
    thing we do know is that the incident was created, when, and by whom (from
    `reported_by`), so that is all this writes.  The entry is stamped with the
    incident's own created_at, so it sorts ahead of everything that follows.

    Where the reporter is unknown the event is attributed to SYSTEM rather than
    guessed at.

    Idempotent: incidents that already have history are skipped, so re-running
    this never duplicates an event.  Mirrors step 5 of
    migrations/add_incident_history.sql, which does the same for the live
    database; this function is the path used by tests and by any incident that
    somehow reaches the app without a trail.

    Returns the number of events written.
    """
    from app.models.incident import Incident

    incidents_without_history = (
        db.query(Incident)
        .filter(
            ~db.query(IncidentHistory.id)
            .filter(IncidentHistory.incident_id == Incident.id)
            .exists()
        )
        .all()
    )

    written = 0
    for incident in incidents_without_history:
        reporter = (
            db.query(User).filter(User.id == incident.reported_by).first()
            if incident.reported_by
            else None
        )
        actor_id, actor_role, actor_name = _actor_fields(reporter)

        entry = IncidentHistory(
            incident_id=incident.id,
            actor_id=actor_id,
            actor_role=actor_role,
            actor_name=actor_name,
            action=HistoryAction.INCIDENT_CREATED,
            old_value=None,
            new_value="REPORTED",
            description=(
                f"Incident reported by {actor_name}."
                if reporter is not None
                else "Incident reported. Reporter unknown — this entry was "
                "reconstructed when the audit trail was introduced."
            ),
            created_at=incident.created_at,
        )
        db.add(entry)
        written += 1

    if written:
        db.flush()
    return written
