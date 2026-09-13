"""
IncidentHistory ORM model — the incident audit trail.

What this is
------------
An append-only log of everything that happened to an incident and who did it.
One row per real operation: a status transition, an assignment, a priority or
SLA change, an image upload, an admin override.

Immutability
------------
Audit rows are written by the service layer and never changed again.  Three
things enforce that, deliberately overlapping:

  1. No API endpoint updates or deletes history.  The router exposes GET only.
  2. The SQLAlchemy mapper events at the bottom of this module raise on any
     UPDATE or DELETE of an IncidentHistory, so a mistake inside the backend
     fails loudly rather than silently rewriting the record.
  3. The SQL migration installs a database trigger that rejects UPDATE and
     DELETE outright, which also covers anything reaching the table outside
     this application.

Actor
-----
actor_id   — FK to users.id.  NULL when the system acted on its own (the
             assignment queue picking an agent) or when the original actor of a
             pre-existing row cannot be determined.
actor_role — USER / AGENT / ADMIN / SYSTEM.  Stored rather than joined so the
             trail still reads correctly if the account is later deleted (the
             FK is ON DELETE SET NULL) or the person changes role.

Both are derived server-side from the authenticated request.  Neither is ever
accepted from a client.

old_value / new_value
---------------------
Short, human-comparable strings ("REPORTED" -> "TRIAGED", "P3" -> "P1",
"24 hours" -> "8 hours") rather than typed columns, because the trail spans
statuses, priorities, SLA windows and agent names.  The `action` column is what
code branches on; these two are for display and diffing.

Timestamps
----------
created_at defaults to clock_timestamp(), not now(): now() is fixed at
transaction start in PostgreSQL, so several events written in one transaction
would share a timestamp and lose their order.  The service layer also passes an
explicit server-derived timestamp.  Readers order by (created_at, id) so the
primary key breaks any remaining tie in true insertion order.
"""

import enum
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    event,
    func,
)

from app.db.base import Base


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

class HistoryAction(str, enum.Enum):
    """
    The closed set of auditable operations.

    A new action type means a new member here and a matching value in the
    `incidenthistoryaction` database enum — arbitrary strings are not accepted,
    so the trail stays queryable and the UI can label every event.
    """

    INCIDENT_CREATED = "INCIDENT_CREATED"
    INCIDENT_TRIAGED = "INCIDENT_TRIAGED"
    PRIORITY_CHANGED = "PRIORITY_CHANGED"
    SLA_CREATED = "SLA_CREATED"
    SLA_UPDATED = "SLA_UPDATED"
    AGENT_ASSIGNED = "AGENT_ASSIGNED"
    AGENT_REASSIGNED = "AGENT_REASSIGNED"
    AGENT_UNASSIGNED = "AGENT_UNASSIGNED"
    STATUS_CHANGED = "STATUS_CHANGED"
    AGENT_AVAILABILITY_CHANGED = "AGENT_AVAILABILITY_CHANGED"
    IMAGE_ADDED = "IMAGE_ADDED"
    INCIDENT_RESOLVED = "INCIDENT_RESOLVED"
    INCIDENT_CLOSED = "INCIDENT_CLOSED"
    INCIDENT_REOPENED = "INCIDENT_REOPENED"
    ADMIN_OVERRIDE = "ADMIN_OVERRIDE"


class ActorRole(str, enum.Enum):
    """
    Who performed an action.

    Mirrors UserRole plus SYSTEM, which covers work no person requested — the
    automatic assignment queue, the SLA set on creation, and the backfilled
    opening event of incidents that predate this table.
    """

    USER = "USER"
    ADMIN = "ADMIN"
    AGENT = "AGENT"
    SYSTEM = "SYSTEM"


#: Status transitions that deserve a more specific action than STATUS_CHANGED.
#: Every status event still carries old_value/new_value, so nothing is lost by
#: labelling these milestones; the timeline just reads as operations rather
#: than as a list of identical "status changed" lines.
STATUS_MILESTONE_ACTIONS: dict = {
    "TRIAGED": HistoryAction.INCIDENT_TRIAGED,
    "RESOLVED": HistoryAction.INCIDENT_RESOLVED,
    "CLOSED": HistoryAction.INCIDENT_CLOSED,
}

#: Actions a citizen should not see on their own incident.  These are internal
#: operations about staffing, not about the incident's progress, and the
#: reporter has no use for them.  Agents and admins see the full trail.
ACTIONS_HIDDEN_FROM_REPORTER: frozenset = frozenset(
    {
        HistoryAction.AGENT_AVAILABILITY_CHANGED,
        HistoryAction.ADMIN_OVERRIDE,
    }
)


class HistoryImmutableError(RuntimeError):
    """Raised when backend code attempts to modify or delete an audit record."""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class IncidentHistory(Base):
    """A single immutable audit entry for one incident."""

    __tablename__ = "incident_history"

    id: int = Column(Integer, primary_key=True, index=True)

    incident_id: int = Column(
        Integer,
        ForeignKey("incidents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # NULL for system-generated events, and for actors that can no longer be
    # resolved.  actor_role always says what kind of actor it was.
    actor_id: int = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    actor_role: str = Column(
        Enum(ActorRole, name="actorrole"),
        nullable=False,
        default=ActorRole.SYSTEM,
    )
    #: Snapshot of the actor's name at the time of the action, so the trail
    #: still names them after an account is deleted.
    actor_name: str = Column(String(100), nullable=True)

    action: str = Column(
        Enum(HistoryAction, name="incidenthistoryaction"),
        nullable=False,
        index=True,
    )

    old_value: str = Column(String(255), nullable=True)
    new_value: str = Column(String(255), nullable=True)

    #: Human-readable sentence describing what happened, composed server-side.
    description: str = Column(Text, nullable=True)

    created_at: datetime = Column(
        DateTime(timezone=True),
        server_default=func.clock_timestamp(),
        nullable=False,
        index=True,
    )


# ---------------------------------------------------------------------------
# Immutability guards
# ---------------------------------------------------------------------------
#
# These fire during flush, before any SQL is emitted.  They exist so that a
# future refactor cannot quietly rewrite the audit trail: editing history is a
# bug, and it should surface as one.

@event.listens_for(IncidentHistory, "before_update", propagate=True)
def _forbid_history_update(mapper, connection, target) -> None:  # noqa: ARG001
    raise HistoryImmutableError(
        f"Incident history is append-only; entry id={target.id} cannot be modified."
    )


@event.listens_for(IncidentHistory, "before_delete", propagate=True)
def _forbid_history_delete(mapper, connection, target) -> None:  # noqa: ARG001
    raise HistoryImmutableError(
        f"Incident history is append-only; entry id={target.id} cannot be deleted."
    )
