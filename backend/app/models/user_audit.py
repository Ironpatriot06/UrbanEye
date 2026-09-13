"""
UserAuditLog ORM model — the account & authorization audit trail.

Why this is separate from incident_history
------------------------------------------
`incident_history` answers "what happened to incident N".  Every row there is
keyed to an incident, and the reader is an incident timeline.  An account being
created, promoted or deactivated belongs to no incident, so writing it there
would require either a fake incident_id or a nullable key that breaks the one
invariant that table relies on.  These are two different questions about two
different subjects, so they are two tables.

What lands here
---------------
USER_REGISTERED        — an account was created (email/password or Google).
USER_ROLE_CHANGED      — an admin changed someone's role.
USER_STATUS_CHANGED    — an admin deactivated or reactivated an account.
GOOGLE_ACCOUNT_LINKED  — a Google identity was attached to an existing account.

Deliberately NOT here: passwords, password hashes, Google tokens, ID tokens,
authorization codes, or the `sub` claim.  The trail records that an identity
was linked, not the credential that proved it.

Actor and target
----------------
Both the id and an email/name snapshot are stored.  The ids are ON DELETE SET
NULL, so the trail still reads correctly after an account is removed — which is
exactly when an audit log matters most.  `actor_id` is NULL when no admin was
involved: self-service registration is attributed to SYSTEM.

Immutability
------------
Same three overlapping guarantees as incident_history: no write endpoint, ORM
mapper events that raise on UPDATE/DELETE, and a database trigger installed by
the migration.
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
from app.models.history import ActorRole  # USER / ADMIN / AGENT / SYSTEM


class UserAuditAction(str, enum.Enum):
    """The closed set of auditable account operations."""

    USER_REGISTERED = "USER_REGISTERED"
    USER_ROLE_CHANGED = "USER_ROLE_CHANGED"
    USER_STATUS_CHANGED = "USER_STATUS_CHANGED"
    GOOGLE_ACCOUNT_LINKED = "GOOGLE_ACCOUNT_LINKED"


class UserAuditImmutableError(RuntimeError):
    """Raised when backend code attempts to modify or delete an audit record."""


class UserAuditLog(Base):
    """A single immutable audit entry about an account or its authorization."""

    __tablename__ = "user_audit_log"

    id: int = Column(Integer, primary_key=True, index=True)

    action: str = Column(
        Enum(UserAuditAction, name="userauditaction"),
        nullable=False,
        index=True,
    )

    # Who did it.  NULL actor_id + SYSTEM role = no person requested this.
    actor_id: int = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    actor_role: str = Column(
        Enum(ActorRole, name="actorrole"),
        nullable=False,
        default=ActorRole.SYSTEM,
    )
    actor_email: str = Column(String(254), nullable=True)
    actor_name: str = Column(String(100), nullable=True)

    # Whom it was done to.
    target_user_id: int = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    target_email: str = Column(String(254), nullable=True)
    target_name: str = Column(String(100), nullable=True)

    old_value: str = Column(String(255), nullable=True)
    new_value: str = Column(String(255), nullable=True)

    #: Human-readable sentence composed server-side.
    description: str = Column(Text, nullable=True)

    created_at: datetime = Column(
        DateTime(timezone=True),
        server_default=func.clock_timestamp(),
        nullable=False,
        index=True,
    )


# ---------------------------------------------------------------------------
# Immutability guards — fire during flush, before any SQL is emitted.
# ---------------------------------------------------------------------------

@event.listens_for(UserAuditLog, "before_update", propagate=True)
def _forbid_user_audit_update(mapper, connection, target) -> None:  # noqa: ARG001
    raise UserAuditImmutableError(
        f"User audit log is append-only; entry id={target.id} cannot be modified."
    )


@event.listens_for(UserAuditLog, "before_delete", propagate=True)
def _forbid_user_audit_delete(mapper, connection, target) -> None:  # noqa: ARG001
    raise UserAuditImmutableError(
        f"User audit log is append-only; entry id={target.id} cannot be deleted."
    )
