"""
User audit service — the only place account/authorization audit rows are written.

Transaction contract
--------------------
`record()` stages and flushes the row but never commits, so the audit entry
lives or dies with the operation that caused it.  Callers commit once, after
both the change and its audit entry are staged.  This mirrors
history_service.record() deliberately: two audit trails that behave differently
under failure would be worse than either.

Trust model
-----------
The actor is the `User` the router resolved from the JWT, or None for a
self-service action no admin performed.  Nothing here reads an actor, a role or
a timestamp from client input.
"""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.history import ActorRole
from app.models.user import User, UserRole
from app.models.user_audit import UserAuditAction, UserAuditLog

#: Label used when no person performed the action (self-service registration).
SYSTEM_ACTOR_NAME = "System"


def _actor_fields(actor: Optional[User]):
    """Resolve (id, role, email, name) for the acting user, or the system."""
    if actor is None:
        return None, ActorRole.SYSTEM, None, SYSTEM_ACTOR_NAME
    role_name = actor.role.value if isinstance(actor.role, UserRole) else str(actor.role)
    return actor.id, ActorRole(role_name), actor.email, actor.name


def record(
    db: Session,
    action: UserAuditAction,
    *,
    target: Optional[User] = None,
    actor: Optional[User] = None,
    old_value: Optional[str] = None,
    new_value: Optional[str] = None,
    description: Optional[str] = None,
) -> UserAuditLog:
    """Append one account audit entry.  Staged and flushed, never committed."""
    actor_id, actor_role, actor_email, actor_name = _actor_fields(actor)

    entry = UserAuditLog(
        action=action,
        actor_id=actor_id,
        actor_role=actor_role,
        actor_email=actor_email,
        actor_name=actor_name,
        target_user_id=target.id if target is not None else None,
        target_email=target.email if target is not None else None,
        target_name=target.name if target is not None else None,
        old_value=old_value,
        new_value=new_value,
        description=description,
        # Server-derived wall clock, passed explicitly so several events in one
        # transaction order by when they happened rather than by when the
        # transaction opened.
        created_at=datetime.now(timezone.utc),
    )
    db.add(entry)
    db.flush()
    return entry


def get_recent(
    db: Session,
    *,
    limit: int = 100,
    target_user_id: Optional[int] = None,
) -> List[UserAuditLog]:
    """
    Return account audit events, newest first.

    Newest-first is the opposite of the incident timeline on purpose: an
    incident is read as a story from the start, whereas an admin reviewing
    account changes wants the most recent ones.
    """
    query = db.query(UserAuditLog)
    if target_user_id is not None:
        query = query.filter(UserAuditLog.target_user_id == target_user_id)
    return (
        query.order_by(UserAuditLog.created_at.desc(), UserAuditLog.id.desc())
        .limit(max(1, min(limit, 500)))
        .all()
    )
