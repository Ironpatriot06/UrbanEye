"""
User service layer.

All database operations for users and authentication live here.
"""

from typing import Optional

from sqlalchemy.orm import Session

from app.core.security import hash_password, verify_password
from app.models.user import User, UserRole
from app.schemas.user import UserCreate


def create_user(db: Session, payload: UserCreate, role: UserRole = UserRole.USER) -> User:
    """Create and persist a new user.  Never call with plain-text password storage."""
    user = User(
        name=payload.name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role=role,
        is_available=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Return the User with the given email, or None."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Return the User with the given id, or None."""
    return db.query(User).filter(User.id == user_id).first()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Return the User if credentials are valid, else None.

    Timing-safe: we always run the hash comparison even if the email is not
    found, to avoid email-enumeration via response timing.
    """
    user = get_user_by_email(db, email)
    if user is None:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def get_all_agents(db: Session) -> list[User]:
    """Return all users with the AGENT role, with their current workload attached.

    `active_incident_count` is a transient attribute (not an ORM column) read by
    the UserRead schema so the admin dashboard can tell an idle available agent
    apart from a busy one.
    """
    from app.services.incident_service import count_open_incidents_for_agent

    agents = (
        db.query(User)
        .filter(User.role == UserRole.AGENT)
        .order_by(User.name)
        .all()
    )
    for agent in agents:
        agent.active_incident_count = count_open_incidents_for_agent(db, agent.id)
    return agents


def get_agent_by_id(db: Session, agent_id: int) -> Optional[User]:
    """Return the AGENT-role user with the given id, or None.

    Returns None for a non-agent user so callers cannot flip availability on an
    admin or citizen account.
    """
    return (
        db.query(User)
        .filter(User.id == agent_id, User.role == UserRole.AGENT)
        .first()
    )


def set_agent_availability(
    db: Session,
    agent: User,
    is_available: bool,
    actor: Optional[User] = None,
) -> User:
    """
    Persist an agent's availability flag.

    This is the ONLY place availability changes.  Authentication deliberately
    does not touch it: an agent who marked themselves unavailable stays
    unavailable across logout and the next login, until they or an admin change
    it explicitly.

    Audit trail
    -----------
    Availability belongs to the agent, not to any one incident, but the audit
    trail is per-incident — so the event is recorded against each incident
    still open on that agent's plate, which is exactly where it is
    operationally relevant: those are the incidents whose handling it affects.
    Finished incidents are left alone; their outcome cannot change now.

    `actor` distinguishes an agent setting their own status from an admin
    overriding it, which is the difference the admin audit view needs to show.
    A call that does not change the flag records nothing.
    """
    from app.models.history import HistoryAction
    from app.models.incident import SLA_FREEZE_STATUSES, Incident
    from app.services import history_service

    changed = agent.is_available != is_available
    agent.is_available = is_available

    if changed:
        open_incidents = (
            db.query(Incident)
            .filter(
                Incident.assigned_agent_id == agent.id,
                Incident.status.notin_(list(SLA_FREEZE_STATUSES)),
            )
            .all()
        )
        state = "Available" if is_available else "Unavailable"
        by_self = actor is not None and actor.id == agent.id
        who = "themselves" if by_self else (actor.name if actor else "the system")

        for incident in open_incidents:
            history_service.record(
                db,
                incident.id,
                HistoryAction.AGENT_AVAILABILITY_CHANGED,
                actor=actor,
                old_value="Available" if not is_available else "Unavailable",
                new_value=state,
                description=(
                    f"{agent.name}, the agent on this incident, was marked "
                    f"{state.lower()} by {who}."
                ),
            )

    db.commit()
    db.refresh(agent)
    return agent
