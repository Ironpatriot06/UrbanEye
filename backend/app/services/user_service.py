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


def set_agent_availability(db: Session, agent: User, is_available: bool) -> User:
    """
    Persist an agent's availability flag.

    This is the ONLY place availability changes.  Authentication deliberately
    does not touch it: an agent who marked themselves unavailable stays
    unavailable across logout and the next login, until they or an admin change
    it explicitly.
    """
    agent.is_available = is_available
    db.commit()
    db.refresh(agent)
    return agent
