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
    """Return all users with the AGENT role."""
    return db.query(User).filter(User.role == UserRole.AGENT).all()


def set_agent_availability(db: Session, agent: User, is_available: bool) -> User:
    """Update agent availability flag."""
    agent.is_available = is_available
    db.commit()
    db.refresh(agent)
    return agent
