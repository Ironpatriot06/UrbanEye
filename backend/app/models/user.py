"""
User ORM model.

Roles
-----
USER  — citizen who reports incidents
ADMIN — system administrator with full access
AGENT — field agent who handles assigned incidents

Agent-specific fields
---------------------
is_available    : whether the agent can receive new incident assignments.
last_assigned_at: timestamp of the most recent assignment, used by the
                  longest-idle-first queue algorithm.  NULL means the agent
                  has never been assigned (highest priority in the queue).
"""

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Integer,
    String,
    func,
)

from app.db.base import Base


class UserRole(str, enum.Enum):
    USER = "USER"
    ADMIN = "ADMIN"
    AGENT = "AGENT"


class User(Base):
    """Application user — citizen, admin, or field agent."""

    __tablename__ = "users"

    id: int = Column(Integer, primary_key=True, index=True)
    name: str = Column(String(100), nullable=False)
    email: str = Column(String(254), unique=True, nullable=False, index=True)
    password_hash: str = Column(String(255), nullable=False)

    role: str = Column(
        Enum(UserRole, name="userrole"),
        nullable=False,
        default=UserRole.USER,
    )

    # Agent-specific fields (NULL / unused for USER and ADMIN)
    is_available: bool = Column(Boolean, nullable=False, default=True)
    last_assigned_at: datetime = Column(DateTime(timezone=True), nullable=True)

    created_at: datetime = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
