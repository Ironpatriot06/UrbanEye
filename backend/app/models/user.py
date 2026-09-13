"""
User ORM model.

Roles
-----
USER  — citizen who reports incidents
ADMIN — system administrator with full access
AGENT — field agent who handles assigned incidents

The role is an AUTHORIZATION property and is never accepted from a client.
Registration and Google sign-in both force UserRole.USER; only an
authenticated ADMIN can change it (see app/api/admin_users.py).

Credentials
-----------
A user may hold a password, a linked Google identity, or both:

  password_hash      bcrypt digest.  NULL for an account created purely by
                     Google sign-in — such an account has no password to
                     verify, and `authenticate_user` refuses it rather than
                     comparing against an empty string.
  google_subject_id  Google's stable, opaque `sub` claim.  NULL until the
                     account signs in with Google at least once.  UNIQUE, so
                     one Google identity can never back two accounts.

`auth_methods` derives the display label from those two columns, which is why
there is no separate auth_provider column to drift out of step with them.

is_active
---------
A deactivated account cannot log in and its existing tokens stop working —
`get_current_user` re-reads this column on every request.  Deactivation is
reversible and preserves the user's incident history, which deletion would not.

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

    # Nullable: a Google-only account never sets a password.  Every account
    # that does have one stores a bcrypt digest, never the plain text.
    password_hash: str = Column(String(255), nullable=True)

    #: Google's `sub` claim — stable for the lifetime of the Google account and
    #: unaffected by the user changing their Google email address.  Matching on
    #: this first is what stops an email change from silently forking or
    #: merging accounts.
    google_subject_id: str = Column(
        String(255), unique=True, nullable=True, index=True
    )

    role: str = Column(
        Enum(UserRole, name="userrole"),
        nullable=False,
        default=UserRole.USER,
    )

    is_active: bool = Column(Boolean, nullable=False, default=True)

    # Agent-specific fields (NULL / unused for USER and ADMIN)
    is_available: bool = Column(Boolean, nullable=False, default=True)
    last_assigned_at: datetime = Column(DateTime(timezone=True), nullable=True)

    last_login_at: datetime = Column(DateTime(timezone=True), nullable=True)

    created_at: datetime = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------

    @property
    def auth_methods(self) -> list[str]:
        """
        Which credentials can sign this account in.

        Derived rather than stored so it cannot contradict the columns it
        describes.  Returns [] only for an account that predates both — which
        the schema does not allow to be created.
        """
        methods: list[str] = []
        if self.password_hash:
            methods.append("EMAIL")
        if self.google_subject_id:
            methods.append("GOOGLE")
        return methods
