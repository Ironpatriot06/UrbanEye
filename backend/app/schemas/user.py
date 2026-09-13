"""
Pydantic schemas for User and authentication.
"""

import re
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.models.user import UserRole
from app.models.user_audit import UserAuditAction

#: Minimum password strength.  Deliberately modest — length plus a letter and a
#: digit — because a long list of character-class rules pushes people towards
#: predictable substitutions without buying much.  Length is the part that
#: matters, and bcrypt covers the rest.
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128


def validate_password_strength(value: str) -> str:
    """Reject passwords that fail the minimum requirements.  Shared by every schema."""
    if len(value) < PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"Password must be at least {PASSWORD_MIN_LENGTH} characters long."
        )
    if len(value) > PASSWORD_MAX_LENGTH:
        raise ValueError(
            f"Password must be at most {PASSWORD_MAX_LENGTH} characters long."
        )
    if not re.search(r"[A-Za-z]", value):
        raise ValueError("Password must contain at least one letter.")
    if not re.search(r"\d", value):
        raise ValueError("Password must contain at least one number.")
    return value


class UserCreate(BaseModel):
    """
    Request body for registering a new account.

    There is deliberately NO role field.  The role is an authorization property
    decided by the server: registration always produces a USER, and a body
    containing `"role": "ADMIN"` is ignored outright — pydantic drops unknown
    keys, and nothing downstream reads one.  Only an authenticated ADMIN can
    change a role, through PATCH /api/v1/admin/users/{id}/role.
    """

    name: str = Field(..., min_length=1, max_length=100, description="Full name.")
    email: EmailStr = Field(..., description="Email address (used as login).")
    password: str = Field(
        ...,
        min_length=PASSWORD_MIN_LENGTH,
        max_length=PASSWORD_MAX_LENGTH,
        description="Password — at least 8 characters, with a letter and a number.",
    )

    @field_validator("password")
    @classmethod
    def _check_password(cls, value: str) -> str:
        return validate_password_strength(value)

    model_config = {"json_schema_extra": {
        "example": {
            "name": "Riya Sharma",
            "email": "riya@example.com",
            "password": "SecurePass1",
        }
    }}


class UserLogin(BaseModel):
    """Request body for login.

    Uses plain str (not EmailStr) so that demo accounts with .local TLDs
    (which email-validator rejects as RFC-reserved) can still log in.
    """

    email: str = Field(..., description="Email address.")
    password: str

    model_config = {"json_schema_extra": {
        "example": {"email": "riya@example.com", "password": "SecurePass1"}
    }}


class TokenResponse(BaseModel):
    """Response returned after a successful login."""

    access_token: str
    token_type: str = "bearer"
    role: UserRole
    user_id: int
    name: str


class UserRead(BaseModel):
    """Public user representation (no password hash)."""

    id: int
    name: str
    email: str
    role: UserRole

    # Agent fields.  `is_available` is persistent state: it is changed only by
    # the agent themselves or by an admin — never as a side effect of login.
    is_available: bool
    last_assigned_at: Optional[datetime] = None

    # Number of incidents assigned to this agent that are not yet RESOLVED or
    # CLOSED.  Lets the admin UI distinguish "available" from "available but
    # already busy".  Populated only on the admin agent listing.
    active_incident_count: int = 0

    created_at: datetime

    model_config = {"from_attributes": True}


class AgentAvailabilityUpdate(BaseModel):
    """
    Request body for setting agent availability.

    Used both by an agent setting their own status and by an admin overriding
    another agent's status.  The value is persisted, so it survives logout and
    subsequent logins.
    """

    is_available: bool


class RegisterResponse(UserRead):
    """
    Response to a successful registration.

    A superset of UserRead: the new account's public fields plus an immediately
    usable session, so the browser can go straight to the citizen dashboard
    instead of bouncing the user back to a login form to retype what they just
    typed.  The role in it is always USER — it is read back from the created
    row, not from anything the client sent.
    """

    access_token: str
    token_type: str = "bearer"
    user_id: int


class AdminUserRead(BaseModel):
    """
    One row of the admin user-management table.

    Carries no credential material of any kind: no password hash, no Google
    subject id, no tokens.  `auth_methods` says *how* the account can sign in
    ("EMAIL", "GOOGLE", or both) without exposing the credentials themselves.
    """

    id: int
    name: str
    email: str
    role: UserRole
    is_active: bool
    auth_methods: List[str]
    created_at: datetime
    last_login_at: Optional[datetime] = None

    # Only meaningful for AGENT rows; shown so an admin can see at a glance why
    # an agent is or is not receiving assignments.
    is_available: bool

    model_config = {"from_attributes": True}


class UserRoleUpdate(BaseModel):
    """
    Request body for changing another user's role.

    The target user is identified by the path, never by the body, and the actor
    is taken from the JWT — so the only thing a client can choose here is the
    new role, and only after passing the ADMIN guard.
    """

    role: UserRole = Field(..., description="The role to grant: USER, AGENT or ADMIN.")

    model_config = {"json_schema_extra": {"example": {"role": "AGENT"}}}


class UserStatusUpdate(BaseModel):
    """Request body for activating or deactivating an account."""

    is_active: bool = Field(
        ..., description="False deactivates the account and invalidates its tokens."
    )


class UserAuditRead(BaseModel):
    """One entry of the account audit trail, as served to an admin."""

    id: int
    action: UserAuditAction
    actor_name: Optional[str] = None
    actor_email: Optional[str] = None
    actor_role: str
    target_user_id: Optional[int] = None
    target_name: Optional[str] = None
    target_email: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}
