"""
Pydantic schemas for User and authentication.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field

from app.models.user import UserRole


class UserCreate(BaseModel):
    """Request body for registering a new citizen account."""

    name: str = Field(..., min_length=1, max_length=100, description="Full name.")
    email: EmailStr = Field(..., description="Email address (used as login).")
    password: str = Field(
        ..., min_length=8, max_length=128, description="Password (min 8 chars)."
    )

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
