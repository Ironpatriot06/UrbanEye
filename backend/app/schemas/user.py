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
    """Request body for login."""

    email: EmailStr
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
    is_available: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AgentAvailabilityUpdate(BaseModel):
    """Request body for toggling agent availability."""

    is_available: bool
