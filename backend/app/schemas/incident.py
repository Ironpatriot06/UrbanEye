"""
Pydantic schemas for the Incident resource.

Separation of concerns
-----------------------
  IncidentCreate  — validated input for POST /incidents (JSON fields only;
                    images are uploaded separately as multipart form fields)
  IncidentStatusUpdate — validated input for PATCH /incidents/{id}/status
  IncidentRead    — response shape (what the API sends back to clients)

Validation rules enforced here
-------------------------------
- title:       1–200 characters, required
- description: max 2 000 characters, optional
- latitude:    –90 to +90
- longitude:   –180 to +180
- category, source, status, severity:  must be a valid enum member
- priority:    1–10 (1 = most urgent)
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.incident import (
    IncidentCategory,
    IncidentSeverity,
    IncidentSource,
    IncidentStatus,
)


# ---------------------------------------------------------------------------
# Shared field definitions
# ---------------------------------------------------------------------------

class _IncidentBase(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Short, human-readable title for the incident.",
        examples=["Pothole on MG Road near bus stop"],
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Additional detail about the incident.",
    )
    category: IncidentCategory = Field(
        IncidentCategory.OTHER,
        description="Type of civic incident.",
    )
    source: IncidentSource = Field(
        IncidentSource.CITIZEN,
        description="How the incident was reported.",
    )
    severity: IncidentSeverity = Field(
        IncidentSeverity.MEDIUM,
        description="Estimated severity level.",
    )
    priority: int = Field(
        5,
        ge=1,
        le=10,
        description="Priority score where 1 is most urgent and 10 is least urgent.",
    )
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="WGS-84 latitude of the incident location.",
        examples=[12.9716],
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="WGS-84 longitude of the incident location.",
        examples=[77.5946],
    )


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class IncidentCreate(_IncidentBase):
    """
    Request body for creating a new incident.

    Note: image files are accepted as multipart/form-data alongside these
    JSON fields.  The `reported_by` field is NOT accepted from the client —
    it is determined from the authenticated user's JWT token.
    """

    model_config = {"json_schema_extra": {
        "example": {
            "title": "Large pothole near City Bus Stop 14",
            "description": "Approximately 30 cm deep pothole causing vehicle damage.",
            "category": "POTHOLE",
            "source": "CITIZEN",
            "severity": "HIGH",
            "priority": 3,
            "latitude": 12.9716,
            "longitude": 77.5946,
        }
    }}


class IncidentStatusUpdate(BaseModel):
    """Request body for updating incident status."""

    status: IncidentStatus = Field(
        ...,
        description="New status to apply to the incident.",
    )

    model_config = {"json_schema_extra": {
        "example": {"status": "IN_PROGRESS"}
    }}


class AgentAssignUpdate(BaseModel):
    """Request body for admin to manually assign/reassign an agent."""

    agent_id: Optional[int] = Field(
        None,
        description="ID of the agent to assign. Set to null to unassign.",
    )


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class IncidentRead(_IncidentBase):
    """
    Full incident representation returned by the API.

    Note: `location` (the raw PostGIS WKB hex) is intentionally excluded —
    clients receive clean latitude/longitude floats instead.
    """

    id: int
    status: IncidentStatus
    reported_by: Optional[int] = None
    reported_by_name: Optional[str] = None
    assigned_agent_id: Optional[int] = None
    assigned_agent_name: Optional[str] = None
    image_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
