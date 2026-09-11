"""
Pydantic schemas for the Incident resource.

Separation of concerns
-----------------------
  IncidentCreate      — citizen input: title, description, category, source, lat/lon.
                        NO severity or priority_level — those are system-determined.
  IncidentStatusUpdate — validated input for PATCH /incidents/{id}/status
  SLAUpdate           — admin-only SLA override
  AgentAssignUpdate   — admin assign/unassign with optional override
  IncidentRead        — response shape (what the API sends back to clients)

Validation rules enforced here
-------------------------------
- title:       1-200 characters, required
- description: max 2000 characters, optional
- latitude:    -90 to +90
- longitude:   -180 to +180
- category, source:  must be a valid enum member

NOTE: Citizens do NOT set severity or priority_level.
      The service layer derives these from category on creation.
      Admins may update priority_level and SLA via dedicated endpoints.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.incident import (
    IncidentCategory,
    IncidentPriority,
    IncidentSeverity,
    IncidentSource,
    IncidentStatus,
    SLAStatus,
)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class IncidentCreate(BaseModel):
    """
    Request body for creating a new incident.

    Note: image files are accepted as multipart/form-data alongside these
    JSON fields.  The `reported_by` field is NOT accepted from the client —
    it is determined from the authenticated user's JWT token.

    priority_level and severity are intentionally excluded — the service layer
    determines these from the incident category and will later be overridden
    by the AI analysis module.
    """

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

    model_config = {"json_schema_extra": {
        "example": {
            "title": "Large pothole near City Bus Stop 14",
            "description": "Approximately 30 cm deep pothole causing vehicle damage.",
            "category": "POTHOLE",
            "source": "CITIZEN",
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
    override_availability: bool = Field(
        False,
        description=(
            "Admin override: if True, allows assigning an unavailable agent. "
            "A confirmation warning should be shown in the UI before sending this flag."
        ),
    )


class SLAUpdate(BaseModel):
    """Admin-only: override SLA hours for an incident."""

    sla_hours: Optional[int] = Field(
        None,
        ge=1,
        le=8760,  # max 1 year
        description="Target resolution time in hours. Set to null to clear.",
    )


class PriorityUpdate(BaseModel):
    """Admin-only: manually set incident priority (also used by future AI module)."""

    priority_level: IncidentPriority = Field(
        ...,
        description="New priority level (P1=CRITICAL, P2=HIGH, P3=MEDIUM, P4=LOW).",
    )


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class IncidentRead(BaseModel):
    """
    Full incident representation returned by the API.

    Note: `location` (the raw PostGIS WKB hex) is intentionally excluded —
    clients receive clean latitude/longitude floats instead.
    """

    id: int
    title: str
    description: Optional[str] = None
    category: IncidentCategory
    source: IncidentSource
    status: IncidentStatus

    # Priority / severity — assigned by the system, never by the reporter
    priority_level: IncidentPriority
    priority_label: str = ""    # "Critical" / "High" / "Medium" / "Low"
    severity: IncidentSeverity  # derived from priority_level
    priority: int               # legacy integer priority

    # SLA — visible to every role, editable only by an admin
    sla_hours: Optional[int] = None
    sla_deadline: Optional[datetime] = None
    sla_status: Optional[SLAStatus] = None

    # Location
    latitude: float
    longitude: float

    # Ownership
    reported_by: Optional[int] = None
    reported_by_name: Optional[str] = None
    assigned_agent_id: Optional[int] = None
    assigned_agent_name: Optional[str] = None

    # Images
    image_count: int = 0

    # Timestamps
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
