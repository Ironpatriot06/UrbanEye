"""
Incident ORM model.

PostGIS notes
-------------
GeoAlchemy2 maps the `location` column to a PostGIS GEOGRAPHY(POINT, 4326)
type.  GEOGRAPHY (as opposed to GEOMETRY) uses metres for all distance
calculations and handles the antimeridian correctly — ideal for city-scale
incident data.

We store latitude and longitude as plain FLOAT columns too because:
  1. API responses and JSON serialisation are trivial.
  2. Client-side filtering / ordering by coordinate is straightforward.
  3. The PostGIS POINT is the authoritative spatial index for proximity queries.

When creating a row the service layer constructs the WKT string
"SRID=4326;POINT(<lon> <lat>)" so PostGIS receives a fully-typed value.

Ownership / assignment
-----------------------
reported_by       — FK to users.id (nullable for backward compat with Phase 1 rows)
assigned_agent_id — FK to users.id (nullable; set by the assignment queue)

Priority / Severity
--------------------
priority_level  — System-determined P1/P2/P3/P4 (maps to CRITICAL/HIGH/MEDIUM/LOW).
                  Citizens cannot set this directly; the service layer sets it on
                  creation.  Future AI modules will update this field.
severity        — Derived from priority_level, kept for backward compatibility.
                  Always set by the service layer, never from user input.

SLA
---
sla_hours    — Target resolution time in hours (auto-set from priority+category).
sla_deadline — Absolute deadline timestamp (created_at + sla_hours).
sla_status   — ON_TRACK / AT_RISK / BREACHED — computed dynamically but also
               cached here for query filtering.
"""

import enum
from datetime import datetime

from geoalchemy2 import Geography
from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)

from app.db.base import Base


# ---------------------------------------------------------------------------
# Enum definitions
# ---------------------------------------------------------------------------

class IncidentSource(str, enum.Enum):
    CITIZEN = "CITIZEN"
    CAMERA = "CAMERA"
    IOT = "IOT"


class IncidentCategory(str, enum.Enum):
    POTHOLE = "POTHOLE"
    FLOOD = "FLOOD"
    FIRE_HAZARD = "FIRE_HAZARD"
    GARBAGE = "GARBAGE"
    STREETLIGHT = "STREETLIGHT"
    OTHER = "OTHER"


class IncidentStatus(str, enum.Enum):
    REPORTED = "REPORTED"
    TRIAGED = "TRIAGED"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class IncidentSeverity(str, enum.Enum):
    """Kept for DB compatibility. Derived from IncidentPriority by the service layer."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentPriority(str, enum.Enum):
    """
    System-determined incident priority.

    P1 = CRITICAL — life-safety, major infrastructure
    P2 = HIGH     — significant disruption, needs fast response
    P3 = MEDIUM   — standard civic issue (default for new reports)
    P4 = LOW      — minor nuisance, non-urgent

    The AI/analysis module will update this field in future phases.
    Citizens can provide a hint via category, but cannot set priority directly.
    """
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class SLAStatus(str, enum.Enum):
    ON_TRACK = "ON_TRACK"
    AT_RISK = "AT_RISK"
    BREACHED = "BREACHED"


# ---------------------------------------------------------------------------
# Status transition rules (enforced by the service layer)
# ---------------------------------------------------------------------------

#: Mapping of current_status -> set of valid next statuses.
#: Admin may use any valid transition from this map.
#: Agent may only use AGENT_TRANSITIONS (a subset).
VALID_TRANSITIONS: dict = {
    IncidentStatus.REPORTED:     {IncidentStatus.TRIAGED, IncidentStatus.ASSIGNED, IncidentStatus.CLOSED},
    IncidentStatus.TRIAGED:      {IncidentStatus.ASSIGNED, IncidentStatus.CLOSED},
    IncidentStatus.ASSIGNED:     {IncidentStatus.IN_PROGRESS, IncidentStatus.TRIAGED, IncidentStatus.CLOSED},
    IncidentStatus.IN_PROGRESS:  {IncidentStatus.RESOLVED, IncidentStatus.ASSIGNED},
    IncidentStatus.RESOLVED:     {IncidentStatus.CLOSED, IncidentStatus.IN_PROGRESS},
    IncidentStatus.CLOSED:       set(),  # terminal state
}

#: Transitions an AGENT is permitted to make (strict subset of VALID_TRANSITIONS).
AGENT_TRANSITIONS: dict = {
    IncidentStatus.ASSIGNED:    {IncidentStatus.IN_PROGRESS},
    IncidentStatus.IN_PROGRESS: {IncidentStatus.RESOLVED},
}

#: Statuses from which no further transition is possible.
TERMINAL_STATUSES: frozenset = frozenset({IncidentStatus.CLOSED})

#: Statuses at which the SLA clock stops.  Once an incident reaches one of
#: these the stored sla_status is frozen instead of being recomputed against
#: `now()`, so an incident resolved inside its window is not retroactively
#: reported as BREACHED.
SLA_FREEZE_STATUSES: frozenset = frozenset(
    {IncidentStatus.RESOLVED, IncidentStatus.CLOSED}
)

#: Map IncidentPriority -> IncidentSeverity (for backward-compat severity column)
PRIORITY_TO_SEVERITY: dict = {
    IncidentPriority.P1: IncidentSeverity.CRITICAL,
    IncidentPriority.P2: IncidentSeverity.HIGH,
    IncidentPriority.P3: IncidentSeverity.MEDIUM,
    IncidentPriority.P4: IncidentSeverity.LOW,
}

#: Inverse of PRIORITY_TO_SEVERITY.  Used only to derive a priority for legacy
#: rows created before the priority_level column existed (see the
#: add_priority_sla_columns.sql migration, which performs the same mapping).
SEVERITY_TO_PRIORITY: dict = {
    IncidentSeverity.CRITICAL: IncidentPriority.P1,
    IncidentSeverity.HIGH:     IncidentPriority.P2,
    IncidentSeverity.MEDIUM:   IncidentPriority.P3,
    IncidentSeverity.LOW:      IncidentPriority.P4,
}

#: Map IncidentPriority -> the legacy integer `priority` column (1 = most urgent)
PRIORITY_TO_INT: dict = {
    IncidentPriority.P1: 1,
    IncidentPriority.P2: 2,
    IncidentPriority.P3: 3,
    IncidentPriority.P4: 4,
}

#: Human-readable priority labels shown in the UI.
PRIORITY_LABELS: dict = {
    IncidentPriority.P1: "Critical",
    IncidentPriority.P2: "High",
    IncidentPriority.P3: "Medium",
    IncidentPriority.P4: "Low",
}

#: An incident is AT_RISK once less than this fraction of its SLA window remains.
SLA_AT_RISK_RATIO: float = 0.2

#: Default SLA hours by priority level (configurable; future AI can override per incident)
DEFAULT_SLA_HOURS: dict = {
    IncidentPriority.P1: 4,
    IncidentPriority.P2: 24,
    IncidentPriority.P3: 72,
    IncidentPriority.P4: 168,  # 1 week
}

#: Category-based priority hint used on creation when no other signal is available
CATEGORY_DEFAULT_PRIORITY: dict = {
    IncidentCategory.FIRE_HAZARD: IncidentPriority.P1,
    IncidentCategory.FLOOD:       IncidentPriority.P2,
    IncidentCategory.POTHOLE:     IncidentPriority.P3,
    IncidentCategory.STREETLIGHT: IncidentPriority.P3,
    IncidentCategory.GARBAGE:     IncidentPriority.P4,
    IncidentCategory.OTHER:       IncidentPriority.P3,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Incident(Base):
    """
    Represents a single civic incident in the system.

    Spatial field
    ~~~~~~~~~~~~~
    `location` is a PostGIS GEOGRAPHY(POINT, 4326) column.  It is populated
    automatically by the service layer from the supplied latitude/longitude
    pair.
    """

    __tablename__ = "incidents"

    id: int = Column(Integer, primary_key=True, index=True)

    # Human-readable fields
    title: str = Column(String(200), nullable=False)
    description: str = Column(Text, nullable=True)

    # Classification
    category: str = Column(
        Enum(IncidentCategory, name="incidentcategory"),
        nullable=False,
        default=IncidentCategory.OTHER,
    )
    source: str = Column(
        Enum(IncidentSource, name="incidentsource"),
        nullable=False,
        default=IncidentSource.CITIZEN,
    )
    status: str = Column(
        Enum(IncidentStatus, name="incidentstatus"),
        nullable=False,
        default=IncidentStatus.REPORTED,
    )

    # Priority — system-determined (P1=CRITICAL ... P4=LOW)
    priority_level: str = Column(
        Enum(IncidentPriority, name="incidentpriority"),
        nullable=False,
        default=IncidentPriority.P3,
    )

    # Severity — derived from priority_level, kept for DB backward compat
    severity: str = Column(
        Enum(IncidentSeverity, name="incidentseverity"),
        nullable=False,
        default=IncidentSeverity.MEDIUM,
    )

    # Legacy integer priority (1-10). Mapped from priority_level by service layer.
    priority: int = Column(Integer, nullable=False, default=5)

    # SLA fields
    sla_hours: int = Column(Integer, nullable=True)
    sla_deadline: datetime = Column(DateTime(timezone=True), nullable=True)
    sla_status: str = Column(String(20), nullable=True)

    # Spatial: plain coordinates (for simple queries / serialisation)
    latitude: float = Column(Float, nullable=False)
    longitude: float = Column(Float, nullable=False)

    # Spatial: PostGIS GEOGRAPHY for proximity / GIS queries
    location = Column(
        Geography(geometry_type="POINT", srid=4326),
        nullable=False,
    )

    # Ownership — nullable so Phase 1 rows without auth remain valid
    reported_by: int = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    assigned_agent_id: int = Column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Timestamps — set by the database server for reliability
    created_at: datetime = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: datetime = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
