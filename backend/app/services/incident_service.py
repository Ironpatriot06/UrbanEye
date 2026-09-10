"""
Incident service layer.

All database interactions for incidents live here.  The API router calls
these functions rather than talking to the database directly.  This keeps
the router thin and makes the business logic easy to unit-test or swap.

PostGIS POINT construction
---------------------------
PostGIS expects coordinates in (longitude, latitude) order inside WKT.
We pass a WKT string with the SRID prefix so SQLAlchemy / GeoAlchemy2
stores a fully-typed GEOGRAPHY value rather than raw bytes.

    "SRID=4326;POINT(<longitude> <latitude>)"
"""

from typing import List, Optional

from sqlalchemy.orm import Session

from app.models.incident import Incident, IncidentStatus
from app.schemas.incident import IncidentCreate, IncidentStatusUpdate


def _make_point_wkt(latitude: float, longitude: float) -> str:
    """Return a WKT string for a PostGIS GEOGRAPHY POINT."""
    return f"SRID=4326;POINT({longitude} {latitude})"


def create_incident(db: Session, payload: IncidentCreate) -> Incident:
    """Persist a new incident and return the created row."""
    incident = Incident(
        title=payload.title,
        description=payload.description,
        category=payload.category,
        source=payload.source,
        severity=payload.severity,
        priority=payload.priority,
        status=IncidentStatus.REPORTED,
        latitude=payload.latitude,
        longitude=payload.longitude,
        location=_make_point_wkt(payload.latitude, payload.longitude),
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return incident


def get_incidents(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    status: Optional[IncidentStatus] = None,
    category: Optional[str] = None,
) -> List[Incident]:
    """
    Return a paginated list of incidents.

    Optional filters:
    - status: restrict to a particular workflow status
    - category: restrict to a category string (case-sensitive enum value)
    """
    query = db.query(Incident)
    if status is not None:
        query = query.filter(Incident.status == status)
    if category is not None:
        query = query.filter(Incident.category == category)
    return query.order_by(Incident.created_at.desc()).offset(skip).limit(limit).all()


def get_incident_by_id(db: Session, incident_id: int) -> Optional[Incident]:
    """Return a single incident by primary key, or None if not found."""
    return db.query(Incident).filter(Incident.id == incident_id).first()


def update_incident_status(
    db: Session,
    incident: Incident,
    payload: IncidentStatusUpdate,
) -> Incident:
    """Apply a status transition and persist."""
    incident.status = payload.status  # type: ignore[assignment]
    db.commit()
    db.refresh(incident)
    return incident
