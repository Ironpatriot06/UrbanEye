"""
Pydantic schemas for the incident audit trail.

There is deliberately no create/update schema here.  History is written only by
the service layer from server-derived facts; the API surface is read-only, so a
request shape for writing one would be a shape for forging one.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.models.history import ActorRole, HistoryAction


class HistoryEventRead(BaseModel):
    """One audit entry as returned by GET /incidents/{id}/history."""

    id: int
    incident_id: int

    #: Display name captured when the event was written.  "System" for events
    #: no person performed.
    actor_name: Optional[str] = None
    actor_id: Optional[int] = None
    actor_role: ActorRole

    action: HistoryAction
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    description: Optional[str] = None

    created_at: datetime

    model_config = {"from_attributes": True}
