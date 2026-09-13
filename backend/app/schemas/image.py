"""
Pydantic schemas for IncidentImage.
"""

from datetime import datetime

from pydantic import BaseModel

from app.models.image import ImageKind


class ImageRead(BaseModel):
    """Image metadata returned by the API (no raw bytes)."""

    id: int
    incident_id: int
    filename: str
    content_type: str

    #: REPORT (what the citizen photographed) or RESOLUTION (the agent's proof
    #: that the work was done).  Lets the UI label the two apart without a
    #: second request.
    kind: ImageKind
    created_at: datetime

    model_config = {"from_attributes": True}
