"""
Pydantic schemas for IncidentImage.
"""

from datetime import datetime

from pydantic import BaseModel


class ImageRead(BaseModel):
    """Image metadata returned by the API (no raw bytes)."""

    id: int
    incident_id: int
    filename: str
    content_type: str
    created_at: datetime

    model_config = {"from_attributes": True}
