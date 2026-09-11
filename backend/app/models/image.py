"""
IncidentImage ORM model.

Storage strategy
----------------
Images are stored as raw bytes (LargeBinary / PostgreSQL BYTEA column) for
this academic milestone.  The service layer validates file type and size
before persisting.  Future phases could move to S3/MinIO without changing
the API interface.

Size limit: 5 MB per image (enforced in the service layer).
Accepted MIME types: image/jpeg, image/png, image/webp
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, LargeBinary, String, func

from app.db.base import Base


class IncidentImage(Base):
    """A single image attached to an incident."""

    __tablename__ = "incident_images"

    id: int = Column(Integer, primary_key=True, index=True)

    incident_id: int = Column(
        Integer, ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    filename: str = Column(String(255), nullable=False)
    content_type: str = Column(String(100), nullable=False)

    # Raw image bytes stored in PostgreSQL BYTEA
    image_data: bytes = Column(LargeBinary, nullable=False)

    created_at: datetime = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
