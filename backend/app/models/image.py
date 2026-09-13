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

Kinds
-----
REPORT     — a photo the citizen attached when reporting the incident: what the
             problem looks like.
RESOLUTION — a photo the assigned agent attached after marking the incident
             resolved: proof that the work was done.

Both kinds live in this one table and are stored, validated, served and
access-controlled identically — the only difference is the label the UI puts on
them, and who is allowed to add one.  A separate table would have duplicated the
byte storage, the validation, the streaming endpoint and its authorization rule
for no gain.
"""

import enum
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    func,
)

from app.db.base import Base


class ImageKind(str, enum.Enum):
    """What an attached image is evidence of."""

    REPORT = "REPORT"
    RESOLUTION = "RESOLUTION"


class IncidentImage(Base):
    """A single image attached to an incident."""

    __tablename__ = "incident_images"

    id: int = Column(Integer, primary_key=True, index=True)

    incident_id: int = Column(
        Integer, ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False, index=True
    )

    filename: str = Column(String(255), nullable=False)
    content_type: str = Column(String(100), nullable=False)

    #: REPORT unless an agent attached this as proof of completed work.  Every
    #: image that existed before resolution proof was introduced is a citizen's
    #: report photo, which is exactly what the default back-fills them to.
    kind: str = Column(
        Enum(ImageKind, name="imagekind"),
        nullable=False,
        default=ImageKind.REPORT,
        server_default=ImageKind.REPORT.value,
        index=True,
    )

    # Raw image bytes stored in PostgreSQL BYTEA
    image_data: bytes = Column(LargeBinary, nullable=False)

    created_at: datetime = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
