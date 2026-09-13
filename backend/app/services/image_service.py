"""
Image service layer.

Validation
----------
- Accepted MIME types: image/jpeg, image/png, image/webp
- Maximum file size: 5 MB

Security note: we validate content_type from the upload header and also
check the first few magic bytes of the file.  We never execute the file
contents.
"""

from typing import List, Optional

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.models.image import ImageKind, IncidentImage

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB

# Magic byte prefixes for JPEG, PNG, WEBP
_MAGIC = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
    b"RIFF": "image/webp",  # RIFF....WEBP
}


def _validate_image(data: bytes, content_type: str) -> None:
    """Raise HTTP 422 if the upload is not a valid supported image."""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{content_type}'. Accepted: JPEG, PNG, WEBP.",
        )
    if len(data) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File too large. Maximum allowed size is 5 MB.",
        )
    # Magic-byte check
    valid = any(data.startswith(magic) for magic in _MAGIC)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File content does not match a supported image format.",
        )


async def save_image(
    db: Session,
    incident_id: int,
    file: UploadFile,
    actor=None,
    kind: ImageKind = ImageKind.REPORT,
) -> IncidentImage:
    """
    Read, validate, and persist an uploaded image for the given incident.

    One path for both kinds: a citizen's report photo and an agent's proof of
    completed work are validated, stored and served identically.  `kind` only
    decides how the image is labelled — for the UI, and in the sentence written
    to the audit trail.

    An IMAGE_ADDED audit entry is written in the same transaction, but only
    after validation passes — a rejected upload attached nothing, so it is not
    something that happened to the incident.

    The entry carries the filename, media type and size only.  The bytes stay
    in incident_images; duplicating them into the audit trail would bloat a
    table that exists to be read, and the image is already addressable by its
    own endpoint.
    """
    from app.models.history import HistoryAction
    from app.services import history_service

    data = await file.read()
    content_type = file.content_type or "application/octet-stream"
    _validate_image(data, content_type)

    image = IncidentImage(
        incident_id=incident_id,
        filename=file.filename or "upload",
        content_type=content_type,
        image_data=data,
        kind=kind,
    )
    db.add(image)
    db.flush()  # assign image.id so the audit entry can name it

    size_kb = max(1, round(len(data) / 1024))
    noun = "Proof-of-work photo" if kind == ImageKind.RESOLUTION else "Photo"
    history_service.record(
        db,
        incident_id,
        HistoryAction.IMAGE_ADDED,
        actor=actor,
        new_value=image.filename,
        description=(
            f"{noun} '{image.filename}' attached ({content_type}, {size_kb} KB, "
            f"image #{image.id})."
        ),
    )

    db.commit()
    db.refresh(image)
    return image


def get_images_for_incident(
    db: Session,
    incident_id: int,
    kind: Optional[ImageKind] = None,
) -> List[IncidentImage]:
    """
    Return images for the given incident, oldest first.

    Both kinds are returned by default: a caller showing an incident wants the
    report photos and the proof of work together, and separating them is the
    UI's job, not a second round trip's.
    """
    query = db.query(IncidentImage).filter(IncidentImage.incident_id == incident_id)
    if kind is not None:
        query = query.filter(IncidentImage.kind == kind)
    return query.order_by(IncidentImage.created_at, IncidentImage.id).all()


def get_image_by_id(
    db: Session, incident_id: int, image_id: int
) -> Optional[IncidentImage]:
    """Return a specific image belonging to an incident."""
    return (
        db.query(IncidentImage)
        .filter(
            IncidentImage.id == image_id,
            IncidentImage.incident_id == incident_id,
        )
        .first()
    )


def count_images_for_incident(db: Session, incident_id: int) -> int:
    """Return the number of images attached to an incident."""
    return (
        db.query(IncidentImage)
        .filter(IncidentImage.incident_id == incident_id)
        .count()
    )
