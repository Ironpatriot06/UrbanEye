"""
Incident Images API router.

Endpoints
---------
GET  /api/v1/incidents/{incident_id}/images
    → List image metadata for an incident (no binary data), both kinds

GET  /api/v1/incidents/{incident_id}/images/{image_id}
    → Stream the raw image bytes (Content-Type: image/*)

POST /api/v1/incidents/{incident_id}/images/resolution
    → The assigned agent attaches proof that the work was done, after they
      have marked the incident RESOLVED

Access control
--------------
Reading (both GET endpoints):
    USER  — only images on their own incidents
    AGENT — only images on incidents assigned to them
    ADMIN — any incident's images

Because proof-of-work images live in the same table and are served by the same
two endpoints, all three roles see them under exactly that rule: the citizen who
reported the incident sees the proof for their own report, the agent sees it on
their own assignment, and the admin sees everything.

Uploading proof (POST):
    AGENT — only the agent the incident is assigned to
    ADMIN — any incident (admins already control every other part of the
            workflow, so withholding this one would be arbitrary)
    USER  — 403.  A citizen cannot attest that work was completed.
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import Response
from sqlalchemy.orm import Session
from typing import List

from app.core.deps import get_current_user
from app.db.database import get_db
from app.models.image import ImageKind
from app.models.incident import IncidentStatus
from app.models.user import UserRole
from app.schemas.image import ImageRead
from app.services import image_service, incident_service

router = APIRouter(tags=["Incident Images"])


def _check_image_access(incident, current_user):
    """Raise 403/404 if the current user cannot access this incident's images."""
    if incident is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found.")
    role = current_user.role
    if role == UserRole.ADMIN:
        return
    if role == UserRole.USER and incident.reported_by != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")
    if role == UserRole.AGENT and incident.assigned_agent_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")


@router.get(
    "/incidents/{incident_id}/images",
    response_model=List[ImageRead],
    summary="List image metadata for an incident",
)
def list_images(
    incident_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> List[ImageRead]:
    incident = incident_service.get_incident_by_id(db, incident_id)
    _check_image_access(incident, current_user)
    return image_service.get_images_for_incident(db, incident_id)


@router.get(
    "/incidents/{incident_id}/images/{image_id}",
    summary="Stream a single image",
    response_class=Response,
)
def get_image(
    incident_id: int,
    image_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    incident = incident_service.get_incident_by_id(db, incident_id)
    _check_image_access(incident, current_user)

    img = image_service.get_image_by_id(db, incident_id, image_id)
    if img is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")

    return Response(
        content=img.image_data,
        media_type=img.content_type,
        headers={"Content-Disposition": f'inline; filename="{img.filename}"'},
    )


@router.post(
    "/incidents/{incident_id}/images/resolution",
    response_model=ImageRead,
    status_code=status.HTTP_201_CREATED,
    summary="Attach proof that the work was completed",
    description=(
        "The assigned agent uploads a photo showing the finished work, after "
        "marking the incident **RESOLVED**.\n\n"
        "The image is stored exactly like a citizen's report photo — same "
        "table, same validation (JPEG/PNG/WEBP, 5 MB), same authenticated "
        "streaming endpoint — and is tagged `RESOLUTION` so the UI can label "
        "it as proof of work. It becomes visible to the reporting citizen, the "
        "assigned agent, and any admin.\n\n"
        "Refused with 422 unless the incident is RESOLVED: proof of completed "
        "work only means something once the work is declared complete."
    ),
)
async def upload_resolution_image(
    incident_id: int,
    file: UploadFile = File(..., description="JPEG, PNG or WEBP, max 5 MB."),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> ImageRead:
    role = current_user.role

    if role == UserRole.USER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Citizens cannot attach proof of work.",
        )

    incident = incident_service.get_incident_by_id(db, incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found."
        )

    if role == UserRole.AGENT and incident.assigned_agent_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only attach proof to incidents assigned to you.",
        )

    if incident.status != IncidentStatus.RESOLVED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Proof of work can only be attached once the incident is marked "
                "Resolved."
            ),
        )

    return await image_service.save_image(
        db,
        incident_id,
        file,
        actor=current_user,
        kind=ImageKind.RESOLUTION,
    )
