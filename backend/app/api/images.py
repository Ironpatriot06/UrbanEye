"""
Incident Images API router.

Endpoints
---------
GET /api/v1/incidents/{incident_id}/images
    → List image metadata for an incident (no binary data)

GET /api/v1/incidents/{incident_id}/images/{image_id}
    → Stream the raw image bytes (Content-Type: image/*)

Access control
--------------
USER  — can only access images for their own incidents
AGENT — can only access images for incidents assigned to them
ADMIN — can access any incident's images
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session
from typing import List

from app.core.deps import get_current_user
from app.db.database import get_db
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
