"""
Admin user-management API router.

Endpoints
---------
GET   /api/v1/admin/users                 — List every account
PATCH /api/v1/admin/users/{user_id}/role  — Grant USER / AGENT / ADMIN
PATCH /api/v1/admin/users/{user_id}/status— Activate or deactivate an account
GET   /api/v1/admin/audit                 — Recent account/authorization events

Authorization
-------------
Every endpoint depends on `require_admin`, which resolves the caller from the
JWT and then checks the ROLE ON THEIR DATABASE ROW.  Consequences worth stating
explicitly, because the whole feature rests on them:

  - No token            -> 401.
  - USER or AGENT token -> 403, whatever the client believes its role to be.
  - A token minted while the holder was an ADMIN stops working the moment an
    admin demotes them, because the role is re-read on every request rather
    than taken from the token's claim.

Nothing here trusts a role, an actor, or a target identity from a request body.
The actor is the authenticated user; the target comes from the path.

Safety rules
------------
Enforced in the service layer (app/services/user_service.py) so they hold for
any caller, and surfaced here as HTTP 409:

  - An admin cannot change their own role or deactivate their own account.
  - The last active ADMIN cannot be demoted or deactivated.

Together these guarantee the system always retains at least one administrator
who can sign in.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from app.core.deps import require_admin
from app.db.database import get_db
from app.models.user import UserRole
from app.schemas.user import (
    AdminUserRead,
    UserAuditRead,
    UserRoleUpdate,
    UserStatusUpdate,
)
from app.services import user_audit_service, user_service
from app.services.user_service import RoleChangeError

router = APIRouter(prefix="/admin", tags=["Admin · User management"])


def _get_target(db: Session, user_id: int):
    user = user_service.get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id={user_id} not found.",
        )
    return user


@router.get(
    "/users",
    response_model=List[AdminUserRead],
    summary="List all users (admin only)",
    description=(
        "Every account, newest first — so an admin opening this page sees who "
        "just registered at the top.\n\n"
        "Returns no credential material: no password hashes, no Google subject "
        "ids, no tokens. `auth_methods` reports how each account can sign in "
        "(`EMAIL`, `GOOGLE`, or both)."
    ),
)
def list_users(
    role: Optional[UserRole] = Query(default=None, description="Filter by role."),
    q: Optional[str] = Query(
        default=None, max_length=100, description="Search name or email."
    ),
    db: Session = Depends(get_db),
    current_user=Depends(require_admin),
) -> List[AdminUserRead]:
    return user_service.list_users(db, role=role, search=q)


@router.patch(
    "/users/{user_id}/role",
    response_model=AdminUserRead,
    summary="Change a user's role (admin only)",
    description=(
        "Grants USER, AGENT or ADMIN to another account. The change takes "
        "effect on that user's next request — their existing token is not "
        "trusted for authorization.\n\n"
        "Refused with 409 if the caller targets their own account, or if the "
        "target is the last active administrator."
    ),
)
def change_role(
    payload: UserRoleUpdate,
    user_id: int = Path(..., description="ID of the user whose role to change."),
    db: Session = Depends(get_db),
    current_user=Depends(require_admin),
) -> AdminUserRead:
    target = _get_target(db, user_id)
    try:
        return user_service.set_user_role(db, target, payload.role, actor=current_user)
    except RoleChangeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.patch(
    "/users/{user_id}/status",
    response_model=AdminUserRead,
    summary="Activate or deactivate a user (admin only)",
    description=(
        "Deactivation is the reversible alternative to deletion: the account "
        "keeps its incidents and audit trail but can no longer sign in, and "
        "any token it already holds stops working immediately.\n\n"
        "Refused with 409 for the caller's own account or the last active "
        "administrator."
    ),
)
def change_status(
    payload: UserStatusUpdate,
    user_id: int = Path(..., description="ID of the user to activate or deactivate."),
    db: Session = Depends(get_db),
    current_user=Depends(require_admin),
) -> AdminUserRead:
    target = _get_target(db, user_id)
    try:
        return user_service.set_user_active(
            db, target, payload.is_active, actor=current_user
        )
    except RoleChangeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@router.get(
    "/audit",
    response_model=List[UserAuditRead],
    summary="Account and authorization audit trail (admin only)",
    description=(
        "Registrations, role changes, activations and Google account links, "
        "newest first. Separate from incident history, which answers a "
        "different question about a different subject.\n\n"
        "Contains no credential material."
    ),
)
def list_audit(
    user_id: Optional[int] = Query(
        default=None, description="Limit to events about one user."
    ),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user=Depends(require_admin),
) -> List[UserAuditRead]:
    return user_audit_service.get_recent(db, limit=limit, target_user_id=user_id)
