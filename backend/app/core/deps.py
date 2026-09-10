"""
FastAPI dependency functions for authentication and authorization.

Usage
-----
Inject into any endpoint that requires authentication:

    @router.get("/protected")
    def protected(current_user: User = Depends(get_current_user)):
        ...

For role enforcement:

    @router.post("/admin-only")
    def admin_only(current_user: User = Depends(require_admin)):
        ...

Design decisions
----------------
- Bearer token is extracted from the `Authorization` header.
- Roles are embedded in the JWT payload so we avoid a DB lookup on every
  request for simple role checks.  The full User object is still fetched
  from the database so that the endpoint has access to all user fields.
- A missing or invalid token returns HTTP 401.
- A valid token for the wrong role returns HTTP 403.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.database import get_db

# We import User lazily inside the function body to avoid circular imports
# (models → db → base; deps → models would create a cycle via main.py).

bearer_scheme = HTTPBearer(auto_error=False)


def _extract_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    """Extract and return the raw JWT string from the Authorization header."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def get_current_user(
    token: str = Depends(_extract_token),
    db: Session = Depends(get_db),
):
    """
    Validate the JWT and return the corresponding User ORM object.

    Raises HTTP 401 if the token is invalid, expired, or the user no longer
    exists in the database.
    """
    from app.models.user import User  # local import to avoid circular deps

    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    email: str | None = payload.get("sub")
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_admin(current_user=Depends(get_current_user)):
    """Raise HTTP 403 unless the current user has the ADMIN role."""
    from app.models.user import UserRole

    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user


def require_agent(current_user=Depends(get_current_user)):
    """Raise HTTP 403 unless the current user has the AGENT role."""
    from app.models.user import UserRole

    if current_user.role != UserRole.AGENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Agent access required.",
        )
    return current_user


def require_admin_or_agent(current_user=Depends(get_current_user)):
    """Raise HTTP 403 unless the current user is ADMIN or AGENT."""
    from app.models.user import UserRole

    if current_user.role not in (UserRole.ADMIN, UserRole.AGENT):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Agent access required.",
        )
    return current_user
