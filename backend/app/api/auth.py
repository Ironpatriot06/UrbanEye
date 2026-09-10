"""
Authentication API router.

Endpoints
---------
POST /api/v1/auth/register  — Create a new citizen account
POST /api/v1/auth/login     — Log in and receive a JWT
GET  /api/v1/auth/me        — Return the currently authenticated user
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.core.security import create_access_token
from app.db.database import get_db
from app.models.user import UserRole
from app.schemas.user import TokenResponse, UserCreate, UserLogin, UserRead
from app.services import user_service

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new citizen account",
    description=(
        "Create a new USER-role account. "
        "Admin and Agent accounts must be created via the setup script."
    ),
)
def register(payload: UserCreate, db: Session = Depends(get_db)) -> UserRead:
    existing = user_service.get_user_by_email(db, payload.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )
    user = user_service.create_user(db, payload, role=UserRole.USER)
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in and receive a JWT access token",
)
def login(payload: UserLogin, db: Session = Depends(get_db)) -> TokenResponse:
    user = user_service.authenticate_user(db, payload.email, payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user.email, "role": user.role})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user.role,
        user_id=user.id,
        name=user.name,
    )


@router.get(
    "/me",
    response_model=UserRead,
    summary="Get the currently authenticated user",
)
def me(current_user=Depends(get_current_user)) -> UserRead:
    return current_user
