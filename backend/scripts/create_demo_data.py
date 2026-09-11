"""
Demo data seeder for UrbanEye+ development.

Creates one ADMIN, one AGENT, and one USER account if they don't already
exist.  Account credentials are read from environment variables (or .env).

Usage:
    cd backend
    source .venv/bin/activate
    python scripts/create_demo_data.py

It is safe to re-run — existing accounts are left untouched.
"""

import sys
import os

# Allow importing app modules from the backend root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import get_settings
from app.db.base import Base
from app.db.database import engine, SessionLocal
from app.models import user as _user_models  # noqa: F401 — register model
from app.models import incident as _incident_models  # noqa: F401
from app.models import image as _image_models  # noqa: F401
from app.models.user import User, UserRole
from app.core.security import hash_password

settings = get_settings()


def upsert_user(db, name: str, email: str, password: str, role: UserRole):
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        print(f"  [skip] {role.value} {email!r} already exists (id={existing.id})")
        return existing

    user = User(
        name=name,
        email=email,
        password_hash=hash_password(password),
        role=role,
        is_available=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    print(f"  [created] {role.value} {email!r} (id={user.id})")
    return user


def main():
    print("UrbanEye+ — Creating demo accounts...")

    # Ensure tables exist
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        upsert_user(
            db,
            name=settings.DEMO_ADMIN_NAME,
            email=settings.DEMO_ADMIN_EMAIL,
            password=settings.DEMO_ADMIN_PASSWORD,
            role=UserRole.ADMIN,
        )
        upsert_user(
            db,
            name=settings.DEMO_AGENT_NAME,
            email=settings.DEMO_AGENT_EMAIL,
            password=settings.DEMO_AGENT_PASSWORD,
            role=UserRole.AGENT,
        )
        upsert_user(
            db,
            name=settings.DEMO_AGENT_NAME_2,
            email=settings.DEMO_AGENT_EMAIL_2,
            password=settings.DEMO_AGENT_PASSWORD_2,
            role=UserRole.AGENT,
        )
        upsert_user(
            db,
            name=settings.DEMO_USER_NAME,
            email=settings.DEMO_USER_EMAIL,
            password=settings.DEMO_USER_PASSWORD,
            role=UserRole.USER,
        )
        print("\nDemo accounts ready:")
        print(f"  ADMIN  → {settings.DEMO_ADMIN_EMAIL}  / {settings.DEMO_ADMIN_PASSWORD}")
        print(f"  AGENT  → {settings.DEMO_AGENT_EMAIL} / {settings.DEMO_AGENT_PASSWORD}")
        print(f"  AGENT  → {settings.DEMO_AGENT_EMAIL_2} / {settings.DEMO_AGENT_PASSWORD_2}")
        print(f"  USER   → {settings.DEMO_USER_EMAIL}  / {settings.DEMO_USER_PASSWORD}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
