"""
Database engine and session management.

Design decisions:
- We use SQLAlchemy's synchronous engine with psycopg2 for simplicity.
  An async engine (asyncpg) can be swapped in later without changing the
  rest of the codebase as long as the dependency (get_db) stays the same.
- Session is scoped per-request via a FastAPI dependency.
- The engine is created once at import time (module-level singleton).
"""

from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.DATABASE_URL,
    # Keep a small pool; the dev server has low concurrency.
    pool_pre_ping=True,   # detect stale connections before use
    echo=settings.DEBUG,  # log SQL only when DEBUG=True
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session.

    Usage::

        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection() -> bool:
    """
    Execute a trivial query to verify the database is reachable.

    Returns True on success, raises an exception on failure.
    Used by the /health/db endpoint.
    """
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return True
