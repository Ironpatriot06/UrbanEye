"""
SQLAlchemy declarative base.

All ORM models inherit from Base.  Keeping Base in its own module avoids
circular-import issues when models are spread across multiple files.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass
