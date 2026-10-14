#!/usr/bin/env python3
"""
Initialise (or bring up to date) the UrbanEye+ database.  Idempotent.

    cd backend && source .venv/bin/activate
    python scripts/init_db.py            # uses DATABASE_URL from .env / the environment

Steps, in the only order that works on BOTH a brand-new and an existing database:
  1. CREATE EXTENSION IF NOT EXISTS postgis   (the postgis/postgis image already
     enables it in POSTGRES_DB; a plain PostgreSQL server needs PostGIS installed)
  2. Base.metadata.create_all()  — creates any missing table, never alters one
  3. every backend/migrations/*.sql file, in the order below.  Each is guarded
     with IF NOT EXISTS / DO blocks, so re-running is safe; on a new database
     they mostly no-op, but they also install what create_all cannot: the
     append-only triggers on incident_history and user_audit_log.

Nothing is dropped or deleted.  On a database created before a migration
existed, the migrations back-fill what they document (see each file's header).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402

from app.db.base import Base  # noqa: E402
from app.db.database import engine  # noqa: E402
from app.main import app  # noqa: E402,F401  — imports every ORM model

MIGRATIONS = [
    "add_user_columns.sql",
    "add_priority_sla_columns.sql",
    "add_incident_history.sql",
    "add_auth_and_user_management.sql",
    "add_resolution_proof_images.sql",
    "add_ml_predictions.sql",
]
MIGRATION_DIR = Path(__file__).resolve().parents[1] / "migrations"


def main() -> int:
    url = engine.url.render_as_string(hide_password=True)
    print(f"database: {url}")
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    print("postgis: ok")
    Base.metadata.create_all(bind=engine)
    print("tables: ok")
    raw = engine.raw_connection()
    try:
        raw.autocommit = True        # each file manages its own BEGIN / COMMIT
        for name in MIGRATIONS:
            sql = (MIGRATION_DIR / name).read_text()
            with raw.cursor() as cur:
                cur.execute(sql)
            print(f"migration {name}: ok")
    finally:
        raw.close()
    with engine.connect() as conn:
        tables = conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1")).scalars().all()
        triggers = conn.execute(text(
            "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal AND tgname LIKE 'trg_%' "
            "ORDER BY 1")).scalars().all()
    print("tables:", ", ".join(tables))
    print("append-only triggers:", ", ".join(triggers) or "NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
