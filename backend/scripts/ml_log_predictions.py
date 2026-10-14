#!/usr/bin/env python3
"""
Log ML predictions for every incident that does not have them yet, and/or
print the shadow-mode evaluation (logged predictions vs observed outcomes).

Predictions are computed as of each incident's created_at, so logging an old
incident now gives exactly what would have been logged when it was reported
(same bundle).  Idempotent: re-running adds nothing already logged.  Reads
incidents and history; writes only ml_predictions.  No scheduler is needed —
run it after deploying a new bundle, or whenever an evaluation is wanted.

    cd backend && source .venv/bin/activate
    python scripts/ml_log_predictions.py            # backfill, then evaluate
    python scripts/ml_log_predictions.py --evaluate-only
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.base import Base  # noqa: E402
from app.db.database import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402,F401  — registers every model with Base
from app.ml.evaluation import shadow_evaluation  # noqa: E402
from app.ml.registry import load_registry  # noqa: E402
from app.ml.service import backfill_predictions  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    Base.metadata.create_all(bind=engine)          # creates ml_predictions if missing
    reg = load_registry()
    print(f"bundle: {reg.bundle_version} (error: {reg.error})")
    db = SessionLocal()
    try:
        if not a.evaluate_only:
            print("backfill:", json.dumps(backfill_predictions(db, a.limit)))
        print(json.dumps(shadow_evaluation(db), indent=2, default=str))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
