-- UrbanEye+ — Migration: ML prediction log (Phase 2C)
--
-- Additive only: creates one new table.  No existing table, column, row or
-- enum is changed.  Safe to run repeatedly (IF NOT EXISTS everywhere).
-- In development the table is also created by Base.metadata.create_all at
-- startup; this file is the record of it for any database managed by hand.
--
-- Usage:
--   docker exec -i urbaneye-postgres psql -U urbaneye -d urbaneye < backend/migrations/add_ml_predictions.sql
--
-- Mirrors app/models/ml_prediction.py.  Keep the two in step.

BEGIN;

CREATE TABLE IF NOT EXISTS ml_predictions (
    id               SERIAL PRIMARY KEY,
    incident_id      INTEGER NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    task             VARCHAR(20) NOT NULL,
    status           VARCHAR(20) NOT NULL,
    method           VARCHAR(64),
    model_version    VARCHAR(64),
    bundle_version   VARCHAR(64),
    sla_hours        INTEGER,
    fallback_reason  VARCHAR(64),
    reasons          JSONB NOT NULL,
    output           JSONB NOT NULL,
    features         JSONB,
    features_as_of   TIMESTAMPTZ NOT NULL,
    generated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    dedupe_key       TEXT NOT NULL,
    CONSTRAINT uq_ml_predictions_incident_key UNIQUE (incident_id, dedupe_key)
);

CREATE INDEX IF NOT EXISTS ix_ml_predictions_incident_id ON ml_predictions (incident_id);

COMMIT;
