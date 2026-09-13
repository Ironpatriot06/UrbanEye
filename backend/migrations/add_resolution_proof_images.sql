-- UrbanEye+ — Migration: proof-of-work images on resolved incidents
--
-- Run this ONCE against the development database after deploying the
-- resolution-proof feature.  It is safe to run repeatedly: every step is
-- guarded by IF NOT EXISTS or a DO block.
--
-- Usage:
--   docker exec -i urbaneye-postgres psql -U urbaneye -d urbaneye < backend/migrations/add_resolution_proof_images.sql
--
-- What it does NOT do
-- -------------------
-- It adds one column to `incident_images` and nothing else.  No image bytes are
-- read, rewritten or deleted; no row is removed; `incidents`, `users`,
-- `incident_history` and `user_audit_log` are untouched.
--
-- Existing images
-- ---------------
-- Every image that already exists was attached by a citizen when reporting an
-- incident — proof-of-work upload did not exist until now — so REPORT is not a
-- guess, it is the only thing they can be.  The column's DEFAULT back-fills
-- them, which is why this migration needs no UPDATE statement.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Enum: what an attached image is evidence of.
--    Mirrors ImageKind in app/models/image.py.  Keep the two in step.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'imagekind') THEN
        CREATE TYPE imagekind AS ENUM ('REPORT', 'RESOLUTION');
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- 2. incident_images.kind
--
--    REPORT     — the citizen's photo of the problem.
--    RESOLUTION — the assigned agent's photo of the finished work.
--
--    Both kinds stay in this one table: they are stored, validated, streamed
--    and access-controlled identically, and only the label differs.  A second
--    table would have duplicated the BYTEA storage and the authorization rule
--    on the serving endpoint for no gain.
--
--    NOT NULL DEFAULT 'REPORT' back-fills every pre-existing row in place.
-- ---------------------------------------------------------------------------
ALTER TABLE incident_images
    ADD COLUMN IF NOT EXISTS kind imagekind NOT NULL DEFAULT 'REPORT';

-- Reading an incident's gallery filters by kind to split the two groups, so the
-- column is indexed alongside the incident_id it is always queried with.
CREATE INDEX IF NOT EXISTS ix_incident_images_kind
    ON incident_images (kind);

COMMIT;

-- ---------------------------------------------------------------------------
-- Verification — run these afterwards to confirm nothing was lost:
--
--   SELECT kind, count(*) FROM incident_images GROUP BY kind;
--   SELECT count(*) FROM incident_images WHERE image_data IS NULL;  -- expect 0
-- ---------------------------------------------------------------------------
