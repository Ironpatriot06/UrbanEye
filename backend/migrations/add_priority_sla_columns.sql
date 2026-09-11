-- UrbanEye+ — Migration: Add system priority + SLA columns to incidents
--
-- Run this ONCE against the development database after deploying Review IV.
-- It is safe to run multiple times (idempotent: IF NOT EXISTS / guarded DO blocks).
--
-- Usage:
--   docker exec -i urbaneye-postgres psql -U urbaneye -d urbaneye < backend/migrations/add_priority_sla_columns.sql
--
-- Backward compatibility
-- ----------------------
-- Existing incident rows only have the legacy `severity` column.  This script
-- backfills `priority_level` from that severity so no existing row is lost or
-- rendered unreadable, then backfills the SLA window from the priority.
-- No row is ever deleted or recreated.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Enum type for the system-determined priority (P1..P4)
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'incidentpriority') THEN
        CREATE TYPE incidentpriority AS ENUM ('P1', 'P2', 'P3', 'P4');
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- 2. New columns (nullable first so existing rows survive the ALTER)
-- ---------------------------------------------------------------------------
ALTER TABLE incidents ADD COLUMN IF NOT EXISTS priority_level incidentpriority;
ALTER TABLE incidents ADD COLUMN IF NOT EXISTS sla_hours      INTEGER;
ALTER TABLE incidents ADD COLUMN IF NOT EXISTS sla_deadline   TIMESTAMP WITH TIME ZONE;
ALTER TABLE incidents ADD COLUMN IF NOT EXISTS sla_status     VARCHAR(20);

-- ---------------------------------------------------------------------------
-- 3. Backfill priority_level from the legacy severity column
--    CRITICAL -> P1, HIGH -> P2, MEDIUM -> P3, LOW -> P4
-- ---------------------------------------------------------------------------
UPDATE incidents
SET priority_level = CASE severity::text
                         WHEN 'CRITICAL' THEN 'P1'
                         WHEN 'HIGH'     THEN 'P2'
                         WHEN 'MEDIUM'   THEN 'P3'
                         WHEN 'LOW'      THEN 'P4'
                         ELSE 'P3'
                     END::incidentpriority
WHERE priority_level IS NULL;

-- ---------------------------------------------------------------------------
-- 4. Lock priority_level down now that every row has a value
-- ---------------------------------------------------------------------------
ALTER TABLE incidents ALTER COLUMN priority_level SET DEFAULT 'P3'::incidentpriority;
ALTER TABLE incidents ALTER COLUMN priority_level SET NOT NULL;

-- ---------------------------------------------------------------------------
-- 5. Backfill the SLA window from the (now populated) priority level
--    P1 = 4h, P2 = 24h, P3 = 72h, P4 = 168h — mirrors DEFAULT_SLA_HOURS
--    in app/models/incident.py.
-- ---------------------------------------------------------------------------
UPDATE incidents
SET sla_hours = CASE priority_level::text
                    WHEN 'P1' THEN 4
                    WHEN 'P2' THEN 24
                    WHEN 'P3' THEN 72
                    WHEN 'P4' THEN 168
                END
WHERE sla_hours IS NULL;

UPDATE incidents
SET sla_deadline = created_at + (sla_hours || ' hours')::interval
WHERE sla_deadline IS NULL AND sla_hours IS NOT NULL;

UPDATE incidents
SET sla_status = CASE
                     WHEN now() >= sla_deadline THEN 'BREACHED'
                     WHEN EXTRACT(EPOCH FROM (sla_deadline - now()))
                          < (sla_hours * 3600 * 0.2) THEN 'AT_RISK'
                     ELSE 'ON_TRACK'
                 END
WHERE sla_status IS NULL AND sla_deadline IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 6. Helpful indexes for admin dashboard filtering
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS ix_incidents_priority_level ON incidents (priority_level);
CREATE INDEX IF NOT EXISTS ix_incidents_sla_deadline   ON incidents (sla_deadline);

COMMIT;
