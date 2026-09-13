-- UrbanEye+ — Migration: Incident history / audit trail
--
-- Run this ONCE against the development database after deploying the audit
-- trail feature.  It is safe to run repeatedly: every step is guarded by
-- IF NOT EXISTS, a DO block, or a NOT EXISTS predicate.
--
-- Usage:
--   docker exec -i urbaneye-postgres psql -U urbaneye -d urbaneye < backend/migrations/add_incident_history.sql
--
-- What it does NOT do
-- -------------------
-- It never modifies or deletes a row in `incidents`, `users` or
-- `incident_images`.  The only write outside the new table is the backfill in
-- step 5, which inserts into `incident_history` alone.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Enum: the controlled set of auditable actions
--    Mirrors HistoryAction in app/models/history.py.  Keep the two in step.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'incidenthistoryaction') THEN
        CREATE TYPE incidenthistoryaction AS ENUM (
            'INCIDENT_CREATED',
            'INCIDENT_TRIAGED',
            'PRIORITY_CHANGED',
            'SLA_CREATED',
            'SLA_UPDATED',
            'AGENT_ASSIGNED',
            'AGENT_REASSIGNED',
            'AGENT_UNASSIGNED',
            'STATUS_CHANGED',
            'AGENT_AVAILABILITY_CHANGED',
            'IMAGE_ADDED',
            'INCIDENT_RESOLVED',
            'INCIDENT_CLOSED',
            'INCIDENT_REOPENED',
            'ADMIN_OVERRIDE'
        );
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- 2. Enum: who acted.  UserRole plus SYSTEM for work no person requested.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'actorrole') THEN
        CREATE TYPE actorrole AS ENUM ('USER', 'ADMIN', 'AGENT', 'SYSTEM');
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- 3. The audit table
--
--    incident_id  CASCADE       — history dies with the incident it describes.
--    actor_id     SET NULL      — deleting a user must not erase the record of
--                                 what they did; actor_role and actor_name
--                                 keep the entry readable afterwards.
--    created_at   clock_timestamp() — NOT now(), which is frozen at
--                                 transaction start, so several events written
--                                 in one transaction would share a timestamp
--                                 and lose their order.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS incident_history (
    id           SERIAL PRIMARY KEY,
    incident_id  INTEGER NOT NULL
                 REFERENCES incidents(id) ON DELETE CASCADE,
    actor_id     INTEGER
                 REFERENCES users(id) ON DELETE SET NULL,
    actor_role   actorrole NOT NULL DEFAULT 'SYSTEM',
    actor_name   VARCHAR(100),
    action       incidenthistoryaction NOT NULL,
    old_value    VARCHAR(255),
    new_value    VARCHAR(255),
    description  TEXT,
    created_at   TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT clock_timestamp()
);

-- ---------------------------------------------------------------------------
-- 4. Indexes
--    The composite (incident_id, created_at, id) serves the one query the
--    read path actually makes: one incident's trail in chronological order,
--    with the primary key breaking ties in true insertion order.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS ix_incident_history_incident_id ON incident_history (incident_id);
CREATE INDEX IF NOT EXISTS ix_incident_history_actor_id    ON incident_history (actor_id);
CREATE INDEX IF NOT EXISTS ix_incident_history_created_at  ON incident_history (created_at);
CREATE INDEX IF NOT EXISTS ix_incident_history_action      ON incident_history (action);
CREATE INDEX IF NOT EXISTS ix_incident_history_timeline
    ON incident_history (incident_id, created_at, id);

-- ---------------------------------------------------------------------------
-- 5. Backfill: give pre-existing incidents an opening event
--
--    The real history of an incident reported before this table existed cannot
--    be reconstructed, and inventing one would make the trail a liar.  The one
--    thing we do know is that it was created, when, and — from reported_by —
--    usually by whom.  That is all this writes.
--
--    Where the reporter is unknown the event is attributed to SYSTEM rather
--    than guessed at.  The entry is stamped with the incident's own created_at
--    so it sorts ahead of everything recorded afterwards.
--
--    Idempotent: incidents that already have any history are skipped.
--    Mirrors history_service.backfill_initial_history().
-- ---------------------------------------------------------------------------
INSERT INTO incident_history (
    incident_id, actor_id, actor_role, actor_name,
    action, old_value, new_value, description, created_at
)
SELECT
    i.id,
    u.id,
    COALESCE(u.role::text, 'SYSTEM')::actorrole,
    COALESCE(u.name, 'System'),
    'INCIDENT_CREATED'::incidenthistoryaction,
    NULL,
    'REPORTED',
    CASE
        WHEN u.id IS NOT NULL THEN 'Incident reported by ' || u.name || '.'
        ELSE 'Incident reported. Reporter unknown — this entry was '
             || 'reconstructed when the audit trail was introduced.'
    END,
    i.created_at
FROM incidents i
LEFT JOIN users u ON u.id = i.reported_by
WHERE NOT EXISTS (
    SELECT 1 FROM incident_history h WHERE h.incident_id = i.id
);

-- ---------------------------------------------------------------------------
-- 6. Immutability at the database level
--
--    The API exposes no way to write history, and the ORM raises on any
--    attempt (see the mapper events in app/models/history.py).  This trigger
--    is the backstop that also covers anything reaching the table outside the
--    application — a psql session, a future script, a mistaken migration.
--
--    INSERT stays permitted; that is how the trail grows.
--
--    Consequence worth knowing: because the trigger fires FOR EACH ROW on
--    DELETE, it also blocks the ON DELETE CASCADE from `incidents`, so an
--    incident that has history can no longer be deleted while these triggers
--    are installed.  The application has no incident-delete path, and refusing
--    to destroy the record of what happened is the behaviour this feature
--    wants — but it is a deliberate choice, not an oversight.  To retire an
--    incident, drop both triggers first, delete, then re-run this migration.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION incident_history_is_append_only()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION
        'incident_history is append-only: % is not permitted on this table',
        TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_incident_history_no_update ON incident_history;
CREATE TRIGGER trg_incident_history_no_update
    BEFORE UPDATE ON incident_history
    FOR EACH ROW EXECUTE FUNCTION incident_history_is_append_only();

DROP TRIGGER IF EXISTS trg_incident_history_no_delete ON incident_history;
CREATE TRIGGER trg_incident_history_no_delete
    BEFORE DELETE ON incident_history
    FOR EACH ROW EXECUTE FUNCTION incident_history_is_append_only();

COMMIT;
