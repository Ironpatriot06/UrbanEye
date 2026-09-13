-- UrbanEye+ — Migration: authentication providers & admin user management
--
-- Run this ONCE against the development database after deploying the
-- authentication / role-management feature.  It is safe to run repeatedly:
-- every step is guarded by IF NOT EXISTS, a DO block, or a NOT EXISTS predicate.
--
-- Usage:
--   docker exec -i urbaneye-postgres psql -U urbaneye -d urbaneye < backend/migrations/add_auth_and_user_management.sql
--
-- What it does NOT do
-- -------------------
-- It does not delete a user, reset a role, clear a password, or touch
-- `incidents`, `incident_images` or `incident_history`.  Existing ADMIN, AGENT
-- and USER accounts — including ones created by scripts/create_demo_data.py —
-- keep their role, their password and their id.  Every column it adds either
-- carries a DEFAULT that back-fills existing rows correctly (is_active = TRUE)
-- or is nullable (google_subject_id, last_login_at).
--
-- The one constraint it RELAXES is users.password_hash NOT NULL, which must go
-- so that an account created purely by Google sign-in can exist without a
-- password.  Relaxing a NOT NULL cannot invalidate any existing row.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. users.is_active
--
--    The reversible alternative to deleting an account: it can no longer sign
--    in, and app/core/deps.py rejects any token it already holds, but its
--    incidents and audit trail survive.
--
--    DEFAULT TRUE means every account that exists today stays usable.  This is
--    the step that keeps the demo admin/agent accounts working.
-- ---------------------------------------------------------------------------
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;

-- ---------------------------------------------------------------------------
-- 2. users.google_subject_id
--
--    Google's `sub` claim: stable for the life of the Google account and
--    unaffected by the user changing their Google email address.  UNIQUE, so
--    one Google identity can never back two accounts.  NULL for every existing
--    account, which is correct — none of them has signed in with Google.
-- ---------------------------------------------------------------------------
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS google_subject_id VARCHAR(255);

CREATE UNIQUE INDEX IF NOT EXISTS ix_users_google_subject_id
    ON users (google_subject_id);

-- ---------------------------------------------------------------------------
-- 3. users.last_login_at
--
--    Stamped by /auth/login and the Google callback.  NULL for accounts that
--    have not signed in since this migration ran — shown as "Never" in the
--    admin table rather than back-filled with a time that never happened.
-- ---------------------------------------------------------------------------
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMP WITH TIME ZONE;

-- ---------------------------------------------------------------------------
-- 4. users.password_hash becomes nullable
--
--    A Google-only account has no password to store.  Storing a placeholder
--    digest instead would be worse: it would look like a credential, and
--    anything that stopped treating NULL specially could then be tricked into
--    verifying against it.  app/services/user_service.authenticate_user
--    refuses an account with no hash outright.
--
--    DROP NOT NULL is idempotent in PostgreSQL — running it on an already
--    nullable column succeeds and changes nothing.
-- ---------------------------------------------------------------------------
ALTER TABLE users
    ALTER COLUMN password_hash DROP NOT NULL;

-- ---------------------------------------------------------------------------
-- 5. Enum: the controlled set of auditable account operations.
--    Mirrors UserAuditAction in app/models/user_audit.py.  Keep the two in step.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'userauditaction') THEN
        CREATE TYPE userauditaction AS ENUM (
            'USER_REGISTERED',
            'USER_ROLE_CHANGED',
            'USER_STATUS_CHANGED',
            'GOOGLE_ACCOUNT_LINKED'
        );
    END IF;
END
$$;

-- `actorrole` already exists — it was created by add_incident_history.sql and
-- is shared by both audit trails.  Created here too so this migration can run
-- against a database that has not had that one applied.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'actorrole') THEN
        CREATE TYPE actorrole AS ENUM ('USER', 'ADMIN', 'AGENT', 'SYSTEM');
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- 6. The account audit table
--
--    Separate from incident_history on purpose: that table answers "what
--    happened to incident N" and every row there is keyed to an incident.  An
--    account being promoted belongs to no incident.
--
--    Both user FKs are ON DELETE SET NULL, and the email/name snapshots stay
--    behind, so the trail still reads correctly after an account is removed —
--    which is exactly when an audit log matters most.
--
--    It holds no credential material: no password hashes, no Google subject
--    ids, no tokens.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_audit_log (
    id              SERIAL PRIMARY KEY,
    action          userauditaction NOT NULL,
    actor_id        INTEGER REFERENCES users(id) ON DELETE SET NULL,
    actor_role      actorrole NOT NULL DEFAULT 'SYSTEM',
    actor_email     VARCHAR(254),
    actor_name      VARCHAR(100),
    target_user_id  INTEGER REFERENCES users(id) ON DELETE SET NULL,
    target_email    VARCHAR(254),
    target_name     VARCHAR(100),
    old_value       VARCHAR(255),
    new_value       VARCHAR(255),
    description     TEXT,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX IF NOT EXISTS ix_user_audit_log_id         ON user_audit_log (id);
CREATE INDEX IF NOT EXISTS ix_user_audit_log_action     ON user_audit_log (action);
CREATE INDEX IF NOT EXISTS ix_user_audit_log_actor      ON user_audit_log (actor_id);
CREATE INDEX IF NOT EXISTS ix_user_audit_log_target     ON user_audit_log (target_user_id);
CREATE INDEX IF NOT EXISTS ix_user_audit_log_created_at ON user_audit_log (created_at);

-- ---------------------------------------------------------------------------
-- 7. Append-only enforcement
--
--    The third of three overlapping guarantees, matching what
--    add_incident_history.sql does for incident history:
--      1. No API endpoint updates or deletes audit rows (GET only).
--      2. SQLAlchemy mapper events raise on UPDATE/DELETE inside the backend.
--      3. This trigger rejects them at the database, covering anything that
--         reaches the table outside this application.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION user_audit_log_is_append_only()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION
        'user_audit_log is append-only: % is not permitted on this table',
        TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_user_audit_log_no_update ON user_audit_log;
CREATE TRIGGER trg_user_audit_log_no_update
    BEFORE UPDATE ON user_audit_log
    FOR EACH ROW EXECUTE FUNCTION user_audit_log_is_append_only();

DROP TRIGGER IF EXISTS trg_user_audit_log_no_delete ON user_audit_log;
CREATE TRIGGER trg_user_audit_log_no_delete
    BEFORE DELETE ON user_audit_log
    FOR EACH ROW EXECUTE FUNCTION user_audit_log_is_append_only();

-- ---------------------------------------------------------------------------
-- 8. Back-fill a registration event for every pre-existing account
--
--    These accounts were created before this trail existed, by
--    scripts/create_demo_data.py or by the old registration endpoint.  We know
--    three things for certain — that the account exists, when it was created,
--    and what role it currently holds — so that is all this records, attributed
--    to SYSTEM rather than to an admin who may not have been involved.
--
--    Guarded by NOT EXISTS on (target_user_id, USER_REGISTERED), so re-running
--    the migration never duplicates an event.
-- ---------------------------------------------------------------------------
INSERT INTO user_audit_log (
    action, actor_id, actor_role, actor_email, actor_name,
    target_user_id, target_email, target_name,
    old_value, new_value, description, created_at
)
SELECT
    'USER_REGISTERED'::userauditaction,
    NULL,
    'SYSTEM'::actorrole,
    NULL,
    'System',
    u.id,
    u.email,
    u.name,
    NULL,
    u.role::text,
    'Account existed before the account audit trail was introduced. '
        || 'Recorded with the role it held at that point; how it was created is not known.',
    u.created_at
FROM users u
WHERE NOT EXISTS (
    SELECT 1 FROM user_audit_log a
    WHERE a.target_user_id = u.id
      AND a.action = 'USER_REGISTERED'::userauditaction
);

COMMIT;

-- ---------------------------------------------------------------------------
-- Verification — run these afterwards to confirm nothing was lost:
--
--   SELECT role, count(*) FROM users GROUP BY role;
--   SELECT id, email, role, is_active, google_subject_id IS NOT NULL AS has_google
--     FROM users ORDER BY id;
--   SELECT count(*) FROM user_audit_log;
-- ---------------------------------------------------------------------------
