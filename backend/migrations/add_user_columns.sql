-- UrbanEye+ — Migration: Add user FK columns to incidents table
-- Run this ONCE against the development database after deploying Review III.
-- It is safe to run multiple times (IF NOT EXISTS).
--
-- Usage:
--   docker exec -i urbaneye-postgres psql -U urbaneye -d urbaneye < backend/migrations/add_user_columns.sql

ALTER TABLE incidents
    ADD COLUMN IF NOT EXISTS reported_by INTEGER REFERENCES users(id) ON DELETE SET NULL;

ALTER TABLE incidents
    ADD COLUMN IF NOT EXISTS assigned_agent_id INTEGER REFERENCES users(id) ON DELETE SET NULL;
