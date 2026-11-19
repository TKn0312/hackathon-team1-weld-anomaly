-- Adds user accounts to the live database (created before accounts existed). Run ONCE:
--   npx wrangler d1 execute hackathon-team1-weld-db --remote --file=migrations/0002_users.sql
-- The checks saved before this have no user_id, so no user sees them any more; they stay in the table.
CREATE TABLE IF NOT EXISTS users (
  id            TEXT PRIMARY KEY,
  username      TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  created_at    TEXT NOT NULL
);
ALTER TABLE predictions ADD COLUMN user_id TEXT REFERENCES users (id);
CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions (user_id, created_at);
