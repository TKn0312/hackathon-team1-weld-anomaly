-- The database for a NEW install. Run with: npx wrangler d1 execute hackathon-team1-weld-db --local --file=schema.sql
-- (The live database was created before user accounts: migrations/0002_users.sql adds them there.)

-- One row per user account. The password itself is never stored, only a salted PBKDF2 hash.
CREATE TABLE IF NOT EXISTS users (
  id            TEXT PRIMARY KEY,             -- a random id (crypto.randomUUID)
  username      TEXT NOT NULL UNIQUE,         -- 3-32 characters a-z 0-9 _ . -
  password_hash TEXT NOT NULL,                -- "pbkdf2$<rounds>$<salt>$<hash>"
  created_at    TEXT NOT NULL                 -- ISO date and time, UTC
);

-- One row per prediction
CREATE TABLE IF NOT EXISTS predictions (
  id          TEXT PRIMARY KEY,               -- a random id (crypto.randomUUID)
  created_at  TEXT NOT NULL,                  -- ISO date and time, UTC
  image_key   TEXT NOT NULL,                  -- where the uploaded image is stored in R2
  file_name   TEXT,                           -- the name of the uploaded file
  score       REAL NOT NULL,                  -- reconstruction error returned by the model server
  threshold   REAL NOT NULL,                  -- threshold used by the model server
  is_anomaly  INTEGER NOT NULL,               -- 1 = anomaly, 0 = normal (SQLite has no boolean)
  risk_level  TEXT,                           -- low / check / high, as decided by the model server
  score_vs_threshold REAL,                    -- the score as a multiple of the threshold
  user_id     TEXT REFERENCES users (id)      -- whose check it is; each user sees only their own rows
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions (user_id, created_at);
