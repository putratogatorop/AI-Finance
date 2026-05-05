-- v_new_2 paper-deploy DB schema (applied on VPS 2026-05-05)
--
-- Tables:
--   v_new_2_signals          — scanner output, one row per rule trigger
--   v_new_2_paper_positions  — open paper positions
--   v_new_2_paper_trades     — closed trades (one per position)
--   v_new_2_paper_account    — equity curve snapshots

CREATE TABLE IF NOT EXISTS v_new_2_signals (
  id              SERIAL PRIMARY KEY,
  signal_time     TIMESTAMPTZ NOT NULL,
  symbol          VARCHAR(20) NOT NULL,
  side            VARCHAR(10) NOT NULL,
  tier            VARCHAR(12),
  is_meme         BOOLEAN DEFAULT FALSE,
  pullback_pct    DOUBLE PRECISION,
  pullback_atr    DOUBLE PRECISION,
  bgm_score       DOUBLE PRECISION NOT NULL,
  d_ema_spread    DOUBLE PRECISION,
  close_at_signal DOUBLE PRECISION NOT NULL,
  taken           BOOLEAN DEFAULT FALSE,
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT v_new_2_signals_unique UNIQUE(signal_time, symbol, side)
);
CREATE INDEX IF NOT EXISTS ix_v_new_2_signals_time ON v_new_2_signals(signal_time DESC);
CREATE INDEX IF NOT EXISTS ix_v_new_2_signals_taken ON v_new_2_signals(taken) WHERE taken = FALSE;

CREATE TABLE IF NOT EXISTS v_new_2_paper_positions (
  id                  SERIAL PRIMARY KEY,
  signal_id           INTEGER REFERENCES v_new_2_signals(id),
  symbol              VARCHAR(20) NOT NULL,
  side                VARCHAR(10) NOT NULL,
  is_meme             BOOLEAN DEFAULT FALSE,
  tier                VARCHAR(12),
  entry_time          TIMESTAMPTZ NOT NULL,
  entry_price         DOUBLE PRECISION NOT NULL,
  bgm_score           DOUBLE PRECISION,
  position_size_pct   DOUBLE PRECISION NOT NULL,
  leverage            DOUBLE PRECISION NOT NULL,
  notional_usd        DOUBLE PRECISION,
  account_at_entry    DOUBLE PRECISION,
  running_extreme     DOUBLE PRECISION,
  profit_lock_active  BOOLEAN DEFAULT FALSE,
  status              VARCHAR(20) DEFAULT 'open',
  created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_v_new_2_pos_status ON v_new_2_paper_positions(status) WHERE status = 'open';

CREATE TABLE IF NOT EXISTS v_new_2_paper_trades (
  id              SERIAL PRIMARY KEY,
  position_id     INTEGER REFERENCES v_new_2_paper_positions(id),
  symbol          VARCHAR(20) NOT NULL,
  side            VARCHAR(10) NOT NULL,
  is_meme         BOOLEAN,
  tier            VARCHAR(12),
  entry_time      TIMESTAMPTZ NOT NULL,
  exit_time       TIMESTAMPTZ NOT NULL,
  entry_price     DOUBLE PRECISION NOT NULL,
  exit_price      DOUBLE PRECISION NOT NULL,
  bars_held       INTEGER,
  exit_reason     VARCHAR(40),
  bgm_score       DOUBLE PRECISION,
  position_size_pct DOUBLE PRECISION,
  leverage        DOUBLE PRECISION,
  pnl_pct         DOUBLE PRECISION NOT NULL,
  pnl_usd         DOUBLE PRECISION,
  realized_at     TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_v_new_2_trades_time ON v_new_2_paper_trades(realized_at DESC);

CREATE TABLE IF NOT EXISTS v_new_2_paper_account (
  id              SERIAL PRIMARY KEY,
  ts              TIMESTAMPTZ NOT NULL,
  equity_usd      DOUBLE PRECISION NOT NULL,
  open_positions  INTEGER NOT NULL,
  realized_trades INTEGER NOT NULL,
  notes           TEXT
);
