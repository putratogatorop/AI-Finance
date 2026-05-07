-- v_spot — paper-trading schema for the 1-2 position spot portfolio.
--
-- Reads signals from v_new_2_signals (read-only, no `taken` flag mutation).
-- Maintains its own positions/trades/account tables, independent of v_new_2_paper_*.
--
-- Strategy locked 2026-05-07 (services/python/results/v_new_2/spot_tiered_backtest_verdict.md):
--   cap=2 concurrent, 50% equity per slot, no leverage, BGM>=0.65, long-only,
--   no tiered profit-taking, ATR-trail exits inherited from v_new_2 paper executor.

CREATE TABLE IF NOT EXISTS v_spot_paper_positions (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER,                     -- FK to v_new_2_signals.id (informational; consumption tracked here)
    symbol VARCHAR(40) NOT NULL,
    side VARCHAR(8) NOT NULL DEFAULT 'long',
    is_meme BOOLEAN DEFAULT FALSE,
    tier VARCHAR(20),
    entry_time TIMESTAMPTZ NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    bgm_score DOUBLE PRECISION,
    position_size_pct DOUBLE PRECISION NOT NULL,    -- 50.0 means 50% of equity at entry
    notional_usd DOUBLE PRECISION NOT NULL,
    account_at_entry DOUBLE PRECISION,
    running_extreme DOUBLE PRECISION NOT NULL,
    profit_lock_active BOOLEAN DEFAULT FALSE,
    status VARCHAR(16) DEFAULT 'open',     -- 'open' | 'closed'
    lstm_regime VARCHAR(20),
    regime_meme_trail DOUBLE PRECISION,
    regime_nonmeme_trail DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_v_spot_positions_status ON v_spot_paper_positions(status);
CREATE INDEX IF NOT EXISTS idx_v_spot_positions_signal ON v_spot_paper_positions(signal_id);
CREATE INDEX IF NOT EXISTS idx_v_spot_positions_symbol ON v_spot_paper_positions(symbol, status);

CREATE TABLE IF NOT EXISTS v_spot_paper_trades (
    id SERIAL PRIMARY KEY,
    position_id INTEGER NOT NULL REFERENCES v_spot_paper_positions(id),
    symbol VARCHAR(40) NOT NULL,
    side VARCHAR(8) NOT NULL,
    is_meme BOOLEAN DEFAULT FALSE,
    tier VARCHAR(20),
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    exit_price DOUBLE PRECISION NOT NULL,
    bars_held INTEGER,
    exit_reason VARCHAR(40),
    bgm_score DOUBLE PRECISION,
    position_size_pct DOUBLE PRECISION NOT NULL,
    pnl_pct DOUBLE PRECISION NOT NULL,
    pnl_usd DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_v_spot_trades_exit_time ON v_spot_paper_trades(exit_time DESC);
CREATE INDEX IF NOT EXISTS idx_v_spot_trades_symbol ON v_spot_paper_trades(symbol);

CREATE TABLE IF NOT EXISTS v_spot_paper_account (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    equity_usd DOUBLE PRECISION NOT NULL,
    open_positions INTEGER NOT NULL,
    realized_trades INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_v_spot_account_ts ON v_spot_paper_account(ts DESC);
