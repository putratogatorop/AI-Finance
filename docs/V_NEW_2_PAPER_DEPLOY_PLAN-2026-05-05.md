# v_new_2 — Paper-Deploy Plan

**Date:** 2026-05-05
**Status:** Plan locked. Code scaffolding starting.
**Goal:** Wire v_new_2 strategy (memes + AI + top-50, BGM thr 0.60, 5× lev, 5 concurrent) to the existing paper_executor on VPS for 30-day forward validation.

---

## Why paper-deploy now

The council audit flagged 5 critical issues; we fixed 4 in code (fixed-horizon labels, funding/slippage, locked holdout, concurrent simulator). The 5th — **forward-OOS reality** — can ONLY be validated by paper trading on real fills with real data. No more in-sample optimization helps.

Backtest claim: U4 at 5×/0.60/mc=5 → 407%/yr CAGR
Realistic live expectation: 150-300%/yr after forward decay + execution drag
Abort signal: paper PF < 1.3 over 30 trades → strategy isn't real

---

## Architecture overview

```
┌──────────────────────────────────────────────────────────┐
│ INGEST (existing service on VPS)                         │
│ Writes 4h candles to Postgres every 4h                   │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ scanner-v_new_2 (NEW service)                            │
│ Every 4h: read top-200 closes, compute features:         │
│   - daily EMA20/50 from 4h close                         │
│   - cycle high/low since last EMA cross                  │
│   - pullback_pct, pullback_atr, rise_pct, rise_atr       │
│   - ATR_14_pct                                           │
│ For each (symbol, current bar): if rule fires (in U4 universe):
│   - score with BGM_long or BGM_short (thr ≥ 0.60)        │
│   - write to v_new_2_signals table                       │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ paper-executor-v_new_2 (NEW service)                     │
│ Reads new signals, opens paper positions:                │
│   - max 5 concurrent                                     │
│   - 1% paper-equity risk per trade × 5× lev = 5% notional│
│   - records entry_price, BGM score                       │
│ Continuously monitors open positions:                    │
│   - exits on ATR-trail (2× ATR for short, 3× for long)   │
│   - profit-lock at +30%: switches to daily EMA flip exit │
│   - 90-day max hold                                      │
│ Writes closed trades to v_new_2_paper_trades table       │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ /paper-v_new_2 (NEW dashboard page on Vercel)            │
│ Shows: equity curve, open positions, trade log, win rate,│
│ realized PF, comparison to backtest expectations         │
└──────────────────────────────────────────────────────────┘
```

---

## Component checklist

### 1. Database schema (Postgres on VPS)

```sql
-- v_new_2_signals: one row per rule-trigger (filter to U4 + thr 0.60)
CREATE TABLE IF NOT EXISTS v_new_2_signals (
  id            SERIAL PRIMARY KEY,
  signal_time   TIMESTAMPTZ NOT NULL,
  symbol        VARCHAR(20) NOT NULL,
  side          VARCHAR(10) NOT NULL,
  pullback_pct  DOUBLE PRECISION,
  pullback_atr  DOUBLE PRECISION,
  bgm_score     DOUBLE PRECISION NOT NULL,
  d_ema_spread  DOUBLE PRECISION,
  taken         BOOLEAN DEFAULT FALSE,
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(signal_time, symbol, side)
);
CREATE INDEX ix_v_new_2_signals_time ON v_new_2_signals(signal_time DESC);
CREATE INDEX ix_v_new_2_signals_taken ON v_new_2_signals(taken) WHERE taken = FALSE;

-- v_new_2_paper_positions: open positions
CREATE TABLE IF NOT EXISTS v_new_2_paper_positions (
  id            SERIAL PRIMARY KEY,
  signal_id     INTEGER REFERENCES v_new_2_signals(id),
  symbol        VARCHAR(20) NOT NULL,
  side          VARCHAR(10) NOT NULL,
  entry_time    TIMESTAMPTZ NOT NULL,
  entry_price   DOUBLE PRECISION NOT NULL,
  bgm_score     DOUBLE PRECISION,
  position_size_pct DOUBLE PRECISION,    -- e.g. 0.01 for 1% risk
  leverage      DOUBLE PRECISION,         -- e.g. 5.0
  running_max_close DOUBLE PRECISION,    -- updated each bar
  profit_lock_active BOOLEAN DEFAULT FALSE,
  status        VARCHAR(20) DEFAULT 'open',  -- open | closed
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- v_new_2_paper_trades: closed trades
CREATE TABLE IF NOT EXISTS v_new_2_paper_trades (
  id            SERIAL PRIMARY KEY,
  position_id   INTEGER REFERENCES v_new_2_paper_positions(id),
  symbol        VARCHAR(20) NOT NULL,
  side          VARCHAR(10) NOT NULL,
  entry_time    TIMESTAMPTZ NOT NULL,
  exit_time     TIMESTAMPTZ NOT NULL,
  entry_price   DOUBLE PRECISION NOT NULL,
  exit_price    DOUBLE PRECISION NOT NULL,
  bars_held     INTEGER,
  exit_reason   VARCHAR(40),
  pnl_pct       DOUBLE PRECISION NOT NULL,   -- per position
  realized_at   TIMESTAMPTZ DEFAULT NOW()
);
```

### 2. Scanner service (`services/python/src/scanners/v_new_2.py`)

- Runs every 4h via supervisor / docker-compose
- Reads last 600 bars (100 days) per symbol from candles table
- Computes daily EMA + cycle features (mirror v_new_2_features.py logic)
- For each symbol: check if pullback rule fires THIS bar
- If yes: load BGM model from disk, score, write signal if ≥ 0.60
- Filter to U4 universe (memes + AI + top-50)

Universe defined statically: 107 symbols (committed JSON file)

### 3. Paper executor (`services/python/src/executors/v_new_2_paper.py`)

- Polls v_new_2_signals every 1 min for new signals (taken=FALSE)
- Opens position if max_concurrent not exceeded
- Monitors open positions every 4h (after candle close):
  - Update running_max_close
  - Activate profit_lock if pnl_pct ≥ 30%
  - Compute trail price; exit if breached
  - Exit if profit_lock_active AND daily EMA flips
  - Exit if 90-day hold elapsed
- On exit: write to v_new_2_paper_trades, mark signal+position closed

### 4. Dashboard page (`services/nextjs/src/app/paper-v_new_2/page.tsx`)

- KPI strip: equity, open positions, realized trades, PF, WR
- Equity curve chart
- Open positions table
- Last 50 closed trades
- Comparison panel: realized vs backtest claim (407%/yr CAGR)

### 5. Docker compose service

Add to `docker-compose.yml`:
```yaml
scanner-v-new-2:
  build: services/python
  command: python -m src.scanners.v_new_2 --interval 14400
  env_file: .env
  depends_on: [postgres]
  restart: unless-stopped

paper-executor-v-new-2:
  build: services/python
  command: python -m src.executors.v_new_2_paper --interval 60
  env_file: .env
  depends_on: [postgres]
  restart: unless-stopped
```

---

## Deployment sequence

**Day 1 (today):**
1. Write scanner + executor scaffolds (this session)
2. Commit + push
3. SCP BGM model files to VPS
4. Apply DB migration (CREATE TABLE statements above)

**Day 2:**
5. Start scanner-v-new-2 service in dry-run mode (writes signals, doesn't open positions)
6. Validate signals match expected behavior on a few bars
7. Enable paper executor

**Days 3-7:**
8. Daily check: signals firing, positions opening/closing, PnL recorded
9. Build /paper-v_new_2 dashboard page
10. Compare realized fills to backtest expectations

**Days 8-14:**
11. Resist tweaks. Accumulate forward data.
12. End of week 2: first checkpoint — 30 trades, PF check.

**Day 30:**
13. Final paper-trade verdict.
14. If PF ≥ 1.3 AND DD < 10% AND no critical bugs → real money decision.
15. Else → diagnose what broke.

---

## Abort criteria

If ANY trigger during paper:

1. **Realized PF < 1.0 over 30+ trades** — strategy doesn't have edge in live market
2. **Slippage + funding > 1% per round-trip** — frictions exceed our model
3. **Max DD > 10% in any week** — risk is bigger than backtest predicted
4. **BGM precision < 60% in top decile** (vs 90% claimed) — model breaks forward
5. **Critical bug** in scanner or executor (signal not firing, wrong entry price, etc.)

---

## What gets deferred

- **RL/LSTM Layer 4** — only after 30 days of clean paper data
- **Real money deployment** — only after 30-60 days paper validation + 1 month at small scale
- **Portfolio-level optimization** (correlation between positions, cross-strategy hedging) — phase 2
- **Mega-trend mode** — keep dormant; activate only if BTC monthly+daily turn bullish
