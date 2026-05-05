# v_new_2 — Paper-Deploy Status (2026-05-05)

## Phase A — Infrastructure (DONE)

✅ **DB migrations applied on VPS**
- 4 tables created: `v_new_2_signals`, `v_new_2_paper_positions`, `v_new_2_paper_trades`, `v_new_2_paper_account`
- Indexes on signal_time, taken, status
- SQL committed to `services/python/scripts/v_new_2_schema.sql`

✅ **BGM honest models on VPS**
- `/opt/ai-finance/services/python/models/v_new_2_bgm_long_honest/{model.joblib, meta.json}` (3.1 MB)
- `/opt/ai-finance/services/python/models/v_new_2_bgm_short_honest/{model.joblib, meta.json}` (2.3 MB)

✅ **Universe + category data on VPS**
- `coingecko_categories.json` — 26 memes, 33 AI tokens
- `universe_top100_membership.parquet` — 309 symbols by mcap rank
- `v_new_2_u4_universe.json` — 107 symbols (memes + AI + top-50)

✅ **VPS data pipeline mapped**
- `asset_prices_15m` table — fresh data through 2026-05-05 15:30 UTC
- 382 distinct assets, 3.8M bars
- `v_new_1_shadow_score.py` already computes the 44 base features — REUSABLE

---

## Phase B — Build (REMAINING WORK)

### B1. Scanner — `v_new_2_live_scanner_prod.py` (TODO)

Adapt `v_new_1_shadow_score.py` pattern:
1. Load 15m candles from `asset_prices_15m`, resample to 4h
2. Filter to U4 universe (107 symbols from JSON)
3. Compute v_new_1 base 44 features (reuse `compute_per_coin_features()` from shadow_score.py)
4. Compute v_new_2 features (daily EMA20/50, cycle high/low, pullback_pct, pullback_atr, atr14_pct)
5. Per-tier rule trigger detection (current ATR config + meme 20% override)
6. Score with BGM (long_honest, short_honest models)
7. Write to `v_new_2_signals` if BGM ≥ 0.60

Cron: every 4h, similar pattern to existing shadow_score cron.

### B2. Paper Executor — `v_new_2_paper_executor.py` (TODO)

```python
# Polls v_new_2_signals every 60s for new signals (taken=FALSE)
# For each new signal:
#   - Check open position count vs max_concurrent (start at 2 for $20 phase)
#   - If room: open paper position
#     * Fetch current Gate.io price (REST API)
#     * Insert v_new_2_paper_positions row, mark signal taken=TRUE
#     * Set running_extreme = entry_price
# 
# Every 4h (after candle close):
#   For each open position:
#     - Fetch latest 4h close from asset_prices_15m
#     - Update running_extreme
#     - Check exit conditions:
#       * If is_meme: pl_20 + 5×ATR trail (winner from sweep)
#         * Activate profit_lock if pnl >= 20%
#         * Exit on 5×ATR adverse from running_max
#       * Else: default ATR trail + pl 30% → trend-flip
#       * Hard timeout: 90 days
#     - On exit: insert v_new_2_paper_trades, mark position closed
# 
# Every hour: snapshot equity to v_new_2_paper_account
```

Position sizing for $20 phase:
- account_balance = $20 (paper)
- risk_per_trade = 5% = $1
- leverage = 10×
- notional_per_trade = $10 (meets Gate.io min order on most coins)
- max_concurrent = 2

### B3. Cron + monitoring (TODO)
- Cron: scanner every 4h
- Cron: executor health check every hour
- Alerting: email/telegram on critical errors

### B4. Dashboard `/paper-v_new_2` (TODO, lower priority)
- KPI strip: equity, open positions, realized trades, PF, WR
- Equity curve chart (from v_new_2_paper_account)
- Open positions table
- Last 50 closed trades

---

## Deployment sequence (when ready)

```
Day 1 (today done):    DB migrations, BGM models, universe — all on VPS

Day 2:                 Build B1 (scanner). Run dry-run mode for 24h. Validate
                       signals fire at expected rate (~50-100/week).

Day 3:                 Build B2 (executor). Test in dry-run (logs but doesn't
                       open positions).

Day 4:                 Enable real paper trading with $20 paper balance.
                       Monitor for 48h.

Day 5-7:               Build B4 (dashboard) while paper data accumulates.

Day 7:                 Decision: $20 → $0-30 outcome. Diagnose if loss > 50%.
                       Otherwise continue 30-day paper validation.

Day 30:                Full decision gate. Real-money scaling decision.
```

---

## Critical paths to verify before $20 deploy

1. **Slippage estimate validation**: Compare backtest closes to actual Gate.io
   live mid-prices for 5 random signals. If diff > 0.30% on average, raise
   slippage assumption in backtest.

2. **Funding-rate magnitudes**: Pull Gate.io funding history for top-10 of
   our universe. Verify our 0.02%/8h LONG drag assumption.

3. **Min order size check**: For each coin in U4 universe, fetch Gate.io
   contract spec. Confirm $10 notional meets min for at least 80% of universe.

4. **Liquidation distance**: At 10× leverage, ensure stop is placed before
   liquidation price with 30%+ buffer. Add 1.5× ATR safety margin.

---

## Honest expectation for $20 phase

| Outcome | Probability | Action |
|---|---|---|
| $20 → $30+ | 20% | scale to $100, increase concurrent |
| $20 → $15-25 | 50% | strategy validated, top up |
| $20 → $5-15 | 25% | need tuning, diagnose what diverged |
| $20 → $0 | 5% | catastrophic — bug or strategy fails |

**Tuition mindset.** $20 is learning expense, not investment. Real lessons
emerge from real fills, real funding, real psychology.

---

## Final v_new_2 strategy config (for reference)

**Universe:** U4 = memes + AI + top-50 (107 symbols)

**Entry rules per tier:**
- top25 (BTC, ETH, SOL...): 2× ATR pullback (long), 4× ATR rise (short)
- 26-50 (large alts):       15% pullback (long), 10% rise (short)
- 51-100 (mid caps):         3× ATR pullback (long), 4× ATR rise (short)
- 101-200 (small caps):      4× ATR pullback (long), 4× ATR rise (short)
- **MEME OVERRIDE**:         **20% pullback** (long), **20% rise** (short)
                            applies to coins in CG memes category regardless of tier

**BGM filter:** v_new_2_bgm_{long,short}_honest at threshold 0.60

**Exit rules:**
- Memes:    5×ATR trail (no profit-lock, no fixed TP) — let parabolic run
- Non-memes: default 2-3× ATR trail + profit-lock at +30% → switches to trend-flip-only

**Hard timeout:** 90 days max hold

**Position sizing ($20 phase):**
- 5% per trade ($1) at 10× leverage = $10 notional
- Max 2 concurrent positions
- Hard kill at $10 account, full stop at $5

**Position sizing ($100+ phase):**
- 1% per trade at 5× leverage
- Max 5 concurrent positions
- Standard ATR-based sizing

**Expected backtest CAGR (held-out 2026 Q1):** +184%/yr at 5× lev with -3% DD.
**Realistic live CAGR after frictions:** 100-130%/yr at 5× lev.
