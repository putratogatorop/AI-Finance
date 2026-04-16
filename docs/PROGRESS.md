# AI-Finance Progress Tracker

## Project Status: Phase 1 Complete — Data Backfill In Progress

### Architecture
```
PostgreSQL (market)  ←──  Python (ML + scheduler)
       ↑                        ↓
  Next.js (dashboard)    Gate.io API (BTC 4h data)
```

---

## Phase 1: Infrastructure + Data Pipeline ✅
- [x] Docker Compose (PostgreSQL, Python, Next.js)
- [x] Config module + scheduler entry point
- [x] PostgreSQL ORM models (6 tables)
- [x] CSV data lake storage
- [x] Binance client (ccxt) — *blocked in Indonesia, using Gate.io*
- [x] CoinGecko client (fundamentals)
- [x] ETL pipeline (CSV → PostgreSQL, hourly → daily)
- [x] Scheduler jobs
- [x] Historical data backfill script
- [x] GitHub Actions CI

## Phase 2: ML Engine ✅
- [x] Feature engineering (technical + fundamental + market context)
- [x] Base model interface (multi-horizon: 7d/14d/28d/42d)
- [x] XGBoost model
- [x] LightGBM model
- [x] LSTM model (PyTorch)
- [x] Ensemble consensus (weighted average, confidence threshold 0.7)
- [x] Signal generation (BUY/SELL/HOLD/EXIT)
- [x] Risk management (8 max positions, 25% single asset, -8% stop-loss, 20% drawdown pause)
- [x] Backtesting engine (no-lookahead, stop-loss, metrics)
- [x] Model versioning (weekly retraining)
- [x] Integration test (end-to-end pipeline)
- **121 Python tests passing, 0 lint errors**

## Phase 3: Dashboard ✅
- [x] Next.js 14 + TypeScript + Tailwind (dark theme)
- [x] Prisma schema (6 tables)
- [x] Format utilities (IDR, USD, %, dates)
- [x] Layout (sidebar, header, responsive)
- [x] Components (PriceChart, PerformanceChart, SignalCard, PositionRow)
- [x] API routes (signals, portfolio, backtest, assets, audit, settings)
- [x] Pages: Overview, Signals, Portfolio, Backtest, Asset Detail, Audit, Settings
- [x] GitHub Actions CI (lint, type-check, test, build, Docker)
- **32 Jest tests passing**

---

## Phase 4: Data Backfill 🔄
**Focus: BTC only, 4-hour candles**

| Item | Status | Details |
|------|--------|---------|
| Data source | Gate.io | Binance blocked in Indonesia (SSL/403) |
| Timeframe | 4h candles | For daily/weekly swing trading |
| Date range | Mar 2024 → present | ~4,500+ candles |
| Raw CSV | `data/raw/btc_4h/YYYY-MM.csv` | Monthly files |
| PostgreSQL | `asset_prices_hourly` | 4h granularity |
| Daily aggregation | `asset_prices_daily` | Aggregated from 4h |
| Fundamentals | `asset_fundamentals` | CoinGecko |

### Run Backfill
```bash
cd services/python
python scripts/backfill_db.py
```
Script is resumable — re-running skips existing rows.

---

## Phase 5: First Backtest ✅ (baseline — needs tuning)

### Run 1 — 2026-04-01 (baseline, no tuning)
| Metric | Value | Assessment |
|--------|-------|------------|
| Total Trades | 15 | OK |
| Win Rate | 20% | Bad — need >45% |
| Total Return | -Rp 745,940 (-4.97%) | Losing money |
| Reward/Risk | 0.66 | Bad — need >1.5 |
| Sharpe Ratio | 0.02 | Near zero |
| Max Drawdown | -4.48% | Acceptable |
| Avg Win | +4.96% | OK |
| Avg Loss | -7.46% | Stop-loss dominated |
| Avg Hold | 18.9 days | Within target range |

**Key finding:** 12/15 trades hit -8% stop-loss (mostly Feb 2026 BTC crash from $76K→$63K). Model generated all BUY signals with avg confidence 0.58 — kept buying into a downtrend.

**Root causes:**
1. Model has no directional filter — predicts positive returns even in downtrends
2. Confidence threshold too low (0.5) — letting weak signals through
3. No trend filter — needs to avoid BUY signals when price < SMA50
4. All 19 signals were BUY, 0 SELL — model is BUY-biased

### Run 2 — Tuned (walk-forward, trend filter, cooldown)
| Metric | Run 1 | Run 2 | Change |
|--------|-------|-------|--------|
| Total Trades | 15 | 3 | Fewer, higher quality |
| Win Rate | 20% | 33% | +13% |
| Total Return | -4.97% | +2.05% | Now profitable |
| Reward/Risk | 0.66 | 5.19 | Excellent |
| Max Drawdown | -4.48% | -1.64% | Much safer |
| Avg Win | +4.96% | +10.03% | Better entries |
| Avg Loss | -7.46% | -1.93% | No more stop-loss wipeouts |

**Changes applied:**
- Walk-forward validation (retrain every 90 days, expanding window, 6 retrains)
- SMA20 trend filter (skip BUY when price < SMA20)
- 14-day cooldown between signals
- Direct model prediction aggregation (2/3 model agreement, avg return >1%)
- Max 1 concurrent position

**Assessment:** Profitable (+2.05%), but only 3 trades in ~14 months is too few. The strategy is very conservative -- only triggers when all conditions align. Good risk control (max drawdown -1.64%).

**Next steps:**
- [ ] Reduce cooldown to 10 days to get more trades
- [ ] Lower avg return threshold from 1% to 0.5%
- [ ] Test with shorter hold periods (7d instead of 14d)
- [ ] Add more data (extend backfill to 2023)
- [ ] Consider adding ETH as second asset for more opportunities

## Phase 6: v2 Redesign 🔄 (Current)
See `docs/PLAN-V2.md` for full expert-informed redesign plan.

**Sprint 1 (next):**
- [ ] Backfill 4h candles for top 5 (ETH, SOL, XRP, BNB)
- [ ] Build baselines: Buy & Hold (M0) + SMA crossover (M1)
- [ ] New stationary feature set (kill fake features, add Tier 1)
- [ ] Reframe as classification: P(up >3% in 14d)

---

## Key Decisions
- **BTC only** for now (add more assets after system proves profitable)
- **4h candles** from Gate.io (Binance blocked)
- **No leverage**, fixed 1M IDR per trade
- **Model decides hold duration** (1-6 weeks)
- **-8% stop-loss**, max 8 positions, 20% drawdown pause

## Database: `market` (local PostgreSQL)
```
postgres:MySQL100%@localhost:5432/market
```

## Test Counts
| Suite | Tests |
|-------|-------|
| Python (pytest) | 121 |
| Next.js (jest) | 32 |
| **Total** | **153** |
