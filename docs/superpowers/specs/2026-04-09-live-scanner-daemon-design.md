# Live Scanner Daemon Design

**Date:** 2026-04-09
**Status:** Approved
**Purpose:** Real-time momentum scanner for paper trading — polls Gate.io every 60s, detects breakouts, runs ML filter, writes signals to PostgreSQL for dashboard display.

## Architecture

```
Gate.io REST API (every 60s)
  → Fetch all USDT pair tickers
  → Compare to rolling 15min candle history (in memory)
  → Detect volume+price breakouts (same logic as backtest)
  → Run pre-trained ML filter (LightGBM, loaded at startup)
  → Write signals >0.65 confidence to scanner_signals table
  → Dashboard auto-refreshes to show new signals
```

## Single File: `services/python/scripts/live_scanner.py`

### Startup
1. Load pre-trained LightGBM model from disk
2. Fetch 15min kline history for all Gate.io USDT pairs (last 200 bars each)
3. Store in memory as rolling buffer per coin
4. Connect to PostgreSQL

### Main Loop (every 60s)
1. Fetch all tickers from Gate.io REST API (one call)
2. For each coin with price/volume change:
   - Update rolling buffer with latest data
   - Every 15 minutes (on candle close): run breakout detection
   - If breakout detected: extract ML features, run filter
   - If ML confidence > 0.65: write to scanner_signals table
3. Log: "Scanned 300 pairs, 2 breakouts detected, 1 passed ML filter"
4. Sleep until next 60s cycle

### Gate.io API Endpoints
- `GET /api/v4/spot/tickers` — all pairs, one call, no auth needed
- `GET /api/v4/spot/candlesticks?currency_pair=BTC_USDT&interval=15m&limit=200` — historical candles per pair (on startup only)

### ML Model
- Pre-trained LightGBM saved as .txt file during backtest
- Loaded once at startup with `lgb.Booster(model_file=...)`
- Uses same 19 features as backtest ML filter
- Threshold: 0.65

### Rolling Buffer
- Dict of {symbol: DataFrame} with last 200 15min candles per coin
- Updated every 15 minutes from Gate.io kline endpoint
- On startup: bulk fetch all pairs' recent history

### Error Handling
- API timeout: retry once, skip cycle
- API rate limit: back off 5 seconds
- DB connection lost: reconnect with retry
- Any crash: log error, continue loop (never exit)

### Database
- Writes to existing `scanner_signals` table
- Fields: symbol, direction, ml_prob, vol_ratio, price_change, signal_time, status='active'

### Logging
- Console + file log to `logs/live_scanner.log`
- Every cycle: "Cycle #N: scanned X pairs, Y breakouts, Z signals"
- On signal: "SIGNAL: PEPE LONG ml=0.78 vol=5.2x price=+4.1%"
