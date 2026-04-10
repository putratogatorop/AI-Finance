# Scanner Long — 3-Strategy Backtest Design

## Goal

Compare 3 entry strategies for catching long trend-following trades on coins that move +20%+. Determine which entry method captures the most upside with a fixed 3:1 R:R (stop -5%, target +15%).

## Context

The `big_movers` table (v2) contains 6,961 records where `move_pct > 0.2` AND `peak_price > start_price` — coins that moved +20%+ upward. The thesis: after a 5% confirmed move, there's still 15%+ remaining. Ride the trend.

## Strategies

### Strategy A: Volume-Triggered + Momentum Confirmation
- Scan 15m bars for volume spike >= 3x 20-bar average
- Confirm price already moved +5% from rolling 96-bar (24h) low
- Entry: next bar open after both conditions met
- Stop: -5% from entry | Target: +15% from entry

### Strategy B: Rolling Price Scan
- Scan for coins that moved +5% in last 32 bars (8 hours)
- Confirm volume: at least 1 bar with vol >= 2x average during the move
- Entry: next bar open after detection
- Stop: -5% | Target: +15%

### Strategy C: Piggyback on Volume Breakouts
- Query `volume_breakouts` table for long signals (direction=1, vol_ratio >= 2)
- Filter: price already up 5%+ in trailing 96 bars at signal time
- Entry: next bar open after signal
- Stop: -5% | Target: +15%

## Architecture

Single Python script: `services/python/scripts/backtest_scanner_long.py`

### Data Flow
1. Load big_movers from DB (move_pct > 0.2, direction = 1) — ground truth for which coins/periods to scan
2. For each big mover, load that coin's 15m CSV via `load_coin()`
3. Run all 3 strategies on the data window (move_start - 48 bars to move_peak + 96 bars)
4. Simulate bar-by-bar: check entry, track stop/target
5. Record results per strategy

### Exit Rules (all strategies)
- Stop loss: -5% from entry price
- Take profit: +15% from entry price (3:1 R:R)
- Timeout: 672 bars (1 week) — exit at market price
- Cooldown: 96 bars per symbol per strategy

### Output
- Console: summary table per strategy (trades, win rate, avg PnL, profit factor, avg bars held)
- Declare winner
- Save all trades to `scanner_long_backtest` DB table for dashboard use

### No-Lookahead Rules
- Entry only uses data available at that bar
- big_movers ground truth only selects which coins/periods to scan — strategies detect independently
- No walk-forward needed yet (strategy comparison, not ML)
