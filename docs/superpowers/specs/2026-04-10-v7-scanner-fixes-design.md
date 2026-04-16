# v7 Scanner Fixes — Isolated A/B Experiment Design

Date: 2026-04-10
Status: Approved

## Context

The v6 momentum scanner (breakout detection + LightGBM ML filter + trailing stop) is unprofitable on aggregate OOS data:
- Raw scanner: 1674 trades, 37.4% WR, PF 0.82
- Best ML filter (@0.70): 61 trades, 50.8% WR, PF 0.95
- 69.5% of filtered trades exit via stop_loss
- R:R is ~1:1 (avg win 4.72% vs avg loss 4.30%)
- Only profitable in 1 of 3 walk-forward windows (trending market)

Four expert analyses identified 7 concrete fixes. This spec defines how to implement and test them in isolation.

## Architecture

Single new script: `services/python/scripts/backtest_scanner_v7.py`

- Imports shared functions from `backtest_momentum_scanner.py` (load_coin, compute_atr, detect_breakouts)
- Overrides `detect_breakouts` and `simulate_pullback_trade` with v7 versions that accept a `fixes: set[str]` parameter
- CLI: `--fixes impatience_stop,breakeven_stop,regime_gate` (comma-separated list, or `all`)
- Runs the full walk-forward ML backtest (reuses logic from `backtest_scanner_ml_filter.py`)
- Prints results in same format so we can directly compare

## The 7 Fixes

### Fix 1: `lower_thresholds` — Stop Buying Climax Candles

**Problem**: Signal fires on a 3.5% move with 4x volume. That bar IS the move — we're the latecomer.

**Change**:
- `VOL_MULT`: 4.0 -> 2.5
- `PRICE_THRESH`: 0.035 -> 0.015
- Add **multi-bar confirmation**: after the signal bar, require the next bar to also close in breakout direction with volume > 1.5x average. If not confirmed within 3 bars, discard signal.

**In code**: New `detect_breakouts_v7()` function that wraps the original with confirmation logic.

### Fix 2: `impatience_stop` — Kill Dead Trades Early

**Problem**: 69.5% of trades hit the full -3% stop. Most show they're dead within 2 hours.

**Change**: If unrealized PnL < +1.5% after 8 bars (2 hours) from entry, exit at market. This converts -3% losers into -0.5% to -1.5% losers.

**In code**: Add check inside the bar loop in `simulate_trade_v7()`:
```python
bars_held = i - entry_bar
if bars_held >= 8 and unrealized < 0.015 and not trail_active:
    exit_reason = "impatience"
    break
```

### Fix 3: `breakeven_stop` — Protect Green Trades

**Problem**: Trades that go +2% then reverse to hit -3% stop. The "went green then died" pattern.

**Change**: Once unrealized hits +2%, move stop to entry + 0.3% (covers fees).

**In code**: Add after unrealized calculation:
```python
if unrealized > 0.02 and not trail_active:
    be_stop = entry_price * (1.003 if direction == 1 else 0.997)
    if direction == 1:
        stop_price = max(stop_price, be_stop)
    else:
        stop_price = min(stop_price, be_stop)
```

### Fix 4: `regime_gate` — Don't Trade in Chop

**Problem**: Strategy only works in trending markets (Window 2: PF 1.94). In ranging markets (Windows 1, 3), breakouts are fakeouts.

**Change**: Compute BTC ADX(14) on daily-equivalent bars (96 x 15min). Only allow trades when ADX > 20. Below 20 = ranging market, sit out.

**In code**: Precompute BTC ADX array, check at signal time:
```python
btc_adx = compute_adx(btc_high, btc_low, btc_close, period=14*96)
# At signal time:
if btc_adx[btc_idx] < 20:
    continue  # Skip — market is ranging
```

Note: ADX computation uses 96-bar aggregation to approximate daily bars from 15min data. We compute directional movement indicators (+DI, -DI) over 14 * 96 = 1344 bars rolling, which is equivalent to 14-day ADX.

Actually, simpler: resample BTC to daily bars first, compute ADX(14) on daily, then map back to 15min bars by date.

### Fix 5: `tight_trail` — Lock More Gains

**Problem**: Trail locks only 30% of gains after +5%. A +10% move can fall back to +3% before trail triggers. Crypto spikes reverse fast.

**Change**:
- After +5%: lock 50% of gains (was 30%)
- After +10%: lock 60% of gains
- HWM tracks on high/low instead of close (catch actual peaks)

**In code**:
```python
# HWM on high/low
if direction == 1:
    if high[i] > hwm:
        hwm = high[i]
else:
    if low[i] < hwm:
        hwm = low[i]

# Tiered trail
if unrealized > 0.10:
    lock_pct = 0.60
elif unrealized > 0.05:
    lock_pct = 0.50
else:
    lock_pct = 0  # Not yet trailing

if lock_pct > 0:
    trail_active = True
    new_stop = entry_price + (hwm - entry_price) * lock_pct  # for longs
```

### Fix 6: `ml_overhaul` — Better Signal Classification

**Problem**: ML learns trivial "trade when BTC is up" rule. Noisy features (hour, day), binary labels lose trade quality info, only 3 walk-forward windows.

**Changes**:
- Drop features: `hour_of_day`, `day_of_week`, `price_change_2bar`, `vol_trend`
- Add features: `atr_norm_breakout` (breakout size / ATR), `dist_from_high` (% from 20-bar high)
- Switch from `LGBMClassifier` to `LGBMRegressor` on `pnl_pct`, threshold predictions at > 0
- Set `scale_pos_weight=1.7` if keeping classifier
- Reduce complexity: `num_leaves=7`, `n_estimators=100`, `min_child_samples=50`
- Switch to 12mo train / 3mo test windows (more windows for statistical significance)

### Fix 7: `breadth_filter` — Market-Wide Confirmation

**Problem**: Trading individual breakouts without knowing if the broader market supports momentum.

**Change**: Compute altcoin breadth — count how many of 200+ coins have positive 7-day return. Only trade when breadth > 60% (broad rally) or < 25% (broad selloff). If 35-55%, market is mixed — sit out.

**In code**: Precompute daily breadth array across all coins, check at signal time.

## Test Protocol

Each fix is tested in isolation first, then combined:

```bash
cd services/python

# Baseline (current v6)
python scripts/backtest_scanner_v7.py --fixes none

# Individual fixes
python scripts/backtest_scanner_v7.py --fixes impatience_stop
python scripts/backtest_scanner_v7.py --fixes breakeven_stop
python scripts/backtest_scanner_v7.py --fixes regime_gate
python scripts/backtest_scanner_v7.py --fixes tight_trail
python scripts/backtest_scanner_v7.py --fixes lower_thresholds
python scripts/backtest_scanner_v7.py --fixes ml_overhaul
python scripts/backtest_scanner_v7.py --fixes breadth_filter

# Best combinations
python scripts/backtest_scanner_v7.py --fixes impatience_stop,breakeven_stop
python scripts/backtest_scanner_v7.py --fixes impatience_stop,breakeven_stop,regime_gate
python scripts/backtest_scanner_v7.py --fixes all
```

Output format per run: trades, WR, PF, expectancy, monthly return, exit reason breakdown.

Results saved to `models/results/v7_experiment_results.json` for comparison.

## Success Criteria

- At least one combination achieves PF > 1.3 on aggregate OOS
- Stop loss rate drops below 50% (from 69.5%)
- R:R improves to > 1.5:1 (from ~1:1)
- Profitable in at least 2 of 3 walk-forward windows (from 1 of 3)

## Non-Goals

- No changes to data pipeline or coin selection
- No new model architecture (still LightGBM)
- No live trading integration yet
- No UI changes (update tournament JSON after results)
