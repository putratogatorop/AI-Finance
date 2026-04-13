# Short Layered Exit Simulation Design

**Date**: 2026-04-10
**Status**: Approved
**Problem**: Current scanner-short uses simple MFE/MAE extremes with 1% SL — can't simulate price path, so SL hunters stop out 1,496 trades that would have been profitable (MFE >= 3%).

## Solution

Bar-by-bar simulation engine with layered exits (multiple SL/TP/trailing stop levels), grid sweep to find optimal parameters.

## Bar-by-Bar Simulation Engine

For each SHORT signal from `volume_breakouts`:
1. Load next 96 bars (24h of 15m candles) from raw CSV at `data/raw/15m/{symbol}/`
2. Walk bar-by-bar applying layered exit logic

### Position Structure

3 layers with fixed weights: [25%, 50%, 25%] (tight / core / runner).

Each layer has 4 parameters:
- **SL %**: stop loss level (price goes UP against short by this %)
- **TP %**: take profit level (price goes DOWN in favor by this %)
- **Trail activation %**: unrealized profit threshold to activate trailing stop
- **Trail distance %**: how far from best price the trail sits

### Per-Bar Logic (for each open layer)

For SHORT positions, using each bar's high (worst) and low (best):

1. Compute bar high as adverse move: `adverse = (bar_high - entry) / entry`
2. Compute bar low as favorable move: `favorable = (entry - bar_low) / entry`
3. **SL check FIRST** (conservative): if `adverse >= layer_sl` → close layer at `-layer_sl`
4. **Trailing check**: if best_favorable_seen >= trail_activation, trailing is active. If `adverse_from_best >= trail_distance` → close layer at `best_favorable - trail_distance`
5. **TP check**: if `favorable >= layer_tp` → close layer at `+layer_tp`
6. Update best_favorable_seen for trailing
7. After 96 bars: close remaining layers at bar close price (time exit)

**SL checked before TP within same bar** — assumes SL hunter hits you first (conservative).

### Final PnL

`total_pnl = Σ(layer_weight × layer_pnl)` where each layer's PnL comes from its own exit.

### Exit Reason

- Dominant exit determines the trade's `exit_reason`: whichever layer type closed the most weight
- `exit_detail` column stores per-layer breakdown: "L1:tp+3.0 L2:trail+4.5 L3:time+1.2"

## Grid Sweep Optimization

### Parameter Ranges

| Param | Layer 1 (25%) | Layer 2 (50%) | Layer 3 (25%) |
|-------|--------------|--------------|--------------|
| SL % | 1, 1.5, 2 | 2, 3, 4 | 4, 5, 7 |
| TP % | 2, 3, 4 | 5, 6, 8 | 8, 10, 13 |
| Trail activate % | 1.5, 2, 3 | 3, 4, 5 | 5, 7, 9 |
| Trail distance % | 0.5, 1, 1.5 | 1, 1.5, 2 | 2, 3, 4 |

### Two-Phase Sweep

**Phase 1**: Sweep each layer independently (81 combos each, other layers use mid-range defaults). Pick top 5 per layer by PF.

**Phase 2**: Cross-layer sweep of top 5 × 5 × 5 = 125 combos. Total: ~370 simulations.

### Evaluation (1-trade/day subset)

Pick best ML probability per day, then evaluate:
- Primary: **Profit Factor** (must be > 1.5)
- Secondary: **Total PnL %**
- Tiebreaker: **Win Rate**

### Walk-Forward

Same as existing pipeline: 18mo train / 6mo test rolling windows. Grid finds sweet spot on train, final numbers from test.

## Script

**One file**: `services/python/scripts/backtest_short_layered.py`

Workflow:
1. Load SHORT signals from `volume_breakouts` (ML >= 0.90, whitelist)
2. For each signal, load raw 15m CSV → extract next 96 bars after signal_time
3. Phase 1: per-layer grid sweep (find top 5 per layer)
4. Phase 2: cross-layer sweep (125 combos)
5. Print best config + stats (all trades + 1/day)
6. Rebuild `ml_backtest_trades` table with winning config results
7. Scanner-short page reads same table — no frontend changes needed

## Database Changes

`ml_backtest_trades` table — add one column:
- `exit_detail` VARCHAR(100) — per-layer breakdown, e.g. "L1:tp+3.0 L2:trail+4.5 L3:sl-3.0"

All existing columns preserved. `exit_reason` = dominant exit type. `pnl_pct` = weighted sum of layers.

## Runtime

~5K signals × 96 bars × 370 combos = well under 5 minutes locally. No GPU needed.
