# Short Strategy v2 — Expert Panel Fixes

**Date**: 2026-04-10
**Status**: Approved
**Problem**: Short strategy PF=1.02, 74% SL hit rate. Expert panel identified: ML confidence inverted, shorts only work when BTC rising, 94% of SL hits in first hour, L1 (1% SL) is pure bleed.

## Fixes

All changes to `services/python/scripts/backtest_short_layered.py`.

### Fix 1: RSI Filter

Add `rsi_14 >= 50` to signal filtering in `load_signals()`. Don't short oversold coins — they bounce into SL hunters. RSI 70+ is ideal but RSI >= 50 keeps enough trade volume.

### Fix 2: BTC Regime Filter

Add `btc_ret_24bar > 0` to signal filtering in `load_signals()`. Shorts only work as mean-reversion plays in a rising BTC market (PF 1.34). When BTC is falling, short trades get squeezed (PF 0.80).

### Fix 3: Delayed Entry

Modify the simulation flow — don't enter immediately at signal candle close:

1. Look at the first 4 bars (1 hour) after signal
2. If any bar's high > signal_close * 1.01 (price rallied >1%), **skip the trade**
3. If no rally, enter at bar 4's close price
4. Simulate remaining 92 bars (not 96) with the delayed entry price

This eliminates 94% of SL hunter losses that happen in the first hour. Implemented in `attach_bar_arrays()` — the function already extracts bars, so it can check the first 4 and adjust entry/bars.

### Fix 4: Remove L1, Go 2-Layer (60/40)

Drop the tight layer (25% at 1% SL — 79% SL rate, pure bleed). New structure:

- **Layer A (60%)**: core — sweeps SL [2,3,4], TP [5,6,8], trail_act [3,4,5], trail_dist [1,1.5,2]
- **Layer B (40%)**: runner — sweeps SL [4,5,7], TP [8,10,13], trail_act [5,7,9], trail_dist [2,3,4]

Grid sweep adapts: Phase 1 sweeps 2 layers (81 each), Phase 2 cross = 5×5 = 25 combos. Much faster.

### Fix 5: ML Retraining — Deferred

Not implemented now. Fixes 1-4 are entry/exit improvements independent of the model. Reassess after seeing v2 results.

## Implementation

Modify `backtest_short_layered.py`:
- `load_signals()`: add RSI >= 50 and btc_ret_24bar > 0 filters
- `attach_bar_arrays()`: add delayed entry logic (check first 4 bars, adjust entry price and bar array)
- `LAYER_GRIDS`, `LAYER_WEIGHTS`, `DEFAULT_LAYERS`: change from 3 layers to 2 (60/40)
- `sweep_phase1()`: loop over 2 layers instead of 3
- `sweep_phase2()`: cross-product of 2 layers (5×5 = 25)
- Re-run full pipeline, results auto-populate scanner-short page

## Expected Impact

- RSI filter: removes oversold bounce traps
- BTC filter: removes bearish-regime trades (PF 0.80 → cut)
- Delayed entry: eliminates first-hour SL hunter kills
- 2-layer: stops L1 bleed, concentrates on layers that work

Trade count will drop significantly (maybe 200-400 from 1,006) but remaining trades should have much higher PF.
