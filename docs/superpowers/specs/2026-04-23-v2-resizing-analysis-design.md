# v2+ML Resizing Analysis Design

**Date**: 2026-04-23
**Status**: Approved — ready for writing-plans
**Problem**: v2+ML scanner produces strong raw edge (PF 6.14 net / 210% over 3-mo OOT at fixed $600 sizing), but current flat sizing (25% notional per trade) ignores ML probability strength and concurrency cap drops signals in cluster months. Question: how much PnL can be unlocked with **same model, same trades, different capital allocation**?

## Goal

Quantify the PnL lift attributable to two sizing changes, with clean attribution:

- **Lever 1 — Kelly-fractional sizing by ml_prob bin**: bigger positions on higher-confidence signals
- **Lever 4 — Concurrency cap 5 → 10**: stop dropping real signals during clustered bursts

## Non-Goals

- No new features, no new model, no feature engineering changes
- No leverage increase (stays at 5×)
- No compounding — all three runs use fixed $600 starting equity
- No TP/SL retuning (stays at 6.5% / 1% audit defaults)
- Not a deploy decision — this is research. Ship floor does not apply.

## Layered Three-Run Attribution

Single trade stream, three sizing replays. Layered so each run adds exactly one lever:

| Run | Sizing | Concurrency | Purpose |
|-----|--------|-------------|---------|
| **A (baseline)** | Flat 25% notional | 5 | Anchor. Reproduces the ~+210% / 3-mo result |
| **B (+Lever 1)** | Half-Kelly by ml_prob bin | 5 | Isolates sizing-by-confidence lift |
| **C (+Lever 1+4)** | Half-Kelly by ml_prob bin | 10 | Adds concurrency unlock on top of B |

Attribution:
- `B - A` = pure sizing lift
- `C - B` = pure concurrency lift
- `C - A` = combined lift

## Input Data

### Source

`scanner_short_v2_ml_filtered` (the dredged-threshold 0.70 OOS set, 1132 trades, 2025-03 → 2026-04). This matches the table the audit used and the monthly-PnL computation we ran in conversation.

**Note**: We use 0.70 threshold (dashboard) NOT 0.75 (audit-honest). Rationale: the goal is to show the PnL envelope of the current production model as-is, not to re-validate thresholds. A separate follow-up can rerun this analysis on 0.75.

### Window

Last 3 calendar months (`signal_time >= max_date - 3 months`). Identical to the audit OOT window for apples-to-apples comparison.

### Columns needed

- `signal_time` — entry timestamp
- `ml_prob` — signal confidence (for Kelly binning)
- `pnl_pct` — realized gross PnL (fees subtracted by us)
- `bars_held` — to derive `exit_time = signal_time + bars_held × 15min`

**Risk**: If `scanner_short_v2_ml_filtered` doesn't carry `bars_held`, fall back to joining `scanner_short_v2` raw trades on `signal_id`. Script must verify schema on load and error cleanly.

## Kelly Sizing Implementation

### Binning

Bin ml_prob into 5 discrete buckets:

- `[0.70, 0.75)`
- `[0.75, 0.80)`
- `[0.80, 0.85)`
- `[0.85, 0.90)`
- `[0.90, 1.00]`

### Per-bin Kelly fraction

For each bin, compute empirical stats from **pre-OOT data only** (never touch OOT for calibration):

- `p_bin` = win rate in bin on pre-OOT
- `b_bin` = avg winner PnL / avg |loser PnL| in bin on pre-OOT

Kelly fraction per bin:

```
f_bin = p_bin − (1 − p_bin) / b_bin
f_bin_half = 0.5 × f_bin   # half-Kelly, industry standard, hardcoded
f_bin_capped = clip(f_bin_half, 0.02, 0.15)  # [2%, 15%] equity
```

### No dredging

- The 5-bin boundaries are declared upfront (not swept)
- Half-Kelly multiplier is hardcoded (not tuned)
- [2%, 15%] cap is hardcoded
- Kelly table is computed ONCE on pre-OOT and applied to OOT. No OOT-informed adjustment.

### Notional per trade

`notional = equity × kelly_fraction × leverage` where `equity = $600` (fixed), `leverage = 5×`.

Example: a 0.92 ml_prob signal in the [0.90, 1.00] bin where empirical Kelly says 10% → `$600 × 0.10 × 5 = $300 notional` vs baseline $150.

## Concurrency Simulation

Sort trades by `signal_time`. Maintain an `open_positions: list[exit_time]`.

Per trade:

1. Prune `open_positions`: remove any exit_time ≤ current signal_time
2. If `len(open_positions) >= MAX_CONCURRENT`: skip this trade, log to `skipped.csv` with reason `concurrency_cap`
3. Else: take trade, append `signal_time + bars_held × 15min` to `open_positions`

`MAX_CONCURRENT` = 5 for runs A and B, 10 for run C.

## Outputs

Following `backtest-protocol.md`, written to `results/<run_id>/` via `write_results()`:

- `metrics.json`:
  ```json
  {
    "scenario_A": {"trades": N, "wr": ..., "pf_net": ..., "total_return_pct": ..., "max_dd_pct": ..., "pf_mc_p5": ..., "pf_mc_p50": ..., "pf_mc_p95": ...},
    "scenario_B": { ... },
    "scenario_C": { ... },
    "attribution": {"lever_1_lift_pct": B-A, "lever_4_lift_pct": C-B, "combined_lift_pct": C-A}
  }
  ```
- `monthly_comparison.csv` — columns: `month, A_pnl_usd, A_ret_pct, B_pnl_usd, B_ret_pct, C_pnl_usd, C_ret_pct`
- `trades_A.csv`, `trades_B.csv`, `trades_C.csv` — per-scenario ledgers with derived columns `notional_usd, pnl_usd, equity_after`
- `kelly_table.csv` — the 5-bin table with p, b, f_raw, f_half, f_capped (auditable)
- `skipped_by_concurrency.csv` — signals dropped from A and C (B has same as A)
- `params.json` — all constants for reproducibility

## Tighter Backtest Standard Compliance

- ✓ **Hold-out**: Kelly table calibrated on pre-OOT only; sizing rule is deterministic after that
- ✓ **Walk-forward**: N/A — fixed deterministic sizing rule, no model training
- ✓ **Feature audit**: N/A — no features, no ML training here
- ✓ **Fees + slippage**: 0.12% round-trip per trade applied to `pnl_pct` before sizing math
- ✓ **Universe-time correct**: inherits from upstream v2+ML trades
- ✓ **Same model**: by construction — this is the identical model, only sizing changes
- ✓ **Monte Carlo**: bootstrap 1000× on trade order within each scenario, report PF p5/p50/p95
- ✗ **Ship floor**: N/A — this is research output, not a deploy spec

## Decision Framework

After the run:

| Outcome | Interpretation | Next step |
|---------|----------------|-----------|
| `B − A < 15%` | ML probabilities not well-calibrated — flat sizing is fine | Drop Kelly direction; explore calibration instead |
| `B − A > 30%` | Kelly is unlocking real sizing alpha | Productionize Kelly table into paper executor |
| `C − B > 20%` | Concurrency cap is actively hurting prod | Lift MAX_CONCURRENT in paper executor |
| `C − A > 80%` | Combined lift is a meaningful prod upgrade | Full writing-plans + deploy ticket |
| All low | Current sizing is near-optimal | Move on to alt-data direction |

## Script Structure

`services/python/scripts/backtest_v2_resized.py` — ~250 lines. Single main() with functions:

- `load_trades()` — read filtered trades from DB, validate schema, slice OOT window
- `build_kelly_table(pre_oot_df)` → DataFrame[bin, p, b, f_capped]
- `simulate(trades_df, kelly_table, max_concurrent, scenario_label)` → trade ledger + metrics dict
- `monte_carlo_pf(pnl_series, iters=1000)` → (p5, p50, p95)
- `main()` — orchestrates 3 runs, writes results

Dependencies: pandas, numpy, sqlalchemy — already in the project. No new installs.

## Rollback

Pure research script. No prod impact. "Rollback" = delete the `results/<run_id>/` folder and the script file.
