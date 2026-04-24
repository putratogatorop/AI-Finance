# SUPERSEDED 2026-04-23

> **This research contains a look-ahead bias.** The `multi_bar_confirm`
> detector used `close.shift(-1)` (peeks 1 bar into the future), and all three
> signals entered at `open_[i]` while computing masks from `close[i]` (1-bar
> timing leak). Fixed and retired via:
>
> - Shared leak-free module: `services/python/src/ml/bigmover_signals.py`
>   with regression test in `services/python/tests/test_bigmover_signals.py`
> - Leak-free 3y reference: `backtest_bigmover_longshort_3y_leakfix.py`
>   (also re-runs self-check at startup)
>
> **Corrected headline numbers on the same 2026-04-24 snapshot** (net of
> 0.12% round-trip fee + 50 bps SL slippage, entry at open[i+1], classifier
> retrained on holdout shortened to end 2026-01-23 so the grid's OOT window
> is truly classifier-unseen):
>
> | Artifact | File |
> |---|---|
> | Leak-free 3y (6 variants) | `_longshort_3y_leakfix_summary_57bcc40d_20260423T162458Z.csv` |
> | Leak-free sizing sweep (36 variants) | `_sizing_sweep_summary_cd7102c0_20260424T034428Z.csv` |
> | Leak-free compounding (8 scenarios) | `_compounding_summary_cd7102c0_20260424T034500Z.csv` |
> | Leak-free 87-variant ML grid | `_grid_summary_bigmover_ml_v1_73760c74_20260424T035336Z.csv` |
>
> **All 6 long+short 3y variants PF 0.84-0.93. All 36 sizing variants PF 0.84-0.95.
> All 8 compounding accounts liquidate to ~$0 from $100 start. Zero of 87 ML
> grid variants clear the CLAUDE.md ship floor of PF_p5 >= 1.30 on a
> classifier-unseen 3-month OOT.** The family has no live edge.
>
> **Paper trading:** `BIGMOVER_ACCOUNTS` in
> `services/python/scripts/paper_executor.py` disabled on 2026-04-23 (each
> entry carries `"enabled": False`). Open bigmover positions drain naturally
> via SL / trail / 48h timeout; the check_and_close_trades loop continues to
> tick after disable. Do NOT re-enable without a clean backtest per
> CLAUDE.md's Tighter Backtest Standard (all 8 gates).
>
> The `PF 2.85`, `+3,035%`, `zero losing months`, `$2.8e+20 uncapped
> compounding` claims below are **invalid** — they were products of the
> look-ahead bias.

---

# Bigmover Signal Research — 2026-04-23 (SUPERSEDED)

## What this folder contains

Two research runs exploring the `bigmover` signal family (`baseline`, `price_accel_atr`, `multi_bar_confirm`) from `backtest_bigmover_tweaks_v1.py`.

### 1. 87-variant ML grid (short-only, 3-month OOT)

Script: `scripts/backtest_bigmover_ml_grid_v1.py`
Summary: `_grid_summary_bigmover_ml_v1_b5c7ce35_20260423T133532Z.csv`
Per-variant folders: `backtest_bigmover_ml_grid_v1_b5c7ce35_*/`

Dimensions tested: 3 signals × 7 ML thresholds × 2 sizing × 2 exits + 3 no-ML anchors = 87 variants.
Window: last 3 months of snapshot (2026-01-23 → 2026-04-23).

**Headline finding:** No-ML anchors beat every ML-filtered variant by total PnL.
The existing `bigmover_classifier.joblib` only matches ~11% of snapshot signals (DB feature rows
were populated with stricter criteria than the baseline detector). ML threshold sweep doesn't
lift PF_p5 enough to compensate for the 10× trade-count loss.

**Top 3 (by total PnL, filter `trades≥100 AND PF_p5≥1.30`):**
1. `price_accel_atr NO-ML fixed_10 trail_3`  → 706 trades, PF 2.03, PF_p5 1.67, +406% return, −12% DD
2. `multi_bar_confirm NO-ML fixed_10 trail_3`  → 732 trades, PF 2.05, PF_p5 1.69, +358% return, −11% DD
3. `baseline NO-ML fixed_10 trail_3`  → 764 trades, PF 1.81, PF_p5 1.53, +334% return, −22% DD

### 2. 6-variant 3-year long+short regime backtest

Script: `scripts/backtest_bigmover_longshort_3y.py`
Summary: `_longshort_3y_summary_b5c7ce35_20260423T142312Z.csv` (post-listing-gate)
Per-variant folders: `backtest_bigmover_longshort_3y_b5c7ce35_*/`

Dimensions: 2 winning signals × 2 directions + 2 regime-gated combos = 6 variants.
Window: full snapshot (2023-04-01 → 2026-04-24).
Sizing+exit held constant: `fixed_10 + trail_3`.

**BTC regime (3-class, daily SMA 50/200):**
- bull: close > SMA_50 > SMA_200 — 396 days (35%)
- bear: close < SMA_50 < SMA_200 — 156 days (14%)
- sideways: otherwise — 567 days (51%)

**Headline:**

| Variant | Trades | WR | PF | Return | MaxDD | Bull PF | Bear PF | Sideways PF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| price_accel_atr SHORT | 4,453 | 64% | 2.45 | +2,358% | −8.9% | 2.67 | 1.89 | 2.59 |
| price_accel_atr LONG | 5,316 | 61% | 2.22 | +2,503% | −9.3% | 2.62 | 2.04 | 1.99 |
| multi_bar_confirm SHORT | 4,107 | 66% | 2.75 | +2,343% | −6.2% | 2.76 | 2.16 | 3.07 |
| **multi_bar_confirm LONG** | 4,938 | 64% | **2.85** | **+3,035%** | −7.5% | **3.34** | 2.54 | 2.59 |
| price_accel_atr COMBO* | 3,002 | 64% | 2.45 | +1,654% | −5.1% | 2.71 | 1.95 | skipped |
| multi_bar_confirm COMBO* | 2,840 | 66% | **2.96** | +1,816% | −5.9% | **3.35** | 2.26 | skipped |

*COMBO = long only in bull + short only in bear + skip sideways.

## Caveats

### Survivorship bias (IMPORTANT)

The universe parquet (`universe_2026-04-24.parquet`) holds top-100 Gate.io futures by
`quote_volume_24h` **at 2026-04-24**. Coins that delisted, rug-pulled, or lost liquidity
between 2023-04 and 2026-04 are NOT in this universe and therefore NOT in the backtest.

The `listed_since` field is NULL for all 100 coins in this snapshot. As a partial mitigation,
the 3y backtest applies a per-coin listing gate: trades can only enter after the coin's first
snapshot candle + 3-day warmup. This catches late-listed coins (61 of 100 have first-candle-date
after 2023-04-21, most via bulk-ingestion on 2026-01-23). Impact: minor — only 10-25 trades
filtered per variant.

The **real** survivorship bias (delisted coins missing entirely) cannot be fixed from the current
snapshot — it requires re-ingesting Gate.io historical futures including delisted ones.

### Reality-adjusted expectations

- Expect live PF to be **40-60% lower** than backtest (crypto backtest/live slippage gap).
- Backtest PF 2.85 → expect live PF 1.6-1.9. Still above 1.30 ship floor.
- Flat sizing (`fixed_10`): notional is constant `$50 per trade` on `$100 starting equity × 10% × 5x leverage`.
  Real paper_executor compounds equity, which would scale both returns AND drawdowns.

### Zero-losing-month signature

The 4 unrestricted variants have 0 losing months across 37 months. The 2 combo variants have
1-2 losing months across 30 months (fewer months because SMA_200 requires 200 days of history).
This extreme consistency is partially explained by: short-only bias was profitable in 2026 Q1,
survivorship bias removes catastrophic longs, asymmetric trail-3 exit caps losses at -5% while
winners run freely. **Still an unusually smooth curve — do NOT interpret as guaranteed live.**

## Files in this folder

| File | Purpose |
|---|---|
| `_grid_summary_bigmover_ml_v1_*.csv` | 87-variant ML grid summary |
| `_longshort_3y_summary_*.csv` | 6-variant 3y summary |
| `backtest_bigmover_ml_grid_v1_*__<variant>/` | Per-variant outputs for ML grid |
| `backtest_bigmover_longshort_3y_*__<variant>/` | Per-variant outputs for 3y grid (includes `monthly_regime_summary.csv` and `regime_summary.csv`) |

## Paper-trading follow-up

Based on these results, the intended next step is wiring 3 signal variants × 2 directions =
6 new paper-trading accounts alongside the existing 5 `scanner_signals_v2` ML-threshold accounts:

- `bigmover_baseline_long`, `bigmover_baseline_short`
- `bigmover_multi_bar_confirm_long`, `bigmover_multi_bar_confirm_short`
- `bigmover_price_accel_atr_long`, `bigmover_price_accel_atr_short`

Implementation: new `scripts/live_scanner_bigmover.py` + new `scanner_signals_bigmover` DB table +
`paper_executor.py` extension to track 6 new independent portfolios. Target PR: separate from
this research commit.
