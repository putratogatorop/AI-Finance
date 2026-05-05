# Ceiling-Find Plan — Find the BGM architectural ceiling before LSTM/RL

**Date:** 2026-05-04
**Status:** Plan locked, execution starting in parallel with v_new_1.5 train
**Goal:** Determine the actual achievable backtest CAGR for the BGM architecture under different (leverage × DD cap × universe) configs, BEFORE committing weeks of work to RL/LSTM. Specifically: find whether the **(-30% DD, 10× leverage, top-200 universe)** operating point clears 500%/yr in backtest. If it does, the BGM has the edge to defend; if not, the architecture is the bottleneck.

---

## Why this plan exists

User's logic (paraphrased): "Backtest already has bias. If even the *biased* backtest can't deliver 1000%/yr, something is structurally wrong, and stacking RL/LSTM on a broken base is wasted work."

I pushed back on the magnitude — properly-constructed backtests aren't 5-10× inflated, more like 20-50% — and the math: 1000%/yr CAGR over 4yr = 14,641× total vs current 2.06× (~7,000× gap). Under -20% DD lock, 1000%/yr is near-impossible. So we agreed:

- **New target operating point:** -30% DD cap, 10× leverage, top-200 universe.
- **Estimated ceiling at that point:** 500-800%/yr (my guess; sweep produces the real number).
- **Decision rule:** if ceiling ≥ 500%/yr → BGM has edge → proceed to v_new_2 LSTM. If 300-500%/yr → BGM partial → LSTM may close gap. If <300%/yr → architecture is bottleneck → pivot to different strategy class.

---

## Risk reality check

10× leverage + -30% DD cap on real $100 of capital:
- One bad year ≈ -30% (account → $70)
- Two bad years compounded ≈ -50% (account → $50)
- Three consecutive bad years could wipe the account entirely

Hitting 500-800%/yr CAGR demands accepting this volatility. Worth flagging once and moving on; user has chosen.

---

## What changes vs the original v_new_1 setup

| Parameter | Old | New (ceiling-find) |
|---|---|---|
| Universe | Top-100 USDT perps | **Top-200** |
| Leverage | 5× | **10×** (sweep also tests 5× / 8× / 15×) |
| Max yearly DD constraint | -20% | **-30%** (sweep also tests -20% / -40%) |
| Position sizing | Linear `score / threshold`, capped 3× | Aggressive concentration sweep: caps {2, 3, 5, 8} + tiered band variants |
| Exit | 7-day timeout, no SL/TP | Add parabolic-trail variant: trail by `max_close − 3× ATR` |
| Friction model | Same Gate.io fees / slippage | Unchanged |

---

## Compute reality

Phase 1+2 input data lives on **Mac, not VPS**:
- 2020-2022 OHLCV archive: 163 symbols on Mac under `services/python/data/binance_archive_2020_2022/raw/`
- 2023-2026 candle snapshots: in `data/snapshots/`
- VPS has only top-100 derived parquets in `data/v_new_1/`, no raw OHLCV

So the top-200 expansion runs **on Mac**. After Phase 1+2 produces `data/v_new_1_top200/{features,labels,...}.parquet`, we SCP to VPS for scoring with v_new_1.5 model + ceiling-find Phase 4.

Phase 1+2 is pandas + parquet, not sustained ML training — M2 Pro handles it without thermal issues.

---

## Steps

1. **Patch `v_new_1_phase1_2_pipeline.py`** to accept `TOP_N` and `OUTPUT_DIR` env vars (default behaviour unchanged for top-100 reuse).
2. **Run Phase 1+2 with TOP_N=200** on Mac → `data/v_new_1_top200/` (~30-60 min wall-clock, IO-heavy + pandas).
3. **Wait for v_new_1.5 verdict** (~2.5h from now). If LONG or SHORT direction passed +5% lift, use the v_new_1.5 model for ceiling-find. Otherwise fall back to baseline-retrained v_new_1.
4. **SCP top-200 parquets to VPS.** Score the top-200 universe with the chosen model → `oos_scored_{long,short}_top200.parquet`. ~10 min on VPS.
5. **Build `v_new_1_phase4_ceiling_find.py`**: Optuna sweep over (leverage, DD cap, sizing, exit) on top-200 scored data. Output Pareto table.
6. **Run sweep** on Mac (data + script local; ~1-2h Optuna). Output `docs/V_NEW_1_CEILING_FIND_VERDICT-{date}.md`.
7. **Decision** per the rule above.

---

## Sweep grid (concrete)

| dim | values |
|---|---|
| `leverage` | 5, 8, 10, 15 |
| `dd_cap_pct` | None, -20, -30, -40 |
| `universe` | top-100 (existing), top-200 (new) |
| `top_k_long` | 5, 10, 15, 20, 30 |
| `top_k_short` | 5, 10, 15, 20, 30 |
| `score_threshold_long` | Optuna search [0.20, 0.45] |
| `score_threshold_short` | Optuna search [0.30, 0.60] |
| `position_size_pct` | Optuna search [0.3, 3.0] |
| `score_scale_cap` | 2, 3, 5, 8 |
| `exit_strategy` | timeout-7d (current), parabolic-trail-3atr |

Optuna objective: maximise 4-yr CAGR subject to per-year DD ≥ `dd_cap_pct`. Hard fail any config that violates DD in any OOS year.

---

## Deliverables

- `services/python/results/ceiling_find_2026-05-04/pareto_table.csv` — every (leverage, DD, universe, exit) combo with measured CAGR + max DD + per-year breakdown.
- `services/python/results/ceiling_find_2026-05-04/best_per_dd_cap.csv` — top config per DD cap.
- `docs/V_NEW_1_CEILING_FIND_VERDICT-{date}.md` — headline ceiling at the -30%/10×/top-200 target, plus the decision (LSTM-go vs architecture-pivot).

---

## What this plan replaces

This **supersedes** the LSTM/RL prioritisation in `V_NEW_2_LSTM_RL_REPLAN-2026-05-04.md` until the ceiling is measured. The original v_new_1.5 train (in flight) still runs — we want a clean BGM with sequence features + CV-fix as the model used for ceiling-find. After the verdict, LSTM/RL only proceeds if the ceiling rule says BGM has measurable edge to defend.
