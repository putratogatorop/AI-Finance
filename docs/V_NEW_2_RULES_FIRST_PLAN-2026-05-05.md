# v_new_2 — Rules-First Architecture Plan

**Date:** 2026-05-05
**Status:** Plan locked, building now.
**Pivot context:** v_new_1.x BGM-only ceiling cannot deliver -30% DD / 10× / 500-1000%/yr (ceiling-find verdict). Pivoting from "BGM-as-primary-signal" to **rules-first → BGM-grades → RL-adjusts-later**.

---

## The architecture (final form)

```
┌────────────────────────────────────────────────────────┐
│  LAYER 1 — RULES (the signal generator)                │
│  Per-coin daily EMA20/50 trend + pullback detection    │
│  Hand-designed, deterministic, interpretable           │
│  → produces DISCRETE entry candidates per (sym, ts)    │
└────────────────────────────────────────────────────────┘
                          ↓ rule fires
┌────────────────────────────────────────────────────────┐
│  LAYER 2 — BGM (the signal grader)                     │
│  LightGBM trained on RULE-TRIGGER BARS ONLY            │
│  Predicts P(profitable) given features at trigger      │
│  → produces QUALITY-GRADED entry list (skip if low)    │
└────────────────────────────────────────────────────────┘
                          ↓ graded
┌────────────────────────────────────────────────────────┐
│  LAYER 3 — RL/LSTM (deferred)                          │
│  Position sizing, exit timing, regime adaptation       │
│  ONLY built after Rules+BGM proven live for 2-4 weeks  │
└────────────────────────────────────────────────────────┘
```

This phase (v_new_2) builds Layer 1 + Layer 2.

---

## Layer 1 — Rules (this iteration's focus)

### Trend filter (per coin)

Daily EMA20 vs Daily EMA50:
- **LONG-allowed:** Daily EMA20 > Daily EMA50
- **SHORT-allowed:** Daily EMA20 < Daily EMA50
- **Neutral:** EMAs within ε% of each other → no trade

Daily EMAs are computed by resampling 4h close → daily close (last close of UTC day), then EWMA. Broadcast back to 4h granularity (forward-fill within day).

### Pullback entry

For LONG (within an uptrend):
- Track `cycle_high` = highest close since the last EMA20/50 cross (true peak of current trend cycle)
- Pullback% = (cycle_high − current_close) / cycle_high
- Trigger: pullback% ≥ X (sweep X ∈ {10%, 15%, 20%, 25%})

For SHORT (within a downtrend):
- Track `cycle_low` = lowest close since the last EMA20/50 cross
- Rise% = (current_close − cycle_low) / cycle_low
- Trigger: rise% ≥ X (sweep X ∈ {10%, 15%, 20%, 25%})

### Exit

Three exit options (sweep separately):
1. **Trend flip** — exit when EMA20 crosses opposite direction
2. **Trail stop** — exit when close ≤ highest_close − 3×ATR_14 (long); mirror for short
3. **Combined** — whichever fires first
4. **+ Hard timeout** — 90 days max hold (catch-all safety)

NO fixed +X% target. Per user direction: let winners run for the full trend.

### Outputs from Layer 1

For each coin × timestamp where rule triggers:
- `direction` (long / short)
- `cycle_high` / `cycle_low` (reference levels)
- `pullback_pct` / `rise_pct` (depth)
- `days_in_trend` (how long since last EMA cross)
- `trend_strength` (EMA20 / EMA50 ratio)

These all become **features for Layer 2 BGM**.

---

## Layer 2 — BGM as quality grader

### Training data

Filter to RULE-TRIGGER BARS ONLY (much smaller, focused dataset). Estimated ~30-100K rows for top-200 universe over 2020-05 → 2024-12 training period.

### Features (proposal)

44 base v_new_1 features
+ 6 rule-derived features above (cycle_high, pullback_pct, days_in_trend, trend_strength, EMA20-EMA50 spread, etc.)
= **~50 features total**

### Target

Realized return AFTER trigger:
- LONG target = `forward_max_pct` over next 60 days
- SHORT target = `−forward_min_pct` over next 60 days
- Binary label: positive if return ≥ 10%

OR continuous regression on actual return — to be decided.

### Validation

Same OOS framework: 2025 H2 + 2026 Q1 (untouched by training).

### Decision

If BGM grading lifts rule-only PF by ≥ 0.10 (e.g., 1.50 → 1.60), promote BGM as Layer 2. Otherwise rules-only.

---

## Implementation order

1. **Build daily-EMA + cycle-high/low feature engineering** on existing top-200 close data → `data/v_new_1_v2/features_v_new_2.parquet` (Mac, ~10 min)

2. **Backtest rules-only**:
   - Strategy variants: pullback ∈ {10/15/20/25}% × exit ∈ {trend-flip / trail-3atr / combined}
   - Multi-OOS: 2025 H2 + 2026 Q1 separately
   - Output: per-variant PF, win rate, CAGR at 1×/3×/5×, max DD
   - **Establishes the rules-only baseline**

3. **Train Layer 2 BGM** on rule-trigger bars only:
   - Features: 50 (44 base + 6 rule-derived)
   - Target: forward-60d return ≥ 10%
   - Same train/cal/OOS split as v_new_1
   - Output: model + meta JSON

4. **Backtest rules + BGM grade**:
   - Same variants from step 2
   - Filter trades by BGM_score ≥ threshold (sweep threshold)
   - Compare to rules-only baseline
   - Output: lift table

5. **Verdict + decision**:
   - If rules+BGM ≥ 1.5× rules-only on PF → promote, build paper-deploy
   - Else → rules-only is the deployable system, BGM is dropped from this layer
   - Either way: write `V_NEW_2_VERDICT-{date}.md`

6. **Paper-deploy** (separate phase, after verdict):
   - Add v_new_2 scanner to scanner-* services on VPS
   - 2-4 weeks live observation
   - Only then evaluate Layer 3 (RL/LSTM)

---

## Compute estimates

| Step | Where | Time |
|---|---|---|
| Daily EMA + cycle features | Mac | 10 min |
| Rules-only backtest sweep | Mac | 30 min |
| BGM Layer 2 training | VPS | 1-2h |
| Rules+BGM backtest | Mac | 15 min |
| Verdict | Mac | 5 min |
| **Total to verdict** | | **~3-4h** |

---

## Promotion gates

Same multi-OOS framework as v_new_1.x:
- Rules-only PF ≥ 1.30 on EITHER 2025 H2 OR 2026 Q1
- Rules+BGM PF ≥ 1.50 on the harder OOS slice
- Max DD at 3× leverage ≤ -25%
- Trade count ≥ 200 across both OOS slices (statistical significance)

If gates pass → paper-deploy decision.
If gates fail on rules-only → architecture broken, deeper rebuild needed (e.g., trigger pattern beyond EMA, or different timeframe).

---

## What we explicitly defer

- **RL/LSTM** — only after rules+BGM has run live ≥ 2 weeks with real data
- **Narrative features (cohort, CG categories)** — already built in v_new_1.6_extras; add as Layer 2 features in next iteration
- **Catalyst features** (token unlocks, listings) — separate data infra, defer
- **Per-coin model** — single global model first; per-coin specialization only if signal warrants it

---

## What survives from v_new_1.x

- ✅ Phase 1+2 data pipeline (top-200, 2020-2026)
- ✅ Multi-OOS framework (2025 H2 + 2026 Q1)
- ✅ CV-fixed temporal CV protocol
- ✅ Optuna hyperparameter search
- ✅ Calibration + threshold-locking protocol
- ✅ Paper-executor + dashboard infrastructure
- ✅ v_new_1.6_extras parquet (cohort + CG features) — available for Layer 2 if needed
