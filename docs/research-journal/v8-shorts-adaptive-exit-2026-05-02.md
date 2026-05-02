# v8 Shorts Adaptive Exit Research (2026-05-02)

## TL;DR

Applied the same `cls_score` filter sweep (Phase 1) from the 2026-05-01 long-detector study to the two short detectors.

| Detector | Best T | Hold Calmar baseline | Hold Calmar filtered | Improvement | Ship? |
|---|---|---|---|---|---|
| `macd_pullback_short` | **0.35** | 24.16 | **40.09** | **+65.9%** | ✅ YES |
| `macd_early_trend_short` | 0.45 | 10.79 | 11.08 | +2.7% | ❌ NO |

---

## Methodology

Single-round sweep following the streamlined plan (no multi-agent debate). Two new matrix builders cloned from `build_macd_pullback_long_3y_per_trade_matrix_v1.py`:
- `build_macd_pullback_short_3y_per_trade_matrix_v1.py`
- `build_macd_early_trend_short_3y_per_trade_matrix_v1.py`

**Snapshot**: `2026-04-01` (3-year candle window, top 100 USDT futures).
**Hold-out**: `2026-01-01 → 2026-04-01` (~3 months).
**Cell pool**: 30 cells (15m × 2 SL × 5 TP, 1h × 2 × 5, 4h × 2 × 5).
**Live cell**: `pnl_15m_2_6` (15m ATR, SL=2, TP=6 — matches deployed E2 exit).

---

## macd_pullback_short

**Detector AUC (hold-out)**: 0.584

**Trade counts**: 2,927 total; 211 in hold-out. 387 had no classifier features (NaN score, excluded by filter).

### Baseline (no filter, live cell)

| Split | n | PF | MDD% | Total PnL% | Calmar |
|---|---|---|---|---|---|
| All | 2927 | — | — | 3671.6 | — |
| Hold-out | 211 | 4.27 | 14.82 | 358.0 | 24.16 |

### Filter sweep (cell = `pnl_15m_2_6`)

| T | n (hold) | PF (hold) | PnL% (hold) | MDD% (hold) | Calmar (hold) |
|---|---|---|---|---|---|
| 0.25 | 117 | 3.69 | 160.8 | 9.10 | 17.67 |
| 0.30 | 104 | 4.28 | 160.9 | 6.18 | 26.04 |
| **0.35** | **92** | **5.82** | **169.6** | **4.23** | **40.09** |
| 0.40 | 79 | 5.30 | 138.0 | 4.23 | 32.63 |
| 0.45 | 65 | 6.09 | 120.0 | 3.86 | 31.08 |
| 0.50 | 60 | 6.78 | 113.6 | 3.86 | 29.43 |

**Winner**: T=0.35. Hold Calmar 24.16 → 40.09 (+65.9%). Hold MDD 14.82% → 4.23% (3.5× reduction). Hold PF 4.27 → 5.82.

**Best single cell at T=0.35** (train sweep): `pnl_15m_2_6` — the live cell is already the best. No cell change needed.

**2-cell variant**: tested. No variant beats 1-cell by ≥5% on hold Calmar with PF ≥ 2.0. Shipping 1-cell only.

### Decision rule application

| Check | Threshold | Result |
|---|---|---|
| Hold Calmar improvement | ≥ +50% | +65.9% ✅ |
| Hold PF at chosen T | ≥ 2.0 | 5.82 ✅ |
| 2-cell beats 1-cell | ≥ +5% Calmar | No — ship 1-cell |

**→ SHIP: `v8_macd_pullback_short_e2` floor = 0.35, cell = `pnl_15m_2_6` (unchanged).**

---

## macd_early_trend_short

**Detector AUC (hold-out)**: 0.671

**Trade counts**: 1,706 total; 102 in hold-out. 366 had no classifier features.

Note: `macd_early_trend_short` is a strict subset of `macd_pullback_short` (adds the ≤10d since bear-flip restriction), so it has ~60% the trade count.

### Baseline (no filter, live cell)

| Split | n | PF | MDD% | Total PnL% | Calmar |
|---|---|---|---|---|---|
| All | 1706 | — | — | 2390.5 | — |
| Hold-out | 102 | 3.17 | 10.83 | 116.8 | 10.79 |

### Filter sweep (cell = `pnl_15m_2_6`)

| T | n (hold) | PF (hold) | PnL% (hold) | MDD% (hold) | Calmar (hold) |
|---|---|---|---|---|---|
| 0.25 | 61 | 2.47 | 51.8 | 9.69 | 5.35 |
| 0.30 | 58 | 2.70 | 54.7 | 6.74 | 8.12 |
| 0.35 | 56 | 2.75 | 53.9 | 6.74 | 7.99 |
| 0.40 | 55 | 2.80 | 54.4 | 6.74 | 8.08 |
| **0.45** | **52** | **3.05** | **55.5** | **5.01** | **11.08** |
| 0.50 | 43 | 2.94 | 44.8 | 5.01 | 8.95 |

**Best T = 0.45**, hold Calmar 10.79 → 11.08 (+2.7%). Nowhere close to the +50% gate. The small hold-out (102 → 52 traded) makes any signal fragile; the filter throws away real edge without compensating.

### Decision rule application

| Check | Threshold | Result |
|---|---|---|
| Hold Calmar improvement | ≥ +50% | +2.7% ❌ |

**→ NOT SHIPPED. Leave `v8_macd_early_trend_short_e2` unfiltered.**

Why filtering fails here: the early-trend restriction already acts as a quality filter (trades must be within 10d of the MACD bear-flip). The detector selects naturally fresh setups, so the classifier's discriminative signal on top is weak. AUC is 0.671 (higher than pullback_short's 0.584), but that AUC is evaluated on the full detector population; within the early-trend window, the classifier doesn't separate winners from losers effectively enough to justify dropping 50% of trades.

---

## Implementation

Shipped via `paper_executor.py` `V8_CLS_SCORE_FLOOR` dict:

```python
V8_CLS_SCORE_FLOOR: dict[str, float] = {
    "v8_macd_pullback_long_e2": float(os.environ.get("V8_MACD_PULLBACK_LONG_CLS_FLOOR", "0.40")),
    "v8_macd_pullback_short_e2": float(os.environ.get("V8_MACD_PULLBACK_SHORT_CLS_FLOOR", "0.35")),
}
```

`macd_early_trend_short` is intentionally absent — future research should revisit only if there is evidence of AUC improvement after retraining on more data.

---

## Files

- Matrix scripts: `scripts/build_macd_pullback_short_3y_per_trade_matrix_v1.py`, `scripts/build_macd_early_trend_short_3y_per_trade_matrix_v1.py`
- Matrix outputs: `results/adaptive_exit_per_trade_matrix_short_v1_bf5c9489_20260502T125451Z/`, `results/adaptive_exit_per_trade_matrix_early_trend_short_v1_bf5c9489_20260502T125434Z/`
- Code change: `scripts/paper_executor.py` — `V8_CLS_SCORE_FLOOR` extended
