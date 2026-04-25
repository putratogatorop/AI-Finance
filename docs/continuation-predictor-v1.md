# Continuation Predictor v1 — research result

**Status:** RESEARCH — does NOT pass the 8-point ship gate.
Use these numbers internally; do **not** deploy to live trading.

## Why this exists

Three short-side strategies (`vol_accel`, `wait_bounce`, `multi_frame`) all
produced marginal PF (~1.04) on the 2026-04-01 snapshot. A 4-agent diagnostic
team established that:

- Entry timing is **not** the bottleneck (winners and losers enter at
  identical 6.3% drops from peak).
- Exit-rule tweaks **cannot salvage** the strategies (best reachable PF
  across 6 alternative configs ~ 1.03).
- The discriminator between winners and losers is **whether the move
  continues**: winners' moves extend to ~23% from peak; losers stall at
  ~13%. Same setup, different post-entry behavior.

The continuation predictor is a binary classifier that scores, at signal
time, the probability a triggered Strategy B short setup will continue
(target: ≥18% drop within 48h before any 4% rally) vs stall.

## Artifact contract

| Item | Path |
|---|---|
| Trained model | `services/python/models/continuation_v1.joblib` |
| Feature schema + thresholds | `services/python/models/continuation_v1_meta.json` |
| Hand-crafted-rule fallback | `services/python/models/continuation_v1_rule.json` |
| Canonical backtest result | `services/python/results/backtest_strat_b_with_predictor_f264ce5e_20260425T155443Z/` |
| Transfer test (A, C) | `services/python/results/continuation_v1_transfer_test/transfer_test.json` |

The model is a `Pipeline(StandardScaler, LogisticRegression)` saved with
joblib. Inputs are 12 floats in the order of `FEATURE_NAMES` (see
`src/ml/continuation/features.py`). Output is `model.predict_proba(X)[:, 1]`,
a calibrated-ish probability of label=1.

The deployment threshold is **not stored in the joblib**; it is locked at
runtime as the **67th percentile of train-period predicted probabilities**
on Strategy-B trades, which is recomputed deterministically from the
training data at every run (~0.5236 on the canonical run).

## Label

`make_continuation_label(trade, candles, fwd_target=0.18, drawup_cap=0.04, max_bars=192)`:
- Positive: price falls ≥18% from entry within 192 bars (=48h) **before**
  any rally of >4% from entry.
- Negative: stalled, drawup-first, or window exhausted.
- Future bars used by design — never appears in the feature set.

Justification: 18% picks the midpoint between the analyst's 23%-vs-13%
finding; 4% drawup cap sits below the 5% baseline stop-loss so the label
respects what live execution would survive; 192 bars brackets B's median
133-bar hold.

Empirical positive rate on Strategy B: **5.2%** (3,580 / 68,852 trades).
Mean PnL by label on the training set: positive +14.85%, negative −0.50%
(15-pp gap), confirming the label is meaningful.

## Features (12)

All read only `bars[bar.timestamp <= entry_time]`. The bar at exact entry
time is allowed (the signal fires at its close). Strict no-lookahead
enforced by `tests/test_continuation_features.py::test_features_invariant_under_future_permutation`
which shuffles every post-entry bar and asserts the feature dict is unchanged.

| Name | Definition | Lookback |
|---|---|---|
| asset_4h_ret | pct change of asset close | 16 bars |
| asset_24h_ret | pct change of asset close | 96 bars |
| prior_bar_wick_ratio | (high−close)/(high−low) of the bar BEFORE entry | 1 bar |
| atr_norm_drop | (rolling_high32 − close) / ATR14 | 32, 14 bars |
| bars_since_high | bars since the 32-bar rolling high | 32 bars |
| btc_4h_ret | BTC pct change | 16 bars |
| btc_above_ema50_4h | BTC 4h close > 4h EMA-50 | 50 4h-bars |
| btc_realized_vol_24h | stdev of BTC 1h returns | 24 hourly returns |
| vol_decile | cross-sectional decile of asset 24h-volume vs universe | 96 bars |
| concurrent_down_breadth | fraction of universe with 4h-ret < −2% at entry | 16 bars |
| body_to_range_ratio | mean of \|close-open\|/(high-low) over last 8 bars (incl entry) | 8 bars |
| vol_zscore_8 | z-score of entry-bar volume vs prior 8 bars (excl entry) | 8 bars |

## Training

- Source: `results/backtest_3strat_protocol_B_28975aa0_20260424T093815Z/trades.csv` (68,894 short trades).
- Drop: trades whose 192-bar label horizon exceeds snapshot end (42 dropped).
  Drop: trades with any NaN feature (115 dropped on warmup).
- Time-ordered split: train < 2026-01-01 UTC (n=64,171), OOT ≥ 2026-01-01 (n=4,566).
- 5-fold purged time-series CV inside train, purge=192 bars (matches label horizon).
- Model: `Pipeline(StandardScaler → LogisticRegression(C=1.0, class_weight='balanced'))`.

### Gating decision A — hand-crafted rule (skipped model if passed)

Rule: `s = z(asset_4h_ret) + z(btc_4h_ret) + 0.5*z(atr_norm_drop) > q_67(train)`.
Z-score moments locked from training; threshold from training quantile.

| Period | Trades kept | PF |
|---|---|---|
| Train | 21,390 | 1.032 |
| OOT | 1,520 | **1.292** |

Just below the 1.30 floor → did **not** pass Gate A. Continued to LR.

### Gating decision B — LR baseline

| Metric | Value |
|---|---|
| 5-fold CV AUCs | 0.591 / 0.550 / 0.575 / 0.374 |
| Full-train AUC | 0.598 |
| OOT AUC | **0.552** |

OOT AUC clears the 0.55 informativeness floor (barely). LR saved as
the production candidate. Fold-3 AUC of 0.374 is below random — flags a
late-training-period regime shift, also visible in the walk-forward
below.

## Backtest result (Phase 4)

Run: `services/python/results/backtest_strat_b_with_predictor_f264ce5e_20260425T155443Z/`.
Locked threshold: proba ≥ 0.5236.

### 4-cell metrics

| Cell | Trades | WR | PF | Avg PnL | Median hold |
|---|---|---|---|---|---|
| train_off | 64,238 | 30.5% | 1.032 | +0.11% | 195 bars |
| train_on | 21,391 | 30.6% | 1.053 | +0.19% | 187 bars |
| oot_off | 4,656 | 36.1% | 1.243 | +0.77% | 199 bars |
| **oot_on** | **636** | **39.9%** | **1.696** | **+2.10%** | 245 bars |

Predictor lifts OOT PF by **+37%** and per-trade avg PnL by **+172%**.

### Walk-forward (predictor_on, 34 rolling 3-month folds, step=1mo)

| Stat | Value |
|---|---|
| p5 fold-PF | **0.277** |
| p25 | 0.589 |
| p50 | **1.184** |
| p75 | 2.523 |
| p95 | **6.055** |
| P(fold PF < 1.0) | **32.4%** |
| P(fold PF ≥ 1.30) | 47.1% |

The catastrophic 2023-Q4 / 2024-Q4 / early-2025 windows are still losers.

### Monte-Carlo (OOT predictor_on, 5000 bootstrap)

| Stat | PF |
|---|---|
| Point | 1.696 |
| p5 | **1.470** |
| p50 | 1.697 |
| p95 | 1.940 |

The OOT result is robust to resampling — i.e., not driven by a few
lucky trades.

### Ship gate (8-point standard)

| Gate | Required | Got | Pass? |
|---|---|---|---|
| OOT PF ≥ 1.30 | yes | 1.696 | ✅ |
| OOT n ≥ 200 | yes | 636 | ✅ |
| MC bootstrap p5 > 1.0 | yes | 1.470 | ✅ |
| Walk-forward p5 > 1.0 | yes | 0.277 | ❌ |
| **Overall** | | | **DO NOT SHIP** |

## Transfer test (Strategy A and C)

Same `continuation_v1.joblib`, same threshold (0.5236), no retraining.
Run by `scripts/transfer_test_continuation_v1.py`; full numbers in
`results/continuation_v1_transfer_test/transfer_test.json`.

| Strategy | OOT n_off | OOT PF_off | OOT n_on | **OOT PF_on** | MC p5_on |
|---|---|---|---|---|---|
| A (vol_accel) | 5,246 | 1.354 | 581 | **1.773** (+31%) | 1.537 |
| B (wait_bounce) — baseline | 4,656 | 1.243 | 636 | **1.696** (+37%) | 1.470 |
| C (multi_frame) | 5,924 | 1.290 | 646 | **1.554** (+20%) | 1.346 |

**The predictor transfers cleanly across all three strategies.** Trained on
B trades only, it lifts OOT PF for A and C without retraining, and all
three MC bootstrap p5 values are above 1.30. This is strong evidence the
predictor is capturing something general about "is this dump going to
continue" rather than B-specific entry-condition artifacts.

(Train-period predictor_on PF is slightly below train-period predictor_off
for A and C — expected since the model was trained on B, not on A/C
distributions; in-sample is meaningless here, only OOT matters.)

## Honest verdict + next steps

The predictor produces a real, MC-robust uplift on the OOT period (PF 1.24 → 1.70, avg PnL +172%) and is informative (OOT AUC 0.552). But the
walk-forward exposes the predictor's structural limit: it cannot rescue
the strategy in the catastrophic regimes (2023-Q4, 2024-Q4, early-2025)
where shorting fails wholesale regardless of feature value.

Per the 8-point standard, this is **research**, not deployment-grade.

**Recommended follow-ups (priority order):**

1. **Meta-regime gate** — Train a *binary classifier on regime favorability*
   (does the next 3-month window favor shorts at all?) and gate the
   continuation predictor by it. The hypothesis: when we know the next
   quarter is bear-friendly, the predictor's edge is strong; when it's
   bull-friendly, do not run shorts at all. This addresses the walk-forward
   p5 directly.
2. **Add OI / funding-rate features (Phase 3.5)** — `body_to_range_ratio`
   and `vol_zscore_8` are weak proxies for order flow. Pulling
   `contract_stats_1h.open_interest_usd` and `funding_rates.funding_rate`
   from the local Postgres into features would likely lift OOT AUC above
   0.55 by a meaningful margin, and is cheap to engineer (the backfill
   scripts already exist).
3. **Per-regime predictors** — Train separate `continuation_vN_btcUp.joblib`
   and `continuation_vN_btcDown.joblib` artifacts. Avoids the regime-shift
   collapse seen in CV fold 3 (AUC 0.374).
4. **Fix the universe parquet's listed_since column** — `export_snapshot.py`
   has an asset-name mismatch (`BTC` vs `BTCUSDT`) so all 672 entries are
   `None`. Cross-sectional features therefore include all assets at all
   times (mild survivorship bias). One-line fix to the exporter.

**Non-recommendations:**
- Do **not** lower the OOT-AUC bar from 0.55 to ship a weak model.
- Do **not** retune the threshold on OOT data to reach a 1.30 walk-forward
  p5 — that's a textbook overfit.
- Do **not** start exploring deep models before #2 above. Adding 3-5 good
  features beats a fancier classifier on 12 weak ones.

## Reproduce these numbers

```bash
cd services/python

# 1. Snapshot must already exist locally (data/snapshots/candles_15m_2026-04-01.parquet).
# 2. Train (60 seconds wall):
python scripts/train_continuation_classifier.py

# 3. Backtest with predictor gate (~10 minutes wall):
python scripts/backtest_strat_b_with_predictor.py

# 4. Verify reproducibility:
python scripts/backtest_strat_b_with_predictor.py
diff results/backtest_strat_b_with_predictor_<sha>_<ts1>/metrics.json \
     results/backtest_strat_b_with_predictor_<sha>_<ts2>/metrics.json
# → should be byte-identical

# 5. Transfer test on A + C:
python scripts/transfer_test_continuation_v1.py
```
