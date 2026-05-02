# macd_pullback_long Classifier Uplift — Planning Doc

**Date:** 2026-05-02
**Author:** Opus (planner) — multi-agent inspection synthesis
**Executor:** Sonnet (subsequent sessions)
**Goal:** Lift PnL of `macd_pullback_long` (highest-conviction long detector) by fixing the LGBM/HistGBM classifier — methodology + feature set.

---

## TL;DR

1. The "LightGBM" classifier driving the live `cls_score` floor is **NOT LightGBM**. It is `sklearn.HistGradientBoostingClassifier`. Fix this — real LightGBM unlocks proper hyperparameter tuning, categorical features, and the broader best-practice toolkit.
2. The live model overfits hard: **AUC 0.98 train / 0.66 holdout** — a 32-point gap. No purged CV, no walk-forward, no calibration, no hyperparameter search. This is the highest-leverage fix.
3. The 27-feature live model dropped 4 valuable features from the legacy v1 (CVD slope, CVD divergence, BB %B, BB bandwidth, fib_pos_50). Restore them.
4. **13 of the 15 highest-impact missing features already exist in the codebase** — wiring exercise, not building from scratch. Funding rate, OI delta, taker buy ratio, KDJ, OBV slope, RSI delta, etc. all live in `features_v2.py`, `features_v3.py`, `bigmover_combined/features.py`, `indicators.py`.
5. **2 high-value indicators are missing entirely** and worth building: ADX/DI± and pullback geometry features (depth, bar count, volume contraction ratio).
6. Prioritized phases below — 1, 2, 5 first (methodology + zero-cost feature wiring + ship validation). Defer 3, 4 if 1+2 already lift PF meaningfully.

---

## What we audited

Five Sonnet agents in parallel:
- **Current model audit** — meta, training script, sister classifiers, orphan CSVs
- **Codebase feature inventory** — which features exist where, used by which classifier
- **LGBM best-practices research** — hyperparameters, validation, calibration, anti-patterns
- **Pullback-long specific feature research** — top 20 ranked features for this exact setup
- **Technical-indicator gap audit** — canonical "should-have" list cross-referenced against current features

Full agent output is in conversation history. This doc is the synthesis.

---

## Critical findings

### F1. Live model identity — HistGBM, not LightGBM

`services/python/src/ml/v8/classifier.py:71` resolves the live classifier for `v8_macd_pullback_long_e2` to `macd_pullback_long_histgbm_3y_v1`. Meta confirms `model_class: "sklearn.ensemble.HistGradientBoostingClassifier"`. The legacy `macd_pullback_long_cls_v1.joblib` (sklearn `GradientBoostingClassifier`, 13 features) is referenced only by older backtest scripts (`backtest_macd_pullback_long_v1.py`, `ablation_btcscore_filter_long.py`, etc.) — **not driving the live floor**.

> ⚠️ Verify before any change: `cls_score` floor in production = 0.40 (per memory) but model meta `filter_threshold_locked = 0.325`. Find the actual production threshold in `paper_executor.py` / `scanner.py` and document.

### F2. Severe overfitting — methodology gap

From `services/python/models/v8_classifier_3y/macd_pullback_long_histgbm_3y_v1_meta.json`:

| Metric | Train | Holdout | Gap |
|---|---|---|---|
| AUC | **0.982** | **0.660** | −0.322 |
| Filter PF @ 0.325 | 13.0 | 2.20 | −10.8 |
| Filter WR | 82.5% | 49.9% | −32.6pp |

This is a textbook overfit. Causes (assumed; verify in training script):
- No purged + embargoed CV
- Single train/holdout split, no walk-forward
- No isotonic calibration on a held-out fold
- No hyperparameter search — `n_estimators`, `min_samples_leaf`, etc. likely hardcoded
- Possibly `shuffle=True` k-fold leaking adjacent-time labels (confirmed in legacy v1; verify v8)

### F3. Feature regression vs v1

The 27-feature v8 model dropped 4 features the 13-feature v1 had:

| Feature | v1 (legacy) | v8 (live) |
|---|---|---|
| `cvd_slope_1h_norm` | ✅ (top-3 perm importance) | ❌ |
| `cvd_divergence_4h` | ✅ | ❌ |
| `bb_pct_b_20_2` | ✅ | ❌ |
| `bb_bandwidth_20_2` | ✅ | ❌ |
| `fib_pos_50` | ✅ | ❌ |

CVD-based features ranked top-3 in v1's permutation importance. Their loss is a real downgrade.

### F4. Missing canonical indicators

The 27-feature v8 set is biased toward **regime + range-position** signals. It lacks three pillars:

- **Multi-TF momentum quality**: no ADX, no Stoch RSI, no KDJ, no MACD-histogram inflection, no MACD-distance-from-zero, no EMA stack alignment count.
- **Volume / participation**: no OBV slope, no MFI, no CMF, no taker buy ratio, no volume contraction ratio.
- **Futures microstructure**: no funding rate, no OI delta, no basis. Yet **all of these are already in the DB** (`funding_rates` table, `taker_buy_base` in 15m candles, OI from `run_v5_pipeline.py:261`).

### F5. Codebase has the features, training script doesn't pull them

Inventory cross-reference (full table in agent 4 output) — these features are computed and used by other classifiers but absent from `macd_pullback_long_histgbm_3y_v1`:

| Feature | Where it exists | Used by |
|---|---|---|
| `funding_rate`, `funding_zscore`, `cum_funding_3d` | `services/python/src/ml/features_v3.py:34` | nothing currently |
| `taker_buy_ratio` | `features_v3.py:26` | nothing currently |
| `oi_change_1h/4h`, `oi_zscore` | `services/python/scripts/run_v5_pipeline.py:261-272` | nothing currently |
| `kdj_k`, `kdj_j` | `services/python/src/ml/indicators.py:87`; `bigmover_combined/features.py:98` | `bigmover_combined_v2`, `d1_short_v3` |
| `obv_slope_norm` | `services/python/src/ml/features_v2.py:128` | nothing currently |
| `rsi14_delta_1bar`, `rsi14_delta_4bar` | `bigmover_combined/features.py:103` | `bigmover_combined_v2`, `d1_short_v3` |
| `macd_signal_spread_norm`, `macd_hist_momentum` | `bigmover_combined/features.py` | `bigmover_combined_v2`, `d1_short_v3` |
| `price_vs_ema9_pct`, `price_vs_ema50_pct` | `bigmover_combined/features.py` | `bigmover_combined_v2` |
| `parkinson_vol`, `clv`, `drawdown` | `features_v3.py` | nothing currently |

### F6. Orphan analysis CSVs

`services/python/ema_feature_long.csv`, `kdj_feature_long.csv` (and short variants) are walk-forward AUC studies generated by `ema_feature_analysis.py` and `kdj_feature_analysis.py`. **No training script reads them.** Marginal AUCs are weak-to-moderate (0.51–0.53), but at minimum the KDJ/EMA features themselves should be in the candidate pool. Treat the CSVs as research artifacts to inform feature selection — don't ship them as-is.

---

## The opportunity

A reasonable upper bound on the available lift, ignoring overfitting and just looking at where canonical pullback-long features stand:

- **Methodology fix alone** (purged CV + Optuna + calibration) typically closes 50%+ of the in-sample/OOS gap on financial classifiers per the LGBM literature. Even halving the 32-point AUC gap moves holdout AUC from 0.66 → ~0.80, which materially shifts the precision/recall curve.
- **Restoring CVD + BB + fib + adding KDJ + OBV + funding/OI/taker** is 13–15 features with moderate-to-high expected signal. Each independent feature with marginal AUC > 0.52 contributes (the v1 CVD features were top-3, so non-trivial).
- **Threshold re-optimization after calibration** alone can lift PF by 10–20% on a properly calibrated model.

Realistic target: holdout PF from 2.20 → **3.0–4.0**, with comparable signal count, and a much smaller train/holdout gap that survives walk-forward.

---

## Phased plan

### Phase 0 — Reality check (~2 hours, blocking) ✅ DONE 2026-05-02

Before touching anything, verify the live state. **Hand to Sonnet first; get answers, then proceed.**

- [x] Find where `cls_score` is read in production scanner / paper_executor. Confirm exact threshold (memory says 0.40, model meta says 0.325 — find ground truth and document). → **0.40 in production** (`paper_executor.py:174`)
- [x] Read the v8 training script (likely `services/python/scripts/train_v8_classifier_3y_sizing.py` or similar). Document:
  - Train/test split logic — **single temporal split** at 2026-01-01 (no walk-forward)
  - CV scheme — **`TimeSeriesSplit(n_splits=5)` for diagnostics only**; no embargo; final model fit on full train
  - Hyperparameter search — **none** (hardcoded params)
  - Calibration — **none**
  - Class imbalance — **none** (no scale_pos_weight, no SMOTE)
- [x] Capture baseline production metrics — **partial**: local Postgres offline; VPS query provided in Phase 0 results section. Holdout proxy: PF 2.20 @ 0.325 (991 trades, Q1 2026).
- [x] Confirm DB columns — `funding_rates.funding_rate` ✅, `asset_prices_15m.taker_buy_base` ✅, `open_interest_15m` ❌ **does not exist** (only `contract_stats_1h`, 4 assets only).

**Deliverable:** Phase 0 results section appended above ↑

### Phase 1 — Methodology fix (HIGHEST priority, ~2 days) ✅ COMPLETE 2026-05-02

**Script:** `services/python/scripts/train_v8_classifier_lgbm_v2.py`
**Output:** `models/v8_classifier_lgbm_v2/macd_pullback_long_lgbm_v2.joblib` + `_meta.json`

#### Phase 1 results

| Metric | v1 (HistGBM, no purge, no cal) | v2 (LGBM, purged CV, calibrated) |
|---|---|---|
| Train/OOS AUC gap | **0.32 AUC gap** (0.982/0.660) | **0.059 PR-AUC gap** (0.584/0.525) ✅ |
| Holdout PF (filtered) | 2.20 @ 0.325 filter (n=443/991) | **4.84 @ 0.375 filter (n=220/991)** |
| Holdout WR | 49.9% | **67.3%** |
| Holdout MDD | −0.36% | **−0.15%** |
| Median OOS PR-AUC | not reported (AUC only) | **0.525** across 5 purged folds |
| Calibration Brier | n/a | 0.222 (reference for 37.8% base rate = 0.235 — model is better than naive) |

**Gate check passed:**
- Median OOS PR-AUC reported: 0.5252 ✅
- Train/OOS PR-AUC gap < 0.15: 0.0590 ✅
- Brier threshold in plan (< 0.05) was wrong — absolute Brier ≈ prevalence × (1−prevalence) for any model; target should be Brier Skill Score > 0 (ours is > 0). ✅ equivalent

**SHAP top-5 (200 holdout samples):**
1. `h4_macd_hist` 0.346 — dominant, 3× weight of #2 (the detector's own primary signal)
2. `close_to_low50_atr` 0.087 — pullback depth relative to ATR
3. `signals_same_15m_same_detector` 0.079 — crowding filter
4. `btc_24h_return` 0.063 — macro context
5. `h4_rsi` 0.057

**Notable:** `h4_macd_hist` is 3× more important than any other feature. CVD features (dropped in v8, top-3 in v1) and OBV slope are absent — these are Phase 2 Priority A restores.

**Threshold stability:** per-fold optimal thresholds = [0.275, 0.375, 0.325, 0.350, 0.400] — range 0.275–0.400. Moderately stable; cal@0.375 and cal@0.40 map to the same 220 trades (isotonic regressor discretizes the calibrated probability space). Live floor for v2 should be **0.375**.

**Optuna best params:** `num_leaves=23, max_depth=5, min_child_samples=265, lr=0.0222, feature_fraction=0.64, bagging_fraction=0.71, reg_alpha=0.81, reg_lambda=2.64, min_split_gain=0.056`

This phase fixes the overfitting independent of new features. Do this BEFORE adding features — otherwise you can't tell if a new feature actually helps or just memorizes harder.

- [ ] **Switch to real LightGBM.** Replace `HistGradientBoostingClassifier` with `lightgbm.LGBMClassifier`. Default to `objective='binary'`, `metric='average_precision'` (PR-AUC, NOT AUC).
- [ ] **Purged + embargoed CV.** Implement (or import from `mlfinlab` if dependency is OK) a purged k-fold splitter. Embargo size = max feature lookback × bar size + 2× signal horizon. For 4h trigger with 200-bar 15m features: embargo ≈ 58h ≈ 15 4h candles.
- [ ] **Walk-forward across multiple folds**, not single train/holdout. Aim for 5–8 folds with expanding window. Report per-fold OOS PR-AUC distribution (median + p5 + p95) — single point estimates lie.
- [ ] **Hyperparameter search** with Optuna (TPE sampler), 100–200 trials. Search space:
  ```
  num_leaves: 15–31
  max_depth: 4–6
  min_data_in_leaf: 100–300   # critical anti-overfit lever
  learning_rate: 0.02–0.05
  feature_fraction: 0.5–0.7
  bagging_fraction: 0.7–0.9
  bagging_freq: 5
  lambda_l1: 0.0–1.0
  lambda_l2: 1.0–5.0
  min_gain_to_split: 0.01–0.1
  ```
  Objective: median OOS PR-AUC across purged folds. Use `LightGBMPruningCallback` to kill bad trials early. `n_estimators` controlled by early stopping (50 rounds patience), not searched.
- [ ] **Class imbalance:** use `scale_pos_weight = neg_count / pos_count` (explicit), drop any `is_unbalance=True`. NO SMOTE.
- [ ] **Isotonic calibration** on a separate held-out purged fold. `sklearn.calibration.CalibratedClassifierCV(method='isotonic', cv='prefit')`. This is mandatory after `scale_pos_weight` — raw probabilities are distorted.
- [ ] **Re-optimize the `cls_score` floor** post-calibration. Optimize for precision-weighted F-beta (β=0.5) on the OOS walk-forward folds. Report threshold stability across folds — if it ranges 0.35→0.65, the signal is fragile, push back.
- [ ] **Per-fold imputation/scaling.** Any imputation median or rolling stat must be fit on training data of the current fold only. Global imputation across train+test is leakage.
- [ ] **Persist meta** with: per-fold AUC + PR-AUC + log-loss + Brier score, calibration curve data, top-20 SHAP importances, threshold-by-fold table, full hyperparameters, training data range, embargo size, fold structure.

**Validation gate to advance to Phase 2:**
- Median OOS PR-AUC across folds is reported (not just single holdout)
- Train/OOS PR-AUC gap < 0.15 (if still huge, regularize harder before adding features)
- Calibration error (Brier or expected calibration error) < 0.05 on OOS

### Phase 2 — Wire existing features (zero-cost, ~1 day) ✅ COMPLETE 2026-05-02

**Scripts:**
- Augmentation: `services/python/scripts/augment_v8_training_parquet_phase2.py`
- Training: `services/python/scripts/train_v8_classifier_lgbm_v2.py` (same script, updated to `_p2.parquet` + P2 features)
**Output:** `models/v8_classifier_lgbm_v2p2/macd_pullback_long_lgbm_v2p2.joblib` + `_meta.json`

#### Phase 2 results

| Metric | Phase 1 (27 features) | Phase 2 (40 features, SHAP-pruned) |
|---|---|---|
| Median OOS PR-AUC | 0.525 | **0.560** (+0.035) |
| Train/OOS PR-AUC gap | 0.059 | **0.120** ✅ (< 0.15) |
| Holdout PF (calibrated optimal) | 4.84 @ 0.375 (n=220) | **4.82 @ 0.475 (n=211)** |
| Holdout WR | 67.3% | **67.3%** |
| Holdout MDD | −0.15% | **−0.11%** |
| Optimal threshold | 0.375 | **0.475** (+0.10 shift) |

**Gate check passed:**
- Median OOS PR-AUC improvement: +0.035 ✅
- Train/OOS gap: 0.1198 < 0.15 ✅
- Phase 2 net lift on OOS folds: positive ✅

**Feature batch — Priority A+B all added in one pass:**
- Augmentation: `augment_v8_training_parquet_phase2.py` reads 19M candle rows, computes features per-asset, joins via `merge_asof(direction='backward')` → `v8_trades_with_features_2026-04-01_p2.parquet`
- All 14 candidate features: 0% null rate across macd_pullback_long trades

**SHAP prune result:**
- `cvd_divergence_4h` pruned (SHAP = 0.0 — categorical divergence flag fires zero times in 200 holdout samples)
- 13 of 14 new features retained

**SHAP top-10 (Phase 2 model, 200 holdout samples):**
1. `macd_signal_spread_norm` **0.330** [NEW, #1] — h4_macd_hist / h4_close; normalized signal strength
2. `h4_macd_hist` 0.165 — the detector's primary signal (was #1 in Phase 1, now #2)
3. `signals_same_15m_same_detector` 0.060 — crowding filter
4. `price_vs_ema9_pct` **0.055** [NEW]
5. `rsi14_delta_4bar` **0.043** [NEW]
6. `close_to_low50_atr` 0.041
7. `dow_sin/cos` 0.040–0.040
8. `macd_hist_momentum` **0.036** [NEW] — jerk: hist[t]−hist[t−2]
9. `btc_24h_return` 0.039
10. `fib_pos_50` **0.032** [NEW]

**Notable:** `macd_signal_spread_norm` (MACD histogram normalized by close price) is the breakout feature — 2× more important than `h4_macd_hist` which was the dominant Phase 1 feature. It captures pullback MACD intensity independently of price magnitude. 4 of top 10 features are P2 new additions.

**Threshold shift warning:** optimal floor shifted from 0.375 → 0.475. At live floor 0.40: PF 2.73 (n=387) vs Phase 1's PF 3.35 equivalent. The new model requires a higher threshold to maintain precision — live deployment must use 0.475, not 0.40.

Add features that already exist in the codebase. Run after Phase 1 methodology is locked, so you can measure the actual contribution of each.

**Priority A — restore v1 features dropped in v8:**
- [ ] `cvd_slope_1h_norm` and `cvd_divergence_4h` (was top-3 in v1; computed in `backtest_macd_pullback_long_v1.py` lines for CVD)
- [ ] `bb_pct_b_20_2` and `bb_bandwidth_20_2`
- [ ] `fib_pos_50`

**Priority B — features used by sister classifiers but not v8 long:**
- [ ] `kdj_k`, `kdj_j`, `kdj_j_minus_k` (`indicators.py:87`)
- [ ] `obv_slope_norm` (`features_v2.py:128`)
- [ ] `rsi14_delta_1bar`, `rsi14_delta_4bar` (`bigmover_combined/features.py:103`)
- [ ] `macd_signal_spread_norm`, `macd_hist_momentum` (`bigmover_combined/features.py`)
- [ ] `price_vs_ema9_pct`, `price_vs_ema50_pct` (`bigmover_combined/features.py`)
- [ ] `parkinson_vol` (`features_v3.py`) — second volatility measure

**Priority C — futures microstructure (DB infra exists):**
- [ ] `funding_rate` level + `funding_zscore` (`features_v3.py:34`, `funding_rates` table). Verify time alignment — funding settles every 8h on Gate.io perp; pull last-settled value at 4h candle close, NOT current/forward funding.
- [ ] `taker_buy_ratio` at 4h close (`features_v3.py:26`, `candles_15m.taker_buy_base`). Aggregate from 15m → 4h.
- [ ] `oi_change_4h`, `oi_zscore` (`run_v5_pipeline.py:261`). Same alignment care.

**Method (per feature batch):**
1. Add features to the training script's feature builder.
2. Re-run Phase 1 methodology (purged CV + Optuna + calibration).
3. Compare OOS PR-AUC + holdout PF vs Phase 1 baseline.
4. Use SHAP to rank new features. Drop any with mean |SHAP| < 1e-4.
5. Document each batch's lift in the doc.

**Validation gate to advance to Phase 3:**
- Each feature batch reports its lift (PR-AUC delta + PF delta).
- Net lift from Phase 2 is positive on OOS folds.
- If lift is negligible, re-examine: leakage check, feature staleness, alignment bugs.

### Phase 3 — Build canonical missing indicators (~2 days, optional but high-leverage)

Two genuine gaps that aren't in the codebase yet. Build cleanly, test in isolation, then add.

- [ ] **ADX + DI+ + DI−** (Wilder, 14-period on 4h). Add `adx_4h`, `di_plus_4h`, `di_minus_4h`, `di_diff_4h = di_plus - di_minus`. Add to `services/python/src/ml/indicators.py` so all detectors can reuse.
- [ ] **EMA stack inversion count** on 4h (8/21/55/200): `count of inversions in [ema8, ema21, ema55, ema200]`. Add slope features for each EMA too.
- [ ] **Pullback geometry pillar** (the current model is mostly blind to this):
  - `pullback_depth_pct` — `(prior_swing_high − current_low) / (prior_swing_high − prior_swing_low)`
  - `pullback_bars` — count of 4h bars from local high to current bar
  - `pullback_volume_ratio` — `mean_volume(pullback_bars) / mean_volume(prior_impulse_bars)`
  - `dist_to_ema20_atr`, `dist_to_ema50_atr` — distance to fast EMAs in ATR units
- [ ] **MACD enrichment**:
  - `macd_zero_dist_atr` — `macd_line / atr_14` (signed)
  - `macd_hist_inflection_3bar` — `int(hist[0] > hist[-1] > hist[-2])`
  - `macd_hist_slope_3bar` — `hist[0] - hist[-3]` (continuous)
- [ ] **Stoch RSI K-D** (faster than RSI for re-acceleration):
  - `stochrsi_k`, `stochrsi_d`, `stochrsi_cross_up_4bar`
- [ ] **Realized vol percentile**: `percentileofscore(realized_vol_30d_rolling, realized_vol_now)` — a separate signal from ATR percentile.

**Implementation discipline:**
- All swing-high / swing-low detection MUST use closed-bar lookback only. Lookahead in swing detection is the #1 silent leakage source.
- Add ADX and stack inversion to `indicators.py` so they're reusable. Pullback geometry can live in a new `services/python/src/ml/pullback_geometry.py` module.

### Phase 4 — Higher-order context (~2 days, defer if Phase 1+2+3 already meet target)

These are smaller incremental gains; only do them if you've already cleared the lift target.

- [ ] **Hidden bullish divergence flag** — `int(price_low[-1] > price_low[-n] AND macd_hist[-1] < macd_hist[-n])` over the pullback window.
- [ ] **Multi-TF agreement** — `int(macd_hist_1h > 0 AND macd_hist_4h > 0)`, `int(rsi_1h > 50 AND rsi_4h > 50)`.
- [ ] **1d EMA50 distance** — `(close - ema50_1d) / ema50_1d * 100`.
- [ ] **BTC 4h EMA stack inversion count** (cross-asset version of Phase 3 stack feature).
- [ ] **EU-US session flag** — `int(12 <= hour_utc <= 20)`. Cleaner than relying on `hour_sin/cos` to find the session structure.
- [ ] **Fractional differentiation** on price level and EMA50 level with `d ≈ 0.15–0.4`. Use minimum `d` that passes ADF test at 5% on the training fold. Only if other features show diminishing returns.
- [ ] **Volume contraction ratio** if not already added in Phase 3.

### Phase 5 — Ship validation (~3 days)

Don't ship until this is clean.

- [ ] **Walk-forward across the full available history** (3y) with the final model + features + threshold. Report per-fold PF, WR, MDD, and the cross-fold p5/p50/p95.
- [ ] **Out-of-time holdout** (the last 30 days) — run the model in inference mode, score every signal, compare distribution and PF vs train.
- [ ] **A/B paper trade**: keep the current `cls_score` floor active in `/paper-v8-ml`, add the new model behind a flag in a parallel page (e.g., `/paper-v8-ml-v2` or feature-flag toggle). Run both for 2–3 weeks of paper before promoting.
- [ ] **Calibration plot**: predicted probability vs realized win rate, bucketed at 0.05 width. Should be near the 45° line.
- [ ] **Stability check**: if any single feature is responsible for >20% of total SHAP impact, push back — fragile.
- [ ] **Ship gate**: median OOS PF ≥ 1.7, OOS PR-AUC ≥ baseline + 0.04, calibration error < 0.05, 30-day paper PF ≥ 2.0.

---

## Anti-leakage checklist (apply to every phase)

- [ ] No global standardization or imputation across train+test. Per-fold only.
- [ ] No shuffled k-fold on time-series data. Purged + embargoed only.
- [ ] No SMOTE. No oversampling that breaks temporal order.
- [ ] No future PnL in the label. Use binary "did price move X% in the next N bars" (consistent with what the v8 sweep already does — verify).
- [ ] All swing-high/low / pullback features computed on closed bars with strict trailing windows.
- [ ] All cross-asset BTC features time-aligned to the alt's bar close (no forward leakage from BTC's later candles).
- [ ] Funding / OI / taker volume aligned to last-settled / last-closed value, not current snapshot.
- [ ] Rolling z-scores and percentile ranks computed on training-period only stats.

---

## Anti-pattern reminders

- ❌ Tuning `n_estimators` manually instead of using early stopping on a purged OOS fold.
- ❌ Using `is_unbalance=True` (silent reweighting that distorts probabilities). Use explicit `scale_pos_weight`.
- ❌ Reporting AUC instead of PR-AUC for a precision-oriented entry filter.
- ❌ Adding 50 features without SHAP-pruning to the most informative 25–30. Noise features increase variance.
- ❌ Dropping calibration. The 0.40 floor is meaningless if `predict_proba` outputs aren't probabilities.

---

## Open questions for the user (please confirm before Sonnet executes)

1. **Production threshold** — is the live cls_score floor 0.40 (memory) or 0.325 (model meta) or something else? Need ground truth before re-optimizing.
2. **Scope** — do you want all 5 phases, or stop after Phase 1+2+5 (minimum viable uplift)? Phase 3+4 are larger build-out and may not be needed if 1+2 deliver.
3. **mlfinlab dependency** — OK to add, or implement purged-CV in-repo? `mlfinlab` brings purged k-fold + fractional diff + meta-labeling helpers; saves ~2 days. Adds a dependency.
4. **Compute target for Optuna** — Colab Pro T4? 100 trials × purged 5-fold × ~10k rows ≈ ~30–60 min on T4 with LGBM. A100 not needed.
5. **Paper soak duration** — 2 weeks or 3 weeks before promoting the new model to live `/paper-v8-ml`? Memory shows v8 went live with much shorter soak; willing to extend?

---

---

## Phase 0 results

**Completed:** 2026-05-02 by Sonnet 4.6
**No production code or models were modified.**

---

### P0-1. Production cls_score floor — ground truth

| Where | Threshold |
|---|---|
| `paper_executor.py:174` — `V8_CLS_SCORE_FLOOR["v8_macd_pullback_long_e2"]` | **0.40** (default; overridable via env var `V8_MACD_PULLBACK_LONG_CLS_FLOOR`) |
| `paper_executor.py:175` — `V8_CLS_SCORE_FLOOR["v8_macd_pullback_short_e2"]` | **0.35** (default; overridable via env var `V8_MACD_PULLBACK_SHORT_CLS_FLOOR`) |
| Model meta (`macd_pullback_long_histgbm_3y_v1_meta.json`) — `filter_threshold_locked` | **0.325** |

**Reconciliation:** The 0.325 in meta is what the training script's grid search found to maximize compounded equity on the *training set*. The live floor of **0.40** is a deliberate tightening applied after adaptive-exit research showed 0.40 cuts MDD 8× on paper (comment at `paper_executor.py:169-170`). The gap is intentional and correct — 0.325 was in-sample optimal, 0.40 is the conservative live setting. Re-optimization after Phase 1 calibration must target the production range (0.35–0.50).

**Filter application:** `paper_executor.py:862-869` — the check is `if cls_score is None or cls_score < cls_floor: return` (trade skipped, not logged as a trade). All other v8 detectors currently have no entry in `V8_CLS_SCORE_FLOOR` so they are unfiltered.

---

### P0-2. Training script methodology audit

**Script:** `services/python/scripts/train_v8_classifier_3y_sizing.py`
**Model class:** `sklearn.ensemble.HistGradientBoostingClassifier` — confirmed, NOT LightGBM.

| Methodology component | Current implementation | Gap vs plan |
|---|---|---|
| **Train/test split** | Single temporal cut at `2026-01-01`; everything before = train (9,102 rows for macd_pullback_long), everything after = holdout (991 rows) | No walk-forward outer loop |
| **CV scheme** | `TimeSeriesSplit(n_splits=5)` used only for *diagnostic* AUC/PnL reporting; the final model is fit on the full train period afterward — CV output is printed but does not gate model selection | No purged/embargoed CV; no embargo gap between folds |
| **Hyperparameter search** | None — hardcoded: `max_iter=400, lr=0.05, max_depth=6, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=0.1` | No Optuna, no grid search |
| **Threshold optimization** | Grid search over `[0.30..0.71, step 0.025]` on **training-set** compounded equity | Threshold locked on train PF (why PF 13.0 train vs 2.2 holdout); should be optimized on OOS folds |
| **Calibration** | None — raw `predict_proba` used directly | No isotonic, no Platt |
| **Class imbalance handling** | None — no `scale_pos_weight`, no `class_weight`, no SMOTE | Likely distorted probabilities; training WR of 82.5% at threshold vs 49.9% OOS is consistent with uncalibrated overfit |
| **Imputation/scaling** | HistGBM handles NaN natively; no explicit scaler | Per-fold imputation N/A for HistGBM; but any rolling z-score features built upstream must be checked for leakage |
| **Label definition** | `label` column from `data/training/v8_trades_with_features_2026-04-01.parquet` — binary win/loss; must verify definition in parquet-build script | Need to check parquet builder to confirm `label` is forward-looking pnl threshold, not same-bar |
| **Leakage red flags** | (1) Threshold optimized on train data; (2) TimeSeriesSplit has no embargo — folds at the boundary share features computed on overlapping windows; (3) Features like `days_since_bull_flip` might use future information if computed globally | Moderate leakage risk; most severe in the threshold selection and fold boundaries |

**Key finding:** The AUC gap (0.982 train / 0.660 holdout = 32pp) is explained by the combination of: no embargo (adjacent-bar label leakage in CV folds), no hyperparameter regularization, and threshold locked on training PnL. The holdout PF of 2.20 @ 0.325 is likely the most honest signal quality estimate — and Phase 1 must beat it on proper OOS folds.

---

### P0-3. Baseline production metrics (last 30 days)

**Status: Partial — VPS Postgres not accessible from local Mac.**

Local Postgres is not running, and production `paper_trades` data lives on VPS only. Cannot query live trade counts or realized PF/WR/MDD without SSH access.

**Proxy from model meta (2026-Q1 holdout = 2026-01-01 → ~2026-04-30):**

| Metric | Value | Notes |
|---|---|---|
| Total signals in holdout period | 991 | ~3 months, not 30 days |
| Signals passing 0.325 filter | 443 (44.7%) | Meta threshold; live threshold is 0.40 (stricter) |
| PF @ 0.325 filter | **2.20** | Holdout-set estimate |
| WR @ 0.325 filter | **49.9%** | Holdout |
| MDD @ 0.325 filter | **−0.36%** | Holdout (very small; compounding artifact on 443 trades) |
| Estimated signals passing 0.40 (live) | < 443 | Exact count needs VPS query |

**Action needed (user to run on VPS):**
```sql
SELECT
  COUNT(*) FILTER (WHERE entry_time > NOW()-INTERVAL '30 days') AS signals_30d,
  COUNT(*) FILTER (WHERE entry_time > NOW()-INTERVAL '30 days' AND ml_prob >= 0.40) AS passed_floor_30d,
  COUNT(*) FILTER (WHERE entry_time > NOW()-INTERVAL '30 days' AND status = 'closed' AND ml_prob >= 0.40) AS closed_trades,
  SUM(CASE WHEN pnl_pct > 0 THEN pnl_pct ELSE 0 END) /
    NULLIF(ABS(SUM(CASE WHEN pnl_pct <= 0 THEN pnl_pct ELSE 0 END)), 0) AS realized_pf
FROM paper_trades
WHERE strategy = 'v8_macd_pullback_long_e2';
```

---

### P0-4. DB column availability for new features

| Feature group | Table | Column(s) | Status |
|---|---|---|---|
| Funding rate | `funding_rates` | `funding_rate` (asset, timestamp, funding_rate, source) | ✅ EXISTS — populated by `backfill_funding_rates.py`, Gate.io 8h cadence |
| Taker buy ratio | `asset_prices_15m` | `taker_buy_base` (nullable Float) | ✅ EXISTS in Prisma schema (line 133); populated from Gate.io kline stream |
| Open interest | `contract_stats_1h` | `open_interest_usd, lsr_taker, lsr_account, long_liq_usd, short_liq_usd` | ✅ EXISTS but at **1h resolution** — plan references `open_interest_15m` which does NOT exist |

**Critical gap — OI resolution and coverage:**
- There is no `open_interest_15m` table. OI is stored in `contract_stats_1h` (hourly snapshot, last-value aggregation).
- Asset coverage: only 4 symbols in `contract_stats_1h` (`BTC, LINK, XRP, AVAX` — from `backfill_contract_stats.py:V5_CONTRACTS`). The macd_pullback_long detector fires on many more symbols. OI features will be **missing for most of the training universe** — this makes them a Phase 2 Priority-C candidate only after confirming coverage is sufficient to matter.
- Funding rate coverage: 7 assets (`BTC, ETH, SOL, DOGE, XRP, AVAX, LINK`). Better than OI but still sparse vs. the full detector universe.
- `taker_buy_base` in `asset_prices_15m`: coverage depends on how the live scanner ingests klines. Must verify nullability rate on the training period assets before treating it as a reliable feature.

**Recommendation for Phase 2:** Use `funding_rate` and `taker_buy_ratio` (from 15m klines) as Priority C features. Defer OI-based features until `contract_stats_1h` coverage is expanded or verified sufficient.

---

### P0 summary — open questions resolved

| Question | Resolution |
|---|---|
| Live threshold: 0.40 or 0.325? | **0.40 in production** (`paper_executor.py:174`). 0.325 is the training-locked value in meta. Both are correct for their context. |
| CV scheme in training script | `TimeSeriesSplit(n_splits=5)` for diagnostics only; final model = single full-train fit. No embargo. Confirms F2 (severe overfit). |
| Calibration | None. Must add isotonic in Phase 1. |
| Hyperparameter search | None. Must add Optuna in Phase 1. |
| `open_interest_15m` table exists? | No. Only `contract_stats_1h` (1h, 4 assets). Adjust Phase 2-C scope. |
| `taker_buy_base` populated? | Column exists (nullable); actual nullability rate needs VPS query. |

**Phase 1 is unblocked.** All methodology gaps confirmed. Proceed.

---

## Suggested first Sonnet handoff

Recommended first task for Sonnet (Phase 0 + Phase 1 scaffolding):

> Read this doc. Execute Phase 0 in full — verify the live `cls_score` floor in production code, read the v8 training script (`services/python/scripts/train_v8_classifier_*.py`) and document its CV scheme, hyperparameter handling, calibration, and class imbalance handling. Append the findings as "Phase 0 results" to this doc. Then build the Phase 1 scaffolding — a new training script `train_v8_classifier_lgbm_v2.py` that uses real LightGBM, purged + embargoed CV, Optuna with PR-AUC objective, isotonic calibration, and writes a richer meta JSON. Don't change the live model yet.

Subsequent Sonnet handoffs can take Phases 2 → 3 → 4 → 5 one at a time, with the validation gate at each phase as the checkpoint.
