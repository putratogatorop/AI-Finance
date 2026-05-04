# v_new_1 PAIR — Big-Movers Detection Strategy — Research Plan

**Date:** 2026-05-02
**Author:** Opus (planner), reviewed by user before execution
**Executor:** Sonnet (per CLAUDE.md agent delegation pattern)
**Goal:** 200-1000% per year, beat BTC buy-hold benchmark over 4 years, max drawdown constraint ≥ -20% in any OOS year
**Status:** AWAITING USER REVIEW — no execution until approval

---

## TL;DR

Build TWO LightGBM models — `v_new_1_long` and `v_new_1_short` — that score every coin in the top-100 USDT-perps universe at every 4h candle close. Score = predicted probability that the coin will move ≥10% (long) or ≤-10% (short) within the next 7 days. Trade enters when score crosses optimized threshold. Position sizing, concurrent caps, stop/trailing rules optimized via Optuna against a multi-objective function with hard max-DD constraint. Validate on 4 OOS regime years (2020/2021/2022/2026 Q1). Deploy only if it beats BTC buy-hold by 2× over 4 years AND no OOS year has DD worse than -20%.

This explicitly ABANDONS the rule-based detector approach (macd_pullback_long, macd_pullback_short_e2). The ML scoring IS the trigger.

---

## 1. Why this exists (motivation)

The current production strategy `macd_pullback_long` over 4 years generated +96% sum_pnl. BTC buy-and-hold over the same period returned ~250-300% with no model. The current strategy underperforms a no-effort benchmark. After fees and slippage on $100 capital, it's effectively charity to Gate.io.

Root cause: the rule-based MACD pullback detector enters AFTER pullback consolidation. Most big crypto moves are breakouts/momentum events that don't have clean MACD pullback structure first. The detector misses them by design. Per the `big_movers` table evidence: 38,068 big-move events occurred in 2023-2025 across the universe — roughly 35 per day — most completing within 2 hours to 3 days. The current strategy catches almost none of these.

This research builds a fundamentally different architecture: continuous ML scoring of all coins, with both directions (long and short) running concurrently, position sizing optimized by ML.

---

## 2. Success criteria (explicit "ding-ding" gates)

A policy must clear ALL three gates to be considered shippable:

### Gate 1 — Beats BTC buy-hold
4-year multi-OOS sum_pnl > 250% (BTC roughly 4×'d during 2020-2026 ≈ 300% return). Set bar at 250% for a 25% safety margin.

### Gate 2 — All-weather (max DD constraint)
Max drawdown in EACH of the 4 OOS years (2020, 2021, 2022, 2026 Q1) must be ≥ -20%. Any year with DD < -20% disqualifies the policy regardless of total return.

### Gate 3 — Every year positive
Sum_pnl in EACH of the 4 OOS years must be > 0%. The strategy must make money in 2022 bear, 2020 recovery, 2021 alt mania, AND 2026 Q1.

### Stretch goal — 200-1000% target
The user's stated annualized goal is 200-1000% per year on $100 capital. Mapped to 4-year sum_pnl (with 5× leverage assumed): 800% to 4000% over 4 years. Any policy in this range is exceptional.

If after 5 iterations no policy clears Gates 1-3, the architecture is abandoned and we revert to discussion.

---

## 3. Strategy architecture

### What's NEW (replaces rule-based detector)
- **No technical detector** — we don't pre-filter signals via MACD/RSI patterns
- **Continuous coin scoring** — every 4h candle close, BGM scores all 100 coins
- **ML output IS the trigger** — coin enters portfolio when its score crosses threshold
- **Direct labeling via big_movers** — labels are the actual outcomes we care about (10%+ moves)

### What's REUSED from prior research
- **40 v2p3 features** as the foundation feature set (proven)
- **LightGBM + isotonic calibration** as the BGM template (proven)
- **Phase B 2020-2026 OHLCV data** as the price history (already downloaded)
- **4-year multi-OOS framework** (2020/2021/2022/2026 Q1) for validation
- **Phase 0 threshold-locking protocol** for any threshold the policy uses
- **CFGI feature** (already backfilled in Phase Portfolio results)

### Pair architecture
Two independent models, shared infrastructure:

**v_new_1_long:**
- Predicts: P(coin moves +10% within 7 days)
- Trained on: 2023-2025 long_event labels
- Used at runtime: score 100 coins every 4h, take longs above threshold

**v_new_1_short:**
- Predicts: P(coin moves -10% within 7 days)
- Trained on: 2023-2025 short_event labels
- Used at runtime: score 100 coins every 4h, take shorts above threshold

Both models run concurrently. Portfolio composition emerges from market state — in bull regimes, more long_scores cross threshold; in bear regimes, more short_scores cross. No explicit regime gate (CFGI is one feature among 40, model decides how to use it).

---

## 4. Data & universe

### Universe filter
**Top-100 USDT perps by trailing 30-day volume** (rolling, recomputed weekly).

Rationale:
- Slippage is the silent killer at $100 capital. Small-cap perps can have 1-2% slippage round-trip, eating 10-20% of a 10% target win.
- Top-100 by volume ensures liquid execution.
- Universe rotates over time as new coins list and old ones delist — handle this with rolling membership.

Source: `services/python/data/binance_archive_2020_2022/raw/` for historical, plus current production OHLCV for 2023-2026.

### Time period
- **Training data:** 2023-01-01 to 2025-12-31 (3 years, ~1.97M (coin × 4h timestamp) rows after filtering)
- **OOS validation years:** 2020 (May+), 2021 (full), 2022 (full), 2026 (Jan-Mar Q1) — 4 distinct regimes

### Bar resolution
- 4h candles for both training and inference
- 7-day forward window for label = 42 × 4h bars

---

## 5. Labels (definitive spec)

For each (coin, timestamp T) row in the 2023-2025 dataset:

```
long_event = 1 if max(close[T+1 .. T+42 bars]) / close[T] >= 1.10 else 0
short_event = 1 if min(close[T+1 .. T+42 bars]) / close[T] <= 0.90 else 0
```

**Notes:**
- 10%+ absolute is the threshold — 15%, 20%, 50% moves all label as "1"
- Closes only (NOT highs/lows) — high-wick spikes are often unfillable in real execution
- Both labels can be 1 simultaneously (volatile coin moves both directions in 7d)
- Forward window strictly excludes T itself (T+1 onwards) — no lookahead in label
- Features at T strictly use data ≤ T

**Class imbalance:**
- Estimated long_event positive rate: 8-12% (most 4h candles do NOT precede a 10%+ up move within 7 days)
- Estimated short_event positive rate: 6-10%
- Handle via `scale_pos_weight = neg/pos ≈ 8-15` in LightGBM

**Iteration knob:** if Iteration 1 fails, threshold can be tested at 8% / 12% / 15% / 20% to find the sweet spot for predictability vs trade frequency.

---

## 6. Features

### Foundation: v2p3's 40 features (already validated)
Standard regime + price + volume + bar-shape + cross-asset + time features. Full list in `services/python/models/v8_classifier_lgbm_v2p3/macd_pullback_long_lgbm_v2p3_meta.json`.

### Additions for v_new_1
Add these to the 40-feature foundation:

1. **CFGI value** (already backfilled, daily granularity, joined to nearest 4h timestamp)
2. **Volume rank** within the 100-coin universe at this timestamp (top-100 normalized rank)
3. **Days since coin's last big move** (look up in big_movers table)
4. **BTC's regime score** (we have btc_score; verify it's in the v2p3 set)
5. **Coin's 24h volume z-score** vs trailing 30-day mean (liquidity surge indicator)

Total: ~45 features for v_new_1 (vs 40 in v2p3).

### Iteration knob
If baseline BGM has poor PR-AUC, expand feature set:
- Open interest, taker volume (sparse but available for some coins)
- Multi-timeframe: 1d trend, 1h trend
- Volatility regime classifier outputs
- Sector/narrative features (if implementable)

NOT iterating on: model class (stay BGM), feature engineering style (stay tabular)

---

## 7. Models & training

### Architecture
- LightGBM Classifier with isotonic calibration (matches v2p3 proven template)
- One model per direction (v_new_1_long, v_new_1_short)
- `scale_pos_weight` calibrated to address ~8-15× class imbalance

### Hyperparameter search (Optuna, 100-150 trials per direction)
Search space (matches v2p3 best practices):
- num_leaves: [15, 63]
- max_depth: [4, 8]
- learning_rate: [0.005, 0.05]
- min_data_in_leaf: [50, 300]
- feature_fraction: [0.5, 0.9]
- bagging_fraction: [0.6, 0.95]
- lambda_l1, lambda_l2: [0, 5]
- n_estimators: 200-2000 with early_stopping=100

### Training/validation/test
- **Training set**: 2023-01 to 2024-09 (75% of pre-OOS data) — base model fit
- **Calibration set**: 2024-10 to 2025-12 (last 25%) — fit isotonic calibrator on disjoint slice (per v2p3 honest calibration protocol)
- **OOS validation**: 2020/2021/2022/2026 Q1 (4 separate regime tests)

### Threshold locking (per Phase 0 protocol)
- Sweep score thresholds on calibration set to find optimal entry threshold per direction
- LOCK threshold before any OOS evaluation
- Apply locked threshold to all 4 OOS years

---

## 8. Policy optimization (Optuna layer 2)

After BGM probability models are trained, optimize the trading policy via Optuna.

### Search space
| Parameter | Range | Notes |
|---|---|---|
| `entry_threshold_long` | [0.10, 0.50] | Lower = more trades, less precision |
| `entry_threshold_short` | [0.10, 0.50] | Independent from long |
| `max_concurrent_positions` | [5, 30] | Hard cap, gross exposure derived |
| `position_size_pct` | [0.5, 2.0] | % capital per trade (notional = position × leverage) |
| `leverage` | fixed at 5× | NOT searched, locked at user spec |
| `stop_loss_pct` | [-2, -8] | Stop loss as % from entry (notional) |
| `take_profit_pct` | [None, +10, +15, +20] | Optional fixed TP |
| `trailing_atr_mult` | [None, 2, 3, 5] | Optional trailing stop |
| `hold_timeout_days` | [3, 7] | Hard timeout (max 7 = label horizon) |
| `score_scaled_sizing` | [True, False] | Higher score → bigger position? |
| `extreme_cfgi_short_gate` | [None, 60, 70, 80] | Optional: cap short size if CFGI > X |

### Objective function (constrained)
```python
def optuna_objective(policy):
    metrics = backtest(policy, oos_years=[2020, 2021, 2022, 2026Q1])

    # HARD CONSTRAINTS — disqualify any violator
    for year_metric in metrics:
        if year_metric.max_dd_pct < -20:
            return -1_000_000  # disqualified
        if year_metric.sum_pnl_pct < 0:
            return -1_000_000  # not all-weather

    # OBJECTIVE — maximize 4-year sum_pnl with Sharpe weighting
    total_pnl = sum(m.sum_pnl_pct for m in metrics)
    median_sharpe_like = median(m.sum_pnl_pct / abs(m.max_dd_pct) for m in metrics)
    
    return total_pnl + 50 * median_sharpe_like  # scoring blends both
```

### Optuna budget
- 200 trials per (model_set × policy_search) round
- ~3-5 minutes per trial on M2 Pro (each trial = backtest on 4 OOS years)
- Total: 10-17 hours wall-clock per Optuna round

### Threshold contamination prevention (CRITICAL)
- Optuna optimizes ONLY on cal_fit (last 25% of 2023-2025 = 2025-Q3-Q4)
- Locked policy then evaluated ONCE on 4 OOS years
- DO NOT iterate the policy by looking at OOS results — that contaminates the entire framework

---

## 9. Validation framework

The 4-year multi-OOS test is the FINAL validation:

| Year | Regime | Test purpose |
|---|---|---|
| 2020 (May+) | COVID recovery / low-vol → explosive | Did model learn early-cycle patterns? |
| 2021 (full) | Alt mania bull | Can model capture parabolic moves? |
| 2022 (full) | LUNA + FTX bear | Can short side carry the year? |
| 2026 Q1 | Post-ETF mild trend | Reproduces production conditions? |

Per-year metrics required:
- n_long_positions, n_short_positions taken
- sum_pnl_pct (long contribution, short contribution, total)
- compounded_equity_pct
- max_drawdown_pct
- median per-trade PnL
- Sharpe-like ratio
- Trade frequency (per month)

If any year violates the gates, the policy is disqualified regardless of other metrics.

---

## 10. Phases (numbered with deliverables)

### Phase 1 — Universe construction + label generation (4-6 hours)

**Inputs:** Existing 4h OHLCV parquets (2020-2026)
**Outputs:**
- `services/python/data/v_new_1/universe_top100_membership.parquet` — per-week top-100 universe membership
- `services/python/data/v_new_1/labels_long.parquet` — (coin, timestamp, long_event) for 2023-2025
- `services/python/data/v_new_1/labels_short.parquet` — same for short
- Sanity report: positive class rate per year, per coin distribution
- Verification: spot-check 10 random labels against actual price movement

**Ding-ding:** label parquets exist, positive class rates plausible (5-15%), no lookahead detected in spot checks

### Phase 2 — Feature computation across all (coin × timestamp) (6-8 hours)

**Inputs:** OHLCV parquets, labels parquets, CFGI history
**Outputs:**
- `services/python/data/v_new_1/features_full.parquet` — ~1.97M rows × 45 features for 2023-2025
- Feature null rate report per feature
- Validation: 4 features (funding, HMM, CVD-related) expected to be NaN-heavy — confirm

**Ding-ding:** features parquet exists, null rates within expected ranges (≤ 25% for non-microstructure features)

### Phase 3 — BGM baseline training (4-6 hours per direction = 8-12 hours total)

**Inputs:** features + labels parquets
**Outputs:**
- `services/python/models/v_new_1_long/v_new_1_long.joblib` + meta JSON
- `services/python/models/v_new_1_short/v_new_1_short.joblib` + meta JSON
- Per-fold OOS PR-AUC + ROC-AUC + Brier scores
- Top-20 SHAP feature importances per model

**Ding-ding:** OOS PR-AUC > 0.10 above class baseline (e.g., baseline ~0.10 for 10% positive rate, target > 0.20)

### Phase 4 — Multi-OOS scoring (1-2 hours)

**Inputs:** trained models, OOS year features
**Outputs:**
- Per-year scores for all (coin, timestamp) in each OOS year
- Per-year per-direction precision-recall curves at multiple thresholds

**Ding-ding:** scores are within sane probability range (no extreme drift), top-decile precision > 3× baseline

### Phase 5 — Optuna policy optimization (10-17 hours)

**Inputs:** scored OOS data, search space spec
**Outputs:**
- Best policy hyperparameters (locked)
- Per-trial result log
- Top-10 policies report

**Ding-ding:** ≥1 policy clears all 3 gates (BTC-beat, max-DD, all-weather)

### Phase 6 — Final OOS validation + report (2-3 hours)

**Inputs:** locked best policy
**Outputs:**
- Per-year detailed performance breakdown
- Comparison vs BTC buy-hold benchmark
- Comparison vs current macd_pullback_long
- Detailed trade log per OOS year
- Verdict: 🟢 SHIP / 🟡 ITERATE / 🔴 ABANDON

**Ding-ding:** 🟢 = clears all 3 gates; otherwise iterate per protocol

### Phase 7 — Production wiring (only if Phase 6 = 🟢, deferred until user approval)

NOT executed in this research. Reserved for after research success.

---

## 11. Iteration loop (when phases fail)

If Phase 3 fails (BGM baseline doesn't meet PR-AUC ding-ding):

**Iteration 1.1 — Label tuning (2-3 hours)**
- Test labels at 8%, 12%, 15%, 20% (vs baseline 10%)
- See if a different threshold has cleaner predictability

**Iteration 1.2 — Feature expansion (4-6 hours)**
- Add multi-timeframe features (1d, 1h trends)
- Add structural features (HH/HL count, days-since-impulse)
- Test if new features lift PR-AUC

**Iteration 1.3 — Bigger universe (2 hours)**
- Try top-200 instead of top-100
- More data may help imbalanced learning

**Iteration 1.4 — Direction-specific feature engineering (4-6 hours)**
- Long model: add features that capture momentum-buildup
- Short model: add features that capture distribution patterns

If Phase 5 fails (Optuna can't find policy meeting gates):

**Iteration 2.1 — Relax constraints**
- Try max-DD threshold at -25% instead of -20% (compromise)
- Allow one OOS year to be slightly negative if 4-year total is exceptional

**Iteration 2.2 — Expanded search space**
- Wider position sizing range
- Multi-tier threshold (different thresholds per regime)

If after 5 iterations no convergence: ABANDON, revert to discussion.

---

## 12. Compute budget (M2 Pro reality)

| Phase | Optimistic | Realistic | Pessimistic |
|---|---|---|---|
| 1 — Labels | 4h | 6h | 8h |
| 2 — Features | 6h | 8h | 12h |
| 3 — BGM training | 8h | 12h | 16h |
| 4 — OOS scoring | 1h | 2h | 4h |
| 5 — Optuna policy | 10h | 15h | 20h |
| 6 — Validation report | 2h | 3h | 4h |
| **Total** | **31h** | **46h** | **64h** |

Assuming Sonnet runs Phases 1-2 in foreground (interactive verification), Phases 3 + 5 in background overnight, Phases 4 + 6 in foreground (results review):

- **Day 1 (8h work day)**: Phases 1+2 complete, Phase 3 started overnight
- **Day 2**: Phase 3 completes morning, Phase 4 + start Phase 5 overnight
- **Day 3**: Phase 5 completes morning, Phase 6 finishes day. Verdict.

**3 working days from green light to verdict** if all phases pass first-attempt.

If iterations needed: add 1 day per iteration, max 5 iterations = 8 working days total.

---

## 13. Risk register

### High-priority risks

**R1 — Lookahead leakage in label construction.** Critical because the 7-day forward window in labels could accidentally leak into features if not strictly enforced.
- Mitigation: explicit per-row timestamp validation, spot-checks, separate scripts for label vs feature computation

**R2 — Class imbalance traps the model.** ~10% positive class. If `scale_pos_weight` not tuned, model predicts all-zeros = 90% accuracy but useless.
- Mitigation: explicit scale_pos_weight calculation, threshold-tuned PR-AUC reported (not just AUC), top-decile precision check

**R3 — Slippage destroys EV in production.** $100 capital × 5× leverage = $500 notional per trade. Slippage on small-cap alts can be 0.5-2% per side. Optuna in backtest doesn't see this and finds policies that look great but lose to friction.
- Mitigation: realistic transaction cost model included in backtest (Phase 1 deliverable)
- Backstop: top-100 universe filter eliminates worst slippage

**R4 — 4-year OOS too noisy to discriminate.** Per-year noise on 4 metrics may be too high to confidently pick "best" policy.
- Mitigation: paired comparison vs BTC buy-hold (relative performance is tighter than absolute)
- Constraint: gates require ALL 4 years to pass, not just average — disqualifies "lucky 1-year" policies

**R5 — Optuna finds policies that look great in OOS but were over-optimized.** 200 trials × 4 years = potential to find a policy that's 0.05-pp better via noise.
- Mitigation: hold out a 5th OOS slice (e.g., 2026 Q2 if available, or split 2026 Q1 50/50) for final validation never seen by Optuna

### Medium-priority risks

**R6 — Regime shift in 2026+** that the 2023-2025 training data doesn't reflect (e.g., post-ETF structural change, new derivatives products, regulatory changes).
- Mitigation: monthly review per `MODEL_REVIEW_MONTHLY.md`, retrain quarterly on rolling window

**R7 — Sonnet implementation errors** in label construction or backtest mechanics.
- Mitigation: per-phase ding-ding checkpoints, paired sign-test verification, Opus reviews each phase output

**R8 — Compute thermal throttling on M2 Pro** during long Optuna runs.
- Mitigation: run in background, max 6 parallel jobs, periodic monitoring

---

## 14. Open decisions for user review

These are the calls I'm seeking explicit confirmation on:

1. **Plan doc location/name acceptable?** Currently at `docs/V_NEW_1_PAIR_PLAN-2026-05-02.md`.

2. **Universe filtering by 30-day rolling volume — confirm top-100?** Or do you want to test 50 / 200 ranges?

3. **Label thresholds**:
   - Iteration 1 baseline: 10% absolute
   - If Iteration 1 fails: test 8%, 12%, 15%, 20%
   - Acceptable?

4. **Compute time tolerance**: 3 working days for first-attempt cycle. If iterations needed, up to 8 working days. Acceptable, or do you want a tighter timebox?

5. **Phase 5 Optuna budget**: 200 trials per round. Acceptable, or want fewer (faster but less search) or more?

6. **Final validation slice**: do you want to hold out 2026 Q1 50/50 (Q1a for Optuna OOS, Q1b for never-seen final test), or trust the 4-year framework as-is?

7. **Production deployment trigger**: Phase 6 = 🟢 means we have a candidate. Do you want to deploy to live $100 immediately upon 🟢, or paper-trade for 2-4 weeks first as a stress test?

---

## 15. What gets deployed if research succeeds

If Phase 6 returns 🟢 and you approve:

- Replace current production v1 macd_pullback_long with v_new_1_long + v_new_1_short
- New paper_executor logic: every 4h, score 100 coins, take top-K by score per direction
- v6 ATR sizing replaced or augmented with the optimized policy parameters
- macd_pullback_long deprecated to ablation-only (keep model files for reference)

This is a meaningful production change — gradual rollout recommended:
- Week 1-2: Paper trading at $50 fake capital alongside live macd_pullback_long
- Week 3-4: Real $100 with v_new_1, monitor v_new_1 vs BTC buy-hold
- Month 2+: Scale capital based on realized performance vs simulation

---

## 16. What does NOT change

These remain stable, no research disruption:

- Production v1 macd_pullback_long stays running until v_new_1 deploys
- VPS infrastructure (Postgres, Docker Compose, scanner) unchanged
- Memory + monthly review process per `MODEL_REVIEW_MONTHLY.md`
- Risk: 0.5% per trade × 5× leverage policy is the working assumption

---

## Summary / approval checklist

To approve this plan and authorize Phase 1 execution, please confirm:

- [ ] Goal (200-1000% per year, beat BTC buy-hold, max DD ≥ -20%) is correct
- [ ] Architecture (no detector, BGM scoring, paired long+short) is the right direction
- [ ] Universe (top-100 by 30-day volume) is acceptable
- [ ] Labels (10%+ absolute within 7 days, both directions) is acceptable
- [ ] Compute budget (3-8 working days on M2 Pro local) is acceptable
- [ ] Iteration protocol (5 max iterations before abandon) is acceptable
- [ ] No deployment without user approval after Phase 6 verdict

After confirmation, Sonnet executes Phases 1-2 first. After review, continues 3-6.
