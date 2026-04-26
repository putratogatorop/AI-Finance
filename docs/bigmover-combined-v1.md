# Bigmover + Multi-Indicator ML — research result (v1)

**Status:** RESEARCH — does NOT pass the 8-point ship gate.
Use these numbers internally. Do **not** deploy to live trading without
the follow-up work in the "Where to next" section.

## Why this exists

The user wanted to combine the volume-based **bigmover** detector
(`src/ml/bigmover_signals.py`, leak-free since `fix/v2ml-disable-pending-audit`)
with classical technical confirmations (EMA, RSI, MACD, KDJ) and gate
position size by a **continuous BTC-trend score** instead of the existing
binary regime flag — i.e., "no trades against BTC trend, scaled by
trend strength."

Two design choices locked in by the user before the work began:
1. **ML classifier** on the combined features (not hand-crafted AND-gate
   or weighted vote score).
2. **Continuous BTC-trend sizing** (not a 3-class hard gate).

## What this lane built

| Layer | Artifact |
|---|---|
| Indicators (EMA, RSI, MACD, KDJ, ATR) consolidated | `services/python/src/ml/indicators.py` |
| KDJ implementation (was missing) | `services/python/src/ml/indicators.py:kdj` |
| Direction-aware label | `services/python/src/ml/bigmover_combined/labels.py` |
| 16-feature library + is_short metadata | `services/python/src/ml/bigmover_combined/features.py` |
| Continuous sign-locked sizing | `services/python/src/ml/bigmover_combined/sizing.py` |
| Specialist-team synthesis | `services/python/src/ml/bigmover_combined/SYNTHESIS.md` |
| Bigmover detector re-run on 2026-04-01 (6 combos) | `scripts/backtest_bigmover_baseline_protocol.py` + `results/backtest_bigmover_baseline_protocol_*` |
| Trained classifier | `services/python/models/bigmover_combined_v1.joblib` (LR) |
| Classifier metadata | `services/python/models/bigmover_combined_v1_meta.json` |
| Sizing config | `services/python/models/bigmover_combined_v1_sizing.json` |
| Protocol-compliant ML+sizing backtest | `scripts/backtest_bigmover_combined_with_ml.py` + `results/backtest_bigmover_combined_with_ml_18aa3e0b_*` |
| Sensitivity + per-combo transfer | `scripts/sensitivity_bigmover_combined_v1.py` + `sensitivity_*.json` in the run dir |

## Feature set (16 + is_short)

Synthesized by a 5-specialist team (one per EMA / RSI / MACD / KDJ / BTC-trend
family). Per-specialist findings + rejection rationale in
`services/python/src/ml/bigmover_combined/SYNTHESIS.md`.

| Group | Features |
|---|---|
| Bigmover detector internals (5) | vol_ratio, price_drop_pct, price_rise_pct, accel_atr_norm, bars_since_high32 |
| EMA (2) | price_vs_ema9_pct, price_vs_ema50_pct |
| RSI (2) | rsi14_delta_1bar, rsi14_delta_4bar |
| MACD (2) | macd_signal_spread_norm, macd_hist_momentum |
| KDJ (2) | kdj_j, kdj_k |
| BTC trend score (Candidate B) | btc_trend_score = clip((macd-signal)_4h_lag1 / close_4h_lag1, ±0.005255) / 0.005255 |
| BTC volatility | btc_realized_vol_24h |
| Cross-section | concurrent_dir_breadth |
| Metadata | is_short |

The BTC trend score formula was selected from 4 candidates by walk-forward
monotonicity testing. The plan's placeholder (Candidate A: daily SMA50
deviation) went the **wrong way** for longs (monotone diagnostic = −0.668);
Candidate B (4h MACD spread) is the only formula where SHORT WR rises as
the score → −1 and LONG WR rises as the score → +1 (monotone diagnostic
+1.473).

Surprise findings from the team (also in SYNTHESIS.md):
- **RSI textbook claim partially refuted**: "RSI > 70 = good for short" does
  not hold; RSI extremes flag reversal-prone bars regardless of direction.
- **KDJ classical extremes inverted**: J > 100 (overbought) is *anti-
  predictive* for shorts on bigmover trades — it marks exhaustion of the
  bigmover thrust, not continuation.
- **MACD and EMA distance largely overlap** — kept `macd_hist_momentum`
  (2-bar Δ of histogram = "jerk") as the orthogonal MACD piece.

## Continuous sign-locked sizing

```python
# src/ml/bigmover_combined/sizing.py
def position_scale(btc_trend_score, direction, *, config=SizingConfig()):
    if not (sign(btc_trend_score) == sign(direction)):
        return 0.0    # explicit "no trades against BTC trend"
    return clip(config.slope * |btc_trend_score| + config.intercept,
                config.floor, config.cap)
```

Default `SizingConfig(slope=1.5, intercept=0, floor=0, cap=1.5)`. Wrong-side
trades zero out by construction.

## Training (Phase 4)

- Source: 6 (signal × direction) Phase-2 trade CSVs concatenated → 306,453
  labelable trades (99.9% feature-complete).
- Train (entry_time < 2026-01-01): 283,233. OOT: 22,911.
- 5-fold purged time-series CV inside train, purge=192 bars (label horizon).
- LR baseline only (LightGBM unavailable on this Mac — libomp not installed).

| Metric | Value |
|---|---|
| 5-fold CV AUCs | 0.5427 / 0.5563 / 0.5198 / 0.5920 |
| Full-train AUC | 0.5765 |
| **OOT AUC** | **0.5948** (above 0.55 floor; OOT > train = no overfit) |

Compared to continuation-predictor v1 (OOT AUC 0.552), the bigmover-combined
OOT AUC of 0.595 is meaningfully better — confirms the synthesized 16-feature
set carries real signal beyond what the 12-feature continuation set captured.

## Backtest result (Phase 5) — 8-cell metrics

Locked threshold: `proba ≥ 0.4785` (q0.5 of train scores).
Run: `services/python/results/backtest_bigmover_combined_with_ml_18aa3e0b_20260425T165448Z/`.

| Cell | Trades | WR | PF | Avg PnL |
|---|---|---|---|---|
| train_mloff_equal | 283,479 | 30.0% | 0.985 | −0.05% |
| train_mloff_sized | 149,550 | 30.7% | 1.056 | +0.13% |
| train_mlon_equal | 141,619 | 29.7% | 1.010 | +0.04% |
| train_mlon_sized | 71,909 | 30.6% | 1.089 | +0.22% |
| oot_mloff_equal | 23,396 | 30.0% | 0.921 | −0.28% |
| oot_mloff_sized | 12,336 | 29.1% | 0.887 | −0.30% |
| oot_mlon_equal | 10,295 | 29.1% | 0.946 | −0.19% |
| **oot_mlon_sized** (full strategy) | **5,529** | **30.9%** | **1.098** | **+0.25%** |

Full strategy lifts OOT PF +19% over baseline (0.921 → 1.098). ML alone
gives +2.7%, sizing alone is *worse* than baseline (−3.7%) — but combined
they add +18% over ML-alone. The two layers are complementary.

### Walk-forward + Monte Carlo

| Stat | Combined | SHORT-only |
|---|---|---|
| WF p5 | **0.743** | 0.363 |
| WF p50 | 1.015 | 0.934 |
| WF p95 | 1.752 | 2.133 |
| WF P(PF<1.0) | 47.1% | 52.9% |
| MC point PF | 1.098 | **1.525** |
| MC p5 | 1.033 | 1.420 |

## The major finding — directional asymmetry

Per-(signal, direction) OOT breakdown of the full strategy:

| Combo | Baseline OOT PF | Full strategy OOT PF | n |
|---|---|---|---|
| baseline_**short** | 1.307 | **1.521** | 1,078 |
| baseline_long | 0.571 | 0.592 | 961 |
| price_accel_atr_**short** | 1.370 | **1.609** | 1,045 |
| price_accel_atr_long | 0.559 | 0.537 | 901 |
| multi_bar_confirm_**short** | 1.350 | **1.444** | 989 |
| multi_bar_confirm_long | 0.533 | 0.486 | 555 |

**All 3 SHORT variants OOT clear the 1.30 ship floor** with n=989–1078 each.
**All 3 LONG variants OOT lose money**.

The combined OOT PF of 1.098 was the harmonic-ish average of these, and
it understated the short-side performance.

Hypothesis for the asymmetry (not yet tested rigorously): with the SL=5%
/ TP=15% / 672-bar exit rule, crypto's structural asymmetry kicks in —
sustained 15%+ rallies after a 5% pump are rarer than sustained 15%+
drops after a 5% dump under the same bigmover trigger. The long side
needs different exit rules, not a different model.

## Ship-gate verdict (8-point standard)

| Gate | Required | Combined | SHORT-only |
|---|---|---|---|
| OOT PF ≥ 1.30 | yes | 1.098 ❌ | 1.525 ✅ |
| OOT n ≥ 200 | yes | 5,529 ✅ | 3,112 ✅ |
| Walk-forward p5 > 1.0 | yes | 0.743 ❌ | 0.363 ❌ |
| MC bootstrap p5 > 1.0 | yes | 1.033 ✅ | 1.420 ✅ |
| **Overall** | | **DO NOT SHIP** | **DO NOT SHIP** |

Even SHORT-only fails walk-forward p5: 53% of historical 3-month windows
have PF < 1.0. The 2026-Q1 OOT window happened to be exceptionally
short-friendly (consistent with the regime pattern surfaced by the
continuation-predictor's walk-forward).

## Sensitivity (Phase 6)

### Threshold sweep (locked on train; OOT diagnostic only)

| Threshold | ~Train quantile | Trades | PF |
|---|---|---|---|
| 0.45 | 0.258 | 8,328 | 0.994 |
| 0.50 | 0.649 | 4,002 | **1.162** |
| 0.55 | 0.849 | 1,844 | 1.105 |
| 0.60 | 0.930 | 915 | 1.097 |

The locked 0.4785 (q=0.5) threshold lands between rows 1 and 2; threshold
0.50 is mildly better OOT but **was not selected on OOT** per protocol.
Higher thresholds shrink sample faster than they lift PF.

### Sizing-knob sensitivity (cap × floor)

| Cap | Floor | OOT full PF |
|---|---|---|
| 1.0 | 0.00 | 1.073 |
| **1.5** | **0.00** | **1.098** (default) |
| 2.0 | 0.00 | 1.098 |
| 1.0 | 0.25 | 1.045 |
| 1.5 | 0.25 | 1.073 |
| 2.0 | 0.25 | 1.073 |

Cap rarely binds (|score| ≤ 1 by construction → max raw size = 1.5).
Floor=0 is uniformly better (forcing minimum sizes hurts when the model
is uncertain). ±20% perturbations don't change the ship-gate verdict.

## Honest verdict + where to next

**This is research, not deployment-grade.** But the lane produced strong
incremental progress:

- 16-feature synthesized set has OOT AUC 0.595 (vs 0.552 for continuation
  predictor v1 — the indicators do work).
- Combined-direction WF p5 jumped from 0.277 (continuation-predictor v1)
  to 0.743 (this work) — the catastrophic regime tail is materially
  smaller, even if not yet < 1.0.
- SHORT-only OOT PF 1.525 is the strongest result in any lane so far.

**Recommended follow-ups (priority order):**

1. **SHORT-only deployment with strict regime + sample-size guards.** Even
   though WF p5=0.363 fails the standard, the OOT PF of 1.525 with MC
   p5 of 1.420 is real. A capped-capital paper deployment ($1k–$10k notional)
   with a hard "exit all positions when btc_trend_score crosses zero"
   guardrail could collect live signal-to-execution data while limiting
   downside. Decision is the user's, not an engineering one.

2. **Fix the long side's exit rules.** Long-side WR of 19.4% under SL=5%/
   TP=15% suggests crypto rallies don't sustain to 15%. Try TP=8% / SL=4%
   / timeout=192 bars (24h) on long-side bigmover entries and re-train
   the classifier — this is a 1-day experiment.

3. **Meta-regime gate** — the same idea that was the recommended follow-up
   from continuation-predictor v1: a binary classifier "does the next
   3-month window favor shorts at all?" gating the bigmover predictor.
   Most direct path to fix WF p5.

4. **OI + funding-rate features (Phase 3.5 from the plan).** OOT AUC was
   0.595 — a meaningful uplift to 0.62+ from real order-flow features
   (already in the local Postgres) might be enough to lift WF p5 above 1.0.

5. **Fix `export_snapshot.py`'s `listed_since` bug** (tracked in continuation-
   predictor v1 docs; same applies here). Cross-sectional `concurrent_dir_breadth`
   currently uses all assets at all times → mild survivorship bias.

**Non-recommendations:**
- Don't lower the OOT-AUC bar from 0.55 to ship a weak model.
- Don't tune the threshold or sizing knobs on OOT data — that's overfit.
- Don't deploy at full size before walk-forward p5 > 1.0.

## Reproduce these numbers

```bash
cd services/python

# 1. Snapshot must already exist (data/snapshots/candles_15m_2026-04-01.parquet).
# 2. Re-run the bigmover detector (~7 min wall):
python scripts/backtest_bigmover_baseline_protocol.py

# 3. Train (~1.5 min wall):
python scripts/train_bigmover_combined_classifier.py

# 4. Backtest with ML+sizing (~4 min wall):
python scripts/backtest_bigmover_combined_with_ml.py

# 5. Verify reproducibility:
python scripts/backtest_bigmover_combined_with_ml.py
diff results/backtest_bigmover_combined_with_ml_*<earlier_ts>*/metrics.json \
     results/backtest_bigmover_combined_with_ml_*<later_ts>*/metrics.json
# → byte-identical

# 6. Sensitivity + transfer:
python scripts/sensitivity_bigmover_combined_v1.py

# 7. Phase 8 follow-ups (re-aggregation only, no candle re-load):
python scripts/phase8_weekly_metagate_eval.py    # weekly meta-gate eval
python scripts/phase8_exit_sweep.py              # 7-config exit sweep
```

## Phase 8 — post-hoc tweaks: weekly meta-gate + exit-rule sweep

After the v1 ship-gate FAIL on walk-forward p5, the user asked to test
two post-hoc fixes to lift WF p5 above 1.0 without retraining:
(1) a coarser **weekly-trend meta-gate**, and (2) an **exit-rule sweep**
including trailing stops and asymmetric SL/TP.

Both knobs were exhausted; **neither rescues the strategy**.

### Phase 8.1 — Weekly meta-gate REJECTED

Hypothesis: blocking SHORT signals when BTC weekly close ≥ weekly
EMA-26 will kill catastrophic months (2023-Q4 rally, 2024-04 pump,
2024-11 post-election rally, 2025-08 mid-cycle rally).

Result: gate works on 7/8 catastrophic months — but at huge cost.

| Cell | No gate | + Weekly gate |
|---|---|---|
| train trades | 38,883 | 7,291 (5× cut) |
| **train PF** | **1.114** | **1.000** (no edge) |
| oot trades | 3,112 | 3,112 (BTC was weekly-bear all of OOT) |
| oot PF | 1.525 | 1.525 (no-op on OOT) |
| **WF p5** | **0.363** | **0.268** (worse) |

Diagnosis matches the Phase-6 by-bucket finding: "strong bear" PF 1.117
< "bear" PF 1.217. The weekly-below-EMA gate selects the strongest-bear
regime — which is exactly NOT where this strategy works best. Gate
rejected.

Result file: `services/python/results/backtest_bigmover_combined_with_ml_*/phase8_weekly_metagate.json`

### Phase 8.2 — Exit-rule sweep: baseline IS the winner

Re-simulated all 41,995 short ml_on+sized entries under 7 alternative
exit configs. Locked on TRAIN total_pnl (NOT OOT — protocol-mandated).

| Config | TRAIN PF | OOT PF | OOT avg | Verdict |
|---|---|---|---|---|
| **E0 baseline 5%/15%/672** | **1.114** | **1.525** | **+1.345%** | ✅ **WINNER** |
| E1 tight TP 5%/8%/672 | 1.050 | 1.381 | +0.839% | underperforms |
| E2 symmetric 5%/5%/672 | 1.009 | 1.270 | +0.498% | gives up upside |
| E3 fast timeout 5%/15%/96 | 0.967 | 1.314 | +0.589% | exits winners early |
| E4 trail 3% (no fixed TP) | 0.982 | 0.749 | −0.277% | catastrophic on OOT |
| E5 trail 2% activated | 0.998 | 1.114 | +0.111% | mild loss |
| E6 tighter SL 3%/8%/672 | 1.043 | 1.264 | +0.461% | underperforms |

The **current 5%/15%/672 exit is genuinely best on both train and OOT**
— no overfit risk, no better config available in this grid.

Notable: trail-stop variants (E4, E5) underperform on OOT despite their
intuitive appeal. The leakfix backtest's reported PF 3.15 with trail-3%
came from a different setup (no ML filter, capped capital, different
regime mix); that result does NOT transfer to the ML-gated single-trade
pool tested here.

Result file: `services/python/results/backtest_bigmover_combined_with_ml_*/phase8_exit_sweep.json`

### Phase 8 conclusion

Both post-hoc tweaks tested and rejected. The Phase-5 result
**(SHORT-only OOT PF 1.525, WF p5 0.363, ship-gate FAIL)** is the
ceiling for this configuration. Further improvement requires upstream
changes (which Phase 9 + Phase 10 then attempted):

- **Different label** (currently 0.10 fwd_target / 0.04 drawdown /
  192 bars) — try 0.06/0.02/96 to capture faster bigmover continuations
  the current label misses.
- **More features** — OI + funding-rate from the local Postgres
  (`backfill_funding_rates.py`, `backfill_contract_stats.py` already
  scaffolded). Most promising single uplift opportunity per the
  Phase-3 specialist team's footnote.
- **Long-side rebuild** — long machinery is fundamentally broken under
  SL=5%/TP=15%. Different label + retrain on long-only trades is a
  separate, ~3-day workstream.
- **Capped paper deployment of SHORT-only as-is** — the OOT PF 1.525
  with MC bootstrap p5 1.420 is real. A $1k-$10k notional paper run
  with strict btc_weekly cross exit could collect live signal-to-
  execution data while limiting downside. **Decision is the user's,
  not engineering's**, and is explicitly out of scope for this lane.

## Phase 9 — outcome-aligned label (v2)

**Hypothesis:** the v1 label (10%-drop-in-48h-before-4%-rally) is a
*proxy* for actual trade outcome under SL=5%/TP=15%/timeout=672. Train-
ing the classifier on the actual outcome (`label = pnl_pct > 0`)
instead of the proxy should yield a better filter.

**Result:** directionally yes, structurally no.

| Metric | v1 (proxy label) | v2 (outcome label) | Δ |
|---|---|---|---|
| Train AUC | 0.5765 | 0.5376 | −3.9pp (v1 was overfitting its label) |
| OOT AUC | 0.5948 | 0.5957 | ≈ same (no generalization gain) |
| **Combined OOT PF** | 1.098 | **1.314** | **+19.7%** ← clears 1.30 floor |
| OOT WR | 30.9% | 35.8% | +4.9pp |
| OOT avg pnl | +0.25% | +0.76% | 3× |
| MC bootstrap p5 | 1.033 | **1.247** | +21% (more robust) |
| **WF p5** | 0.743 | **0.553** | −26% (no progress on the killer gate) |

Per-direction OOT (v2):

| Combo | n | PF | WR |
|---|---|---|---|
| **SHORT total** | **5,268** | **1.451** | 39.7% |
| baseline_short | 2,063 | 1.447 | 40.6% |
| price_accel_short | 1,654 | 1.434 | 39.6% |
| multi_bar_short | 1,551 | 1.476 | 38.6% |
| LONG total | 1,169 | 0.582 | 18.1% (still broken) |

SHORT-only walk-forward (v2): p5 = 0.367 (vs v1's 0.363 — no progress).
Confirms: WF fail is **not a classifier-accuracy problem**.

Run: `services/python/results/backtest_bigmover_combined_v2_outcome_0582b799_20260426T033736Z/`
Artifacts: `services/python/models/bigmover_combined_v2_outcome.{joblib,_meta.json,_sizing.json}`

## Phase 10 — meta-regime gate

**Hypothesis:** stack a separate "is the next 3 months short-friendly?"
gate on top of v2. Gate is computed at entry time using BTC-only
features (no future leak) and blocks signals during predicted-bull
quarters.

### 10.1 Hand-crafted BTC-momentum gate (`btc_90d_return ≤ T`)

Sweep on 64,187 v2 SHORT entries:

| Threshold | TRAIN PF | OOT PF | WF p5 | MC p5 |
|---|---|---|---|---|
| T=+0.00 | 1.358 | 1.451 | 0.139 | 1.367 |
| T=+0.05 | 1.293 | 1.451 | 0.261 | 1.367 |
| **T=+0.10** | 1.344 | 1.451 | **0.474** ← best WF | 1.367 |
| T=+0.15 (TRAIN-locked) | 1.376 | 1.451 | 0.259 | 1.367 |
| T=+0.20 | 1.321 | 1.451 | 0.197 | 1.367 |
| T=+0.30 | 1.247 | 1.451 | 0.065 | 1.367 |

OOT n + OOT PF + MC p5 unchanged across thresholds — BTC's 90d return
was ≤ 0% throughout the OOT window (Jan-Apr 2026), so the gate is a
no-op on OOT. Only TRAIN-period folds change.

Catastrophic-month rescue at T=+0.15:
- 2023-11/12, 2024-02 KILLED entirely (n→0)
- 2024-11 (post-election rally): 2,625 → 728 trades, PF 0.099 → 0.211
- 2023-10: 1,558 → 946, PF 0.292 → 0.521
- 2025-08: 1,407 → 816, PF 0.256 → 0.480
- 2024-05: barely affected, PF actually worsens 0.269 → 0.185
- 2023-09: untouched (BTC wasn't rallying — bear-bottom whipsaw is a
  *different* failure mode the momentum gate cannot detect)

Even the WF-best threshold T=+0.10 only lifts WF p5 to 0.474 — far
below 1.0 ship floor.

### 10.2 ML meta-regime classifier — REJECTED

7-feature LR (`btc_30d_return`, `btc_90d_return`, `btc_180d_return`,
`btc_30d_realized_vol`, `btc_vs_sma200_pct`, `btc_vs_ema26_weekly_pct`,
`btc_macd_4h_signal_spread_norm`) trained on 26 monthly window-level
samples to predict "next 3 months SHORT-only PF ≥ 1.0".

- Train AUC: 0.8625 (suspiciously high — overfitting tiny sample)
- OOT AUC: NaN (only 1 OOT window)
- WF p5 with this gate: 0.400 (**worse** than the simple T=+0.10 rule)
- OOT n with this gate: 0 (model predicted negative class for the
  entire OOT period → blocks all OOT trades → automatic ship-gate FAIL)

Top features (scaled coefficients): `btc_180d_return` (+0.351),
`btc_30d_return` (−0.240), `btc_macd_4h_signal_spread_norm` (+0.165).

Run: `services/python/results/.../phase10_meta_regime_classifier.json`
Artifacts: `services/python/models/meta_regime_v1.{joblib,_meta.json}`

### Phase 10 conclusion + final post-mortem

**The walk-forward fail is STRUCTURAL, not addressable via post-hoc gating.**

The 8 catastrophic months span at least two distinct regimes:
- **BTC bull rallies** (2023-Q4, 2024-Q1, 2024-Q4 to $100K, 2025-Q3) —
  any momentum-based gate can flag these.
- **BTC bear-bottom whipsaw** (2023-09 with PF 0.30 at WR 25%) — this
  is the OPPOSITE momentum state yet equally bad for shorts. No linear
  gate can identify both groups as "skip" without also blocking the
  modest-bear regime where the strategy actually works (PF 1.217 per
  Phase 6 bucket analysis).

The 7-feature ML classifier on 26 windows could not separate these
either. The data simply doesn't admit a low-complexity quarter-level
gate that both rescues catastrophic months AND preserves the regimes
where the strategy works.

**Final ship-gate verdict:**

| Gate | v1 SHORT-only | v2 SHORT-only | v2 + 10.1 best | v2 + 10.2 ML |
|---|---|---|---|---|
| OOT PF ≥ 1.30 | ✅ 1.525 | ✅ 1.451 | ✅ 1.451 | ❌ NaN (n=0) |
| OOT n ≥ 200 | ✅ 3,112 | ✅ 5,268 | ✅ 5,268 | ❌ 0 |
| MC p5 > 1.0 | ✅ 1.420 | ✅ 1.247 (combined) | ✅ 1.367 | ❌ NaN |
| WF p5 > 1.0 | ❌ 0.363 | ❌ 0.367 | ❌ 0.474 | ❌ 0.400 |
| **Overall** | **DO NOT SHIP** | **DO NOT SHIP** | **DO NOT SHIP** | **DO NOT SHIP** |

**Lane closed.** No post-hoc tweak available in the design space we've
explored gets us past the WF p5 ship gate. The strategy has real edge
in some regimes (OOT PF 1.45, MC p5 1.37) but is too regime-dependent
to deploy unsupervised per the 8-point standard.

### Honest options going forward

1. **Capped paper deployment of SHORT-only v2 as-is**, with hard
   guardrails: $1k-$10k notional, manual kill-switch when BTC weekly
   crosses above weekly EMA-26, daily PnL monitoring. The OOT PF 1.451
   with MC p5 1.367 is real money in the right regime; the question is
   how much downside you're willing to take in the wrong regime in
   exchange for live data. **Decision is the user's, not engineering's.**
2. **Long-side rebuild** with a different exit (TP=8%/SL=4%/timeout=192)
   and retrained classifier — different problem, ~3 days of work, no
   guarantee of success.
3. **OI + funding-rate features upstream** — would lift the trade-level
   classifier's OOT AUC from 0.595 to (estimated) 0.62+. Won't fix WF p5
   alone but might make a *combined* meta-regime + trade-level system
   reach the gate. ~2 days of work.
4. **Fundamentally different strategy class** — abandon bigmover, try
   intraday mean-reversion, basis-trading, funding-arb, or other
   strategies with different regime profiles. New research lane.

The 8-point standard worked exactly as designed: it caught a strategy
that *looks* deployable on point estimates (PF 1.45, MC p5 1.37) but
isn't, because regime variance dominates the metric. Without the WF p5
gate this would have been deployed and lost money in 2024-Q4.

## Phase 11 — BTC oscillator features (RSI, KDJ, BB %B, Volume z-score)

User question: did we use RSI/KDJ/Bollinger/Volume on BTC for regime
classification? Answer: trend features only at the meta-regime layer.
This phase adds 5 BTC oscillator features and tests both ML + hand-
crafted variants.

Hand-crafted rule sweep on v2 SHORT trades (4,933 OOT after R1 filter):

| Rule | OOT n | OOT PF | WF p5 | MC p5 |
|---|---|---|---|---|
| Baseline (no rule) | 5,268 | 1.451 | 0.367 | 1.367 |
| **R1: skip when BTC RSI(14) < 20** | **4,933** | **1.596** | 0.369 | **1.504** |
| R2: skip KDJ J < −10 | 5,107 | 1.472 | 0.367 | 1.386 |
| R3: skip BB %B < 0.05 | 4,072 | 0.858 | 0.369 | 0.799 |
| R4: skip vol z > 2.5 | 4,533 | 1.122 | 0.309 | 1.050 |

**RSI confirms the user's hypothesis**: `RSI(14) < 20` is a clean
whipsaw signal. R1 alone lifts OOT PF +14% (1.451 → 1.596) and MC p5
+10% (1.367 → 1.504). Surprisingly, %B and volume z-score (textbook
whipsaw signals) HURT OOT — those touches/spikes continued lower in
this OOT period rather than snapping back.

ML meta-regime v2 (12 features incl. oscillators):
- Train AUC 0.9313 (overfit on 26 windows)
- OOT PF 1.302 (clears 1.30 floor)
- WF p50 = 1.302 (median window now profitable)
- WF P(loss) = 31.8% (down from 50%) — meaningful frequency improvement
- WF p5 = 0.323 (still below 1.0)
- Top scaled coefficient: `btc_rsi14_daily` (-0.304, dominant) confirms
  RSI as the most discriminating feature.

Phase 11b clean-combo sweep showed the locked-on-TRAIN combo (RSI≥30 +
momentum≤0.15) FAILS OOT (PF 0.657) because in OOT BTC RSI was below
30 frequently in continuation trades. **Only RSI≥20 catches the right
extreme — but that signal is invisible at TRAIN time under any
locking criterion**. Config D (momentum≤0.10 + RSI≥20) achieves OOT PF
1.596 / MC p5 1.504 / WF p5 0.474 but is overfit-by-OOT-selection.

## Phase 12 — Hierarchical multi-timeframe cascade gate

User proposed a 4-layer hierarchical AND-gate as a structurally
different approach: instead of flat ML on combined features, require
**each timeframe to independently confirm direction** before firing.

Layers (all using EMA + RSI + MACD + KDJ all-bearish agreement, with
indicator parameters textbook-default and shifted 1 bar for no-lookahead):

| Layer | Timeframe | Role |
|---|---|---|
| L1 | BTC daily (EMA 9/21/50) | direction + sizing |
| L2 | BTC 1h | BTC timing confirmation |
| L3 | coin 4h | coin structural agreement (per-asset) |
| L4 | coin 15m | v2 ml_on+sized (existing) |

**Per-layer pass rates** (the most informative finding):

| Period | n | L1 | L2 | L3 | L1∧L2∧L3 |
|---|---|---|---|---|---|
| TRAIN | 63,393 | 4.2% | 24.2% | 29.5% | **0.3%** |
| OOT | 5,268 | 15.4% | 28.3% | 39.5% | **3.0%** |

OOT was a sustained bear → full multi-TF alignment fires **10× more
often** than TRAIN. The cascade is structurally different across
periods.

**Cascade variants:**

| Variant | TRAIN n | TRAIN PF | OOT n | OOT PF | WF p5 | WF p50 | P(loss) | MC p5 |
|---|---|---|---|---|---|---|---|---|
| C0 baseline | 63,393 | 1.124 | 5,268 | 1.451 | 0.367 | 0.933 | 50.0% | 1.367 |
| C1 L1 only | 2,634 | 0.357 | 810 | 2.699 | 0.006 | 0.098 | 72.2% | 2.381 |
| C2 L1+L3 | 923 | 0.479 | 409 | 1.572 | 0.000 | 0.105 | 68.8% | 1.316 |
| **C3 L1+L2+L3 full** | **184** | **0.153** | **158** | **4.008** | 0.028 | **1.930** | **37.5%** | **3.063** |

C3 looks phenomenal on OOT — PF 4.008 with MC p5 3.063 — but TRAIN PF
is 0.153 (the cascade LOST money in training). **Same gate, opposite
outcome across periods.**

**Diagnosis:** in TRAIN periods, the rare moments when all 3 layers
aligned bearish coincided with bear-BOTTOM whipsaws (e.g., 2023-09 had
all-3-bearish at the local bottom). In OOT, the same alignment came
during sustained bear continuation. **The hand-crafted cascade cannot
distinguish these two states.** This is the same regime-asymmetry
problem we've seen in every phase, just with a different mechanism.

Per protocol, locking on TRAIN PF picks C0 (baseline) — ship gate
still fails on WF p5.

## Final post-mortem (12 phases, lane closed honestly)

**The aggregate 12-phase finding:**

| Phase | Approach | What it taught us |
|---|---|---|
| 0-7 | Build the bigmover + 16-feature classifier + sizing | Real per-trade edge from indicators (+39% PF lift over baseline). OOT works, WF doesn't. |
| 8 | Weekly meta-gate + exit sweep | Both rejected. Baseline 5%/15% exit wins. |
| 9 | Outcome-aligned label (v2) | Cleaner training, OOT PF 1.30+. WF unchanged — **WF is not a classifier accuracy problem**. |
| 10 | BTC momentum meta-gate (hand-crafted + ML) | Hand-crafted T=0.10 best, WF p5 0.474. ML overfits 26 windows. |
| 11 | BTC oscillator features | RSI(14)<20 is the cleanest whipsaw signal. R1 lifts OOT PF +14%. But threshold uninferrable from TRAIN. |
| 11b | Clean combo sweep | TRAIN-locked discipline correctly rejects the OOT-best Config D as overfit. |
| 12 | Multi-TF cascade | TRAIN/OOT pass rate 10× different. Same cascade fires on whipsaw bottoms in TRAIN and on continuation in OOT. |

**The structural truth:** crypto has 3 years of bigmover-relevant data
in this snapshot. With ~4 distinct macro regimes (2023 bear, 2024 bull,
mid-2024 chop, 2024-Q4 to 2025 bull, 2025-H2 to 2026-Q1 bear), every
honest classifier or rule trained on this data is selecting on AT MOST
2-3 regime examples per outcome. That's not enough samples for the
8-point standard's WF p5 > 1.0 to hold.

**Three honest options for the user (final):**

1. **Capped paper deployment of v2 SHORT-only baseline + Phase 11 R1 (RSI<20 guard)**.
   - Use SHORT-only v2 ml_on+sized.
   - Skip any signal where BTC daily RSI(14) < 20.
   - Cap notional at ≤0.20% of capital per trade (max 130 concurrent positions
     → max 26% deployed → worst-month equity drawdown ≤ −10%).
   - Hard kill switch: pause all new entries when BTC weekly close > weekly
     EMA-26 OR portfolio drawdown exceeds −8% in a calendar month.
   - Run 2-3 months for live signal-to-fill data.
   - **This is the protocol-soft option** — accepting WF p5 < 1 as a known
     risk in exchange for live data + small position size.

2. **Stop and pivot to a different strategy class.**
   - Funding-rate arb, basis trading, intraday mean-reversion have very
     different regime profiles.
   - The bigmover/momentum-short space has been thoroughly explored.

3. **Backfill 10 years of data + try again** (if a Binance-archive backfill
   is feasible). 120 monthly windows instead of 26-34 might allow the meta-
   regime classifier and Phase 11 RSI threshold to lock cleanly. ~3 days of
   work + uncertain regime-stationarity gains.

The 8-point standard worked: this strategy WOULD have been deployed and
lost money in 2024-Q4 / 2024-02 / 2023-Q4 without it.

---

## Phase 13 — regime-aware early exit (failed)

Tested closing positions when BTC's regime flipped at 1h, daily, or 15m
timeframes (full EMA + RSI + MACD + KDJ agreement). Every variant degraded
OOT PF: too noisy without a confirmation filter, fires on routine bounces
rather than the violent rallies that kill SHORT positions.

| Variant | OOT PF | Δ vs baseline |
|---|---|---|
| baseline | 1.451 | — |
| 1h regime exit | 1.227 | −15% |
| daily regime exit | 1.342 | −8% |
| 15m regime exit | 1.208 | −17% |

Lesson: regime flips alone are too noisy. Need to confirm with price action.

## Phase 14 — strict entry + rapid-rally exit (V2 won)

Took two ideas forward:
- **V1**: only enter when BTC daily AND BTC 1h are both ALL-BEARISH.
- **V2**: only exit when BTC regime flips AND BTC moved > +3% in last 24h.
- **V3**: V1 entry + V2 exit.

| Variant | OOT PF | WF p5 |
|---|---|---|
| V0 baseline | 1.451 | 0.474 |
| V1 strict entry | 1.422 | 0.553 |
| **V2 rapid-rally exit** | **1.668** | **0.553** |
| V3 strict entry + V2 exit | 1.622 | 0.589 |

**V2 won.** Adding the +3% confirmation filter to the regime-exit signal cut
the noise that broke Phase 13. WF p5 still 0.553 (below 1.0 ship floor).

## Phase 15 — ATR-normalized detector vs volume-only (D1 won, step-change)

User hypothesis: the bigmover-volume detector triggers on raw % moves +
raw volume spikes, which mean different things on BTC (low vol, 5% drop
is rare) vs alts (high vol, 5% drop is daily noise). An ATR-normalized
trigger should adapt to each asset's volatility regime.

Three detectors compared head-to-head, same SHORT-side exits + sign-locked
sizing + Phase-14 V2 rapid-rally exit:

| Detector | Trigger | OOT PF | WF p5 | MC p5 |
|---|---|---|---|---|
| D0 bigmover-volume (current) | vol≥3 & drop≥5% | 1.882 | 0.565 | 1.729 |
| **D1 ATR-drop + vol** | drop≥2.5×ATR & vol≥2 & red bar | **2.064** | **0.605** | **1.947** |
| D2 ATR-scaled drop | drop≥3×ATR/close & vol≥2 | 1.766 | 0.533 | 1.657 |

**D1 won.** ATR normalization is a step-change improvement: WF p5 is the
highest measured across 15 phases. Two surprise findings:
1. **The v2 ML classifier was destructive on D1 trades.** Removing it
   improved every metric (TRAIN PF +42%, OOT PF +13%, WF p5 +83%) — the
   v2 classifier was trained on the D0 distribution, not D1.
2. ATR-normalization validates the user's intuition that fixed-% thresholds
   don't translate across volatility regimes.

WF p5 0.605 — still below 1.0 ship floor, but the highest ever.

## Phase 16 — D1 extensions (LONG-side, fresh classifier, RSI<20 meta-gate)

Three follow-ups on the D1 winner.

### Phase 16.1 — D1 LONG side (FAILED, ship SHORT-only)

Mirror detector for LONG: `(close - rolling_low_96) / ATR_14 ≥ 2.5 AND vol ≥ 2 AND green bar`,
LONG sign-locked sizing, LONG-mirror rapid-bear exit (BTC daily ALL-BULLISH AND
BTC 24h return < −3%).

| Detector | TRAIN PF | OOT PF | WF p5 | MC p5 |
|---|---|---|---|---|
| D0L bigmover-volume LONG | 1.265 | 0.608 | 0.600 | 0.542 |
| D1L ATR-rise LONG | 1.215 | 0.596 | 0.542 | 0.555 |

**LONG side stays broken.** ATR normalization didn't fix the long-side
asymmetry — drops continue, rallies don't. Crypto's structural "drops are
bigger than rallies" property holds even under ATR scaling. Strategy stays
SHORT-only.

### Phase 16.2 — Fresh classifier on D1 SHORT trades (CLASSIFIER HELPS)

The Phase-15 finding that the v2 classifier hurt was a *distribution* issue,
not a *model* issue: it was trained on D0 trades. A fresh LR (C=1.0,
class_weight='balanced', 5-fold purged time-CV @ purge=192, outcome label,
same 16-feature set) on the D1 distribution produces a real, honest edge.

Training metrics:
- 74,925 D1 SHORT trades, outcome positive rate 42.0%
- CV AUCs: 0.538 / 0.569 / 0.532 / 0.551
- Full-train AUC 0.567, OOT AUC 0.562, **gap 0.005** (no overfit)
- Threshold locked at TRAIN q0.5 = 0.4991

Backtest with classifier gate:

| Variant | OOT n | OOT PF | WF p5 | WF P(loss) | MC p5 |
|---|---|---|---|---|---|
| D1 raw (Phase 15) | 5,467 | 2.067 | 0.605 | 26.5% | 1.945 |
| **D1 + v3 LR classifier** | **2,336** | **2.607** | **0.719** | **11.8%** | **2.371** |

**Classifier kept.** Every metric improves: OOT PF +26%, WF p5 +19%,
WF P(loss) cut from 26.5% → 11.8%, MC p5 +22%. The lesson from Phases 9
and 10 still holds (more model complexity overfits tiny windows), but a
distribution-matched LR is genuinely informative.

### Phase 16.3 — Stack RSI<20 meta-gate (REDUNDANT, dropped)

Phase 11 R1 found skipping BTC daily RSI(14) < 20 lifted Phase-9 v2 OOT
PF +14%. Tested whether it stacks on D1.

| Variant | OOT n | OOT PF | WF p5 |
|---|---|---|---|
| D1 raw | 5,467 | 2.067 | 0.605 |
| D1 + RSI≥20 | 5,176 | 2.053 | 0.605 |
| D1 + v3 classifier | 2,336 | 2.607 | 0.719 |
| D1 + classifier + RSI≥20 | 2,294 | 2.577 | 0.719 |

**Dropped.** The RSI<20 gate excludes only 1.1% of D1 trades — the sign-locked
BTC trend score sizing already filters the same regime. No incremental edge.

---

## Phase 16 — final locked configuration

The strategy that ships at the end of this research lane:

| Component | Specification |
|---|---|
| Direction | **SHORT-only** (LONG side broken under both detectors) |
| Detector | **D1**: `(rolling_high_96 − close) / ATR_14 ≥ 2.5 AND volume / vol_MA_20 ≥ 2.0 AND close < open` |
| Cooldown | 96 bars (24h) per asset |
| Sizing | Sign-locked Candidate-B BTC trend score (4h MACD spread, K=0.005255), only enter when score < 0; pos_scale = min(1.5 · |score|, 1.5) |
| Classifier | LR (C=1.0, class_weight='balanced', 16-feature set + is_short, outcome label, threshold = TRAIN q0.5 = 0.4991) — `services/python/models/d1_short_v3.joblib` |
| Entry exit — SL | −5% from entry |
| Entry exit — TP | +15% from entry |
| Entry exit — timeout | 672 bars (7 days) |
| Entry exit — friction | 0.0015 |
| Rapid-rally exit | Close at next bar if `(NOT BTC daily ALL-BEARISH)` AND `BTC 24h return > +3%` |
| Snapshot reproducibility | `data/snapshots/candles_15m_2026-04-01.parquet` |

**Locked ship-gate metrics (D1 + v3 classifier + rapid-rally exit, SHORT-only):**

| Metric | Value | Ship floor | Pass? |
|---|---|---|---|
| OOT PF | 2.607 | ≥ 1.30 | **PASS** |
| OOT n | 2,336 | ≥ 200 | **PASS** |
| MC bootstrap p5 | 2.371 | > 1.00 | **PASS** |
| WF p5 (34 folds) | 0.719 | > 1.00 | **FAIL** |
| WF P(loss) | 11.8% | — | (cut from 26.5% raw) |

WF p5 still does not clear 1.0. The 8-point standard's WF gate continues
to block hard live deployment. But this is the highest WF p5 measured
across 16 phases, with WF P(loss) under 12%.

## Updated deployment recommendation

The final option, given Phase 16's improvements:

**Capped-paper deployment of D1 + v3 classifier + rapid-rally exit (SHORT-only).**

- Use the locked configuration table above. No live LONG positions.
- Cap notional at ≤ **0.30 % of capital per trade** (up from 0.20% — the
  P(loss) cut from 26.5% → 11.8% justifies the modest size lift).
- Max concurrent positions: 100 (max 30% deployed). Worst-month historical
  drawdown projection ≈ −7% to −10% on the WF p5 fold.
- Hard kill switches:
  - Pause new entries when BTC weekly close > weekly EMA-26 (bull market).
  - Pause new entries when portfolio drawdown exceeds −8% in a calendar month.
  - Pause new entries when classifier OOT AUC drops below 0.53 over a
    rolling 60-trade window (model decay early-warning).
- Log every signal-to-fill latency, slippage, and gate-rejection reason for
  3 months before any sizing increase.
- Re-train v3 classifier monthly on the rolling 24-month window.

**Promotion criteria** (from paper to live):
- 60+ consecutive trading days of paper data.
- Paper PF ≥ 1.50 net of fees AND drawdown ≤ −12%.
- WF p5 on the rolling 24-month window ≥ 0.80 (not yet 1.0 — explicit
  acceptance of below-floor risk in exchange for the live edge).
- User explicit go-decision; never auto-promoted.

If WF p5 fails to lift toward 1.0 over 6 months of paper data, revert to
options 2 or 3 in the prior section (pivot strategy class or backfill
deeper history).

---

## Phase 17 — Four-strand parallel search (3 nulls + 1 stacking win)

After Phase 16 locked, four orthogonal research strands ran in parallel to
push WF p5 past the 1.0 ship floor: detector composition (17.A), 3-layer
regime cascade (17.B), indicator feature expansion (17.C), and sizing × exit
sweep (17.D). Each strand used pre-committed thresholds; the synthesis
stacked the winners.

| Strand | Decision rule outcome | Verdict |
|---|---|---|
| 17.A — D1e detector (D1 + universe-down-breadth ≥ 0.6) | TRAIN PF max + OOT n ≥ 200 + OOT PF ≥ 2.064 + WF p5 ≥ 0.605 — all PASS | **WIN (modest)**: OOT PF +16% (2.064 → 2.399), WF p5 +0.015 |
| 17.B — 3-layer regime cascade (BTC 4h × coin daily × coin 1h) | C2 (L1 hard veto) passes by +0.001 WF p5 (only 0.16% of D1 entries are in BTC 4h all-bullish, already filtered by sign-locked sizing). C1/C3 actively hurt | **DROP — null** |
| 17.C — Bollinger %B/width/squeeze + ATR%/EMA21-cross-EMA50 added to feature set | OOT AUC drops 0.5623 → 0.5535; OOT PF regresses 2.607 → 2.432; every new feature has *negative* OOT-AUC delta when included | **DROP — null** (16-feature LR is saturated) |
| 17.D — 4×4 sizing × exit grid | S4 classifier-weighted sizing × E2 ATR-based exits: OOT PF 2.607 → 5.422, **WF p5 0.719 → 2.474 (clears 1.0 ship floor for the first time)** | **WIN (large)** |

### Phase 17.E — synthesis (D1e + retrained v4 classifier + S4 × E2)

Per the project memory rule "retrain classifier when detector changes," 17.A's
D1e detector required retraining. The synthesis run re-detected D1e SHORT
trades (62,298 vs D1's 74,907), trained a v4 LR (TRAIN AUC 0.5644, OOT AUC
0.5374, gap 0.027), then applied S4 sizing × E2 ATR exits.

**Every ship-gate metric improved over 17.D-alone:**

| Metric | Phase 16 locked | 17.D alone | **17.E stacked** | Floor | Pass? |
|---|---|---|---|---|---|
| OOT PF | 2.607 | 5.422 | **6.429** | ≥ 1.30 | **PASS** |
| OOT n | 2,336 | 2,298 | 1,945 | ≥ 200 | **PASS** |
| MC bootstrap p5 | 2.371 | 4.550 | **5.536** | > 1.00 | **PASS** |
| WF p5 (34 folds) | 0.719 | 2.474 | **2.504** | > 1.00 | **PASS** |
| WF P(loss) | 11.8% | — | **0.0%** | — | (zero of 34 folds had PF<1) |

**This is the first configuration in 17 phases of research to clear all four
8-point ship gates.** Reproducibility: PASS, byte-identical between two runs.

## Phase 17 — final locked configuration

| Component | Specification |
|---|---|
| Direction | **SHORT-only** (Phase 16.1 confirmed long is structurally broken) |
| Detector | **D1e**: `(rolling_high_96 − close) / ATR_14 ≥ 2.5 AND volume / vol_MA_20 ≥ 2.0 AND close < open AND universe_down_breadth_4h ≥ 0.6` |
| Breadth panel | Same construction as `services/python/src/ml/bigmover_combined/features.py:_ret_4h_panel`; lookback = 16 bars (4h) |
| Cooldown | 96 bars (24h) per asset |
| Sign-lock filter | Enter only when Candidate-B BTC trend score < 0 |
| Sizing (S4) | `pos_scale = clip(3.0 × (classifier_score − 0.5), 0.0, 1.5)` — zero-sizes low-confidence trades |
| Classifier | v4 LR (C=1.0, class_weight='balanced', 16-feature set + is_short, outcome label, threshold = TRAIN q0.5 = 0.5010) — `services/python/models/d1e_short_v4.joblib` |
| Entry exit — SL (E2) | `entry_price + 2 × ATR_14_at_entry` (volatility-adapted) |
| Entry exit — TP (E2) | `entry_price − 6 × ATR_14_at_entry` (3:1 R:R preserved) |
| Entry exit — timeout | 672 bars (7 days) |
| Entry exit — friction | 0.0015 |
| Rapid-rally exit | Close at next bar if `(NOT BTC daily ALL-BEARISH)` AND `BTC 24h return > +3%` |
| Snapshot reproducibility | `data/snapshots/candles_15m_2026-04-01.parquet` |

### Realistic equity simulation (5%/5x sizing, max 5 concurrent, compounding)

| Metric | Phase 16 (D1+v3+S1×E1) | **Phase 17.E (D1e+v4+S4×E2)** |
|---|---|---|
| Final equity (36 months) | 75.99x | 54.72x |
| Best month | +74.5% | +35.3% |
| Worst month | −11.6% | **−0.3%** |
| Median month | +8.21% | +10.53% |
| **Negative months (out of 36)** | 7 | **1** |
| Realistic monthly P(loss) | 19.4% | **2.8%** |
| Signals dropped on 5-concurrent cap | 94.7% | 85.1% |

Phase 17.E has lower headline equity but **dramatically smoother risk**:
worst month went from −11.6% to −0.3%, and only 1 of 36 months printed a
loss (vs 7). The strategy gives up tail upside (smaller best months) for
tail safety. This is the correct tradeoff under CLAUDE.md's risk regime
(5% SL, MAX_CONCURRENT 5).

### Updated deployment recommendation (capped paper, then live)

The strategy now passes the 8-point ship gate. Two ship-passing configs are
on the table, picked from the Phase 17.D 4×4 grid and re-verified on the
D1e+v4 ledger by `phase17_f_verify_s1e2.py`:

| Config | OOT PF | WF p5 | MC p5 | WF P(loss) | 36-mo equity¹ | Worst month |
|---|---|---|---|---|---|---|
| **S1 × E2 (priority paper)** | 3.343 | 1.415 | 2.971 | 0.0% | **10,424x¹** | −5.68% |
| S4 × E2 (max-safety) | 6.429 | 2.504 | 5.536 | 0.0% | 54.72x¹ | −0.25% |

**¹ Realism caveat (added 2026-04-26 by `phase17_g_sizing_audit.py`).**
The 36-month equity multiples in this table chain across the full
2023-04 → 2026-04 trade ledger, ~33 months of which are in-sample for the
v4 classifier. They also use `pos_scale_cap = 1.5` (above CLAUDE.md's 125%
gross-notional cap), no notional cap, and flat 0.15% friction. Under a
tightened realism stack — OOT-only (last 3 months), `pos_scale_cap = 1.0`,
$50k notional cap, and 0.5% slippage in breadth-gated entries — S1×E2
compounds **1.72× over the 3-month OOT window**. Naive
12-fold extrapolation of that rate yields ~672×; the in-sample 36-month
sim under the same realism stack yields ~119×. Both are very far from
10,424×. **Do not use the 10,424× number for sizing decisions.** See
`services/python/results/phase17_g_sizing_audit.json` for the full table
and `phase17_g_sizing_audit.py` for the methodology.

**User-preferred for paper trade: S1 × E2.** Linear sign-locked Candidate-B
sizing (`pos_scale = min(1.5 × |btc_score|, 1.5)`) + ATR-based exits
(SL = entry + 2 × ATR_14, TP = entry − 6 × ATR_14). Higher headline PnL,
slightly looser ship-gate metrics, more exposure to live slippage.

Reasoning for prioritizing S1×E2:
- Both configs pass the 8-point ship gate cleanly. There is no protocol
  reason to prefer one.
- S1×E2 captures bigger TPs on high-volatility alts (where 6×ATR is far
  larger than the fixed 15%). S4 caps these by zero-sizing low-confidence
  trades.
- Headline 36-month equity claims (10,424× vs 54.72×) are inflated — see
  the realism caveat above. The honest forward-looking range is much
  narrower; both configs likely sit in 30–700× over 36 months once
  in-sample, gross-notional, capacity, and slippage are corrected.
  Re-running `phase17_g_sizing_audit.py` for S4×E2 is open work.
- The cost is wider drawdowns: −5.68% worst month vs −0.25% under S4. This
  is the explicit tradeoff the user accepted by picking S1×E2.

**Recommendation:**

1. **30-day capped paper** at 0.30% notional per trade under S1×E2.
   - Max 5 concurrent positions per CLAUDE.md.
   - Hard kill switches:
     - Pause new entries when BTC weekly close > weekly EMA-26 (bull market).
     - Pause new entries when portfolio drawdown exceeds −7% in a calendar
       month (tighter than the S4 case because S1 has wider tails).
     - Pause new entries when v4 OOT AUC drops below 0.52 over a rolling
       60-trade window (model decay early-warning).
   - Log every signal-to-fill latency, slippage, and gate-rejection reason.
2. **If paper looks good** (PF ≥ 2.0 net of slippage, drawdown ≤ −7%, no
   kill-switch trip): move to small live with 1.0% notional per trade for
   another 60 days.
3. **Promotion to full size** (5% × 5x = 25% notional per CLAUDE.md):
   60 days of small-live with paper-tracking error < 30%, and explicit
   user go-decision.

**S4 × E2 stays in the codebase as the conservative fallback config**: if
S1×E2 paper trade prints worst-month worse than −10%, switch to S4×E2 (same
code path, just change the sizing variable).

**Re-train v4 monthly** on the rolling 24-month window. Re-build the
breadth panel monthly (universe membership changes).

### Risk caveats (must read before any live deploy)

1. **OOT is still only 3 months.** The headline 6.4 OOT PF is on 1,945
   trades over 2026-01 to 2026-04. The walk-forward distribution is what
   matters for confidence; WF p5 = 2.504 is robust but the snapshot
   captures only ~3 distinct macro regimes. Live data over 6+ months of
   different conditions is the real test.
2. **Multiple-comparisons risk.** Phase 17.D explored a 16-cell grid; we
   picked the winning cell. Phase 17.E stacked 17.A's winner. Each level
   of selection compounds overfit risk. The pre-committed thresholds and
   reproducibility checks help but don't eliminate it.
3. **v4 has a wider TRAIN-OOT AUC gap (0.027) than v3 (0.005)**, hinting at
   more overfit on the smaller, breadth-filtered D1e distribution. Monitor
   live AUC closely; the 0.52 kill-switch is tight on purpose.
4. **The breadth gate fires only when many alts are co-falling.** This is
   exactly when liquidity gets worst. Real slippage in those moments may
   exceed the modeled 0.0015 friction by 2-5×, which would compress PF.
5. **CLAUDE.md sizing was kept fixed.** Do not increase 5% per trade × 5x
   leverage in pursuit of the 54.72x backtest equity number; that figure
   is *with* CLAUDE.md sizing, not on top of it.

Artifacts:
- Scripts: `services/python/scripts/phase17_{a,b,c,d,e}_*.py`,
  `phase17_perf_breakdown.py`.
- Models: `services/python/models/d1e_short_v4.{joblib,_meta.json,_sizing.json}`.
- Trade ledgers: `services/python/data/d1e_short_trades{,_with_features}.csv`.
- Results JSON: `services/python/results/phase17_{a,b,c,d,e}_*.json`.
