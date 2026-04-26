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
