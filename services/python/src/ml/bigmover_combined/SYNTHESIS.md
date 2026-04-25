# Phase-3 specialist team — synthesis (bigmover-combined-ml v1)

5 specialists analyzed the 6 bigmover baseline trade CSVs (3 signals × 2
directions) on the 2026-04-01 snapshot. Each ranked candidate features by
marginal AUC and walk-forward stability across 8 non-overlapping ~4.5-month
windows. This document records their findings and the rejection rationale
so the final feature set in `features.py` is auditable.

## Final feature set (16 + 1 metadata)

| Group | Feature | AUC (best dir) | Walk-forward p5 | Picked because |
|---|---|---|---|---|
| Bigmover internals | vol_ratio, price_drop_pct, price_rise_pct, accel_atr_norm, bars_since_high32 | n/a | n/a | These are the detector's own internals. They go into the model so it can learn interactions with the other features. |
| EMA | **price_vs_ema9_pct** | 0.534 (short) | 0.513 | Strongest single EMA feature on shorts; flips sign cleanly for longs. 12.5pp WR spread top-vs-bottom decile. |
| EMA | **price_vs_ema50_pct** | 0.524 (short) | 0.512 | Slow-context complement to EMA9. Stable in all 8 walk-forward windows. |
| RSI | **rsi14_delta_1bar** | 0.555 (long) / 0.440 (short) | 0.50–0.58 | Cleanest direction-flipping AUC; 9.8pp Δ-WR long, 10.5pp Δ-WR short. Deterministic, sign-flips by direction as theory predicts. |
| RSI | **rsi14_delta_4bar** | 0.522 (long) / 0.450 (short) | 0.43–0.55 (tightest band) | Near-orthogonal horizon (1h vs 15m), tightest WF band of any feature. Adds little correlation cost on top of 1-bar delta. |
| MACD | **macd_signal_spread_norm** | 0.491 (short) / 0.483 (long) | ~0.47–0.54 | Monotonic on long-side (0.296 → 0.263 WR by quartile). Sign-flips by direction. |
| MACD | **macd_hist_momentum** | 0.485 (short) / 0.506 (long) | 0.46–0.53 | Captures *jerk* (2-bar Δ of histogram), the only MACD feature that's truly orthogonal to EMA distance. |
| KDJ | **kdj_j** | 0.527 (short) / 0.511 (long) | 0.51–0.56 | Best AUC on shorts; 8/8 WF windows stable above 0.51. Single feature covers both directions with opposite sign (low-J helps shorts, high-J helps longs). |
| KDJ | **kdj_k** | 0.515 (long) / 0.523 (short) | 0.50–0.54 | Best on longs; smoother than J (no extreme outliers). Position-within-channel info, complementary to EMA distance. |
| BTC | **btc_trend_score** *(Candidate B)* | n/a (sizing/gate) | n/a | Monotone diagnostic +1.473 — only formula where LONG WR rises monotonically with score AND SHORT WR falls monotonically. Wrong-side-trade WR penalty: SHORTS at score>+0.6 get WR=0.273 vs SHORTS at score<-0.6 WR=0.351 (+7.8 pp). LONGS less sensitive (+1.8 pp). |
| BTC | btc_realized_vol_24h | (unranked) | (unranked) | Standard regime context; cheap to compute. |
| Cross-section | concurrent_dir_breadth | (unranked) | (unranked) | Distinguishes asset-specific vs market-wide setups. |
| Metadata | is_short | n/a | n/a | Direction dispatch flag for the single-classifier setup. |

## Rejected candidates (and why)

- `price_vs_ema20_pct` / `ema20_minus_ema50_atr` — EMA9 + EMA50 already cover
  short-end and slow-end; adding EMA20 is correlated, no marginal AUC.
- `ema_cross_state`, `ema20_slope`, `ema_ribbon_score` — minimal Δ-WR,
  near-coin-flip walk-forward (≤3pp).
- `rsi14` (level), `rsi14_zone` (overbought/oversold flag),
  `rsi14_divergence_8bar` — AUC ~0.51-0.54, near-noise. Textbook claim
  "RSI > 70 = good for short" was *partially refuted* on bigmover trades:
  RSI extremes flag reversal-prone bars regardless of direction; oversold
  beats overbought for shorts (counter to textbook).
- `macd_norm` (level) — redundant with EMA distance features.
- `macd_hist_sign_change`, `macd_zero_cross_4bar`, `macd_signal_cross_4bar` —
  AUCs straddle 0.5 in walk-forward, no edge.
- `kdj_j_minus_k`, `kdj_k_above_d`, `kdj_extreme_zone`, `kdj_cross_4bar` —
  highly correlated with raw J/K; cross-event flags don't help. Notably the
  classical J>100 (overbought) flag is *anti*-predictive: it marks
  exhaustion of the bigmover thrust, not continuation.
- BTC trend score Candidates A (daily SMA50 deviation), C (multi-TF
  consensus), D (24h-return z-score) — A and D went *the wrong way* for
  longs (high score → low WR, the opposite of intent); C is too
  quantized to give useful gradient.

## Locked parameters (frozen for reproducibility)

```python
EMA_FAST = 9
EMA_SLOW = 50
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
KDJ_N, KDJ_K_SMOOTH, KDJ_D_SMOOTH = 9, 3, 3
# Candidate B trend score:
BTC_TREND_K = 0.005255           # 90th-percentile of |MACD-signal spread / close| on 4h BTC
BTC_TREND_TIMEFRAME = "4h"
```

The Phase-3 specialist scratch scripts and CSVs (e.g., `kdj_feature_short.csv`,
`ema_feature_short.csv`) are intentionally NOT committed — they were
exploration artifacts, not deployment artifacts. The committed audit trail is
this synthesis + the locked parameters in `features.py`.
