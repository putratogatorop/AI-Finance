# V4 Design Spec: LightGBM 15-Min Crypto Futures Trading System

**Date:** 2026-04-07
**Status:** Approved
**Target:** +10-20%/month at 3-5x leverage on Gate.io perpetual futures
**Approach:** Supervised ML (LightGBM) signal generation + rule-based execution + adaptive risk management

> This spec synthesizes advice from 8 expert agents: RL specialist, quant trader, ML engineer, market structure expert, crypto hedge fund PM, adaptive systems architect, strategy architect, risk manager, feature engineer, and execution specialist.

---

## 1. Why This Approach (Lessons from v3 RL Failure)

| v3 Problem | v4 Solution |
|---|---|
| LSTM training broken (PPO shuffled batches) | LightGBM — no temporal state issues |
| 178 alts, 534-dim action space | 5 alts, model outputs probability only |
| 505 days of 4h data | 3 years of 15min data (~105K candles/coin) |
| BTC-alt propagation dead on 4h | 15min can still catch short-term lead-lag |
| 94 trades/45 days = 28% fee drag | 5-15 trades/week, high conviction only |
| Reward function chaos | No reward — supervised classification with clear target |
| 0% win rate on holdout | Walk-forward with 24+ out-of-sample periods |

## 2. Architecture Overview

```
Data Pipeline (15min OHLCV + Funding + OI from Binance Vision / Gate.io)
    |
    v
Feature Engineering (30 features per coin, recalibrated for 15min)
    |
    v
LightGBM Models (1 per coin, walk-forward retrain every 2 weeks)
    |
    v
Confidence Score System (combines model output + regime + accuracy)
    |
    v
Entry Filters (chase filter, volatility, RSI, correlation gate)
    |
    v
Position Sizing (conviction-scaled, ATR-based stops, correlation-adjusted)
    |
    v
Execution Engine (limit orders, trailing stops, time exits)
    |
    v
Risk Management (5-tier drawdown, circuit breakers, anti-tilt, sleep guard)
    |
    v
Performance Tracker + Adaptive Retraining (triggered, not calendar)
```

## 3. Coins & Data

### Universe: 5 Liquid Alts
Pick from: SOL, DOGE, XRP, AVAX, LINK, SUI, PEPE — criteria: high volume on Gate.io, perpetual contract available, tight spreads.

### Data Requirements
- **15-minute OHLCV**: 3 years, from Binance Vision (free)
- **Funding rate history**: from Gate.io API (every 8h)
- **Open interest history**: from Gate.io API
- **Taker buy volume**: from Binance Vision (taker_buy_base column)
- **BTC + ETH**: same data as reference assets

### Data Volume
- 3 years x 96 candles/day x 365 days = ~105,000 candles per coin
- 5 coins + BTC + ETH = 7 assets total
- Storage: ~500MB in PostgreSQL

## 4. Feature Set (30 Features Per Coin)

All periods calibrated for 15-minute candles.

### Momentum (5)
1. `ret_4` — 1-hour return
2. `ret_16` — 4-hour return
3. `ret_96` — 1-day return
4. `ret_672` — 7-day return
5. `ret_4_lag1` — lagged 1h return (autocorrelation)

### Trend / Mean-Reversion (4)
6. `price_to_sma_96` — distance from 1-day mean
7. `price_to_sma_672` — distance from 7-day mean
8. `ema_ratio` — EMA(48)/EMA(104) (MACD equivalent)
9. `macd_hist_norm` — MACD histogram, price-normalized

### Oscillators (2)
10. `rsi_norm` — RSI(56) normalized to [-1,1]
11. `bb_pct` — Bollinger Band position, BB(80)

### Volatility (4)
12. `vol_4h` — 4-hour realized vol
13. `vol_1d` — 1-day realized vol
14. `vol_ratio` — vol_4h / vol_1d (compression/expansion)
15. `parkinson_vol` — Parkinson volatility (H/L range)

### Volume (2)
16. `volume_ratio` — current / SMA(96) volume
17. `taker_buy_ratio` — buy-side aggression (from Binance Vision)

### Microstructure (2)
18. `clv` — close location value [-1, +1]
19. `drawdown` — distance from 7-day high

### Funding Rate (4) — HIGHEST PRIORITY NEW FEATURES
20. `funding_rate` — raw current rate
21. `funding_ma_3d` — 3-day moving average
22. `funding_zscore` — rolling z-score (32-day window)
23. `cum_funding_3d` — cumulative cost over 3 days

### Open Interest (1)
24. `oi_change_pct` — 4-hour OI change

### Cross-Asset (3)
25. `btc_ret_96` — BTC 1-day return (broadcast)
26. `btc_residual` — coin return minus beta * BTC return
27. `altcoin_dispersion` — std of returns across 5 coins

### Time/Session (2)
28. `hour_sin` — cyclical hour encoding
29. `hour_cos` — cyclical hour encoding

### Interaction (1)
30. `funding_x_rsi` — funding_zscore * rsi_norm (extreme sentiment + extreme price)

## 5. Model: LightGBM

### Target Variable
```
target = classify(forward_return_16bars / realized_vol_96bars)
  SHORT: bottom 20%
  FLAT:  middle 60%
  LONG:  top 20%
```
Predicts risk-adjusted 4-hour forward returns. The 60% FLAT zone ensures we only trade on strong signals.

### Training
- **Separate model per coin** (5 models)
- **Walk-forward**: train 6 months, validate 1 month, test 1 month, slide 2 weeks
- **~24 out-of-sample test periods** from 3 years of data
- **Retrain every 2 weeks** (or triggered by performance degradation)

### Hyperparameters
```python
params = {
    "objective": "multiclass",
    "num_class": 3,
    "num_leaves": 47,
    "min_child_samples": 200,
    "learning_rate": 0.01,
    "feature_fraction": 0.6,
    "max_depth": 7,
    "n_estimators": 2000,
    "early_stopping_rounds": 50,
}
```

### Signal Conversion
```python
signal = model.predict_proba(features)  # [p_short, p_flat, p_long]
conviction = signal[2] - signal[0]      # ranges -1 to +1
# Only trade when abs(conviction) > 0.35
```

## 6. Confidence Score System

Combines model output with regime context:

| Component | Weight | Source |
|---|---|---|
| Model signal strength | 50% | abs(conviction) scaled |
| Regime agreement | 20% | trend alignment, Hurst, funding contrarian |
| Feature quality | 10% | vol regime penalty |
| Recent model accuracy | 20% | rolling 50-trade accuracy |

### Thresholds
| Score | Label | Action | Leverage |
|---|---|---|---|
| 0.00-0.55 | No/weak signal | Skip | 0x |
| 0.55-0.65 | Slight edge | Min size | 0.3-0.7x |
| 0.65-0.75 | Moderate | Standard | 0.7-1.5x |
| 0.75-0.85 | Strong | Above avg | 1.5-3.0x |
| 0.85-0.95 | High | Large | 3.0-4.5x |
| 0.95-1.00 | Maximum | Cap at max | 4.5-5.0x |

## 7. Position Sizing

```
risk_per_trade = base_risk * conviction_multiplier * vol_multiplier * correlation_multiplier
position_value = risk_amount / stop_distance_pct
leverage = position_value / equity (capped at max_leverage)
```

### Stop Loss: ATR-Based
- Low vol: 2.5x ATR
- Normal: 2.0x ATR
- High vol: 1.5x ATR
- Hard cap: min 2%, max 10%

### Take Profit
- TP = stop_distance * 2.0 (minimum 1.5x reward:risk)

### Trailing Stop
- Activate at +1.5 ATR profit → move stop to breakeven
- After +3 ATR → trail at 2 ATR behind high-water mark

## 8. Entry Filters

Before placing any trade:
1. **Chase filter**: reject if price moved >1.5% in last hour or >2.5% in 2 hours
2. **RSI filter**: no longs above RSI 72, no shorts below RSI 28
3. **Volatility filter**: no entries when ATR > 2x average
4. **Correlation gate**: reject if >3 same-direction positions with correlation >0.70
5. **Sleep window**: no new entries within 2 hours of bedtime
6. **Portfolio risk**: total risk < 15% of equity

## 9. Execution

### Order Types
- Normal entry: LIMIT (post-only, save fees)
- High conviction: LIMIT with market fallback after 30min
- Stop loss: STOP-MARKET on exchange (always)
- Take profit: LIMIT (GTC)
- Signal flip exit: MARKET (urgent)

### Exit Priority Stack
1. Hard stop loss (2.5% on price)
2. Model signal flip (conviction reverses)
3. Trailing stop (after activation)
4. Take profit (5% on price)
5. Time exit (16h soft, 24h hard)

### Fees
- Maker: 0.015% (limit orders)
- Taker: 0.05% (market orders)
- Target: mostly maker = 0.03% round-trip avg
- Budget: <8% annual fee drag

## 10. Risk Management

### 5-Tier Drawdown System
| Drawdown | Action | Max Positions | Max Leverage |
|---|---|---|---|
| 0-5% | Normal | 5 | 5x |
| 5-10% | Reduce | 3 | 3x |
| 10-15% | Half | 2 | 2x |
| 15-20% | Minimal | 1 | 1.5x |
| 20%+ | HALT | 0 | 0x, 48h cooldown |
| 25%+ | KILL SWITCH | 0 | 0x, 7d cooldown, manual restart |

### Circuit Breakers
- BTC drops 8% in 1h → close all, 24h cooldown
- BTC drops 12% in 4h → close all, 24h cooldown
- Portfolio drops 5% in 1h → close all, 24h cooldown
- Stablecoin depeg (USDT < $0.995) → close all, withdraw
- Extreme funding (>0.3%/8h) → close affected position
- API failure (3 consecutive) → close all

### Anti-Tilt
- 5 consecutive losses → 24h cooldown
- 4 losses in 24h → 24h cooldown
- Win rate <25% over 20 trades → halt until manual review
- Human can override max 2 signals/day

### Sleep Guard (8h unattended)
- All positions must have exchange-level stop orders
- Max 8% drawdown in 8h → emergency close-all
- No new entries within 2h of sleep
- 60-second API heartbeat
- Telegram alerts for any trigger

### Safe Compounding
- Quarter Kelly for position sizing
- Only compound 50% of profits into sizing
- Cap monthly capital growth at 10%
- After losing month: no capital reduction (floor at base)

## 11. Adaptive Layer

### Performance Tracking
- Log every prediction + outcome to PostgreSQL
- Rolling metrics: Sharpe, accuracy, calibration per model

### Triggered Retraining
- Retrain when: rolling Sharpe < 0.3, or accuracy drops 10% from peak, or 30 days since last retrain
- Minimum 5 days between retrains (no thrashing)

### Feature Decay Monitoring
- Track rolling IC per feature (30-day window)
- Flag features with IC < 0.01 for 4 consecutive weeks
- Gate (don't drop) decaying features — multiply by decay score

### Dynamic Ensemble Weights (Phase 2)
- If multiple models per coin: weight by recent performance
- Regime-conditional weights (trend regime favors different models than mean-reversion)

### Alerts
- Tier 1 (immediate/Telegram): drawdown breach, model divergence, data failure, NaN in features
- Tier 2 (daily): rolling Sharpe, accuracy, feature stability, regime changes
- Tier 3 (weekly): feature importance drift, model agreement, backtest vs live gap

## 12. Implementation Timeline

| Week | Task | Details |
|---|---|---|
| 1-2 | Data pipeline | Backfill 3yr 15min OHLCV + funding + OI from Binance Vision & Gate.io |
| 3-4 | Feature engineering v2 | 30 features per coin, recalibrated for 15min, funding rate features |
| 5-6 | LightGBM walk-forward | Separate model per coin, 24+ OOS periods, on Colab |
| 7-8 | Backtester | Vectorized, realistic fees/funding/slippage, no look-ahead |
| 9-10 | Risk management + execution | Drawdown tiers, circuit breakers, entry filters, trailing stops |
| 11-12 | Paper trading | Connect to Gate.io API, trade with $0, validate execution |
| 13+ | Live with $50 | Small capital, prove it works with real money |

## 13. Success Criteria (Before Going Live)

Walk-forward backtest must show ALL of these:

| Metric | Minimum |
|---|---|
| Sharpe ratio (after fees) | > 1.5 |
| Profit factor | > 1.4 |
| Win rate | > 48% |
| Max drawdown | < 25% |
| Profitable months | > 60% of OOS months |
| Avg trades per week | 5-15 |
| Avg hold time | 4-16 hours |
| Fee drag / gross return | < 20% |

**Red flag if Sharpe > 3.0 or win rate > 65% — likely look-ahead bias or bug.**

## 14. Graduation Ladder

| Phase | Config | Requirements to Graduate |
|---|---|---|
| Month 1-3 | Conservative: 3x leverage, 1.5% risk/trade, threshold 0.60 | 3 consecutive profitable months |
| Month 4-6 | Moderate: 4x leverage, 2% risk/trade, threshold 0.57 | 3 more profitable months |
| Month 7+ | Aggressive: 5x leverage, 2.5% risk/trade, threshold 0.55 | Cumulative returns positive |
| Any month >15% DD | Revert to previous phase for 1 month | |

## 15. Market Longevity Assessment

Expert consensus: **5-8 year window** for these edges.

| Edge | Expected Lifespan |
|---|---|
| Funding rate mean-reversion | 5-7 years (structural to perp mechanism) |
| Retail FOMO/panic patterns | 3-5 years (new retail waves keep coming) |
| Altcoin microstructure on smaller venues | 8-10 years |
| Cross-coin momentum dispersion | 3-5 years |
| Volatility regime switching | 7+ years (most traders don't adapt sizing) |

Crypto stays inefficient longer than forex because: 24/7 retail-dominated flow, exchange fragmentation (50+ venues), perpetual swap mechanics (unique to crypto), constant new token launches, regulatory asymmetry across jurisdictions.

## 16. What the AI Watches That Humans Can't

The system monitors all 5 coins simultaneously every 15 minutes:
1. Multi-timeframe trend alignment (4 timeframes x 5 coins)
2. Volume anomaly detection (z-scores)
3. Order flow imbalance (from candle microstructure)
4. Cross-asset correlation regime
5. Volatility regime classification
6. Funding rate momentum and extremes
7. RSI divergence across all coins
8. Liquidation level clustering
9. Mean-reversion vs momentum regime (Hurst exponent)
10. BTC dominance and correlation lead

No FOMO. No greed. No revenge trading. No sleeping. Just math.
