# V5 Design Spec: Two-Stage Extreme Move Predictor

**Date:** 2026-04-08
**Status:** Approved
**Target:** 70% backtest win rate (expecting 55-60% live), Sharpe > 2.5 in backtest
**Approach:** Two-stage ML — first predict IF a big move is coming, then predict WHICH direction

---

## 1. Why This Approach (Lessons from v4)

| v4 Problem | v5 Solution |
|---|---|
| Tried to predict direction all the time (99% noise) | Only predict when a big move is imminent |
| 29 features were all lagging (price-derived) | Add leading indicators: OI, liquidations, large trades |
| Model predicted FLAT 99% of the time | Stage 1 is binary (big move yes/no), no FLAT class |
| 58% accuracy was illusion (just predicting majority) | Stage 1 has ~20% positive rate, Stage 2 is ~50/50 |
| Features couldn't distinguish direction | Direction prediction only on extreme-move rows (clearer signal) |

## 2. Architecture

```
Stage 1: Volatility Predictor (binary)
  "Will price move > 2x ATR(4h) in next 4 hours?"
  Input: ~25 features (OI, liquidations, volume, funding, volatility)
  Output: probability of extreme move
  Threshold: only fire when p > 0.6

        ↓ (when Stage 1 fires)

Stage 2: Direction Predictor (binary)
  "LONG or SHORT?"
  Input: ~30 features (same + funding direction + momentum + oscillators)
  Output: probability of UP vs DOWN
  Trade: long if p_up > 0.6, short if p_up < 0.4, skip if uncertain
```

## 3. Data Sources

### Gate.io API — NEW (backfill 3 years)

**Open Interest (hourly):**
- Endpoint: `GET /api/v4/futures/usdt/contract_stats`
- Fields: `open_interest`, `top_lsr_account`, `top_lsr_size`
- No API key needed for public endpoints
- ~26K rows per coin (1h bars, 3 years)

**Liquidation History:**
- Endpoint: `GET /api/v4/futures/usdt/liq_orders`
- Fields: `size`, `time`, `side` (long/short liquidated)
- Aggregate to hourly: sum long liq volume, sum short liq volume
- ~26K rows per coin

### Binance Vision CDN — NEW

**Aggregated Trades:**
- URL: `https://data.binance.vision/data/spot/monthly/aggTrades/{SYMBOL}/`
- Monthly zip files, same pattern as kline downloads
- Fields: `price`, `quantity`, `is_buyer_maker`, `timestamp`
- Aggregate to hourly: large trade count (qty > 95th percentile), large trade imbalance (buy vs sell)

### Free APIs — NEW

**Fear & Greed Index:**
- URL: `https://api.alternative.me/fng/?limit=1095&format=json`
- Daily values, 3 years history in one call
- Forward-fill to hourly

### Existing Data (already in PostgreSQL)
- 15min OHLCV → aggregate to 1h (done in v4.2)
- Funding rates from Gate.io (done in v4.0)

## 4. Feature Set

### Stage 1 Features: Volatility Prediction (~25 features)

**Open Interest (3) — NEW, HIGHEST PRIORITY:**
1. `oi_change_1h` — hourly OI change (%)
2. `oi_change_4h` — 4-hour OI change (%)
3. `oi_zscore` — OI change z-score over 7-day rolling window

**Liquidations (3) — NEW:**
4. `liq_long_1h` — long liquidation volume in last hour (log-scaled)
5. `liq_short_1h` — short liquidation volume in last hour (log-scaled)
6. `liq_imbalance` — (long_liq - short_liq) / total_liq, ranges -1 to +1

**Large Trades (2) — NEW:**
7. `large_trade_ratio` — large trades / total trades in last hour
8. `large_trade_imbalance` — (large_buy - large_sell) / total_large, ranges -1 to +1

**Trade Activity (1) — NEW:**
9. `trade_count_ratio` — hourly trade count / 24h SMA (spike detection)

**Funding Rate (4) — EXISTING:**
10. `funding_rate` — raw current rate
11. `funding_ma_3d` — 3-day moving average
12. `funding_zscore` — rolling z-score (32-day window)
13. `cum_funding_3d` — cumulative cost over 3 days

**Volatility (4) — EXISTING:**
14. `vol_4h` — 4-hour realized volatility
15. `vol_1d` — 1-day realized volatility
16. `vol_ratio` — vol_4h / vol_1d (compression/expansion signal)
17. `parkinson_vol` — Parkinson high-low volatility

**Volume (2) — EXISTING:**
18. `volume_ratio` — current / 24h SMA volume
19. `taker_buy_ratio` — buy-side aggression

**Fear & Greed (2) — NEW:**
20. `fear_greed` — daily index value (0-100), forward-filled hourly
21. `fear_greed_change_3d` — 3-day change in index

### Stage 2 Features: Direction Prediction (~30 features)

All Stage 1 features PLUS:

**Long/Short Ratio (2) — NEW:**
22. `ls_ratio` — top trader long/short account ratio
23. `ls_ratio_change_4h` — 4-hour change in ratio (crowd positioning shift)

**Momentum (3) — EXISTING:**
24. `ret_1` — 1-hour log return
25. `ret_4` — 4-hour log return
26. `ret_24` — 1-day log return

**Oscillators (3) — EXISTING:**
27. `rsi_norm` — RSI(14) normalized to [-1, 1]
28. `bb_pct` — Bollinger Band position
29. `clv` — close location value

**Interaction (1):**
30. `funding_x_liq` — funding_zscore * liq_imbalance (extreme funding + liquidation cascade)

## 5. Target Variables

### Stage 1: Extreme Move Detection

```python
atr_4h = rolling_atr(close, high, low, period=4)  # 4-bar ATR on 1h = 4h volatility
forward_4h_return = abs(close.shift(-4) / close - 1)
target_stage1 = (forward_4h_return > 2 * atr_4h).astype(int)
```

Expected positive rate: ~15-20% (extreme moves are the top quintile)

### Stage 2: Direction (only where Stage 1 = 1)

```python
forward_4h_return_signed = close.shift(-4) / close - 1
target_stage2 = (forward_4h_return_signed > 0).astype(int)  # 1=UP, 0=DOWN
```

Expected split: ~50/50 (direction on big moves is roughly balanced)

## 6. Model Configuration

### Stage 1: LightGBM Binary Classifier

```python
stage1_params = {
    "objective": "binary",
    "num_leaves": 63,
    "min_child_samples": 100,
    "learning_rate": 0.01,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": -1,
    "n_estimators": 3000,
    "scale_pos_weight": 4.0,  # ~20% positive rate → weight = 80/20 = 4
    "verbose": -1,
}
```

### Stage 2: LightGBM Binary Classifier

```python
stage2_params = {
    "objective": "binary",
    "num_leaves": 31,          # simpler — less data (only extreme-move rows)
    "min_child_samples": 50,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 5,
    "n_estimators": 2000,
    "verbose": -1,
}
```

Stage 2 is intentionally simpler (fewer leaves, shallower) because it trains on much less data (~20% of Stage 1's training set).

## 7. Walk-Forward Configuration (1h bars)

```
Train: 6 months = 4,368 bars
Val:   1 month  = 720 bars
Test:  1 month  = 720 bars
Slide: 2 weeks  = 336 bars
Embargo: 4 bars (= 4h target horizon)
```

~61 folds from 3 years of data.

Both stages use the same walk-forward splits. Stage 2 trains on the subset of training rows where `target_stage1 = 1`.

## 8. Trading Rules

### Entry
- Stage 1 probability > 0.6 (extreme move likely)
- Stage 2 probability > 0.6 (direction is UP → LONG) or < 0.4 (direction is DOWN → SHORT)
- Combined confidence = stage1_prob * abs(stage2_prob - 0.5) * 2

### Position Sizing
```python
atr = ATR(14) on 1h bars
stop_distance = 2.0 * atr
risk_amount = equity * 0.01  # 1% equity risk
position_size = risk_amount / stop_distance
# Cap at 3x leverage
position_size = min(position_size, equity * 3.0)
```

### Exit (priority order)
1. **Stop loss:** entry -/+ 2x ATR
2. **Take profit:** entry +/- 4x ATR (2:1 reward:risk)
3. **Funding normalization:** if entered partly due to funding extreme (z-score > 2), close when z-score < 1
4. **Time exit:** 24h max hold
5. **Stage 1 flip:** if Stage 1 probability drops below 0.3 on next bar, close early

### Risk Limits
- Max 2 concurrent positions across all coins
- Max 1 position per coin
- No new entries if portfolio drawdown > 10%
- No entries within 1 hour of a previous stop-loss (anti-tilt)

## 9. Coins

BTC, LINK, XRP, AVAX — the 4 coins that showed signal in v4.1/v4.2.

## 10. Success Criteria (Backtest Targets)

Aiming high because live trading degrades performance:

| Metric | Backtest Target | Live Expectation |
|---|---|---|
| Win rate | > 70% | 55-60% |
| Sharpe ratio | > 2.5 | > 1.5 |
| Profit factor | > 2.0 | > 1.4 |
| Max drawdown | < 15% | < 25% |
| Trades per week | 4-10 | 4-10 |
| Avg hold time | 2-8 hours | 2-8 hours |

**Red flags:** Win rate > 85% or Sharpe > 5.0 = likely leakage or bug.

## 11. Implementation Phases

### Phase 1: Data Backfill (Week 1)
- Backfill Gate.io: OI, long/short ratio, liquidations (3 years, 4 coins)
- Backfill Binance: aggregated trades (3 years, 4 coins)
- Backfill Fear & Greed index
- Store all in PostgreSQL

### Phase 2: Feature Engineering (Week 2)
- Compute all 30 features on 1h bars
- Build Stage 1 and Stage 2 targets
- Save per-coin Parquet files

### Phase 3: Training + Backtest (Week 3)
- Walk-forward train both stages
- Run backtester with two-stage signal
- Evaluate against 70% WR target

### Phase 4: Iterate (Week 4+)
- If targets not met: feature ablation, hyperparameter tuning, threshold optimization
- If targets met: paper trading on Gate.io
