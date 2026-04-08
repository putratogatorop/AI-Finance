# V6 Design Spec: Momentum Scanner + BTC-Alt Lead-Lag

**Date:** 2026-04-09
**Status:** Approved
**Target:** 10-15% monthly in backtest, expecting 5-8% live
**Approach:** Two complementary strategies that REACT to observable moves, not predict direction

---

## 1. Why This Approach (Lessons from v4/v5)

| What Failed | Why | V6 Solution |
|---|---|---|
| ML predicting direction | Signal-to-noise too low on any timeframe | Don't predict — detect moves already happening |
| 7 coins, all the time | 99% of bars are noise | Scan 200+ coins, only trade when something unusual happens |
| 29 features, LightGBM | Features are lagging, already priced in | Signal is VOLUME + PRICE ACCELERATION — observable in real-time |
| Target 48% win rate | Too low for 5-10% monthly | Target 45-55% WR with 3:1+ R:R on momentum trades |

## 2. Strategy 1: Momentum Scanner (Primary)

### Signal Detection

Scan all ~200 Gate.io USDT futures pairs every 60 seconds.

**Trigger conditions (ALL must be true):**
1. Current 15min volume > 3x the 20-period average volume
2. Volume confirmed on 2 consecutive 15min candles (not just a wick)
3. Price moved > 3% in last 30 minutes (2 candles)
4. Coin has > $500K 24h volume (liquidity filter)
5. Coin is NOT already up > 15% today (too late filter)
6. BTC daily trend aligns: BTC > 20 EMA for longs, BTC < 20 EMA for shorts

**Direction:** Same as the move (long breakouts, short breakdowns). No prediction needed.

### Entry

Do NOT chase the first candle. Wait for pullback.

1. Alert fires when trigger conditions met
2. Wait for first 20-30% retracement of the initial spike (1-4 candles)
3. Enter when price makes a higher-low (long) or lower-high (short)
4. Scale in: 50% on pullback entry, 50% when momentum resumes (next candle confirms direction)
5. If no pullback within 4 candles (1 hour), skip the trade

### Exit

1. **Stop loss:** Below the pullback low (long) or above pullback high (short). Typically 3-5% from entry.
2. **TP1:** Take 50% profit at 1.5R (7-10% from entry)
3. **Trail:** Move stop to breakeven after TP1, then trail at 2x ATR(14) on 15min
4. **Time exit:** Hard close after 8 hours max
5. **Momentum death:** If volume drops below 1x average on 2 consecutive candles, close

### Expected Performance

- Alerts per day: 5-15 across 200 coins
- Trades taken (after pullback filter): 1-3 per day
- Win rate: 40-55%
- Average winner: 10-25% (leveraged)
- Average loser: 3-5% (leveraged)
- R:R: 3:1+
- Monthly return at 3x leverage: 8-15% in backtest

## 3. Strategy 2: BTC-Alt Lead-Lag (Secondary)

### Signal Detection

Monitor BTC specifically for significant moves.

**Trigger conditions:**
1. BTC 2-hour rolling return > 1.5 standard deviations of 14-day rolling volatility (~1.5-2.5%)
2. BTC holds the move for 2 consecutive 15min candles (30min confirmation)
3. BTC OI rising during the move (real directional flow, not just a wick)

### Alt Selection

When BTC trigger fires:
1. Compute rolling 72h beta for each alt (LINK, XRP, AVAX, SOL)
2. Compute expected alt move: `expected = beta * btc_move`
3. Compute actual alt move so far
4. Pick the alt with the largest gap: `catch_up_remaining = expected - actual`
5. Skip if alt already moved > 50% of expected (too late)

### Entry

1. Enter 50% immediately on the selected alt
2. Add 50% when alt's 15min candle confirms direction
3. Use limit orders when possible (save fees)

### Exit

1. **Stop:** 1.5x ATR on 1h chart. Also exit if BTC retraces > 50% of triggering move.
2. **TP:** When alt achieves 70-80% of beta-expected move
3. **Time exit:** 6 hours max
4. **Signal flip:** Exit if BTC reverses direction

### Expected Performance

- Signals per week: 2-4 (BTC big moves)
- Win rate: 55-62%
- R:R: 1.3-1.8:1
- Monthly return at 3x leverage: 4-7%

## 4. Combined System

### Capital Allocation

- Momentum scanner: 70% of capital (higher frequency, more opportunities)
- BTC-alt lead-lag: 30% of capital (fewer but higher-conviction trades)
- Both strategies can fire simultaneously, but max 2 concurrent positions total

### Risk Management

- Max position size: 15% of equity as margin at 3x leverage
- Max 2 concurrent positions
- Max loss per trade: 2% of equity ($12 on $600)
- Weekly drawdown limit: 5% → reduce position size by half
- Monthly drawdown limit: 10% → stop trading 48h
- Never counter-trend (BTC above 20 EMA = longs only, below = shorts only)

### Position Sizing

```
atr = ATR(14) on 15min bars
stop_distance = distance from entry to stop level
risk_amount = equity * 0.02  # 2% equity risk
position_size = risk_amount / stop_distance
position_size = min(position_size, equity * 3.0)  # cap at 3x leverage
```

## 5. Data Requirements

### For Backtesting (already have)
- 200+ coins × 3 years 15min OHLCV from Binance Vision (data/raw/4h/ — need to re-download as 15min)
- Actually: we only downloaded 15min for 7 v4 coins. Need to extend to 200+ coins.

### For Live Trading (Gate.io API)
- REST ticker endpoint: all pairs in one call, poll every 60s
- WebSocket: optional, for faster detection
- Order placement API: limit + market orders

## 6. Backtest Design

### Historical Simulation

For each 15min candle timestamp across 3 years:
1. Compute volume ratio and price change for ALL coins
2. Fire alerts when trigger conditions met
3. Simulate pullback entry on subsequent candles
4. Track positions with stops, TP, trailing, time exit
5. Log all trades

### Walk-Forward

- 12 months in-sample (parameter calibration): volume threshold, pullback %, stop distance
- 3 months out-of-sample (validation)
- Roll quarterly, 8+ OOS windows

### Parameters to Optimize (MAX 5 to avoid overfit)

1. Volume threshold multiplier (default: 3x)
2. Price change threshold (default: 3% in 30min)
3. Pullback retracement % to enter (default: 25%)
4. ATR stop multiplier (default: see per-strategy)
5. Max hold time (default: 8h for scanner, 6h for lead-lag)

### Backtest Targets (aiming high for live degradation)

| Metric | Backtest Target | Live Expectation |
|---|---|---|
| Monthly return | > 10-15% | 5-8% |
| Win rate | > 50% | 40-50% |
| Profit factor | > 2.0 | > 1.5 |
| Max drawdown | < 15% | < 20% |
| Trades per week | 5-15 | 5-15 |
| Avg hold time | 2-8 hours | 2-8 hours |

**Red flags:** Monthly return > 30% or WR > 70% = likely look-ahead bias.

## 7. Data Gap: Need 15min Data for 200+ Coins

We currently have 15min OHLCV for only 7 coins (v4 assets). The momentum scanner needs ALL ~200 coins.

**Solution:** Re-run backfill_binance_vision.py (already exists for 200+ coins at 4h). Modify to download 15min instead. This is the same script from Week 1, just with different interval.

However, 200 coins × 36 months × 15min = massive data. For backtesting, we can:
- Download 15min for the top 50 most liquid coins (covers 90% of the big movers)
- Use 4h data we already have for the remaining 150 as a rough filter
- Full 200-coin 15min download can run overnight

## 8. Implementation Phases

### Phase 1: Data (extend 15min to top 50 coins)
- Modify backfill_binance_15m.py symbol list
- Download and store in data/raw/15m/

### Phase 2: Momentum Scanner Backtester
- Scan all coins at each timestamp
- Detect breakouts, simulate pullback entries
- Track trades with existing backtester

### Phase 3: BTC-Alt Lead-Lag Backtester
- Detect BTC big moves
- Select lagging alts, simulate entries
- Track trades

### Phase 4: Combined Results
- Run both strategies on same timeline
- Apply combined risk rules
- Report final metrics vs targets

### Phase 5: Live System (if backtest passes)
- Gate.io API integration
- Real-time scanner loop
- Order management
- Telegram alerts
