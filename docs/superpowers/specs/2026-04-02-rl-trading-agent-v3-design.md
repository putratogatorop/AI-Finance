# v3: RL Trading Agent with Population-Based Training

## Vision

A population of RL agents evolves to trade ~198 crypto alt-coins, exploiting the structural dependency between BTC and altcoins. BTC moves first, alts follow with amplified reaction and a 1-6 candle lag on 4h timeframe. The agent learns to front-run this propagation delay.

BTC and ETH are **read-only indicators** (the compass). Alts are the only tradeable assets.

The agent operates like a 9-to-5 worker: checks the market at 9 AM and 10 PM WIB, manages up to 10 positions, sets SL/TP orders, then walks away. No screen-watching.

## Capital Scaling

| Phase | Capital | Max Positions | Timeline |
|-------|---------|---------------|----------|
| Proof of concept | 2-3M IDR | 5 | Month 1 |
| Validated | 10M IDR (~$600) | 10 | Month 2-3 |
| Scaled | 100M IDR (~$6,000) | 10 (bigger size) | Month 4+ |

## Primary Edge: BTC-Alt Propagation Delay

The agent is NOT predicting market direction. It exploits a structural market inefficiency:

- BTC drops 2% → 1-4 hours later alts drop 10-20% → agent shorts alts before they catch up
- BTC recovers 2% → 1-4 hours later alts bounce 10-20% → agent longs alts before they bounce
- BTC flat → agent mean-reverts alts that drifted from fair value
- BTC uncertain → agent reduces exposure, collects SL/TP on existing trades

This edge exists because retail traders react slowly to BTC signals. The LSTM learns this pattern naturally from the data through thousands of training episodes.

## Trading Mechanics

### Schedule

- **Window 1:** 9:00 AM WIB — scan all alts, open/close/adjust positions, set SL/TP
- **Window 2:** 10:00 PM WIB — second pass, adjust
- **Between windows:** Agent sleeps. SL/TP orders execute automatically on exchange.
- **Result:** 730 decision points per year, ~3,650-7,300 individual trades/year

### Position Constraints

| Parameter | Value |
|-----------|-------|
| Max concurrent positions | 10 |
| Max single position | 10% of capital |
| Max total exposure | 80% of capital (20% cash buffer) |
| Every position must have SL/TP | Mandatory, no naked positions |
| Stop-loss range | 2-8% from entry |
| Take-profit range | 3-15% from entry |

## Features (43 total)

### Per-Alt Features (28)

| # | Feature | Category |
|---|---------|----------|
| 1-7 | Log returns (3, 6, 12, 24, 42, 84, 168 candles) | Momentum |
| 8-11 | Price-to-SMA ratios (20, 50, 100, 200) | Trend |
| 12 | RSI(14) normalized | Momentum |
| 13 | MACD histogram normalized | Momentum |
| 14-15 | BB width, %B | Volatility |
| 16-18 | Volatility (6, 24, 72 candle) | Volatility |
| 19 | ATR normalized | Volatility |
| 20 | Volume ratio (short/long) | Volume |
| 21 | CLV (close location value) | Microstructure |
| 22 | Skewness (24 candle) | Distribution |
| 23 | Drawdown from 168-candle high | Trend |
| 24 | Above SMA50 | Regime |
| 25 | Above SMA200 | Regime |
| 26 | BTC-alt beta (rolling 30d) | Cross-token |
| 27 | BTC-alt correlation (rolling 30d) | Cross-token |
| 28 | Residual: alt return - (beta x BTC return) | Cross-token |

### BTC/ETH Indicator Features (9)

| # | Feature | Purpose |
|---|---------|---------|
| 29-31 | BTC return (1, 2, 3 candles ago) | Leading signal |
| 32 | BTC volatility (24 candle) | Vol regime |
| 33 | BTC RSI | Overbought/oversold |
| 34-36 | ETH return (1, 2, 3 candles ago) | Secondary leader |
| 37 | ETH/BTC ratio change | Rotation signal |
| 38 | BTC above SMA50 | BTC trend regime |

### Portfolio State Features (5)

| # | Feature | Purpose |
|---|---------|---------|
| 39 | Cash ratio (cash / total capital) | Exposure level |
| 40 | Open positions count / max | Capacity remaining |
| 41 | Current portfolio PnL % | Session performance |
| 42 | Avg position holding time | Duration awareness |
| 43 | Max open position drawdown | Worst current position |

All features are stationary (z-scores, ratios, normalized). No raw prices. No calendar/date features. Agent cannot cheat.

## Training Architecture

### Population-Based Training

```
PHASE 1: Fast Prototype (Colab Pro A100, ~8 hours)
  Population: 10 agents
  Architecture: LSTM (64 hidden, 2 layers) + PPO policy head
  Episodes: 100 generations x 10 episodes per agent

  Evolution loop:
    1. Each agent plays 10 episodes (random 6-month windows)
    2. Rank all 10 by reward
    3. Bottom 3 die
    4. Top 3 clone + mutate weights (5% noise)
    5. Middle 4 get experience replay from top 3
    6. Save checkpoint to Google Drive
    7. Repeat

PHASE 2: Full Training (if Phase 1 promising, ~40 hours)
  Population: 20 agents
  500 generations
  Add hyperparameter evolution (LR, SL/TP ranges, position sizing)
```

### Agent Architecture

```
Input per window:
  198 alts x 28 features = 5,544
  + 9 BTC/ETH indicators
  + 5 portfolio state
  = 5,563 inputs

  LSTM sees last 10 windows (5 days of history)

  LSTM (64 hidden, 2 layers, dropout 0.3)
    → Policy head (linear layers)
      → Per-alt score: [-1, +1] (negative=short, positive=long)
      → SL distance: [0.02, 0.08]
      → TP distance: [0.03, 0.15]
    
  Top-K selection:
    → Long top 5 by score, short bottom 5
    → Position size proportional to |score|
    → Skip if score magnitude < threshold (stay flat)
```

### Reward Function

```
R = Sharpe x 0.6 + Normalized_Return x 0.3 - Drawdown_Penalty x 0.1

Where:
  Sharpe: annualized Sharpe ratio of the episode
  Normalized_Return: total_return / max_possible_return (keeps profit motivation)
  Drawdown_Penalty: extra -1.0 if max drawdown exceeds 20%
```

### Walk-Forward (No Cheating)

```
Total data: 2022-01 to 2026-03 (4+ years, bear + bull markets)

Training episodes:  2022-01 to 2025-06 (3.5 years)
  - Random start date within range
  - Play forward 6 months
  - Never sees data beyond current candle

Holdout test:       2025-07 to 2026-03 (SEALED)
  - Run ONLY once after all training
  - Go/no-go decision for real money

Anti-cheat rules:
  - No date/year/month features
  - All features normalized
  - Dead coins included (LUNA, FTT die mid-episode)
  - BTC/ETH are indicators, not tradeable
  - Agent cannot peek at future candles
```

### Checkpoint/Resume for Colab

```
After every generation:
  - Save all agents' weights to Google Drive
  - Save evolution state (rankings, generation count)
  - Save training metrics

Colab disconnects:
  - Reconnect
  - Load latest checkpoint from Google Drive
  - Resume from generation N
  - Continue training
```

## Gym Environment

```
class CryptoTradingEnv (OpenAI Gym-style):

  State space:
    198 alts x 28 features + 9 BTC/ETH + 5 portfolio = 5,563 per window
    LSTM context: last 10 windows

  Action space:
    Per alt: score [-1, +1] (direction + magnitude)
    SL distance: [0.02, 0.08]
    TP distance: [0.03, 0.15]
    Top-K selection applied post-output

  Step = one trading window (12 hours):
    1. Agent receives observation
    2. Agent outputs actions
    3. Environment simulates 12 hours:
       - Check SL/TP hits against actual high/low of 4h candles
       - Apply fills with 0.1% taker fee
       - Apply 0.05% slippage
       - Deduct funding rate: 0.01% every 8h on open positions
       - Update portfolio
    4. Return: new state, reward, done

  Episode = 6 months (~365 windows):
    - Random start date within training range
    - Starting capital: 10M IDR
    - Episode ends at 6 months OR drawdown > 30%

  Realism:
    - SL/TP checked against intra-period high/low (not just close)
    - Dead coins become untradeable (zero volume)
    - Funding rates deducted on open futures positions
    - Fees + slippage on every trade
```

## Live Deployment

### Infrastructure (reuse existing Docker setup)

```
Python Service (existing container):
  Cron 9:00 AM WIB:
    - Load latest 4h candles for all alts + BTC + ETH
    - Compute 42 features
    - Run winning agent forward pass
    - Execute trades via Gate.io futures API
    - Set SL/TP orders
    - Log decisions to PostgreSQL

  Cron 10:00 PM WIB:
    - Same flow, second window

  Cron every 4h:
    - Ingest latest candles
    - Sync SL/TP fills with exchange

PostgreSQL (existing):
  - trades, positions, signals, audit_log

Next.js Dashboard (existing):
  - Tournament page (already built)
  - Live positions + PnL
  - Agent decisions log

Alerts:
  - Telegram notification for every trade
```

### Exchange

Gate.io perpetual futures (Binance blocked in Indonesia).
- Long + short capability
- API for order placement + SL/TP
- 0.1% taker fee

### Circuit Breakers

| Trigger | Action |
|---------|--------|
| Daily loss > 5% | Halt until next day |
| Weekly loss > 10% | Halt for 3 days |
| Monthly loss > 15% | Halt, retrain, manual review |
| Max drawdown > 25% | Full stop, manual intervention |

## Success Criteria (Go/No-Go)

Evaluated on sealed holdout (2025-07 to 2026-03):

| Metric | Minimum | Target |
|--------|---------|--------|
| Sharpe ratio | > 0.5 | > 1.0 |
| Total return | > 0% | > 15% |
| Max drawdown | < 25% | < 15% |
| Win rate | > 48% | > 52% |
| Trade count | > 200 | > 500 |
| Beat Buy & Hold | Yes | Yes |

If holdout fails all minimums: go back, do not deploy. No exceptions.

## Timeline

```
Week 1: Build trading gym + features
  - Gym environment with realistic SL/TP/fees/funding
  - 43 features (28 alt + 9 BTC/ETH + 5 portfolio + 1 residual)
  - Test locally, upload to Colab

Week 2: Phase 1 training (Colab Pro)
  - 10 agents, 100 generations (~8 hours)
  - Evaluate on holdout
  - Iterate if needed

Week 3: Phase 2 training (if Phase 1 promising)
  - 20 agents, 500 generations (~40 hours)
  - Final holdout evaluation
  - Go/no-go decision

Week 4: Deploy
  - Wire winning agent to Gate.io API
  - Dashboard updates for live tracking
  - Start with 2-3M IDR
  - Monitor daily
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| RL framework | Stable-Baselines3 (PPO) or custom PyTorch |
| LSTM backbone | PyTorch |
| Gym environment | Custom (OpenAI Gym interface) |
| Training compute | Colab Pro (A100) |
| Checkpoint storage | Google Drive |
| Exchange | Gate.io futures API |
| Backend | Python + PostgreSQL (existing) |
| Dashboard | Next.js (existing) |
| Deployment | Docker Compose (existing) |
