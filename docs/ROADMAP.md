# AI-Finance Roadmap

## Vision

A crypto trading system that **starts profitable and gets smarter over time** — like OpenAI Five, but for trading. The AI continuously learns from its own trades, adapting to market changes and improving its win rate and profit capture.

---

## Phase 1: Paper Trading (NOW)
**Goal:** Prove the system works with real-time market data, zero risk.

- [ ] Build live scanner daemon (Python, runs every 60s)
- [ ] Connect to Gate.io API (read market data, no trading yet)
- [ ] Scanner writes signals to PostgreSQL → Dashboard displays them
- [ ] Manual decision: look at signal on dashboard → open Gate.io app → trade manually
- [ ] Track all signals and outcomes in DB
- [ ] Run for 2-4 weeks, collect 50-100 signals
- [ ] Validate: does live WR match backtest WR (~65%)?

**Tech:** Python scanner daemon + existing Next.js dashboard + Gate.io REST API

---

## Phase 2: Auto-Trading with $50 (Month 2)
**Goal:** Automate execution on Gate.io with tiny capital.

- [ ] Gate.io API: place limit/market orders programmatically
- [ ] Auto-execute when ML filter says confidence > 0.65
- [ ] ATR trailing stop managed by the bot
- [ ] Staged partial exits automated
- [ ] Telegram alerts on every entry/exit
- [ ] Kill switch: stop if drawdown > 10%
- [ ] Run for 2-4 weeks with $50

**Success criteria:** Positive PnL after fees on $50 over 2 weeks.

---

## Phase 3: Scale to $600 + Weekly Retrain (Month 3)
**Goal:** Full capital deployment with adaptive ML.

- [ ] Scale to $600 capital
- [ ] **Weekly ML retrain** (every Sunday):
  - Retrain LightGBM on last 6 months of signals (sliding window)
  - Validate new model on last 2 weeks holdout
  - Deploy only if new model beats current on holdout metrics
  - Auto-rollback if live PF drops below 1.5
- [ ] Model drift detection:
  - Track rolling WR and PF over 50-trade windows
  - Alert if WR < 55% or PF < 1.5 for 2 consecutive windows
  - Trigger early retrain
- [ ] Feature importance monitoring (detect when market regime changes)

**Tech:** Cron job Sunday midnight → retrain → validate → deploy

---

## Phase 4: RL Trade Management Agent (Month 4-6)
**Goal:** AI learns optimal trade management from live data.

**Prerequisites:** 500+ live trades collected from Phases 2-3.

- [ ] Train offline RL agent (Fitted Q-Iteration / DQN):
  - **State:** unrealized PnL/ATR, bars held, volume trend, vol regime, trail distance
  - **Actions:** hold, partial exit 25%, tighten trail, loosen trail, close all
  - **Reward:** realized PnL per unit of risk (position-level Sharpe)
- [ ] Compare RL policy vs fixed rules on holdout trades
- [ ] A/B test: run 50% of trades with RL, 50% with fixed rules
- [ ] Deploy RL only if it beats fixed rules by > 5% on net PnL
- [ ] Continue collecting data → retrain RL monthly

**Architecture:** DQN with small MLP (2 layers, 64 units), experience replay.

---

## Phase 5: Full Adaptive System (Month 6+)
**Goal:** Self-improving AI that adapts to market changes.

- [ ] Weekly retrain of ML filter (already running from Phase 3)
- [ ] Monthly retrain of RL trade management agent
- [ ] Regime detection model (trending/choppy/volatile):
  - Automatically adjusts risk sizing per regime
  - RL agent receives regime as state input
- [ ] New feature discovery pipeline:
  - Monitor Binance orderbook for leading signals
  - Add social sentiment (CryptoPanic API)
  - Test new features → add if they improve holdout PF
- [ ] Portfolio-level RL:
  - Manage multiple concurrent positions
  - Learn correlation risk (don't go all-in on correlated alts)
  - Optimize capital allocation across coins

**The dream:** A system that started with fixed rules, evolved to adaptive ML, and eventually runs a multi-strategy portfolio that improves every week — with minimal human intervention.

---

## Success Metrics Per Phase

| Phase | WR Target | PF Target | Monthly Return | Capital |
|-------|-----------|-----------|----------------|---------|
| 1 Paper | Validate ~65% | Validate ~5.0 | $0 (tracking only) | $0 |
| 2 Auto $50 | > 55% | > 2.0 | > 5% | $50 |
| 3 Scale $600 | > 60% | > 3.0 | > 10% | $600 |
| 4 RL Agent | > 65% | > 4.0 | > 15% | $600-2000 |
| 5 Full Auto | > 70% | > 5.0 | > 20% | $2000+ |

**Baseline comparison:** Indonesian government bonds = 6%/year = 0.5%/month. Every phase must massively beat this.

---

## Architecture

```
Phase 1-2: Scanner → ML Filter → Dashboard → Manual/Auto Trade
Phase 3:   Scanner → ML Filter (weekly retrain) → Auto Trade
Phase 4:   Scanner → ML Filter → RL Agent (monthly retrain) → Auto Trade
Phase 5:   Scanner → ML Filter → RL Agent → Portfolio Manager → Multi-Trade
                ↑                    ↑              ↑
          Weekly retrain      Monthly retrain   Regime detection
```

Each layer improves independently. The scanner finds opportunities, ML filters quality, RL optimizes execution, portfolio manager handles risk. Each learns from its own data at its own pace.
