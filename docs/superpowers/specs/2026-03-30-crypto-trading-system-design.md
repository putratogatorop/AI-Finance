# AI-Finance: Crypto Trading System — Design Spec

**Date:** 2026-03-30
**Status:** Draft
**Author:** User + Claude

---

## Overview

A local, production-grade ML-driven crypto trading system that generates buy/sell signals for swing trading (1-6 week holds). The system uses an ensemble of ML models trained on technical and fundamental features, with a Next.js dashboard for reviewing signals and managing a portfolio.

**Key principles:**
- No leverage, fixed 1M IDR (~$60) per entry
- Model decides hold duration (1-6 weeks) per trade
- Hourly data collection, daily signals — review once per day
- Backtest-validated before live trading with real money
- 100% local, Docker-based, production-grade

---

## 1. System Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Docker Compose                     │
│                                                      │
│  ┌──────────────┐  ┌────────────┐  ┌───────────┐   │
│  │ Next.js      │  │ Python     │  │ PostgreSQL│   │
│  │ Container    │  │ Container  │  │ Container │   │
│  │ :3000        │  │ (no port)  │  │ :5432     │   │
│  │ API routes + │  │ Background │  │           │   │
│  │ Dashboard    │  │ ML service │  │           │   │
│  └──────┬───────┘  └─────┬──────┘  └─────┬─────┘   │
│         │                │               │          │
│         └────── reads ───┴── writes ─────┘          │
│                                                      │
│  Shared volume: ./data/raw/ (CSV data lake)         │
│                                                      │
│  Flow: Python writes signals/data → DB              │
│        Next.js reads via Prisma → serves to UI      │
└──────────────────────────────────────────────────────┘
```

### Components

- **Next.js (Frontend + API layer):** Dashboard UI + all API routes (via Next.js built-in API routes). Queries PostgreSQL directly via Prisma. Serves all client requests — no proxy to Python.
- **Python (Background ML + Data service):** ML inference, data pipeline, signal generation, scheduling. Runs as a background process (no HTTP server, no REST API). Writes results (signals, features, cleaned data) directly to PostgreSQL.
- **PostgreSQL:** Data warehouse — cleaned data, features, signals, portfolio, audit trail. Shared communication layer between Python (writes) and Next.js (reads via Prisma).
- **APScheduler:** Runs daily data fetch, weekly model retrain, signal generation inside the Python container
- **Docker Compose:** One command to spin up all services
- **CI/CD:** GitHub Actions workflows (Python: pytest/mypy/ruff, Next.js: eslint/type-check/build), pre-commit hooks

### Future expansion

- Add IDX stock support when a reliable data source is found
- Cloud deployment ready — Docker + CI/CD already in place

---

## 2. Data Pipeline

### Data Sources

| Source | Data | Cost |
|---|---|---|
| **Binance API** | OHLCV candles (hourly + daily), volume, order book depth | Free |
| **CoinGecko API** | Market cap rankings, categories, metadata, fear & greed index | Free tier |

### Data Lake (CSV) — Raw Storage

```
data/raw/
├── hourly/                ← 1-hour candles (primary)
│   ├── BTC/
│   │   ├── 2026-03-25.csv   ← 24 rows per file
│   │   └── ...
│   └── ETH/
│       └── ...
│
├── daily/                 ← daily OHLCV (aggregated)
│   ├── BTC/
│   │   ├── 2026-03-25.csv
│   │   └── ...
│   └── ...
│
├── fundamentals/          ← market metadata snapshots
│   ├── BTC/
│   │   ├── 2026-03-27.csv
│   │   └── ...
│   └── ...
│
└── index/                 ← total market / BTC dominance
    └── market/
        └── ...
```

- One CSV per asset per date
- Append-only, raw truth — safety net if DB needs rebuilding
- Shared Docker volume accessible by Python container

### Data Warehouse (PostgreSQL)

| Table | Content |
|---|---|
| `asset_prices_hourly` | Deduplicated hourly OHLCV |
| `asset_prices_daily` | Aggregated daily OHLCV |
| `asset_fundamentals` | Market cap, volume rank, category |
| `asset_features` | Computed ML features |
| `signals` | Model outputs (action, confidence, hold duration) |
| `portfolio` | Open/closed positions, P&L |
| `audit_log` | Every signal, every decision, timestamped |

### ETL Flow

```
Binance/CoinGecko → CSV (data lake) → ETL (clean/validate) → PostgreSQL (warehouse)
```

### Schedule

- **Hourly:** Fetch latest 1-hour candle for all 20 assets, save to CSV data lake, run ETL into PostgreSQL
- **Daily (after 00:00 UTC):** Aggregate hourly → daily, generate signals, re-evaluate open positions
- **Weekly (Sunday):** Refresh fundamental data, retrain models

---

## 3. Crypto Universe

**Top 20 by market cap** — dynamically tracked. Current typical set:

BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, DOT, MATIC, LINK, UNI, ATOM, LTC, APT, NEAR, OP, ARB, FIL, INJ

- List refreshed weekly via CoinGecko market cap rankings
- Auto-rotates: if a coin drops out of top 20, close any open position at next signal cycle

---

## 4. ML Engine (Ensemble)

### Models

| Model | Role | Strength |
|---|---|---|
| **XGBoost** | Baseline, feature importance | Fast, great with tabular data |
| **LightGBM** | Second opinion | Handles large feature sets |
| **LSTM** | Sequential patterns | Captures time-series dependencies |

### Features

**Technical:**
- Moving averages: SMA/EMA 5, 10, 20, 50 day
- RSI, MACD, Bollinger Bands
- Volume trends, price momentum (1w, 2w, 4w returns)
- Volatility: ATR, standard deviation

**On-chain / Market context (free):**
- BTC dominance, total market cap trend
- Fear & greed index
- Cross-asset correlations (BTC/ETH, altcoin rotation patterns)

**Fundamental:**
- Market cap rank
- Volume rank
- Category/sector momentum

### Target Variable

Multi-horizon forward returns:
- 1-week return
- 2-week return
- 4-week return
- 6-week return

### Consensus Logic

- Each model produces a confidence score per horizon
- Final signal = weighted average across models
- Only fire signal when ensemble agreement exceeds confidence threshold (configurable, default 0.7)
- Model selects the best horizon for each trade

### Signal Output

```json
{
  "asset": "ETH",
  "action": "BUY",
  "confidence": 0.84,
  "suggested_hold_days": 18,
  "stop_loss_pct": -8,
  "expected_return_pct": 12,
  "model_agreement": "3/3",
  "timestamp": "2026-03-30T00:15:00Z"
}
```

### Retraining

- Weekly (Sunday), rolling window of last 12 months
- Model versioning — every trained model saved with timestamp
- New model only replaces old if backtest metrics are equal or better

---

## 5. Risk Management

### Per-Trade

- **Fixed position size:** 1,000,000 IDR (~$60) per entry
- **Hard stop-loss:** -8% from entry price (configurable)
- **Dynamic exit:** Model re-evaluates open positions daily. If confidence drops below threshold → signal early exit

### Portfolio-Level

- **Max single asset exposure:** 25% of total capital
- **Max concurrent positions:** 8 (= max 8M IDR exposed)
- **Portfolio drawdown pause:** If total portfolio drops 20% from peak, pause new signals until recovery (configurable)

### Alert Levels

- **Red** — stop-loss hit or model says exit now
- **Yellow** — model confidence dropping, watch closely
- **Green** — on track, hold

---

## 6. Backtesting Engine

### Process

1. Run full ensemble against 2-3 years of historical crypto data
2. Simulate trades as live system would: signals, entries, exits, stop-losses
3. No lookahead bias — model only sees data available at time of decision
4. Fixed 1M IDR per trade to make results tangible

### Output Metrics

| Metric | Description |
|---|---|
| Total return | Overall profit/loss in IDR |
| Win rate | % of profitable trades |
| Reward/Risk ratio | Average win size vs average loss size |
| Sharpe ratio | Risk-adjusted returns |
| Max drawdown | Worst peak-to-trough drop |
| Monthly breakdown | Per-month performance in IDR |
| Per-asset breakdown | Which coins perform best |

### Validation Flow

```
Backtest (2-3 years historical) → Satisfied with results? → Live with small capital (1M IDR/trade)
```

No paper trading phase — straight to live with small money after backtest validation.

---

## 7. Dashboard (Next.js)

All dashboard data is served by Next.js API routes (`/api/*`) that query PostgreSQL directly via Prisma. Python writes data to the DB in the background; Next.js reads it on demand. No inter-service HTTP calls.

### Pages

| Page | Content | Data Source (via Prisma) |
|---|---|---|
| **Overview** | Portfolio value (IDR), total P&L, open positions count, today's signals | `portfolio`, `signals` tables |
| **Signals** | Active buy/sell signals with confidence, suggested hold, model agreement | `signals` table |
| **Portfolio** | Current holdings, entry price, current price, P&L per position, stop-loss status | `portfolio`, `asset_prices_daily` tables |
| **Backtest** | Historical performance charts, metrics table, monthly breakdown | `audit_log`, `signals` tables |
| **Asset Detail** | Per-coin chart, feature values, model scores, trade history | `asset_prices_hourly`, `asset_features`, `signals` tables |
| **Audit Log** | Every signal, every action, timestamps, model reasoning | `audit_log` table |
| **Settings** | Risk parameters: stop-loss %, max positions, confidence threshold, retrain schedule | Config table or env vars |

### Key Features

- Red/Yellow/Green signal urgency indicators
- One-click acknowledge on signals (seen, acted, or skipped)
- Backtest re-run with different parameters
- Mobile-responsive — check signals from phone
- All amounts displayed in IDR
- All API routes in Next.js (`/api/signals`, `/api/portfolio`, etc.) — no Python proxy layer

---

## 8. Tech Stack

### Python Service (Background ML + Data — no HTTP server)
- **Python 3.12+**
- **APScheduler** — job scheduling (entry point for the container)
- **yfinance** — future IDX support
- **ccxt or python-binance** — Binance API
- **pycoingecko** — CoinGecko API
- **pandas, numpy** — data processing
- **scikit-learn, xgboost, lightgbm** — ML models
- **tensorflow/pytorch** — LSTM model
- **SQLAlchemy** — ORM for PostgreSQL (writes signals, features, cleaned data to DB)
- **ruff, mypy, pytest** — code quality
- **Communication:** Python writes to PostgreSQL; Next.js reads via Prisma. No REST API, no FastAPI.

### Next.js App (Frontend + ALL API routes)
- **Next.js 14+** — React framework with built-in API routes (handles all backend HTTP endpoints)
- **TypeScript** — type safety
- **Tailwind CSS** — styling
- **Recharts or Tremor** — data visualization
- **Prisma** — database ORM (queries PostgreSQL directly for signals, portfolio, etc.)
- **ESLint** — code quality

### Infrastructure
- **Docker + Docker Compose** — containerization
- **PostgreSQL 16** — data warehouse
- **GitHub Actions** — CI/CD pipelines
- **Pre-commit hooks** — linting, type checking

---

## 9. Future Roadmap

1. **Phase 1:** Build and validate crypto trading system (this spec)
2. **Phase 2:** Add IDX stock support when reliable data source is found
3. **Phase 3:** Optimize leverage based on live track record (if desired)
4. **Phase 4:** Cloud deployment for always-on operation
