# AI-Finance v2: ML Trading Tournament

## Vision
Run a 4-year simulation (2022-2026) across 200+ crypto tokens, making thousands of trades, testing 7-10 ML approaches head-to-head. Pick the winner. Deploy with real money.

## Data Source Decision
**Primary: Binance Data Vision** (`data.binance.vision`)
- Static CDN downloads — no API key, no rate limits
- Works from Indonesia (tested, HTTP 200)
- Pre-packaged CSV in ZIP files per symbol/month
- 400+ USDT pairs with 4+ years of history
- Format: `{SYMBOL}-4h-{YYYY-MM}.csv` with OHLCV + volume + trades

**Why single source:** Consistent pricing, no cross-exchange discrepancies in backtest. All models train and trade on the same price data.

**Fallback:** Gate.io via ccxt (already proven, ~200 pairs)

---

## Phase 1: Data Collection ← CURRENT PRIORITY

### 1A. Bulk 4h Candle Download
```bash
cd services/python
python scripts/backfill_binance_vision.py
```

| Item | Target |
|------|--------|
| Symbols | 200+ USDT pairs (includes dead coins: LUNA, FTT, UST) |
| Timeframe | 4h candles |
| Period | Jan 2022 — Apr 2026 (51 months) |
| Expected rows | ~200 tokens × 8,766 candles = ~1.75M rows |
| Storage | ~500MB CSV, ~1-2GB in PostgreSQL with indexes |
| Download time | ~20-40 min (8 parallel workers, no rate limit) |
| Output | `data/raw/4h/{SYMBOL}/{SYMBOL}-4h-{YYYY-MM}.csv` |
| Manifest | `data/raw/4h/manifest.csv` — which symbols have data |

### 1B. Upsert to PostgreSQL (after CSV download)
- [ ] Build upsert script: CSV → `asset_prices_hourly` table
- [ ] Aggregate 4h → daily → `asset_prices_daily` table
- [ ] Verify row counts match manifest

### 1C. Supplementary Data (Phase 2, after ML tournament)
| Data | Source | Priority |
|------|--------|----------|
| Funding rates | Binance public API | P1 |
| Open interest | Bybit API | P1 |
| Fear & Greed | alternative.me | P2 |
| Macro (DXY, SPX, VIX) | yfinance | P2 |
| BTC on-chain | blockchain.info | P3 |

---

## Phase 2: Feature Engineering

### Kill List (remove from current code)
- ❌ Fake `market_context` features (np.random in run_backtest.py)
- ❌ Raw SMA/EMA values (non-stationary)
- ❌ Raw BB upper/middle/lower
- ❌ Raw volume_sma_20

### Tier 1 Features (from OHLCV only, no new data needed, ~20-25)
- Log returns: 3d, 7d, 14d, 28d
- Price-to-SMA ratios: 10, 20, 50
- RSI(14), MACD histogram (normalized by price)
- BB width, %B
- ATR normalized by price
- Volume ratio (short/long), OBV slope
- Volatility: 10d, 30d, ratio
- Parkinson/Garman-Klass volatility
- Rolling skewness, kurtosis (24 candles)
- Close location value, shadow ratios
- Consecutive direction count
- Trend regime: above/below SMA50, SMA200, golden cross

### Tier 2 Features (cross-token, no new data)
- BTC return (lagged 1-2 candles, BTC leads alts)
- BTC-token rolling correlation and beta
- Sector average return (sector rotation)
- Market breadth (advance/decline ratio across all tokens)
- ETH/BTC ratio trend

### Target Feature Count
- Per model: 30-40 features after selection
- Start with ~50-60 raw, prune via correlation filter + importance

---

## Phase 3: ML Tournament

### Tournament Structure
```
PHASE A: Data Prep (do once)
  - All 200+ tokens, 4h, 4 years
  - Compute ALL features
  - Time splits:
      Train:      2022-01 to 2024-06  (2.5 years)
      Validation:  2024-07 to 2025-06  (1 year — tune here)
      Test:        2025-07 to 2026-03  (holdout — SEALED ENVELOPE)

PHASE B: Walk-Forward Training
  - For each model:
      - Expanding window, retrain monthly
      - Predict next month
      - Record all predictions with timestamps
      - Deduct 0.2% fees per round-trip trade

PHASE C: Evaluate on Validation Set
  - Rank ALL models on identical metrics
  - Select top 3

PHASE D: Final Test (holdout)
  - Run ONLY top 3 on sealed test set
  - This is unbiased performance estimate
  - If all 3 fail → go back, don't deploy
```

### Model Zoo

**Tier 0 — Baselines (no ML, must beat these)**
| ID | Model | Description |
|----|-------|-------------|
| B0 | Buy & Hold | Per-coin baseline |
| B1 | SMA Crossover | Long when price > SMA50, flat below |
| B2 | Momentum | Buy top 10 coins by 30d return, rebalance weekly |

**Tier 1 — Primary Contenders**
| ID | Model | Type | Why |
|----|-------|------|-----|
| M1 | LightGBM Classifier | Gradient-boosted tree | Best for tabular, fast |
| M2 | XGBoost Classifier | Gradient-boosted tree | Different regularization |
| M3 | Ridge/Logistic Regression | Linear | Interpretable, catches simple patterns |
| M4 | Random Forest | Bagged trees | Less overfit, OOB estimate |

**Tier 2 — Add if Tier 1 shows promise**
| ID | Model | Type | Why |
|----|-------|------|-----|
| M5 | LSTM/GRU | Recurrent NN | Sequential patterns |
| M6 | LightGBM (vol-adjusted target) | Tree variant | Different target framing |
| M7 | Stacking Ensemble | Meta-learner | Combine top 2-3 models |

### Classification Target
- **Primary:** P(token up >2% in next 5 days) → binary
- **Secondary:** Trinomial {short, flat, long} based on return thresholds
- Use `predict_proba` for confidence, NOT custom sigmoid

### Comparison Metrics (ranked by Sharpe)
| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Risk-adjusted return (primary ranking) |
| Total Return | Raw cumulative performance |
| Max Drawdown | Worst peak-to-trough |
| Win Rate | % profitable trades |
| Profit Factor | Gross profit / gross loss |
| # Trades | Must be 100+ for statistical significance |
| Return per Trade | After 0.2% round-trip fees |

### Anti-Overfitting Rules
1. Holdout test set is SEALED — touch only once at the very end
2. Bonferroni correction: if testing 10 models, p < 0.005
3. Walk-forward (never single train/test split)
4. If train Sharpe / test Sharpe > 3x → model is memorizing, discard
5. Track feature importance stability across folds

---

## Phase 4: Simulation (100M IDR = ~$6,000)

### Capital & Position Sizing
| Parameter | Value |
|-----------|-------|
| Starting capital | 100,000,000 IDR (~$6,000) |
| Position size | Inverse-volatility (target 0.2% daily risk per position) |
| Max single token | 5% of capital |
| Max concurrent positions | 15-20 |
| Max sector exposure | 20% |
| Max gross exposure | 100% (no leverage) |

### Regime Filter (4 gates, upstream of ALL models)
```
Gate 1: Token price > 50-SMA (trend)
Gate 2: 2.5x ATR / price < 10% (volatility ceiling)
Gate 3: ROC(20) > -12% (momentum floor)
Gate 4: Model probability > threshold (per-model tuned)
ALL MUST PASS.
```

### Stop-Loss (ATR-based)
```
Distance: 2.5 × ATR(14)
Floor: 3% (never tighter)
Ceiling: 25% (altcoins are volatile)
Trailing: activate at +8%, trail at -6% from high
Stagnation: exit if flat (±3%) after 21 days
Max hold: 42 days
```

### Circuit Breakers
```
Daily portfolio loss > 5% → HALT all trading
Weekly loss > 10% → HALT
3 consecutive stop-losses → 14-day pause
Monthly loss > 5M IDR → pause rest of month
```

---

## Phase 5: Survivorship Bias Handling

### The Problem
Top 100 coins from Jan 2022 includes LUNA, FTT, UST — all went to zero. If we only backtest on TODAY's top 100, results are artificially inflated.

### Solution
- Download ALL available symbols from Binance Data Vision (~200+)
- Include known dead coins: LUNA, FTT, UST, ANC, CEL
- Tokens enter the universe when their data starts, NOT retroactively
- Tokens stay in the universe even after they crash
- Manifest tracks first/last available month per symbol

---

## Phase 6: Go Live

### Deployment Timeline
```
Sprint 1: Download data, build baselines → this week
Sprint 2: Build ML models, run tournament → next week
Sprint 3: Risk framework, full backtest → week 3
Sprint 4: Dashboard + paper trading → week 4
Sprint 5+: Live trading with winning model → week 5+
```

### Live Architecture
```
Every 4h: Ingest latest candles for all tokens
Daily 00:15: Compute features → run model → generate signals
Daily 00:25: Risk check → persist signals → Telegram alert
Weekly Sunday: Retrain model on latest data
```

### Cost
```
Binance Data Vision: Free
Gate.io trading fees: ~$19/month at 5 trades/day
Electricity: ~$10/month
Total: ~$30/month
```

---

## Current Status

### Completed ✅
- [x] Infrastructure: Python + PostgreSQL + Next.js + Docker (Plans 1-3)
- [x] 121 Python tests + 32 Jest tests passing
- [x] BTC 4h candles: 4,567 rows in PostgreSQL
- [x] Baseline backtest: SMA200 = +14.2%, Buy&Hold = +17.1%
- [x] ML backtest v1-v3: learned that current approach doesn't work
- [x] 6-expert consultation: redesign plan agreed
- [x] Binance Data Vision: confirmed accessible from Indonesia
- [x] Bulk downloader script: `scripts/backfill_binance_vision.py`
- [x] **200 tokens × 4 years of 4h data: 1,602,352 candles, 8,829 CSV files (227 MB)** ✅
  - Downloaded in 6 minutes from Binance Data Vision
  - Stored in `data/raw/4h/{SYMBOL}/{SYMBOL}-4h-{YYYY-MM}.csv`
  - Manifest: `data/raw/4h/manifest.csv`

### In Progress 🔄
- [x] CSV → PostgreSQL upsert script: `scripts/upsert_csv_to_pg.py` — running (200 tokens)
- [x] New stationary feature engineering: `src/ml/features_v2.py` — 45 features, all stationary ✅
  - Tested on BTC: 6,305 valid samples, 35.3% positive rate
- [x] Feature export for Colab: `data/features/tournament_data.parquet` (229 MB) ✅
  - 1,110,364 samples, 200 tokens, 28 features, 39.5% positive target rate
  - Also exported as `tournament_data.csv.gz` (258 MB)
  - Upload to Google Colab for ML tournament (local machine too slow for walk-forward on 1M+ rows)
- [ ] ML tournament on Colab ← NEXT
  - Upload parquet to Colab
  - Run 7 models (3 baselines + 4 ML) with walk-forward
  - Colab gives free GPU + faster CPU for training

### Next Up 🔲
- [ ] Build baselines (B0, B1, B2)
- [ ] Build ML models (M1-M4)
- [ ] Run tournament
- [ ] Pick winner
- [ ] Paper trade → live
