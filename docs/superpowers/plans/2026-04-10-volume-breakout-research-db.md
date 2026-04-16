# Volume Breakout Research Database — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `volume_breakouts` database table and backfill script that captures every 2x+ volume spike across 198 coins on 3 timeframes (15m, 1h, 4h), recording what happens to price at 6 forward horizons (12h to 1 month).

**Architecture:** SQLAlchemy model + Prisma schema for the table. Single backfill script reads existing 15m CSVs, resamples to 1h/4h, detects volume spikes, computes forward returns and MFE/MAE, batch-inserts to PostgreSQL.

**Tech Stack:** Python 3.12, SQLAlchemy, pandas, numpy, PostgreSQL. Reuses `load_coin` from `backtest_momentum_scanner.py`.

**Spec:** `docs/superpowers/specs/2026-04-10-volume-breakout-research-db-design.md`

---

### Task 1: Add VolumeBreakout SQLAlchemy Model

**Files:**
- Modify: `services/python/src/db/models.py`

- [ ] **Step 1: Add VolumeBreakout class to models.py**

Append after the `FearGreedIndex` class at the bottom of the file:

```python
class VolumeBreakout(Base):
    __tablename__ = "volume_breakouts"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "signal_time", name="uq_vb_sym_tf_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(4), nullable=False, index=True)
    signal_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    # Breakout bar OHLCV
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    quote_volume: Mapped[float] = mapped_column(Float, nullable=False)
    trades: Mapped[int] = mapped_column(Integer, nullable=True)

    # Volume context
    vol_ratio: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    vol_avg_20: Mapped[float] = mapped_column(Float, nullable=False)

    # Price context
    price_change_1bar: Mapped[float] = mapped_column(Float, nullable=True)
    price_change_2bar: Mapped[float] = mapped_column(Float, nullable=True)
    atr_14: Mapped[float] = mapped_column(Float, nullable=True)
    bar_range_pct: Mapped[float] = mapped_column(Float, nullable=True)
    upper_wick_pct: Mapped[float] = mapped_column(Float, nullable=True)
    lower_wick_pct: Mapped[float] = mapped_column(Float, nullable=True)
    body_pct: Mapped[float] = mapped_column(Float, nullable=True)
    direction: Mapped[int] = mapped_column(Integer, nullable=False)

    # Wider context
    dist_from_20_high: Mapped[float] = mapped_column(Float, nullable=True)
    dist_from_20_low: Mapped[float] = mapped_column(Float, nullable=True)
    price_vs_ema_50: Mapped[float] = mapped_column(Float, nullable=True)
    rsi_14: Mapped[float] = mapped_column(Float, nullable=True)
    volatility_20: Mapped[float] = mapped_column(Float, nullable=True)

    # BTC context
    btc_price: Mapped[float] = mapped_column(Float, nullable=True)
    btc_ret_24bar: Mapped[float] = mapped_column(Float, nullable=True)
    btc_ret_96bar: Mapped[float] = mapped_column(Float, nullable=True)
    btc_vol_ratio: Mapped[float] = mapped_column(Float, nullable=True)

    # Forward returns
    fwd_ret_12h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_ret_24h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_ret_48h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_ret_1w: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_ret_2w: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_ret_1m: Mapped[float] = mapped_column(Float, nullable=True)

    # Forward extremes (MFE/MAE)
    fwd_max_gain_12h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_max_gain_24h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_max_gain_48h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_max_gain_1w: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_max_loss_12h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_max_loss_24h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_max_loss_48h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_max_loss_1w: Mapped[float] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.now
    )
```

- [ ] **Step 2: Create the table in PostgreSQL**

```bash
cd services/python
python -c "
from sqlalchemy import create_engine
from src.db.models import Base
engine = create_engine('postgresql://postgres:MySQL100%25@localhost:5432/market')
Base.metadata.create_all(engine)
print('Table volume_breakouts created')
"
```

Expected: `Table volume_breakouts created`

- [ ] **Step 3: Commit**

```bash
git add services/python/src/db/models.py
git commit -m "feat: add VolumeBreakout SQLAlchemy model for research database"
```

---

### Task 2: Add Prisma Schema

**Files:**
- Modify: `services/nextjs/prisma/schema.prisma`

- [ ] **Step 1: Add VolumeBreakout model to Prisma schema**

Append at the end of `schema.prisma`:

```prisma
model VolumeBreakout {
  id              Int       @id @default(autoincrement())
  symbol          String    @db.VarChar(20)
  timeframe       String    @db.VarChar(4)
  signal_time     DateTime  @db.Timestamptz() @map("signal_time")

  open            Float
  high            Float
  low             Float
  close           Float
  volume          Float
  quote_volume    Float     @map("quote_volume")
  trades          Int?

  vol_ratio       Float     @map("vol_ratio")
  vol_avg_20      Float     @map("vol_avg_20")

  price_change_1bar   Float?  @map("price_change_1bar")
  price_change_2bar   Float?  @map("price_change_2bar")
  atr_14              Float?  @map("atr_14")
  bar_range_pct       Float?  @map("bar_range_pct")
  upper_wick_pct      Float?  @map("upper_wick_pct")
  lower_wick_pct      Float?  @map("lower_wick_pct")
  body_pct            Float?  @map("body_pct")
  direction           Int     @db.SmallInt

  dist_from_20_high   Float?  @map("dist_from_20_high")
  dist_from_20_low    Float?  @map("dist_from_20_low")
  price_vs_ema_50     Float?  @map("price_vs_ema_50")
  rsi_14              Float?  @map("rsi_14")
  volatility_20       Float?  @map("volatility_20")

  btc_price           Float?  @map("btc_price")
  btc_ret_24bar       Float?  @map("btc_ret_24bar")
  btc_ret_96bar       Float?  @map("btc_ret_96bar")
  btc_vol_ratio       Float?  @map("btc_vol_ratio")

  fwd_ret_12h         Float?  @map("fwd_ret_12h")
  fwd_ret_24h         Float?  @map("fwd_ret_24h")
  fwd_ret_48h         Float?  @map("fwd_ret_48h")
  fwd_ret_1w          Float?  @map("fwd_ret_1w")
  fwd_ret_2w          Float?  @map("fwd_ret_2w")
  fwd_ret_1m          Float?  @map("fwd_ret_1m")

  fwd_max_gain_12h    Float?  @map("fwd_max_gain_12h")
  fwd_max_gain_24h    Float?  @map("fwd_max_gain_24h")
  fwd_max_gain_48h    Float?  @map("fwd_max_gain_48h")
  fwd_max_gain_1w     Float?  @map("fwd_max_gain_1w")
  fwd_max_loss_12h    Float?  @map("fwd_max_loss_12h")
  fwd_max_loss_24h    Float?  @map("fwd_max_loss_24h")
  fwd_max_loss_48h    Float?  @map("fwd_max_loss_48h")
  fwd_max_loss_1w     Float?  @map("fwd_max_loss_1w")

  created_at      DateTime  @default(now()) @db.Timestamptz() @map("created_at")

  @@unique([symbol, timeframe, signal_time], name: "uq_vb_sym_tf_time")
  @@index([symbol])
  @@index([timeframe])
  @@index([signal_time])
  @@index([vol_ratio])
  @@map("volume_breakouts")
}
```

- [ ] **Step 2: Generate Prisma client**

```bash
cd services/nextjs
npx prisma generate
```

Expected: `Generated Prisma Client`

- [ ] **Step 3: Commit**

```bash
git add services/nextjs/prisma/schema.prisma
git commit -m "feat: add VolumeBreakout Prisma model"
```

---

### Task 3: Write the Backfill Script — Core Structure and Indicator Functions

**Files:**
- Create: `services/python/scripts/backfill_volume_breakouts.py`

- [ ] **Step 1: Create the script with imports, config, and indicator functions**

```python
# scripts/backfill_volume_breakouts.py
"""Backfill volume_breakouts table from historical 15m CSV data.

Scans 198 coins for volume spikes (>= 2x avg) on 15m, 1h, and 4h timeframes.
Records breakout context and forward returns at 6 horizons (12h to 1 month).
"""

import logging
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, ".")
from scripts.backtest_momentum_scanner import compute_atr, load_coin
from src.db.models import Base, VolumeBreakout

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
RAW_DIR = "C:/Users/togat/Desktop/AI-Finance/data/raw/15m"
VOL_RATIO_MIN = 2.0
BATCH_SIZE = 1000

# Forward return horizons in bars per timeframe
HORIZONS = {
    "15m": {"12h": 48, "24h": 96, "48h": 192, "1w": 672, "2w": 1344, "1m": 2880},
    "1h":  {"12h": 12, "24h": 24, "48h": 48,  "1w": 168, "2w": 336,  "1m": 720},
    "4h":  {"12h": 3,  "24h": 6,  "48h": 12,  "1w": 42,  "2w": 84,   "1m": 180},
}

# MFE/MAE horizons (only up to 1w — beyond that is too noisy)
MFE_HORIZONS = {"12h", "24h", "48h", "1w"}

# Cooldown per timeframe (bars) — prevent double-counting same move
COOLDOWN = {"15m": 8, "1h": 4, "4h": 2}


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI as array. Returns NaN for first `period` bars."""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def resample_to_timeframe(df_15m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample 15m bars to 1h or 4h. Returns DataFrame with same columns."""
    if tf == "15m":
        return df_15m.copy()

    rule = "1h" if tf == "1h" else "4h"
    df = df_15m.set_index("open_time").resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "trades": "sum",
    }).dropna(subset=["close"]).reset_index()
    return df
```

- [ ] **Step 2: Add the breakout detection and feature extraction function**

Append to the same file:

```python
def detect_and_extract(
    df: pd.DataFrame,
    timeframe: str,
    btc_close: np.ndarray,
    btc_volume: np.ndarray,
    btc_times: np.ndarray,
) -> list[dict]:
    """Detect volume spikes and extract all features + forward returns."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    quote_vol = df["quote_volume"].values.astype(float)
    trade_counts = df["trades"].values if "trades" in df.columns else np.zeros(len(close))
    times = df["open_time"].values

    n = len(close)
    if n < 50:
        return []

    # Precompute indicators
    vol_ma = pd.Series(volume).rolling(20).mean().values
    atr = compute_atr(high, low, close, period=14)
    ema_50 = pd.Series(close).ewm(span=50).mean().values
    rsi = compute_rsi(close, period=14)
    returns = np.zeros(n)
    returns[1:] = (close[1:] - close[:-1]) / close[:-1]
    volatility = pd.Series(returns).rolling(20).std().values

    horizons = HORIZONS[timeframe]
    cooldown = COOLDOWN[timeframe]
    records = []
    last_signal_bar = -cooldown - 1

    for i in range(21, n):
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            continue

        ratio = volume[i] / vol_ma[i]
        if ratio < VOL_RATIO_MIN:
            continue

        # Cooldown
        if i - last_signal_bar < cooldown:
            continue
        last_signal_bar = i

        # Direction
        direction = 1 if close[i] >= close[i - 1] else -1

        # Bar shape
        bar_range = high[i] - low[i]
        if bar_range == 0:
            bar_range_pct = 0.0
            upper_wick_pct = 0.0
            lower_wick_pct = 0.0
            body_pct = 0.0
        else:
            bar_range_pct = bar_range / close[i]
            body = abs(close[i] - df["open"].values[i])
            if close[i] >= df["open"].values[i]:  # Green candle
                upper_wick_pct = (high[i] - close[i]) / bar_range
                lower_wick_pct = (df["open"].values[i] - low[i]) / bar_range
            else:  # Red candle
                upper_wick_pct = (high[i] - df["open"].values[i]) / bar_range
                lower_wick_pct = (close[i] - low[i]) / bar_range
            body_pct = body / bar_range

        # Price context
        price_change_1bar = returns[i] if i >= 1 else None
        price_change_2bar = (close[i] - close[i - 2]) / close[i - 2] if i >= 2 else None

        # Distance from 20-bar high/low
        if i >= 20:
            high_20 = np.max(high[i - 20:i])
            low_20 = np.min(low[i - 20:i])
            dist_from_20_high = (close[i] - high_20) / high_20 if high_20 > 0 else 0
            dist_from_20_low = (close[i] - low_20) / low_20 if low_20 > 0 else 0
        else:
            dist_from_20_high = None
            dist_from_20_low = None

        price_vs_ema = (close[i] / ema_50[i] - 1) if not np.isnan(ema_50[i]) and ema_50[i] > 0 else None

        # BTC context
        sig_time = times[i]
        btc_idx = np.searchsorted(btc_times, sig_time)
        btc_idx = min(btc_idx, len(btc_close) - 1)
        btc_p = float(btc_close[btc_idx])

        btc_vol_ma = np.mean(btc_volume[max(0, btc_idx - 20):btc_idx]) if btc_idx >= 20 else None
        btc_vr = float(btc_volume[btc_idx] / btc_vol_ma) if btc_vol_ma and btc_vol_ma > 0 else None

        btc_r24 = None
        if btc_idx >= 24 and btc_close[btc_idx - 24] > 0:
            btc_r24 = (btc_close[btc_idx] - btc_close[btc_idx - 24]) / btc_close[btc_idx - 24]
        btc_r96 = None
        if btc_idx >= 96 and btc_close[btc_idx - 96] > 0:
            btc_r96 = (btc_close[btc_idx] - btc_close[btc_idx - 96]) / btc_close[btc_idx - 96]

        # Forward returns
        fwd = {}
        for label, bars in horizons.items():
            j = i + bars
            if j < n and close[i] > 0:
                ret = (close[j] - close[i]) / close[i]
                if direction == -1:
                    ret = -ret  # Flip for shorts — "did price move in our direction?"
                fwd[f"fwd_ret_{label}"] = ret
            else:
                fwd[f"fwd_ret_{label}"] = None

        # MFE / MAE (max favorable / adverse excursion)
        for label, bars in horizons.items():
            if label not in MFE_HORIZONS:
                continue
            j_end = min(i + bars + 1, n)
            if j_end <= i + 1 or close[i] == 0:
                fwd[f"fwd_max_gain_{label}"] = None
                fwd[f"fwd_max_loss_{label}"] = None
                continue

            fwd_high = np.max(high[i + 1:j_end])
            fwd_low = np.min(low[i + 1:j_end])

            if direction == 1:
                fwd[f"fwd_max_gain_{label}"] = (fwd_high - close[i]) / close[i]
                fwd[f"fwd_max_loss_{label}"] = (fwd_low - close[i]) / close[i]
            else:
                fwd[f"fwd_max_gain_{label}"] = (close[i] - fwd_low) / close[i]
                fwd[f"fwd_max_loss_{label}"] = (close[i] - fwd_high) / close[i]

        record = {
            "symbol": None,  # Filled by caller
            "timeframe": timeframe,
            "signal_time": pd.Timestamp(times[i]).to_pydatetime(),
            "open": float(df["open"].values[i]),
            "high": float(high[i]),
            "low": float(low[i]),
            "close": float(close[i]),
            "volume": float(volume[i]),
            "quote_volume": float(quote_vol[i]),
            "trades": int(trade_counts[i]) if not np.isnan(trade_counts[i]) else None,
            "vol_ratio": float(ratio),
            "vol_avg_20": float(vol_ma[i]),
            "price_change_1bar": float(price_change_1bar) if price_change_1bar is not None else None,
            "price_change_2bar": float(price_change_2bar) if price_change_2bar is not None else None,
            "atr_14": float(atr[i]) if not np.isnan(atr[i]) else None,
            "bar_range_pct": float(bar_range_pct),
            "upper_wick_pct": float(upper_wick_pct),
            "lower_wick_pct": float(lower_wick_pct),
            "body_pct": float(body_pct),
            "direction": direction,
            "dist_from_20_high": float(dist_from_20_high) if dist_from_20_high is not None else None,
            "dist_from_20_low": float(dist_from_20_low) if dist_from_20_low is not None else None,
            "price_vs_ema_50": float(price_vs_ema) if price_vs_ema is not None else None,
            "rsi_14": float(rsi[i]) if not np.isnan(rsi[i]) else None,
            "volatility_20": float(volatility[i]) if not np.isnan(volatility[i]) else None,
            "btc_price": btc_p,
            "btc_ret_24bar": float(btc_r24) if btc_r24 is not None else None,
            "btc_ret_96bar": float(btc_r96) if btc_r96 is not None else None,
            "btc_vol_ratio": float(btc_vr) if btc_vr is not None else None,
            **fwd,
        }
        records.append(record)

    return records
```

- [ ] **Step 3: Commit**

```bash
git add services/python/scripts/backfill_volume_breakouts.py
git commit -m "feat: add volume breakout detection and feature extraction"
```

---

### Task 4: Write the Backfill Script — Main Loop and DB Insert

**Files:**
- Modify: `services/python/scripts/backfill_volume_breakouts.py`

- [ ] **Step 1: Add main function with coin loop and batch insert**

Append to the script:

```python
def main():
    start_t = time.time()
    logger.info("Volume Breakout Research Backfill")
    logger.info(f"Data: {RAW_DIR}")
    logger.info(f"Vol ratio min: {VOL_RATIO_MIN}x")

    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Clear existing data for clean backfill
    deleted = session.execute(text("DELETE FROM volume_breakouts")).rowcount
    session.commit()
    logger.info(f"Cleared {deleted} existing rows")

    # Load BTC
    from pathlib import Path
    btc_dir = Path(RAW_DIR) / "BTCUSDT"
    btc_df = load_coin(btc_dir)
    if btc_df is None:
        logger.error("Cannot load BTC data!")
        return
    btc_close = btc_df["close"].values.astype(float)
    btc_volume = btc_df["volume"].values.astype(float)
    btc_times = btc_df["open_time"].values
    logger.info(f"BTC loaded: {len(btc_df)} bars")

    # Process all coins
    symbol_dirs = sorted([d for d in Path(RAW_DIR).iterdir() if d.is_dir()])
    logger.info(f"Processing {len(symbol_dirs)} coins across 3 timeframes...")

    total_inserted = 0
    coins_processed = 0
    batch = []

    for sym_dir in symbol_dirs:
        symbol = sym_dir.name
        df_15m = load_coin(sym_dir)
        if df_15m is None or len(df_15m) < 50:
            continue

        coins_processed += 1

        for tf in ["15m", "1h", "4h"]:
            df_tf = resample_to_timeframe(df_15m, tf)
            if len(df_tf) < 50:
                continue

            records = detect_and_extract(
                df=df_tf,
                timeframe=tf,
                btc_close=btc_close,
                btc_volume=btc_volume,
                btc_times=btc_times,
            )

            for rec in records:
                rec["symbol"] = symbol
                batch.append(rec)

            if len(batch) >= BATCH_SIZE:
                session.bulk_insert_mappings(VolumeBreakout, batch)
                session.commit()
                total_inserted += len(batch)
                batch = []

        if coins_processed % 20 == 0:
            logger.info(f"  {coins_processed} coins done, {total_inserted} rows inserted...")

    # Flush remaining
    if batch:
        session.bulk_insert_mappings(VolumeBreakout, batch)
        session.commit()
        total_inserted += len(batch)

    session.close()
    elapsed = time.time() - start_t
    logger.info(f"\nBackfill complete in {elapsed:.0f}s")
    logger.info(f"Coins processed: {coins_processed}")
    logger.info(f"Total rows inserted: {total_inserted}")

    # Print summary stats
    session = Session()
    for tf in ["15m", "1h", "4h"]:
        count = session.execute(
            text("SELECT COUNT(*) FROM volume_breakouts WHERE timeframe = :tf"),
            {"tf": tf},
        ).scalar()
        avg_vr = session.execute(
            text("SELECT AVG(vol_ratio) FROM volume_breakouts WHERE timeframe = :tf"),
            {"tf": tf},
        ).scalar()
        logger.info(f"  {tf}: {count} events, avg vol_ratio={avg_vr:.1f}x")

    # Forward return distribution sanity check
    for horizon in ["12h", "24h", "48h", "1w"]:
        col = f"fwd_ret_{horizon}"
        stats = session.execute(
            text(f"""
                SELECT
                    COUNT(*) FILTER (WHERE {col} IS NOT NULL) as n,
                    AVG({col}) as mean,
                    percentile_cont(0.25) WITHIN GROUP (ORDER BY {col}) as p25,
                    percentile_cont(0.50) WITHIN GROUP (ORDER BY {col}) as median,
                    percentile_cont(0.75) WITHIN GROUP (ORDER BY {col}) as p75
                FROM volume_breakouts
                WHERE timeframe = '15m'
            """)
        ).fetchone()
        logger.info(
            f"  15m fwd_ret_{horizon}: n={stats[0]}, "
            f"mean={stats[1]:.4f}, p25={stats[2]:.4f}, "
            f"median={stats[3]:.4f}, p75={stats[4]:.4f}"
        )
    session.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the backfill**

```bash
cd services/python
python scripts/backfill_volume_breakouts.py
```

Expected output:
- Processes 198 coins across 3 timeframes
- Inserts 100k-280k rows
- Prints per-timeframe counts and forward return distribution
- Completes in under 60 minutes

- [ ] **Step 3: Verify data in PostgreSQL**

```bash
cd services/python
python -c "
from sqlalchemy import create_engine, text
engine = create_engine('postgresql://postgres:MySQL100%25@localhost:5432/market')
with engine.connect() as conn:
    total = conn.execute(text('SELECT COUNT(*) FROM volume_breakouts')).scalar()
    print(f'Total rows: {total}')
    for tf in ['15m', '1h', '4h']:
        n = conn.execute(text(\"SELECT COUNT(*) FROM volume_breakouts WHERE timeframe = :tf\"), {'tf': tf}).scalar()
        print(f'  {tf}: {n}')
    # Check a sample row
    row = conn.execute(text('SELECT symbol, timeframe, signal_time, vol_ratio, fwd_ret_24h, fwd_max_gain_24h FROM volume_breakouts ORDER BY vol_ratio DESC LIMIT 5')).fetchall()
    for r in row:
        print(f'  {r[0]} {r[1]} {r[2]} vol={r[3]:.1f}x fwd24h={r[4]:.2%} mfe24h={r[5]:.2%}')
"
```

Expected: 100k+ total rows, sample rows with reasonable forward returns.

- [ ] **Step 4: Commit**

```bash
git add services/python/scripts/backfill_volume_breakouts.py
git commit -m "feat: add volume breakout backfill script — main loop and DB insert"
```

---

### Task 5: Quick Sanity Analysis

**Files:**
- No new files — just SQL queries to validate the data makes sense

- [ ] **Step 1: Run sanity queries**

```bash
cd services/python
python -c "
from sqlalchemy import create_engine, text
engine = create_engine('postgresql://postgres:MySQL100%25@localhost:5432/market')
with engine.connect() as conn:
    # 1. Do higher vol_ratio spikes lead to bigger moves?
    print('=== Vol ratio vs forward returns (15m, direction-adjusted) ===')
    rows = conn.execute(text('''
        SELECT
            CASE
                WHEN vol_ratio < 3 THEN '2-3x'
                WHEN vol_ratio < 5 THEN '3-5x'
                WHEN vol_ratio < 10 THEN '5-10x'
                ELSE '10x+'
            END as bucket,
            COUNT(*) as n,
            ROUND(AVG(fwd_ret_24h)::numeric, 4) as avg_24h,
            ROUND(AVG(fwd_ret_48h)::numeric, 4) as avg_48h,
            ROUND(AVG(fwd_ret_1w)::numeric, 4) as avg_1w,
            ROUND(AVG(fwd_max_gain_24h)::numeric, 4) as avg_mfe_24h
        FROM volume_breakouts
        WHERE timeframe = '15m' AND fwd_ret_24h IS NOT NULL
        GROUP BY 1 ORDER BY 1
    ''')).fetchall()
    for r in rows:
        print(f'  {r[0]}: n={r[1]}, ret24h={r[2]}, ret48h={r[3]}, ret1w={r[4]}, mfe24h={r[5]}')

    # 2. Which timeframe has best signal quality?
    print()
    print('=== Timeframe comparison ===')
    rows = conn.execute(text('''
        SELECT
            timeframe,
            COUNT(*) as n,
            ROUND(AVG(fwd_ret_24h)::numeric, 4) as avg_24h,
            ROUND(AVG(fwd_ret_48h)::numeric, 4) as avg_48h,
            ROUND(AVG(fwd_max_gain_24h)::numeric, 4) as avg_mfe_24h,
            ROUND(AVG(fwd_max_loss_24h)::numeric, 4) as avg_mae_24h
        FROM volume_breakouts
        WHERE fwd_ret_24h IS NOT NULL
        GROUP BY 1 ORDER BY 1
    ''')).fetchall()
    for r in rows:
        print(f'  {r[0]}: n={r[1]}, ret24h={r[2]}, ret48h={r[3]}, mfe24h={r[4]}, mae24h={r[5]}')

    # 3. How many events have positive forward returns?
    print()
    print('=== Win rates by horizon (15m, direction-adjusted) ===')
    for h in ['12h', '24h', '48h', '1w', '2w', '1m']:
        col = f'fwd_ret_{h}'
        row = conn.execute(text(f'''
            SELECT
                COUNT(*) FILTER (WHERE {col} > 0) as wins,
                COUNT(*) FILTER (WHERE {col} IS NOT NULL) as total
            FROM volume_breakouts WHERE timeframe = \\'15m\\'
        ''')).fetchone()
        wr = row[0] / row[1] * 100 if row[1] > 0 else 0
        print(f'  {h}: {row[0]}/{row[1]} = {wr:.1f}% positive')
"
```

This tells us:
- Do bigger volume spikes lead to better follow-through?
- Which timeframe has the best signal?
- What's the natural win rate at each horizon?

- [ ] **Step 2: Commit results to MODEL_EVOLUTION.md**

Update `models/results/MODEL_EVOLUTION.md` with the actual findings from the sanity analysis. Add a new section at the top: "Volume Breakout Research (2026-04-10)" with the real numbers.
