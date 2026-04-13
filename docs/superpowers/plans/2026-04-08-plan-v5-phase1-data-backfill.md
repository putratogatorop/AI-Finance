# V5 Phase 1: Data Backfill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backfill 3 new data sources (Gate.io contract stats, Binance aggregated trades, Fear & Greed index) into PostgreSQL for the v5 two-stage extreme move predictor.

**Architecture:** Gate.io `contract_stats` endpoint provides OI, liquidations, and long/short ratio in one call at 5-min intervals (aggregate to 1h). Binance Vision provides aggregated trades as monthly CSVs (aggregate to 1h large-trade metrics). Fear & Greed is one API call for all history. All stored in new PostgreSQL tables.

**Tech Stack:** Python 3.12, SQLAlchemy, PostgreSQL, urllib/httpx

---

## API Facts (verified)

| Source | Endpoint | Interval | History | Auth |
|---|---|---|---|---|
| Gate.io contract_stats | `/api/v4/futures/usdt/contract_stats` | 5 min | Jan 2024+ (~2 years) | No |
| Binance Vision aggTrades | CDN zip files | tick-level | 2023+ | No |
| Fear & Greed | `api.alternative.me/fng` | daily | 2018+ | No |

Gate.io `contract_stats` returns per record: `open_interest`, `open_interest_usd`, `long_liq_usd`, `short_liq_usd`, `lsr_taker`, `lsr_account`, `top_lsr_size`, `top_lsr_account`, `mark_price`, `long_liq_size`, `short_liq_size`.

## File Structure

| File | Responsibility |
|---|---|
| `src/db/models.py` | Add `ContractStats1h` and `FearGreedIndex` models |
| `tests/test_db_models_v5.py` | Test new models |
| `scripts/backfill_contract_stats.py` | Pull Gate.io contract_stats, aggregate 5min→1h, upsert to PG |
| `tests/test_backfill_contract_stats.py` | Test API parsing and aggregation |
| `scripts/backfill_agg_trades.py` | Download Binance aggTrades, compute hourly large-trade metrics |
| `tests/test_backfill_agg_trades.py` | Test trade aggregation logic |
| `scripts/backfill_fear_greed.py` | Pull Fear & Greed index, upsert to PG |
| `scripts/validate_v5_data.py` | Validate all v5 data completeness |

## Coins: BTC, LINK, XRP, AVAX

---

### Task 1: New Database Models

**Files:**
- Modify: `services/python/src/db/models.py`
- Create: `services/python/tests/test_db_models_v5.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_db_models_v5.py
import sys
sys.path.insert(0, ".")

from datetime import datetime, UTC
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from src.db.models import Base, ContractStats1h, FearGreedIndex


def make_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def test_contract_stats_roundtrip():
    engine = make_engine()
    with Session(engine) as session:
        row = ContractStats1h(
            asset="BTC",
            timestamp=datetime(2024, 6, 1, 0, 0, tzinfo=UTC),
            open_interest_usd=4_000_000_000.0,
            long_liq_usd=500_000.0,
            short_liq_usd=300_000.0,
            lsr_taker=1.05,
            lsr_account=0.95,
            top_lsr_size=1.10,
            top_lsr_account=0.88,
        )
        session.add(row)
        session.commit()

        result = session.query(ContractStats1h).filter_by(asset="BTC").first()
        assert result is not None
        assert result.open_interest_usd == 4_000_000_000.0
        assert result.lsr_taker == 1.05


def test_fear_greed_roundtrip():
    engine = make_engine()
    with Session(engine) as session:
        row = FearGreedIndex(
            date=datetime(2024, 6, 1, tzinfo=UTC),
            value=25,
            classification="Extreme Fear",
        )
        session.add(row)
        session.commit()

        result = session.query(FearGreedIndex).first()
        assert result is not None
        assert result.value == 25
        assert result.classification == "Extreme Fear"
```

- [ ] **Step 2: Run test, verify fails**

Run: `cd services/python && python -m pytest tests/test_db_models_v5.py -v`

- [ ] **Step 3: Add models to src/db/models.py**

Add after the existing `FundingRate` class:

```python
class ContractStats1h(Base):
    __tablename__ = "contract_stats_1h"
    __table_args__ = (UniqueConstraint("asset", "timestamp", name="uq_cstats_asset_ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    open_interest_usd: Mapped[float] = mapped_column(Float, nullable=False)
    long_liq_usd: Mapped[float] = mapped_column(Float, nullable=True)
    short_liq_usd: Mapped[float] = mapped_column(Float, nullable=True)
    lsr_taker: Mapped[float] = mapped_column(Float, nullable=True)
    lsr_account: Mapped[float] = mapped_column(Float, nullable=True)
    top_lsr_size: Mapped[float] = mapped_column(Float, nullable=True)
    top_lsr_account: Mapped[float] = mapped_column(Float, nullable=True)


class FearGreedIndex(Base):
    __tablename__ = "fear_greed_index"
    __table_args__ = (UniqueConstraint("date", name="uq_fgi_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    value: Mapped[int] = mapped_column(Integer, nullable=False)
    classification: Mapped[str] = mapped_column(String(50), nullable=False)
```

- [ ] **Step 4: Run test, verify passes**
- [ ] **Step 5: Commit**

```bash
git add services/python/src/db/models.py services/python/tests/test_db_models_v5.py
git commit -m "feat: add ContractStats1h and FearGreedIndex database models"
```

---

### Task 2: Backfill Gate.io Contract Stats

**Files:**
- Create: `services/python/scripts/backfill_contract_stats.py`
- Create: `services/python/tests/test_backfill_contract_stats.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_backfill_contract_stats.py
import sys
sys.path.insert(0, ".")

from datetime import datetime, UTC


SAMPLE_RESPONSE = [
    {
        "time": 1717200000, "open_interest_usd": 4000000000.0,
        "long_liq_usd": 50000.0, "short_liq_usd": 30000.0,
        "lsr_taker": 1.05, "lsr_account": 0.95,
        "top_lsr_size": 1.10, "top_lsr_account": 0.88,
        "open_interest": 50000, "mark_price": 70000.0,
        "long_liq_size": 5, "short_liq_size": 3,
        "long_liq_amount": 50000, "short_liq_amount": 30000,
    },
    {
        "time": 1717200300, "open_interest_usd": 4010000000.0,
        "long_liq_usd": 10000.0, "short_liq_usd": 5000.0,
        "lsr_taker": 1.02, "lsr_account": 0.97,
        "top_lsr_size": 1.08, "top_lsr_account": 0.90,
        "open_interest": 50100, "mark_price": 70050.0,
        "long_liq_size": 1, "short_liq_size": 0,
        "long_liq_amount": 10000, "short_liq_amount": 5000,
    },
]


def test_parse_contract_stats():
    from scripts.backfill_contract_stats import parse_contract_stats

    rows = parse_contract_stats("BTC", SAMPLE_RESPONSE)
    assert len(rows) == 2
    assert rows[0]["asset"] == "BTC"
    assert rows[0]["open_interest_usd"] == 4000000000.0
    assert rows[0]["lsr_taker"] == 1.05
    assert isinstance(rows[0]["timestamp"], datetime)


def test_aggregate_to_1h():
    from scripts.backfill_contract_stats import aggregate_5min_to_1h
    import pandas as pd
    import numpy as np

    # 12 records = 1 hour of 5-min data
    ts = pd.date_range("2024-06-01 00:00", periods=12, freq="5min", tz="UTC")
    data = []
    for i, t in enumerate(ts):
        data.append({
            "asset": "BTC", "timestamp": t,
            "open_interest_usd": 4e9 + i * 1e6,
            "long_liq_usd": 10000.0 + i * 1000,
            "short_liq_usd": 5000.0 + i * 500,
            "lsr_taker": 1.0 + i * 0.01,
            "lsr_account": 0.9 + i * 0.01,
            "top_lsr_size": 1.1, "top_lsr_account": 0.88,
        })

    df = pd.DataFrame(data)
    result = aggregate_5min_to_1h(df)

    assert len(result) == 1
    # OI should be last value of the hour
    assert result.iloc[0]["open_interest_usd"] == 4e9 + 11 * 1e6
    # Liquidations should be summed
    expected_liq_long = sum(10000.0 + i * 1000 for i in range(12))
    assert abs(result.iloc[0]["long_liq_usd"] - expected_liq_long) < 0.01
    # LSR should be mean
    expected_lsr = np.mean([1.0 + i * 0.01 for i in range(12)])
    assert abs(result.iloc[0]["lsr_taker"] - expected_lsr) < 0.001
```

- [ ] **Step 2: Run test, verify fails**

- [ ] **Step 3: Implement backfill_contract_stats.py**

```python
# scripts/backfill_contract_stats.py
"""Backfill Gate.io contract stats: OI, liquidations, long/short ratio.

Endpoint: GET /api/v4/futures/usdt/contract_stats
Returns 5-minute interval data. We aggregate to 1h before storing.

Output: contract_stats_1h table in PostgreSQL
"""

import json
import logging
import sys
import time
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.db.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
GATEIO_BASE = "https://api.gateio.ws/api/v4/futures/usdt/contract_stats"

V5_CONTRACTS = {
    "BTC": "BTC_USDT",
    "LINK": "LINK_USDT",
    "XRP": "XRP_USDT",
    "AVAX": "AVAX_USDT",
}

START_DATE = datetime(2024, 1, 1, tzinfo=UTC)
PAGE_LIMIT = 1000  # Max records per request


def parse_contract_stats(asset: str, data: list[dict]) -> list[dict]:
    rows = []
    for item in data:
        rows.append({
            "asset": asset,
            "timestamp": datetime.fromtimestamp(item["time"], tz=UTC),
            "open_interest_usd": float(item.get("open_interest_usd", 0)),
            "long_liq_usd": float(item.get("long_liq_usd", 0) or 0),
            "short_liq_usd": float(item.get("short_liq_usd", 0) or 0),
            "lsr_taker": float(item.get("lsr_taker", 0) or 0),
            "lsr_account": float(item.get("lsr_account", 0) or 0),
            "top_lsr_size": float(item.get("top_lsr_size", 0) or 0),
            "top_lsr_account": float(item.get("top_lsr_account", 0) or 0),
        })
    return rows


def aggregate_5min_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5-min contract stats to 1h bars."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    agg = df.resample("1h").agg({
        "asset": "first",
        "open_interest_usd": "last",       # Snapshot: use last value
        "long_liq_usd": "sum",             # Liquidations: sum over hour
        "short_liq_usd": "sum",
        "lsr_taker": "mean",               # Ratios: average over hour
        "lsr_account": "mean",
        "top_lsr_size": "mean",
        "top_lsr_account": "mean",
    }).dropna(subset=["asset"])

    return agg.reset_index()


def fetch_stats_page(contract: str, from_ts: int, to_ts: int) -> list[dict]:
    url = f"{GATEIO_BASE}?contract={contract}&from={from_ts}&to={to_ts}&limit={PAGE_LIMIT}"
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    resp = urlopen(req, timeout=15)
    return json.loads(resp.read().decode("utf-8"))


def upsert_stats_batch(engine, rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = text("""
        INSERT INTO contract_stats_1h
            (asset, timestamp, open_interest_usd, long_liq_usd, short_liq_usd,
             lsr_taker, lsr_account, top_lsr_size, top_lsr_account)
        VALUES
            (:asset, :timestamp, :open_interest_usd, :long_liq_usd, :short_liq_usd,
             :lsr_taker, :lsr_account, :top_lsr_size, :top_lsr_account)
        ON CONFLICT (asset, timestamp) DO UPDATE SET
            open_interest_usd = EXCLUDED.open_interest_usd,
            long_liq_usd = EXCLUDED.long_liq_usd,
            short_liq_usd = EXCLUDED.short_liq_usd,
            lsr_taker = EXCLUDED.lsr_taker,
            lsr_account = EXCLUDED.lsr_account,
            top_lsr_size = EXCLUDED.top_lsr_size,
            top_lsr_account = EXCLUDED.top_lsr_account
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)


def backfill_asset(engine, asset: str, contract: str) -> int:
    now = datetime.now(UTC)
    cursor = START_DATE
    total = 0

    while cursor < now:
        # 1000 records at 5min = ~3.5 days per request
        window_end = min(cursor + timedelta(days=3), now)
        from_ts = int(cursor.timestamp())
        to_ts = int(window_end.timestamp())

        try:
            data = fetch_stats_page(contract, from_ts, to_ts)
            if data:
                parsed = parse_contract_stats(asset, data)
                df = pd.DataFrame(parsed)
                hourly = aggregate_5min_to_1h(df)
                rows = hourly.to_dict("records")
                upsert_stats_batch(engine, rows)
                total += len(rows)
                logger.info(f"  {asset}: +{len(rows)} hours ({cursor.date()} -> {window_end.date()})")
            else:
                logger.info(f"  {asset}: no data ({cursor.date()} -> {window_end.date()})")
        except HTTPError as e:
            logger.error(f"  {asset}: HTTP {e.code} ({cursor.date()})")
        except Exception as e:
            logger.error(f"  {asset}: error: {e}")

        cursor = window_end
        time.sleep(0.3)

    return total


def main():
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)

    logger.info("Gate.io Contract Stats Backfill (OI + Liquidations + LSR)")
    logger.info(f"Assets: {list(V5_CONTRACTS.keys())}")
    logger.info(f"Start: {START_DATE.date()}")

    for asset, contract in V5_CONTRACTS.items():
        logger.info(f"\nBackfilling {asset} ({contract})...")
        count = backfill_asset(engine, asset, contract)
        logger.info(f"  {asset}: {count} total hourly records")

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM contract_stats_1h")).scalar()
        assets = conn.execute(text("SELECT COUNT(DISTINCT asset) FROM contract_stats_1h")).scalar()
    logger.info(f"\nDONE! {total:,} records, {assets} assets")
    engine.dispose()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, verify passes**
- [ ] **Step 5: Commit**

```bash
git add services/python/scripts/backfill_contract_stats.py services/python/tests/test_backfill_contract_stats.py
git commit -m "feat: add Gate.io contract stats backfill (OI, liquidations, LSR)"
```

- [ ] **Step 6: Run the backfill**

Run: `cd services/python && python scripts/backfill_contract_stats.py`
Expected: ~4 coins x ~250 API calls (~2 years / 3 days per call). Takes ~5-10 min with rate limiting. ~17,520 hourly records per coin.

---

### Task 3: Backfill Fear & Greed Index

**Files:**
- Create: `services/python/scripts/backfill_fear_greed.py`

- [ ] **Step 1: Implement**

```python
# scripts/backfill_fear_greed.py
"""Backfill Fear & Greed Index from alternative.me API.

One API call gets all history. Daily values stored in fear_greed_index table.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from urllib.request import Request, urlopen

from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.db.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"


def main():
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)

    logger.info("Fear & Greed Index Backfill")

    req = Request(FNG_URL, headers={"User-Agent": "Mozilla/5.0"})
    resp = urlopen(req, timeout=30)
    payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data", [])
    logger.info(f"Fetched {len(data)} daily records")

    rows = []
    for item in data:
        ts = int(item["timestamp"])
        rows.append({
            "date": datetime.fromtimestamp(ts, tz=UTC),
            "value": int(item["value"]),
            "classification": item["value_classification"],
        })

    if rows:
        sql = text("""
            INSERT INTO fear_greed_index (date, value, classification)
            VALUES (:date, :value, :classification)
            ON CONFLICT (date) DO UPDATE SET
                value = EXCLUDED.value, classification = EXCLUDED.classification
        """)
        with engine.begin() as conn:
            conn.execute(sql, rows)

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM fear_greed_index")).scalar()
        min_d = conn.execute(text("SELECT MIN(date) FROM fear_greed_index")).scalar()
        max_d = conn.execute(text("SELECT MAX(date) FROM fear_greed_index")).scalar()

    logger.info(f"DONE! {total:,} records [{min_d} -> {max_d}]")
    engine.dispose()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add services/python/scripts/backfill_fear_greed.py
git commit -m "feat: add Fear & Greed index backfill"
```

- [ ] **Step 3: Run it**

Run: `cd services/python && python scripts/backfill_fear_greed.py`
Expected: ~2000+ daily records, one API call, instant.

---

### Task 4: Backfill Binance Aggregated Trades

**Files:**
- Create: `services/python/scripts/backfill_agg_trades.py`
- Create: `services/python/tests/test_backfill_agg_trades.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_backfill_agg_trades.py
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd


def test_compute_hourly_trade_metrics():
    from scripts.backfill_agg_trades import compute_hourly_trade_metrics

    n = 1000
    rng = np.random.RandomState(42)
    ts = pd.date_range("2024-06-01", periods=n, freq="5s", tz="UTC")
    qty = rng.exponential(1.0, n)  # Most trades small, few large
    is_buyer = rng.choice([True, False], n)

    df = pd.DataFrame({
        "timestamp": ts,
        "quantity": qty,
        "is_buyer_maker": is_buyer,
    })

    result = compute_hourly_trade_metrics(df)

    assert "large_trade_ratio" in result.columns
    assert "large_trade_imbalance" in result.columns
    assert "trade_count_ratio" in result.columns
    assert len(result) >= 1

    # large_trade_ratio should be between 0 and 1
    assert result["large_trade_ratio"].dropna().min() >= 0
    assert result["large_trade_ratio"].dropna().max() <= 1

    # imbalance should be between -1 and 1
    valid = result["large_trade_imbalance"].dropna()
    assert valid.min() >= -1.01
    assert valid.max() <= 1.01
```

- [ ] **Step 2: Run test, verify fails**

- [ ] **Step 3: Implement backfill_agg_trades.py**

```python
# scripts/backfill_agg_trades.py
"""Backfill Binance aggregated trades and compute hourly large-trade metrics.

Downloads aggTrades from Binance Vision CDN, computes per-hour:
- large_trade_ratio: fraction of trades above 95th percentile size
- large_trade_imbalance: (large_buy - large_sell) / total_large
- trade_count_ratio: hourly count / 24h SMA count

Stores in asset_prices_15m is not practical for tick data.
Instead, we compute hourly metrics and store as a CSV/Parquet
that gets merged during feature building.

Output: data/features/v5/agg_trades/{ASSET}_hourly_trades.parquet
"""

import csv
import io
import logging
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/monthly/aggTrades"
OUTPUT_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/features/v5/agg_trades")
START_YEAR = 2024
START_MONTH = 1

V5_SYMBOLS = ["BTCUSDT", "LINKUSDT", "XRPUSDT", "AVAXUSDT"]


def compute_hourly_trade_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly large-trade metrics from tick-level aggTrades.

    Input: DataFrame with timestamp, quantity, is_buyer_maker
    Output: DataFrame with hourly large_trade_ratio, large_trade_imbalance, trade_count_ratio
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # Define "large" as > 95th percentile of quantity (expanding to avoid lookahead)
    qty = df["quantity"]
    large_threshold = qty.expanding(min_periods=100).quantile(0.95)
    df["is_large"] = qty > large_threshold

    # Hourly aggregation
    hourly = df.resample("1h").agg(
        total_trades=("quantity", "count"),
        large_count=("is_large", "sum"),
        large_buy=("is_large", lambda x: ((x) & (~df.loc[x.index, "is_buyer_maker"])).sum()),
        large_sell=("is_large", lambda x: ((x) & (df.loc[x.index, "is_buyer_maker"])).sum()),
    )

    hourly["large_trade_ratio"] = hourly["large_count"] / hourly["total_trades"].replace(0, np.nan)
    total_large = (hourly["large_buy"] + hourly["large_sell"]).replace(0, np.nan)
    hourly["large_trade_imbalance"] = (hourly["large_buy"] - hourly["large_sell"]) / total_large

    # Trade count ratio: current hour / 24h SMA
    hourly["trade_count_sma"] = hourly["total_trades"].rolling(24, min_periods=1).mean()
    hourly["trade_count_ratio"] = hourly["total_trades"] / hourly["trade_count_sma"].replace(0, np.nan)

    result = hourly[["large_trade_ratio", "large_trade_imbalance", "trade_count_ratio"]].reset_index()
    return result


def download_and_process_month(symbol: str, month: str) -> pd.DataFrame | None:
    """Download one month of aggTrades and compute hourly metrics."""
    url = f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{month}.zip"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urlopen(req, timeout=60)
        zip_data = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                # aggTrades CSV: agg_trade_id, price, quantity, first_trade_id, last_trade_id, timestamp, is_buyer_maker
                df = pd.read_csv(
                    f, header=None,
                    names=["agg_id", "price", "quantity", "first_id", "last_id", "timestamp", "is_buyer_maker"],
                    usecols=["timestamp", "quantity", "is_buyer_maker"],
                )

        # Convert timestamp (milliseconds or microseconds)
        ts_raw = df["timestamp"].iloc[0]
        if ts_raw > 1e13:
            df["timestamp"] = pd.to_datetime(df["timestamp"] / 1000, unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        df["quantity"] = df["quantity"].astype(float)
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)

        return compute_hourly_trade_metrics(df)

    except HTTPError as e:
        if e.code == 404:
            return None
        logger.error(f"HTTP {e.code} for {symbol} {month}")
        return None
    except Exception as e:
        logger.error(f"Error processing {symbol} {month}: {e}")
        return None


def generate_months(start_year, start_month):
    now = datetime.now(timezone.utc)
    months = []
    y, m = start_year, start_month
    while (y, m) <= (now.year, now.month):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    months = generate_months(START_YEAR, START_MONTH)

    logger.info("Binance Aggregated Trades Backfill")
    logger.info(f"Symbols: {V5_SYMBOLS}")
    logger.info(f"Months: {len(months)} ({months[0]} to {months[-1]})")

    for symbol in V5_SYMBOLS:
        logger.info(f"\nProcessing {symbol}...")
        all_hourly = []

        for month in months:
            logger.info(f"  {symbol} {month}...")
            hourly = download_and_process_month(symbol, month)
            if hourly is not None and len(hourly) > 0:
                all_hourly.append(hourly)
                logger.info(f"    {len(hourly)} hourly records")
            else:
                logger.info(f"    no data")
            time.sleep(0.5)

        if all_hourly:
            combined = pd.concat(all_hourly, ignore_index=True)
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            out_path = OUTPUT_DIR / f"{symbol}_hourly_trades.parquet"
            combined.to_parquet(out_path, index=False)
            logger.info(f"  {symbol}: {len(combined):,} total hourly records -> {out_path}")
        else:
            logger.warning(f"  {symbol}: no data at all!")

    logger.info("\nDONE!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, verify passes**
- [ ] **Step 5: Commit**

```bash
git add services/python/scripts/backfill_agg_trades.py services/python/tests/test_backfill_agg_trades.py
git commit -m "feat: add Binance aggregated trades backfill with large-trade metrics"
```

- [ ] **Step 6: Run the backfill**

Run: `cd services/python && python scripts/backfill_agg_trades.py`
Expected: Downloads ~100 zip files (4 symbols x ~27 months). aggTrades files are LARGE (100MB+ per month for BTC). This will take 30-60 minutes. Output: 4 Parquet files in `data/features/v5/agg_trades/`.

**WARNING:** BTC aggTrades are very large (~500MB per month compressed). If this is too slow, we can skip BTC aggTrades and use only the existing `taker_buy_ratio` from OHLCV for BTC, while using aggTrades for the 3 alts.

---

### Task 5: Data Validation

**Files:**
- Create: `services/python/scripts/validate_v5_data.py`

- [ ] **Step 1: Implement**

```python
# scripts/validate_v5_data.py
"""Validate v5 data completeness."""

import logging
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
TRADES_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/features/v5/agg_trades")
V5_ASSETS = ["BTC", "LINK", "XRP", "AVAX"]


def main():
    engine = create_engine(DB_URL)
    issues = []

    logger.info("=" * 60)
    logger.info("V5 Data Validation")
    logger.info("=" * 60)

    # 1. Contract stats
    logger.info("\n--- Contract Stats (1h) ---")
    with engine.connect() as conn:
        for asset in V5_ASSETS:
            r = conn.execute(text(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) "
                "FROM contract_stats_1h WHERE asset = :a"
            ), {"a": asset}).fetchone()
            count, min_ts, max_ts = r
            status = "OK" if count >= 10000 else "LOW"
            if status == "LOW":
                issues.append(f"{asset}: only {count:,} contract_stats rows")
            logger.info(f"  {asset}: {count:,} rows [{min_ts} -> {max_ts}] {status}")

    # 2. Fear & Greed
    logger.info("\n--- Fear & Greed Index ---")
    with engine.connect() as conn:
        r = conn.execute(text(
            "SELECT COUNT(*), MIN(date), MAX(date) FROM fear_greed_index"
        )).fetchone()
        count, min_d, max_d = r
        status = "OK" if count >= 1000 else "LOW"
        if status == "LOW":
            issues.append(f"Fear & Greed: only {count} records")
        logger.info(f"  {count:,} records [{min_d} -> {max_d}] {status}")

    # 3. Aggregated trades
    logger.info("\n--- Aggregated Trade Metrics ---")
    for symbol in ["BTCUSDT", "LINKUSDT", "XRPUSDT", "AVAXUSDT"]:
        path = TRADES_DIR / f"{symbol}_hourly_trades.parquet"
        if path.exists():
            import pandas as pd
            df = pd.read_parquet(path)
            logger.info(f"  {symbol}: {len(df):,} hourly records")
        else:
            issues.append(f"{symbol}: no aggTrades file")
            logger.info(f"  {symbol}: MISSING")

    # 4. Existing data (sanity check)
    logger.info("\n--- Existing 15m OHLCV + Funding ---")
    with engine.connect() as conn:
        for asset in V5_ASSETS:
            ohlcv = conn.execute(text(
                "SELECT COUNT(*) FROM asset_prices_15m WHERE asset = :a"
            ), {"a": asset}).scalar()
            funding = conn.execute(text(
                "SELECT COUNT(*) FROM funding_rates WHERE asset = :a"
            ), {"a": asset}).scalar()
            logger.info(f"  {asset}: {ohlcv:,} OHLCV, {funding:,} funding rates")

    logger.info("\n" + "=" * 60)
    if issues:
        logger.warning(f"ISSUES ({len(issues)}):")
        for i in issues:
            logger.warning(f"  - {i}")
    else:
        logger.info("ALL CHECKS PASSED")

    engine.dispose()
    return len(issues) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
```

- [ ] **Step 2: Commit**

```bash
git add services/python/scripts/validate_v5_data.py
git commit -m "feat: add v5 data validation script"
```

- [ ] **Step 3: Run validation after all backfills complete**

Run: `cd services/python && python scripts/validate_v5_data.py`
