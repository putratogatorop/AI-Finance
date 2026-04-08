# V4 Week 1-2: Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backfill 3 years of 15-minute OHLCV candles (Binance Vision) and funding rate history (Gate.io) for 7 assets into PostgreSQL.

**Architecture:** Download 15min CSVs from Binance Vision static CDN (no API key, works from Indonesia). Pull funding rates from Gate.io REST API. Store in two new PostgreSQL tables. Existing 4h data and tables remain untouched.

**Tech Stack:** Python 3.12, SQLAlchemy, PostgreSQL, urllib (Binance Vision CDN), httpx (Gate.io API)

---

## File Structure

| File | Responsibility |
|---|---|
| `src/db/models.py` | Add `AssetPrice15m` and `FundingRate` SQLAlchemy models |
| `tests/test_db_models_v4.py` | Test new models create/query correctly |
| `scripts/backfill_binance_15m.py` | Download 15min OHLCV CSVs from Binance Vision |
| `tests/test_backfill_binance_15m.py` | Test download logic with mocked HTTP |
| `scripts/upsert_15m_to_pg.py` | Load 15min CSVs into PostgreSQL |
| `tests/test_upsert_15m.py` | Test upsert logic with real DB |
| `scripts/backfill_funding_rates.py` | Pull funding rate history from Gate.io API |
| `tests/test_backfill_funding.py` | Test Gate.io API parsing and DB insert |
| `scripts/validate_v4_data.py` | Validate data completeness and quality |

## Assets

```python
V4_SYMBOLS_USDT = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT"]
V4_ASSETS = ["BTC", "ETH", "SOL", "DOGE", "XRP", "AVAX", "LINK"]
```

---

### Task 1: Add New Database Models

**Files:**
- Modify: `services/python/src/db/models.py`
- Create: `services/python/tests/test_db_models_v4.py`

- [ ] **Step 1: Write failing test for AssetPrice15m model**

```python
# tests/test_db_models_v4.py
from datetime import datetime, UTC
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from src.db.models import Base, AssetPrice15m, FundingRate


def make_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def test_asset_price_15m_roundtrip():
    engine = make_engine()
    with Session(engine) as session:
        row = AssetPrice15m(
            asset="BTC",
            timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            open=42000.0,
            high=42100.0,
            low=41900.0,
            close=42050.0,
            volume=123.45,
            quote_volume=5_180_000.0,
            trades=1500,
            taker_buy_base=60.0,
            taker_buy_quote=2_520_000.0,
        )
        session.add(row)
        session.commit()

        result = session.query(AssetPrice15m).filter_by(asset="BTC").first()
        assert result is not None
        assert result.close == 42050.0
        assert result.trades == 1500
        assert result.taker_buy_base == 60.0


def test_funding_rate_roundtrip():
    engine = make_engine()
    with Session(engine) as session:
        row = FundingRate(
            asset="SOL",
            timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            funding_rate=0.0001,
            source="gateio",
        )
        session.add(row)
        session.commit()

        result = session.query(FundingRate).filter_by(asset="SOL").first()
        assert result is not None
        assert result.funding_rate == 0.0001
        assert result.source == "gateio"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_db_models_v4.py -v`
Expected: FAIL with `ImportError: cannot import name 'AssetPrice15m'`

- [ ] **Step 3: Add AssetPrice15m and FundingRate models**

Add to `src/db/models.py` after the existing `AssetPriceDaily` class:

```python
class AssetPrice15m(Base):
    __tablename__ = "asset_prices_15m"
    __table_args__ = (UniqueConstraint("asset", "timestamp", name="uq_15m_asset_ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    quote_volume: Mapped[float] = mapped_column(Float, nullable=True)
    trades: Mapped[int] = mapped_column(Integer, nullable=True)
    taker_buy_base: Mapped[float] = mapped_column(Float, nullable=True)
    taker_buy_quote: Mapped[float] = mapped_column(Float, nullable=True)


class FundingRate(Base):
    __tablename__ = "funding_rates"
    __table_args__ = (UniqueConstraint("asset", "timestamp", "source", name="uq_funding_asset_ts_src"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    funding_rate: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_db_models_v4.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add services/python/src/db/models.py services/python/tests/test_db_models_v4.py
git commit -m "feat: add AssetPrice15m and FundingRate database models"
```

---

### Task 2: Backfill 15-Minute OHLCV from Binance Vision

**Files:**
- Create: `services/python/scripts/backfill_binance_15m.py`
- Create: `services/python/tests/test_backfill_binance_15m.py`

- [ ] **Step 1: Write failing test for download and parse logic**

```python
# tests/test_backfill_binance_15m.py
import csv
import io
import zipfile
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, ".")


def make_fake_zip(rows: list[list[str]]) -> bytes:
    """Create a fake Binance Vision zip with CSV data."""
    buf = io.BytesIO()
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    for row in rows:
        writer.writerow(row)
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("BTCUSDT-15m-2024-01.csv", csv_buf.getvalue())
    return buf.getvalue()


def test_parse_binance_csv():
    from scripts.backfill_binance_15m import parse_binance_csv

    raw = "1704067200000,42000.0,42100.0,41900.0,42050.0,123.45,1704068099999,5180000.0,1500,60.0,2520000.0,0\n"
    rows = parse_binance_csv(raw)
    assert len(rows) == 1
    assert rows[0]["open"] == 42000.0
    assert rows[0]["close"] == 42050.0
    assert rows[0]["trades"] == 1500
    assert rows[0]["taker_buy_base"] == 60.0


def test_generate_months():
    from scripts.backfill_binance_15m import generate_months

    months = generate_months(2024, 1, 2024, 3)
    assert months == ["2024-01", "2024-02", "2024-03"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_backfill_binance_15m.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement backfill_binance_15m.py**

```python
# scripts/backfill_binance_15m.py
"""Download 15-minute candles from Binance Vision for v4 assets.

Output: data/raw/15m/{SYMBOL}/{SYMBOL}-15m-{YYYY-MM}.csv
Source: https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/15m/
"""

import csv
import io
import logging
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    "C:/Users/togat/Desktop/AI-Finance/data/raw/15m",
))
START_YEAR = 2023
START_MONTH = 4  # April 2023 → 3 years to April 2026
WORKERS = 4

V4_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT",
    "XRPUSDT", "AVAXUSDT", "LINKUSDT",
]

CLEAN_HEADERS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


def generate_months(start_year: int, start_month: int,
                    end_year: int | None = None, end_month: int | None = None) -> list[str]:
    now = datetime.utcnow()
    ey = end_year or now.year
    em = end_month or now.month
    months = []
    y, m = start_year, start_month
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def parse_binance_csv(raw_text: str) -> list[dict]:
    reader = csv.reader(io.StringIO(raw_text))
    rows = []
    for row in reader:
        if len(row) < 12:
            continue
        rows.append({
            "open_time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "close_time": int(row[6]),
            "quote_volume": float(row[7]),
            "trades": int(row[8]),
            "taker_buy_base": float(row[9]),
            "taker_buy_quote": float(row[10]),
        })
    return rows


def download_one(symbol: str, month: str) -> tuple[str, str, int, str]:
    out_dir = OUTPUT_DIR / symbol
    out_file = out_dir / f"{symbol}-15m-{month}.csv"

    if out_file.exists() and out_file.stat().st_size > 100:
        return (symbol, month, 0, "skipped")

    url = f"{BASE_URL}/{symbol}/15m/{symbol}-15m-{month}.zip"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urlopen(req, timeout=30)
        zip_data = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                raw = f.read().decode("utf-8")

        parsed = parse_binance_csv(raw)
        if not parsed:
            return (symbol, month, 0, "empty")

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CLEAN_HEADERS)
            for row in parsed:
                writer.writerow([
                    row["open_time"], row["open"], row["high"], row["low"],
                    row["close"], row["volume"], row["close_time"],
                    row["quote_volume"], row["trades"], row["taker_buy_base"],
                    row["taker_buy_quote"], 0,
                ])

        return (symbol, month, len(parsed), "ok")

    except HTTPError as e:
        if e.code == 404:
            return (symbol, month, 0, "not_found")
        return (symbol, month, 0, f"http_{e.code}")
    except Exception as e:
        return (symbol, month, 0, f"error: {str(e)[:80]}")


def main():
    months = generate_months(START_YEAR, START_MONTH)
    logger.info("Binance Vision 15m Downloader (v4)")
    logger.info(f"Symbols: {V4_SYMBOLS}")
    logger.info(f"Months: {len(months)} ({months[0]} to {months[-1]})")
    logger.info(f"Output: {OUTPUT_DIR}")

    tasks = [(sym, month) for sym in V4_SYMBOLS for month in months]
    results = {"ok": 0, "skipped": 0, "not_found": 0, "error": 0}
    total_rows = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(download_one, s, m): (s, m) for s, m in tasks}
        done = 0
        total = len(futures)
        last_log = time.time()

        for future in as_completed(futures):
            sym, month, rows, status = future.result()
            done += 1
            total_rows += rows

            if status == "ok":
                results["ok"] += 1
            elif status == "skipped":
                results["skipped"] += 1
            elif status == "not_found":
                results["not_found"] += 1
            else:
                results["error"] += 1

            if time.time() - last_log > 5 or done == total:
                last_log = time.time()
                pct = done / total * 100
                logger.info(
                    f"  [{done}/{total}] {pct:.0f}% | "
                    f"ok={results['ok']} skip={results['skipped']} "
                    f"miss={results['not_found']} err={results['error']} | "
                    f"{total_rows:,} rows"
                )

    logger.info("=" * 60)
    logger.info(f"DONE! Total candle rows: {total_rows:,}")
    logger.info(f"Downloaded: {results['ok']}, Skipped: {results['skipped']}, "
                f"Not found: {results['not_found']}, Errors: {results['error']}")

    # Save manifest
    manifest = OUTPUT_DIR / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "months", "first_month", "last_month"])
        for sym in V4_SYMBOLS:
            sym_dir = OUTPUT_DIR / sym
            files = sorted(sym_dir.glob("*.csv")) if sym_dir.exists() else []
            if files:
                first = files[0].stem.split("-15m-")[1]
                last = files[-1].stem.split("-15m-")[1]
                writer.writerow([sym, len(files), first, last])
    logger.info(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_backfill_binance_15m.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add services/python/scripts/backfill_binance_15m.py services/python/tests/test_backfill_binance_15m.py
git commit -m "feat: add 15-minute OHLCV backfill from Binance Vision"
```

- [ ] **Step 6: Run the backfill script**

Run: `cd services/python && python scripts/backfill_binance_15m.py`
Expected: Downloads ~252 zip files (7 symbols x 36 months). ~735,000 total candle rows. Takes ~5-10 minutes.

- [ ] **Step 7: Verify downloaded data**

Run: `ls data/raw/15m/ && wc -l data/raw/15m/BTCUSDT/*.csv | tail -1`
Expected: 7 symbol directories, ~36 CSV files per symbol, ~2,880 rows per file (96 candles/day x 30 days).

- [ ] **Step 8: Commit data manifest**

```bash
git add data/raw/15m/manifest.csv
git commit -m "data: add 15m OHLCV manifest for v4 assets"
```

---

### Task 3: Upsert 15-Minute CSVs into PostgreSQL

**Files:**
- Create: `services/python/scripts/upsert_15m_to_pg.py`
- Create: `services/python/tests/test_upsert_15m.py`

- [ ] **Step 1: Write failing test for upsert logic**

```python
# tests/test_upsert_15m.py
import sys
sys.path.insert(0, ".")

from datetime import datetime, UTC
from sqlalchemy import create_engine, text
from src.db.models import Base, AssetPrice15m


def make_pg_engine():
    """Use the real local PostgreSQL for integration test."""
    url = "postgresql://postgres:MySQL100%25@localhost:5432/market"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return engine


def test_upsert_15m_batch():
    from scripts.upsert_15m_to_pg import upsert_15m_batch

    engine = make_pg_engine()

    # Clean slate
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM asset_prices_15m WHERE asset = 'TEST'"))

    rows = [
        {
            "asset": "TEST", "timestamp": datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
            "volume": 50.0, "quote_volume": 5000.0, "trades": 100,
            "taker_buy_base": 25.0, "taker_buy_quote": 2500.0,
        },
        {
            "asset": "TEST", "timestamp": datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
            "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5,
            "volume": 60.0, "quote_volume": 6000.0, "trades": 120,
            "taker_buy_base": 30.0, "taker_buy_quote": 3000.0,
        },
    ]

    count = upsert_15m_batch(engine, rows)
    assert count == 2

    # Upsert again — should not duplicate
    count2 = upsert_15m_batch(engine, rows)
    assert count2 == 2

    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM asset_prices_15m WHERE asset = 'TEST'")
        ).scalar()
        assert total == 2

    # Cleanup
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM asset_prices_15m WHERE asset = 'TEST'"))

    engine.dispose()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_upsert_15m.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement upsert_15m_to_pg.py**

```python
# scripts/upsert_15m_to_pg.py
"""Upsert 15m CSVs from data/raw/15m/ into asset_prices_15m table."""

import csv
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.db.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")
BATCH_SIZE = 5000

V4_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT"]


def parse_symbol(symbol_usdt: str) -> str:
    return symbol_usdt.replace("USDT", "")


def upsert_15m_batch(engine, rows: list[dict]) -> int:
    if not rows:
        return 0

    sql = text("""
        INSERT INTO asset_prices_15m
            (asset, timestamp, open, high, low, close, volume,
             quote_volume, trades, taker_buy_base, taker_buy_quote)
        VALUES
            (:asset, :timestamp, :open, :high, :low, :close, :volume,
             :quote_volume, :trades, :taker_buy_base, :taker_buy_quote)
        ON CONFLICT (asset, timestamp) DO UPDATE SET
            open = EXCLUDED.open, high = EXCLUDED.high,
            low = EXCLUDED.low, close = EXCLUDED.close,
            volume = EXCLUDED.volume, quote_volume = EXCLUDED.quote_volume,
            trades = EXCLUDED.trades, taker_buy_base = EXCLUDED.taker_buy_base,
            taker_buy_quote = EXCLUDED.taker_buy_quote
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)

    return len(rows)


def load_symbol_csvs(symbol_dir: Path) -> list[dict]:
    symbol_usdt = symbol_dir.name
    asset = parse_symbol(symbol_usdt)
    csv_files = sorted(symbol_dir.glob("*.csv"))
    rows = []

    for csv_file in csv_files:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts_ms = int(row["open_time"])
                    dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
                    rows.append({
                        "asset": asset,
                        "timestamp": dt,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                        "quote_volume": float(row.get("quote_volume", 0)),
                        "trades": int(row.get("trades", 0)),
                        "taker_buy_base": float(row.get("taker_buy_base", 0)),
                        "taker_buy_quote": float(row.get("taker_buy_quote", 0)),
                    })
                except (ValueError, KeyError, OSError):
                    continue

    return rows


def main():
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)

    logger.info(f"Upserting 15m candles from {RAW_DIR}")

    total_rows = 0
    start = time.time()

    for symbol_usdt in V4_SYMBOLS:
        sym_dir = RAW_DIR / symbol_usdt
        if not sym_dir.exists():
            logger.warning(f"  {symbol_usdt}: directory not found, skipping")
            continue

        rows = load_symbol_csvs(sym_dir)
        asset = parse_symbol(symbol_usdt)

        # Batch upsert
        upserted = 0
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            upserted += upsert_15m_batch(engine, batch)

        total_rows += upserted
        logger.info(f"  {asset}: {upserted:,} rows upserted")

    # Final DB counts
    with engine.connect() as conn:
        db_total = conn.execute(text("SELECT COUNT(*) FROM asset_prices_15m")).scalar()
        db_assets = conn.execute(text("SELECT COUNT(DISTINCT asset) FROM asset_prices_15m")).scalar()
        min_ts = conn.execute(text("SELECT MIN(timestamp) FROM asset_prices_15m")).scalar()
        max_ts = conn.execute(text("SELECT MAX(timestamp) FROM asset_prices_15m")).scalar()

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"DONE in {elapsed:.0f}s!")
    logger.info(f"  PostgreSQL: {db_total:,} rows, {db_assets} assets")
    logger.info(f"  Range: {min_ts} → {max_ts}")

    engine.dispose()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_upsert_15m.py -v`
Expected: PASS (1 test). Requires local PostgreSQL running.

- [ ] **Step 5: Commit**

```bash
git add services/python/scripts/upsert_15m_to_pg.py services/python/tests/test_upsert_15m.py
git commit -m "feat: add upsert script for 15m candles to PostgreSQL"
```

- [ ] **Step 6: Run the upsert**

Run: `cd services/python && python scripts/upsert_15m_to_pg.py`
Expected: ~735,000 rows upserted across 7 assets. Takes 1-3 minutes.

- [ ] **Step 7: Commit**

```bash
git commit --allow-empty -m "data: upsert 15m OHLCV into PostgreSQL (735K rows)"
```

---

### Task 4: Backfill Funding Rates from Gate.io

**Files:**
- Create: `services/python/scripts/backfill_funding_rates.py`
- Create: `services/python/tests/test_backfill_funding.py`

- [ ] **Step 1: Write failing test for Gate.io API parsing**

```python
# tests/test_backfill_funding.py
import sys
sys.path.insert(0, ".")

from datetime import datetime, UTC


SAMPLE_GATEIO_RESPONSE = [
    {"t": 1704067200, "r": "0.000100", "T": "1704067200"},
    {"t": 1704096000, "r": "-0.000050", "T": "1704096000"},
]


def test_parse_gateio_funding():
    from scripts.backfill_funding_rates import parse_gateio_funding

    rows = parse_gateio_funding("SOL", SAMPLE_GATEIO_RESPONSE)
    assert len(rows) == 2
    assert rows[0]["asset"] == "SOL"
    assert rows[0]["funding_rate"] == 0.0001
    assert rows[0]["source"] == "gateio"
    assert isinstance(rows[0]["timestamp"], datetime)


def test_parse_gateio_funding_empty():
    from scripts.backfill_funding_rates import parse_gateio_funding

    rows = parse_gateio_funding("SOL", [])
    assert rows == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_backfill_funding.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement backfill_funding_rates.py**

Gate.io API endpoint: `GET /api/v4/futures/usdt/funding_rate`
Params: `contract` (e.g. "SOL_USDT"), `limit` (max 1000), `from`/`to` (unix timestamps).
No API key needed for public endpoints.

```python
# scripts/backfill_funding_rates.py
"""Backfill funding rate history from Gate.io API.

Gate.io provides funding rates every 8 hours for perpetual contracts.
Endpoint: GET https://api.gateio.ws/api/v4/futures/usdt/funding_rate
No API key required.
"""

import json
import logging
import sys
import time
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.db.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
GATEIO_BASE = "https://api.gateio.ws/api/v4/futures/usdt/funding_rate"

# Gate.io contract names
V4_CONTRACTS = {
    "BTC": "BTC_USDT",
    "ETH": "ETH_USDT",
    "SOL": "SOL_USDT",
    "DOGE": "DOGE_USDT",
    "XRP": "XRP_USDT",
    "AVAX": "AVAX_USDT",
    "LINK": "LINK_USDT",
}

START_DATE = datetime(2023, 4, 1, tzinfo=UTC)
PAGE_LIMIT = 1000


def parse_gateio_funding(asset: str, data: list[dict]) -> list[dict]:
    rows = []
    for item in data:
        ts = int(item["t"])
        rate = float(item["r"])
        rows.append({
            "asset": asset,
            "timestamp": datetime.fromtimestamp(ts, tz=UTC),
            "funding_rate": rate,
            "source": "gateio",
        })
    return rows


def fetch_funding_page(contract: str, from_ts: int, to_ts: int) -> list[dict]:
    url = f"{GATEIO_BASE}?contract={contract}&from={from_ts}&to={to_ts}&limit={PAGE_LIMIT}"
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    resp = urlopen(req, timeout=15)
    return json.loads(resp.read().decode("utf-8"))


def upsert_funding_batch(engine, rows: list[dict]) -> int:
    if not rows:
        return 0

    sql = text("""
        INSERT INTO funding_rates (asset, timestamp, funding_rate, source)
        VALUES (:asset, :timestamp, :funding_rate, :source)
        ON CONFLICT (asset, timestamp, source) DO UPDATE SET
            funding_rate = EXCLUDED.funding_rate
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)

    return len(rows)


def backfill_asset(engine, asset: str, contract: str) -> int:
    now = datetime.now(UTC)
    cursor = START_DATE
    total = 0

    while cursor < now:
        window_end = min(cursor + timedelta(days=90), now)
        from_ts = int(cursor.timestamp())
        to_ts = int(window_end.timestamp())

        try:
            data = fetch_funding_page(contract, from_ts, to_ts)
            rows = parse_gateio_funding(asset, data)
            if rows:
                upsert_funding_batch(engine, rows)
                total += len(rows)
                logger.info(f"  {asset}: +{len(rows)} rates ({cursor.date()} → {window_end.date()})")
            else:
                logger.info(f"  {asset}: no data ({cursor.date()} → {window_end.date()})")
        except HTTPError as e:
            logger.error(f"  {asset}: HTTP {e.code} ({cursor.date()} → {window_end.date()})")
        except Exception as e:
            logger.error(f"  {asset}: error: {e}")

        cursor = window_end
        time.sleep(0.3)  # Rate limit

    return total


def main():
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)

    logger.info("Gate.io Funding Rate Backfill")
    logger.info(f"Assets: {list(V4_CONTRACTS.keys())}")
    logger.info(f"Start: {START_DATE.date()}")

    grand_total = 0
    for asset, contract in V4_CONTRACTS.items():
        logger.info(f"Backfilling {asset} ({contract})...")
        count = backfill_asset(engine, asset, contract)
        grand_total += count
        logger.info(f"  {asset}: {count} total rates")

    with engine.connect() as conn:
        db_total = conn.execute(text("SELECT COUNT(*) FROM funding_rates")).scalar()
        db_assets = conn.execute(text("SELECT COUNT(DISTINCT asset) FROM funding_rates")).scalar()

    logger.info("=" * 60)
    logger.info(f"DONE! PostgreSQL: {db_total:,} funding rates, {db_assets} assets")
    engine.dispose()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_backfill_funding.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add services/python/scripts/backfill_funding_rates.py services/python/tests/test_backfill_funding.py
git commit -m "feat: add Gate.io funding rate backfill script"
```

- [ ] **Step 6: Run the funding rate backfill**

Run: `cd services/python && python scripts/backfill_funding_rates.py`
Expected: ~7 assets x ~1,095 days / ~0.33 days per rate = ~23,000 funding rate records. Takes ~2-5 minutes due to API rate limiting.

- [ ] **Step 7: Commit**

```bash
git commit --allow-empty -m "data: backfill Gate.io funding rates (3 years, 7 assets)"
```

---

### Task 5: Data Validation Script

**Files:**
- Create: `services/python/scripts/validate_v4_data.py`

- [ ] **Step 1: Implement validation script**

```python
# scripts/validate_v4_data.py
"""Validate v4 data completeness and quality.

Checks:
1. Row counts per asset (15m candles)
2. Date range coverage
3. Gap detection (missing 15-min intervals)
4. Funding rate coverage
5. Basic data quality (no NaN, no zero prices)
"""

import logging
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
V4_ASSETS = ["BTC", "ETH", "SOL", "DOGE", "XRP", "AVAX", "LINK"]

# 3 years of 15min candles: ~105,000 per coin. Allow 90% as passing.
MIN_ROWS_15M = 90_000
MIN_FUNDING_ROWS = 2_000  # ~3 rates/day x 1095 days = ~3285, allow margin


def main():
    engine = create_engine(DB_URL)
    issues = []

    logger.info("=" * 60)
    logger.info("V4 Data Validation")
    logger.info("=" * 60)

    # 1. 15m candle counts
    logger.info("\n--- 15m Candle Counts ---")
    with engine.connect() as conn:
        for asset in V4_ASSETS:
            result = conn.execute(text(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) "
                "FROM asset_prices_15m WHERE asset = :asset"
            ), {"asset": asset}).fetchone()
            count, min_ts, max_ts = result
            status = "OK" if count >= MIN_ROWS_15M else "LOW"
            if status == "LOW":
                issues.append(f"{asset}: only {count:,} 15m rows (need {MIN_ROWS_15M:,})")
            logger.info(f"  {asset}: {count:,} rows [{min_ts} → {max_ts}] {status}")

    # 2. Gap detection (check for gaps > 1 hour in 15m data)
    logger.info("\n--- Gap Detection (>1h gaps) ---")
    with engine.connect() as conn:
        for asset in V4_ASSETS:
            gaps = conn.execute(text("""
                SELECT gap_start, gap_end, gap_minutes FROM (
                    SELECT
                        timestamp AS gap_start,
                        LEAD(timestamp) OVER (ORDER BY timestamp) AS gap_end,
                        EXTRACT(EPOCH FROM
                            LEAD(timestamp) OVER (ORDER BY timestamp) - timestamp
                        ) / 60 AS gap_minutes
                    FROM asset_prices_15m
                    WHERE asset = :asset
                ) sub
                WHERE gap_minutes > 60
                ORDER BY gap_minutes DESC
                LIMIT 5
            """), {"asset": asset}).fetchall()

            if gaps:
                logger.info(f"  {asset}: {len(gaps)} gaps >1h (showing top 5):")
                for g in gaps:
                    logger.info(f"    {g[0]} → {g[1]} ({g[2]:.0f} min)")
                    issues.append(f"{asset}: gap {g[2]:.0f}min at {g[0]}")
            else:
                logger.info(f"  {asset}: no gaps >1h ✓")

    # 3. Funding rate counts
    logger.info("\n--- Funding Rate Counts ---")
    with engine.connect() as conn:
        for asset in V4_ASSETS:
            result = conn.execute(text(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) "
                "FROM funding_rates WHERE asset = :asset"
            ), {"asset": asset}).fetchone()
            count, min_ts, max_ts = result
            status = "OK" if count >= MIN_FUNDING_ROWS else "LOW"
            if status == "LOW":
                issues.append(f"{asset}: only {count:,} funding rates (need {MIN_FUNDING_ROWS:,})")
            logger.info(f"  {asset}: {count:,} rates [{min_ts} → {max_ts}] {status}")

    # 4. Data quality
    logger.info("\n--- Data Quality ---")
    with engine.connect() as conn:
        zero_prices = conn.execute(text(
            "SELECT COUNT(*) FROM asset_prices_15m WHERE close = 0 OR open = 0"
        )).scalar()
        null_prices = conn.execute(text(
            "SELECT COUNT(*) FROM asset_prices_15m WHERE close IS NULL"
        )).scalar()
        logger.info(f"  Zero prices: {zero_prices}")
        logger.info(f"  Null prices: {null_prices}")
        if zero_prices > 0:
            issues.append(f"{zero_prices} rows with zero prices")
        if null_prices > 0:
            issues.append(f"{null_prices} rows with null prices")

    # Summary
    logger.info("\n" + "=" * 60)
    if issues:
        logger.warning(f"ISSUES FOUND ({len(issues)}):")
        for issue in issues:
            logger.warning(f"  ⚠ {issue}")
    else:
        logger.info("ALL CHECKS PASSED ✓")

    engine.dispose()
    return len(issues) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
```

- [ ] **Step 2: Commit**

```bash
git add services/python/scripts/validate_v4_data.py
git commit -m "feat: add v4 data validation script"
```

- [ ] **Step 3: Run validation**

Run: `cd services/python && python scripts/validate_v4_data.py`
Expected: Shows counts per asset, gaps, and quality checks. May show issues to investigate.

---

### Task 6: Fix Any Data Issues

This is a catch-all task after validation.

- [ ] **Step 1: Review validation output**

Check for: missing symbols, large gaps, low row counts, zero prices.

- [ ] **Step 2: Fix issues**

Common fixes:
- Missing months: re-run backfill script (it skips already-downloaded files)
- Gate.io API errors: increase timeout, retry with smaller date windows
- Zero prices: delete bad rows and re-download that month

- [ ] **Step 3: Re-run validation until clean**

Run: `cd services/python && python scripts/validate_v4_data.py`
Expected: ALL CHECKS PASSED

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "data: v4 data pipeline complete — 15m OHLCV + funding rates for 7 assets"
```
