yes# Plan 1: Infrastructure + Data Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the Docker-based infrastructure, PostgreSQL data warehouse, CSV data lake, and hourly data ingestion from Binance and CoinGecko with automated scheduling.

**Architecture:** Three Docker containers (Python background service, PostgreSQL, placeholder Next.js) orchestrated by Docker Compose. Python runs as a background scheduler/ML service (not an API server) — it fetches hourly crypto data from Binance and market metadata from CoinGecko, stores raw CSVs in a shared volume (data lake), then cleans and loads into PostgreSQL (data warehouse). APScheduler handles hourly/daily/weekly job scheduling. Next.js handles all API routes.

**Tech Stack:** Python 3.12+, SQLAlchemy, APScheduler, ccxt (Binance), pycoingecko, pandas, PostgreSQL 16, Docker, Docker Compose, pytest, ruff, mypy

---

## File Structure

```
AI-Finance/
├── docker-compose.yml                    # Orchestrates all 3 containers
├── .env.example                          # Environment variable template
├── .gitignore                            # Git ignore rules
├── .github/
│   └── workflows/
│       └── python-ci.yml                 # GitHub Actions: pytest, ruff, mypy
│
├── services/
│   ├── python/
│   │   ├── Dockerfile                    # Python container build
│   │   ├── pyproject.toml                # Python dependencies + tool config
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                   # Scheduler entry point + DB init
│   │   │   ├── config.py                 # Settings loaded from env vars
│   │   │   ├── db/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── engine.py             # SQLAlchemy engine + session factory
│   │   │   │   ├── models.py             # All SQLAlchemy ORM models
│   │   │   │   └── migrations.py         # Table creation on startup
│   │   │   ├── data/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── binance_client.py     # Binance API wrapper (hourly OHLCV)
│   │   │   │   ├── coingecko_client.py   # CoinGecko API wrapper (market data)
│   │   │   │   ├── csv_storage.py        # Read/write CSV data lake
│   │   │   │   └── etl.py               # CSV → PostgreSQL ETL pipeline
│   │   │   ├── scheduler/
│   │   │   │   ├── __init__.py
│   │   │   │   └── jobs.py              # Scheduled job definitions
│   │   └── tests/
│   │       ├── __init__.py
│   │       ├── conftest.py               # Shared fixtures (DB, test data)
│   │       ├── test_config.py
│   │       ├── test_binance_client.py
│   │       ├── test_coingecko_client.py
│   │       ├── test_csv_storage.py
│   │       ├── test_etl.py
│   │       ├── test_db_models.py
│   │       └── test_scheduler_jobs.py
│   │
│   └── nextjs/
│       └── Dockerfile                    # Placeholder Next.js container
│
├── data/
│   └── raw/                              # CSV data lake (Docker shared volume)
│       ├── hourly/
│       ├── daily/
│       ├── fundamentals/
│       └── index/
│
└── docs/
    └── superpowers/
        ├── specs/
        └── plans/
```

---

### Task 1: Project Scaffolding + Docker Compose

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `services/python/Dockerfile`
- Create: `services/python/pyproject.toml`
- Create: `services/python/src/__init__.py`
- Create: `services/nextjs/Dockerfile`

- [ ] **Step 1: Initialize git repository**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git init
```

- [ ] **Step 2: Create .gitignore**

Create `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
dist/
build/

# Environment
.env

# Data lake (large files, don't commit)
data/raw/hourly/
data/raw/daily/
data/raw/fundamentals/
data/raw/index/

# ML models
models/

# IDE
.vscode/
.idea/

# Docker
*.log

# OS
.DS_Store
Thumbs.db

# Next.js
services/nextjs/node_modules/
services/nextjs/.next/
```

- [ ] **Step 3: Create .env.example**

Create `.env.example`:

```env
# PostgreSQL
POSTGRES_USER=aifinance
POSTGRES_PASSWORD=aifinance_dev
POSTGRES_DB=aifinance
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Data Lake
DATA_LAKE_PATH=/app/data/raw

# Binance (no API key needed for public data)
BINANCE_RATE_LIMIT_MS=100

# CoinGecko
COINGECKO_API_KEY=

# Scheduler
SCHEDULER_ENABLED=true

# App
LOG_LEVEL=INFO
POSITION_SIZE_IDR=1000000
```

- [ ] **Step 4: Create pyproject.toml**

Create `services/python/pyproject.toml`:

```toml
[project]
name = "ai-finance-python"
version = "0.1.0"
description = "Crypto trading ML engine and data pipeline"
requires-python = ">=3.12"
dependencies = [
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "ccxt>=4.0.0",
    "pycoingecko>=3.1.0",
    "apscheduler>=3.10.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.7.0",
    "mypy>=1.13.0",
    "pytest-cov>=6.0.0",
]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 5: Create Python Dockerfile**

Create `services/python/Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY src/ src/
COPY tests/ tests/

CMD ["python", "-m", "src.main"]
```

- [ ] **Step 6: Create placeholder Next.js Dockerfile**

Create `services/nextjs/Dockerfile`:

```dockerfile
FROM node:20-alpine

WORKDIR /app

# Placeholder — will be built out in Plan 3 (Dashboard)
RUN echo '{"name":"ai-finance-dashboard","version":"0.1.0","scripts":{"dev":"echo Dashboard placeholder - see Plan 3"}}' > package.json

EXPOSE 3000

CMD ["echo", "Next.js dashboard placeholder - implement in Plan 3"]
```

- [ ] **Step 7: Create docker-compose.yml**

Create `docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    env_file: .env
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-aifinance}"]
      interval: 5s
      timeout: 5s
      retries: 5

  python:
    build:
      context: ./services/python
      dockerfile: Dockerfile
    env_file: .env
    volumes:
      - ./services/python/src:/app/src
      - ./services/python/tests:/app/tests
      - ./data/raw:/app/data/raw
    depends_on:
      postgres:
        condition: service_healthy

  nextjs:
    build:
      context: ./services/nextjs
      dockerfile: Dockerfile
    ports:
      - "3000:3000"

volumes:
  postgres_data:
```

- [ ] **Step 8: Create src/__init__.py**

Create `services/python/src/__init__.py`:

```python
```

- [ ] **Step 9: Create data lake directories**

```bash
mkdir -p data/raw/hourly data/raw/daily data/raw/fundamentals data/raw/index
```

- [ ] **Step 10: Commit**

```bash
git add .gitignore .env.example docker-compose.yml \
  services/python/Dockerfile services/python/pyproject.toml services/python/src/__init__.py \
  services/nextjs/Dockerfile
git commit -m "chore: scaffold project with Docker Compose, Python service, PostgreSQL"
```

---

### Task 2: Configuration + App Entry Point

**Files:**
- Create: `services/python/src/config.py`
- Create: `services/python/src/main.py`
- Create: `services/python/tests/__init__.py`
- Create: `services/python/tests/test_config.py`

- [ ] **Step 1: Write the failing test for config**

Create `services/python/tests/__init__.py`:

```python
```

Create `services/python/tests/test_config.py`:

```python
import os
from unittest.mock import patch


def test_settings_loads_defaults():
    env = {
        "POSTGRES_USER": "testuser",
        "POSTGRES_PASSWORD": "testpass",
        "POSTGRES_DB": "testdb",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "DATA_LAKE_PATH": "/tmp/data",
    }
    with patch.dict(os.environ, env, clear=False):
        from src.config import Settings

        settings = Settings()
        assert settings.POSTGRES_USER == "testuser"
        assert settings.POSTGRES_DB == "testdb"
        assert settings.DATA_LAKE_PATH == "/tmp/data"
        assert settings.POSITION_SIZE_IDR == 1_000_000
        assert settings.SCHEDULER_ENABLED is True


def test_settings_database_url():
    env = {
        "POSTGRES_USER": "u",
        "POSTGRES_PASSWORD": "p",
        "POSTGRES_DB": "d",
        "POSTGRES_HOST": "h",
        "POSTGRES_PORT": "5432",
        "DATA_LAKE_PATH": "/tmp/data",
    }
    with patch.dict(os.environ, env, clear=False):
        from src.config import Settings

        settings = Settings()
        assert settings.database_url == "postgresql://u:p@h:5432/d"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd services/python && pip install -e ".[dev]" && pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: Implement config.py**

Create `services/python/src/config.py`:

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    # Data Lake
    DATA_LAKE_PATH: str = "/app/data/raw"

    # Binance
    BINANCE_RATE_LIMIT_MS: int = 100

    # CoinGecko
    COINGECKO_API_KEY: str = ""

    # Scheduler
    SCHEDULER_ENABLED: bool = True

    # App
    LOG_LEVEL: str = "INFO"
    POSITION_SIZE_IDR: int = 1_000_000

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Top 20 crypto by market cap — refreshed weekly by CoinGecko job
    DEFAULT_ASSETS: list[str] = [
        "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX",
        "DOT", "LINK", "UNI", "ATOM", "LTC", "APT", "NEAR",
        "OP", "ARB", "FIL", "INJ", "MATIC",
    ]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Implement main.py (scheduler entry point)**

Create `services/python/src/main.py`:

```python
import logging
import signal
import sys
import threading

from src.config import Settings

logger = logging.getLogger(__name__)


def get_settings() -> Settings:
    return Settings()


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.LOG_LEVEL)
    logger.info("AI-Finance Python service starting...")

    # Keep the process running until interrupted
    shutdown_event = threading.Event()

    def handle_signal(signum, frame):
        logger.info("Received shutdown signal")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("AI-Finance Python service running (scheduler will be wired up in Task 9)")
    shutdown_event.wait()
    logger.info("AI-Finance Python service shutting down...")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add services/python/src/config.py services/python/src/main.py \
  services/python/tests/__init__.py services/python/tests/test_config.py
git commit -m "feat: add config module and scheduler entry point shell"
```

---

### Task 3: Database Models + Engine

**Files:**
- Create: `services/python/src/db/__init__.py`
- Create: `services/python/src/db/engine.py`
- Create: `services/python/src/db/models.py`
- Create: `services/python/src/db/migrations.py`
- Create: `services/python/tests/conftest.py`
- Create: `services/python/tests/test_db_models.py`

- [ ] **Step 1: Write the failing test for DB models**

Create `services/python/tests/conftest.py`:

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Base


@pytest.fixture
def db_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    session_factory = sessionmaker(bind=db_engine)
    session = session_factory()
    yield session
    session.close()
```

Create `services/python/tests/test_db_models.py`:

```python
from datetime import datetime, timezone

from src.db.models import AssetPriceHourly, AssetPriceDaily, AssetFundamental, Signal, Portfolio, AuditLog


def test_insert_hourly_price(db_session):
    price = AssetPriceHourly(
        asset="BTC",
        timestamp=datetime(2026, 3, 30, 10, 0, tzinfo=timezone.utc),
        open=70000.0,
        high=70500.0,
        low=69800.0,
        close=70200.0,
        volume=1500.5,
    )
    db_session.add(price)
    db_session.commit()

    result = db_session.query(AssetPriceHourly).first()
    assert result is not None
    assert result.asset == "BTC"
    assert result.close == 70200.0


def test_insert_daily_price(db_session):
    price = AssetPriceDaily(
        asset="ETH",
        date=datetime(2026, 3, 30, tzinfo=timezone.utc),
        open=3500.0,
        high=3600.0,
        low=3450.0,
        close=3550.0,
        volume=25000.0,
    )
    db_session.add(price)
    db_session.commit()

    result = db_session.query(AssetPriceDaily).first()
    assert result is not None
    assert result.asset == "ETH"


def test_insert_fundamental(db_session):
    fund = AssetFundamental(
        asset="BTC",
        fetched_at=datetime(2026, 3, 30, tzinfo=timezone.utc),
        market_cap=1_400_000_000_000.0,
        market_cap_rank=1,
        total_volume_24h=35_000_000_000.0,
        circulating_supply=19_800_000.0,
        category="Layer 1",
    )
    db_session.add(fund)
    db_session.commit()

    result = db_session.query(AssetFundamental).first()
    assert result is not None
    assert result.market_cap_rank == 1


def test_insert_signal(db_session):
    signal = Signal(
        asset="ETH",
        action="BUY",
        confidence=0.84,
        suggested_hold_days=18,
        stop_loss_pct=-8.0,
        expected_return_pct=12.0,
        model_agreement="3/3",
        created_at=datetime(2026, 3, 30, tzinfo=timezone.utc),
    )
    db_session.add(signal)
    db_session.commit()

    result = db_session.query(Signal).first()
    assert result is not None
    assert result.action == "BUY"
    assert result.confidence == 0.84


def test_insert_portfolio(db_session):
    position = Portfolio(
        asset="SOL",
        action="BUY",
        entry_price=180.0,
        entry_amount_idr=1_000_000,
        quantity=5555.56,
        stop_loss_price=165.6,
        status="open",
        opened_at=datetime(2026, 3, 30, tzinfo=timezone.utc),
    )
    db_session.add(position)
    db_session.commit()

    result = db_session.query(Portfolio).first()
    assert result is not None
    assert result.status == "open"
    assert result.entry_amount_idr == 1_000_000


def test_insert_audit_log(db_session):
    log = AuditLog(
        event_type="signal_generated",
        asset="ETH",
        details='{"action": "BUY", "confidence": 0.84}',
        created_at=datetime(2026, 3, 30, tzinfo=timezone.utc),
    )
    db_session.add(log)
    db_session.commit()

    result = db_session.query(AuditLog).first()
    assert result is not None
    assert result.event_type == "signal_generated"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_db_models.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.db'`

- [ ] **Step 3: Implement DB models**

Create `services/python/src/db/__init__.py`:

```python
```

Create `services/python/src/db/models.py`:

```python
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AssetPriceHourly(Base):
    __tablename__ = "asset_prices_hourly"
    __table_args__ = (UniqueConstraint("asset", "timestamp", name="uq_hourly_asset_ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)


class AssetPriceDaily(Base):
    __tablename__ = "asset_prices_daily"
    __table_args__ = (UniqueConstraint("asset", "date", name="uq_daily_asset_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)


class AssetFundamental(Base):
    __tablename__ = "asset_fundamentals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    market_cap: Mapped[float] = mapped_column(Float, nullable=True)
    market_cap_rank: Mapped[int] = mapped_column(Integer, nullable=True)
    total_volume_24h: Mapped[float] = mapped_column(Float, nullable=True)
    circulating_supply: Mapped[float] = mapped_column(Float, nullable=True)
    category: Mapped[str] = mapped_column(String(100), nullable=True)


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(10), nullable=False)  # BUY, SELL, HOLD, EXIT
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    suggested_hold_days: Mapped[int] = mapped_column(Integer, nullable=False)
    stop_loss_pct: Mapped[float] = mapped_column(Float, nullable=False)
    expected_return_pct: Mapped[float] = mapped_column(Float, nullable=False)
    model_agreement: Mapped[str] = mapped_column(String(10), nullable=False)
    acknowledged: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Portfolio(Base):
    __tablename__ = "portfolio"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    entry_amount_idr: Mapped[int] = mapped_column(Integer, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=True)
    exit_amount_idr: Mapped[int] = mapped_column(Integer, nullable=True)
    pnl_idr: Mapped[int] = mapped_column(Integer, nullable=True)
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(10), nullable=False)  # open, closed, stopped
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    signal_id: Mapped[int] = mapped_column(Integer, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    asset: Mapped[str] = mapped_column(String(20), nullable=True)
    details: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_db_models.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Implement engine.py and migrations.py**

Create `services/python/src/db/engine.py`:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import Settings


def create_db_engine(settings: Settings):
    return create_engine(settings.database_url, pool_pre_ping=True, pool_size=5)


def create_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine)
```

Create `services/python/src/db/migrations.py`:

```python
import logging

from sqlalchemy import Engine

from src.db.models import Base

logger = logging.getLogger(__name__)


def run_migrations(engine: Engine) -> None:
    logger.info("Running database migrations...")
    Base.metadata.create_all(engine)
    logger.info("Database migrations complete.")
```

- [ ] **Step 6: Commit**

```bash
git add services/python/src/db/ services/python/tests/conftest.py services/python/tests/test_db_models.py
git commit -m "feat: add PostgreSQL ORM models for prices, signals, portfolio, audit log"
```

---

### Task 4: CSV Data Lake Storage

**Files:**
- Create: `services/python/src/data/__init__.py`
- Create: `services/python/src/data/csv_storage.py`
- Create: `services/python/tests/test_csv_storage.py`

- [ ] **Step 1: Write the failing test**

Create `services/python/tests/test_csv_storage.py`:

```python
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data.csv_storage import CsvStorage


@pytest.fixture
def csv_storage(tmp_path):
    return CsvStorage(base_path=str(tmp_path))


def test_save_hourly_creates_file(csv_storage):
    rows = [
        {
            "timestamp": "2026-03-30T10:00:00+00:00",
            "open": 70000.0,
            "high": 70500.0,
            "low": 69800.0,
            "close": 70200.0,
            "volume": 1500.5,
        },
        {
            "timestamp": "2026-03-30T11:00:00+00:00",
            "open": 70200.0,
            "high": 70800.0,
            "low": 70100.0,
            "close": 70600.0,
            "volume": 1200.3,
        },
    ]
    csv_storage.save_hourly("BTC", "2026-03-30", rows)

    file_path = Path(csv_storage.base_path) / "hourly" / "BTC" / "2026-03-30.csv"
    assert file_path.exists()

    df = pd.read_csv(file_path)
    assert len(df) == 2
    assert df.iloc[0]["close"] == 70200.0


def test_save_hourly_appends_to_existing(csv_storage):
    row1 = [{"timestamp": "2026-03-30T10:00:00+00:00", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0}]
    row2 = [{"timestamp": "2026-03-30T11:00:00+00:00", "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "volume": 200.0}]

    csv_storage.save_hourly("BTC", "2026-03-30", row1)
    csv_storage.save_hourly("BTC", "2026-03-30", row2)

    file_path = Path(csv_storage.base_path) / "hourly" / "BTC" / "2026-03-30.csv"
    df = pd.read_csv(file_path)
    assert len(df) == 2


def test_save_daily_creates_file(csv_storage):
    rows = [
        {
            "date": "2026-03-30",
            "open": 70000.0,
            "high": 71000.0,
            "low": 69000.0,
            "close": 70500.0,
            "volume": 35000.0,
        }
    ]
    csv_storage.save_daily("BTC", "2026-03-30", rows)

    file_path = Path(csv_storage.base_path) / "daily" / "BTC" / "2026-03-30.csv"
    assert file_path.exists()


def test_save_fundamentals_creates_file(csv_storage):
    data = {
        "fetched_at": "2026-03-30T00:00:00+00:00",
        "market_cap": 1_400_000_000_000.0,
        "market_cap_rank": 1,
        "total_volume_24h": 35_000_000_000.0,
        "circulating_supply": 19_800_000.0,
        "category": "Layer 1",
    }
    csv_storage.save_fundamentals("BTC", "2026-03-30", data)

    file_path = Path(csv_storage.base_path) / "fundamentals" / "BTC" / "2026-03-30.csv"
    assert file_path.exists()


def test_read_hourly_returns_dataframe(csv_storage):
    rows = [{"timestamp": "2026-03-30T10:00:00+00:00", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0}]
    csv_storage.save_hourly("ETH", "2026-03-30", rows)

    df = csv_storage.read_hourly("ETH", "2026-03-30")
    assert len(df) == 1
    assert df.iloc[0]["close"] == 1.5


def test_read_hourly_missing_returns_empty(csv_storage):
    df = csv_storage.read_hourly("ETH", "2026-01-01")
    assert len(df) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_csv_storage.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data'`

- [ ] **Step 3: Implement csv_storage.py**

Create `services/python/src/data/__init__.py`:

```python
```

Create `services/python/src/data/csv_storage.py`:

```python
from pathlib import Path

import pandas as pd


class CsvStorage:
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path

    def _ensure_dir(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def save_hourly(self, asset: str, date: str, rows: list[dict]) -> None:
        file_path = Path(self.base_path) / "hourly" / asset / f"{date}.csv"
        self._ensure_dir(file_path)

        new_df = pd.DataFrame(rows)

        if file_path.exists():
            existing_df = pd.read_csv(file_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
            combined.to_csv(file_path, index=False)
        else:
            new_df.to_csv(file_path, index=False)

    def save_daily(self, asset: str, date: str, rows: list[dict]) -> None:
        file_path = Path(self.base_path) / "daily" / asset / f"{date}.csv"
        self._ensure_dir(file_path)
        pd.DataFrame(rows).to_csv(file_path, index=False)

    def save_fundamentals(self, asset: str, date: str, data: dict) -> None:
        file_path = Path(self.base_path) / "fundamentals" / asset / f"{date}.csv"
        self._ensure_dir(file_path)
        pd.DataFrame([data]).to_csv(file_path, index=False)

    def read_hourly(self, asset: str, date: str) -> pd.DataFrame:
        file_path = Path(self.base_path) / "hourly" / asset / f"{date}.csv"
        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_csv(file_path)

    def read_daily(self, asset: str, date: str) -> pd.DataFrame:
        file_path = Path(self.base_path) / "daily" / asset / f"{date}.csv"
        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_csv(file_path)

    def read_fundamentals(self, asset: str, date: str) -> pd.DataFrame:
        file_path = Path(self.base_path) / "fundamentals" / asset / f"{date}.csv"
        if not file_path.exists():
            return pd.DataFrame()
        return pd.read_csv(file_path)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_csv_storage.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add services/python/src/data/__init__.py services/python/src/data/csv_storage.py \
  services/python/tests/test_csv_storage.py
git commit -m "feat: add CSV data lake storage with hourly, daily, fundamentals support"
```

---

### Task 5: Binance Client

**Files:**
- Create: `services/python/src/data/binance_client.py`
- Create: `services/python/tests/test_binance_client.py`

- [ ] **Step 1: Write the failing test**

Create `services/python/tests/test_binance_client.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.data.binance_client import BinanceClient


@pytest.fixture
def client():
    return BinanceClient(rate_limit_ms=0)


def test_symbol_for_asset(client):
    assert client.symbol_for_asset("BTC") == "BTC/USDT"
    assert client.symbol_for_asset("ETH") == "ETH/USDT"


def test_parse_ohlcv_row(client):
    # ccxt returns: [timestamp_ms, open, high, low, close, volume]
    raw = [1711785600000, 70000.0, 70500.0, 69800.0, 70200.0, 1500.5]
    parsed = client.parse_ohlcv_row(raw)

    assert parsed["open"] == 70000.0
    assert parsed["high"] == 70500.0
    assert parsed["low"] == 69800.0
    assert parsed["close"] == 70200.0
    assert parsed["volume"] == 1500.5
    assert "timestamp" in parsed


@patch("src.data.binance_client.ccxt.binance")
def test_fetch_hourly_ohlcv(mock_binance_cls, client):
    mock_exchange = MagicMock()
    mock_binance_cls.return_value = mock_exchange
    mock_exchange.fetch_ohlcv.return_value = [
        [1711785600000, 70000.0, 70500.0, 69800.0, 70200.0, 1500.5],
        [1711789200000, 70200.0, 70800.0, 70100.0, 70600.0, 1200.3],
    ]

    new_client = BinanceClient(rate_limit_ms=0)
    rows = new_client.fetch_hourly_ohlcv("BTC", limit=2)

    assert len(rows) == 2
    assert rows[0]["close"] == 70200.0
    assert rows[1]["close"] == 70600.0
    mock_exchange.fetch_ohlcv.assert_called_once_with("BTC/USDT", "1h", limit=2)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_binance_client.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.binance_client'`

- [ ] **Step 3: Implement binance_client.py**

Create `services/python/src/data/binance_client.py`:

```python
import time
from datetime import datetime, timezone

import ccxt


class BinanceClient:
    def __init__(self, rate_limit_ms: int = 100) -> None:
        self.exchange = ccxt.binance({"enableRateLimit": True})
        self.rate_limit_ms = rate_limit_ms

    def symbol_for_asset(self, asset: str) -> str:
        return f"{asset}/USDT"

    def parse_ohlcv_row(self, raw: list) -> dict:
        timestamp_ms, open_price, high, low, close, volume = raw
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        return {
            "timestamp": dt.isoformat(),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }

    def fetch_hourly_ohlcv(self, asset: str, limit: int = 24) -> list[dict]:
        symbol = self.symbol_for_asset(asset)
        raw_data = self.exchange.fetch_ohlcv(symbol, "1h", limit=limit)

        if self.rate_limit_ms > 0:
            time.sleep(self.rate_limit_ms / 1000)

        return [self.parse_ohlcv_row(row) for row in raw_data]

    def fetch_daily_ohlcv(self, asset: str, limit: int = 1) -> list[dict]:
        symbol = self.symbol_for_asset(asset)
        raw_data = self.exchange.fetch_ohlcv(symbol, "1d", limit=limit)

        if self.rate_limit_ms > 0:
            time.sleep(self.rate_limit_ms / 1000)

        return [self.parse_ohlcv_row(row) for row in raw_data]

    def fetch_historical_hourly(
        self, asset: str, since: datetime, limit: int = 500
    ) -> list[dict]:
        symbol = self.symbol_for_asset(asset)
        since_ms = int(since.timestamp() * 1000)
        raw_data = self.exchange.fetch_ohlcv(symbol, "1h", since=since_ms, limit=limit)

        if self.rate_limit_ms > 0:
            time.sleep(self.rate_limit_ms / 1000)

        return [self.parse_ohlcv_row(row) for row in raw_data]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_binance_client.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add services/python/src/data/binance_client.py services/python/tests/test_binance_client.py
git commit -m "feat: add Binance client for hourly/daily OHLCV fetching via ccxt"
```

---

### Task 6: CoinGecko Client

**Files:**
- Create: `services/python/src/data/coingecko_client.py`
- Create: `services/python/tests/test_coingecko_client.py`

- [ ] **Step 1: Write the failing test**

Create `services/python/tests/test_coingecko_client.py`:

```python
from unittest.mock import MagicMock, patch

import pytest

from src.data.coingecko_client import CoinGeckoClient


@pytest.fixture
def client():
    return CoinGeckoClient(api_key="")


MOCK_MARKET_DATA = [
    {
        "id": "bitcoin",
        "symbol": "btc",
        "name": "Bitcoin",
        "market_cap": 1_400_000_000_000,
        "market_cap_rank": 1,
        "total_volume": 35_000_000_000,
        "circulating_supply": 19_800_000,
    },
    {
        "id": "ethereum",
        "symbol": "eth",
        "name": "Ethereum",
        "market_cap": 400_000_000_000,
        "market_cap_rank": 2,
        "total_volume": 15_000_000_000,
        "circulating_supply": 120_000_000,
    },
]


def test_asset_to_coingecko_id(client):
    assert client.asset_to_id("BTC") == "bitcoin"
    assert client.asset_to_id("ETH") == "ethereum"
    assert client.asset_to_id("SOL") == "solana"


def test_parse_market_data(client):
    parsed = client.parse_market_data(MOCK_MARKET_DATA[0])

    assert parsed["asset"] == "BTC"
    assert parsed["market_cap"] == 1_400_000_000_000
    assert parsed["market_cap_rank"] == 1
    assert parsed["total_volume_24h"] == 35_000_000_000
    assert parsed["circulating_supply"] == 19_800_000


@patch("src.data.coingecko_client.CoinGeckoAPI")
def test_fetch_top_n_assets(mock_cg_cls, client):
    mock_cg = MagicMock()
    mock_cg_cls.return_value = mock_cg
    mock_cg.get_coins_markets.return_value = MOCK_MARKET_DATA

    new_client = CoinGeckoClient(api_key="")
    assets = new_client.fetch_top_n_assets(n=2)

    assert len(assets) == 2
    assert assets[0] == "BTC"
    assert assets[1] == "ETH"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_coingecko_client.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.coingecko_client'`

- [ ] **Step 3: Implement coingecko_client.py**

Create `services/python/src/data/coingecko_client.py`:

```python
from datetime import datetime, timezone

from pycoingecko import CoinGeckoAPI

SYMBOL_TO_ID: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "LTC": "litecoin",
    "APT": "aptos",
    "NEAR": "near",
    "OP": "optimism",
    "ARB": "arbitrum",
    "FIL": "filecoin",
    "INJ": "injective-protocol",
    "MATIC": "matic-network",
}

ID_TO_SYMBOL: dict[str, str] = {v: k for k, v in SYMBOL_TO_ID.items()}


class CoinGeckoClient:
    def __init__(self, api_key: str = "") -> None:
        self.cg = CoinGeckoAPI()

    def asset_to_id(self, asset: str) -> str:
        return SYMBOL_TO_ID.get(asset, asset.lower())

    def id_to_asset(self, cg_id: str) -> str:
        return ID_TO_SYMBOL.get(cg_id, cg_id.upper())

    def parse_market_data(self, raw: dict) -> dict:
        symbol = self.id_to_asset(raw["id"])
        return {
            "asset": symbol,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "market_cap": raw.get("market_cap"),
            "market_cap_rank": raw.get("market_cap_rank"),
            "total_volume_24h": raw.get("total_volume"),
            "circulating_supply": raw.get("circulating_supply"),
            "category": None,
        }

    def fetch_top_n_assets(self, n: int = 20) -> list[str]:
        markets = self.cg.get_coins_markets(
            vs_currency="usd", order="market_cap_desc", per_page=n, page=1
        )
        return [self.id_to_asset(coin["id"]) for coin in markets]

    def fetch_fundamentals(self, assets: list[str]) -> list[dict]:
        ids = [self.asset_to_id(a) for a in assets]
        ids_str = ",".join(ids)
        markets = self.cg.get_coins_markets(
            vs_currency="usd", ids=ids_str, order="market_cap_desc", per_page=len(assets), page=1
        )
        return [self.parse_market_data(coin) for coin in markets]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_coingecko_client.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add services/python/src/data/coingecko_client.py services/python/tests/test_coingecko_client.py
git commit -m "feat: add CoinGecko client for market data and top-N asset discovery"
```

---

### Task 7: ETL Pipeline (CSV → PostgreSQL)

**Files:**
- Create: `services/python/src/data/etl.py`
- Create: `services/python/tests/test_etl.py`

- [ ] **Step 1: Write the failing test**

Create `services/python/tests/test_etl.py`:

```python
from datetime import datetime, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data.csv_storage import CsvStorage
from src.data.etl import EtlPipeline
from src.db.models import AssetPriceDaily, AssetPriceHourly, Base


@pytest.fixture
def db_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    factory = sessionmaker(bind=db_engine)
    session = factory()
    yield session
    session.close()


@pytest.fixture
def csv_storage(tmp_path):
    return CsvStorage(base_path=str(tmp_path))


@pytest.fixture
def etl(csv_storage, db_session):
    return EtlPipeline(csv_storage=csv_storage, session=db_session)


def test_load_hourly_csv_to_db(etl, csv_storage, db_session):
    rows = [
        {"timestamp": "2026-03-30T10:00:00+00:00", "open": 70000.0, "high": 70500.0, "low": 69800.0, "close": 70200.0, "volume": 1500.5},
        {"timestamp": "2026-03-30T11:00:00+00:00", "open": 70200.0, "high": 70800.0, "low": 70100.0, "close": 70600.0, "volume": 1200.3},
    ]
    csv_storage.save_hourly("BTC", "2026-03-30", rows)

    loaded = etl.load_hourly("BTC", "2026-03-30")
    assert loaded == 2

    results = db_session.query(AssetPriceHourly).all()
    assert len(results) == 2
    assert results[0].asset == "BTC"
    assert results[0].close == 70200.0


def test_load_hourly_deduplicates(etl, csv_storage, db_session):
    rows = [
        {"timestamp": "2026-03-30T10:00:00+00:00", "open": 70000.0, "high": 70500.0, "low": 69800.0, "close": 70200.0, "volume": 1500.5},
    ]
    csv_storage.save_hourly("BTC", "2026-03-30", rows)

    etl.load_hourly("BTC", "2026-03-30")
    etl.load_hourly("BTC", "2026-03-30")

    results = db_session.query(AssetPriceHourly).all()
    assert len(results) == 1


def test_aggregate_hourly_to_daily(etl, csv_storage, db_session):
    rows = [
        {"timestamp": "2026-03-30T00:00:00+00:00", "open": 70000.0, "high": 70500.0, "low": 69800.0, "close": 70200.0, "volume": 1000.0},
        {"timestamp": "2026-03-30T01:00:00+00:00", "open": 70200.0, "high": 71000.0, "low": 69500.0, "close": 70800.0, "volume": 2000.0},
    ]
    csv_storage.save_hourly("BTC", "2026-03-30", rows)
    etl.load_hourly("BTC", "2026-03-30")

    etl.aggregate_daily("BTC", "2026-03-30")

    daily = db_session.query(AssetPriceDaily).first()
    assert daily is not None
    assert daily.open == 70000.0       # first candle open
    assert daily.high == 71000.0       # max high
    assert daily.low == 69500.0        # min low
    assert daily.close == 70800.0      # last candle close
    assert daily.volume == 3000.0      # sum volume
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_etl.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.etl'`

- [ ] **Step 3: Implement etl.py**

Create `services/python/src/data/etl.py`:

```python
import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy.orm import Session

from src.data.csv_storage import CsvStorage
from src.db.models import AssetPriceDaily, AssetPriceHourly

logger = logging.getLogger(__name__)


class EtlPipeline:
    def __init__(self, csv_storage: CsvStorage, session: Session) -> None:
        self.csv_storage = csv_storage
        self.session = session

    def load_hourly(self, asset: str, date: str) -> int:
        df = self.csv_storage.read_hourly(asset, date)
        if df.empty:
            return 0

        loaded = 0
        for _, row in df.iterrows():
            ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            exists = (
                self.session.query(AssetPriceHourly)
                .filter_by(asset=asset, timestamp=ts)
                .first()
            )
            if exists:
                continue

            price = AssetPriceHourly(
                asset=asset,
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            self.session.add(price)
            loaded += 1

        self.session.commit()
        logger.info(f"Loaded {loaded} hourly rows for {asset} on {date}")
        return loaded

    def aggregate_daily(self, asset: str, date: str) -> None:
        hourly_rows = (
            self.session.query(AssetPriceHourly)
            .filter(
                AssetPriceHourly.asset == asset,
                AssetPriceHourly.timestamp >= pd.Timestamp(f"{date}T00:00:00+00:00").to_pydatetime(),
                AssetPriceHourly.timestamp < pd.Timestamp(f"{date}T00:00:00+00:00").to_pydatetime() + pd.Timedelta(days=1),
            )
            .order_by(AssetPriceHourly.timestamp)
            .all()
        )

        if not hourly_rows:
            return

        date_dt = pd.Timestamp(f"{date}T00:00:00+00:00").to_pydatetime()

        daily = AssetPriceDaily(
            asset=asset,
            date=date_dt,
            open=hourly_rows[0].open,
            high=max(r.high for r in hourly_rows),
            low=min(r.low for r in hourly_rows),
            close=hourly_rows[-1].close,
            volume=sum(r.volume for r in hourly_rows),
        )

        existing = (
            self.session.query(AssetPriceDaily)
            .filter_by(asset=asset, date=date_dt)
            .first()
        )
        if existing:
            existing.open = daily.open
            existing.high = daily.high
            existing.low = daily.low
            existing.close = daily.close
            existing.volume = daily.volume
        else:
            self.session.add(daily)

        self.session.commit()
        logger.info(f"Aggregated daily for {asset} on {date}")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_etl.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add services/python/src/data/etl.py services/python/tests/test_etl.py
git commit -m "feat: add ETL pipeline for CSV-to-PostgreSQL with hourly-to-daily aggregation"
```

---

### Task 8: Scheduler Jobs

**Files:**
- Create: `services/python/src/scheduler/__init__.py`
- Create: `services/python/src/scheduler/jobs.py`
- Create: `services/python/tests/test_scheduler_jobs.py`

- [ ] **Step 1: Write the failing test**

Create `services/python/tests/test_scheduler_jobs.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.scheduler.jobs import JobRunner


@pytest.fixture
def mock_deps():
    return {
        "binance_client": MagicMock(),
        "coingecko_client": MagicMock(),
        "csv_storage": MagicMock(),
        "etl": MagicMock(),
        "assets": ["BTC", "ETH"],
    }


def test_hourly_job_fetches_and_stores(mock_deps):
    mock_deps["binance_client"].fetch_hourly_ohlcv.return_value = [
        {"timestamp": "2026-03-30T10:00:00+00:00", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0}
    ]

    runner = JobRunner(**mock_deps)
    runner.run_hourly_ingestion()

    assert mock_deps["binance_client"].fetch_hourly_ohlcv.call_count == 2  # BTC + ETH
    assert mock_deps["csv_storage"].save_hourly.call_count == 2
    assert mock_deps["etl"].load_hourly.call_count == 2


def test_daily_aggregation_job(mock_deps):
    runner = JobRunner(**mock_deps)
    runner.run_daily_aggregation()

    assert mock_deps["etl"].aggregate_daily.call_count == 2  # BTC + ETH


def test_weekly_fundamentals_job(mock_deps):
    mock_deps["coingecko_client"].fetch_fundamentals.return_value = [
        {"asset": "BTC", "market_cap": 1_400_000_000_000, "market_cap_rank": 1,
         "total_volume_24h": 35_000_000_000, "circulating_supply": 19_800_000,
         "category": None, "fetched_at": "2026-03-30T00:00:00+00:00"},
    ]
    mock_deps["coingecko_client"].fetch_top_n_assets.return_value = ["BTC", "ETH"]

    runner = JobRunner(**mock_deps)
    runner.run_weekly_fundamentals()

    mock_deps["coingecko_client"].fetch_top_n_assets.assert_called_once_with(n=20)
    mock_deps["coingecko_client"].fetch_fundamentals.assert_called_once()
    assert mock_deps["csv_storage"].save_fundamentals.call_count >= 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_scheduler_jobs.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.scheduler'`

- [ ] **Step 3: Implement scheduler jobs**

Create `services/python/src/scheduler/__init__.py`:

```python
```

Create `services/python/src/scheduler/jobs.py`:

```python
import logging
from datetime import datetime, timezone

from src.data.binance_client import BinanceClient
from src.data.coingecko_client import CoinGeckoClient
from src.data.csv_storage import CsvStorage
from src.data.etl import EtlPipeline

logger = logging.getLogger(__name__)


class JobRunner:
    def __init__(
        self,
        binance_client: BinanceClient,
        coingecko_client: CoinGeckoClient,
        csv_storage: CsvStorage,
        etl: EtlPipeline,
        assets: list[str],
    ) -> None:
        self.binance = binance_client
        self.coingecko = coingecko_client
        self.csv = csv_storage
        self.etl = etl
        self.assets = assets

    def run_hourly_ingestion(self) -> None:
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        logger.info(f"Starting hourly ingestion for {len(self.assets)} assets")

        for asset in self.assets:
            try:
                rows = self.binance.fetch_hourly_ohlcv(asset, limit=1)
                self.csv.save_hourly(asset, date_str, rows)
                self.etl.load_hourly(asset, date_str)
                logger.info(f"Hourly ingestion complete for {asset}")
            except Exception:
                logger.exception(f"Hourly ingestion failed for {asset}")

    def run_daily_aggregation(self) -> None:
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        logger.info(f"Starting daily aggregation for {len(self.assets)} assets")

        for asset in self.assets:
            try:
                self.etl.aggregate_daily(asset, date_str)
            except Exception:
                logger.exception(f"Daily aggregation failed for {asset}")

    def run_weekly_fundamentals(self) -> None:
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        logger.info("Starting weekly fundamentals refresh")

        try:
            top_assets = self.coingecko.fetch_top_n_assets(n=20)
            self.assets = top_assets
            logger.info(f"Updated asset list: {top_assets}")

            fundamentals = self.coingecko.fetch_fundamentals(self.assets)
            for fund in fundamentals:
                self.csv.save_fundamentals(fund["asset"], date_str, fund)
            logger.info(f"Saved fundamentals for {len(fundamentals)} assets")
        except Exception:
            logger.exception("Weekly fundamentals refresh failed")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_scheduler_jobs.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add services/python/src/scheduler/ services/python/tests/test_scheduler_jobs.py
git commit -m "feat: add scheduler jobs for hourly ingestion, daily aggregation, weekly fundamentals"
```

---

### Task 9: Wire Up main.py with Scheduler

**Files:**
- Modify: `services/python/src/main.py`

- [ ] **Step 1: Update main.py to initialize DB, start scheduler, and keep running**

Replace `services/python/src/main.py` with:

```python
import logging
import signal
import sys
import threading

from apscheduler.schedulers.background import BackgroundScheduler

from src.config import Settings
from src.data.binance_client import BinanceClient
from src.data.coingecko_client import CoinGeckoClient
from src.data.csv_storage import CsvStorage
from src.data.etl import EtlPipeline
from src.db.engine import create_db_engine, create_session_factory
from src.db.migrations import run_migrations
from src.scheduler.jobs import JobRunner

logger = logging.getLogger(__name__)


def get_settings() -> Settings:
    return Settings()


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.LOG_LEVEL)
    logger.info("AI-Finance Python service starting...")

    # Initialize database
    engine = create_db_engine(settings)
    run_migrations(engine)
    session_factory = create_session_factory(engine)
    session = session_factory()

    scheduler = None

    if settings.SCHEDULER_ENABLED:
        binance = BinanceClient(rate_limit_ms=settings.BINANCE_RATE_LIMIT_MS)
        coingecko = CoinGeckoClient(api_key=settings.COINGECKO_API_KEY)
        csv_storage = CsvStorage(base_path=settings.DATA_LAKE_PATH)
        etl = EtlPipeline(csv_storage=csv_storage, session=session)
        job_runner = JobRunner(
            binance_client=binance,
            coingecko_client=coingecko,
            csv_storage=csv_storage,
            etl=etl,
            assets=list(settings.DEFAULT_ASSETS),
        )

        scheduler = BackgroundScheduler()
        scheduler.add_job(job_runner.run_hourly_ingestion, "interval", hours=1, id="hourly_ingestion")
        scheduler.add_job(job_runner.run_daily_aggregation, "cron", hour=0, minute=30, id="daily_aggregation")
        scheduler.add_job(job_runner.run_weekly_fundamentals, "cron", day_of_week="sun", hour=1, id="weekly_fundamentals")
        scheduler.start()
        logger.info("Scheduler started with hourly, daily, and weekly jobs")

    # Keep the process running until interrupted
    shutdown_event = threading.Event()

    def handle_signal(signum, frame):
        logger.info("Received shutdown signal")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("AI-Finance Python service running...")
    shutdown_event.wait()

    # Cleanup
    if scheduler:
        scheduler.shutdown()
    session.close()
    engine.dispose()
    logger.info("AI-Finance Python service shut down.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests PASSED

- [ ] **Step 3: Commit**

```bash
git add services/python/src/main.py
git commit -m "feat: wire up scheduler and DB init in main.py entry point"
```

---

### Task 10: GitHub Actions CI + Auto-Merge Workflow

**Files:**
- Create: `.github/workflows/python-ci.yml`
- Create: `.github/workflows/auto-merge.yml`

- [ ] **Step 1: Create CI workflow**

Create `.github/workflows/python-ci.yml`:

```yaml
name: Python CI

on:
  push:
    branches: [main, 'feat/**', 'fix/**', 'chore/**']
    paths:
      - 'services/python/**'
  pull_request:
    branches: [main]
    paths:
      - 'services/python/**'

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: services/python

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint with ruff
        run: ruff check src/ tests/

      - name: Type check with mypy
        run: mypy src/ --ignore-missing-imports

      - name: Run tests
        run: pytest tests/ -v --tb=short
```

- [ ] **Step 2: Create auto-merge workflow**

Create `.github/workflows/auto-merge.yml`:

```yaml
name: Auto Merge to Main

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  auto-merge:
    runs-on: ubuntu-latest
    if: github.actor == github.repository_owner
    permissions:
      contents: write
      pull-requests: write
    steps:
      - name: Enable auto-merge
        uses: peter-evans/enable-pull-request-automerge@v3
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          merge-method: squash
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/python-ci.yml .github/workflows/auto-merge.yml
git commit -m "ci: add GitHub Actions CI + auto-merge workflow for branch-based development"
```

**Git workflow for all tasks:**
1. Create branch: `git checkout -b feat/task-N-description`
2. Make changes, commit to branch
3. Push: `git push -u origin feat/task-N-description`
4. Create PR: `gh pr create --title "..." --body "..."`
5. CI runs → if green, auto-merge squashes into main
6. Pull latest main: `git checkout main && git pull`

---

### Task 11: Historical Data Backfill Script

**Files:**
- Create: `services/python/src/data/backfill.py`
- Create: `services/python/tests/test_backfill.py`

This script fetches 2-3 years of historical hourly data for backtesting.

- [ ] **Step 1: Write the failing test**

Create `services/python/tests/test_backfill.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock, call

import pytest

from src.data.backfill import Backfiller


@pytest.fixture
def mock_deps():
    return {
        "binance_client": MagicMock(),
        "csv_storage": MagicMock(),
    }


def test_backfill_generates_date_range(mock_deps):
    backfiller = Backfiller(**mock_deps)
    dates = backfiller.generate_date_range("2026-03-28", "2026-03-30")
    assert dates == ["2026-03-28", "2026-03-29", "2026-03-30"]


def test_backfill_fetches_for_each_date(mock_deps):
    mock_deps["binance_client"].fetch_historical_hourly.return_value = [
        {"timestamp": "2026-03-28T00:00:00+00:00", "open": 1.0, "high": 2.0,
         "low": 0.5, "close": 1.5, "volume": 100.0}
    ]

    backfiller = Backfiller(**mock_deps)
    backfiller.backfill_asset("BTC", "2026-03-28", "2026-03-28")

    mock_deps["binance_client"].fetch_historical_hourly.assert_called_once()
    mock_deps["csv_storage"].save_hourly.assert_called_once_with(
        "BTC", "2026-03-28",
        [{"timestamp": "2026-03-28T00:00:00+00:00", "open": 1.0, "high": 2.0,
          "low": 0.5, "close": 1.5, "volume": 100.0}]
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_backfill.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.backfill'`

- [ ] **Step 3: Implement backfill.py**

Create `services/python/src/data/backfill.py`:

```python
import logging
import time
from datetime import datetime, timedelta, timezone

from src.data.binance_client import BinanceClient
from src.data.csv_storage import CsvStorage

logger = logging.getLogger(__name__)


class Backfiller:
    def __init__(self, binance_client: BinanceClient, csv_storage: CsvStorage) -> None:
        self.binance = binance_client
        self.csv = csv_storage

    def generate_date_range(self, start_date: str, end_date: str) -> list[str]:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates

    def backfill_asset(self, asset: str, start_date: str, end_date: str) -> int:
        dates = self.generate_date_range(start_date, end_date)
        total_saved = 0

        for date_str in dates:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                rows = self.binance.fetch_historical_hourly(asset, since=dt, limit=24)

                if rows:
                    # Group rows by date (in case they span midnight)
                    date_rows: dict[str, list[dict]] = {}
                    for row in rows:
                        row_date = row["timestamp"][:10]
                        date_rows.setdefault(row_date, []).append(row)

                    for rd, rd_rows in date_rows.items():
                        self.csv.save_hourly(asset, rd, rd_rows)
                        total_saved += len(rd_rows)

                logger.info(f"Backfilled {asset} for {date_str}: {len(rows)} candles")
                time.sleep(0.5)  # Rate limit between days
            except Exception:
                logger.exception(f"Backfill failed for {asset} on {date_str}")

        return total_saved

    def backfill_all(self, assets: list[str], start_date: str, end_date: str) -> dict[str, int]:
        results = {}
        for asset in assets:
            logger.info(f"Starting backfill for {asset} from {start_date} to {end_date}")
            count = self.backfill_asset(asset, start_date, end_date)
            results[asset] = count
            logger.info(f"Completed backfill for {asset}: {count} total candles")
        return results
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_backfill.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add services/python/src/data/backfill.py services/python/tests/test_backfill.py
git commit -m "feat: add historical data backfill script for 2-3 years of hourly candles"
```

---

### Task 12: Final Integration Test + Run All Tests

**Files:**
- No new files — run all existing tests

- [ ] **Step 1: Run full test suite**

```bash
cd services/python && pytest tests/ -v --tb=short
```

Expected: All tests PASSED (20+ tests)

- [ ] **Step 2: Run ruff linter**

```bash
ruff check src/ tests/
```

Expected: No errors

- [ ] **Step 3: Verify Docker Compose builds**

```bash
cd /c/Users/togat/Desktop/AI-Finance
cp .env.example .env
docker compose build
```

Expected: All 3 images build successfully

- [ ] **Step 4: Verify Docker Compose starts and scheduler runs**

```bash
docker compose up -d
# Check that the python container is running and scheduler started
sleep 10
docker compose logs python | grep "Scheduler started"
```

Expected: Log output shows "Scheduler started with hourly, daily, and weekly jobs"

- [ ] **Step 5: Stop Docker and commit**

```bash
docker compose down
git add -A
git commit -m "chore: verify full integration — all tests pass, Docker builds and starts"
```
