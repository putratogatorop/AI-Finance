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
            "open": 70000.0, "high": 70500.0, "low": 69800.0,
            "close": 70200.0, "volume": 1500.5,
        },
        {
            "timestamp": "2026-03-30T11:00:00+00:00",
            "open": 70200.0, "high": 70800.0, "low": 70100.0,
            "close": 70600.0, "volume": 1200.3,
        },
    ]
    csv_storage.save_hourly("BTC", "2026-03-30", rows)
    file_path = Path(csv_storage.base_path) / "hourly" / "BTC" / "2026-03-30.csv"
    assert file_path.exists()
    df = pd.read_csv(file_path)
    assert len(df) == 2
    assert df.iloc[0]["close"] == 70200.0


def test_save_hourly_appends_to_existing(csv_storage):
    row1 = [
        {"timestamp": "2026-03-30T10:00:00+00:00", "open": 1.0, "high": 2.0,
         "low": 0.5, "close": 1.5, "volume": 100.0}
    ]
    row2 = [
        {"timestamp": "2026-03-30T11:00:00+00:00", "open": 1.5, "high": 2.5,
         "low": 1.0, "close": 2.0, "volume": 200.0}
    ]
    csv_storage.save_hourly("BTC", "2026-03-30", row1)
    csv_storage.save_hourly("BTC", "2026-03-30", row2)
    file_path = Path(csv_storage.base_path) / "hourly" / "BTC" / "2026-03-30.csv"
    df = pd.read_csv(file_path)
    assert len(df) == 2


def test_save_daily_creates_file(csv_storage):
    rows = [
        {"date": "2026-03-30", "open": 70000.0, "high": 71000.0,
         "low": 69000.0, "close": 70500.0, "volume": 35000.0}
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
    rows = [
        {"timestamp": "2026-03-30T10:00:00+00:00", "open": 1.0, "high": 2.0,
         "low": 0.5, "close": 1.5, "volume": 100.0}
    ]
    csv_storage.save_hourly("ETH", "2026-03-30", rows)
    df = csv_storage.read_hourly("ETH", "2026-03-30")
    assert len(df) == 1
    assert df.iloc[0]["close"] == 1.5


def test_read_hourly_missing_returns_empty(csv_storage):
    df = csv_storage.read_hourly("ETH", "2026-01-01")
    assert len(df) == 0
