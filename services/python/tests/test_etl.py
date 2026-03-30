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
    assert daily.open == 70000.0
    assert daily.high == 71000.0
    assert daily.low == 69500.0
    assert daily.close == 70800.0
    assert daily.volume == 3000.0
