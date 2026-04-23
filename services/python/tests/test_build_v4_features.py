import sys

sys.path.insert(0, ".")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from src.db.models import Base


def test_load_15m_from_pg():
    from scripts.build_v4_features import load_15m_data

    engine = create_engine("postgresql://postgres:MySQL100%25@localhost:5432/market")
    try:
        Base.metadata.create_all(engine)
        df = load_15m_data(engine, "BTC")
    except OperationalError as e:
        pytest.skip(f"Postgres not reachable in this env: {e}")
    if len(df) == 0:
        pytest.skip("Postgres asset_prices_15m is empty in this env")
    assert len(df) > 100_000
    assert "close" in df.columns
    assert "taker_buy_base" in df.columns
    assert "timestamp" in df.columns


def test_load_funding_from_pg():
    from scripts.build_v4_features import load_funding_data

    engine = create_engine("postgresql://postgres:MySQL100%25@localhost:5432/market")
    try:
        df = load_funding_data(engine, "BTC")
    except OperationalError as e:
        pytest.skip(f"Postgres not reachable in this env: {e}")
    if len(df) == 0:
        pytest.skip("Postgres funding table is empty in this env")
    assert len(df) > 3000
    assert "funding_rate" in df.columns
    assert "timestamp" in df.columns
