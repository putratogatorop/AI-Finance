import sys
sys.path.insert(0, ".")

import pandas as pd
from sqlalchemy import create_engine, text
from src.db.models import Base


def test_load_15m_from_pg():
    from scripts.build_v4_features import load_15m_data

    engine = create_engine("postgresql://postgres:MySQL100%25@localhost:5432/market")
    Base.metadata.create_all(engine)

    df = load_15m_data(engine, "BTC")
    assert len(df) > 100_000
    assert "close" in df.columns
    assert "taker_buy_base" in df.columns
    assert "timestamp" in df.columns


def test_load_funding_from_pg():
    from scripts.build_v4_features import load_funding_data

    engine = create_engine("postgresql://postgres:MySQL100%25@localhost:5432/market")
    df = load_funding_data(engine, "BTC")
    assert len(df) > 3000
    assert "funding_rate" in df.columns
    assert "timestamp" in df.columns
