import sys

sys.path.insert(0, ".")

from datetime import UTC, datetime

from sqlalchemy import create_engine, text

from src.db.models import Base


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
