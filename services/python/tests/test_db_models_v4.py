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
