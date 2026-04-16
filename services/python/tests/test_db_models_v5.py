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
