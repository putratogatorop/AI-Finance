from datetime import UTC, datetime

from src.db.models import (
    AssetFundamental,
    AssetPriceDaily,
    AssetPriceHourly,
    AuditLog,
    Portfolio,
    Signal,
)


def test_insert_hourly_price(db_session):
    price = AssetPriceHourly(
        asset="BTC",
        timestamp=datetime(2026, 3, 30, 10, 0, tzinfo=UTC),
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
        date=datetime(2026, 3, 30, tzinfo=UTC),
        open=3500.0, high=3600.0, low=3450.0, close=3550.0, volume=25000.0,
    )
    db_session.add(price)
    db_session.commit()
    result = db_session.query(AssetPriceDaily).first()
    assert result is not None
    assert result.asset == "ETH"


def test_insert_fundamental(db_session):
    fund = AssetFundamental(
        asset="BTC",
        fetched_at=datetime(2026, 3, 30, tzinfo=UTC),
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
        asset="ETH", action="BUY", confidence=0.84,
        suggested_hold_days=18, stop_loss_pct=-8.0,
        expected_return_pct=12.0, model_agreement="3/3",
        created_at=datetime(2026, 3, 30, tzinfo=UTC),
    )
    db_session.add(signal)
    db_session.commit()
    result = db_session.query(Signal).first()
    assert result is not None
    assert result.action == "BUY"
    assert result.confidence == 0.84


def test_insert_portfolio(db_session):
    position = Portfolio(
        asset="SOL", action="BUY", entry_price=180.0,
        entry_amount_idr=1_000_000, quantity=5555.56,
        stop_loss_price=165.6, status="open",
        opened_at=datetime(2026, 3, 30, tzinfo=UTC),
    )
    db_session.add(position)
    db_session.commit()
    result = db_session.query(Portfolio).first()
    assert result is not None
    assert result.status == "open"
    assert result.entry_amount_idr == 1_000_000


def test_insert_audit_log(db_session):
    log = AuditLog(
        event_type="signal_generated", asset="ETH",
        details='{"action": "BUY", "confidence": 0.84}',
        created_at=datetime(2026, 3, 30, tzinfo=UTC),
    )
    db_session.add(log)
    db_session.commit()
    result = db_session.query(AuditLog).first()
    assert result is not None
    assert result.event_type == "signal_generated"
