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
    action: Mapped[str] = mapped_column(String(10), nullable=False)
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
    status: Mapped[str] = mapped_column(String(10), nullable=False)
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
    __table_args__ = (
        UniqueConstraint("asset", "timestamp", "source", name="uq_funding_asset_ts_src"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    funding_rate: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)


class ContractStats1h(Base):
    __tablename__ = "contract_stats_1h"
    __table_args__ = (UniqueConstraint("asset", "timestamp", name="uq_cstats_asset_ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    asset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    open_interest_usd: Mapped[float] = mapped_column(Float, nullable=False)
    long_liq_usd: Mapped[float] = mapped_column(Float, nullable=True)
    short_liq_usd: Mapped[float] = mapped_column(Float, nullable=True)
    lsr_taker: Mapped[float] = mapped_column(Float, nullable=True)
    lsr_account: Mapped[float] = mapped_column(Float, nullable=True)
    top_lsr_size: Mapped[float] = mapped_column(Float, nullable=True)
    top_lsr_account: Mapped[float] = mapped_column(Float, nullable=True)


class FearGreedIndex(Base):
    __tablename__ = "fear_greed_index"
    __table_args__ = (UniqueConstraint("date", name="uq_fgi_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    value: Mapped[int] = mapped_column(Integer, nullable=False)
    classification: Mapped[str] = mapped_column(String(50), nullable=False)


class VolumeBreakout(Base):
    __tablename__ = "volume_breakouts"
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "signal_time", name="uq_vb_sym_tf_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timeframe: Mapped[str] = mapped_column(String(4), nullable=False, index=True)
    signal_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Breakout bar OHLCV
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    quote_volume: Mapped[float] = mapped_column(Float, nullable=False)
    trades: Mapped[int] = mapped_column(Integer, nullable=True)

    # Volume context
    vol_ratio: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    vol_avg_20: Mapped[float] = mapped_column(Float, nullable=False)

    # Price context
    price_change_1bar: Mapped[float] = mapped_column(Float, nullable=True)
    price_change_2bar: Mapped[float] = mapped_column(Float, nullable=True)
    atr_14: Mapped[float] = mapped_column(Float, nullable=True)
    bar_range_pct: Mapped[float] = mapped_column(Float, nullable=True)
    upper_wick_pct: Mapped[float] = mapped_column(Float, nullable=True)
    lower_wick_pct: Mapped[float] = mapped_column(Float, nullable=True)
    body_pct: Mapped[float] = mapped_column(Float, nullable=True)
    direction: Mapped[int] = mapped_column(Integer, nullable=False)

    # Wider context
    dist_from_20_high: Mapped[float] = mapped_column(Float, nullable=True)
    dist_from_20_low: Mapped[float] = mapped_column(Float, nullable=True)
    price_vs_ema_50: Mapped[float] = mapped_column(Float, nullable=True)
    rsi_14: Mapped[float] = mapped_column(Float, nullable=True)
    volatility_20: Mapped[float] = mapped_column(Float, nullable=True)

    # BTC context
    btc_price: Mapped[float] = mapped_column(Float, nullable=True)
    btc_ret_24bar: Mapped[float] = mapped_column(Float, nullable=True)
    btc_ret_96bar: Mapped[float] = mapped_column(Float, nullable=True)
    btc_vol_ratio: Mapped[float] = mapped_column(Float, nullable=True)

    # Forward returns
    fwd_ret_12h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_ret_24h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_ret_48h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_ret_1w: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_ret_2w: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_ret_1m: Mapped[float] = mapped_column(Float, nullable=True)

    # Forward extremes (MFE/MAE)
    fwd_max_gain_12h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_max_gain_24h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_max_gain_48h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_max_gain_1w: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_max_loss_12h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_max_loss_24h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_max_loss_48h: Mapped[float] = mapped_column(Float, nullable=True)
    fwd_max_loss_1w: Mapped[float] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.now
    )
