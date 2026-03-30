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
                asset=asset, timestamp=ts,
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
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
            asset=asset, date=date_dt,
            open=hourly_rows[0].open,
            high=max(r.high for r in hourly_rows),
            low=min(r.low for r in hourly_rows),
            close=hourly_rows[-1].close,
            volume=sum(r.volume for r in hourly_rows),
        )

        existing = self.session.query(AssetPriceDaily).filter_by(asset=asset, date=date_dt).first()
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
