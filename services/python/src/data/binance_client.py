import time
from datetime import UTC, datetime
from typing import Any

import ccxt


class BinanceClient:
    def __init__(self, rate_limit_ms: int = 100) -> None:
        self.exchange = ccxt.binance({"enableRateLimit": True})
        self.rate_limit_ms = rate_limit_ms

    def symbol_for_asset(self, asset: str) -> str:
        return f"{asset}/USDT"

    def parse_ohlcv_row(self, raw: list[Any]) -> dict[str, Any]:
        timestamp_ms, open_price, high, low, close, volume = raw
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
        return {
            "timestamp": dt.isoformat(),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }

    def fetch_hourly_ohlcv(self, asset: str, limit: int = 24) -> list[dict[str, Any]]:
        symbol = self.symbol_for_asset(asset)
        raw_data = self.exchange.fetch_ohlcv(symbol, "1h", limit=limit)
        if self.rate_limit_ms > 0:
            time.sleep(self.rate_limit_ms / 1000)
        return [self.parse_ohlcv_row(row) for row in raw_data]

    def fetch_daily_ohlcv(self, asset: str, limit: int = 1) -> list[dict[str, Any]]:
        symbol = self.symbol_for_asset(asset)
        raw_data = self.exchange.fetch_ohlcv(symbol, "1d", limit=limit)
        if self.rate_limit_ms > 0:
            time.sleep(self.rate_limit_ms / 1000)
        return [self.parse_ohlcv_row(row) for row in raw_data]

    def fetch_historical_hourly(
        self, asset: str, since: datetime, limit: int = 500
    ) -> list[dict[str, Any]]:
        symbol = self.symbol_for_asset(asset)
        since_ms = int(since.timestamp() * 1000)
        raw_data = self.exchange.fetch_ohlcv(symbol, "1h", since=since_ms, limit=limit)
        if self.rate_limit_ms > 0:
            time.sleep(self.rate_limit_ms / 1000)
        return [self.parse_ohlcv_row(row) for row in raw_data]
