from unittest.mock import MagicMock, patch

import pytest

from src.data.binance_client import BinanceClient


@pytest.fixture
def client():
    return BinanceClient(rate_limit_ms=0)


def test_symbol_for_asset(client):
    assert client.symbol_for_asset("BTC") == "BTC/USDT"
    assert client.symbol_for_asset("ETH") == "ETH/USDT"


def test_parse_ohlcv_row(client):
    raw = [1711785600000, 70000.0, 70500.0, 69800.0, 70200.0, 1500.5]
    parsed = client.parse_ohlcv_row(raw)
    assert parsed["open"] == 70000.0
    assert parsed["high"] == 70500.0
    assert parsed["low"] == 69800.0
    assert parsed["close"] == 70200.0
    assert parsed["volume"] == 1500.5
    assert "timestamp" in parsed


@patch("src.data.binance_client.ccxt.binance")
def test_fetch_hourly_ohlcv(mock_binance_cls, client):
    mock_exchange = MagicMock()
    mock_binance_cls.return_value = mock_exchange
    mock_exchange.fetch_ohlcv.return_value = [
        [1711785600000, 70000.0, 70500.0, 69800.0, 70200.0, 1500.5],
        [1711789200000, 70200.0, 70800.0, 70100.0, 70600.0, 1200.3],
    ]
    new_client = BinanceClient(rate_limit_ms=0)
    rows = new_client.fetch_hourly_ohlcv("BTC", limit=2)
    assert len(rows) == 2
    assert rows[0]["close"] == 70200.0
    assert rows[1]["close"] == 70600.0
    mock_exchange.fetch_ohlcv.assert_called_once_with("BTC/USDT", "1h", limit=2)
