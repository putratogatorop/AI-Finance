from unittest.mock import MagicMock

import pytest

from src.data.backfill import Backfiller


@pytest.fixture
def mock_deps():
    return {
        "binance_client": MagicMock(),
        "csv_storage": MagicMock(),
    }


def test_backfill_generates_date_range(mock_deps):
    backfiller = Backfiller(**mock_deps)
    dates = backfiller.generate_date_range("2026-03-28", "2026-03-30")
    assert dates == ["2026-03-28", "2026-03-29", "2026-03-30"]


def test_backfill_fetches_for_each_date(mock_deps):
    mock_deps["binance_client"].fetch_historical_hourly.return_value = [
        {"timestamp": "2026-03-28T00:00:00+00:00", "open": 1.0, "high": 2.0,
         "low": 0.5, "close": 1.5, "volume": 100.0}
    ]
    backfiller = Backfiller(**mock_deps)
    backfiller.backfill_asset("BTC", "2026-03-28", "2026-03-28")
    mock_deps["binance_client"].fetch_historical_hourly.assert_called_once()
    mock_deps["csv_storage"].save_hourly.assert_called_once_with(
        "BTC", "2026-03-28",
        [{"timestamp": "2026-03-28T00:00:00+00:00", "open": 1.0, "high": 2.0,
          "low": 0.5, "close": 1.5, "volume": 100.0}]
    )
