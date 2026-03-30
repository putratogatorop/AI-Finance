from unittest.mock import MagicMock

import pytest

from src.scheduler.jobs import JobRunner


@pytest.fixture
def mock_deps():
    return {
        "binance_client": MagicMock(),
        "coingecko_client": MagicMock(),
        "csv_storage": MagicMock(),
        "etl": MagicMock(),
        "assets": ["BTC", "ETH"],
    }


def test_hourly_job_fetches_and_stores(mock_deps):
    mock_deps["binance_client"].fetch_hourly_ohlcv.return_value = [
        {
            "timestamp": "2026-03-30T10:00:00+00:00",
            "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0,
        }
    ]
    runner = JobRunner(**mock_deps)
    runner.run_hourly_ingestion()
    assert mock_deps["binance_client"].fetch_hourly_ohlcv.call_count == 2
    assert mock_deps["csv_storage"].save_hourly.call_count == 2
    assert mock_deps["etl"].load_hourly.call_count == 2


def test_daily_aggregation_job(mock_deps):
    runner = JobRunner(**mock_deps)
    runner.run_daily_aggregation()
    assert mock_deps["etl"].aggregate_daily.call_count == 2


def test_weekly_fundamentals_job(mock_deps):
    mock_deps["coingecko_client"].fetch_fundamentals.return_value = [
        {"asset": "BTC", "market_cap": 1_400_000_000_000, "market_cap_rank": 1,
         "total_volume_24h": 35_000_000_000, "circulating_supply": 19_800_000,
         "category": None, "fetched_at": "2026-03-30T00:00:00+00:00"},
    ]
    mock_deps["coingecko_client"].fetch_top_n_assets.return_value = ["BTC", "ETH"]
    runner = JobRunner(**mock_deps)
    runner.run_weekly_fundamentals()
    mock_deps["coingecko_client"].fetch_top_n_assets.assert_called_once_with(n=20)
    mock_deps["coingecko_client"].fetch_fundamentals.assert_called_once()
    assert mock_deps["csv_storage"].save_fundamentals.call_count >= 1
