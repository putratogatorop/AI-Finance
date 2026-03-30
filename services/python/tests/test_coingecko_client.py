from unittest.mock import MagicMock, patch

import pytest

from src.data.coingecko_client import CoinGeckoClient


@pytest.fixture
def client():
    return CoinGeckoClient(api_key="")


MOCK_MARKET_DATA = [
    {"id": "bitcoin", "symbol": "btc", "name": "Bitcoin", "market_cap": 1_400_000_000_000, "market_cap_rank": 1, "total_volume": 35_000_000_000, "circulating_supply": 19_800_000},
    {"id": "ethereum", "symbol": "eth", "name": "Ethereum", "market_cap": 400_000_000_000, "market_cap_rank": 2, "total_volume": 15_000_000_000, "circulating_supply": 120_000_000},
]


def test_asset_to_coingecko_id(client):
    assert client.asset_to_id("BTC") == "bitcoin"
    assert client.asset_to_id("ETH") == "ethereum"
    assert client.asset_to_id("SOL") == "solana"


def test_parse_market_data(client):
    parsed = client.parse_market_data(MOCK_MARKET_DATA[0])
    assert parsed["asset"] == "BTC"
    assert parsed["market_cap"] == 1_400_000_000_000
    assert parsed["market_cap_rank"] == 1
    assert parsed["total_volume_24h"] == 35_000_000_000
    assert parsed["circulating_supply"] == 19_800_000


@patch("src.data.coingecko_client.CoinGeckoAPI")
def test_fetch_top_n_assets(mock_cg_cls, client):
    mock_cg = MagicMock()
    mock_cg_cls.return_value = mock_cg
    mock_cg.get_coins_markets.return_value = MOCK_MARKET_DATA
    new_client = CoinGeckoClient(api_key="")
    assets = new_client.fetch_top_n_assets(n=2)
    assert len(assets) == 2
    assert assets[0] == "BTC"
    assert assets[1] == "ETH"
