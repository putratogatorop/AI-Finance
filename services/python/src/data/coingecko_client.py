from datetime import UTC, datetime
from typing import Any

from pycoingecko import CoinGeckoAPI

SYMBOL_TO_ID: dict[str, str] = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin", "SOL": "solana",
    "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin", "AVAX": "avalanche-2",
    "DOT": "polkadot", "LINK": "chainlink", "UNI": "uniswap", "ATOM": "cosmos",
    "LTC": "litecoin", "APT": "aptos", "NEAR": "near", "OP": "optimism",
    "ARB": "arbitrum", "FIL": "filecoin", "INJ": "injective-protocol", "MATIC": "matic-network",
}

ID_TO_SYMBOL: dict[str, str] = {v: k for k, v in SYMBOL_TO_ID.items()}


class CoinGeckoClient:
    def __init__(self, api_key: str = "") -> None:
        self.cg = CoinGeckoAPI()

    def asset_to_id(self, asset: str) -> str:
        return SYMBOL_TO_ID.get(asset, asset.lower())

    def id_to_asset(self, cg_id: str) -> str:
        return ID_TO_SYMBOL.get(cg_id, cg_id.upper())

    def parse_market_data(self, raw: dict[str, Any]) -> dict[str, Any]:
        symbol = self.id_to_asset(raw["id"])
        return {
            "asset": symbol,
            "fetched_at": datetime.now(UTC).isoformat(),
            "market_cap": raw.get("market_cap"),
            "market_cap_rank": raw.get("market_cap_rank"),
            "total_volume_24h": raw.get("total_volume"),
            "circulating_supply": raw.get("circulating_supply"),
            "category": None,
        }

    def fetch_top_n_assets(self, n: int = 20) -> list[str]:
        markets = self.cg.get_coins_markets(
            vs_currency="usd", order="market_cap_desc", per_page=n, page=1
        )
        return [self.id_to_asset(coin["id"]) for coin in markets]

    def fetch_fundamentals(self, assets: list[str]) -> list[dict[str, Any]]:
        ids = [self.asset_to_id(a) for a in assets]
        ids_str = ",".join(ids)
        markets = self.cg.get_coins_markets(
            vs_currency="usd",
            ids=ids_str,
            order="market_cap_desc",
            per_page=len(assets),
            page=1,
        )
        return [self.parse_market_data(coin) for coin in markets]
