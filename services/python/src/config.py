from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    # Data Lake
    DATA_LAKE_PATH: str = "/app/data/raw"

    # Binance
    BINANCE_RATE_LIMIT_MS: int = 100

    # CoinGecko
    COINGECKO_API_KEY: str = ""

    # Scheduler
    SCHEDULER_ENABLED: bool = True

    # App
    LOG_LEVEL: str = "INFO"
    POSITION_SIZE_IDR: int = 1_000_000

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Top 20 crypto by market cap — refreshed weekly by CoinGecko job
    DEFAULT_ASSETS: list[str] = [
        "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX",
        "DOT", "LINK", "UNI", "ATOM", "LTC", "APT", "NEAR",
        "OP", "ARB", "FIL", "INJ", "MATIC",
    ]
