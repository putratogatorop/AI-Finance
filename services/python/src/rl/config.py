"""RL trading gym configuration constants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RLConfig:
    """Immutable configuration for the RL trading environment."""

    # Position constraints
    MAX_POSITIONS: int = 20
    MAX_SINGLE_POSITION_PCT: float = 0.10
    MAX_EXPOSURE_PCT: float = 0.80
    SCORE_THRESHOLD: float = 0.15

    # SL/TP ranges
    SL_MIN: float = 0.02
    SL_MAX: float = 0.08
    TP_MIN: float = 0.03
    TP_MAX: float = 0.15

    # Costs
    TAKER_FEE: float = 0.001
    SLIPPAGE: float = 0.0005
    FUNDING_RATE_8H: float = 0.0001

    # Capital (10M IDR)
    STARTING_CAPITAL: float = 10_000_000

    # Episode
    EPISODE_WINDOWS: int = 365
    MAX_DRAWDOWN_KILL: float = 0.30
    WINDOW_HOURS: int = 12
    CANDLES_PER_WINDOW: int = 3

    # Reward weights
    REWARD_SHARPE_W: float = 0.6
    REWARD_RETURN_W: float = 0.3
    REWARD_DD_PENALTY_W: float = 0.1
    REWARD_DD_THRESHOLD: float = 0.20

    # Features
    N_ALT_FEATURES: int = 28
    N_INDICATOR_FEATURES: int = 9
    N_PORTFOLIO_FEATURES: int = 5
    LSTM_LOOKBACK: int = 10

    # Universe
    INDICATOR_ASSETS: tuple[str, ...] = ("BTC", "ETH")
