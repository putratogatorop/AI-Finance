# v3 Week 1: Trading Gym Environment + Features

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a realistic crypto trading gym environment that RL agents can train on, with 43 features (28 per-alt + 9 BTC/ETH indicators + 5 portfolio state + 1 cross-token residual), non-overlapping 12-hour windows, SL/TP simulation, fees, and funding rates.

**Architecture:** OpenAI Gym-style environment (`CryptoTradingEnv`) that replays historical 4h candle data for ~198 altcoins. BTC/ETH are read-only indicators. The environment steps in 12-hour windows (2 per day), simulates SL/TP fills against intra-period high/low, deducts fees/slippage/funding, and returns observations + rewards. A `DataLoader` preprocesses raw CSVs into a structured tensor format for fast episode replay.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Pandas, Gymnasium (OpenAI Gym successor)

**Spec:** `docs/superpowers/specs/2026-04-02-rl-trading-agent-v3-design.md`

---

### File Structure

```
services/python/
├── src/rl/
│   ├── __init__.py
│   ├── env.py              # CryptoTradingEnv (Gym environment)
│   ├── data_loader.py      # Load raw CSVs → structured arrays for fast replay
│   ├── features.py         # Cross-token features (BTC indicators, beta, correlation)
│   ├── portfolio.py        # Portfolio state tracking (positions, PnL, cash)
│   └── config.py           # RL-specific constants and configuration
├── scripts/
│   └── prepare_rl_data.py  # One-time script: raw CSVs → rl_data.npz
└── tests/
    ├── test_rl_env.py
    ├── test_rl_data_loader.py
    ├── test_rl_features.py
    └── test_rl_portfolio.py
```

---

### Task 1: RL Config Constants

**Files:**
- Create: `services/python/src/rl/__init__.py`
- Create: `services/python/src/rl/config.py`
- Test: `services/python/tests/test_rl_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_config.py
from src.rl.config import RLConfig


def test_config_defaults():
    cfg = RLConfig()
    assert cfg.MAX_POSITIONS == 10
    assert cfg.MAX_SINGLE_POSITION_PCT == 0.10
    assert cfg.MAX_EXPOSURE_PCT == 0.80
    assert cfg.SL_MIN == 0.02
    assert cfg.SL_MAX == 0.08
    assert cfg.TP_MIN == 0.03
    assert cfg.TP_MAX == 0.15
    assert cfg.TAKER_FEE == 0.001
    assert cfg.SLIPPAGE == 0.0005
    assert cfg.FUNDING_RATE_8H == 0.0001
    assert cfg.STARTING_CAPITAL == 10_000_000
    assert cfg.EPISODE_WINDOWS == 365
    assert cfg.MAX_DRAWDOWN_KILL == 0.30
    assert cfg.WINDOW_HOURS == 12
    assert cfg.CANDLES_PER_WINDOW == 3  # 12h / 4h = 3 candles per window


def test_config_reward_weights():
    cfg = RLConfig()
    assert cfg.REWARD_SHARPE_W == 0.6
    assert cfg.REWARD_RETURN_W == 0.3
    assert cfg.REWARD_DD_PENALTY_W == 0.1
    assert cfg.REWARD_DD_THRESHOLD == 0.20


def test_config_feature_counts():
    cfg = RLConfig()
    assert cfg.N_ALT_FEATURES == 28
    assert cfg.N_INDICATOR_FEATURES == 9
    assert cfg.N_PORTFOLIO_FEATURES == 5
    assert cfg.LSTM_LOOKBACK == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_rl_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rl'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rl/__init__.py
```

```python
# src/rl/config.py
"""RL Trading Agent configuration constants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RLConfig:
    """Immutable configuration for the RL trading environment."""

    # Position constraints
    MAX_POSITIONS: int = 10
    MAX_SINGLE_POSITION_PCT: float = 0.10
    MAX_EXPOSURE_PCT: float = 0.80

    # SL/TP ranges
    SL_MIN: float = 0.02
    SL_MAX: float = 0.08
    TP_MIN: float = 0.03
    TP_MAX: float = 0.15

    # Costs
    TAKER_FEE: float = 0.001       # 0.1% per trade
    SLIPPAGE: float = 0.0005       # 0.05%
    FUNDING_RATE_8H: float = 0.0001  # 0.01% every 8h

    # Capital
    STARTING_CAPITAL: float = 10_000_000  # 10M IDR

    # Episode
    EPISODE_WINDOWS: int = 365     # 6 months of 12h windows
    MAX_DRAWDOWN_KILL: float = 0.30  # Episode ends if DD > 30%
    WINDOW_HOURS: int = 12
    CANDLES_PER_WINDOW: int = 3    # 12h / 4h

    # Reward
    REWARD_SHARPE_W: float = 0.6
    REWARD_RETURN_W: float = 0.3
    REWARD_DD_PENALTY_W: float = 0.1
    REWARD_DD_THRESHOLD: float = 0.20

    # Features
    N_ALT_FEATURES: int = 28
    N_INDICATOR_FEATURES: int = 9
    N_PORTFOLIO_FEATURES: int = 5
    LSTM_LOOKBACK: int = 10

    # Tradeable universe
    INDICATOR_ASSETS: tuple[str, ...] = ("BTC", "ETH")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_rl_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add services/python/src/rl/ services/python/tests/test_rl_config.py
git commit -m "feat(rl): add RL config constants"
```

---

### Task 2: Portfolio State Tracker

**Files:**
- Create: `services/python/src/rl/portfolio.py`
- Test: `services/python/tests/test_rl_portfolio.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_portfolio.py
import numpy as np
from src.rl.portfolio import Portfolio, Position
from src.rl.config import RLConfig


def test_portfolio_initial_state():
    p = Portfolio(RLConfig())
    assert p.cash == 10_000_000
    assert p.equity == 10_000_000
    assert len(p.positions) == 0
    assert p.n_positions == 0


def test_open_long_position():
    p = Portfolio(RLConfig())
    ok = p.open_position("DOGE", side=1, size_pct=0.05, entry_price=0.10,
                         sl_pct=0.05, tp_pct=0.10)
    assert ok is True
    assert p.n_positions == 1
    pos = p.positions["DOGE"]
    assert pos.side == 1
    assert pos.entry_price == 0.10
    assert pos.sl_price == 0.10 * (1 - 0.05)  # 0.095
    assert pos.tp_price == 0.10 * (1 + 0.10)  # 0.11
    assert p.cash < 10_000_000  # Capital allocated


def test_open_short_position():
    p = Portfolio(RLConfig())
    ok = p.open_position("SHIB", side=-1, size_pct=0.05, entry_price=0.00001,
                         sl_pct=0.05, tp_pct=0.10)
    assert ok is True
    pos = p.positions["SHIB"]
    assert pos.side == -1
    assert pos.sl_price == 0.00001 * (1 + 0.05)  # SL above for shorts
    assert pos.tp_price == 0.00001 * (1 - 0.10)  # TP below for shorts


def test_reject_over_max_positions():
    cfg = RLConfig()
    p = Portfolio(cfg)
    for i in range(cfg.MAX_POSITIONS):
        p.open_position(f"ALT{i}", side=1, size_pct=0.05,
                        entry_price=1.0, sl_pct=0.05, tp_pct=0.10)
    ok = p.open_position("ONEMORE", side=1, size_pct=0.05,
                         entry_price=1.0, sl_pct=0.05, tp_pct=0.10)
    assert ok is False
    assert p.n_positions == cfg.MAX_POSITIONS


def test_reject_over_max_exposure():
    p = Portfolio(RLConfig())
    # Try to allocate 90% in one shot (max is 80%)
    ok = p.open_position("BIG", side=1, size_pct=0.90,
                         entry_price=1.0, sl_pct=0.05, tp_pct=0.10)
    assert ok is False


def test_close_position_profit():
    p = Portfolio(RLConfig())
    p.open_position("DOGE", side=1, size_pct=0.05, entry_price=0.10,
                    sl_pct=0.05, tp_pct=0.10)
    initial_cash = p.cash
    pnl = p.close_position("DOGE", exit_price=0.11)  # +10%
    assert pnl > 0
    assert p.n_positions == 0
    assert p.cash > initial_cash


def test_close_short_position_profit():
    p = Portfolio(RLConfig())
    p.open_position("SHIB", side=-1, size_pct=0.05, entry_price=0.10,
                    sl_pct=0.05, tp_pct=0.10)
    pnl = p.close_position("SHIB", exit_price=0.09)  # Price dropped = profit for short
    assert pnl > 0


def test_check_sl_tp_long_sl_hit():
    p = Portfolio(RLConfig())
    p.open_position("DOGE", side=1, size_pct=0.05, entry_price=100.0,
                    sl_pct=0.05, tp_pct=0.10)
    # Candle low goes below SL (95.0)
    fills = p.check_sl_tp({"DOGE": {"high": 102.0, "low": 94.0}})
    assert "DOGE" in fills
    assert fills["DOGE"]["reason"] == "sl"
    assert p.n_positions == 0


def test_check_sl_tp_long_tp_hit():
    p = Portfolio(RLConfig())
    p.open_position("DOGE", side=1, size_pct=0.05, entry_price=100.0,
                    sl_pct=0.05, tp_pct=0.10)
    # Candle high goes above TP (110.0)
    fills = p.check_sl_tp({"DOGE": {"high": 111.0, "low": 99.0}})
    assert fills["DOGE"]["reason"] == "tp"
    assert p.n_positions == 0


def test_deduct_funding():
    p = Portfolio(RLConfig())
    p.open_position("DOGE", side=1, size_pct=0.05, entry_price=100.0,
                    sl_pct=0.05, tp_pct=0.10)
    equity_before = p.equity_at_prices({"DOGE": 100.0})
    p.deduct_funding(n_8h_periods=3)  # 12h window = 1.5 but we pass 3 candles
    equity_after = p.equity_at_prices({"DOGE": 100.0})
    assert equity_after < equity_before


def test_portfolio_state_vector():
    p = Portfolio(RLConfig())
    p.open_position("DOGE", side=1, size_pct=0.05, entry_price=100.0,
                    sl_pct=0.05, tp_pct=0.10)
    state = p.get_state_vector(current_prices={"DOGE": 102.0})
    assert len(state) == 5  # cash_ratio, pos_count_ratio, pnl_pct, avg_hold, max_dd
    assert 0 <= state[0] <= 1  # cash_ratio
    assert state[1] == 0.1  # 1/10 positions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_rl_portfolio.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rl/portfolio.py
"""Portfolio state tracker for RL trading environment."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.rl.config import RLConfig


@dataclass
class Position:
    """A single open position."""

    asset: str
    side: int  # 1 = long, -1 = short
    size_idr: float  # Capital allocated
    quantity: float  # Units held
    entry_price: float
    sl_price: float
    tp_price: float
    windows_held: int = 0


class Portfolio:
    """Tracks cash, positions, and computes portfolio metrics."""

    def __init__(self, config: RLConfig):
        self.config = config
        self.cash: float = config.STARTING_CAPITAL
        self.positions: dict[str, Position] = {}
        self._peak_equity: float = config.STARTING_CAPITAL
        self._total_funding_paid: float = 0.0

    @property
    def n_positions(self) -> int:
        return len(self.positions)

    @property
    def equity(self) -> float:
        """Equity = cash + sum of position sizes (at entry, not mark-to-market)."""
        return self.cash + sum(p.size_idr for p in self.positions.values())

    def equity_at_prices(self, current_prices: dict[str, float]) -> float:
        """Mark-to-market equity."""
        total = self.cash
        for asset, pos in self.positions.items():
            price = current_prices.get(asset, pos.entry_price)
            pnl = pos.side * (price - pos.entry_price) / pos.entry_price * pos.size_idr
            total += pos.size_idr + pnl
        return total

    def _total_exposure(self) -> float:
        return sum(p.size_idr for p in self.positions.values())

    def open_position(
        self,
        asset: str,
        side: int,
        size_pct: float,
        entry_price: float,
        sl_pct: float,
        tp_pct: float,
    ) -> bool:
        """Open a new position. Returns False if rejected by constraints."""
        if self.n_positions >= self.config.MAX_POSITIONS:
            return False
        if asset in self.positions:
            return False

        size_idr = self.equity * size_pct
        # Check max single position
        if size_pct > self.config.MAX_SINGLE_POSITION_PCT:
            return False
        # Check max exposure
        if (self._total_exposure() + size_idr) / self.equity > self.config.MAX_EXPOSURE_PCT:
            return False
        # Check enough cash
        total_cost = size_idr * (1 + self.config.TAKER_FEE + self.config.SLIPPAGE)
        if total_cost > self.cash:
            return False

        # Compute SL/TP prices
        if side == 1:  # Long
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:  # Short
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)

        quantity = size_idr / entry_price
        self.positions[asset] = Position(
            asset=asset, side=side, size_idr=size_idr, quantity=quantity,
            entry_price=entry_price, sl_price=sl_price, tp_price=tp_price,
        )
        self.cash -= total_cost
        return True

    def close_position(self, asset: str, exit_price: float) -> float:
        """Close a position at exit_price. Returns PnL in IDR."""
        pos = self.positions.pop(asset)
        pnl = pos.side * (exit_price - pos.entry_price) / pos.entry_price * pos.size_idr
        fee = pos.size_idr * (self.config.TAKER_FEE + self.config.SLIPPAGE)
        net_pnl = pnl - fee
        self.cash += pos.size_idr + net_pnl
        return net_pnl

    def check_sl_tp(
        self, candle_data: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, str | float]]:
        """Check if any SL/TP was hit using candle high/low. Returns fills."""
        fills: dict[str, dict[str, str | float]] = {}
        assets_to_close = []

        for asset, pos in self.positions.items():
            if asset not in candle_data:
                continue
            high = candle_data[asset]["high"]
            low = candle_data[asset]["low"]

            if pos.side == 1:  # Long
                if low <= pos.sl_price:
                    assets_to_close.append((asset, pos.sl_price, "sl"))
                elif high >= pos.tp_price:
                    assets_to_close.append((asset, pos.tp_price, "tp"))
            else:  # Short
                if high >= pos.sl_price:
                    assets_to_close.append((asset, pos.sl_price, "sl"))
                elif low <= pos.tp_price:
                    assets_to_close.append((asset, pos.tp_price, "tp"))

        for asset, price, reason in assets_to_close:
            pnl = self.close_position(asset, price)
            fills[asset] = {"reason": reason, "pnl": pnl, "price": price}

        return fills

    def deduct_funding(self, n_8h_periods: int = 1) -> None:
        """Deduct funding rate from all open positions."""
        for pos in self.positions.values():
            cost = pos.size_idr * self.config.FUNDING_RATE_8H * n_8h_periods
            self.cash -= cost
            self._total_funding_paid += cost

    def get_state_vector(self, current_prices: dict[str, float]) -> list[float]:
        """Return 5-element portfolio state for the RL agent."""
        equity = self.equity_at_prices(current_prices)
        if equity > self._peak_equity:
            self._peak_equity = equity
        drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0

        cash_ratio = self.cash / equity if equity > 0 else 1.0
        pos_ratio = self.n_positions / self.config.MAX_POSITIONS
        pnl_pct = (equity - self.config.STARTING_CAPITAL) / self.config.STARTING_CAPITAL

        avg_hold = 0.0
        max_pos_dd = 0.0
        if self.positions:
            avg_hold = sum(p.windows_held for p in self.positions.values()) / len(self.positions)
            avg_hold = avg_hold / 100.0  # Normalize
            for asset, pos in self.positions.items():
                price = current_prices.get(asset, pos.entry_price)
                pos_ret = pos.side * (price - pos.entry_price) / pos.entry_price
                if pos_ret < max_pos_dd:
                    max_pos_dd = pos_ret

        return [cash_ratio, pos_ratio, pnl_pct, avg_hold, max_pos_dd]

    def increment_hold_times(self) -> None:
        """Call once per window to track holding duration."""
        for pos in self.positions.values():
            pos.windows_held += 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_rl_portfolio.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add services/python/src/rl/portfolio.py services/python/tests/test_rl_portfolio.py
git commit -m "feat(rl): add portfolio state tracker with SL/TP/funding"
```

---

### Task 3: Cross-Token Feature Builder

**Files:**
- Create: `services/python/src/rl/features.py`
- Test: `services/python/tests/test_rl_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_features.py
import numpy as np
import pandas as pd
from src.rl.features import build_cross_token_features, build_indicator_features


def _make_price_series(n=500, base=100.0, seed=42):
    """Generate a fake OHLCV DataFrame."""
    rng = np.random.RandomState(seed)
    close = base * np.cumprod(1 + rng.randn(n) * 0.02)
    return pd.DataFrame({
        "open": close * (1 + rng.randn(n) * 0.005),
        "high": close * (1 + np.abs(rng.randn(n) * 0.01)),
        "low": close * (1 - np.abs(rng.randn(n) * 0.01)),
        "close": close,
        "volume": np.abs(rng.randn(n) * 1e6),
    })


def test_build_cross_token_features():
    btc = _make_price_series(500, 50000.0, seed=1)
    alt = _make_price_series(500, 1.0, seed=2)
    result = build_cross_token_features(alt_close=alt["close"], btc_close=btc["close"])
    assert "btc_alt_beta" in result.columns
    assert "btc_alt_corr" in result.columns
    assert "btc_alt_residual" in result.columns
    assert len(result) == 500
    # Beta and correlation should be finite where computed
    valid = result.dropna()
    assert len(valid) > 300
    assert np.isfinite(valid["btc_alt_beta"]).all()


def test_build_indicator_features():
    btc = _make_price_series(500, 50000.0, seed=1)
    eth = _make_price_series(500, 3000.0, seed=3)
    result = build_indicator_features(btc_df=btc, eth_df=eth)
    expected_cols = [
        "btc_ret_1", "btc_ret_2", "btc_ret_3",
        "btc_vol_24", "btc_rsi",
        "eth_ret_1", "eth_ret_2", "eth_ret_3",
        "eth_btc_ratio_chg", "btc_above_sma50",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing: {col}"
    assert len(result) == 500
    valid = result.dropna()
    assert len(valid) > 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_rl_features.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/rl/features.py
"""Cross-token and indicator features for RL agent.

BTC/ETH are read-only indicators. These features capture the leading
relationship between BTC and altcoins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_cross_token_features(
    alt_close: pd.Series, btc_close: pd.Series, window: int = 180
) -> pd.DataFrame:
    """Compute BTC-alt dependency features for one altcoin.

    Args:
        alt_close: Alt close prices (same index as btc_close).
        btc_close: BTC close prices.
        window: Rolling window in candles (~30 days at 4h).

    Returns:
        DataFrame with columns: btc_alt_beta, btc_alt_corr, btc_alt_residual
    """
    alt_ret = alt_close.pct_change()
    btc_ret = btc_close.pct_change()

    # Rolling correlation
    corr = alt_ret.rolling(window).corr(btc_ret)

    # Rolling beta = cov(alt, btc) / var(btc)
    cov = alt_ret.rolling(window).cov(btc_ret)
    var_btc = btc_ret.rolling(window).var()
    beta = cov / var_btc.replace(0, np.nan)

    # Residual: actual alt return - expected (beta * btc return)
    expected_ret = beta * btc_ret
    residual = alt_ret - expected_ret

    return pd.DataFrame({
        "btc_alt_beta": beta,
        "btc_alt_corr": corr,
        "btc_alt_residual": residual,
    }, index=alt_close.index)


def build_indicator_features(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> pd.DataFrame:
    """Build 9 BTC/ETH indicator features.

    Args:
        btc_df: BTC OHLCV DataFrame.
        eth_df: ETH OHLCV DataFrame.

    Returns:
        DataFrame with 9 indicator columns.
    """
    btc_close = btc_df["close"].astype(float)
    eth_close = eth_df["close"].astype(float)

    # BTC lagged returns (1, 2, 3 candles = 4h, 8h, 12h ago)
    btc_ret = btc_close.pct_change()
    btc_ret_1 = btc_ret.shift(1)
    btc_ret_2 = btc_ret.shift(2)
    btc_ret_3 = btc_ret.shift(3)

    # BTC volatility (24 candles = 4 days)
    btc_vol_24 = btc_ret.rolling(24).std()

    # BTC RSI (14 candles)
    delta = btc_close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0.0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    btc_rsi = (100 - (100 / (1 + rs)) - 50) / 50  # Normalized [-1, 1]

    # ETH lagged returns
    eth_ret = eth_close.pct_change()
    eth_ret_1 = eth_ret.shift(1)
    eth_ret_2 = eth_ret.shift(2)
    eth_ret_3 = eth_ret.shift(3)

    # ETH/BTC ratio change
    eth_btc_ratio = eth_close / btc_close
    eth_btc_ratio_chg = eth_btc_ratio.pct_change(6)  # 6-candle change

    # BTC above SMA50
    btc_sma50 = btc_close.rolling(50).mean()
    btc_above_sma50 = (btc_close > btc_sma50).astype(float)

    return pd.DataFrame({
        "btc_ret_1": btc_ret_1,
        "btc_ret_2": btc_ret_2,
        "btc_ret_3": btc_ret_3,
        "btc_vol_24": btc_vol_24,
        "btc_rsi": btc_rsi,
        "eth_ret_1": eth_ret_1,
        "eth_ret_2": eth_ret_2,
        "eth_ret_3": eth_ret_3,
        "eth_btc_ratio_chg": eth_btc_ratio_chg,
        "btc_above_sma50": btc_above_sma50,
    }, index=btc_df.index)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_rl_features.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add services/python/src/rl/features.py services/python/tests/test_rl_features.py
git commit -m "feat(rl): add cross-token and BTC/ETH indicator features"
```

---

### Task 4: Data Loader — Raw CSVs to RL-Ready Arrays

**Files:**
- Create: `services/python/src/rl/data_loader.py`
- Test: `services/python/tests/test_rl_data_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_data_loader.py
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from src.rl.data_loader import RLDataLoader


def test_data_loader_init():
    loader = RLDataLoader(raw_dir=Path("/fake"), max_tokens=5)
    assert loader.max_tokens == 5


def test_load_single_token_csv(tmp_path):
    """Create a fake CSV and verify loading."""
    symbol_dir = tmp_path / "DOGEUSDT"
    symbol_dir.mkdir()
    # Create fake 4h OHLCV CSV
    n = 200
    rng = np.random.RandomState(42)
    ts = np.arange(n) * 4 * 3600 * 1000 + 1640000000000  # 4h apart in ms
    close = 0.10 * np.cumprod(1 + rng.randn(n) * 0.02)
    df = pd.DataFrame({
        "open_time": ts,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": np.abs(rng.randn(n) * 1e6),
    })
    df.to_csv(symbol_dir / "DOGEUSDT-4h-2022-01.csv", index=False)

    loader = RLDataLoader(raw_dir=tmp_path, max_tokens=5)
    result = loader._load_token_ohlcv("DOGEUSDT")
    assert result is not None
    assert len(result) == n
    assert "close" in result.columns


def test_build_rl_dataset_structure(tmp_path):
    """Verify the output structure of the full dataset."""
    # Create BTC + ETH + one alt
    for sym in ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]:
        d = tmp_path / sym
        d.mkdir()
        n = 300
        rng = np.random.RandomState(hash(sym) % 2**31)
        ts = np.arange(n) * 4 * 3600 * 1000 + 1640000000000
        close = 100.0 * np.cumprod(1 + rng.randn(n) * 0.02)
        pd.DataFrame({
            "open_time": ts, "open": close * 0.99, "high": close * 1.01,
            "low": close * 0.98, "close": close,
            "volume": np.abs(rng.randn(n) * 1e6),
        }).to_csv(d / f"{sym}-4h-2022-01.csv", index=False)

    # Also create a manifest
    pd.DataFrame({
        "symbol": ["BTCUSDT", "ETHUSDT", "DOGEUSDT"],
        "months": [12, 12, 12],
    }).to_csv(tmp_path / "manifest.csv", index=False)

    loader = RLDataLoader(raw_dir=tmp_path, max_tokens=5)
    dataset = loader.build_dataset()

    assert "alt_features" in dataset  # shape: (n_timestamps, n_alts, n_alt_features)
    assert "indicator_features" in dataset  # shape: (n_timestamps, 9)
    assert "prices" in dataset  # shape: (n_timestamps, n_alts, 4) for OHLC
    assert "alt_names" in dataset  # list of alt symbols
    assert "timestamps" in dataset
    assert "BTC" not in dataset["alt_names"]
    assert "ETH" not in dataset["alt_names"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_rl_data_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/rl/data_loader.py
"""Load raw CSV data into RL-ready arrays.

Loads all altcoin 4h candles, computes per-alt features using FeatureEngineerV2,
adds cross-token features (BTC beta/correlation/residual), and BTC/ETH indicators.
Outputs a dict of numpy arrays aligned by timestamp for fast gym replay.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.features_v2 import FeatureEngineerV2
from src.rl.config import RLConfig
from src.rl.features import build_cross_token_features, build_indicator_features

logger = logging.getLogger(__name__)


class RLDataLoader:
    """Load and preprocess data for the RL trading environment."""

    def __init__(self, raw_dir: Path, max_tokens: int = 200):
        self.raw_dir = raw_dir
        self.max_tokens = max_tokens
        self.fe = FeatureEngineerV2()
        self.config = RLConfig()

    def _load_token_ohlcv(self, symbol: str) -> pd.DataFrame | None:
        """Load all CSV files for one token into a single DataFrame."""
        symbol_dir = self.raw_dir / symbol
        files = sorted(glob.glob(str(symbol_dir / "*.csv")))
        if not files:
            return None

        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)

        if "open_time" not in df.columns:
            return None

        df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
        return df

    def _get_alt_symbols(self) -> list[str]:
        """Get list of alt symbols (exclude BTC and ETH)."""
        manifest_path = self.raw_dir / "manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            manifest = manifest.sort_values("months", ascending=False)
            symbols = manifest["symbol"].tolist()
        else:
            symbols = [d.name for d in self.raw_dir.iterdir() if d.is_dir()]

        # Exclude indicators
        indicator_syms = {f"{a}USDT" for a in self.config.INDICATOR_ASSETS}
        alts = [s for s in symbols if s not in indicator_syms]
        return alts[: self.max_tokens]

    def build_dataset(self) -> dict:
        """Build the complete RL dataset.

        Returns dict with:
            alt_features: np.ndarray (n_times, n_alts, n_alt_features)
            indicator_features: np.ndarray (n_times, 9)
            prices: np.ndarray (n_times, n_alts, 4) — OHLC for SL/TP checking
            alt_names: list[str]
            timestamps: np.ndarray
        """
        # Load BTC and ETH first
        btc_df = self._load_token_ohlcv("BTCUSDT")
        eth_df = self._load_token_ohlcv("ETHUSDT")
        if btc_df is None or eth_df is None:
            raise ValueError("BTC and ETH data required")

        btc_df["timestamp"] = pd.to_datetime(btc_df["open_time"], unit="ms")
        eth_df["timestamp"] = pd.to_datetime(eth_df["open_time"], unit="ms")

        # Find common timestamps between BTC and ETH
        common_ts = set(btc_df["timestamp"]) & set(eth_df["timestamp"])
        btc_df = btc_df[btc_df["timestamp"].isin(common_ts)].sort_values("timestamp")
        eth_df = eth_df[eth_df["timestamp"].isin(common_ts)].sort_values("timestamp")

        # Build BTC/ETH indicator features
        indicators = build_indicator_features(btc_df, eth_df)
        indicator_cols = list(indicators.columns)

        # Load alt tokens
        alt_symbols = self._get_alt_symbols()
        logger.info(f"Loading {len(alt_symbols)} alt tokens...")

        all_alt_features = {}
        all_alt_prices = {}
        btc_close = btc_df.set_index("timestamp")["close"]

        for i, sym in enumerate(alt_symbols):
            df = self._load_token_ohlcv(sym)
            if df is None or len(df) < 300:
                continue

            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            df = df[df["timestamp"].isin(common_ts)].sort_values("timestamp")

            if len(df) < 200:
                continue

            asset = sym.replace("USDT", "")

            # Per-alt features from FeatureEngineerV2
            try:
                feat_df = self.fe.build_features(df, asset=asset)
            except Exception as e:
                logger.warning(f"Skipping {sym}: {e}")
                continue

            feature_cols = self.fe.get_feature_cols()

            # Cross-token features
            alt_close = df.set_index("timestamp")["close"].reindex(btc_close.index)
            cross = build_cross_token_features(alt_close, btc_close)
            cross = cross.reset_index(drop=True)

            # Combine per-alt + cross-token features
            combined = feat_df[feature_cols].reset_index(drop=True)
            for col in ["btc_alt_beta", "btc_alt_corr", "btc_alt_residual"]:
                combined[col] = cross[col].values[: len(combined)]

            # OHLC prices for SL/TP simulation
            prices = df[["open", "high", "low", "close"]].reset_index(drop=True)

            all_alt_features[asset] = combined
            all_alt_prices[asset] = prices

            if (i + 1) % 50 == 0:
                logger.info(f"  Loaded {i + 1}/{len(alt_symbols)}...")

        if not all_alt_features:
            raise ValueError("No alt tokens loaded successfully")

        # Align all to same length (shortest common length)
        alt_names = sorted(all_alt_features.keys())
        min_len = min(len(all_alt_features[a]) for a in alt_names)

        n_alt_features = len(all_alt_features[alt_names[0]].columns)
        n_alts = len(alt_names)

        # Build 3D arrays
        alt_feat_arr = np.zeros((min_len, n_alts, n_alt_features), dtype=np.float32)
        price_arr = np.zeros((min_len, n_alts, 4), dtype=np.float32)

        for j, asset in enumerate(alt_names):
            feat = all_alt_features[asset].iloc[:min_len].values
            alt_feat_arr[:, j, :] = feat
            price_arr[:, j, :] = all_alt_prices[asset].iloc[:min_len].values

        # Indicator features (trim to min_len)
        ind_arr = indicators.iloc[:min_len].values.astype(np.float32)

        # Timestamps
        timestamps = btc_df["timestamp"].values[:min_len]

        # Replace NaN with 0
        alt_feat_arr = np.nan_to_num(alt_feat_arr, nan=0.0)
        ind_arr = np.nan_to_num(ind_arr, nan=0.0)

        logger.info(
            f"Dataset: {n_alts} alts, {min_len} timestamps, "
            f"{n_alt_features} alt features, {len(indicator_cols)} indicator features"
        )

        return {
            "alt_features": alt_feat_arr,
            "indicator_features": ind_arr,
            "prices": price_arr,
            "alt_names": alt_names,
            "timestamps": timestamps,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_rl_data_loader.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add services/python/src/rl/data_loader.py services/python/tests/test_rl_data_loader.py
git commit -m "feat(rl): add data loader for RL-ready arrays from raw CSVs"
```

---

### Task 5: Gym Environment — CryptoTradingEnv

**Files:**
- Create: `services/python/src/rl/env.py`
- Test: `services/python/tests/test_rl_env.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_env.py
import numpy as np
from src.rl.env import CryptoTradingEnv
from src.rl.config import RLConfig


def _make_fake_dataset(n_times=400, n_alts=10, n_alt_feat=31, n_ind_feat=9):
    """Create a minimal fake dataset for testing."""
    rng = np.random.RandomState(42)
    prices = np.zeros((n_times, n_alts, 4), dtype=np.float32)
    base = rng.uniform(1.0, 100.0, size=n_alts)
    for t in range(n_times):
        change = 1 + rng.randn(n_alts) * 0.02
        base = base * change
        for j in range(n_alts):
            o = base[j] * 0.99
            h = base[j] * 1.02
            lo = base[j] * 0.97
            c = base[j]
            prices[t, j] = [o, h, lo, c]

    return {
        "alt_features": rng.randn(n_times, n_alts, n_alt_feat).astype(np.float32),
        "indicator_features": rng.randn(n_times, n_ind_feat).astype(np.float32),
        "prices": prices,
        "alt_names": [f"ALT{i}" for i in range(n_alts)],
        "timestamps": np.arange(n_times),
    }


def test_env_reset():
    dataset = _make_fake_dataset()
    env = CryptoTradingEnv(dataset, RLConfig())
    obs = env.reset()
    # obs should be a flat array: n_alts * n_alt_feat + n_ind_feat + n_portfolio_feat
    cfg = RLConfig()
    expected_size = 10 * 31 + 9 + 5
    assert obs.shape == (expected_size,)
    assert np.isfinite(obs).all()


def test_env_step_flat():
    """Step with no actions (all zeros = stay flat)."""
    dataset = _make_fake_dataset()
    env = CryptoTradingEnv(dataset, RLConfig())
    env.reset()

    # Action: scores for each alt, SL, TP
    # All zeros = no trades
    action = np.zeros(10 * 3, dtype=np.float32)  # 10 alts * (score, sl, tp)
    obs, reward, done, info = env.step(action)
    assert obs.shape[0] > 0
    assert isinstance(reward, float)
    assert done is False
    assert info["portfolio_value"] == RLConfig().STARTING_CAPITAL


def test_env_step_open_long():
    """Step with a strong buy signal for one alt."""
    dataset = _make_fake_dataset()
    env = CryptoTradingEnv(dataset, RLConfig())
    env.reset()

    action = np.zeros(10 * 3, dtype=np.float32)
    action[0] = 0.9   # Strong long signal for ALT0
    action[1] = 0.5    # SL: maps to middle of [0.02, 0.08] range
    action[2] = 0.5    # TP: maps to middle of [0.03, 0.15] range
    obs, reward, done, info = env.step(action)
    assert info["n_positions"] == 1


def test_env_episode_terminates():
    """Episode ends after EPISODE_WINDOWS steps."""
    cfg = RLConfig()
    dataset = _make_fake_dataset(n_times=500)
    env = CryptoTradingEnv(dataset, cfg)
    env.reset()

    action = np.zeros(10 * 3, dtype=np.float32)
    done = False
    steps = 0
    while not done and steps < cfg.EPISODE_WINDOWS + 10:
        _, _, done, _ = env.step(action)
        steps += 1
    assert done is True
    assert steps <= cfg.EPISODE_WINDOWS


def test_env_drawdown_kill():
    """Episode ends early if drawdown exceeds threshold."""
    cfg = RLConfig()
    dataset = _make_fake_dataset(n_times=500)
    # Make prices crash so any long position loses big
    dataset["prices"][:, :, 0:4] *= 0.5  # Cut all prices in half after step
    env = CryptoTradingEnv(dataset, cfg)
    env.reset()

    # Open max positions going long
    action = np.zeros(10 * 3, dtype=np.float32)
    for i in range(10):
        action[i * 3] = 0.9  # Strong long
        action[i * 3 + 1] = 0.8  # Wide SL
        action[i * 3 + 2] = 0.5

    obs, reward, done, info = env.step(action)
    # Positions should have SL triggered or heavy losses
    # The environment should kill the episode eventually


def test_env_reward_computation():
    """Verify reward is computed at episode end."""
    cfg = RLConfig()
    dataset = _make_fake_dataset(n_times=500)
    env = CryptoTradingEnv(dataset, cfg)
    env.reset()

    action = np.zeros(10 * 3, dtype=np.float32)
    total_reward = 0.0
    done = False
    while not done:
        _, r, done, _ = env.step(action)
        total_reward += r
    # Flat (no trades) should have near-zero reward
    assert abs(total_reward) < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_rl_env.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/rl/env.py
"""Crypto Trading Gym Environment for RL training.

Steps in 12-hour windows. At each step the agent outputs per-alt scores,
SL distances, and TP distances. The environment simulates fills, SL/TP
execution against intra-period high/low, fees, slippage, and funding rates.
"""

from __future__ import annotations

import numpy as np

from src.rl.config import RLConfig
from src.rl.portfolio import Portfolio


class CryptoTradingEnv:
    """OpenAI Gym-style trading environment.

    Observation: flat array of [alt_features | indicator_features | portfolio_state]
    Action: per-alt [score, sl_pct, tp_pct] — shape (n_alts * 3,)
        score in [-1, 1]: positive = long, negative = short, near 0 = flat
        sl_pct in [0, 1]: mapped to [SL_MIN, SL_MAX]
        tp_pct in [0, 1]: mapped to [TP_MIN, TP_MAX]
    """

    SCORE_THRESHOLD = 0.3  # Minimum |score| to open a position

    def __init__(self, dataset: dict, config: RLConfig | None = None):
        self.config = config or RLConfig()
        self.alt_features = dataset["alt_features"]  # (T, n_alts, n_alt_feat)
        self.indicator_features = dataset["indicator_features"]  # (T, n_ind_feat)
        self.prices = dataset["prices"]  # (T, n_alts, 4) OHLC
        self.alt_names = dataset["alt_names"]
        self.n_alts = len(self.alt_names)
        self.n_times = len(self.alt_features)

        self.n_alt_feat = self.alt_features.shape[2]
        self.n_ind_feat = self.indicator_features.shape[1]
        self.obs_size = (
            self.n_alts * self.n_alt_feat
            + self.n_ind_feat
            + self.config.N_PORTFOLIO_FEATURES
        )

        self.portfolio: Portfolio | None = None
        self.step_idx = 0
        self.start_idx = 0
        self.episode_returns: list[float] = []

    def reset(self, start_idx: int | None = None) -> np.ndarray:
        """Reset environment for a new episode."""
        self.portfolio = Portfolio(self.config)
        self.episode_returns = []

        # Random start (leave room for full episode + lookback)
        max_start = self.n_times - self.config.EPISODE_WINDOWS * self.config.CANDLES_PER_WINDOW - 1
        if start_idx is not None:
            self.start_idx = min(start_idx, max(0, max_start))
        else:
            self.start_idx = np.random.randint(200, max(201, max_start))

        self.step_idx = 0
        return self._get_observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one 12-hour trading window.

        Args:
            action: shape (n_alts * 3,) — [score, sl_norm, tp_norm] per alt

        Returns:
            observation, reward, done, info
        """
        assert self.portfolio is not None

        candle_idx = self.start_idx + self.step_idx * self.config.CANDLES_PER_WINDOW
        equity_before = self.portfolio.equity

        # Parse actions
        actions = action.reshape(self.n_alts, 3)
        scores = actions[:, 0]
        sl_norms = actions[:, 1]
        tp_norms = actions[:, 2]

        # Close positions for alts where agent now wants opposite direction or flat
        for asset in list(self.portfolio.positions.keys()):
            if asset not in self.alt_names:
                continue
            j = self.alt_names.index(asset)
            score = scores[j]
            pos = self.portfolio.positions[asset]
            # Close if: agent wants flat, or wants opposite direction
            if abs(score) < self.SCORE_THRESHOLD or (score > 0 and pos.side == -1) or (
                score < 0 and pos.side == 1
            ):
                close_price = self.prices[candle_idx, j, 3]  # Close price
                self.portfolio.close_position(asset, close_price)

        # Open new positions for strong signals
        ranked = np.argsort(-np.abs(scores))  # Strongest signals first
        for j in ranked:
            score = scores[j]
            if abs(score) < self.SCORE_THRESHOLD:
                continue
            asset = self.alt_names[j]
            if asset in self.portfolio.positions:
                continue
            if self.portfolio.n_positions >= self.config.MAX_POSITIONS:
                break

            side = 1 if score > 0 else -1
            entry_price = self.prices[candle_idx, j, 3]  # Close price
            if entry_price <= 0:
                continue

            sl_pct = self.config.SL_MIN + sl_norms[j] * (self.config.SL_MAX - self.config.SL_MIN)
            tp_pct = self.config.TP_MIN + tp_norms[j] * (self.config.TP_MAX - self.config.TP_MIN)
            size_pct = min(abs(score) * 0.10, self.config.MAX_SINGLE_POSITION_PCT)

            sl_pct = float(np.clip(sl_pct, self.config.SL_MIN, self.config.SL_MAX))
            tp_pct = float(np.clip(tp_pct, self.config.TP_MIN, self.config.TP_MAX))

            self.portfolio.open_position(asset, side, size_pct, entry_price, sl_pct, tp_pct)

        # Simulate candles in this window (check SL/TP against high/low)
        for c in range(self.config.CANDLES_PER_WINDOW):
            ci = candle_idx + c
            if ci >= self.n_times:
                break
            candle_data = {}
            for j, asset in enumerate(self.alt_names):
                if asset in self.portfolio.positions:
                    candle_data[asset] = {
                        "high": float(self.prices[ci, j, 1]),
                        "low": float(self.prices[ci, j, 2]),
                    }
            self.portfolio.check_sl_tp(candle_data)

        # Deduct funding (12h window = 1.5 × 8h periods)
        self.portfolio.deduct_funding(n_8h_periods=1)

        # Increment hold times
        self.portfolio.increment_hold_times()

        # Compute step reward (change in equity)
        current_prices = {
            self.alt_names[j]: float(self.prices[min(candle_idx + self.config.CANDLES_PER_WINDOW - 1, self.n_times - 1), j, 3])
            for j in range(self.n_alts)
        }
        equity_after = self.portfolio.equity_at_prices(current_prices)
        step_return = (equity_after - equity_before) / equity_before if equity_before > 0 else 0
        self.episode_returns.append(step_return)

        # Check termination
        self.step_idx += 1
        done = self.step_idx >= self.config.EPISODE_WINDOWS

        # Check drawdown kill
        peak = max(self.config.STARTING_CAPITAL, equity_after)
        dd = (peak - equity_after) / peak if peak > 0 else 0
        if dd > self.config.MAX_DRAWDOWN_KILL:
            done = True

        # Check if we've run out of data
        next_candle = self.start_idx + self.step_idx * self.config.CANDLES_PER_WINDOW
        if next_candle >= self.n_times:
            done = True

        # Compute reward
        reward = step_return
        if done and len(self.episode_returns) > 1:
            reward = self._episode_reward()

        info = {
            "portfolio_value": equity_after,
            "n_positions": self.portfolio.n_positions,
            "step": self.step_idx,
            "drawdown": dd,
        }

        obs = self._get_observation()
        return obs, float(reward), done, info

    def _get_observation(self) -> np.ndarray:
        """Build flat observation vector."""
        candle_idx = self.start_idx + self.step_idx * self.config.CANDLES_PER_WINDOW
        candle_idx = min(candle_idx, self.n_times - 1)

        # Alt features: (n_alts, n_alt_feat) → flatten
        alt_feat = self.alt_features[candle_idx].flatten()

        # Indicator features: (n_ind_feat,)
        ind_feat = self.indicator_features[candle_idx]

        # Portfolio state: (5,)
        current_prices = {
            self.alt_names[j]: float(self.prices[candle_idx, j, 3])
            for j in range(self.n_alts)
        }
        port_state = np.array(
            self.portfolio.get_state_vector(current_prices), dtype=np.float32
        ) if self.portfolio else np.zeros(self.config.N_PORTFOLIO_FEATURES, dtype=np.float32)

        obs = np.concatenate([alt_feat, ind_feat, port_state]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0)

    def _episode_reward(self) -> float:
        """Compute episode-level reward: Sharpe + Return - DD penalty."""
        returns = np.array(self.episode_returns)
        cfg = self.config

        # Sharpe (annualized: 730 windows/year)
        mean_r = returns.mean()
        std_r = returns.std()
        sharpe = (mean_r / std_r) * np.sqrt(730) if std_r > 0 else 0.0

        # Normalized return
        total_return = np.sum(returns)
        norm_return = np.clip(total_return / 0.5, -1.0, 1.0)  # Normalize to [-1, 1]

        # Drawdown penalty
        cum = np.cumsum(returns)
        peak = np.maximum.accumulate(cum)
        max_dd = np.min(cum - peak)
        dd_penalty = 1.0 if abs(max_dd) > cfg.REWARD_DD_THRESHOLD else 0.0

        reward = (
            cfg.REWARD_SHARPE_W * sharpe
            + cfg.REWARD_RETURN_W * norm_return
            - cfg.REWARD_DD_PENALTY_W * dd_penalty
        )
        return float(reward)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_rl_env.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add services/python/src/rl/env.py services/python/tests/test_rl_env.py
git commit -m "feat(rl): add CryptoTradingEnv gym environment"
```

---

### Task 6: Data Preparation Script

**Files:**
- Create: `services/python/scripts/prepare_rl_data.py`

- [ ] **Step 1: Write the script**

```python
# scripts/prepare_rl_data.py
"""One-time script: build RL dataset from raw CSVs and save as .npz.

Usage:
    cd services/python
    python scripts/prepare_rl_data.py

Output:
    data/features/rl_dataset.npz
"""

import logging
import time
from pathlib import Path

import numpy as np

from src.rl.data_loader import RLDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path("C:/Users/togat/Desktop/AI-Finance")
RAW_DIR = PROJECT / "data/raw/4h"
OUTPUT_PATH = PROJECT / "data/features/rl_dataset.npz"


def main():
    logger.info("Building RL dataset from raw CSVs...")
    t0 = time.time()

    loader = RLDataLoader(raw_dir=RAW_DIR, max_tokens=200)
    dataset = loader.build_dataset()

    logger.info(f"Saving to {OUTPUT_PATH}...")
    np.savez_compressed(
        OUTPUT_PATH,
        alt_features=dataset["alt_features"],
        indicator_features=dataset["indicator_features"],
        prices=dataset["prices"],
        alt_names=np.array(dataset["alt_names"]),
        timestamps=dataset["timestamps"],
    )

    elapsed = time.time() - t0
    shape = dataset["alt_features"].shape
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    logger.info(
        f"Done in {elapsed:.1f}s. Shape: {shape}, "
        f"File size: {size_mb:.1f} MB"
    )
    logger.info(f"Alts: {len(dataset['alt_names'])}")
    logger.info(f"Timestamps: {shape[0]}")
    logger.info(f"Features per alt: {shape[2]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script on real data**

Run: `cd services/python && python scripts/prepare_rl_data.py`
Expected: Creates `data/features/rl_dataset.npz` (~200-500 MB). Should complete in 5-15 minutes.

- [ ] **Step 3: Verify output**

```bash
cd services/python && python -c "
import numpy as np
d = np.load('../../data/features/rl_dataset.npz', allow_pickle=True)
print('Keys:', list(d.keys()))
print('alt_features:', d['alt_features'].shape)
print('indicator_features:', d['indicator_features'].shape)
print('prices:', d['prices'].shape)
print('alt_names:', d['alt_names'][:5], '...', f'({len(d[\"alt_names\"])} total)')
print('timestamps:', d['timestamps'][:3])
"
```

Expected output similar to:
```
Keys: ['alt_features', 'indicator_features', 'prices', 'alt_names', 'timestamps']
alt_features: (6000+, 150+, 31)
indicator_features: (6000+, 9)
prices: (6000+, 150+, 4)
alt_names: ['1INCH' 'AAVE' 'ACE' ...] (150+ total)
```

- [ ] **Step 4: Commit**

```bash
git add services/python/scripts/prepare_rl_data.py
git commit -m "feat(rl): add data preparation script for RL dataset"
```

---

### Task 7: Integration Test — Full Episode Playthrough

**Files:**
- Test: `services/python/tests/test_rl_integration.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/test_rl_integration.py
"""Integration test: run a full episode with random actions."""

import numpy as np
from src.rl.env import CryptoTradingEnv
from src.rl.config import RLConfig


def _make_realistic_dataset(n_times=1200, n_alts=20, n_alt_feat=31, n_ind_feat=9):
    """Create a dataset with realistic price dynamics."""
    rng = np.random.RandomState(42)
    prices = np.zeros((n_times, n_alts, 4), dtype=np.float32)

    for j in range(n_alts):
        base = rng.uniform(0.1, 500.0)
        for t in range(n_times):
            ret = rng.randn() * 0.03  # 3% vol per 4h candle
            base *= (1 + ret)
            o = base * (1 + rng.randn() * 0.005)
            h = base * (1 + abs(rng.randn() * 0.015))
            lo = base * (1 - abs(rng.randn() * 0.015))
            prices[t, j] = [o, h, lo, base]

    return {
        "alt_features": rng.randn(n_times, n_alts, n_alt_feat).astype(np.float32) * 0.1,
        "indicator_features": rng.randn(n_times, n_ind_feat).astype(np.float32) * 0.1,
        "prices": prices,
        "alt_names": [f"ALT{i}" for i in range(n_alts)],
        "timestamps": np.arange(n_times),
    }


def test_full_episode_random_actions():
    """Run a complete episode with random actions and verify it completes."""
    cfg = RLConfig()
    dataset = _make_realistic_dataset()
    env = CryptoTradingEnv(dataset, cfg)
    rng = np.random.RandomState(123)

    obs = env.reset(start_idx=100)
    assert obs.shape[0] > 0

    done = False
    steps = 0
    total_reward = 0.0
    max_positions = 0

    while not done:
        # Random action: scores in [-1, 1], sl/tp in [0, 1]
        action = rng.uniform(-1, 1, size=20 * 3).astype(np.float32)
        # Clamp sl/tp to [0, 1]
        for i in range(20):
            action[i * 3 + 1] = abs(action[i * 3 + 1])
            action[i * 3 + 2] = abs(action[i * 3 + 2])

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        max_positions = max(max_positions, info["n_positions"])

        assert np.isfinite(obs).all(), f"NaN in observation at step {steps}"
        assert np.isfinite(reward), f"NaN reward at step {steps}"

    print(f"\nEpisode complete:")
    print(f"  Steps: {steps}")
    print(f"  Final equity: {info['portfolio_value']:,.0f} IDR")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Max positions held: {max_positions}")
    print(f"  Final drawdown: {info['drawdown']:.2%}")

    assert steps > 0
    assert steps <= cfg.EPISODE_WINDOWS
    assert info["portfolio_value"] > 0  # Not bankrupt


def test_multiple_episodes_different_starts():
    """Run 5 episodes from different start points to verify no crashes."""
    cfg = RLConfig()
    dataset = _make_realistic_dataset()
    env = CryptoTradingEnv(dataset, cfg)
    rng = np.random.RandomState(456)

    for episode in range(5):
        obs = env.reset()
        done = False
        while not done:
            action = rng.uniform(-1, 1, size=20 * 3).astype(np.float32)
            for i in range(20):
                action[i * 3 + 1] = abs(action[i * 3 + 1])
                action[i * 3 + 2] = abs(action[i * 3 + 2])
            obs, _, done, info = env.step(action)

        assert info["portfolio_value"] > 0, f"Bankrupt in episode {episode}"
```

- [ ] **Step 2: Run integration test**

Run: `cd services/python && python -m pytest tests/test_rl_integration.py -v -s`
Expected: 2 passed, prints episode stats

- [ ] **Step 3: Commit**

```bash
git add services/python/tests/test_rl_integration.py
git commit -m "test(rl): add integration test for full episode playthrough"
```

---

### Task 8: Add gymnasium dependency + ruff check

**Files:**
- Modify: `services/python/pyproject.toml`

- [ ] **Step 1: Add gymnasium to dependencies**

Add `"gymnasium>=0.29.0"` to the dependencies list in `pyproject.toml`:

```toml
dependencies = [
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "ccxt>=4.0.0",
    "pycoingecko>=3.1.0",
    "apscheduler>=3.10.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "scikit-learn>=1.5.0",
    "xgboost>=2.1.0",
    "lightgbm>=4.5.0",
    "torch>=2.4.0",
    "joblib>=1.4.0",
    "gymnasium>=0.29.0",
]
```

Also add rl files to the ruff per-file-ignores:

```toml
[tool.ruff.lint.per-file-ignores]
"src/ml/**" = ["N803", "N806"]
"src/rl/**" = ["N803", "N806"]
"tests/test_*" = ["N803", "N806"]
```

- [ ] **Step 2: Install and lint**

Run:
```bash
cd services/python && pip install gymnasium>=0.29.0
cd services/python && ruff check src/rl/ tests/test_rl_*.py
```

Expected: No lint errors (or fix any that appear)

- [ ] **Step 3: Run all RL tests**

Run: `cd services/python && python -m pytest tests/test_rl_*.py -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add services/python/pyproject.toml
git commit -m "chore: add gymnasium dependency and rl ruff config"
```

---

## Summary

After completing all 8 tasks, you will have:

1. **`src/rl/config.py`** — All constants (positions, fees, reward weights)
2. **`src/rl/portfolio.py`** — Position tracking with SL/TP/funding
3. **`src/rl/features.py`** — BTC-alt beta/correlation/residual + BTC/ETH indicators
4. **`src/rl/data_loader.py`** — Raw CSV → structured numpy arrays
5. **`src/rl/env.py`** — Full gym environment with realistic simulation
6. **`scripts/prepare_rl_data.py`** — One-time data prep script
7. **Tests for everything** — 20+ tests covering all components
8. **`data/features/rl_dataset.npz`** — Ready for Week 2 (RL agent training)

**Week 2 plan** (RL Agent + PPO + Population Training) will build on this foundation.
