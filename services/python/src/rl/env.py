"""OpenAI Gym-style crypto trading environment stepping in 12-hour windows."""

from __future__ import annotations

import numpy as np

from src.rl.config import RLConfig
from src.rl.portfolio import Portfolio

SCORE_THRESHOLD = 0.3


class CryptoTradingEnv:
    """Gym-style environment for crypto trading with 12-hour decision windows."""

    def __init__(
        self,
        dataset: dict,
        config: RLConfig | None = None,
    ) -> None:
        self.cfg = config or RLConfig()
        self.dataset = dataset

        # Unpack dataset
        self.alt_features: np.ndarray = dataset["alt_features"]      # (T, n_alts, n_alt_feat)
        self.ind_features: np.ndarray = dataset["indicator_features"]  # (T, n_ind_feat)
        self.prices: np.ndarray = dataset["prices"]                   # (T, n_alts, 4)
        self.alt_names: list[str] = dataset["alt_names"]
        self.timestamps = dataset["timestamps"]

        # Derive dimensions
        self.n_times = self.alt_features.shape[0]
        self.n_alts = self.alt_features.shape[1]
        self.n_alt_feat = self.alt_features.shape[2]
        self.n_ind_feat = self.ind_features.shape[1]
        self.obs_size = (
            self.n_alts * self.n_alt_feat
            + self.n_ind_feat
            + self.cfg.N_PORTFOLIO_FEATURES
        )

        # State (set by reset)
        self.portfolio: Portfolio | None = None
        self.start_idx: int = 0
        self.step_idx: int = 0
        self.episode_returns: list[float] = []
        self.peak_equity: float = 0.0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self, start_idx: int | None = None) -> np.ndarray:
        """Reset env and return initial observation."""
        self.portfolio = Portfolio(self.cfg)
        self.step_idx = 0
        self.episode_returns = []
        self.peak_equity = float(self.cfg.STARTING_CAPITAL)

        max_safe = (
            self.n_times
            - self.cfg.EPISODE_WINDOWS * self.cfg.CANDLES_PER_WINDOW
            - 1
        )
        if start_idx is not None:
            self.start_idx = start_idx
        else:
            self.start_idx = np.random.randint(200, max(201, max_safe))

        return self._get_observation()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one 12-hour window. Returns (obs, reward, done, info)."""
        assert self.portfolio is not None, "Call reset() before step()."

        action_2d = action.reshape(self.n_alts, 3)
        scores = action_2d[:, 0]
        sl_norms = action_2d[:, 1]
        tp_norms = action_2d[:, 2]

        candle_idx = self.start_idx + self.step_idx * self.cfg.CANDLES_PER_WINDOW

        # Current close prices
        close_prices = self.prices[candle_idx, :, 3]  # OHLC -> close
        current_prices = {
            self.alt_names[i]: float(close_prices[i])
            for i in range(self.n_alts)
        }

        equity_before = self.portfolio.equity_at_prices(current_prices)

        # --- 1) Close positions where agent wants opposite dir or flat ---
        assets_to_close = []
        for asset, pos in self.portfolio.positions.items():
            idx = self.alt_names.index(asset)
            score = scores[idx]
            if abs(score) < SCORE_THRESHOLD:
                # Agent wants flat -> close
                assets_to_close.append(asset)
            else:
                desired_side = 1 if score > 0 else -1
                if desired_side != pos.side:
                    assets_to_close.append(asset)

        for asset in assets_to_close:
            idx = self.alt_names.index(asset)
            exit_price = float(close_prices[idx])
            if exit_price > 0:
                self.portfolio.close_position(asset, exit_price)

        # --- 2) Open new positions ranked by |score| descending ---
        ranked = np.argsort(-np.abs(scores))
        for i in ranked:
            score = scores[i]
            if abs(score) < SCORE_THRESHOLD:
                continue
            asset = self.alt_names[i]
            if asset in self.portfolio.positions:
                continue
            if self.portfolio.n_positions >= self.cfg.MAX_POSITIONS:
                break

            side = 1 if score > 0 else -1
            entry_price = float(close_prices[i])
            if entry_price <= 0:
                continue

            size_pct = min(
                abs(score) * 0.10, self.cfg.MAX_SINGLE_POSITION_PCT
            )
            sl_pct = self.cfg.SL_MIN + sl_norms[i] * (
                self.cfg.SL_MAX - self.cfg.SL_MIN
            )
            tp_pct = self.cfg.TP_MIN + tp_norms[i] * (
                self.cfg.TP_MAX - self.cfg.TP_MIN
            )

            self.portfolio.open_position(
                asset=asset,
                side=side,
                size_pct=size_pct,
                entry_price=entry_price,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
            )

        # --- 3) Simulate CANDLES_PER_WINDOW candles (SL/TP checks) ---
        for c in range(self.cfg.CANDLES_PER_WINDOW):
            ci = candle_idx + c
            if ci >= self.n_times:
                break
            candle_data: dict[str, dict[str, float]] = {}
            for j, name in enumerate(self.alt_names):
                candle_data[name] = {
                    "open": float(self.prices[ci, j, 0]),
                    "high": float(self.prices[ci, j, 1]),
                    "low": float(self.prices[ci, j, 2]),
                    "close": float(self.prices[ci, j, 3]),
                }
            self.portfolio.check_sl_tp(candle_data)

        # --- 4) Deduct funding (1 period per window) ---
        self.portfolio.deduct_funding(n_8h_periods=1)

        # --- 5) Increment hold times ---
        self.portfolio.increment_hold_times()

        # --- 6) Compute step return ---
        # Recompute close prices at end of window
        end_ci = min(
            candle_idx + self.cfg.CANDLES_PER_WINDOW - 1,
            self.n_times - 1,
        )
        end_close = self.prices[end_ci, :, 3]
        end_prices = {
            self.alt_names[i]: float(end_close[i])
            for i in range(self.n_alts)
        }
        equity_after = self.portfolio.equity_at_prices(end_prices)
        step_return = (
            (equity_after - equity_before) / equity_before
            if equity_before > 0
            else 0.0
        )
        self.episode_returns.append(step_return)

        # --- 7) Advance step ---
        self.step_idx += 1

        # --- 8) Check done conditions ---
        next_candle = (
            self.start_idx + self.step_idx * self.cfg.CANDLES_PER_WINDOW
        )
        self.peak_equity = max(self.peak_equity, equity_after)
        drawdown = (
            (self.peak_equity - equity_after) / self.peak_equity
            if self.peak_equity > 0 else 0.0
        )

        done = False
        if self.step_idx >= self.cfg.EPISODE_WINDOWS:
            done = True
        elif drawdown > self.cfg.MAX_DRAWDOWN_KILL:
            done = True
        elif next_candle + self.cfg.CANDLES_PER_WINDOW > self.n_times:
            done = True

        # --- 9) Reward: dense per-step + episode bonus ---
        # Per-step: scaled PnL change (gives PPO signal every step)
        reward = step_return * 10.0
        # Penalize large drawdown per step
        if drawdown > self.cfg.REWARD_DD_THRESHOLD:
            reward -= 0.1
        # Episode-end bonus
        if done:
            reward += self._episode_reward()

        # --- 10) Info ---
        info = {
            "portfolio_value": equity_after,
            "n_positions": self.portfolio.n_positions,
            "step": self.step_idx,
            "drawdown": drawdown,
        }

        obs = self._get_observation() if not done else np.zeros(self.obs_size)
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        """Build flat observation vector."""
        candle_idx = self.start_idx + self.step_idx * self.cfg.CANDLES_PER_WINDOW
        candle_idx = min(candle_idx, self.n_times - 1)

        alt_flat = self.alt_features[candle_idx].flatten()
        ind_flat = self.ind_features[candle_idx]

        close_prices = self.prices[candle_idx, :, 3]
        current_prices = {
            self.alt_names[i]: float(close_prices[i])
            for i in range(self.n_alts)
        }
        port_vec = np.array(
            self.portfolio.get_state_vector(current_prices), dtype=np.float64
        )

        obs = np.concatenate([alt_flat, ind_flat, port_vec])
        obs = np.nan_to_num(obs, nan=0.0)
        return obs

    def _episode_reward(self) -> float:
        """Compute shaped episode reward from returns history."""
        if not self.episode_returns:
            return 0.0

        returns = np.array(self.episode_returns)

        # Sharpe
        mean_r = returns.mean()
        std_r = returns.std()
        sharpe = (mean_r / std_r) * np.sqrt(730) if std_r > 0 else 0.0

        # Normalized return
        norm_return = np.clip(returns.sum() / 0.5, -1.0, 1.0)

        # Drawdown penalty
        cum = np.cumsum(returns)
        running_max = np.maximum.accumulate(cum)
        max_dd = np.min(cum - running_max)
        dd_penalty = 1.0 if abs(max_dd) > self.cfg.REWARD_DD_THRESHOLD else 0.0

        reward = (
            self.cfg.REWARD_SHARPE_W * sharpe
            + self.cfg.REWARD_RETURN_W * norm_return
            - self.cfg.REWARD_DD_PENALTY_W * dd_penalty
        )
        return float(reward)
