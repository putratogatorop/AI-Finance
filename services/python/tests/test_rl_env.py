"""Tests for CryptoTradingEnv."""

import numpy as np

from src.rl.config import RLConfig
from src.rl.env import CryptoTradingEnv


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _make_fake_dataset(
    n_times: int = 400,
    n_alts: int = 10,
    n_alt_feat: int = 31,
    n_ind_feat: int = 9,
    seed: int = 42,
) -> dict:
    """Generate a fake dataset compatible with CryptoTradingEnv."""
    rng = np.random.RandomState(seed)

    alt_features = rng.randn(n_times, n_alts, n_alt_feat).astype(np.float32)
    indicator_features = rng.randn(n_times, n_ind_feat).astype(np.float32)

    # Realistic prices: random walk per alt
    prices = np.zeros((n_times, n_alts, 4), dtype=np.float64)
    for a in range(n_alts):
        base = rng.uniform(100, 50000)
        close = base
        for t in range(n_times):
            change = rng.normal(0, 0.02) * close
            close = max(close + change, 1.0)
            high = close * (1 + abs(rng.normal(0, 0.005)))
            low = close * (1 - abs(rng.normal(0, 0.005)))
            opn = low + rng.random() * (high - low)
            prices[t, a] = [opn, high, low, close]

    alt_names = [f"ALT{i}" for i in range(n_alts)]
    timestamps = list(range(n_times))

    return {
        "alt_features": alt_features,
        "indicator_features": indicator_features,
        "prices": prices,
        "alt_names": alt_names,
        "timestamps": timestamps,
    }


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
class TestCryptoTradingEnv:
    def test_env_reset(self):
        """Obs shape = n_alts * n_alt_feat + n_ind_feat + 5, all finite."""
        ds = _make_fake_dataset()
        env = CryptoTradingEnv(ds)
        obs = env.reset(start_idx=200)

        expected_size = 10 * 31 + 9 + 5
        assert obs.shape == (expected_size,)
        assert env.obs_size == expected_size
        assert np.all(np.isfinite(obs))

    def test_env_step_flat(self):
        """Step with all-zero action => done=False, portfolio unchanged."""
        ds = _make_fake_dataset()
        env = CryptoTradingEnv(ds)
        env.reset(start_idx=200)

        action = np.zeros(10 * 3)
        obs, reward, done, info = env.step(action)

        assert done is False
        assert info["n_positions"] == 0
        assert info["step"] == 1
        assert obs.shape == (env.obs_size,)

    def test_env_step_open_long(self):
        """Step with score=0.9 for first alt => n_positions >= 1."""
        ds = _make_fake_dataset()
        env = CryptoTradingEnv(ds)
        env.reset(start_idx=200)

        action = np.zeros(10 * 3)
        # First alt: score=0.9, sl_norm=0.5, tp_norm=0.5
        action[0] = 0.9
        action[1] = 0.5
        action[2] = 0.5
        obs, reward, done, info = env.step(action)

        assert done is False
        assert info["n_positions"] >= 1

    def test_env_episode_terminates(self):
        """Run until done, verify steps <= EPISODE_WINDOWS."""
        ds = _make_fake_dataset(n_times=2000)
        cfg = RLConfig(EPISODE_WINDOWS=20)
        env = CryptoTradingEnv(ds, config=cfg)
        env.reset(start_idx=200)

        steps = 0
        done = False
        while not done:
            action = np.zeros(10 * 3)
            _, _, done, info = env.step(action)
            steps += 1
            if steps > cfg.EPISODE_WINDOWS + 5:
                break

        assert done is True
        assert steps <= cfg.EPISODE_WINDOWS

    def test_env_reward_computation(self):
        """Flat episode => total reward near zero."""
        ds = _make_fake_dataset(n_times=2000)
        cfg = RLConfig(EPISODE_WINDOWS=10)
        env = CryptoTradingEnv(ds, config=cfg)
        env.reset(start_idx=200)

        total_reward = 0.0
        done = False
        while not done:
            action = np.zeros(10 * 3)
            _, reward, done, _ = env.step(action)
            total_reward += reward

        # No positions => returns are only funding costs (tiny)
        # Episode reward should be close to zero
        assert abs(total_reward) < 5.0


class TestTopNFiltering:
    def test_top_n_reduces_alts(self):
        """top_n=5 reduces n_alts from 10 to 5."""
        ds = _make_fake_dataset(n_alts=10)
        env = CryptoTradingEnv(ds, top_n=5)
        assert env.n_alts == 5

    def test_top_n_selects_by_volume(self):
        """top_n picks alts with highest total close price volume."""
        ds = _make_fake_dataset(n_alts=10)
        ds["prices"][:, 0, :] *= 1000
        env = CryptoTradingEnv(ds, top_n=3)
        assert "ALT0" in env.alt_names

    def test_top_n_none_keeps_all(self):
        """top_n=None keeps all alts."""
        ds = _make_fake_dataset(n_alts=10)
        env = CryptoTradingEnv(ds, top_n=None)
        assert env.n_alts == 10

    def test_top_n_obs_size_correct(self):
        """Obs size adjusts for fewer alts."""
        ds = _make_fake_dataset(n_alts=10, n_alt_feat=28)
        env = CryptoTradingEnv(ds, top_n=5)
        obs = env.reset(start_idx=200)
        expected = 5 * 28 + 9 + 5
        assert obs.shape == (expected,)
        assert env.obs_size == expected
