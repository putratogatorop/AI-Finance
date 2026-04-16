"""Integration test: run a full episode with random actions."""
import numpy as np

from src.rl.agent import LSTMPPOAgent
from src.rl.config import RLConfig
from src.rl.env import CryptoTradingEnv


def _make_realistic_dataset(n_times=1200, n_alts=20, n_alt_feat=31, n_ind_feat=9):
    rng = np.random.RandomState(42)
    prices = np.zeros((n_times, n_alts, 4), dtype=np.float32)
    for j in range(n_alts):
        base = rng.uniform(0.1, 500.0)
        for t in range(n_times):
            ret = rng.randn() * 0.03
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
        action = rng.uniform(-1, 1, size=20 * 3).astype(np.float32)
        for i in range(20):
            action[i * 3 + 1] = abs(action[i * 3 + 1])
            action[i * 3 + 2] = abs(action[i * 3 + 2])
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        max_positions = max(max_positions, info["n_positions"])
        assert np.isfinite(obs).all(), f"NaN in obs at step {steps}"
        assert np.isfinite(reward), f"NaN reward at step {steps}"

    assert steps > 0
    assert steps <= cfg.EPISODE_WINDOWS
    assert info["portfolio_value"] > 0


def test_multiple_episodes_different_starts():
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


def test_full_episode_top_n():
    """Full episode with top_n=5 filtering and PPO update."""
    ds = _make_realistic_dataset(n_times=2000, n_alts=20, n_alt_feat=28)
    cfg = RLConfig(EPISODE_WINDOWS=10)
    env = CryptoTradingEnv(ds, config=cfg, top_n=5)

    agent = LSTMPPOAgent(obs_size=env.obs_size, n_alts=env.n_alts)

    obs = env.reset(start_idx=200)
    agent.reset_hidden()
    done = False
    total_reward = 0.0

    while not done:
        action, log_prob, value = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    assert done is True
    assert info["step"] <= cfg.EPISODE_WINDOWS
    assert isinstance(total_reward, (float, np.floating))
