"""Tests for population-based training."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.rl.population import PopulationTrainer


def _make_fake_dataset(
    n_times: int = 1200,
    n_alts: int = 10,
    n_alt_feat: int = 31,
    n_ind_feat: int = 9,
) -> dict:
    """Create a fake dataset with realistic random-walk prices."""
    rng = np.random.default_rng(42)

    # Random-walk prices per alt: start around 100, random log-normal steps
    prices = np.zeros((n_times, n_alts, 4), dtype=np.float64)
    for a in range(n_alts):
        log_returns = rng.normal(0.0, 0.01, size=n_times)
        close = 100.0 * np.exp(np.cumsum(log_returns))
        for t in range(n_times):
            noise = rng.uniform(0.005, 0.02)
            o = close[t] * (1 + rng.uniform(-noise, noise))
            h = max(o, close[t]) * (1 + rng.uniform(0, noise))
            low = min(o, close[t]) * (1 - rng.uniform(0, noise))
            prices[t, a] = [o, h, low, close[t]]

    alt_features = rng.standard_normal((n_times, n_alts, n_alt_feat))
    indicator_features = rng.standard_normal((n_times, n_ind_feat))
    alt_names = [f"ALT{i}" for i in range(n_alts)]
    timestamps = list(range(n_times))

    return {
        "alt_features": alt_features,
        "indicator_features": indicator_features,
        "prices": prices,
        "alt_names": alt_names,
        "timestamps": timestamps,
    }


# Use small population and few episodes for fast tests
POP_SIZE = 3
EPISODES = 1


def test_population_init():
    """Verify agents and scores have correct length after init."""
    ds = _make_fake_dataset()
    trainer = PopulationTrainer(
        ds, population_size=POP_SIZE, episodes_per_agent=EPISODES
    )
    assert len(trainer.agents) == POP_SIZE
    assert len(trainer.scores) == POP_SIZE


def test_evaluate_agent():
    """evaluate_agent should return a finite float."""
    ds = _make_fake_dataset()
    trainer = PopulationTrainer(
        ds, population_size=POP_SIZE, episodes_per_agent=EPISODES
    )
    reward = trainer.evaluate_agent(trainer.agents[0], n_episodes=1)
    assert isinstance(reward, float)
    assert np.isfinite(reward)


def test_evolve_population():
    """After evolve, population size should be preserved."""
    ds = _make_fake_dataset()
    trainer = PopulationTrainer(
        ds, population_size=5, episodes_per_agent=EPISODES
    )
    # Assign fake scores so evolve has something to rank
    trainer.scores = [1.0, 5.0, 2.0, 0.5, 3.0]
    trainer.evolve()
    assert len(trainer.agents) == 5
    assert len(trainer.scores) == 5


def test_run_generation():
    """run_generation should return a stats dict with expected keys."""
    ds = _make_fake_dataset()
    trainer = PopulationTrainer(
        ds, population_size=POP_SIZE, episodes_per_agent=EPISODES
    )
    stats = trainer.run_generation()
    expected_keys = {
        "generation", "best_reward", "mean_reward", "worst_reward", "elapsed_s"
    }
    assert expected_keys == set(stats.keys())
    assert stats["generation"] == 1


def test_train_loop_checkpoint(tmp_path: Path):
    """Train 2 generations with checkpoint dir, verify checkpoint exists."""
    ds = _make_fake_dataset()
    ckpt_dir = tmp_path / "ckpt"
    trainer = PopulationTrainer(
        ds,
        population_size=POP_SIZE,
        episodes_per_agent=EPISODES,
        checkpoint_dir=ckpt_dir,
    )
    stats_list = trainer.train(n_generations=2)
    assert len(stats_list) == 2
    assert (ckpt_dir / "metadata.json").exists()
    assert (ckpt_dir / "scores.json").exists()
    assert (ckpt_dir / "agent_0.pt").exists()
