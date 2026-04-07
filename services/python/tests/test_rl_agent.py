"""Tests for LSTMPPOAgent actor-critic network."""

import numpy as np
import pytest
import torch

from src.rl.agent import LSTMPPOAgent

OBS_SIZE = 324
N_ALTS = 10
ACTION_SIZE = N_ALTS * 3


@pytest.fixture()
def agent() -> LSTMPPOAgent:
    """Create a fresh agent for each test."""
    torch.manual_seed(42)
    return LSTMPPOAgent(obs_size=OBS_SIZE, n_alts=N_ALTS)


# ------------------------------------------------------------------
def test_agent_init(agent: LSTMPPOAgent) -> None:
    """Verify obs_size and n_alts are stored correctly."""
    assert agent.obs_size == OBS_SIZE
    assert agent.n_alts == N_ALTS
    assert agent.action_size == ACTION_SIZE


def test_agent_act_shape(agent: LSTMPPOAgent) -> None:
    """Action shape = (n_alts*3,), log_prob is float, value is float."""
    obs = np.random.default_rng(0).standard_normal(OBS_SIZE).astype(np.float32)
    action, log_prob, value = agent.act(obs)

    assert action.shape == (ACTION_SIZE,)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_agent_act_deterministic(agent: LSTMPPOAgent) -> None:
    """Same obs with deterministic=True gives identical actions."""
    agent.eval()  # disable dropout for deterministic forward pass
    obs = np.random.default_rng(1).standard_normal(OBS_SIZE).astype(np.float32)

    agent.reset_hidden()
    a1, _, _ = agent.act(obs, deterministic=True)

    agent.reset_hidden()
    a2, _, _ = agent.act(obs, deterministic=True)

    np.testing.assert_array_equal(a1, a2)


def test_agent_act_stochastic_varies(agent: LSTMPPOAgent) -> None:
    """Different seeds produce different stochastic actions."""
    obs = np.random.default_rng(2).standard_normal(OBS_SIZE).astype(np.float32)

    torch.manual_seed(100)
    agent.reset_hidden()
    a1, _, _ = agent.act(obs, deterministic=False)

    torch.manual_seed(999)
    agent.reset_hidden()
    a2, _, _ = agent.act(obs, deterministic=False)

    assert not np.array_equal(a1, a2)


def test_agent_evaluate_batch(agent: LSTMPPOAgent) -> None:
    """Batch of 8 observations — verify output shapes."""
    batch = 8
    obs_batch = torch.randn(batch, OBS_SIZE)
    act_batch = torch.randn(batch, ACTION_SIZE)

    log_probs, values, entropy = agent.evaluate(obs_batch, act_batch)

    assert log_probs.shape == (batch,)
    assert values.shape == (batch,)
    assert entropy.dim() == 0  # scalar


def test_agent_hidden_state_reset(agent: LSTMPPOAgent) -> None:
    """Verify reset_hidden sets hidden to None."""
    obs = np.random.default_rng(3).standard_normal(OBS_SIZE).astype(np.float32)
    agent.act(obs)
    assert agent.hidden is not None

    agent.reset_hidden()
    assert agent.hidden is None


def test_agent_parameter_count(agent: LSTMPPOAgent) -> None:
    """Total parameters between 10K and 5M."""
    total = sum(p.numel() for p in agent.parameters())
    assert 10_000 < total < 5_000_000, f"param count {total} out of range"
