"""Tests for PPO buffer and update logic."""

from __future__ import annotations

import torch

from src.rl.agent import LSTMPPOAgent
from src.rl.ppo import PPOBuffer, ppo_update

OBS_SIZE = 100
N_ALTS = 3
ACTION_SIZE = N_ALTS * 3


def _make_agent() -> LSTMPPOAgent:
    return LSTMPPOAgent(obs_size=OBS_SIZE, n_alts=N_ALTS)


def _random_transition(done: bool = False):
    obs = torch.randn(OBS_SIZE)
    action = torch.randn(ACTION_SIZE)
    log_prob = -1.5
    reward = 1.0
    value = 0.5
    return obs, action, log_prob, reward, value, done


def test_buffer_add_and_get():
    buf = PPOBuffer()
    for i in range(6):
        done = i == 5
        buf.add(*_random_transition(done=done))

    batch = buf.get()
    assert batch["obs"].shape == (6, OBS_SIZE)
    assert batch["actions"].shape == (6, ACTION_SIZE)
    assert batch["old_log_probs"].shape == (6,)
    assert batch["returns"].shape == (6,)
    assert batch["advantages"].shape == (6,)


def test_buffer_gae_computation():
    buf = PPOBuffer()
    for i in range(10):
        obs = torch.randn(OBS_SIZE)
        action = torch.randn(ACTION_SIZE)
        reward = float(i % 3)
        value = 0.5 * reward
        done = i == 9
        buf.add(obs, action, -1.0, reward, value, done)

    batch = buf.get()
    adv = batch["advantages"]
    # After normalization, mean should be near 0
    assert abs(adv.mean().item()) < 1e-5


def test_ppo_update_runs():
    agent = _make_agent()
    buf = PPOBuffer()
    for i in range(20):
        obs = torch.randn(OBS_SIZE)
        action = torch.randn(ACTION_SIZE)
        log_prob = -1.0
        reward = 1.0
        value = 0.5
        done = i == 19
        buf.add(obs, action, log_prob, reward, value, done)

    batch = buf.get()
    result = ppo_update(agent, batch, epochs=2, batch_size=10)

    assert "policy_loss" in result
    assert "value_loss" in result
    assert "entropy" in result
    assert isinstance(result["policy_loss"], float)
    assert isinstance(result["value_loss"], float)
    assert isinstance(result["entropy"], float)


def test_ppo_update_improves():
    agent = _make_agent()

    # Fill buffer: constant reward=1, value=0 => large value error initially
    buf = PPOBuffer()
    for i in range(50):
        obs = torch.randn(OBS_SIZE)
        action = torch.randn(ACTION_SIZE)
        done = i == 49
        buf.add(obs, action, -1.0, 1.0, 0.0, done)

    batch = buf.get()

    # Measure value loss before training
    with torch.no_grad():
        _, values_before, _ = agent.evaluate(batch["obs"], batch["actions"])
        loss_before = torch.nn.functional.mse_loss(
            values_before, batch["returns"]
        ).item()

    ppo_update(agent, batch, epochs=10, batch_size=25, lr=1e-3)

    # Measure value loss after training
    with torch.no_grad():
        _, values_after, _ = agent.evaluate(batch["obs"], batch["actions"])
        loss_after = torch.nn.functional.mse_loss(
            values_after, batch["returns"]
        ).item()

    assert loss_after < loss_before, (
        f"Value loss should decrease: {loss_before:.4f} -> {loss_after:.4f}"
    )
