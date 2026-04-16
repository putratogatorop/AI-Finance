"""Tests for checkpoint save/load functionality."""

from __future__ import annotations

from pathlib import Path

import torch

from src.rl.agent import LSTMPPOAgent
from src.rl.checkpoint import load_checkpoint, save_checkpoint

OBS_SIZE = 100
N_ALTS = 3


def _make_agents(n: int = 3) -> list[LSTMPPOAgent]:
    return [LSTMPPOAgent(obs_size=OBS_SIZE, n_alts=N_ALTS) for _ in range(n)]


def test_save_and_load_roundtrip(tmp_path: Path):
    """Save then load should return matching scores and metadata."""
    agents = _make_agents(3)
    scores = [1.0, 2.5, -0.3]
    metadata = {"generation": 5, "best_reward": 2.5}

    save_checkpoint(tmp_path / "ckpt", agents, scores, metadata)
    result = load_checkpoint(tmp_path / "ckpt", OBS_SIZE, N_ALTS)

    assert result is not None
    loaded_agents, loaded_scores, loaded_metadata = result
    assert len(loaded_agents) == 3
    assert loaded_scores == scores
    assert loaded_metadata == metadata


def test_weights_preserved(tmp_path: Path):
    """Loaded agent weights should match saved agent weights."""
    agents = _make_agents(2)
    scores = [0.0, 0.0]
    metadata = {"generation": 0}

    # Grab original weight tensor for comparison
    orig_weight = agents[0].actor_mean[0].weight.clone()

    save_checkpoint(tmp_path / "ckpt", agents, scores, metadata)
    loaded_agents, _, _ = load_checkpoint(tmp_path / "ckpt", OBS_SIZE, N_ALTS)

    loaded_weight = loaded_agents[0].actor_mean[0].weight
    assert torch.allclose(orig_weight, loaded_weight)


def test_nonexistent_returns_none(tmp_path: Path):
    """Loading from a nonexistent directory should return None."""
    result = load_checkpoint(tmp_path / "no_such_dir", OBS_SIZE, N_ALTS)
    assert result is None


def test_files_exist(tmp_path: Path):
    """After save, expected files should exist on disk."""
    agents = _make_agents(2)
    ckpt_dir = tmp_path / "ckpt"

    save_checkpoint(ckpt_dir, agents, [0.0, 0.0], {"generation": 1})

    assert (ckpt_dir / "metadata.json").exists()
    assert (ckpt_dir / "scores.json").exists()
    assert (ckpt_dir / "agent_0.pt").exists()
    assert (ckpt_dir / "agent_1.pt").exists()
