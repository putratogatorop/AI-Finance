"""Checkpoint save/load for population-based RL training."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.rl.agent import LSTMPPOAgent


def save_checkpoint(
    checkpoint_dir: Path,
    agents: list[LSTMPPOAgent],
    scores: list[float],
    metadata: dict,
) -> None:
    """Save population checkpoint to disk.

    Creates checkpoint_dir if it doesn't exist. Saves each agent's
    state_dict, the score list, and arbitrary metadata.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for i, agent in enumerate(agents):
        torch.save(agent.state_dict(), checkpoint_dir / f"agent_{i}.pt")

    with open(checkpoint_dir / "scores.json", "w") as f:
        json.dump(scores, f)

    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_checkpoint(
    checkpoint_dir: Path,
    obs_size: int,
    n_alts: int,
    hidden_size: int = 64,
    num_layers: int = 2,
) -> tuple[list[LSTMPPOAgent], list[float], dict] | None:
    """Load population checkpoint from disk.

    Returns None if metadata.json doesn't exist (no checkpoint found).
    Otherwise returns (agents, scores, metadata).
    """
    checkpoint_dir = Path(checkpoint_dir)
    metadata_path = checkpoint_dir / "metadata.json"

    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    with open(checkpoint_dir / "scores.json") as f:
        scores = json.load(f)

    agents: list[LSTMPPOAgent] = []
    i = 0
    while True:
        agent_path = checkpoint_dir / f"agent_{i}.pt"
        if not agent_path.exists():
            break
        agent = LSTMPPOAgent(
            obs_size=obs_size,
            n_alts=n_alts,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        state_dict = torch.load(agent_path, weights_only=True)
        agent.load_state_dict(state_dict)
        agents.append(agent)
        i += 1

    return agents, scores, metadata
