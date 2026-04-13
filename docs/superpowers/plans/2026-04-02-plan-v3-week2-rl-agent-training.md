# v3 Week 2: RL Agent + Population-Based Training

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the LSTM+PPO RL agent and population-based training loop that runs on Colab Pro, with checkpoint/resume to Google Drive. Produces a trained agent ready for holdout evaluation.

**Architecture:** Each agent has an LSTM backbone (encodes temporal market state) feeding a PPO policy head (outputs per-alt trading scores + SL/TP distances). A population of 10 agents evolves: bottom 3 die each generation, top 3 reproduce with weight mutation, middle 4 receive experience from winners. Checkpoints save to Google Drive after every generation for Colab crash recovery.

**Tech Stack:** PyTorch, NumPy, Google Colab Pro (A100 GPU)

**Spec:** `docs/superpowers/specs/2026-04-02-rl-trading-agent-v3-design.md`

**Depends on:** Week 1 gym environment (`src/rl/env.py`, `src/rl/config.py`, `src/rl/portfolio.py`)

---

### File Structure

```
services/python/
├── src/rl/
│   ├── agent.py            # LSTMPPOAgent: neural network + action selection
│   ├── ppo.py              # PPO update logic (clipped surrogate loss)
│   ├── population.py       # PopulationTrainer: evolution + selection loop
│   └── checkpoint.py       # Save/load population to disk or Google Drive
├── scripts/
│   └── train_rl.py         # Main training entry point (runs locally or on Colab)
├── notebooks/
│   └── rl_training.ipynb   # Colab notebook (mounts Drive, runs training)
└── tests/
    ├── test_rl_agent.py
    ├── test_rl_ppo.py
    └── test_rl_population.py
```

---

### Task 1: LSTM+PPO Agent Network

**Files:**
- Create: `services/python/src/rl/agent.py`
- Test: `services/python/tests/test_rl_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_agent.py
import numpy as np
import torch
from src.rl.agent import LSTMPPOAgent


def test_agent_init():
    agent = LSTMPPOAgent(obs_size=324, n_alts=10)
    assert agent.obs_size == 324
    assert agent.n_alts == 10


def test_agent_act_shape():
    agent = LSTMPPOAgent(obs_size=324, n_alts=10)
    obs = np.random.randn(324).astype(np.float32)
    action, log_prob, value = agent.act(obs)
    # Action: n_alts * 3 (score, sl, tp per alt)
    assert action.shape == (30,)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_agent_act_deterministic():
    agent = LSTMPPOAgent(obs_size=324, n_alts=10)
    obs = np.random.randn(324).astype(np.float32)
    a1, _, _ = agent.act(obs, deterministic=True)
    a2, _, _ = agent.act(obs, deterministic=True)
    np.testing.assert_array_equal(a1, a2)


def test_agent_act_stochastic_varies():
    agent = LSTMPPOAgent(obs_size=324, n_alts=10)
    obs = np.random.randn(324).astype(np.float32)
    torch.manual_seed(1)
    a1, _, _ = agent.act(obs, deterministic=False)
    torch.manual_seed(2)
    a2, _, _ = agent.act(obs, deterministic=False)
    assert not np.array_equal(a1, a2)


def test_agent_evaluate_batch():
    agent = LSTMPPOAgent(obs_size=324, n_alts=10)
    obs_batch = torch.randn(8, 324)
    act_batch = torch.randn(8, 30)
    log_probs, values, entropy = agent.evaluate(obs_batch, act_batch)
    assert log_probs.shape == (8,)
    assert values.shape == (8,)
    assert entropy.shape == ()
    assert entropy.item() > 0  # Entropy should be positive


def test_agent_hidden_state_reset():
    agent = LSTMPPOAgent(obs_size=324, n_alts=10)
    agent.reset_hidden()
    assert agent.hidden is None


def test_agent_parameter_count():
    agent = LSTMPPOAgent(obs_size=324, n_alts=10, hidden_size=64, num_layers=2)
    total_params = sum(p.numel() for p in agent.parameters())
    # Should be manageable — under 500K params
    assert total_params < 500_000
    assert total_params > 10_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_rl_agent.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/rl/agent.py
"""LSTM + PPO Agent for crypto trading.

The agent observes market features + portfolio state, processes through an LSTM
to capture temporal patterns, then outputs trading actions via a PPO policy head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class LSTMPPOAgent(nn.Module):
    """LSTM backbone with PPO actor-critic heads.

    Input: flat observation vector (alt_features + indicators + portfolio_state)
    Output: per-alt actions [score, sl_pct, tp_pct] via Gaussian policy
    """

    def __init__(
        self,
        obs_size: int,
        n_alts: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.n_alts = n_alts
        self.action_size = n_alts * 3  # score + sl + tp per alt
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=obs_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Actor head (policy): outputs mean of Gaussian for each action dimension
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),
            nn.Tanh(),  # Bound scores to [-1, 1]
        )
        # Learnable log std
        self.actor_log_std = nn.Parameter(torch.zeros(self.action_size))

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Hidden state (maintained across steps within an episode)
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset_hidden(self) -> None:
        """Reset LSTM hidden state (call at episode start)."""
        self.hidden = None

    def _forward_lstm(self, obs: torch.Tensor) -> torch.Tensor:
        """Run observation through LSTM, return final hidden output."""
        # obs shape: (batch, obs_size) or (obs_size,)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)  # (1, 1, obs_size)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_size)

        lstm_out, self.hidden = self.lstm(obs, self.hidden)
        return lstm_out[:, -1, :]  # (batch, hidden_size)

    def act(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, float, float]:
        """Select action for a single observation.

        Returns: (action, log_prob, value)
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs)
            features = self._forward_lstm(obs_t)

            action_mean = self.actor_mean(features).squeeze(0)
            action_std = self.actor_log_std.exp().clamp(min=0.01)

            if deterministic:
                action = action_mean
                dist = Normal(action_mean, action_std)
                log_prob = dist.log_prob(action).sum().item()
            else:
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum().item()

            value = self.critic(features).squeeze().item()

            # Clamp action: scores to [-1, 1], sl/tp to [0, 1]
            action_np = action.numpy()
            for i in range(self.n_alts):
                action_np[i * 3] = np.clip(action_np[i * 3], -1.0, 1.0)
                action_np[i * 3 + 1] = np.clip(action_np[i * 3 + 1], 0.0, 1.0)
                action_np[i * 3 + 2] = np.clip(action_np[i * 3 + 2], 0.0, 1.0)

        return action_np, log_prob, value

    def evaluate(
        self, obs_batch: torch.Tensor, action_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update (batch mode).

        Args:
            obs_batch: (batch_size, obs_size)
            action_batch: (batch_size, action_size)

        Returns: (log_probs, values, entropy)
        """
        # Reset hidden for batch evaluation (stateless)
        saved_hidden = self.hidden
        self.hidden = None

        features = self._forward_lstm(obs_batch)  # (batch, hidden)

        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp().clamp(min=0.01)

        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(action_batch).sum(dim=-1)  # (batch,)
        entropy = dist.entropy().sum(dim=-1).mean()  # scalar
        values = self.critic(features).squeeze(-1)  # (batch,)

        self.hidden = saved_hidden
        return log_probs, values, entropy
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_rl_agent.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add services/python/src/rl/agent.py services/python/tests/test_rl_agent.py
git commit -m "feat(rl): add LSTM+PPO agent network"
```

---

### Task 2: PPO Update Logic

**Files:**
- Create: `services/python/src/rl/ppo.py`
- Test: `services/python/tests/test_rl_ppo.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_ppo.py
import torch
from src.rl.agent import LSTMPPOAgent
from src.rl.ppo import PPOBuffer, ppo_update


def test_buffer_add_and_get():
    buf = PPOBuffer()
    for _ in range(5):
        buf.add(
            obs=torch.randn(324),
            action=torch.randn(30),
            log_prob=-1.5,
            reward=0.01,
            value=0.5,
            done=False,
        )
    buf.add(obs=torch.randn(324), action=torch.randn(30),
            log_prob=-1.2, reward=0.02, value=0.6, done=True)

    batch = buf.get(gamma=0.99, lam=0.95)
    assert batch["obs"].shape == (6, 324)
    assert batch["actions"].shape == (6, 30)
    assert batch["returns"].shape == (6,)
    assert batch["advantages"].shape == (6,)
    assert batch["old_log_probs"].shape == (6,)


def test_buffer_gae_computation():
    """Verify GAE advantages are computed and normalized."""
    buf = PPOBuffer()
    for i in range(10):
        buf.add(
            obs=torch.randn(100), action=torch.randn(10),
            log_prob=-1.0, reward=0.01 * (i + 1), value=0.5, done=(i == 9),
        )
    batch = buf.get(gamma=0.99, lam=0.95)
    # Advantages should be normalized (mean ~0, std ~1)
    adv = batch["advantages"]
    assert abs(adv.mean().item()) < 0.5
    assert adv.std().item() > 0


def test_ppo_update_runs():
    agent = LSTMPPOAgent(obs_size=100, n_alts=3)
    buf = PPOBuffer()
    for i in range(20):
        buf.add(
            obs=torch.randn(100), action=torch.randn(9),
            log_prob=-1.0, reward=0.01, value=0.5, done=(i == 19),
        )

    batch = buf.get(gamma=0.99, lam=0.95)
    stats = ppo_update(agent, batch, epochs=2, batch_size=10, lr=3e-4)
    assert "policy_loss" in stats
    assert "value_loss" in stats
    assert "entropy" in stats
    assert isinstance(stats["policy_loss"], float)


def test_ppo_update_improves():
    """After PPO update, value predictions should be closer to returns."""
    agent = LSTMPPOAgent(obs_size=100, n_alts=3)
    buf = PPOBuffer()
    for i in range(50):
        buf.add(
            obs=torch.randn(100), action=torch.randn(9),
            log_prob=-1.0, reward=0.05, value=0.0, done=(i == 49),
        )

    batch = buf.get(gamma=0.99, lam=0.95)

    # Value loss before
    with torch.no_grad():
        _, values_before, _ = agent.evaluate(batch["obs"], batch["actions"])
    loss_before = (values_before - batch["returns"]).pow(2).mean().item()

    ppo_update(agent, batch, epochs=5, batch_size=16, lr=1e-3)

    # Value loss after
    with torch.no_grad():
        _, values_after, _ = agent.evaluate(batch["obs"], batch["actions"])
    loss_after = (values_after - batch["returns"]).pow(2).mean().item()

    assert loss_after < loss_before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_rl_ppo.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/rl/ppo.py
"""PPO (Proximal Policy Optimization) update logic.

Implements the clipped surrogate objective and GAE advantage estimation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.rl.agent import LSTMPPOAgent


class PPOBuffer:
    """Stores trajectory data for one episode and computes GAE advantages."""

    def __init__(self) -> None:
        self.obs: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get(self, gamma: float = 0.99, lam: float = 0.95) -> dict[str, torch.Tensor]:
        """Compute GAE advantages and returns, return as batch tensors."""
        n = len(self.rewards)
        advantages = torch.zeros(n)
        returns = torch.zeros(n)

        # GAE computation (reverse pass)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(n)):
            mask = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
            next_value = self.values[t]

        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "obs": torch.stack(self.obs),
            "actions": torch.stack(self.actions),
            "old_log_probs": torch.tensor(self.log_probs),
            "returns": returns,
            "advantages": advantages,
        }

    def clear(self) -> None:
        self.__init__()


def ppo_update(
    agent: LSTMPPOAgent,
    batch: dict[str, torch.Tensor],
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> dict[str, float]:
    """Run PPO update on collected batch.

    Returns dict with policy_loss, value_loss, entropy averages.
    """
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    obs = batch["obs"]
    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"]
    returns = batch["returns"]
    advantages = batch["advantages"]

    n = len(obs)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(epochs):
        indices = torch.randperm(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            new_log_probs, values, entropy = agent.evaluate(obs[idx], actions[idx])

            # Clipped surrogate objective
            ratio = (new_log_probs - old_log_probs[idx]).exp()
            surr1 = ratio * advantages[idx]
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages[idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(values, returns[idx])

            # Total loss
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss": total_value_loss / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_rl_ppo.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add services/python/src/rl/ppo.py services/python/tests/test_rl_ppo.py
git commit -m "feat(rl): add PPO buffer and update logic"
```

---

### Task 3: Checkpoint Save/Load

**Files:**
- Create: `services/python/src/rl/checkpoint.py`
- Test: `services/python/tests/test_rl_checkpoint.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_checkpoint.py
import json
from pathlib import Path

import torch

from src.rl.agent import LSTMPPOAgent
from src.rl.checkpoint import save_checkpoint, load_checkpoint


def test_save_and_load_checkpoint(tmp_path):
    agents = [LSTMPPOAgent(obs_size=100, n_alts=3) for _ in range(3)]
    scores = [0.5, 0.8, 0.3]
    metadata = {"generation": 5, "best_reward": 0.8}

    save_checkpoint(tmp_path, agents, scores, metadata)

    loaded_agents, loaded_scores, loaded_meta = load_checkpoint(
        tmp_path, obs_size=100, n_alts=3
    )
    assert len(loaded_agents) == 3
    assert loaded_scores == scores
    assert loaded_meta["generation"] == 5
    assert loaded_meta["best_reward"] == 0.8


def test_checkpoint_preserves_weights(tmp_path):
    agent = LSTMPPOAgent(obs_size=100, n_alts=3)
    original_params = {k: v.clone() for k, v in agent.state_dict().items()}

    save_checkpoint(tmp_path, [agent], [1.0], {"generation": 1})
    loaded, _, _ = load_checkpoint(tmp_path, obs_size=100, n_alts=3)

    for key in original_params:
        torch.testing.assert_close(
            loaded[0].state_dict()[key], original_params[key]
        )


def test_load_nonexistent_returns_none(tmp_path):
    result = load_checkpoint(tmp_path / "nonexistent", obs_size=100, n_alts=3)
    assert result is None


def test_checkpoint_files_exist(tmp_path):
    agents = [LSTMPPOAgent(obs_size=100, n_alts=3)]
    save_checkpoint(tmp_path, agents, [0.5], {"generation": 1})
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "agent_0.pt").exists()
    assert (tmp_path / "scores.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_rl_checkpoint.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/rl/checkpoint.py
"""Checkpoint save/load for population-based training.

Saves agent weights, scores, and metadata to a directory.
Compatible with Google Drive for Colab crash recovery.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from src.rl.agent import LSTMPPOAgent

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: Path,
    agents: list[LSTMPPOAgent],
    scores: list[float],
    metadata: dict,
) -> None:
    """Save entire population state to disk."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save each agent's weights
    for i, agent in enumerate(agents):
        torch.save(agent.state_dict(), checkpoint_dir / f"agent_{i}.pt")

    # Save scores
    with open(checkpoint_dir / "scores.json", "w") as f:
        json.dump(scores, f)

    # Save metadata
    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Checkpoint saved: {len(agents)} agents, gen={metadata.get('generation', '?')}")


def load_checkpoint(
    checkpoint_dir: Path,
    obs_size: int,
    n_alts: int,
    hidden_size: int = 64,
    num_layers: int = 2,
) -> tuple[list[LSTMPPOAgent], list[float], dict] | None:
    """Load population state from disk. Returns None if not found."""
    checkpoint_dir = Path(checkpoint_dir)
    meta_path = checkpoint_dir / "metadata.json"

    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        metadata = json.load(f)

    with open(checkpoint_dir / "scores.json") as f:
        scores = json.load(f)

    agents = []
    i = 0
    while (checkpoint_dir / f"agent_{i}.pt").exists():
        agent = LSTMPPOAgent(
            obs_size=obs_size, n_alts=n_alts,
            hidden_size=hidden_size, num_layers=num_layers,
        )
        agent.load_state_dict(torch.load(checkpoint_dir / f"agent_{i}.pt", weights_only=True))
        agents.append(agent)
        i += 1

    logger.info(
        f"Checkpoint loaded: {len(agents)} agents, gen={metadata.get('generation', '?')}"
    )
    return agents, scores, metadata
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_rl_checkpoint.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add services/python/src/rl/checkpoint.py services/python/tests/test_rl_checkpoint.py
git commit -m "feat(rl): add checkpoint save/load for population training"
```

---

### Task 4: Population Trainer

**Files:**
- Create: `services/python/src/rl/population.py`
- Test: `services/python/tests/test_rl_population.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_population.py
import numpy as np
import torch
from src.rl.agent import LSTMPPOAgent
from src.rl.population import PopulationTrainer
from src.rl.config import RLConfig


def _make_fake_dataset(n_times=1200, n_alts=10, n_alt_feat=31, n_ind_feat=9):
    rng = np.random.RandomState(42)
    prices = np.zeros((n_times, n_alts, 4), dtype=np.float32)
    for j in range(n_alts):
        base = rng.uniform(1.0, 100.0)
        for t in range(n_times):
            base *= 1 + rng.randn() * 0.02
            prices[t, j] = [base * 0.99, base * 1.02, base * 0.97, base]
    return {
        "alt_features": rng.randn(n_times, n_alts, n_alt_feat).astype(np.float32) * 0.1,
        "indicator_features": rng.randn(n_times, n_ind_feat).astype(np.float32) * 0.1,
        "prices": prices,
        "alt_names": [f"ALT{i}" for i in range(n_alts)],
        "timestamps": np.arange(n_times),
    }


def test_population_init():
    dataset = _make_fake_dataset()
    trainer = PopulationTrainer(dataset, population_size=5)
    assert len(trainer.agents) == 5
    assert len(trainer.scores) == 5


def test_evaluate_agent():
    dataset = _make_fake_dataset()
    trainer = PopulationTrainer(dataset, population_size=3)
    reward = trainer.evaluate_agent(trainer.agents[0], n_episodes=2)
    assert isinstance(reward, float)
    assert np.isfinite(reward)


def test_evolve_population():
    dataset = _make_fake_dataset()
    trainer = PopulationTrainer(dataset, population_size=6)
    # Set fake scores
    trainer.scores = [0.1, 0.5, 0.3, 0.8, 0.2, 0.9]
    trainer.evolve()
    assert len(trainer.agents) == 6  # Population size preserved
    assert len(trainer.scores) == 6


def test_run_generation():
    dataset = _make_fake_dataset()
    trainer = PopulationTrainer(dataset, population_size=4, episodes_per_agent=1)
    stats = trainer.run_generation()
    assert "best_reward" in stats
    assert "mean_reward" in stats
    assert "generation" in stats
    assert stats["generation"] == 1


def test_train_loop_checkpoint(tmp_path):
    dataset = _make_fake_dataset()
    trainer = PopulationTrainer(
        dataset, population_size=3, episodes_per_agent=1,
        checkpoint_dir=tmp_path,
    )
    trainer.train(n_generations=2)
    assert trainer.generation == 2
    # Checkpoint should exist
    assert (tmp_path / "metadata.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_rl_population.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# src/rl/population.py
"""Population-based training for RL trading agents.

Evolves a population of LSTM+PPO agents: evaluate each on random episodes,
rank by reward, bottom die and get replaced by mutated copies of top agents.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np
import torch

from src.rl.agent import LSTMPPOAgent
from src.rl.checkpoint import load_checkpoint, save_checkpoint
from src.rl.config import RLConfig
from src.rl.env import CryptoTradingEnv
from src.rl.ppo import PPOBuffer, ppo_update

logger = logging.getLogger(__name__)


class PopulationTrainer:
    """Manages a population of agents and evolves them."""

    def __init__(
        self,
        dataset: dict,
        population_size: int = 10,
        episodes_per_agent: int = 10,
        mutation_noise: float = 0.05,
        kill_fraction: float = 0.3,
        checkpoint_dir: Path | None = None,
        config: RLConfig | None = None,
    ):
        self.config = config or RLConfig()
        self.dataset = dataset
        self.population_size = population_size
        self.episodes_per_agent = episodes_per_agent
        self.mutation_noise = mutation_noise
        self.kill_count = max(1, int(population_size * kill_fraction))
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.generation = 0

        # Compute dimensions from dataset
        n_alts = len(dataset["alt_names"])
        n_alt_feat = dataset["alt_features"].shape[2]
        n_ind_feat = dataset["indicator_features"].shape[1]
        self.obs_size = n_alts * n_alt_feat + n_ind_feat + self.config.N_PORTFOLIO_FEATURES
        self.n_alts = n_alts

        # Try to resume from checkpoint
        self.agents: list[LSTMPPOAgent] = []
        self.scores: list[float] = [0.0] * population_size

        if self.checkpoint_dir:
            loaded = load_checkpoint(self.checkpoint_dir, self.obs_size, self.n_alts)
            if loaded:
                self.agents, self.scores, meta = loaded
                self.generation = meta.get("generation", 0)
                self.population_size = len(self.agents)
                logger.info(f"Resumed from generation {self.generation}")

        # Create new agents if needed
        while len(self.agents) < self.population_size:
            self.agents.append(LSTMPPOAgent(self.obs_size, self.n_alts))
            if len(self.scores) < len(self.agents):
                self.scores.append(0.0)

    def evaluate_agent(self, agent: LSTMPPOAgent, n_episodes: int | None = None) -> float:
        """Run agent through episodes, return mean reward. Also does PPO update."""
        n_episodes = n_episodes or self.episodes_per_agent
        env = CryptoTradingEnv(self.dataset, self.config)
        total_reward = 0.0

        for _ in range(n_episodes):
            obs = env.reset()
            agent.reset_hidden()
            buffer = PPOBuffer()
            done = False
            episode_reward = 0.0

            while not done:
                action, log_prob, value = agent.act(obs)
                next_obs, reward, done, _info = env.step(action)

                buffer.add(
                    obs=torch.FloatTensor(obs),
                    action=torch.FloatTensor(action),
                    log_prob=log_prob,
                    reward=reward,
                    value=value,
                    done=done,
                )
                obs = next_obs
                episode_reward += reward

            # PPO update after each episode
            if len(buffer.obs) > 10:
                batch = buffer.get(gamma=0.99, lam=0.95)
                ppo_update(agent, batch, epochs=3, batch_size=32, lr=3e-4)

            total_reward += episode_reward

        return total_reward / n_episodes

    def evolve(self) -> None:
        """Evolutionary step: kill bottom agents, reproduce top agents."""
        ranked_idx = np.argsort(self.scores)  # Ascending (worst first)
        bottom = ranked_idx[: self.kill_count]
        top = ranked_idx[-self.kill_count :]

        # Replace bottom with mutated copies of top
        for i, kill_idx in enumerate(bottom):
            parent_idx = top[i % len(top)]
            parent = self.agents[parent_idx]

            # Clone parent
            child = LSTMPPOAgent(self.obs_size, self.n_alts)
            child.load_state_dict(copy.deepcopy(parent.state_dict()))

            # Mutate weights
            with torch.no_grad():
                for param in child.parameters():
                    noise = torch.randn_like(param) * self.mutation_noise
                    param.add_(noise)

            self.agents[kill_idx] = child
            self.scores[kill_idx] = 0.0

    def run_generation(self) -> dict:
        """Evaluate all agents, evolve, return stats."""
        self.generation += 1
        t0 = time.time()

        # Evaluate each agent
        for i, agent in enumerate(self.agents):
            self.scores[i] = self.evaluate_agent(agent)

        # Evolve
        self.evolve()

        elapsed = time.time() - t0
        stats = {
            "generation": self.generation,
            "best_reward": float(max(self.scores)),
            "mean_reward": float(np.mean(self.scores)),
            "worst_reward": float(min(self.scores)),
            "elapsed_s": round(elapsed, 1),
        }

        logger.info(
            f"Gen {self.generation}: best={stats['best_reward']:.4f}, "
            f"mean={stats['mean_reward']:.4f}, {elapsed:.1f}s"
        )

        # Checkpoint
        if self.checkpoint_dir:
            save_checkpoint(
                self.checkpoint_dir,
                self.agents,
                self.scores,
                {"generation": self.generation, **stats},
            )

        return stats

    def train(self, n_generations: int = 100) -> list[dict]:
        """Run full training loop."""
        all_stats = []
        target = self.generation + n_generations

        while self.generation < target:
            stats = self.run_generation()
            all_stats.append(stats)

        return all_stats

    def get_best_agent(self) -> LSTMPPOAgent:
        """Return the agent with the highest score."""
        best_idx = int(np.argmax(self.scores))
        return self.agents[best_idx]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_rl_population.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add services/python/src/rl/population.py services/python/tests/test_rl_population.py
git commit -m "feat(rl): add population-based training with evolution"
```

---

### Task 5: Training Script

**Files:**
- Create: `services/python/scripts/train_rl.py`

- [ ] **Step 1: Write the script**

```python
# scripts/train_rl.py
"""RL Population-Based Training script.

Runs locally or on Colab. Loads rl_dataset.npz, trains population of agents,
saves checkpoints for crash recovery.

Usage:
    python scripts/train_rl.py                          # Defaults: 10 agents, 100 gens
    python scripts/train_rl.py --population 20 --generations 500  # Phase 2
    python scripts/train_rl.py --checkpoint /content/drive/MyDrive/ai-finance/checkpoints
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RL Population Training")
    parser.add_argument("--data", type=str, default=None, help="Path to rl_dataset.npz")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--population", type=int, default=10, help="Population size")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per agent per gen")
    parser.add_argument("--mutation", type=float, default=0.05, help="Mutation noise std")
    args = parser.parse_args()

    # Resolve paths
    project = Path("C:/Users/togat/Desktop/AI-Finance")
    data_path = Path(args.data) if args.data else project / "data/features/rl_dataset.npz"
    ckpt_dir = Path(args.checkpoint) if args.checkpoint else project / "models/rl_checkpoints"

    logger.info("=" * 60)
    logger.info("RL POPULATION-BASED TRAINING")
    logger.info("=" * 60)
    logger.info(f"Data: {data_path}")
    logger.info(f"Checkpoint: {ckpt_dir}")
    logger.info(f"Population: {args.population}, Generations: {args.generations}")
    logger.info(f"Episodes/agent: {args.episodes}, Mutation: {args.mutation}")

    # Load dataset
    logger.info("\nLoading dataset...")
    t0 = time.time()
    data = np.load(str(data_path), allow_pickle=True)
    dataset = {
        "alt_features": data["alt_features"],
        "indicator_features": data["indicator_features"],
        "prices": data["prices"],
        "alt_names": data["alt_names"].tolist(),
        "timestamps": data["timestamps"],
    }
    logger.info(
        f"Loaded in {time.time() - t0:.1f}s: "
        f"{dataset['alt_features'].shape[1]} alts, "
        f"{dataset['alt_features'].shape[0]} timestamps, "
        f"{dataset['alt_features'].shape[2]} features/alt"
    )

    # Train
    from src.rl.population import PopulationTrainer

    trainer = PopulationTrainer(
        dataset=dataset,
        population_size=args.population,
        episodes_per_agent=args.episodes,
        mutation_noise=args.mutation,
        checkpoint_dir=ckpt_dir,
    )

    logger.info(f"\nStarting from generation {trainer.generation}...")
    t_start = time.time()
    all_stats = trainer.train(n_generations=args.generations)

    total_time = time.time() - t_start
    logger.info(f"\nTraining complete in {total_time / 60:.1f} minutes")
    logger.info(f"Final best reward: {all_stats[-1]['best_reward']:.4f}")
    logger.info(f"Final mean reward: {all_stats[-1]['mean_reward']:.4f}")

    # Save best agent separately
    best = trainer.get_best_agent()
    best_path = ckpt_dir / "best_agent.pt"
    import torch
    torch.save(best.state_dict(), best_path)
    logger.info(f"Best agent saved to {best_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for s in all_stats[-5:]:
        print(
            f"  Gen {s['generation']:>4}: best={s['best_reward']:>8.4f}, "
            f"mean={s['mean_reward']:>8.4f}, {s['elapsed_s']:.1f}s"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses args**

Run: `cd services/python && python scripts/train_rl.py --help`
Expected: Shows help with all arguments

- [ ] **Step 3: Commit**

```bash
git add services/python/scripts/train_rl.py
git commit -m "feat(rl): add training entry point script"
```

---

### Task 6: Colab Training Notebook

**Files:**
- Create: `notebooks/rl_training.ipynb`

- [ ] **Step 1: Create the notebook**

Create a Jupyter notebook with these cells:

**Cell 1 (markdown):**
```markdown
# RL Trading Agent — Population-Based Training
Colab Pro (A100 GPU). Trains population of LSTM+PPO agents on crypto data.
Checkpoints to Google Drive for crash recovery.
```

**Cell 2 (code): Mount Drive + Clone repo**
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/putratogatorop/AI-Finance.git /content/AI-Finance 2>/dev/null || (cd /content/AI-Finance && git pull)
%cd /content/AI-Finance/services/python

!pip install -q torch numpy pandas
```

**Cell 3 (code): Upload or copy dataset**
```python
import os
from pathlib import Path

# Option A: Copy from Drive (if you uploaded rl_dataset.npz to Drive)
DRIVE_DATA = Path("/content/drive/MyDrive/ai-finance/rl_dataset.npz")
LOCAL_DATA = Path("/content/AI-Finance/data/features/rl_dataset.npz")

if DRIVE_DATA.exists():
    print("Copying dataset from Drive...")
    !cp "{DRIVE_DATA}" "{LOCAL_DATA}"
else:
    # Option B: Upload directly
    from google.colab import files
    print("Upload rl_dataset.npz:")
    uploaded = files.upload()
    !mkdir -p /content/AI-Finance/data/features/
    !mv rl_dataset.npz /content/AI-Finance/data/features/

import numpy as np
d = np.load(str(LOCAL_DATA), allow_pickle=True)
print(f"Dataset: {d['alt_features'].shape[1]} alts, {d['alt_features'].shape[0]} timestamps")
```

**Cell 4 (code): Phase 1 Training**
```python
CHECKPOINT_DIR = "/content/drive/MyDrive/ai-finance/rl_checkpoints"

!python scripts/train_rl.py \
    --data /content/AI-Finance/data/features/rl_dataset.npz \
    --checkpoint {CHECKPOINT_DIR} \
    --population 10 \
    --generations 100 \
    --episodes 10
```

**Cell 5 (code): View results**
```python
import json
from pathlib import Path

ckpt = Path(CHECKPOINT_DIR)
if (ckpt / "metadata.json").exists():
    with open(ckpt / "metadata.json") as f:
        meta = json.load(f)
    print(f"Generation: {meta['generation']}")
    print(f"Best reward: {meta['best_reward']:.4f}")
    print(f"Mean reward: {meta['mean_reward']:.4f}")
else:
    print("No checkpoint found yet")
```

**Cell 6 (code): Phase 2 (run after Phase 1 shows promise)**
```python
# Uncomment to run Phase 2 (500 generations, 20 agents)
# !python scripts/train_rl.py \
#     --data /content/AI-Finance/data/features/rl_dataset.npz \
#     --checkpoint {CHECKPOINT_DIR} \
#     --population 20 \
#     --generations 500 \
#     --episodes 10
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/rl_training.ipynb
git commit -m "feat(rl): add Colab training notebook"
```

---

### Task 7: Holdout Evaluation Script

**Files:**
- Create: `services/python/scripts/evaluate_rl.py`

- [ ] **Step 1: Write the script**

```python
# scripts/evaluate_rl.py
"""Evaluate trained RL agent on holdout period.

Loads best_agent.pt, runs deterministic episode on holdout data,
outputs performance metrics for go/no-go decision.

Usage:
    python scripts/evaluate_rl.py --checkpoint models/rl_checkpoints
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RL agent on holdout")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--holdout-start", type=float, default=0.85,
                        help="Fraction of data where holdout starts (0.85 = last 15%%)")
    args = parser.parse_args()

    project = Path("C:/Users/togat/Desktop/AI-Finance")
    data_path = Path(args.data) if args.data else project / "data/features/rl_dataset.npz"
    ckpt_dir = Path(args.checkpoint) if args.checkpoint else project / "models/rl_checkpoints"

    # Load dataset
    logger.info("Loading dataset...")
    data = np.load(str(data_path), allow_pickle=True)
    dataset = {
        "alt_features": data["alt_features"],
        "indicator_features": data["indicator_features"],
        "prices": data["prices"],
        "alt_names": data["alt_names"].tolist(),
        "timestamps": data["timestamps"],
    }

    n_times = dataset["alt_features"].shape[0]
    holdout_idx = int(n_times * args.holdout_start)
    logger.info(f"Total timestamps: {n_times}, holdout from: {holdout_idx}")

    # Slice holdout data
    holdout = {
        "alt_features": dataset["alt_features"][holdout_idx:],
        "indicator_features": dataset["indicator_features"][holdout_idx:],
        "prices": dataset["prices"][holdout_idx:],
        "alt_names": dataset["alt_names"],
        "timestamps": dataset["timestamps"][holdout_idx:],
    }

    # Load agent
    from src.rl.agent import LSTMPPOAgent
    from src.rl.config import RLConfig
    from src.rl.env import CryptoTradingEnv

    cfg = RLConfig()
    n_alts = len(holdout["alt_names"])
    n_alt_feat = holdout["alt_features"].shape[2]
    n_ind_feat = holdout["indicator_features"].shape[1]
    obs_size = n_alts * n_alt_feat + n_ind_feat + cfg.N_PORTFOLIO_FEATURES

    agent = LSTMPPOAgent(obs_size, n_alts)
    best_path = ckpt_dir / "best_agent.pt"
    if not best_path.exists():
        logger.error(f"No best_agent.pt found at {best_path}")
        sys.exit(1)

    agent.load_state_dict(torch.load(best_path, weights_only=True))
    agent.eval()
    logger.info("Loaded best agent")

    # Run deterministic episode on holdout
    env = CryptoTradingEnv(holdout, cfg)
    obs = env.reset(start_idx=0)
    agent.reset_hidden()

    done = False
    returns = []
    trades = 0
    trade_pnls = []

    while not done:
        action, _, _ = agent.act(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        returns.append(reward)
        trades = info.get("n_positions", 0)

    # Compute metrics
    returns = np.array(returns)
    equity_curve = np.cumprod(1 + returns)
    total_return = float(equity_curve[-1] - 1) * 100

    mean_r = returns.mean()
    std_r = returns.std()
    sharpe = (mean_r / std_r) * np.sqrt(730) if std_r > 0 else 0.0

    peak = np.maximum.accumulate(equity_curve)
    max_dd = float(np.min((equity_curve - peak) / peak)) * 100

    win_rate = float((returns > 0).sum() / max(len(returns), 1)) * 100

    # Go/no-go
    metrics = {
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate": round(win_rate, 2),
        "n_windows": len(returns),
        "final_equity": round(float(info["portfolio_value"]), 0),
    }

    go = (
        sharpe > 0.5
        and total_return > 0
        and max_dd > -25
        and win_rate > 48
    )

    print("\n" + "=" * 60)
    print("HOLDOUT EVALUATION")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\n  GO/NO-GO: {'GO' if go else 'NO-GO'}")
    print("=" * 60)

    # Save results
    results_path = ckpt_dir / "holdout_results.json"
    with open(results_path, "w") as f:
        json.dump({**metrics, "go": go}, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses args**

Run: `cd services/python && python scripts/evaluate_rl.py --help`
Expected: Shows help

- [ ] **Step 3: Commit**

```bash
git add services/python/scripts/evaluate_rl.py
git commit -m "feat(rl): add holdout evaluation script with go/no-go decision"
```

---

## Summary

After completing all 7 tasks, you will have:

| File | Purpose |
|------|---------|
| `src/rl/agent.py` | LSTM+PPO neural network with act/evaluate |
| `src/rl/ppo.py` | PPO buffer + clipped surrogate update |
| `src/rl/checkpoint.py` | Save/load population to disk/Drive |
| `src/rl/population.py` | Population evolution: evaluate, rank, kill, reproduce |
| `scripts/train_rl.py` | Training entry point (local or Colab) |
| `scripts/evaluate_rl.py` | Holdout evaluation with go/no-go |
| `notebooks/rl_training.ipynb` | Colab notebook for training |
| 3 test files | ~20 tests covering agent, PPO, population |

**Training workflow:**
1. Upload `rl_dataset.npz` to Google Drive
2. Open `rl_training.ipynb` in Colab Pro
3. Run Phase 1 (10 agents, 100 gens, ~8 hours)
4. Check results, iterate if needed
5. Run Phase 2 (20 agents, 500 gens, ~40 hours)
6. Run `evaluate_rl.py` on holdout → go/no-go decision
