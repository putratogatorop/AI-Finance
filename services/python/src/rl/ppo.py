"""PPO buffer and update logic for RL crypto trading."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam

from src.rl.agent import LSTMPPOAgent


class PPOBuffer:
    """Rollout buffer that stores transitions and computes GAE advantages."""

    def __init__(self) -> None:
        self.clear()

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Append a single transition to the buffer."""
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get(self, gamma: float = 0.99, lam: float = 0.95) -> dict:
        """Compute GAE advantages and return a batch dict.

        Returns dict with keys: obs, actions, old_log_probs, returns, advantages.
        """
        n = len(self.rewards)
        advantages = torch.zeros(n)
        gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
            else:
                next_value = self.values[t + 1]
            not_done = 1.0 - float(self.dones[t])
            delta = (
                self.rewards[t]
                + gamma * next_value * not_done
                - self.values[t]
            )
            gae = delta + gamma * lam * not_done * gae
            advantages[t] = gae

        values_t = torch.tensor(self.values, dtype=torch.float32)
        returns = advantages + values_t

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "obs": torch.stack(self.obs),
            "actions": torch.stack(self.actions),
            "old_log_probs": torch.tensor(
                self.log_probs, dtype=torch.float32
            ),
            "returns": returns,
            "advantages": advantages,
        }

    def clear(self) -> None:
        """Reinitialize all storage lists."""
        self.obs: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []


def ppo_update(
    agent: LSTMPPOAgent,
    batch: dict,
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> dict[str, float]:
    """Run PPO policy gradient update on the agent.

    Returns dict with average policy_loss, value_loss, entropy.
    """
    optimizer = Adam(agent.parameters(), lr=lr)
    device = agent.device

    obs = batch["obs"].to(device)
    actions = batch["actions"].to(device)
    old_log_probs = batch["old_log_probs"].to(device)
    returns = batch["returns"].to(device)
    advantages = batch["advantages"].to(device)

    n = obs.shape[0]
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    for _ in range(epochs):
        indices = torch.randperm(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            mb_obs = obs[idx]
            mb_actions = actions[idx]
            mb_old_lp = old_log_probs[idx]
            mb_returns = returns[idx]
            mb_adv = advantages[idx]

            new_log_probs, values, entropy = agent.evaluate(
                mb_obs, mb_actions
            )

            ratio = torch.exp(new_log_probs - mb_old_lp)
            surr1 = ratio * mb_adv
            surr2 = (
                torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, mb_returns)

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            num_updates += 1

    return {
        "policy_loss": total_policy_loss / max(num_updates, 1),
        "value_loss": total_value_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
    }
