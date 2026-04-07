"""LSTM + PPO actor-critic agent network for RL crypto trading."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class LSTMPPOAgent(nn.Module):
    """Actor-critic network with LSTM backbone for PPO training."""

    def __init__(
        self,
        obs_size: int,
        n_alts: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.obs_size = obs_size
        self.n_alts = n_alts
        self.action_size = n_alts * 3
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = torch.device(device)

        # LSTM backbone (obs_size is small enough to feed directly)
        self.lstm = nn.LSTM(
            input_size=obs_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Actor head — outputs action means in [-1, 1]
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),
            nn.Tanh(),
        )

        # Initialize output layer with small weights to prevent Tanh saturation
        nn.init.uniform_(self.actor_mean[-2].weight, -0.003, 0.003)
        nn.init.zeros_(self.actor_mean[-2].bias)

        # Learnable log standard deviation
        self.actor_log_std = nn.Parameter(
            torch.full((self.action_size,), -0.5)
        )

        # Critic head — value function
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # LSTM hidden state (managed across steps)
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None

        # Move entire model to device
        self.to(self.device)

    # ------------------------------------------------------------------
    def reset_hidden(self) -> None:
        """Reset LSTM hidden state to None."""
        self.hidden = None

    # ------------------------------------------------------------------
    def _forward_lstm(self, obs: torch.Tensor) -> torch.Tensor:
        """Run observation through LSTM, return last hidden output.

        Handles 1-D (single obs), 2-D (batch of obs) inputs by adding
        the required sequence dimension.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)

        lstm_out, self.hidden = self.lstm(obs, self.hidden)
        return lstm_out[:, -1, :]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """Select an action given a single observation.

        Returns:
            action: numpy array of shape (action_size,)
            log_prob: scalar log-probability of the action
            value: scalar value estimate
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        h = self._forward_lstm(obs_t)

        # Actor
        action_mean = self.actor_mean(h).squeeze(0)
        action_std = torch.nan_to_num(
            torch.exp(self.actor_log_std.clamp(min=-5.0, max=2.0)),
            nan=0.01,
        ).clamp(min=0.01)
        action_mean = torch.nan_to_num(action_mean, nan=0.0)
        dist = Normal(action_mean, action_std)

        if deterministic:
            action = action_mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum().item()
        value = self.critic(h).squeeze().item()

        # Clamp outputs per triple (score, sl, tp)
        action_np = action.cpu().numpy()
        for i in range(self.n_alts):
            base = i * 3
            action_np[base] = np.clip(action_np[base], -1.0, 1.0)       # score
            action_np[base + 1] = np.clip(action_np[base + 1], 0.0, 1.0)  # sl
            action_np[base + 2] = np.clip(action_np[base + 2], 0.0, 1.0)  # tp

        return action_np, log_prob, value

    # ------------------------------------------------------------------
    def evaluate(
        self,
        obs_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a batch of (obs, action) pairs for PPO update.

        Returns:
            log_probs: (batch,) per-sample log probabilities
            values: (batch,) value estimates
            entropy: scalar mean entropy
        """
        # Stateless for batch evaluation
        saved_hidden = self.hidden
        self.hidden = None

        obs_batch = obs_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        h = self._forward_lstm(obs_batch)

        action_mean = self.actor_mean(h)
        # Clamp log_std to prevent exploding/vanishing std
        action_std = torch.nan_to_num(
            torch.exp(self.actor_log_std.clamp(min=-5.0, max=2.0)),
            nan=0.01,
        ).clamp(min=0.01)
        # Guard against NaN from Tanh saturation
        action_mean = torch.nan_to_num(action_mean, nan=0.0)
        dist = Normal(action_mean, action_std)

        log_probs = dist.log_prob(action_batch).sum(dim=-1)  # (batch,)
        values = self.critic(h).squeeze(-1)                   # (batch,)
        entropy = dist.entropy().mean()                        # scalar

        # Restore hidden
        self.hidden = saved_hidden

        return log_probs, values, entropy
