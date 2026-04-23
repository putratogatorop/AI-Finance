"""Population-based training with evolutionary selection for RL agents."""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.rl.agent import LSTMPPOAgent
from src.rl.checkpoint import load_checkpoint, save_checkpoint
from src.rl.config import RLConfig
from src.rl.env import CryptoTradingEnv
from src.rl.ppo import PPOBuffer, ppo_update

logger = logging.getLogger(__name__)


class PopulationTrainer:
    """Evolutionary population trainer that combines PPO with selection."""

    def __init__(
        self,
        dataset: dict[str, Any],
        population_size: int = 10,
        episodes_per_agent: int = 10,
        mutation_noise: float = 0.05,
        kill_fraction: float = 0.3,
        checkpoint_dir: Path | None = None,
        config: RLConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.dataset = dataset
        self.population_size = population_size
        self.episodes_per_agent = episodes_per_agent
        self.mutation_noise = mutation_noise
        self.kill_fraction = kill_fraction
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.cfg = config or RLConfig()
        self.generation = 0
        self.device = torch.device(device)

        # Compute obs_size from dataset
        alt_features = dataset["alt_features"]
        ind_features = dataset["indicator_features"]
        n_alts = alt_features.shape[1]
        n_alt_feat = alt_features.shape[2]
        n_ind_feat = ind_features.shape[1]
        self.n_alts = n_alts
        self.obs_size = (
            n_alts * n_alt_feat + n_ind_feat + self.cfg.N_PORTFOLIO_FEATURES
        )

        # Try loading from checkpoint
        self.agents: list[LSTMPPOAgent] = []
        self.scores: list[float] = []

        if self.checkpoint_dir is not None:
            result = load_checkpoint(
                self.checkpoint_dir, self.obs_size, self.n_alts
            )
            if result is not None:
                self.agents, self.scores, meta = result
                self.generation = meta.get("generation", 0)

        # Move loaded agents to device
        for agent in self.agents:
            agent.device = self.device
            agent.to(self.device)

        # Fill up to population_size
        while len(self.agents) < self.population_size:
            self.agents.append(
                LSTMPPOAgent(
                    obs_size=self.obs_size, n_alts=self.n_alts, device=self.device,
                )
            )
            self.scores.append(0.0)

    def evaluate_agent(
        self, agent: LSTMPPOAgent, n_episodes: int | None = None
    ) -> float:
        """Evaluate an agent over multiple episodes with PPO updates.

        Returns mean episode reward.
        """
        n_episodes = n_episodes or self.episodes_per_agent
        env = CryptoTradingEnv(self.dataset, config=self.cfg)
        episode_rewards: list[float] = []

        for _ in range(n_episodes):
            obs = env.reset()
            agent.reset_hidden()
            buf = PPOBuffer()
            done = False
            ep_reward = 0.0

            while not done:
                action, log_prob, value = agent.act(obs)
                next_obs, reward, done, _info = env.step(action)

                buf.add(
                    obs=torch.as_tensor(obs, dtype=torch.float32),
                    action=torch.as_tensor(action, dtype=torch.float32),
                    log_prob=log_prob,
                    reward=reward,
                    value=value,
                    done=done,
                )
                obs = next_obs
                ep_reward += reward

            episode_rewards.append(ep_reward)

            # PPO update after each episode
            if len(buf.obs) > 0:
                batch = buf.get()
                ppo_update(agent, batch)

        return float(np.mean(episode_rewards))

    def evolve(self) -> None:
        """Evolutionary selection: kill bottom agents, replace with mutated top agents."""
        kill_count = int(self.population_size * self.kill_fraction)
        if kill_count == 0:
            return

        # Rank by score ascending — bottom indices are worst
        ranked_indices = np.argsort(self.scores)
        dead_indices = ranked_indices[:kill_count].tolist()
        top_indices = ranked_indices[kill_count:].tolist()

        for dead_idx in dead_indices:
            # Pick a random survivor to clone
            parent_idx = top_indices[
                np.random.randint(len(top_indices))
            ]
            self.agents[dead_idx] = copy.deepcopy(self.agents[parent_idx])
            self.agents[dead_idx].lstm.flatten_parameters()

            # Add mutation noise to all parameters
            with torch.no_grad():
                for param in self.agents[dead_idx].parameters():
                    param.add_(
                        torch.randn_like(param) * self.mutation_noise
                    )

            self.scores[dead_idx] = 0.0

    def run_generation(self) -> dict[str, Any]:
        """Run one generation: evaluate, evolve, checkpoint.

        Returns stats dict with generation, best/mean/worst reward, elapsed_s.
        """
        self.generation += 1
        t0 = time.time()

        # Evaluate all agents
        for i, agent in enumerate(self.agents):
            self.scores[i] = self.evaluate_agent(agent)

        # Evolve
        self.evolve()

        # Checkpoint
        if self.checkpoint_dir is not None:
            save_checkpoint(
                self.checkpoint_dir,
                self.agents,
                self.scores,
                {
                    "generation": self.generation,
                    "population_size": self.population_size,
                    "best_reward": float(max(self.scores)),
                },
            )

        elapsed = time.time() - t0
        return {
            "generation": self.generation,
            "best_reward": float(max(self.scores)),
            "mean_reward": float(np.mean(self.scores)),
            "worst_reward": float(min(self.scores)),
            "elapsed_s": elapsed,
        }

    def train(self, n_generations: int) -> list[dict[str, Any]]:
        """Run multiple generations and return stats list."""
        stats_list: list[dict[str, Any]] = []
        gen_times: list[float] = []
        for i in range(n_generations):
            stats = self.run_generation()
            stats_list.append(stats)
            gen_times.append(stats["elapsed_s"])

            avg_time = sum(gen_times) / len(gen_times)
            remaining = (n_generations - i - 1) * avg_time
            eta_min = remaining / 60

            logger.info(
                f"Gen {stats['generation']:>4}/{self.generation + n_generations - i - 1:>4} | "
                f"best={stats['best_reward']:>8.4f} mean={stats['mean_reward']:>8.4f} | "
                f"{stats['elapsed_s']:.1f}s/gen | ETA {eta_min:.0f}min"
            )
        return stats_list

    def get_best_agent(self) -> LSTMPPOAgent:
        """Return the agent with the highest score."""
        best_idx = int(np.argmax(self.scores))
        return self.agents[best_idx]
