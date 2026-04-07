"""Single-agent PPO training for RL crypto trading.

Usage (local):
    cd services/python
    python scripts/train_ppo.py --top-n 20

Usage (Colab):
    !python scripts/train_ppo.py --data /content/drive/MyDrive/ai-finance/rl_dataset.npz \
        --checkpoint /content/drive/MyDrive/ai-finance/rl_checkpoints \
        --top-n 20 --generations 200 --episodes 10
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        return dev
    logger.info("No GPU detected, using CPU")
    return torch.device("cpu")


def collect_episode(agent, env):
    """Run one episode, return rollout buffer and stats."""
    from src.rl.ppo import PPOBuffer

    obs = env.reset()
    agent.reset_hidden()
    buf = PPOBuffer()
    done = False
    ep_reward = 0.0
    positions_opened = 0

    while not done:
        action, log_prob, value = agent.act(obs)
        next_obs, reward, done, info = env.step(action)

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
        positions_opened = max(positions_opened, info["n_positions"])

    stats = {
        "reward": ep_reward,
        "steps": info["step"],
        "max_positions": positions_opened,
        "final_value": info["portfolio_value"],
        "max_drawdown": info["drawdown"],
    }
    return buf, stats


def main():
    parser = argparse.ArgumentParser(description="Single-agent PPO Training")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--ppo-batch", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_path = Path(args.data) if args.data else project_root / "data/features/rl_dataset.npz"
    ckpt_dir = Path(args.checkpoint) if args.checkpoint else project_root / "models/rl_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu") if args.cpu else detect_device()

    logger.info("=" * 60)
    logger.info("SINGLE-AGENT PPO TRAINING (v3.1)")
    logger.info("=" * 60)
    logger.info(f"Device:      {device}")
    logger.info(f"Data:        {data_path}")
    logger.info(f"Checkpoint:  {ckpt_dir}")
    logger.info(f"Top-N alts:  {args.top_n}")
    logger.info(f"Generations: {args.generations}, Episodes/gen: {args.episodes}")
    logger.info(f"PPO epochs:  {args.ppo_epochs}, batch: {args.ppo_batch}, lr: {args.lr}")

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
    logger.info(f"Loaded in {time.time() - t0:.1f}s: {dataset['alt_features'].shape}")

    # Create env and agent
    from src.rl.agent import LSTMPPOAgent
    from src.rl.env import CryptoTradingEnv
    from src.rl.ppo import PPOBuffer, ppo_update

    env = CryptoTradingEnv(dataset, top_n=args.top_n)
    logger.info(f"Env: {env.n_alts} alts, obs_size={env.obs_size}, action_size={env.n_alts * 3}")

    # Try loading checkpoint
    agent_path = ckpt_dir / "ppo_agent.pt"
    meta_path = ckpt_dir / "ppo_meta.json"
    start_gen = 0

    agent = LSTMPPOAgent(
        obs_size=env.obs_size, n_alts=env.n_alts, device=device,
    )

    if agent_path.exists() and meta_path.exists():
        state_dict = torch.load(agent_path, weights_only=True, map_location=device)
        agent.load_state_dict(state_dict)
        with open(meta_path) as f:
            meta = json.load(f)
        start_gen = meta.get("generation", 0)
        logger.info(f"Resumed from generation {start_gen}")
    else:
        logger.info("Starting fresh agent")

    param_count = sum(p.numel() for p in agent.parameters())
    logger.info(f"Parameters: {param_count:,}")

    # Training loop
    t_start = time.time()
    gen_times = []
    best_reward = float("-inf")

    for gen in range(start_gen, start_gen + args.generations):
        t_gen = time.time()

        # Collect episodes
        all_buffers = []
        ep_stats = []
        for _ in range(args.episodes):
            buf, stats = collect_episode(agent, env)
            all_buffers.append(buf)
            ep_stats.append(stats)

        # Merge buffers into one big batch
        merged = PPOBuffer()
        for buf in all_buffers:
            for i in range(len(buf.obs)):
                merged.add(
                    obs=buf.obs[i],
                    action=buf.actions[i],
                    log_prob=buf.log_probs[i],
                    reward=buf.rewards[i],
                    value=buf.values[i],
                    done=buf.dones[i],
                )

        # PPO update on merged batch
        batch = merged.get()
        ppo_stats = ppo_update(
            agent, batch,
            epochs=args.ppo_epochs,
            batch_size=args.ppo_batch,
            lr=args.lr,
        )

        # Stats
        mean_reward = np.mean([s["reward"] for s in ep_stats])
        mean_positions = np.mean([s["max_positions"] for s in ep_stats])
        mean_dd = np.mean([s["max_drawdown"] for s in ep_stats])
        mean_steps = np.mean([s["steps"] for s in ep_stats])
        elapsed = time.time() - t_gen
        gen_times.append(elapsed)

        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(agent.cpu().state_dict(), ckpt_dir / "best_agent.pt")
            agent.to(device)

        # Checkpoint every generation
        torch.save(agent.cpu().state_dict(), agent_path)
        agent.to(device)
        with open(meta_path, "w") as f:
            json.dump({"generation": gen + 1, "best_reward": best_reward}, f)

        avg_time = sum(gen_times) / len(gen_times)
        remaining = (args.generations - (gen - start_gen) - 1) * avg_time
        eta_min = remaining / 60

        logger.info(
            f"Gen {gen + 1:>4} | "
            f"reward={mean_reward:>8.3f} best={best_reward:>8.3f} | "
            f"pos={mean_positions:.1f} dd={mean_dd:.3f} steps={mean_steps:.0f} | "
            f"ploss={ppo_stats['policy_loss']:.4f} vloss={ppo_stats['value_loss']:.4f} "
            f"ent={ppo_stats['entropy']:.4f} | "
            f"{elapsed:.1f}s ETA {eta_min:.0f}min"
        )

    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  Best reward: {best_reward:.4f}")
    print(f"  Checkpoint:  {ckpt_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
