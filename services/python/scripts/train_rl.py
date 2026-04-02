"""RL Population-Based Training.

Usage:
    cd services/python
    python scripts/train_rl.py
    python scripts/train_rl.py --population 20 --generations 500
    python scripts/train_rl.py --checkpoint /content/drive/MyDrive/ai-finance/rl_checkpoints
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RL Population Training")
    parser.add_argument("--data", type=str, default=None, help="Path to rl_dataset.npz")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--mutation", type=float, default=0.05)
    args = parser.parse_args()

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

    from src.rl.population import PopulationTrainer

    trainer = PopulationTrainer(
        dataset=dataset,
        population_size=args.population,
        episodes_per_agent=args.episodes,
        mutation_noise=args.mutation,
        checkpoint_dir=ckpt_dir,
    )

    logger.info(f"\nStarting from generation {trainer.generation}...")
    all_stats = trainer.train(n_generations=args.generations)

    best = trainer.get_best_agent()
    torch.save(best.state_dict(), ckpt_dir / "best_agent.pt")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for s in all_stats[-5:]:
        print(f"  Gen {s['generation']:>4}: best={s['best_reward']:>8.4f}, mean={s['mean_reward']:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
