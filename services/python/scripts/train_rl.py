"""RL Population-Based Training.

Usage (local):
    cd services/python
    python scripts/train_rl.py
    python scripts/train_rl.py --population 20 --generations 500

Usage (Colab — paste in a cell):
    !git clone https://github.com/<you>/AI-Finance.git
    %cd AI-Finance/services/python
    !pip install -q torch gymnasium numpy pandas
    !python scripts/train_rl.py --data ../../data/features/rl_dataset.npz \\
        --checkpoint /content/drive/MyDrive/ai-finance/rl_checkpoints
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


def detect_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return dev
    logger.info("No GPU detected, using CPU")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="RL Population Training")
    parser.add_argument("--data", type=str, default=None, help="Path to rl_dataset.npz")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--mutation", type=float, default=0.05)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU available")
    args = parser.parse_args()

    # Resolve paths relative to the script's project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # services/python/scripts -> AI-Finance
    data_path = Path(args.data) if args.data else project_root / "data/features/rl_dataset.npz"
    ckpt_dir = Path(args.checkpoint) if args.checkpoint else project_root / "models/rl_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu") if args.cpu else detect_device()

    logger.info("=" * 60)
    logger.info("RL POPULATION-BASED TRAINING")
    logger.info("=" * 60)
    logger.info(f"Device:     {device}")
    logger.info(f"Data:       {data_path}")
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
        device=device,
    )

    logger.info(f"\nStarting from generation {trainer.generation}...")
    logger.info(f"Obs size: {trainer.obs_size}, Alts: {trainer.n_alts}")
    t_start = time.time()

    all_stats = trainer.train(n_generations=args.generations)

    total_time = time.time() - t_start
    best = trainer.get_best_agent()
    best_path = ckpt_dir / "best_agent.pt"
    torch.save(best.cpu().state_dict(), best_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  Best agent: {best_path}")
    for s in all_stats[-5:]:
        print(
            f"  Gen {s['generation']:>4}: "
            f"best={s['best_reward']:>8.4f}, mean={s['mean_reward']:>8.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
