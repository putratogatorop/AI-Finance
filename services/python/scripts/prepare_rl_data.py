"""One-time script: build RL dataset from raw CSVs and save as .npz.

Usage: cd services/python && python scripts/prepare_rl_data.py
Output: data/features/rl_dataset.npz
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from src.rl.data_loader import RLDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = Path("C:/Users/togat/Desktop/AI-Finance")
RAW_DIR = PROJECT / "data/raw/4h"
OUTPUT_PATH = PROJECT / "data/features/rl_dataset.npz"

def main():
    logger.info("Building RL dataset from raw CSVs...")
    t0 = time.time()
    loader = RLDataLoader(raw_dir=RAW_DIR, max_tokens=200)
    dataset = loader.build_dataset()

    logger.info(f"Saving to {OUTPUT_PATH}...")
    np.savez_compressed(
        OUTPUT_PATH,
        alt_features=dataset["alt_features"],
        indicator_features=dataset["indicator_features"],
        prices=dataset["prices"],
        alt_names=np.array(dataset["alt_names"]),
        timestamps=dataset["timestamps"],
    )

    elapsed = time.time() - t0
    shape = dataset["alt_features"].shape
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    logger.info(f"Done in {elapsed:.1f}s. Shape: {shape}, File: {size_mb:.1f} MB")
    logger.info(f"Alts: {len(dataset['alt_names'])}, Timestamps: {shape[0]}, Features/alt: {shape[2]}")

if __name__ == "__main__":
    main()
