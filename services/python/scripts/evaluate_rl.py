"""Evaluate trained RL agent on holdout period.

Usage: cd services/python && python scripts/evaluate_rl.py
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
                        help="Fraction of data where holdout starts")
    args = parser.parse_args()

    project = Path("C:/Users/togat/Desktop/AI-Finance")
    data_path = Path(args.data) if args.data else project / "data/features/rl_dataset.npz"
    ckpt_dir = Path(args.checkpoint) if args.checkpoint else project / "models/rl_checkpoints"

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
    logger.info(f"Total: {n_times}, holdout from index {holdout_idx}")

    holdout = {k: v[holdout_idx:] if hasattr(v, '__getitem__') and k != "alt_names" else v
               for k, v in dataset.items()}

    from src.rl.agent import LSTMPPOAgent
    from src.rl.config import RLConfig
    from src.rl.env import CryptoTradingEnv

    cfg = RLConfig()
    n_alts = len(holdout["alt_names"])
    obs_size = (n_alts * holdout["alt_features"].shape[2]
                + holdout["indicator_features"].shape[1]
                + cfg.N_PORTFOLIO_FEATURES)

    agent = LSTMPPOAgent(obs_size, n_alts)
    best_path = ckpt_dir / "best_agent.pt"
    if not best_path.exists():
        logger.error(f"No best_agent.pt at {best_path}")
        sys.exit(1)
    agent.load_state_dict(torch.load(best_path, weights_only=True))
    agent.eval()

    env = CryptoTradingEnv(holdout, cfg)
    obs = env.reset(start_idx=0)
    agent.reset_hidden()

    done = False
    returns = []
    while not done:
        action, _, _ = agent.act(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        returns.append(reward)

    returns = np.array(returns)
    equity = np.cumprod(1 + returns)
    total_ret = float(equity[-1] - 1) * 100
    mean_r, std_r = returns.mean(), returns.std()
    sharpe = (mean_r / std_r) * np.sqrt(730) if std_r > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min((equity - peak) / peak)) * 100
    win_rate = float((returns > 0).sum() / max(len(returns), 1)) * 100

    go = sharpe > 0.5 and total_ret > 0 and max_dd > -25 and win_rate > 48

    print("\n" + "=" * 60)
    print("HOLDOUT EVALUATION")
    print("=" * 60)
    print(f"  Sharpe:      {sharpe:.3f}")
    print(f"  Return:      {total_ret:.2f}%")
    print(f"  Max DD:      {max_dd:.2f}%")
    print(f"  Win Rate:    {win_rate:.2f}%")
    print(f"  Windows:     {len(returns)}")
    print(f"  Final Value: {info['portfolio_value']:,.0f} IDR")
    print(f"\n  VERDICT: {'GO' if go else 'NO-GO'}")
    print("=" * 60)

    with open(ckpt_dir / "holdout_results.json", "w") as f:
        json.dump({"sharpe": round(sharpe, 3), "total_return_pct": round(total_ret, 2),
                    "max_drawdown_pct": round(max_dd, 2), "win_rate": round(win_rate, 2),
                    "go": go}, f, indent=2)


if __name__ == "__main__":
    main()
