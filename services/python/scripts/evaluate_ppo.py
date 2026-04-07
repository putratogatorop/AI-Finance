"""Evaluate trained PPO agent on holdout data.

Usage (local):
    cd services/python
    python scripts/evaluate_ppo.py --top-n 20

Usage (with checkpoint from Drive):
    python scripts/evaluate_ppo.py --data /path/to/rl_dataset.npz \
        --agent /path/to/rl_checkpoints/best_agent.pt --top-n 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_episode(agent, env, start_idx: int) -> dict:
    """Run one deterministic episode from a fixed start index."""
    obs = env.reset(start_idx=start_idx)
    agent.reset_hidden()
    done = False

    step_returns = []
    trades_opened = 0
    trades_closed = 0
    positions_log = []

    starting_equity = env.cfg.STARTING_CAPITAL

    while not done:
        action, _, _ = agent.act(obs, deterministic=True)
        prev_positions = set(env.portfolio.positions.keys())
        obs, reward, done, info = env.step(action)
        new_positions = set(env.portfolio.positions.keys())

        opened = new_positions - prev_positions
        closed = prev_positions - new_positions
        trades_opened += len(opened)
        trades_closed += len(closed)

        step_returns.append(env.episode_returns[-1] if env.episode_returns else 0.0)

    final_equity = info["portfolio_value"]
    returns = np.array(step_returns)

    # Total return
    total_return_pct = ((final_equity - starting_equity) / starting_equity) * 100

    # Sharpe ratio (annualized, 2 decisions/day, 365 days)
    mean_r = returns.mean()
    std_r = returns.std()
    sharpe = (mean_r / std_r) * np.sqrt(730) if std_r > 1e-10 else 0.0

    # Max drawdown
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum - running_max
    max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    # Win rate (positive step returns)
    nonzero = returns[returns != 0]
    win_rate = (nonzero > 0).mean() * 100 if len(nonzero) > 0 else 0.0

    return {
        "start_idx": start_idx,
        "steps": info["step"],
        "total_return_pct": total_return_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "win_rate_pct": win_rate,
        "trades_opened": trades_opened,
        "trades_closed": trades_closed,
        "final_equity": final_equity,
    }


def compute_btc_buy_hold(dataset, start_idx: int, n_windows: int, candles_per_window: int) -> float:
    """Compute BTC buy-and-hold return for the same period."""
    # BTC is not in the filtered dataset, so use the original prices
    # We compute from the first alt's price period as a proxy
    # Actually, we need BTC prices from the indicator features
    # Since we don't have raw BTC price in the dataset, we'll skip this
    # and just report agent performance
    return float("nan")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--agent", type=str, default=None, help="Path to best_agent.pt")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--holdout-pct", type=float, default=0.2,
                        help="Last N%% of data used as holdout (default 20%%)")
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Number of holdout episodes to run")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    data_path = Path(args.data) if args.data else project_root / "data/features/rl_dataset.npz"
    ckpt_dir = project_root / "models/rl_checkpoints"

    if args.agent:
        agent_path = Path(args.agent)
    else:
        agent_path = ckpt_dir / "best_agent.pt"

    device = torch.device("cpu") if args.cpu else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    logger.info("=" * 60)
    logger.info("PPO AGENT EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Data:     {data_path}")
    logger.info(f"Agent:    {agent_path}")
    logger.info(f"Top-N:    {args.top_n}")
    logger.info(f"Holdout:  last {args.holdout_pct * 100:.0f}% of data")
    logger.info(f"Episodes: {args.n_episodes}")

    # Load dataset
    data = np.load(str(data_path), allow_pickle=True)
    dataset = {
        "alt_features": data["alt_features"],
        "indicator_features": data["indicator_features"],
        "prices": data["prices"],
        "alt_names": data["alt_names"].tolist(),
        "timestamps": data["timestamps"],
    }
    n_times = dataset["alt_features"].shape[0]
    logger.info(f"Dataset: {n_times} candles, {len(dataset['alt_names'])} alts")

    # Create env
    from src.rl.agent import LSTMPPOAgent
    from src.rl.config import RLConfig
    from src.rl.env import CryptoTradingEnv

    cfg = RLConfig()
    env = CryptoTradingEnv(dataset, config=cfg, top_n=args.top_n)
    logger.info(f"Env: {env.n_alts} alts, obs_size={env.obs_size}")

    # Load agent
    agent = LSTMPPOAgent(obs_size=env.obs_size, n_alts=env.n_alts, device=device)
    state_dict = torch.load(str(agent_path), weights_only=True, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    logger.info(f"Loaded agent: {sum(p.numel() for p in agent.parameters()):,} params")

    # Compute holdout range
    # Training uses random start between 200 and max_safe
    # Holdout = last holdout_pct of the data
    holdout_start = int(n_times * (1 - args.holdout_pct))
    max_safe = n_times - cfg.EPISODE_WINDOWS * cfg.CANDLES_PER_WINDOW - 1

    # Ensure holdout_start is valid
    holdout_start = min(holdout_start, max_safe)
    holdout_end = max_safe

    logger.info(f"Holdout candle range: {holdout_start} - {holdout_end}")
    logger.info(f"Holdout = candles {holdout_start}-{n_times} "
                f"({(n_times - holdout_start) * 4 / 24:.0f} days)")

    # Evenly space episodes across holdout range
    if args.n_episodes == 1:
        start_indices = [holdout_start]
    else:
        start_indices = np.linspace(
            holdout_start, holdout_end, args.n_episodes, dtype=int
        ).tolist()

    # Run episodes
    logger.info(f"\nRunning {args.n_episodes} holdout episodes (deterministic)...\n")
    results = []
    for i, start_idx in enumerate(start_indices):
        result = run_episode(agent, env, start_idx)
        results.append(result)
        logger.info(
            f"  Episode {i + 1:>2}/{args.n_episodes} | "
            f"start={start_idx:>5} | "
            f"return={result['total_return_pct']:>+7.2f}% | "
            f"sharpe={result['sharpe']:>6.2f} | "
            f"dd={result['max_drawdown_pct']:>5.1f}% | "
            f"win={result['win_rate_pct']:>5.1f}% | "
            f"trades={result['trades_opened']}"
        )

    # Summary
    avg_return = np.mean([r["total_return_pct"] for r in results])
    avg_sharpe = np.mean([r["sharpe"] for r in results])
    avg_dd = np.mean([r["max_drawdown_pct"] for r in results])
    avg_win = np.mean([r["win_rate_pct"] for r in results])
    avg_trades = np.mean([r["trades_opened"] for r in results])

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (holdout data)")
    print("=" * 60)
    print(f"  Episodes:       {args.n_episodes}")
    print(f"  Avg Return:     {avg_return:>+.2f}%")
    print(f"  Avg Sharpe:     {avg_sharpe:>.2f}")
    print(f"  Avg Max DD:     {avg_dd:>.1f}%")
    print(f"  Avg Win Rate:   {avg_win:>.1f}%")
    print(f"  Avg Trades:     {avg_trades:>.0f}")
    print()

    # Pass/fail criteria
    checks = {
        "Sharpe > 0.5": avg_sharpe > 0.5,
        "Return > 0%": avg_return > 0,
        "Max DD < 30%": avg_dd < 30,
        "Win Rate > 40%": avg_win > 40,
    }
    all_pass = all(checks.values())

    print("  PASS/FAIL CRITERIA:")
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {name}")

    print()
    if all_pass:
        print("  >>> ALL CHECKS PASSED — agent is a candidate for live testing")
    else:
        print("  >>> SOME CHECKS FAILED — needs more training or tuning")
    print("=" * 60)

    # Save results
    results_path = ckpt_dir / "evaluation_results.json" if not args.agent else Path(args.agent).parent / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "summary": {
                "avg_return_pct": float(avg_return),
                "avg_sharpe": float(avg_sharpe),
                "avg_max_drawdown_pct": float(avg_dd),
                "avg_win_rate_pct": float(avg_win),
                "avg_trades": float(avg_trades),
                "all_pass": all_pass,
            },
            "episodes": results,
        }, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
