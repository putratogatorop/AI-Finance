"""Layered exit backtest: SHORT-only, bar-by-bar sim with trailing stops."""
import sys

sys.path.insert(0, ".")

import glob
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
DATA_DIR = "../../data/raw/15m"
THRESHOLD = 0.90
BARS_24H = 96  # 24h of 15m candles


# ---------------------------------------------------------------------------
# Task 1 — CSV Loader
# ---------------------------------------------------------------------------


def load_symbol_bars(symbol: str) -> pd.DataFrame:
    """Load all 15m CSVs for *symbol*, concat, convert open_time, sort."""
    pattern = f"{DATA_DIR}/{symbol}/{symbol}-15m-*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found for {symbol} at {pattern}")

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    # Older CSVs use milliseconds (13 digits), newer ones microseconds (16 digits).
    # Normalise everything to microseconds before converting.
    ts = df["open_time"].values.copy()
    mask_ms = ts < 1_000_000_000_000_000  # < 1e15 → milliseconds
    ts[mask_ms] *= 1000  # ms → us
    df["open_time"] = pd.to_datetime(ts, unit="us", utc=True)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_bars_after_signal(
    bars_df: pd.DataFrame, signal_time, n_bars: int = 96
) -> np.ndarray | None:
    """Return (n_bars, 3) array of [high, low, close] starting AFTER signal_time.

    Uses searchsorted for O(log n) lookup. Returns None if not enough bars remain.
    """
    idx = bars_df["open_time"].searchsorted(signal_time, side="right")
    if idx + n_bars > len(bars_df):
        return None
    chunk = bars_df.iloc[idx : idx + n_bars]
    return chunk[["high", "low", "close"]].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Task 2 — Single-Trade Layered Sim
# ---------------------------------------------------------------------------


@dataclass
class LayerConfig:
    weight: float
    sl: float
    tp: float
    trail_act: float
    trail_dist: float


def sim_trade(
    entry: float, bars: np.ndarray, layers: list[LayerConfig]
) -> tuple[float, str, str]:
    """Simulate a SHORT trade with layered exits.

    Parameters
    ----------
    entry : float
        Entry price.
    bars : np.ndarray
        Shape (n_bars, 3) with columns [high, low, close].
    layers : list[LayerConfig]
        Exit layers with weight, SL, TP, trailing activation & distance.

    Returns
    -------
    total_pnl : float
        Weighted sum of per-layer PnL (as fraction, e.g. 0.05 = +5 %).
    exit_reason : str
        Dominant exit type by weight.
    exit_detail : str
        Per-layer breakdown like "L1:sl_hit-1.5 L2:tp_hit+6.0 L3:trail_hit+4.5".
    """
    n_layers = len(layers)
    closed = [False] * n_layers
    layer_pnl = [0.0] * n_layers
    layer_reason = ["time_exit"] * n_layers
    best_fav = [0.0] * n_layers  # best favorable move seen per layer

    for bar in bars:
        bar_high, bar_low, bar_close = bar[0], bar[1], bar[2]
        adverse = (bar_high - entry) / entry  # price UP → bad for short
        favorable = (entry - bar_low) / entry  # price DOWN → good for short

        for i, layer in enumerate(layers):
            if closed[i]:
                continue

            # 1. SL check (conservative — SL first within same bar)
            if adverse >= layer.sl:
                layer_pnl[i] = -layer.sl
                layer_reason[i] = "sl_hit"
                closed[i] = True
                continue

            # 2. Trailing stop check
            if best_fav[i] >= layer.trail_act:
                # Trailing is active — check retracement from best
                current_fav_from_high = (entry - bar_high) / entry
                retracement = best_fav[i] - current_fav_from_high
                if retracement >= layer.trail_dist:
                    layer_pnl[i] = best_fav[i] - layer.trail_dist
                    layer_reason[i] = "trail_hit"
                    closed[i] = True
                    continue

            # 3. TP check
            if favorable >= layer.tp:
                layer_pnl[i] = layer.tp
                layer_reason[i] = "tp_hit"
                closed[i] = True
                continue

            # 4. Update best favorable
            if favorable > best_fav[i]:
                best_fav[i] = favorable

        # Early exit if all layers closed
        if all(closed):
            break

    # Time-exit remaining layers at last close
    last_close = bars[-1, 2]
    for i in range(n_layers):
        if not closed[i]:
            layer_pnl[i] = (entry - last_close) / entry
            layer_reason[i] = "time_exit"
            closed[i] = True

    # Aggregate
    total_pnl = sum(layers[i].weight * layer_pnl[i] for i in range(n_layers))

    # Dominant exit reason by weight
    reason_weights: dict[str, float] = {}
    for i in range(n_layers):
        reason_weights[layer_reason[i]] = (
            reason_weights.get(layer_reason[i], 0.0) + layers[i].weight
        )
    exit_reason = max(reason_weights, key=reason_weights.get)  # type: ignore[arg-type]

    # Detail string
    parts = []
    for i in range(n_layers):
        parts.append(f"L{i+1}:{layer_reason[i]}{layer_pnl[i]*100:+.1f}")
    exit_detail = " ".join(parts)

    return total_pnl, exit_reason, exit_detail


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Smoke test CSV loader
    bars = load_symbol_bars("ENJUSDT")
    logger.info(
        f"ENJUSDT: {len(bars)} bars, {bars['open_time'].min()} to {bars['open_time'].max()}"
    )
    test_time = bars["open_time"].iloc[100]
    chunk = get_bars_after_signal(bars, test_time, 96)
    logger.info(f"Chunk shape: {chunk.shape if chunk is not None else 'None'}")

    # Test sim_trade with synthetic bars
    entry = 100.0
    test_bars = np.array(
        [
            [100.5, 99.5, 99.8],  # bar 1: slight dip, slight spike
            [100.3, 98.0, 98.5],  # bar 2: drops to 98 (2% favorable)
            [99.0, 95.0, 95.5],   # bar 3: drops to 95 (5% favorable)
            [97.0, 94.5, 96.0],   # bar 4: bounces to 97 from 94.5
            [98.0, 95.5, 97.5],   # bar 5: retraces to 98
        ]
    )

    layers = [
        LayerConfig(weight=0.25, sl=0.02, tp=0.03, trail_act=0.02, trail_dist=0.01),
        LayerConfig(weight=0.50, sl=0.03, tp=0.06, trail_act=0.04, trail_dist=0.02),
        LayerConfig(weight=0.25, sl=0.05, tp=0.10, trail_act=0.06, trail_dist=0.03),
    ]

    pnl, reason, detail = sim_trade(entry, test_bars, layers)
    logger.info(f"PnL: {pnl*100:+.2f}%  Reason: {reason}  Detail: {detail}")
