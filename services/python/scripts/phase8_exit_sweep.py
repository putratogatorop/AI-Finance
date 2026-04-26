"""Phase 8.2 — exit-rule sweep on the SHORT-only bigmover ml_on+sized pipeline.

Reuses entries from Phase-5's trades.csv (which has every short entry's
symbol + entry_time + ml_proba + pos_scale). Re-simulates each entry with
7 alternative exit configurations (parameter sweep). The locked classifier
threshold and locked sizing config stay frozen — only exit rules vary.

Locks the winning exit on TRAIN-period total sized PnL (NOT OOT — the
protocol mandates locking on train). OOT is read-only diagnosis.

Run from services/python/:
    python scripts/phase8_exit_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DATE = "2026-04-01"
CANDLES_PATH = REPO_ROOT / "data/snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
PHASE5_RUN_DIR = (
    REPO_ROOT / "services/python/results"
    / "backtest_bigmover_combined_with_ml_18aa3e0b_20260425T165448Z"
)
TRADES_CSV = PHASE5_RUN_DIR / "trades.csv"
OUT_PATH = PHASE5_RUN_DIR / "phase8_exit_sweep.json"

FRICTION_PCT = 0.0015
WF_FOLD_MONTHS = 3
RANDOM_SEED = 42
MC_N_RESAMPLES = 5000


# Exit-config grid. Each config has:
#   sl: stop-loss as fractional adverse move (positive number)
#   tp: take-profit as fractional favorable move (positive number, or None for trail-only)
#   timeout: max bars in trade
#   trail: trailing-stop fractional pullback from best favorable (positive, or None)
#   trail_activation: required favorable move before trail activates (or None for immediate)
EXIT_CONFIGS = {
    "E0_baseline_5_15_672":      dict(sl=0.05, tp=0.15, timeout=672, trail=None, trail_activation=None),
    "E1_tight_tp_5_8_672":       dict(sl=0.05, tp=0.08, timeout=672, trail=None, trail_activation=None),
    "E2_symmetric_5_5_672":      dict(sl=0.05, tp=0.05, timeout=672, trail=None, trail_activation=None),
    "E3_fast_timeout_5_15_96":   dict(sl=0.05, tp=0.15, timeout=96,  trail=None, trail_activation=None),
    "E4_trail_3pct":             dict(sl=0.05, tp=None, timeout=672, trail=0.03, trail_activation=0.0),
    "E5_trail_2pct_act_2pct":    dict(sl=0.05, tp=None, timeout=672, trail=0.02, trail_activation=0.02),
    "E6_tighter_sl_3_8_672":     dict(sl=0.03, tp=0.08, timeout=672, trail=None, trail_activation=None),
}


# --- Helpers ----------------------------------------------------------------


def _pf(pnls: np.ndarray) -> float:
    p = np.asarray(pnls, dtype=float)
    if len(p) == 0:
        return float("nan")
    w = p[p > 0].sum()
    l = -p[p < 0].sum()
    return float("inf") if l == 0 else float(w / l)


def _cell_metrics(pnls: np.ndarray) -> dict:
    if len(pnls) == 0:
        return {"trades": 0, "wr": 0.0, "pf": float("nan"),
                "avg_pnl_pct": 0.0, "total_pnl_pct": 0.0}
    return {
        "trades": int(len(pnls)),
        "wr": float((pnls > 0).mean()),
        "pf": _pf(pnls),
        "avg_pnl_pct": float(pnls.mean()),
        "total_pnl_pct": float(pnls.sum()),
    }


def _simulate_short_with_exit(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, open_: np.ndarray,
    enter_bar: int,
    *,
    sl: float, tp: float | None, timeout: int,
    trail: float | None, trail_activation: float | None,
) -> dict | None:
    """Generic short-side simulator with optional trailing stop.

    Trade is opened at `open_[enter_bar]` (already shifted +1 from signal bar
    in the source trades.csv). Exit semantics, in priority order per bar:
        1. Hard SL: high[bar] >= entry * (1 + sl) -> -sl - friction
        2. Hard TP: low[bar] <= entry * (1 - tp) -> +tp - friction (if tp not None)
        3. Trail stop: tracks best favorable (lowest low). After
           `trail_activation` favorable move, exits when high[bar] crosses
           best_low * (1 + trail). Counts as -delta favorable - friction.
        4. Otherwise continue to next bar.
    On timeout, exit at close[last]. Friction subtracted on every trade.
    """
    n = len(close)
    if enter_bar >= n:
        return None
    entry_price = float(open_[enter_bar])
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    sl_price = entry_price * (1 + sl)
    tp_price = entry_price * (1 - tp) if tp is not None else None
    last_bar = min(enter_bar + 1 + timeout, n)
    best_low = entry_price
    trail_armed = (trail is not None and (trail_activation is None or trail_activation == 0.0))
    for bar in range(enter_bar + 1, last_bar):
        # 1. SL
        if high[bar] >= sl_price:
            return {
                "pnl_pct": -sl - FRICTION_PCT,
                "exit_reason": "stop_loss",
                "bars_held": int(bar - enter_bar),
            }
        # 2. Hard TP
        if tp_price is not None and low[bar] <= tp_price:
            return {
                "pnl_pct": tp - FRICTION_PCT,
                "exit_reason": "take_profit",
                "bars_held": int(bar - enter_bar),
            }
        # Update best favorable
        if low[bar] < best_low:
            best_low = float(low[bar])
        # Activate trail once favorable move >= trail_activation
        if trail is not None and not trail_armed and trail_activation is not None:
            favorable_pct = (entry_price - best_low) / entry_price
            if favorable_pct >= trail_activation:
                trail_armed = True
        # 3. Trail (only after activation)
        if trail is not None and trail_armed:
            trail_trigger = best_low * (1 + trail)
            if high[bar] >= trail_trigger:
                return {
                    "pnl_pct": (entry_price - trail_trigger) / entry_price - FRICTION_PCT,
                    "exit_reason": "trail_stop",
                    "bars_held": int(bar - enter_bar),
                }
    # Timeout
    exit_bar = last_bar - 1
    return {
        "pnl_pct": (entry_price - float(close[exit_bar])) / entry_price - FRICTION_PCT,
        "exit_reason": "timeout",
        "bars_held": int(exit_bar - enter_bar),
    }


def _resimulate_combo(
    entries: pd.DataFrame, candles: pd.DataFrame, *, exit_cfg: dict,
) -> pd.DataFrame:
    """Re-simulate every short entry under the supplied exit_cfg."""
    cs = candles.sort_values(["asset", "timestamp"])
    by_symbol = {}
    for asset, sub in cs.groupby("asset", sort=False):
        ts = pd.to_datetime(sub["timestamp"], utc=True).dt.tz_convert(None).to_numpy()
        by_symbol[asset] = (
            ts,
            sub["open"].to_numpy(dtype=float),
            sub["high"].to_numpy(dtype=float),
            sub["low"].to_numpy(dtype=float),
            sub["close"].to_numpy(dtype=float),
        )
    rows = []
    for r in entries.itertuples(index=False):
        sym = r.symbol
        if sym not in by_symbol:
            continue
        ts_arr, open_, high, low, close = by_symbol[sym]
        et = pd.Timestamp(r.entry_time)
        if et.tzinfo is not None:
            et = et.tz_convert(None)
        # entry_time in trades.csv is the OPEN of the entered bar.
        # Find that index.
        target = np.datetime64(et)
        i = int(np.searchsorted(ts_arr, target))
        if i >= len(ts_arr) or ts_arr[i] != target:
            # If not exactly aligned, fall back to nearest >=.
            if i >= len(ts_arr):
                continue
        out = _simulate_short_with_exit(
            close, high, low, open_, i,
            sl=exit_cfg["sl"], tp=exit_cfg["tp"], timeout=exit_cfg["timeout"],
            trail=exit_cfg["trail"], trail_activation=exit_cfg["trail_activation"],
        )
        if out is None:
            continue
        out["symbol"] = sym
        out["entry_time"] = str(r.entry_time)
        out["proba"] = float(r.proba)
        out["pos_scale"] = float(r.pos_scale)
        out["sized_pnl"] = out["pnl_pct"] * out["pos_scale"]
        out["period"] = r.period
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    started = time.monotonic()

    print(f"loading Phase-5 trades from {TRADES_CSV}", flush=True)
    df = pd.read_csv(TRADES_CSV)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)

    # Filter to SHORT-only ml_on+sized (the deployable subset).
    threshold = float(
        df[(df["period"] == "train") & df["proba"].notna()]["proba"].quantile(0.5)
    )
    print(f"  locked predictor threshold: {threshold:.4f}", flush=True)
    short_full = df[
        (df["direction"] == "short")
        & (df["proba"] >= threshold)
        & (df["pos_scale"] > 0)
    ][["symbol", "entry_time", "proba", "pos_scale", "period"]].reset_index(drop=True)
    print(f"  short ml_on+sized entries: {len(short_full):,}", flush=True)

    print(f"\nloading candles from {CANDLES_PATH}", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)

    # ---- Sweep ---------------------------------------------------------
    sweep_results = {}
    print(f"\n=== Sweeping {len(EXIT_CONFIGS)} exit configs on {len(short_full):,} entries ===")
    for name, cfg in EXIT_CONFIGS.items():
        t0 = time.monotonic()
        sim = _resimulate_combo(short_full, candles, exit_cfg=cfg)
        wall = time.monotonic() - t0
        train_pnls = sim[sim["period"] == "train"]["sized_pnl"].to_numpy()
        oot_pnls = sim[sim["period"] == "oot"]["sized_pnl"].to_numpy()
        sweep_results[name] = {
            "config": cfg,
            "train_metrics": _cell_metrics(train_pnls),
            "oot_metrics": _cell_metrics(oot_pnls),
            "wall_seconds": wall,
        }
        tm = sweep_results[name]["train_metrics"]
        om = sweep_results[name]["oot_metrics"]
        print(f"  {name:<28}  train PF={tm['pf']:.3f} (n={tm['trades']:>5})  "
              f"OOT PF={om['pf']:.3f} (n={om['trades']:>4})  "
              f"OOT avg={om['avg_pnl_pct']*100:+.3f}%  "
              f"wall={wall:.1f}s", flush=True)

    # ---- Lock winner on TRAIN total_pnl_pct ----------------------------
    locked_name = max(
        sweep_results,
        key=lambda k: (
            sweep_results[k]["train_metrics"]["total_pnl_pct"]
            if np.isfinite(sweep_results[k]["train_metrics"]["pf"]) else -np.inf
        ),
    )
    locked_cfg = EXIT_CONFIGS[locked_name]
    print(f"\nLOCKED on TRAIN total_pnl: {locked_name}", flush=True)
    print(f"  config: {locked_cfg}", flush=True)
    print(f"  train: PF={sweep_results[locked_name]['train_metrics']['pf']:.3f}  "
          f"total_pnl={sweep_results[locked_name]['train_metrics']['total_pnl_pct']*100:+.0f}%",
          flush=True)
    print(f"  OOT:   PF={sweep_results[locked_name]['oot_metrics']['pf']:.3f}  "
          f"n={sweep_results[locked_name]['oot_metrics']['trades']}", flush=True)

    # Train -> OOT gap (overfit warning)
    train_oot_gap = {
        name: {
            "train_pf": r["train_metrics"]["pf"],
            "oot_pf": r["oot_metrics"]["pf"],
            "delta_pf": (
                r["oot_metrics"]["pf"] - r["train_metrics"]["pf"]
                if np.isfinite(r["oot_metrics"]["pf"]) and np.isfinite(r["train_metrics"]["pf"])
                else float("nan")
            ),
        }
        for name, r in sweep_results.items()
    }

    out = {
        "predictor_threshold": threshold,
        "exit_configs": EXIT_CONFIGS,
        "sweep_results": sweep_results,
        "locked_winner": {
            "name": locked_name,
            "config": locked_cfg,
            "train_metrics": sweep_results[locked_name]["train_metrics"],
            "oot_metrics": sweep_results[locked_name]["oot_metrics"],
        },
        "train_oot_pf_gap": train_oot_gap,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
