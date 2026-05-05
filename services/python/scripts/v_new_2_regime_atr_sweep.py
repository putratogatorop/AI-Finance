"""v_new_2 — Regime-conditional ATR multiplier sweep.

Goal: determine whether the optimal ATR trail multiplier varies by market regime
at entry time. If different regimes clearly prefer different multipliers, we can
build a simple regime-conditional exit policy that improves PF/PnL over a fixed
multiplier without full RL complexity.

Approach:
  1. Use existing trade entries from v_new_2_trades_scored_honest.parquet
     (bgm_honest >= 0.60, same as live filter) — avoids re-detecting signals
  2. Re-simulate exits bar-by-bar using features_v_new_2_v2.parquet
     (has close + atr14_pct per bar)
  3. Test 6 ATR multipliers: 1.5, 2, 3, 5, 8, 10
  4. Bucket entries into regime cells: cfgi (fear/neutral/greed) × btc_momentum (bear/bull)
  5. For each regime cell: which multiplier yields best sum_pnl + PF?
  6. Build a "conditional policy": assign each trade its regime-best multiplier
  7. Compare conditional vs fixed 2×, 3×, 5× on sum_pnl, PF, WR, max_DD

Outputs:
  results/v_new_2/regime_atr_sweep_raw.csv    — all trades × multipliers
  results/v_new_2/regime_atr_sweep.csv        — regime × multiplier aggregates
  results/v_new_2/regime_atr_verdict.md       — human-readable verdict
"""
from __future__ import annotations

import os
import pathlib
import time
from typing import Optional

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                        str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))
RESULTS = ROOT / "services" / "python" / "results" / "v_new_2"
RESULTS.mkdir(parents=True, exist_ok=True)

BGM_THRESHOLD   = 0.60
TIMEOUT_BARS    = 90 * 6          # 90 days × 6 bars/day on 4h
FRICTION_RT     = 0.003           # 0.1% fee + 0.2% slippage round-trip
FUNDING_LONG    = 0.0002 / 2      # per 4h bar (0.02% per 8h)
FUNDING_SHORT   = -0.00005 / 2

ATR_MULTS = [1.5, 2.0, 3.0, 5.0, 8.0, 10.0]

# Regime bucketing — two most predictive features from BGM SHAP analysis
CFGI_BINS   = [0, 35, 65, 100]
CFGI_LABELS = ["fear", "neutral", "greed"]
BTC_BINS    = [-999, -0.01, 0.01, 999]
BTC_LABELS  = ["bear", "flat", "bull"]


def _ts() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


def _regime_bucket(cfgi: float, btc_ret: float) -> str:
    if cfgi < 35:
        c = "fear"
    elif cfgi < 65:
        c = "neutral"
    else:
        c = "greed"
    if btc_ret < -0.01:
        b = "bear"
    elif btc_ret > 0.01:
        b = "bull"
    else:
        b = "flat"
    return f"{c}_{b}"


def _simulate_exit(
    bars_close: np.ndarray,
    bars_atr: np.ndarray,
    bars_trend: np.ndarray,         # +1 = in trend, 0/NaN = flipped
    entry_idx: int,
    side: str,
    trail_mult: float,
) -> tuple[int, str, float]:
    """Simulate ATR-trail exit from entry_idx forward.

    Returns (exit_idx, exit_reason, raw_pnl_frac).
    raw_pnl_frac is pre-friction — caller subtracts fees + funding.
    """
    n = len(bars_close)
    entry_price = bars_close[entry_idx]
    if np.isnan(entry_price) or entry_price <= 0:
        return entry_idx, "bad_entry", 0.0

    running_extreme = entry_price

    for i in range(entry_idx + 1, n):
        p = bars_close[i]
        if np.isnan(p):
            continue

        if side == "long":
            running_extreme = max(running_extreme, p)
            pnl_frac = p / entry_price - 1.0
        else:
            running_extreme = min(running_extreme, p)
            pnl_frac = entry_price / p - 1.0

        a = bars_atr[i] / 100.0 if not np.isnan(bars_atr[i]) else 0.02

        # ATR trail stop
        if side == "long":
            stop = running_extreme * (1.0 - trail_mult * a)
            if p <= stop:
                return i, "atr_trail", pnl_frac
        else:
            stop = running_extreme * (1.0 + trail_mult * a)
            if p >= stop:
                return i, "atr_trail", pnl_frac

        # Timeout
        if i - entry_idx >= TIMEOUT_BARS:
            return i, "timeout", pnl_frac

    return n - 1, "end_of_data", (bars_close[n-1] / entry_price - 1.0) if side == "long" else (entry_price / bars_close[n-1] - 1.0)


def _metrics(pnl_arr: np.ndarray) -> dict:
    if len(pnl_arr) == 0:
        return {"n": 0, "sum_pnl_pct": 0, "pf": 0, "wr": 0, "max_dd_pct": 0}
    wins   = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    # Use log-returns for equity curve to avoid float overflow on fat-tail trades
    log_ret = np.log1p(np.clip(pnl_arr, -0.99, 10.0))
    log_eq  = np.cumsum(log_ret)
    log_rm  = np.maximum.accumulate(log_eq)
    dd_log  = log_eq - log_rm          # always <= 0
    max_dd  = float(np.exp(dd_log.min()) - 1.0) * 100  # convert back to %
    return {
        "n":           int(len(pnl_arr)),
        "sum_pnl_pct": float(pnl_arr.sum() * 100),
        "pf":          float(wins.sum() / abs(losses.sum())) if len(losses) else float("inf"),
        "wr":          float((pnl_arr > 0).mean()),
        "max_dd_pct":  max_dd,
    }


def main():
    t0 = time.time()
    print(f"{_ts()} v_new_2 regime-conditional ATR sweep")

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"{_ts()} Loading trades_scored_honest …")
    trades = pd.read_parquet(DATA_DIR / "v_new_2_trades_scored_honest.parquet")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"],  utc=True)

    # Apply same live filter as paper executor
    trades = trades[trades["bgm_honest"] >= BGM_THRESHOLD].copy()
    print(f"  {len(trades):,} trades after bgm >= {BGM_THRESHOLD}")

    print(f"{_ts()} Loading features_v_new_2_v2 (bar-level) …")
    bars = pd.read_parquet(DATA_DIR / "features_v_new_2_v2.parquet",
                           columns=["symbol", "timestamp", "close", "atr14_pct",
                                    "d_trend_long", "d_trend_short"])
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Build per-symbol lookup: {symbol: (timestamps_arr, close_arr, atr_arr, trend_long_arr, trend_short_arr)}
    print(f"{_ts()} Building per-symbol bar index …")
    sym_data: dict[str, tuple] = {}
    for sym, g in bars.groupby("symbol", sort=False):
        g = g.reset_index(drop=True)
        # Store timestamps as int64 ns for tz-agnostic searchsorted.
        # Cast via datetime64[ns] to normalise parquet μs storage → ns.
        sym_data[sym] = (
            g["timestamp"].values.astype("datetime64[ns]").astype("int64"),
            g["close"].values.astype(float),
            g["atr14_pct"].values.astype(float),
            g["d_trend_long"].values.astype(float),
            g["d_trend_short"].values.astype(float),
        )

    # ── Regime bucketing ───────────────────────────────────────────────────────
    trades["regime"] = trades.apply(
        lambda r: _regime_bucket(
            float(r["cfgi_value"]) if not pd.isna(r["cfgi_value"]) else 50.0,
            float(r["btc_24h_return"]) if not pd.isna(r["btc_24h_return"]) else 0.0,
        ),
        axis=1,
    )
    print(f"\n{_ts()} Regime distribution:")
    print(trades["regime"].value_counts().to_string())
    print()

    # ── Re-simulate exits for each trade × ATR multiplier ─────────────────────
    print(f"{_ts()} Simulating exits ({len(trades):,} trades × {len(ATR_MULTS)} multipliers) …")
    raw_rows = []

    skipped = 0
    for idx, row in trades.iterrows():
        sym   = row["symbol"]
        side  = row["side"]
        entry = row["entry_time"]

        if sym not in sym_data:
            skipped += 1
            continue

        ts_arr, cl_arr, atr_arr, tl_arr, ts_arr2 = sym_data[sym]
        trend_arr = tl_arr if side == "long" else ts_arr2

        # Convert entry_time to int64 ns for tz-agnostic comparison
        entry_ns = int(entry.value)  # pandas Timestamp.value is int64 ns
        match = np.searchsorted(ts_arr, entry_ns)
        if match >= len(ts_arr):
            skipped += 1
            continue
        # Tolerate up to 1 bar off (4h = 14_400_000_000_000 ns)
        if abs(ts_arr[match] - entry_ns) > 6 * 3_600_000_000_000:
            skipped += 1
            continue

        entry_idx = int(match)
        bars_held_base = max(int(row["bars_held"]), 1)

        for mult in ATR_MULTS:
            exit_idx, reason, raw_pnl = _simulate_exit(
                cl_arr, atr_arr, trend_arr, entry_idx, side, mult
            )
            bars_held = exit_idx - entry_idx
            funding_rate = FUNDING_LONG if side == "long" else FUNDING_SHORT
            pnl_net = raw_pnl - FRICTION_RT - funding_rate * bars_held

            raw_rows.append({
                "symbol":     sym,
                "side":       side,
                "tier":       row["tier"],
                "is_meme":    bool(row["is_meme"]),
                "entry_time": entry,
                "regime":     row["regime"],
                "cfgi":       float(row["cfgi_value"]) if not pd.isna(row["cfgi_value"]) else 50.0,
                "btc_ret":    float(row["btc_24h_return"]) if not pd.isna(row["btc_24h_return"]) else 0.0,
                "breadth_up": float(row["breadth_up"]) if not pd.isna(row["breadth_up"]) else 0.5,
                "atr_mult":   mult,
                "pnl_pct":    pnl_net,
                "bars_held":  bars_held,
                "exit_reason": reason,
            })

    print(f"  Done. Skipped {skipped} trades (symbol not in bars data).")
    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(RESULTS / "regime_atr_sweep_raw.csv", index=False)
    print(f"  Raw results: {len(raw_df):,} rows → {RESULTS / 'regime_atr_sweep_raw.csv'}")

    # ── Aggregate: regime × multiplier ────────────────────────────────────────
    print(f"\n{_ts()} Aggregating by regime × ATR multiplier …")
    agg_rows = []
    regimes = sorted(raw_df["regime"].unique())
    for regime in regimes:
        for mult in ATR_MULTS:
            sub = raw_df[(raw_df["regime"] == regime) & (raw_df["atr_mult"] == mult)]
            if len(sub) == 0:
                continue
            m = _metrics(sub["pnl_pct"].values)
            m["regime"] = regime
            m["atr_mult"] = mult
            agg_rows.append(m)

    agg_df = pd.DataFrame(agg_rows)[["regime","atr_mult","n","sum_pnl_pct","pf","wr","max_dd_pct"]]
    agg_df.to_csv(RESULTS / "regime_atr_sweep.csv", index=False)

    # ── Best multiplier per regime ─────────────────────────────────────────────
    best_mult: dict[str, float] = {}
    for regime in regimes:
        sub = agg_df[agg_df["regime"] == regime]
        if len(sub) == 0:
            best_mult[regime] = 3.0
            continue
        # primary sort: sum_pnl_pct descending (absolute PnL first, per project rule)
        best_row = sub.sort_values("sum_pnl_pct", ascending=False).iloc[0]
        best_mult[regime] = float(best_row["atr_mult"])

    print(f"\n  Best ATR multiplier per regime:")
    for r, m in sorted(best_mult.items()):
        row_m = agg_df[(agg_df["regime"] == r) & (agg_df["atr_mult"] == m)].iloc[0]
        print(f"    {r:<18} → {m}×  (sum_pnl={row_m['sum_pnl_pct']:+.0f}%  PF={row_m['pf']:.2f}  n={row_m['n']})")

    # ── Conditional policy vs fixed baselines ─────────────────────────────────
    print(f"\n{_ts()} Comparing conditional policy vs fixed baselines …")

    # One row per trade (use the simulated exit, sorted by entry_time)
    base_trades = raw_df[raw_df["atr_mult"] == 2.0][["symbol","side","entry_time","regime","pnl_pct"]].copy()
    base_trades = base_trades.rename(columns={"pnl_pct": "pnl_2x"})

    policy_rows = []
    for (sym, side, entry, regime), g in raw_df.groupby(["symbol","side","entry_time","regime"]):
        best = best_mult.get(regime, 3.0)
        row_best = g[g["atr_mult"] == best]
        if len(row_best) == 0:
            continue
        policy_rows.append({
            "symbol":     sym,
            "side":       side,
            "entry_time": entry,
            "regime":     regime,
            "pnl_conditional": float(row_best.iloc[0]["pnl_pct"]),
        })
        for mult in [2.0, 3.0, 5.0]:
            row_f = g[g["atr_mult"] == mult]
            if len(row_f) > 0:
                policy_rows[-1][f"pnl_{mult}x"] = float(row_f.iloc[0]["pnl_pct"])

    policy_df = pd.DataFrame(policy_rows).sort_values("entry_time").reset_index(drop=True)

    print(f"\n  === Policy comparison (all splits) ===")
    summary_rows = []
    for label, col in [("conditional", "pnl_conditional"),
                        ("fixed_2x",   "pnl_2.0x"),
                        ("fixed_3x",   "pnl_3.0x"),
                        ("fixed_5x",   "pnl_5.0x")]:
        if col not in policy_df.columns:
            continue
        m = _metrics(policy_df[col].dropna().values)
        m["policy"] = label
        summary_rows.append(m)
        print(f"    {label:<18} n={m['n']:5d}  sum_pnl={m['sum_pnl_pct']:+8.0f}%  "
              f"PF={m['pf']:.2f}  WR={m['wr']:.1%}  MaxDD={m['max_dd_pct']:.1f}%")

    # OOS only (Q1 2026)
    oos_mask = policy_df["entry_time"] >= pd.Timestamp("2026-01-01", tz="UTC")
    if oos_mask.sum() > 10:
        print(f"\n  === OOS only (Q1 2026) ===")
        oos = policy_df[oos_mask]
        for label, col in [("conditional", "pnl_conditional"),
                            ("fixed_2x",   "pnl_2.0x"),
                            ("fixed_3x",   "pnl_3.0x"),
                            ("fixed_5x",   "pnl_5.0x")]:
            if col not in oos.columns:
                continue
            m = _metrics(oos[col].dropna().values)
            print(f"    {label:<18} n={m['n']:5d}  sum_pnl={m['sum_pnl_pct']:+8.0f}%  "
                  f"PF={m['pf']:.2f}  WR={m['wr']:.1%}  MaxDD={m['max_dd_pct']:.1f}%")

    # ── Write verdict ──────────────────────────────────────────────────────────
    verdict_path = RESULTS / "regime_atr_verdict.md"
    with open(verdict_path, "w") as f:
        f.write("# v_new_2 Regime-Conditional ATR Sweep — Verdict\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Trades: {len(trades):,} (bgm >= {BGM_THRESHOLD})\n")
        f.write(f"ATR multipliers tested: {ATR_MULTS}\n\n")

        f.write("## Regime distribution\n")
        for r, cnt in trades["regime"].value_counts().items():
            f.write(f"  {r}: {cnt}\n")
        f.write("\n")

        f.write("## Best multiplier per regime\n")
        f.write("| regime | best_mult | sum_pnl% | PF | WR | n |\n")
        f.write("|--------|-----------|----------|----|----|---|\n")
        for r, m in sorted(best_mult.items()):
            row_m = agg_df[(agg_df["regime"] == r) & (agg_df["atr_mult"] == m)].iloc[0]
            f.write(f"| {r} | {m}× | {row_m['sum_pnl_pct']:+.0f}% | "
                    f"{row_m['pf']:.2f} | {row_m['wr']:.1%} | {int(row_m['n'])} |\n")

        f.write("\n## Full regime × multiplier table\n")
        f.write(agg_df.round(2).to_csv(index=False))
        f.write("\n")

        f.write("## Policy comparison (all splits)\n")
        f.write("| policy | n | sum_pnl% | PF | WR | MaxDD% |\n")
        f.write("|--------|---|----------|----|----|--------|\n")
        for m in summary_rows:
            f.write(f"| {m['policy']} | {m['n']} | {m['sum_pnl_pct']:+.0f}% | "
                    f"{m['pf']:.2f} | {m['wr']:.1%} | {m['max_dd_pct']:.1f}% |\n")

        f.write("\n## Decision\n")
        cond = next((m for m in summary_rows if m["policy"] == "conditional"), None)
        best_fixed = max((m for m in summary_rows if m["policy"] != "conditional"),
                         key=lambda x: x["sum_pnl_pct"], default=None)
        if cond and best_fixed:
            delta = cond["sum_pnl_pct"] - best_fixed["sum_pnl_pct"]
            if delta > 50:
                f.write(f"**PROCEED** — conditional policy beats best fixed by {delta:+.0f}pp sum_pnl. "
                        f"Train regime-exit ML model.\n")
            elif delta > 0:
                f.write(f"**MARGINAL** — conditional beats best fixed by only {delta:+.0f}pp. "
                        f"Consider per-meme/non-meme split before committing to ML.\n")
            else:
                f.write(f"**FIXED WINS** — conditional underperforms best fixed by {abs(delta):.0f}pp. "
                        f"Fixed ATR trail is already near-optimal; skip exit ML.\n")

    print(f"\n{_ts()} Verdict → {verdict_path}")
    print(f"{_ts()} Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
