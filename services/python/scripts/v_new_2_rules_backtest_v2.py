"""v_new_2 — Rules-only backtest v2 (with ATR + mcap tier + realistic compounding).

Improvements over v1 backtest:
  - Pullback can be %-based OR ATR-based
  - Exit can be: trend_flip | fixed_target_pct | atr_trail | combined
  - Per-tier reporting (top25 / 26-50 / 51-100 / 101-200)
  - Caps per-trade loss at -99% (no negative equity arithmetic)
  - Reports SUM_PNL (additive) for full-history; compounded only for OOS slices

Variants:
  pullback type/level:
    %  ∈ {5, 10, 15, 20}
    ATR ∈ {1, 2, 3, 4}  (units of atr14_pct)
  exit method:
    trend_flip          — exit when EMA20/50 cross opposite
    target_20pct        — exit at +20% / -20%
    target_30pct        — exit at +30% / -30%
    target_50pct        — exit at +50% / -50%
    atr_trail_2         — exit at running_max - 2*atr14
    atr_trail_3         — exit at running_max - 3*atr14
  + 90-day timeout safety on all
  mcap tier ∈ {top25, 26-50, 51-100, 101-200}
  side ∈ {long, short}

Output:
  services/python/results/v_new_2/rules_backtest_v2.csv
  services/python/results/v_new_2/rules_v2_verdict.md
"""
from __future__ import annotations

import os
import pathlib
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                        str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))
RESULTS = ROOT / "services" / "python" / "results" / "v_new_2"
RESULTS.mkdir(parents=True, exist_ok=True)

PULLBACK_PCT = [5.0, 10.0, 15.0, 20.0]
PULLBACK_ATR = [1.0, 2.0, 3.0, 4.0]
EXITS = ["trend_flip", "target_20pct", "target_30pct", "target_50pct",
         "atr_trail_2", "atr_trail_3"]
TIERS = ["top25", "26-50", "51-100", "101-200"]
TIMEOUT_BARS = 90 * 6   # 90 days × 6 bars/day
FRICTION = 0.001
MAX_LOSS = -0.99


def _ts() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


def _summary(pnls: np.ndarray, leverage: float):
    if len(pnls) == 0:
        return {"n":0}
    p = pnls * leverage
    p = np.clip(p, MAX_LOSS, 10.0)
    eq = np.cumprod(1.0 + p)
    rm = np.maximum.accumulate(eq)
    safe = np.where(rm > 1e-10, rm, 1e-10)
    dd = (eq - rm) / safe
    wins = p[p > 0]
    losses = p[p < 0]
    return {
        "n": int(len(p)),
        "wr": float((p > 0).mean()),
        "sum_pnl_pct": float(p.sum() * 100.0),
        "compounded_pct": float((eq[-1] - 1.0) * 100.0),
        "max_dd_pct": float(dd.min() * 100.0),
        "avg_win_pct": float(wins.mean() * 100) if len(wins) else 0.0,
        "avg_loss_pct": float(losses.mean() * 100) if len(losses) else 0.0,
        "pf": float(wins.sum() / abs(losses.sum())) if len(losses) else float("inf"),
    }


def _backtest_symbol(
    df_sym: pd.DataFrame, side: str, pb_kind: str, pb_level: float, exit_method: str,
) -> list[dict]:
    """Run one symbol's series for one variant. Returns list of trade dicts."""
    n = len(df_sym)
    if n < 50:
        return []
    closes = df_sym["close"].values
    timestamps = df_sym["timestamp"].values
    atr_pct = df_sym["atr14_pct"].values   # 4h ATR proxy in %
    if side == "long":
        trend = df_sym["d_trend_long"].values
        pb_pct_arr = df_sym["pullback_pct"].values
        pb_atr_arr = df_sym["pullback_atr"].values
    else:
        trend = df_sym["d_trend_short"].values
        pb_pct_arr = df_sym["rise_pct"].values
        pb_atr_arr = df_sym["rise_atr"].values

    # Trigger column based on pullback kind
    if pb_kind == "pct":
        trig_arr = pb_pct_arr
    else:  # atr
        trig_arr = pb_atr_arr

    trades: list[dict] = []
    i = 1
    while i < n - 1:
        # Skip if not in trend OR threshold not met OR previous bar already triggered
        if pd.isna(trend[i]) or trend[i] != 1:
            i += 1; continue
        if pd.isna(trig_arr[i]) or trig_arr[i] < pb_level:
            i += 1; continue
        prev = trig_arr[i-1]
        if not pd.isna(prev) and prev >= pb_level:
            i += 1; continue   # already triggered last bar — wait for new cross

        entry_idx = i
        entry_price = closes[entry_idx]
        if pd.isna(entry_price) or entry_price <= 0:
            i += 1; continue

        # Walk forward to find exit
        running_extreme = entry_price   # max for long, min for short
        exit_idx = None
        exit_reason = None
        max_idx = min(n - 1, entry_idx + TIMEOUT_BARS)
        for k in range(entry_idx + 1, max_idx + 1):
            p = closes[k]
            if pd.isna(p):
                continue
            if side == "long":
                running_extreme = max(running_extreme, p)
            else:
                running_extreme = min(running_extreme, p)

            if exit_method == "trend_flip":
                if not pd.isna(trend[k]) and trend[k] != 1:
                    exit_idx = k; exit_reason = "trend_flip"; break
            elif exit_method.startswith("target_"):
                tgt = float(exit_method.replace("target_", "").replace("pct", "")) / 100.0
                if side == "long" and p >= entry_price * (1 + tgt):
                    exit_idx = k; exit_reason = "target"; break
                if side == "short" and p <= entry_price * (1 - tgt):
                    exit_idx = k; exit_reason = "target"; break
            elif exit_method.startswith("atr_trail_"):
                mult = float(exit_method.replace("atr_trail_", ""))
                a = atr_pct[k] / 100.0 if not pd.isna(atr_pct[k]) else 0.02
                if side == "long":
                    trail = running_extreme * (1 - mult * a)
                    if p <= trail:
                        exit_idx = k; exit_reason = "atr_trail"; break
                else:
                    trail = running_extreme * (1 + mult * a)
                    if p >= trail:
                        exit_idx = k; exit_reason = "atr_trail"; break
        if exit_idx is None:
            exit_idx = max_idx
            exit_reason = "timeout"
        exit_price = closes[exit_idx]
        if pd.isna(exit_price) or exit_price <= 0:
            i = max_idx + 1; continue
        if side == "long":
            pnl = (exit_price / entry_price) - 1.0
        else:
            pnl = (entry_price / exit_price) - 1.0
        pnl -= FRICTION
        trades.append({
            "symbol": df_sym["symbol"].iloc[0],
            "tier": df_sym["mcap_tier"].iloc[0],
            "side": side,
            "entry_time": pd.Timestamp(timestamps[entry_idx]),
            "exit_time": pd.Timestamp(timestamps[exit_idx]),
            "bars_held": int(exit_idx - entry_idx),
            "exit_reason": exit_reason,
            "pnl_pct": float(pnl),
        })
        i = exit_idx + 1
    return trades


def main() -> None:
    t0 = time.time()
    print(f"{_ts()} v_new_2 rules backtest v2 — start")
    df = pd.read_parquet(DATA_DIR / "features_v_new_2_v2.parquet",
                          columns=["symbol","timestamp","close","mcap_tier",
                                   "d_trend_long","d_trend_short",
                                   "pullback_pct","rise_pct",
                                   "pullback_atr","rise_atr","atr14_pct"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    print(f"  loaded {len(df):,} rows")

    sym_groups = {sym: g.reset_index(drop=True) for sym, g in df.groupby("symbol")}
    print(f"  symbols: {len(sym_groups)}")

    rows = []
    variant_count = 0
    pb_variants = [("pct", v) for v in PULLBACK_PCT] + [("atr", v) for v in PULLBACK_ATR]

    for side in ["long", "short"]:
        for pb_kind, pb_level in pb_variants:
            for exit_method in EXITS:
                t_var = time.time()
                # Run on ALL symbols, then aggregate by tier
                all_trades_by_tier = {t: [] for t in TIERS}
                for sym, g in sym_groups.items():
                    if len(g) == 0: continue
                    tier = g["mcap_tier"].iloc[0]
                    if tier not in TIERS:
                        continue
                    trades = _backtest_symbol(g, side, pb_kind, pb_level, exit_method)
                    all_trades_by_tier[tier].extend(trades)

                for tier in TIERS:
                    trs = all_trades_by_tier[tier]
                    pnls = np.array([t["pnl_pct"] for t in trs]) if trs else np.array([])
                    s_full = _summary(pnls, 1.0)
                    s_3x = _summary(pnls, 3.0)
                    s_5x = _summary(pnls, 5.0)
                    # OOS slices — compare in numpy datetime64 to avoid tz issues
                    h2_start = np.datetime64("2025-07-01")
                    h2_end = np.datetime64("2026-01-01")
                    q1_end = np.datetime64("2026-04-01")
                    h2_pnls = np.array([t["pnl_pct"] for t in trs
                                          if h2_start <= np.datetime64(t["entry_time"], "ns") < h2_end])
                    q1_pnls = np.array([t["pnl_pct"] for t in trs
                                          if h2_end <= np.datetime64(t["entry_time"], "ns") < q1_end])
                    s_h2 = _summary(h2_pnls, 1.0)
                    s_q1 = _summary(q1_pnls, 1.0)
                    rows.append({
                        "side": side, "pb_kind": pb_kind, "pb_level": pb_level,
                        "exit_method": exit_method, "tier": tier,
                        "n_total": s_full["n"],
                        "full_wr": s_full.get("wr",0)*100,
                        "full_pf": s_full.get("pf",0),
                        "full_sum_pnl_pct": s_full.get("sum_pnl_pct",0),
                        "full_avg_win_pct": s_full.get("avg_win_pct",0),
                        "full_avg_loss_pct": s_full.get("avg_loss_pct",0),
                        "full_3x_compd_pct": s_3x.get("compounded_pct",0),
                        "full_3x_dd_pct": s_3x.get("max_dd_pct",0),
                        "full_5x_compd_pct": s_5x.get("compounded_pct",0),
                        "full_5x_dd_pct": s_5x.get("max_dd_pct",0),
                        "h2_n": s_h2["n"], "h2_wr": s_h2.get("wr",0)*100, "h2_pf": s_h2.get("pf",0),
                        "h2_sum_pnl_pct": s_h2.get("sum_pnl_pct",0),
                        "q1_n": s_q1["n"], "q1_wr": s_q1.get("wr",0)*100, "q1_pf": s_q1.get("pf",0),
                        "q1_sum_pnl_pct": s_q1.get("sum_pnl_pct",0),
                    })
                variant_count += 1
                if variant_count % 16 == 0:
                    print(f"  {_ts()} {variant_count} variants done  elapsed={time.time()-t0:.1f}s")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "rules_backtest_v2.csv", index=False)
    print(f"\n  saved: {RESULTS / 'rules_backtest_v2.csv'}")

    # Verdict — top variants per tier per side, filtered by sample size
    md = []
    md.append(f"# v_new_2 — Rules-only Backtest v2 ({time.strftime('%Y-%m-%d')})\n\n")
    md.append("ATR + %-based pullback × multiple exit methods × 4 mcap tiers × LONG+SHORT.\n")
    md.append(f"Trade-cap: {MAX_LOSS*100:.0f}% per trade. Friction: {FRICTION*100:.2f}%. 90d timeout.\n\n")

    for side in ["long", "short"]:
        for tier in TIERS:
            sub = df_out[(df_out["side"]==side) & (df_out["tier"]==tier) & (df_out["n_total"]>=50)]
            if len(sub) == 0:
                continue
            sub = sub.sort_values("full_pf", ascending=False).head(10)
            md.append(f"## {side.upper()} — tier {tier} (top 10 by full PF, n≥50)\n\n")
            md.append("| pb_kind | pb_level | exit | n | WR | PF | avg_win | avg_loss | full_sum | H2_n / H2_PF / H2_sum | Q1_n / Q1_PF / Q1_sum |\n")
            md.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
            for _, r in sub.iterrows():
                md.append(f"| {r['pb_kind']} | {r['pb_level']:.1f} | {r['exit_method']} | "
                          f"{int(r['n_total'])} | {r['full_wr']:.1f}% | {r['full_pf']:.2f} | "
                          f"{r['full_avg_win_pct']:+.1f}% | {r['full_avg_loss_pct']:+.1f}% | "
                          f"{r['full_sum_pnl_pct']:+.0f}% | "
                          f"{int(r['h2_n'])}/{r['h2_pf']:.2f}/{r['h2_sum_pnl_pct']:+.0f}% | "
                          f"{int(r['q1_n'])}/{r['q1_pf']:.2f}/{r['q1_sum_pnl_pct']:+.0f}% |\n")
            md.append("\n")

    md.append("## How to read\n")
    md.append("- `full_sum` = sum of all trade %-returns (additive, not compounded). Honest measure of total signal over 6 years.\n")
    md.append("- Annualised CAGR ≈ full_sum / 6 / N_concurrent_positions, very roughly.\n")
    md.append("- `full_3x_compd` is naive sequential compounding — use as relative ranking, not absolute.\n")
    md.append("- We're looking for: PF ≥ 1.5, both H2 and Q1 PF ≥ 1.2, and consistent across tiers.\n")
    (RESULTS / "rules_v2_verdict.md").write_text("".join(md))
    print(f"  saved: {RESULTS / 'rules_v2_verdict.md'}")
    print(f"\n{_ts()} DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
