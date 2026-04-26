"""Realism audit of the S1×E2 "10,424× over 36 months" headline equity number.

The companion script `phase17_sizing_compare.py` reproducibly computes the
10,424× figure via:

    r = pnl_pct × pos_scale × NOTIONAL_PCT     (NOTIONAL_PCT = 0.25)
    equity *= 1 + r

Internal arithmetic is correct. This audit script asks the next question:
how does that number hold up when the methodology is tightened?

Realism stack applied here (each layer in addition to the previous):

    L0  Reported            full 36-mo, pos_scale_cap=1.5, no notional cap, friction=0.0015
    L1  + OOT-only          restrict to entry_time >= TRAIN_OOT_BOUNDARY (3 months)
    L2  + pos cap @ 1.0     CLAUDE.md gross-notional rule (5 × 1.0 × 25% = 125%)
    L3  + $50k notional cap once equity_usd × notional_pct × pos_scale > $50k, fill the cap
    L4  + 0.5% slippage     scaled friction in breadth-gated entries (0.005 instead of 0.0015)

L0 should reproduce the 10,424× from phase17_sizing_compare.py exactly.
L4 is the realistic forward projection.

Run from services/python/:
    uv run python scripts/phase17_g_sizing_audit.py

Cache: stores prepared sim DataFrame in
       services/python/data/_phase17_audit_sim.parquet
Re-running is fast after the first invocation. Pass --rebuild to force
re-simulation of exits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.bigmover_combined.features import FEATURE_NAMES
from src.ml.indicators import atr as _atr, ema as _ema, kdj as _kdj, macd as _macd, rsi as _rsi

# ── Identical constants to phase17_sizing_compare.py ──────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICES_PY = Path(__file__).resolve().parents[1]
SNAPSHOT_DATE = "2026-04-01"
TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
CANDLES_PATH = REPO_ROOT / "data" / "snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
TRADES_PATH = SERVICES_PY / "data" / "d1e_short_trades_with_features.csv"
MODEL_PATH = SERVICES_PY / "models" / "d1e_short_v4.joblib"
META_PATH = SERVICES_PY / "models" / "d1e_short_v4_meta.json"
SIM_CACHE_PATH = SERVICES_PY / "data" / "_phase17_audit_sim.parquet"
RESULTS_PATH = SERVICES_PY / "results" / "phase17_g_sizing_audit.json"

NOTIONAL_PCT = 0.25
MAX_CONCURRENT = 5
ATR_PERIOD = 14
ATR_SL_MULT, ATR_TP_MULT = 2.0, 6.0
TIMEOUT_BARS = 672
FRICTION_PCT_REPORTED = 0.0015
RAPID_RALLY_PCT = 0.03
RAPID_RALLY_LOOKBACK_15M = 96


def _build_rapid_exit_panel(candles_15m):
    close = candles_15m["close"].astype(float)
    btc_daily = candles_15m.resample("1D", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    ema_fast = _ema(btc_daily["close"], 9).shift(1)
    ema_mid = _ema(btc_daily["close"], 21).shift(1)
    ema_slow = _ema(btc_daily["close"], 50).shift(1)
    ema_cond = (btc_daily["close"].shift(1) < ema_fast) & (ema_fast < ema_mid) & (ema_mid < ema_slow)
    rsi_cond = _rsi(btc_daily["close"], 14).shift(1) < 50.0
    md = _macd(btc_daily["close"], 12, 26, 9)
    macd_cond = md["macd"].shift(1) < md["signal"].shift(1)
    kdj_v = _kdj(btc_daily["high"], btc_daily["low"], btc_daily["close"], n=9, k_smooth=3, d_smooth=3)
    kdj_cond = kdj_v["k"].shift(1) < kdj_v["d"].shift(1)
    daily_bear = (ema_cond & rsi_cond & macd_cond & kdj_cond).fillna(False)
    daily_bear_15m = daily_bear.reindex(close.index, method="ffill").fillna(False)
    btc_24h_ret = close.pct_change(RAPID_RALLY_LOOKBACK_15M)
    return ((~daily_bear_15m) & (btc_24h_ret > RAPID_RALLY_PCT)).fillna(False).astype(bool)


def _sim_E2_zero_friction(close, high, low, open_, atr14, enter_bar, *, rally_flag):
    """Same E2 exit as phase17_sizing_compare._sim_E2 but stores GROSS pnl
    (no friction subtracted) so the audit can sweep friction post-hoc."""
    n = len(close)
    if enter_bar >= n - 1:
        return None
    entry_price = float(open_[enter_bar])
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    a = float(atr14[enter_bar]) if enter_bar < len(atr14) else float("nan")
    if not np.isfinite(a) or a <= 0:
        return None
    sl_price = entry_price + ATR_SL_MULT * a
    tp_price = entry_price - ATR_TP_MULT * a
    if tp_price <= 0:
        return None
    last_bar = min(enter_bar + 1 + TIMEOUT_BARS, n)
    for bar in range(enter_bar + 1, last_bar):
        if high[bar] >= sl_price:
            pnl = (entry_price - sl_price) / entry_price
            return {"pnl_gross": pnl, "bars": int(bar - enter_bar)}
        if low[bar] <= tp_price:
            pnl = (entry_price - tp_price) / entry_price
            return {"pnl_gross": pnl, "bars": int(bar - enter_bar)}
        if rally_flag is not None and bool(rally_flag[bar]):
            ep = float(close[bar])
            return {"pnl_gross": (entry_price - ep) / entry_price, "bars": int(bar - enter_bar)}
    eb = last_bar - 1
    return {"pnl_gross": (entry_price - float(close[eb])) / entry_price, "bars": int(eb - enter_bar)}


def prepare_sim(rebuild: bool = False) -> pd.DataFrame:
    """Build the post-classifier-gate, post-E2-resimulation sim DataFrame.
    Cached in parquet because exit re-simulation is the slow step (~1 min)."""
    if SIM_CACHE_PATH.exists() and not rebuild:
        sim = pd.read_parquet(SIM_CACHE_PATH)
        sim["entry_time"] = pd.to_datetime(sim["entry_time"], utc=True)
        print(f"loaded cached sim: {len(sim):,} trades from {SIM_CACHE_PATH}")
        return sim

    df = pd.read_csv(TRADES_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    threshold = float(meta["threshold_train_q50"])
    feat_cols = FEATURE_NAMES + ["is_short"]
    X = df[feat_cols].to_numpy(dtype=float)
    df["score"] = model.predict_proba(X)[:, 1]
    gated = df[df["score"] >= threshold].reset_index(drop=True)
    print(f"D1e + v4 classifier kept {len(gated):,} of {len(df):,}")

    print("re-simulating E2 exits (gross, no friction)...", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    btc = (candles[candles["asset"] == "BTCUSDT"].sort_values("timestamp").set_index("timestamp"))
    btc_15m = btc[["open", "high", "low", "close"]].astype(float)
    rapid_exit_15m = _build_rapid_exit_panel(btc_15m)
    rapid_naive = rapid_exit_15m.copy()
    rapid_naive.index = pd.to_datetime(rapid_naive.index, utc=True).tz_convert(None)

    rows = []
    for asset, gtrades in gated.groupby("asset"):
        sub = (candles[candles["asset"] == asset]
               .sort_values("timestamp").reset_index(drop=True))
        if len(sub) < 200:
            continue
        ts_naive = pd.to_datetime(sub["timestamp"], utc=True).dt.tz_convert(None).to_numpy()
        close = sub["close"].to_numpy(dtype=float)
        high = sub["high"].to_numpy(dtype=float)
        low = sub["low"].to_numpy(dtype=float)
        open_ = sub["open"].to_numpy(dtype=float)
        atr14 = _atr(pd.Series(high), pd.Series(low), pd.Series(close), ATR_PERIOD).to_numpy()
        rally = (rapid_naive.reindex(pd.DatetimeIndex(ts_naive), method="ffill")
                 .fillna(False).to_numpy().astype(bool))
        for _, r in gtrades.iterrows():
            i = int(r["enter_bar"])
            t2 = _sim_E2_zero_friction(close, high, low, open_, atr14, i, rally_flag=rally)
            if t2 is None:
                continue
            rows.append({
                "entry_time": r["entry_time"],
                "asset": r["asset"],
                "btc_score": float(r["btc_score"]),
                "score": float(r["score"]),
                "e2_pnl_gross": float(t2["pnl_gross"]),
                "e2_bars": int(t2["bars"]),
            })
    sim = pd.DataFrame(rows)
    sim.to_parquet(SIM_CACHE_PATH, index=False)
    print(f"  exits computed for {len(sim):,} trades — cached at {SIM_CACHE_PATH}")
    return sim


def run_equity(sim: pd.DataFrame, *,
               pos_scale_cap: float,
               notional_cap_usd: float | None,
               friction_pct: float,
               oot_only: bool,
               starting_equity_usd: float = 100.0,
               notional_pct: float = NOTIONAL_PCT,
               max_concurrent: int = MAX_CONCURRENT) -> dict:
    """Run an equity simulation under one realism configuration.

    pos_scale = min(pos_scale_cap × |btc_score|, pos_scale_cap)
    per-trade notional_usd = equity_usd × notional_pct × pos_scale
    if notional_cap_usd is set: notional_usd = min(notional_usd, notional_cap_usd)
    return on equity = pnl_after_friction × (notional_usd / equity_usd)
    """
    df = sim.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    if oot_only:
        df = df[df["entry_time"] >= TRAIN_OOT_BOUNDARY].copy()
    df = df.sort_values("entry_time").reset_index(drop=True)
    df["exit_time"] = df["entry_time"] + pd.to_timedelta(df["e2_bars"] * 15, unit="m")

    pos_scale_arr = np.minimum(pos_scale_cap * np.abs(df["btc_score"].to_numpy()), pos_scale_cap)
    pnl_net = df["e2_pnl_gross"].to_numpy() - friction_pct

    equity_usd = float(starting_equity_usd)
    open_exits: list = []
    rows = []
    accepted_count = 0
    for i, t in df.iterrows():
        open_exits = [x for x in open_exits if x > t["entry_time"]]
        accepted = len(open_exits) < max_concurrent
        if accepted:
            notional_usd = equity_usd * notional_pct * float(pos_scale_arr[i])
            if notional_cap_usd is not None and notional_usd > notional_cap_usd:
                notional_usd = float(notional_cap_usd)
            r = float(pnl_net[i]) * (notional_usd / equity_usd) if equity_usd > 0 else 0.0
            equity_usd *= (1.0 + r)
            open_exits.append(t["exit_time"])
            accepted_count += 1
            rows.append({"entry_time": t["entry_time"], "equity_usd": equity_usd,
                         "pnl_net": float(pnl_net[i]), "pos_scale": float(pos_scale_arr[i]),
                         "notional_usd": notional_usd, "accepted": True})
        else:
            rows.append({"entry_time": t["entry_time"], "equity_usd": equity_usd,
                         "pnl_net": float(pnl_net[i]), "pos_scale": float(pos_scale_arr[i]),
                         "notional_usd": float("nan"), "accepted": False})
    out = pd.DataFrame(rows)
    final_multiple = equity_usd / starting_equity_usd
    accepted = out[out["accepted"]].copy()
    accepted["month"] = accepted["entry_time"].dt.tz_convert("UTC").dt.strftime("%Y-%m")
    monthly = accepted.groupby("month").agg(end_equity=("equity_usd", "last"))
    end_eq = monthly["end_equity"].values
    start_eq = np.concatenate([[float(starting_equity_usd)], end_eq[:-1]])
    monthly["ret"] = (end_eq / start_eq) - 1.0
    rets = monthly["ret"]
    return {
        "final_multiple": final_multiple,
        "accepted_trades": int(accepted_count),
        "total_signals": int(len(df)),
        "best_month": float(rets.max()) if len(rets) else float("nan"),
        "worst_month": float(rets.min()) if len(rets) else float("nan"),
        "median_month": float(rets.median()) if len(rets) else float("nan"),
        "p_loss_monthly": float((rets < 0).mean()) if len(rets) else float("nan"),
        "n_months": int(len(rets)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="Force re-simulation of exits even if cache exists.")
    ap.add_argument("--starting-equity-usd", type=float, default=100.0,
                    help="USD starting equity for the simulation (matches paper executor).")
    args = ap.parse_args()

    sim = prepare_sim(rebuild=args.rebuild)

    # ── Realism stack ─────────────────────────────────────────────────────
    SCENARIOS = [
        # (label, kwargs)
        ("L0  Reported (matches phase17_sizing_compare.py)", dict(
            pos_scale_cap=1.5, notional_cap_usd=None,
            friction_pct=FRICTION_PCT_REPORTED, oot_only=False)),
        ("L1  + OOT-only (entry >= 2026-01-01)", dict(
            pos_scale_cap=1.5, notional_cap_usd=None,
            friction_pct=FRICTION_PCT_REPORTED, oot_only=True)),
        ("L2  + pos_scale cap 1.0 (CLAUDE.md gross-notional)", dict(
            pos_scale_cap=1.0, notional_cap_usd=None,
            friction_pct=FRICTION_PCT_REPORTED, oot_only=True)),
        ("L3  + $50k notional cap (Gate.io tier-2 liquidity)", dict(
            pos_scale_cap=1.0, notional_cap_usd=50_000.0,
            friction_pct=FRICTION_PCT_REPORTED, oot_only=True)),
        ("L4  + 0.5% slippage (breadth-gated worst-liquidity)", dict(
            pos_scale_cap=1.0, notional_cap_usd=50_000.0,
            friction_pct=0.005, oot_only=True)),
        # Reference: full 36-month run with the same realism stack
        # (still in-sample for the classifier; reported only to show the
        # capacity-cap × scaled-slippage ceiling under L4 constraints).
        ("REF full-36mo with L4 realism (in-sample, ceiling check)", dict(
            pos_scale_cap=1.0, notional_cap_usd=50_000.0,
            friction_pct=0.005, oot_only=False)),
    ]

    print(f"\n{'='*92}")
    print(f"PHASE 17.G — S1×E2 EQUITY REALISM AUDIT")
    print(f"{'='*92}")
    print(f"  starting equity USD : ${args.starting_equity_usd:,.0f}")
    print(f"  trade ledger        : {len(sim):,} v4-gated trades over "
          f"{sim['entry_time'].min().date()} → {sim['entry_time'].max().date()}")
    print(f"  TRAIN_OOT_BOUNDARY  : {TRAIN_OOT_BOUNDARY.date()}")
    print()
    print(f"  {'Layer':<55} {'Multiple':>12} {'Worst mo':>10} {'P(loss)':>9} {'N trades':>10}")
    print(f"  {'-'*55} {'-'*12} {'-'*10} {'-'*9} {'-'*10}")

    results = []
    for label, kw in SCENARIOS:
        r = run_equity(sim, starting_equity_usd=args.starting_equity_usd, **kw)
        results.append({"label": label, **kw, **r})
        worst = r["worst_month"] * 100 if np.isfinite(r["worst_month"]) else float("nan")
        ploss = r["p_loss_monthly"] * 100 if np.isfinite(r["p_loss_monthly"]) else float("nan")
        print(f"  {label:<55} {r['final_multiple']:>11.2f}x {worst:>+9.2f}% "
              f"{ploss:>7.1f}% {r['accepted_trades']:>10,}")

    print()
    by_label = {r["label"]: r for r in results}
    L0 = results[0]
    L4 = next(r for r in results if r["label"].startswith("L4"))
    REF = next((r for r in results if r["label"].startswith("REF")), None)

    print(f"  Reported headline (L0, in-sample 36mo)        : {L0['final_multiple']:>11.2f}x  "
          f"({L0['n_months']} months)")
    print(f"  OOT realistic (L4, 3-month forward-looking)   : {L4['final_multiple']:>11.2f}x  "
          f"({L4['n_months']} months)")
    if REF is not None:
        print(f"  In-sample 36mo under same L4 realism (REF)    : {REF['final_multiple']:>11.2f}x  "
              f"({REF['n_months']} months)")
        ratio_in = L0["final_multiple"] / REF["final_multiple"] if REF["final_multiple"] > 0 else float("nan")
        print(f"  Realism alone shaves the headline by          : {ratio_in:>11.1f}x  "
              f"(in-sample, no OOT restriction)")
        ratio_oot = L0["final_multiple"] / (L4["final_multiple"] ** 12) if L4["final_multiple"] > 0 else float("nan")
        print(f"  Headline vs naive OOT-extrapolation (L4^12)   : {ratio_oot:>11.1f}x")

    if L4["n_months"] > 0 and L4["final_multiple"] > 0:
        proj_36 = L4["final_multiple"] ** (36 / L4["n_months"])
        print(f"\n  Naive 36-mo projection from OOT (L4 ^ 12)     : {proj_36:>11.2f}x")
        print(f"  Note: assumes 2026-Q1 conditions repeat 12x in a row.")
        print(f"        WF p5 PF = 1.415 (3-mo profit factor, NOT equity mult);")
        print(f"        the WF p5 fold's equity mult will be lower than its PF.")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "snapshot_date": SNAPSHOT_DATE,
        "starting_equity_usd": args.starting_equity_usd,
        "train_oot_boundary": TRAIN_OOT_BOUNDARY.isoformat(),
        "scenarios": [{k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                       for k, v in r.items()} for r in results],
    }, indent=2, default=str))
    print(f"\n  results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
