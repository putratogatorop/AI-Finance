"""Phase 17.F — Verify ship-gate metrics for the S1×E2 alternative on the
D1e+v4 trade ledger.

The Phase 17.D 4×4 grid was computed on the D1+v3 ledger. The realistic-
equity comparison in phase17_sizing_compare.py showed (b) S1×E2 producing
the highest 36-month equity (10,424x) on the D1e+v4 ledger. Before adopting
S1×E2 as the priority paper-trade config, CLAUDE.md requires a backtest of
the ship-gate metrics on that exact ledger.

Computes (TRAIN/OOT PF, OOT n, WF p5/p50/p95/P(loss), MC bootstrap p5) for
S1×E2 on D1e+v4 trades, and compares to the locked S4×E2 config.

Run from services/python/:
    uv run python scripts/phase17_f_verify_s1e2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.bigmover_combined.features import FEATURE_NAMES
from src.ml.indicators import atr as _atr, ema as _ema, kdj as _kdj, macd as _macd, rsi as _rsi

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICES_PY = Path(__file__).resolve().parents[1]
SNAPSHOT_DATE = "2026-04-01"
TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM_SEED = 42
MC_N_RESAMPLES = 5000
WF_FOLD_MONTHS = 3

CANDLES_PATH = REPO_ROOT / "data" / "snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
TRADES_PATH = SERVICES_PY / "data" / "d1e_short_trades_with_features.csv"
MODEL_PATH = SERVICES_PY / "models" / "d1e_short_v4.joblib"
META_PATH = SERVICES_PY / "models" / "d1e_short_v4_meta.json"
OUT_PATH = SERVICES_PY / "results" / "phase17_f_verify_s1e2.json"

ATR_PERIOD = 14
ATR_SL_MULT, ATR_TP_MULT = 2.0, 6.0
TIMEOUT_BARS = 672
FRICTION_PCT = 0.0015
RAPID_RALLY_PCT = 0.03
RAPID_RALLY_LOOKBACK_15M = 96


def _pf(p):
    p = np.asarray(p, dtype=float)
    if len(p) == 0:
        return float("nan")
    w = p[p > 0].sum()
    l = -p[p < 0].sum()
    return float("inf") if l == 0 else float(w / l)


def _cell(p):
    p = np.asarray(p, dtype=float)
    if len(p) == 0:
        return {"trades": 0, "wr": 0.0, "pf": float("nan"), "avg": 0.0, "total": 0.0}
    return {"trades": int(len(p)), "wr": float((p > 0).mean()), "pf": _pf(p),
            "avg": float(p.mean()), "total": float(p.sum())}


def _walkforward(df, pnl_col):
    if len(df) == 0:
        return {"n_folds_used": 0}
    et = pd.to_datetime(df["entry_time"], utc=True)
    s = pd.Timestamp("2023-04-01", tz="UTC")
    end_global = pd.Timestamp(SNAPSHOT_DATE, tz="UTC")
    fold_w = relativedelta(months=WF_FOLD_MONTHS)
    month = relativedelta(months=1)
    pfs = []
    s_ = s
    while s_ + fold_w <= end_global:
        sub = df[(et >= s_) & (et < s_ + fold_w)]
        if len(sub) >= 30:
            p = _pf(sub[pnl_col].to_numpy())
            if np.isfinite(p):
                pfs.append(p)
        s_ += month
    pfs = np.array(pfs)
    if len(pfs) == 0:
        return {"n_folds_used": 0}
    return {"n_folds_used": int(len(pfs)),
            "p5": float(np.percentile(pfs, 5)), "p50": float(np.percentile(pfs, 50)),
            "p95": float(np.percentile(pfs, 95)),
            "p_lt_1": float((pfs < 1.0).mean()),
            "p_gte_130": float((pfs >= 1.30).mean())}


def _monte_carlo(p, n=MC_N_RESAMPLES):
    p = np.asarray(p, dtype=float)
    if len(p) == 0:
        return {"n": 0}
    rng = np.random.default_rng(RANDOM_SEED)
    pfs = np.array([_pf(rng.choice(p, len(p), replace=True)) for _ in range(n)])
    finite = pfs[np.isfinite(pfs)]
    return {"n": int(len(p)), "point_pf": _pf(p),
            "p5": float(np.percentile(finite, 5)) if len(finite) else float("nan"),
            "p50": float(np.percentile(finite, 50)) if len(finite) else float("nan"),
            "p95": float(np.percentile(finite, 95)) if len(finite) else float("nan")}


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


def _sim_E2(close, high, low, open_, atr14, enter_bar, *, rally_flag):
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
            return {"pnl_pct": (entry_price - sl_price) / entry_price - FRICTION_PCT,
                    "exit": "sl", "bars": int(bar - enter_bar)}
        if low[bar] <= tp_price:
            return {"pnl_pct": (entry_price - tp_price) / entry_price - FRICTION_PCT,
                    "exit": "tp", "bars": int(bar - enter_bar)}
        if rally_flag is not None and bool(rally_flag[bar]):
            ep = float(close[bar])
            return {"pnl_pct": (entry_price - ep) / entry_price - FRICTION_PCT,
                    "exit": "rapid", "bars": int(bar - enter_bar)}
    eb = last_bar - 1
    return {"pnl_pct": (entry_price - float(close[eb])) / entry_price - FRICTION_PCT,
            "exit": "timeout", "bars": int(eb - enter_bar)}


def _summarize(df, label, pnl_col):
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["period"] = np.where(df["entry_time"] < TRAIN_OOT_BOUNDARY, "train", "oot")
    train_p = df[df["period"] == "train"][pnl_col].to_numpy()
    oot_p = df[df["period"] == "oot"][pnl_col].to_numpy()
    train_m = _cell(train_p)
    oot_m = _cell(oot_p)
    wf = _walkforward(df, pnl_col)
    mc = _monte_carlo(oot_p)
    print(f"  {label:<24} TRAIN n={train_m['trades']:>5} PF={train_m['pf']:.3f}  "
          f"OOT n={oot_m['trades']:>5} PF={oot_m['pf']:.3f}  "
          f"WF p5={wf.get('p5', float('nan')):.3f}  P(loss)={wf.get('p_lt_1', float('nan'))*100:.1f}%  "
          f"MC p5={mc.get('p5', float('nan')):.3f}", flush=True)
    return {"train": train_m, "oot": oot_m, "wf": wf, "mc": mc}


def main():
    df = pd.read_csv(TRADES_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    threshold = float(meta["threshold_train_q50"])
    feat_cols = FEATURE_NAMES + ["is_short"]
    X = df[feat_cols].to_numpy(dtype=float)
    df["score"] = model.predict_proba(X)[:, 1]
    gated = df[df["score"] >= threshold].reset_index(drop=True)
    print(f"D1e + v4 kept {len(gated):,} of {len(df):,}")

    print("re-simulating E2 exits...", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    btc_15m = (candles[candles["asset"] == "BTCUSDT"]
               .sort_values("timestamp").set_index("timestamp")[["open", "high", "low", "close"]]
               .astype(float))
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
        rally = rapid_naive.reindex(pd.DatetimeIndex(ts_naive), method="ffill").fillna(False).to_numpy().astype(bool)
        for _, r in gtrades.iterrows():
            i = int(r["enter_bar"])
            t = _sim_E2(close, high, low, open_, atr14, i, rally_flag=rally)
            if t is None:
                continue
            rec = r.to_dict()
            rec["e2_pnl_pct"] = t["pnl_pct"]
            rec["e2_bars"] = int(t["bars"])
            rec["e2_exit"] = t["exit"]
            rows.append(rec)
    sim = pd.DataFrame(rows)
    sim["entry_time"] = pd.to_datetime(sim["entry_time"], utc=True)
    print(f"  E2 ok: {len(sim):,}/{len(gated):,}")

    btc_score = sim["btc_score"].to_numpy(dtype=float)
    score = sim["score"].to_numpy(dtype=float)
    s1 = np.minimum(1.5 * np.abs(btc_score), 1.5)
    s4 = np.clip(3.0 * (score - 0.5), 0.0, 1.5)
    sim["s1_e2_sized"] = sim["e2_pnl_pct"].to_numpy() * s1
    sim["s4_e2_sized"] = sim["e2_pnl_pct"].to_numpy() * s4

    print("\n=== Ship-gate verification — D1e + v4, S1×E2 vs S4×E2 ===")
    s1e2 = _summarize(sim, "S1 × E2", "s1_e2_sized")
    s4e2 = _summarize(sim, "S4 × E2 (current lock)", "s4_e2_sized")

    def _gates(m, label):
        g = {
            "oot_pf_ge_130": m["oot"]["pf"] >= 1.30 if np.isfinite(m["oot"]["pf"]) else False,
            "oot_n_ge_200": m["oot"]["trades"] >= 200,
            "wf_p5_gt_1": m["wf"].get("p5", -1) > 1.0,
            "mc_p5_gt_1": np.isfinite(m["mc"].get("p5", float("nan"))) and m["mc"].get("p5", -1) > 1.0,
        }
        g["overall"] = all(g.values())
        print(f"\n  Ship gates for {label}:")
        for k, v in g.items():
            print(f"    {k}: {'PASS' if v else 'FAIL'}")
        return g

    g_s1e2 = _gates(s1e2, "S1 × E2")
    g_s4e2 = _gates(s4e2, "S4 × E2")

    print(f"\nS1 × E2 verdict: {'SHIP-GATE PASSES (all 4 floors clear)' if g_s1e2['overall'] else 'SHIP-GATE FAILS — DO NOT ADOPT'}")
    print(f"S4 × E2 verdict: {'SHIP-GATE PASSES (all 4 floors clear)' if g_s4e2['overall'] else 'SHIP-GATE FAILS'}")

    out = {
        "phase": "17.F — verify S1×E2 vs S4×E2 on D1e+v4",
        "snapshot": SNAPSHOT_DATE,
        "n_gated_trades": int(len(gated)),
        "n_e2_simulated": int(len(sim)),
        "S1_E2": {"metrics": s1e2, "ship_gates": g_s1e2},
        "S4_E2": {"metrics": s4e2, "ship_gates": g_s4e2},
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
