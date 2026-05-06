"""Build LSTM sequence dataset for v_new_2 regime classifier.

For each of the 21k backtest trades, extract the last SEQ_LEN×4h bars of
per-symbol context + market-wide features. Labels derive from trade pnl at
the widest trail (atr_mult=10) — proxy for "how productive was this regime."

Label scheme (quartile-based, ~25/50/25 balance):
  0 = bear      pnl < Q25  (~-2.3%)
  1 = neutral   Q25 <= pnl <= Q75  (~15.4%)
  2 = bull_mania  pnl > Q75

Features per bar (7):
  close_ret      — symbol 4h return
  atr14_pct      — symbol ATR proxy (%)
  d_ema_spread_pct — EMA20/50 spread (trend momentum)
  trend_strength   — EMA trend confidence
  cfgi_norm        — Fear&Greed index normalized 0–1 (daily, broadcast)
  btc_ret_4h       — BTC 4h return (market-wide proxy)
  side_sign        — +1 long / -1 short (constant across sequence, tells model direction)

Output (written to DATA_DIR/v_new_2_lstm/):
  X.npy        (N, SEQ_LEN, N_FEATURES)  float32
  y.npy        (N,)                       int8  0/1/2
  meta.parquet (N rows)  symbol, side, entry_time, pnl_pct, label

Run from services/python/:
  python3 scripts/v_new_2_lstm_build_dataset.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                        str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))
SWEEP_CSV = ROOT / "services" / "python" / "results" / "v_new_2" / "regime_atr_sweep_raw.csv"
CFGI_PARQUET = ROOT / "services" / "python" / "data" / "cfgi_historical.parquet"
OUT_DIR = ROOT / "services" / "python" / "data" / "v_new_2_lstm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 30       # 30 × 4h = 5 days of context before entry
N_FEATURES = 6     # close_ret, atr14_pct, d_ema_spread_pct, trend_strength, btc_ret_4h, side_sign
MIN_BARS = 20      # skip trade if < MIN_BARS of history available

# Label = CFGI quartile at entry time (market-wide sentiment from sweep data)
# Q25=26, Q75=72 → ~25/50/25 class balance
# cfgi intentionally EXCLUDED from input features so LSTM must learn to infer
# sentiment from price sequences alone (non-circular, genuinely predictive)
CFGI_BULL_THR = 72.0   # CFGI > 72 → bull_mania (2)
CFGI_BEAR_THR = 26.0   # CFGI < 26 → bear (0)


def _ts() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


def main() -> None:
    print(f"{_ts()} Building LSTM regime dataset")

    # ── Load features ────────────────────────────────────────────────────────
    print(f"{_ts()} Loading features parquet…")
    feats = pd.read_parquet(DATA_DIR / "features_v_new_2_v2.parquet",
                            columns=["symbol", "timestamp", "close",
                                     "atr14_pct", "d_ema_spread_pct", "trend_strength"])
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
    feats = feats.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # close_ret per symbol
    feats["close_ret"] = feats.groupby("symbol")["close"].pct_change().fillna(0.0)
    print(f"  {len(feats):,} rows, {feats['symbol'].nunique()} symbols")

    # ── BTC market proxy ─────────────────────────────────────────────────────
    btc = feats[feats["symbol"] == "BTCUSDT"][["timestamp", "close_ret"]].copy()
    btc = btc.rename(columns={"close_ret": "btc_ret_4h"}).set_index("timestamp")
    if len(btc) == 0:
        # fallback: BTCUSDT might be named differently
        btc_cands = [s for s in feats["symbol"].unique() if "BTC" in s and "USD" in s]
        print(f"  BTCUSDT not found, trying: {btc_cands[:3]}")
        if btc_cands:
            btc = feats[feats["symbol"] == btc_cands[0]][["timestamp", "close_ret"]].copy()
            btc = btc.rename(columns={"close_ret": "btc_ret_4h"}).set_index("timestamp")
    print(f"  BTC proxy rows: {len(btc)}")

    # CFGI is used only for labels (loaded with trade records from sweep_raw)
    # NOT used as input feature — LSTM must infer regime from price sequences

    # ── Trade records + CFGI labels ───────────────────────────────────────────
    print(f"{_ts()} Loading trade records…")
    sweep = pd.read_csv(SWEEP_CSV)
    sweep["entry_time"] = pd.to_datetime(sweep["entry_time"], utc=True)
    trades = sweep[sweep["atr_mult"] == 10.0].copy()
    trades = trades.drop_duplicates(subset=["symbol", "side", "entry_time"]).reset_index(drop=True)
    print(f"  {len(trades):,} unique trades")

    # Label by CFGI at entry (already in sweep_raw)
    def _cfgi_label(cfgi_val: float) -> int:
        if cfgi_val > CFGI_BULL_THR:
            return 2   # bull_mania
        if cfgi_val < CFGI_BEAR_THR:
            return 0   # bear
        return 1       # neutral

    trades["label"] = trades["cfgi"].apply(_cfgi_label)
    print(f"  Label dist (CFGI regime): {trades['label'].value_counts().sort_index().to_dict()}")
    q25, q75 = float(CFGI_BEAR_THR), float(CFGI_BULL_THR)

    # ── Build per-symbol lookup for fast slicing ──────────────────────────────
    print(f"{_ts()} Indexing symbol groups…")
    sym_groups: dict[str, pd.DataFrame] = {}
    for sym, g in feats.groupby("symbol"):
        sym_groups[sym] = g.reset_index(drop=True)

    # ── Build sequences ───────────────────────────────────────────────────────
    print(f"{_ts()} Building sequences (n={len(trades):,})…")
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta_rows: list[dict] = []
    skipped = 0

    for idx, row in trades.iterrows():
        sym: str = row["symbol"]
        entry_time: pd.Timestamp = row["entry_time"]
        side: str = row["side"]
        side_sign: float = 1.0 if side == "long" else -1.0

        g = sym_groups.get(sym)
        if g is None:
            skipped += 1
            continue

        # Find position of last bar BEFORE entry (strict <)
        ts_arr = g["timestamp"].values
        pos = int(np.searchsorted(ts_arr, np.datetime64(entry_time), side="left"))
        # pos is the index where entry_time would be inserted;
        # bars at pos-SEQ_LEN .. pos-1 are the context window
        if pos < MIN_BARS:
            skipped += 1
            continue

        start = max(0, pos - SEQ_LEN)
        window = g.iloc[start:pos]
        if len(window) < MIN_BARS:
            skipped += 1
            continue

        # Pad to SEQ_LEN with zeros at the front if window is shorter
        pad = SEQ_LEN - len(window)

        # Build feature matrix — cfgi intentionally excluded (LSTM infers from price)
        cr     = window["close_ret"].values.astype(np.float32)
        atr    = window["atr14_pct"].values.astype(np.float32)
        ema    = window["d_ema_spread_pct"].values.astype(np.float32)
        ts_str = window["trend_strength"].values.astype(np.float32)

        # BTC 4h return aligned to window timestamps (market-wide proxy)
        if len(btc) > 0:
            btc_vals = btc["btc_ret_4h"].reindex(window["timestamp"], method="ffill").fillna(0.0).values.astype(np.float32)
        else:
            btc_vals = np.zeros(len(window), dtype=np.float32)

        side_arr = np.full(len(window), side_sign, dtype=np.float32)

        seq = np.stack([cr, atr, ema, ts_str, btc_vals, side_arr], axis=1)  # (L, 6)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        if pad > 0:
            seq = np.pad(seq, ((pad, 0), (0, 0)), mode="constant", constant_values=0.0)

        X_list.append(seq)
        y_list.append(row["label"])
        meta_rows.append({
            "symbol": sym,
            "side": side,
            "entry_time": entry_time,
            "pnl_pct": row["pnl_pct"],
            "label": row["label"],
            "regime": row.get("regime", ""),
        })

        if (idx + 1) % 5000 == 0:
            print(f"  {_ts()} processed {idx+1:,} / {len(trades):,}  skipped={skipped}")

    print(f"\n{_ts()} Skipped: {skipped} / {len(trades)}  Built: {len(X_list)}")

    X = np.array(X_list, dtype=np.float32)   # (N, SEQ_LEN, N_FEATURES)
    y = np.array(y_list, dtype=np.int8)
    meta = pd.DataFrame(meta_rows)

    print(f"  X shape: {X.shape}   y shape: {y.shape}")
    print(f"  Label dist: {np.bincount(y.astype(int))}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(OUT_DIR / "X.npy", X)
    np.save(OUT_DIR / "y.npy", y)
    meta.to_parquet(OUT_DIR / "meta.parquet", index=False)

    # Save label thresholds for inference
    import json
    with open(OUT_DIR / "label_thresholds.json", "w") as f:
        json.dump({"cfgi_bear_thr": float(q25), "cfgi_bull_thr": float(q75),
                   "n_features": N_FEATURES, "seq_len": SEQ_LEN,
                   "features": ["close_ret","atr14_pct","d_ema_spread_pct",
                                 "trend_strength","btc_ret_4h","side_sign"]}, f, indent=2)

    print(f"\n{_ts()} Saved to {OUT_DIR}")
    print(f"  X.npy: {X.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
