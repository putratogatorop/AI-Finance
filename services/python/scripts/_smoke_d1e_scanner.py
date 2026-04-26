"""Smoke test for live_scanner_d1e.py — runs the detector against the
2026-04-01 snapshot for a few sample D1e trades from the research ledger
and confirms the live detector produces matching internals + classifier
scores.

This is NOT a deployment test (we have no live DB locally). It verifies the
detect_d1e_short() + classifier wiring is correct before we ship.

Run from services/python/:
    uv run python scripts/_smoke_d1e_scanner.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.ml.bigmover_combined.features import FeatureContext, FEATURE_NAMES
from scripts.live_scanner_d1e import (
    ATR_DROP_MULT, VOL_RATIO_MIN, BREADTH_MIN,
    detect_d1e_short, load_classifier, score_signal,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DATE = "2026-04-01"
CANDLES_PATH = REPO_ROOT / "data" / "snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
LEDGER_PATH = Path("data/d1e_short_trades.csv")


def main() -> None:
    print(f"loading snapshot candles: {CANDLES_PATH}")
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    print(f"  {len(candles):,} rows, {candles['asset'].nunique()} assets")

    print("building FeatureContext (BTC trend + breadth panel)...")
    ctx = FeatureContext(candles=candles, btc_asset_key="BTCUSDT")
    print("  done")

    print(f"loading research ledger: {LEDGER_PATH}")
    ledger = pd.read_csv(LEDGER_PATH)
    ledger["entry_time"] = pd.to_datetime(ledger["entry_time"], utc=True)
    # Pick a handful of recent OOT trades (post-2026-01-01)
    sample = ledger[ledger["entry_time"] >= "2026-01-01"].sample(n=8, random_state=42)
    print(f"  sampled {len(sample)} OOT trades from {len(ledger):,} total")

    model, threshold, feat_order = load_classifier()
    print(f"v4: threshold={threshold:.4f} features={len(feat_order)}")
    print()

    matched_detector = 0
    matched_features = 0
    BAR_15M = pd.Timedelta(minutes=15)
    for _, row in sample.iterrows():
        sym = row["asset"]
        # Trade ledger stores entry_time = bar i+1 (the FILL bar). The
        # detector fires at the previous bar's close, so signal_time = entry_time - 15m.
        entry_time = row["entry_time"]
        signal_time = entry_time - BAR_15M
        ledger_btc_score = float(row["btc_score"])
        ledger_atr = float(row["atr14_at_entry"])

        # Build a per-asset df ending at the SIGNAL bar (where detector fired).
        sub = (candles[candles["asset"] == sym]
               .sort_values("timestamp")
               .reset_index(drop=True))
        idx = int(np.searchsorted(sub["timestamp"].values,
                                  np.datetime64(signal_time.tz_convert(None)),
                                  side="right")) - 1
        if idx < 100:
            print(f"  {sym} @ {entry_time}: not enough history, skip")
            continue
        df_at_entry = sub.iloc[: idx + 1].copy()

        hit = detect_d1e_short(ctx, sym, df_at_entry, entry_time=signal_time)
        if hit is None:
            # Diagnose which gate killed it
            view = ctx._asset_view(sym)
            entry_naive = signal_time.tz_convert(None)
            i = int(np.searchsorted(view.ts, np.datetime64(entry_naive), side="right")) - 1
            if i < 100:
                print(f"  {sym} @ {entry_time}: REJECTED reason=insufficient_history (i={i})")
                continue
            c, o = float(view.close[i]), float(view.open[i])
            rh = float(view.rolling_high96[i]) if i < len(view.rolling_high96) else float("nan")
            atr = float(view.atr14[i]) if i < len(view.atr14) else float("nan")
            volma = float(view.vol_ma20[i]) if i < len(view.vol_ma20) else float("nan")
            vol = float(view.volume[i])
            drop_norm = (rh - c) / atr if (np.isfinite(rh) and atr > 0) else float("nan")
            vr = vol / volma if volma > 0 else float("nan")
            btc_score = ctx._btc_trend_score_15m.asof(signal_time) if ctx._btc_trend_score_15m is not None else float("nan")
            try:
                row = ctx._ret_4h_panel.loc[signal_time]
                breadth = float((row < 0).sum() / row.notna().sum()) if row.notna().any() else float("nan")
            except KeyError:
                breadth = float("nan")
            print(
                f"  {sym} @ {entry_time}: REJECTED  red={c<o} drop={drop_norm:.2f}xATR (need>=2.5) "
                f"vol={vr:.2f}x (need>=2.0) breadth={breadth:.2f} (need>=0.6) btc={btc_score:.3f} (need<0)"
            )
            continue
        matched_detector += 1

        cls_score = score_signal(model, feat_order, hit["features"])
        kept = bool(np.isfinite(cls_score) and cls_score >= threshold)

        # Compare detector internals to the research ledger
        atr_match = abs(hit["atr14_at_entry"] - ledger_atr) / max(abs(ledger_atr), 1e-9) < 1e-3
        btc_match = abs(hit["btc_score"] - ledger_btc_score) < 1e-3
        if atr_match and btc_match:
            matched_features += 1
        print(
            f"  {sym} @ {entry_time}: drop={hit['drop_atr_norm']:.2f}xATR "
            f"vol={hit['vol_ratio']:.2f}x breadth={hit['breadth_4h']:.2f} "
            f"btc={hit['btc_score']:+.3f} cls={cls_score:.4f} "
            f"kept={kept} atr_match={atr_match} btc_match={btc_match}"
        )

    print()
    print(f"DETECTOR matched on {matched_detector}/{len(sample)} sampled trades")
    print(f"FEATURE PARITY (atr+btc_score) on {matched_features}/{len(sample)} sampled trades")
    if matched_detector >= len(sample) * 0.7 and matched_features >= len(sample) * 0.7:
        print("SMOKE OK")
        return
    print("SMOKE FAIL — divergence between live detector and research ledger; investigate")
    sys.exit(1)


if __name__ == "__main__":
    main()
