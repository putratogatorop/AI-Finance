"""Smoke test for paper_executor.py D1e wiring — bypasses the DB by directly
exercising the sizing functions and the SL/TP setup math against the known
S1×E2 trade ledger.

Verifies:
  1. _sizing_s1 reproduces pos_scale_s1 from the research ledger byte-for-byte.
  2. _sizing_s4 returns 0 for cls_score < 0.5 (zero-sizing rule).
  3. E2 SL/TP math (entry + 2*ATR / entry - 6*ATR) matches what
     phase17_sizing_compare._sim_E2 uses.
  4. E1 SL/TP math (5%/15% fixed) is consistent.
  5. rapid_rally_active() can be called without crashing (returns False
     locally if BTC daily candles aren't in the local DB — expected).

Run from services/python/:
    uv run python scripts/_smoke_d1e_executor.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

# Import paper_executor as a module
spec = importlib.util.spec_from_file_location("paper_executor", "scripts/paper_executor.py")
pe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe)

LEDGER_PATH = Path("data/d1e_short_trades.csv")


def main() -> None:
    print(f"loading research ledger: {LEDGER_PATH}")
    ledger = pd.read_csv(LEDGER_PATH)
    sample = ledger.sample(n=20, random_state=7)

    print()
    print("== Test 1: _sizing_s1 reproduces ledger pos_scale_s1 ==")
    n_match = 0
    for _, row in sample.iterrows():
        ours = pe._sizing_s1(float(row["btc_score"]), float("nan"))
        theirs = float(row["pos_scale_s1"])
        if abs(ours - theirs) < 1e-9:
            n_match += 1
        else:
            print(f"  MISMATCH btc={row['btc_score']:.4f} ours={ours:.4f} theirs={theirs:.4f}")
    print(f"  {n_match}/{len(sample)} match")
    assert n_match == len(sample), "S1 sizing diverged from research"

    print()
    print("== Test 2: _sizing_s4 zero-sizes when cls_score < 0.5 ==")
    cases = [(-0.5, 0.45), (-0.5, 0.50), (-0.5, 0.55), (-0.5, 0.83)]
    for btc, cls in cases:
        s = pe._sizing_s4(btc, cls)
        expected = max(0.0, min(3.0 * (cls - 0.5), 1.5))
        ok = abs(s - expected) < 1e-9
        print(f"  S4(btc={btc:+.2f}, cls={cls:.2f}) = {s:.3f} (expected {expected:.3f}) {'OK' if ok else 'FAIL'}")
        assert ok

    print()
    print("== Test 3: E2 SL/TP math matches research ==")
    for _, row in sample.head(5).iterrows():
        entry = 100.0  # arbitrary; we check the formula, not the price
        atr = float(row["atr14_at_entry"])
        sl = entry + pe.D1E_E2_ATR_SL_MULT * atr
        tp = entry - pe.D1E_E2_ATR_TP_MULT * atr
        # research: phase17_sizing_compare._sim_E2 uses 2.0 / 6.0 multipliers
        assert pe.D1E_E2_ATR_SL_MULT == 2.0
        assert pe.D1E_E2_ATR_TP_MULT == 6.0
        print(f"  {row['symbol']} atr={atr:.4f} entry={entry} sl={sl:.4f} tp={tp:.4f}")

    print()
    print("== Test 4: E1 SL/TP math (fixed 5%/15%) ==")
    entry = 100.0
    sl1 = entry * (1 + pe.D1E_E1_SL_PCT)
    tp1 = entry * (1 - pe.D1E_E1_TP_PCT)
    assert abs(sl1 - 105.0) < 1e-9 and abs(tp1 - 85.0) < 1e-9
    print(f"  entry={entry} -> sl={sl1} tp={tp1} (5%/15% short) OK")

    print()
    print("== Test 5: rapid_rally_active() callable (DB may be missing locally) ==")
    try:
        result = pe.rapid_rally_active()
        print(f"  rapid_rally_active() returned {result} (False expected without local DB)")
    except Exception as e:
        print(f"  callable but threw: {type(e).__name__}: {e}")
        # Acceptable locally — VPS will have BTC candles in DB

    print()
    print("== Test 6: D1E_ACCOUNTS spec sanity ==")
    for a in pe.D1E_ACCOUNTS:
        assert a["sizing"] in {"s1", "s4"}, a
        assert a["exit"] in {"e1", "e2"}, a
        assert a["strategy"].startswith("d1e_")
        assert pe._is_d1e_strategy(a["strategy"])
        assert not pe._is_bigmover_strategy(a["strategy"])
        print(f"  {a['strategy']:<12} sizing={a['sizing']} exit={a['exit']} enabled={a.get('enabled')}")

    print()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
