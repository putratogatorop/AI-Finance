"""v_new_1.6 — Merge narrative extras onto base features.

Reads features_full_v1_6_extras.parquet (cohort + 10 category cols + cohort
diagnostic + membership count) and merges onto either:
- features_full_v1_5.parquet  (if --base v1_5; v_new_1.5 won the verdict)
- features_full.parquet       (if --base v1     ; v_new_1.5 didn't pass)

Output: features_full_v1_6.parquet, ready for phase3_train with V_NEW_1_6_MODE=1
(and V_NEW_1_6_WITH_SEQUENCE=1 if base is v1_5).

USAGE:
    services/python/.venv/bin/python3 \\
        services/python/scripts/v_new_1_6_merge_features.py --base v1_5
    services/python/.venv/bin/python3 \\
        services/python/scripts/v_new_1_6_merge_features.py --base v1
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                    str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=["v1", "v1_5"], required=True)
    args = parser.parse_args()

    t0 = time.time()
    base_name = "features_full.parquet" if args.base == "v1" else "features_full_v1_5.parquet"
    base_path = DATA / base_name
    extras_path = DATA / "features_full_v1_6_extras.parquet"
    out_path = DATA / "features_full_v1_6.parquet"

    print(f"[merge] base = {base_path}")
    print(f"[merge] extras = {extras_path}")
    if not base_path.exists():
        print(f"[merge] ERROR: base parquet missing")
        sys.exit(1)
    if not extras_path.exists():
        print(f"[merge] ERROR: extras parquet missing — run v_new_1_6_build_extras.py first")
        sys.exit(1)

    base = pd.read_parquet(base_path)
    extras = pd.read_parquet(extras_path)
    print(f"[merge] base shape:   {base.shape}")
    print(f"[merge] extras shape: {extras.shape}")

    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    extras["timestamp"] = pd.to_datetime(extras["timestamp"], utc=True)

    # Drop pre-existing v_new_1.6 cols if any (idempotent reruns)
    extras_cols = [c for c in extras.columns if c not in ("symbol", "timestamp")]
    pre = [c for c in extras_cols if c in base.columns]
    if pre:
        print(f"[merge] dropping {len(pre)} pre-existing v_new_1.6 cols from base")
        base = base.drop(columns=pre)

    out = base.merge(extras[["symbol", "timestamp"] + extras_cols],
                     on=["symbol", "timestamp"], how="left")
    print(f"[merge] result shape: {out.shape}")

    nan_pct = {c: round(100.0 * out[c].isna().mean(), 1) for c in extras_cols}
    print(f"[merge] NaN% in extras after join: {nan_pct}")

    out.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    sz_mb = out_path.stat().st_size / 1e6
    print(f"[merge] DONE in {elapsed:.1f}s — wrote {out_path} ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
