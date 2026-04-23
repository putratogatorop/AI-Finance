"""One-off backfill: catch up on Gate.io pairs that were never ingested.

Reads universe from the 2026-04-23 snapshot (the freshest list of Gate.io
USDT-perp pairs we have). For each tradable pair (vol >= $500k, not delisting,
not leveraged) that has zero candles in postgres, pulls 90 days of history
from /spot/candlesticks.

Run from services/python/:
    python scripts/backfill_missing_pairs.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from ingest_candles import (  # noqa: E402, I001
    BOOTSTRAP_DAYS,
    CANDLE_SECONDS,
    DB_CONN,
    MAX_BARS_PER_REQUEST,
    REQUEST_SLEEP_SEC,
    fetch_candles,
    upsert_candles,
)

LEVERAGED_RE = re.compile(r"[35][LS]_USDT$")
MIN_QUOTE_VOL_24H = 500_000


def main() -> None:
    snapshot_path = (
        Path(__file__).resolve().parents[3] / "data" / "snapshots" / "universe_2026-04-23.parquet"
    )
    universe = pd.read_parquet(snapshot_path)
    print(f"Universe rows: {len(universe)}")

    tradable = universe[
        (~universe["symbol"].str.contains(LEVERAGED_RE, na=False))
        & (~universe["in_delisting"].fillna(False))
        & (universe["quote_volume_24h"].fillna(0) >= MIN_QUOTE_VOL_24H)
    ].copy()
    tradable = tradable.sort_values("quote_volume_24h", ascending=False)
    print(f"Tradable pairs: {len(tradable)}")

    conn = psycopg2.connect(**DB_CONN)
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT asset FROM asset_prices_15m")
        existing = {row[0] for row in cur.fetchall()}
    print(f"Existing assets in DB: {len(existing)}")

    missing = []
    for _, row in tradable.iterrows():
        sym = row["symbol"]
        asset = sym.replace("_", "")  # BTC_USDT -> BTCUSDT
        if asset not in existing:
            missing.append((sym, asset, row["quote_volume_24h"]))
    print(f"Missing pairs to backfill: {len(missing)}")
    print()

    now_ts = int(time.time())
    start_ts = now_ts - BOOTSTRAP_DAYS * 86400
    total_rows = 0
    failed = []

    for i, (pair, asset, vol) in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] {pair} (24h vol ${vol:,.0f}) ...")
        try:
            cursor_ts = start_ts
            pair_rows = 0
            while cursor_ts < now_ts:
                rows = fetch_candles(pair, from_ts=cursor_ts, limit=MAX_BARS_PER_REQUEST)
                if not rows:
                    break
                pair_rows += upsert_candles(conn, asset, rows)
                last_ts = int(rows[-1]["timestamp"].timestamp())
                if last_ts <= cursor_ts:
                    break
                cursor_ts = last_ts + CANDLE_SECONDS
                time.sleep(REQUEST_SLEEP_SEC)
            print(f"    +{pair_rows} bars")
            total_rows += pair_rows
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            failed.append(asset)
            continue

    print()
    print(f"Done. Total bars inserted: {total_rows:,}")
    print(f"Failed: {len(failed)}: {', '.join(failed[:20])}")
    conn.close()


if __name__ == "__main__":
    main()
