"""One-off backfill: catch up on Gate.io pairs that were never ingested.

Reads universe from the 2026-04-23 snapshot (the freshest list of Gate.io
USDT-perp pairs we have). For each tradable pair (vol >= $500k, not delisting,
not leveraged) that has zero candles in postgres, pulls 90 days of history
from /spot/candlesticks.

Run from services/python/:
    python scripts/backfill_missing_pairs.py
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import psycopg2

# Force stdout to UTF-8 — some Gate.io USDT-perp symbols contain CJK characters
# (e.g., Chinese-named meme coins) which crash cp1252 default on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from ingest_candles import (  # noqa: E402, I001
    BOOTSTRAP_DAYS,
    CANDLE_SECONDS,
    MAX_BARS_PER_REQUEST,
    REQUEST_SLEEP_SEC,
    fetch_candles,
    upsert_candles,
)

# Build DB_CONN locally with URL-decoded password (ingest_candles.py's DB_CONN
# leaves the password URL-encoded — works for SQLAlchemy callers but not for
# direct psycopg2.connect, which expects the literal raw password).
_DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market"
)
_p = urlparse(_DB_URL)
DB_CONN = dict(
    host=_p.hostname,
    port=_p.port or 5432,
    dbname=(_p.path or "/").lstrip("/"),
    user=_p.username,
    password=unquote(_p.password) if _p.password else None,
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
    total_rows = 0
    failed = []

    for i, (pair, asset, vol) in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] {pair} (24h vol ${vol:,.0f}) ...")
        try:
            # Try the full BOOTSTRAP_DAYS window first; if that returns 0
            # bars (recent listing), retry with progressively shorter windows.
            pair_rows = 0
            for try_days in (BOOTSTRAP_DAYS, 30, 7, 1):
                cursor_ts = now_ts - try_days * 86400
                pair_rows_this_try = 0
                while cursor_ts < now_ts:
                    rows = fetch_candles(pair, from_ts=cursor_ts, limit=MAX_BARS_PER_REQUEST)
                    if not rows:
                        break
                    pair_rows_this_try += upsert_candles(conn, asset, rows)
                    last_ts = int(rows[-1]["timestamp"].timestamp())
                    if last_ts <= cursor_ts:
                        break
                    cursor_ts = last_ts + CANDLE_SECONDS
                    time.sleep(REQUEST_SLEEP_SEC)
                if pair_rows_this_try > 0:
                    pair_rows = pair_rows_this_try
                    if try_days < BOOTSTRAP_DAYS:
                        print(f"    (only ~{try_days}d of history available)")
                    break
            print(f"    +{pair_rows} bars")
            total_rows += pair_rows
            if pair_rows == 0:
                failed.append(asset)
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
