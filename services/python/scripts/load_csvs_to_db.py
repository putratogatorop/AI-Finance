"""One-shot backfill: load all 198 coins from data/raw/15m/ CSVs into asset_prices_15m.

Uses psycopg2 COPY for ~100x speedup over row-by-row inserts.
Idempotent — uses a staging table + INSERT...ON CONFLICT DO NOTHING.

Run from services/python/:
    python scripts/load_csvs_to_db.py
"""

import io
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2

sys.path.insert(0, ".")

DB_CONN = dict(host="localhost", port=5432, dbname="market",
               user="postgres", password="MySQL100%")
CSV_ROOT = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")


def load_coin_csvs(coin_dir: Path) -> pd.DataFrame:
    """Load and concat all monthly CSVs for one coin."""
    dfs = []
    for csv_path in sorted(coin_dir.glob("*.csv")):
        try:
            df = pd.read_csv(
                csv_path,
                names=["open_time", "open", "high", "low", "close", "volume",
                       "close_time", "quote_volume", "trades",
                       "taker_buy_base", "taker_buy_quote", "ignore"],
                header=0,
                dtype={"open_time": "int64"},
            )
            dfs.append(df)
        except Exception as e:
            print(f"  skip {csv_path.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def copy_coin_to_db(conn, coin: str, df: pd.DataFrame) -> int:
    """Bulk-load one coin's DataFrame into asset_prices_15m via COPY + staging."""
    if df.empty:
        return 0

    # Prep: handle ms (pre-2025) AND microseconds (2025+) CSVs
    df = df.copy()
    # Detect unit: ms timestamps are ~13 digits, us are ~16 digits
    df.loc[df["open_time"] > 2_000_000_000_000, "open_time"] //= 1000  # us -> ms
    # Filter to valid range (year 2020-2030 in ms)
    df = df[(df["open_time"] > 1_500_000_000_000) & (df["open_time"] < 2_000_000_000_000)]
    if df.empty:
        return 0
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["asset"] = coin
    df = df[[
        "asset", "timestamp", "open", "high", "low", "close",
        "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"
    ]]
    df = df.drop_duplicates(subset=["asset", "timestamp"])

    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, sep="\t", na_rep="\\N")
    buf.seek(0)

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE stage_prices (LIKE asset_prices_15m INCLUDING DEFAULTS)
            ON COMMIT DROP
        """)
        cur.execute("ALTER TABLE stage_prices DROP COLUMN id")
        cur.copy_expert("""
            COPY stage_prices (asset, timestamp, open, high, low, close,
                               volume, quote_volume, trades, taker_buy_base, taker_buy_quote)
            FROM STDIN WITH (FORMAT CSV, DELIMITER E'\t', NULL '\\N')
        """, buf)
        cur.execute("""
            INSERT INTO asset_prices_15m (asset, timestamp, open, high, low, close,
                volume, quote_volume, trades, taker_buy_base, taker_buy_quote)
            SELECT asset, timestamp, open, high, low, close,
                volume, quote_volume, trades, taker_buy_base, taker_buy_quote
            FROM stage_prices
            ON CONFLICT (asset, timestamp) DO NOTHING
        """)
        inserted = cur.rowcount
    conn.commit()
    return inserted


def main():
    start = time.time()
    coin_dirs = sorted([d for d in CSV_ROOT.iterdir() if d.is_dir()])
    print(f"Found {len(coin_dirs)} coin directories")

    conn = psycopg2.connect(**DB_CONN)

    total_inserted = 0
    for i, coin_dir in enumerate(coin_dirs, 1):
        coin = coin_dir.name
        t0 = time.time()
        df = load_coin_csvs(coin_dir)
        if df.empty:
            print(f"[{i:3d}/{len(coin_dirs)}] {coin:20s} — no data")
            continue
        inserted = copy_coin_to_db(conn, coin, df)
        total_inserted += inserted
        elapsed = time.time() - t0
        print(f"[{i:3d}/{len(coin_dirs)}] {coin:20s} {len(df):>8,} rows -> "
              f"{inserted:>8,} inserted ({elapsed:.1f}s)")

    conn.close()
    total_elapsed = time.time() - start
    print(f"\nDone. {total_inserted:,} rows inserted in {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
