"""Incremental candle ingestion — keeps asset_prices_15m always-current.

On startup: backfills the gap between last DB bar and now from Gate.io.
Every 15 min: fetches the latest few closed bars per coin, upserts to DB.

Uses ON CONFLICT DO NOTHING — safe to re-run anytime.

Run from services/python/:
    python scripts/ingest_candles.py
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psycopg2
from psycopg2.extras import execute_values

sys.path.insert(0, ".")

DB_CONN = dict(host="localhost", port=5432, dbname="market",
               user="postgres", password="MySQL100%")
GATEIO_BASE = "https://api.gateio.ws/api/v4"

CANDLE_INTERVAL_STR = "15m"
CANDLE_SECONDS = 15 * 60
MAX_BARS_PER_REQUEST = 1000     # Gate.io limit
REQUEST_SLEEP_SEC = 0.1         # be polite to API
POLL_INTERVAL_SEC = 60 * 15     # run ingest every 15 min
LIVE_BARS_PER_COIN = 4          # on each tick, fetch last 4 bars (handles missed cycles)

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/ingest_candles.log", mode="a"),
    ],
)
logger = logging.getLogger("ingest")


def fetch_json(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            req = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
            resp = urlopen(req, timeout=15)
            return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:200]
            except Exception:
                pass
            logger.warning(f"HTTP {e.code} for {url}: {body}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except (URLError, TimeoutError) as e:
            logger.warning(f"Net error ({url[:120]}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def fetch_candles(pair: str, from_ts: int | None = None, limit: int = MAX_BARS_PER_REQUEST):
    """Fetch 15m candles. If from_ts given, fetch forward from there; else latest N."""
    url = f"{GATEIO_BASE}/spot/candlesticks?currency_pair={pair}&interval={CANDLE_INTERVAL_STR}"
    if from_ts is not None:
        # Gate.io counts the end bar — (limit-1) * CANDLE_SECONDS = 1000 bars max
        to_ts = min(from_ts + (limit - 1) * CANDLE_SECONDS, int(time.time()))
        url += f"&from={from_ts}&to={to_ts}"
    else:
        url += f"&limit={limit}"
    data = fetch_json(url)
    if not data or not isinstance(data, list):
        return []
    rows = []
    for c in data:
        # Gate.io: [timestamp, volume, close, high, low, open, quote_volume, is_window_closed]
        try:
            rows.append({
                "timestamp": datetime.fromtimestamp(int(c[0]), tz=timezone.utc),
                "volume": float(c[1]),
                "close": float(c[2]),
                "high": float(c[3]),
                "low": float(c[4]),
                "open": float(c[5]),
                "quote_volume": float(c[6]) if len(c) > 6 else 0.0,
            })
        except (ValueError, IndexError):
            continue
    return rows


def _get_extreme_ts_per_asset(conn, direction: str = "DESC") -> dict[str, datetime]:
    """Get MAX (DESC) or MIN (ASC) timestamp per asset using fast per-asset index lookups.

    Uses an explicit loop over distinct assets because PG's planner can't
    turn MAX/DISTINCT ON into loose index scans for a composite index — it falls
    back to a seq scan on all 19M rows. 198 index lookups @ ~0.1ms each = ~20ms.
    """
    # Loose index scan to enumerate distinct assets using composite (asset, timestamp) index.
    # Plain SELECT DISTINCT asset triggers a seq scan on 19M rows; this does ~200 index seeks.
    with conn.cursor() as cur:
        cur.execute("""
            WITH RECURSIVE t AS (
                (SELECT asset FROM asset_prices_15m ORDER BY asset LIMIT 1)
                UNION ALL
                (SELECT (SELECT asset FROM asset_prices_15m WHERE asset > t.asset
                         ORDER BY asset LIMIT 1)
                 FROM t WHERE t.asset IS NOT NULL)
            )
            SELECT asset FROM t WHERE asset IS NOT NULL
        """)
        assets = [r[0] for r in cur.fetchall()]
        out: dict[str, datetime] = {}
        for asset in assets:
            cur.execute(
                f"SELECT timestamp FROM asset_prices_15m WHERE asset = %s "
                f"ORDER BY timestamp {direction} LIMIT 1",
                (asset,),
            )
            row = cur.fetchone()
            if row:
                out[asset] = row[0]
    return out


def get_asset_last_timestamps(conn) -> dict[str, datetime]:
    """Return {asset: max_timestamp}. ~20ms via per-asset index seeks."""
    return _get_extreme_ts_per_asset(conn, "DESC")


def upsert_candles(conn, asset: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    values = [
        (asset, r["timestamp"], r["open"], r["high"], r["low"], r["close"],
         r["volume"], r["quote_volume"])
        for r in rows
    ]
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO asset_prices_15m
                (asset, timestamp, open, high, low, close, volume, quote_volume)
            VALUES %s
            ON CONFLICT (asset, timestamp) DO NOTHING
        """, values)
        inserted = cur.rowcount
    conn.commit()
    return inserted


def asset_to_pair(asset: str) -> str:
    """BTCUSDT -> BTC_USDT."""
    if asset.endswith("USDT"):
        return asset[:-4] + "_USDT"
    return asset


STRICT_FLAG_FILE = Path("logs/.last_strict_scan")
LENIENT_LOOKBACK_DAYS = 30


def find_gaps_for_asset(conn, asset: str, since_ts: datetime) -> list[tuple[int, int]]:
    """Find missing 15-min timestamps for an asset since `since_ts`.

    Returns list of (from_epoch, to_epoch) ranges to fetch. Consecutive missing
    bars are merged into one range; isolated gaps become single-bar ranges.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT timestamp FROM asset_prices_15m
            WHERE asset = %s AND timestamp >= %s
            ORDER BY timestamp
        """, (asset, since_ts))
        have = [r[0] for r in cur.fetchall()]

    if not have:
        # No data at all in window — treat whole window as one gap
        return [(int(since_ts.timestamp()), int(time.time()))]

    # Expected: since_ts aligned to 15m boundary, stepping by CANDLE_SECONDS to now
    now_ts = int(time.time()) // CANDLE_SECONDS * CANDLE_SECONDS
    # Align since_ts to 15m boundary (ceil)
    since_epoch = int(since_ts.timestamp())
    since_epoch = ((since_epoch + CANDLE_SECONDS - 1) // CANDLE_SECONDS) * CANDLE_SECONDS

    have_set = {int(ts.timestamp()) for ts in have}
    missing = []
    cur_ts = since_epoch
    while cur_ts < now_ts:
        if cur_ts not in have_set:
            missing.append(cur_ts)
        cur_ts += CANDLE_SECONDS

    if not missing:
        return []

    # Merge consecutive missing timestamps into ranges
    ranges: list[tuple[int, int]] = []
    range_start = missing[0]
    prev = missing[0]
    for ts in missing[1:]:
        if ts == prev + CANDLE_SECONDS:
            prev = ts
        else:
            ranges.append((range_start, prev))
            range_start = ts
            prev = ts
    ranges.append((range_start, prev))
    return ranges


def fill_gaps(conn, strict: bool):
    """CDC-style gap filler.

    Strict: scan full history for gaps (slow, thorough).
    Lenient: scan last LENIENT_LOOKBACK_DAYS for gaps (fast).
    """
    mode = "STRICT (full history)" if strict else f"LENIENT (last {LENIENT_LOOKBACK_DAYS} days)"
    logger.info(f"CDC gap scan: {mode}")

    last_ts_map = get_asset_last_timestamps(conn)
    if not last_ts_map:
        logger.warning("No assets in DB — skipping gap scan")
        return

    if strict:
        # Scan from each asset's earliest timestamp
        since_map = _get_extreme_ts_per_asset(conn, "ASC")
    else:
        cutoff = datetime.now(timezone.utc).timestamp() - LENIENT_LOOKBACK_DAYS * 86400
        cutoff_dt = datetime.fromtimestamp(cutoff, tz=timezone.utc)
        since_map = {a: cutoff_dt for a in last_ts_map}

    total_filled = 0
    total_assets = len(since_map)
    for i, (asset, since_ts) in enumerate(sorted(since_map.items()), 1):
        gaps = find_gaps_for_asset(conn, asset, since_ts)
        if not gaps:
            continue

        pair = asset_to_pair(asset)
        filled = 0
        for from_ts, to_ts in gaps:
            # Fetch in 1000-bar chunks
            cursor_ts = from_ts
            while cursor_ts <= to_ts:
                rows = fetch_candles(pair, from_ts=cursor_ts, limit=MAX_BARS_PER_REQUEST)
                if not rows:
                    break
                filled += upsert_candles(conn, asset, rows)
                last_fetched = int(rows[-1]["timestamp"].timestamp())
                if last_fetched <= cursor_ts:
                    break
                cursor_ts = last_fetched + CANDLE_SECONDS
                time.sleep(REQUEST_SLEEP_SEC)

        total_filled += filled
        if filled > 0 or i % 25 == 0:
            logger.info(f"  gap-fill {i}/{total_assets}: {asset} "
                        f"+{filled} bars ({len(gaps)} ranges)")

    logger.info(f"Gap-fill complete: {total_filled} bars across {total_assets} assets")


def should_run_strict() -> bool:
    """True on first run of calendar day."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        last = STRICT_FLAG_FILE.read_text().strip()
        if last == today:
            return False
    except FileNotFoundError:
        pass
    STRICT_FLAG_FILE.write_text(today)
    return True


def backfill_gap(conn):
    """For each asset, fetch missing bars from last_ts to now."""
    last_ts = get_asset_last_timestamps(conn)
    if not last_ts:
        logger.warning("No assets in DB yet — run load_csvs_to_db.py first")
        return

    now_ts = int(time.time())
    total_inserted = 0
    total_assets = len(last_ts)

    for i, (asset, last) in enumerate(sorted(last_ts.items()), 1):
        pair = asset_to_pair(asset)
        from_ts = int(last.timestamp()) + CANDLE_SECONDS
        gap_bars = max(0, (now_ts - from_ts) // CANDLE_SECONDS)

        if gap_bars < 1:
            continue

        coin_inserted = 0
        # Paginate — Gate.io caps 1000 bars per call
        cursor_ts = from_ts
        pages = 0
        while cursor_ts < now_ts and pages < 10:  # safety cap: max 10k bars per coin per cycle
            rows = fetch_candles(pair, from_ts=cursor_ts, limit=MAX_BARS_PER_REQUEST)
            if not rows:
                break
            coin_inserted += upsert_candles(conn, asset, rows)
            last_fetched = rows[-1]["timestamp"]
            cursor_ts = int(last_fetched.timestamp()) + CANDLE_SECONDS
            pages += 1
            time.sleep(REQUEST_SLEEP_SEC)

        total_inserted += coin_inserted
        if i % 20 == 0 or i == total_assets:
            logger.info(f"  backfill {i}/{total_assets}: "
                        f"{asset} +{coin_inserted} bars (total {total_inserted})")

    logger.info(f"Backfill complete: {total_inserted} new bars across {total_assets} assets")


def tick_ingest(conn):
    """Fetch last N bars for every asset, upsert."""
    last_ts = get_asset_last_timestamps(conn)
    total_inserted = 0
    for asset in last_ts.keys():
        pair = asset_to_pair(asset)
        rows = fetch_candles(pair, from_ts=None, limit=LIVE_BARS_PER_COIN)
        if rows:
            total_inserted += upsert_candles(conn, asset, rows)
        time.sleep(REQUEST_SLEEP_SEC)
    return total_inserted


def main():
    logger.info("=" * 60)
    logger.info("CANDLE INGEST DAEMON")
    logger.info("=" * 60)

    conn = psycopg2.connect(**DB_CONN)

    # CDC gap scan: strict on first run of the day, lenient otherwise
    strict = should_run_strict()
    t0 = time.time()
    fill_gaps(conn, strict=strict)
    logger.info(f"Gap scan took {time.time() - t0:.0f}s")

    # Trailing backfill (catches anything between last bar and right now)
    logger.info("Starting trailing backfill...")
    t0 = time.time()
    backfill_gap(conn)
    logger.info(f"Backfill complete in {time.time() - t0:.0f}s")

    logger.info(f"Entering live loop — polling every {POLL_INTERVAL_SEC}s")

    cycle = 0
    while True:
        try:
            cycle += 1
            t0 = time.time()
            inserted = tick_ingest(conn)
            elapsed = time.time() - t0
            logger.info(f"Cycle {cycle}: +{inserted} bars ({elapsed:.0f}s)")

            sleep = max(0, POLL_INTERVAL_SEC - elapsed)
            time.sleep(sleep)

        except KeyboardInterrupt:
            logger.info("Ingest daemon stopped by user.")
            break
        except Exception as e:
            logger.error(f"Cycle {cycle} error: {e}", exc_info=True)
            time.sleep(60)

    conn.close()


if __name__ == "__main__":
    main()
