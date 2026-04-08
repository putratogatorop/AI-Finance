"""Download 15-minute candles from Binance Vision for v4 assets.

Output: data/raw/15m/{SYMBOL}/{SYMBOL}-15m-{YYYY-MM}.csv
Source: https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/15m/
"""

import csv
import io
import logging
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    "C:/Users/togat/Desktop/AI-Finance/data/raw/15m",
))
START_YEAR = 2023
START_MONTH = 4  # April 2023 → 3 years to April 2026
WORKERS = 4

V4_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT",
    "XRPUSDT", "AVAXUSDT", "LINKUSDT",
]

CLEAN_HEADERS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


def generate_months(start_year: int, start_month: int,
                    end_year: int | None = None, end_month: int | None = None) -> list[str]:
    now = datetime.now(timezone.utc)
    ey = end_year or now.year
    em = end_month or now.month
    months = []
    y, m = start_year, start_month
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def parse_binance_csv(raw_text: str) -> list[dict]:
    reader = csv.reader(io.StringIO(raw_text))
    rows = []
    for row in reader:
        if len(row) < 12:
            continue
        rows.append({
            "open_time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "close_time": int(row[6]),
            "quote_volume": float(row[7]),
            "trades": int(row[8]),
            "taker_buy_base": float(row[9]),
            "taker_buy_quote": float(row[10]),
        })
    return rows


def download_one(symbol: str, month: str) -> tuple[str, str, int, str]:
    out_dir = OUTPUT_DIR / symbol
    out_file = out_dir / f"{symbol}-15m-{month}.csv"

    if out_file.exists() and out_file.stat().st_size > 100:
        return (symbol, month, 0, "skipped")

    url = f"{BASE_URL}/{symbol}/15m/{symbol}-15m-{month}.zip"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urlopen(req, timeout=30)
        zip_data = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                raw = f.read().decode("utf-8")

        parsed = parse_binance_csv(raw)
        if not parsed:
            return (symbol, month, 0, "empty")

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CLEAN_HEADERS)
            for row in parsed:
                writer.writerow([
                    row["open_time"], row["open"], row["high"], row["low"],
                    row["close"], row["volume"], row["close_time"],
                    row["quote_volume"], row["trades"], row["taker_buy_base"],
                    row["taker_buy_quote"], 0,
                ])

        return (symbol, month, len(parsed), "ok")

    except HTTPError as e:
        if e.code == 404:
            return (symbol, month, 0, "not_found")
        return (symbol, month, 0, f"http_{e.code}")
    except Exception as e:
        return (symbol, month, 0, f"error: {str(e)[:80]}")


def main():
    months = generate_months(START_YEAR, START_MONTH)
    logger.info("Binance Vision 15m Downloader (v4)")
    logger.info(f"Symbols: {V4_SYMBOLS}")
    logger.info(f"Months: {len(months)} ({months[0]} to {months[-1]})")
    logger.info(f"Output: {OUTPUT_DIR}")

    tasks = [(sym, month) for sym in V4_SYMBOLS for month in months]
    results = {"ok": 0, "skipped": 0, "not_found": 0, "error": 0}
    total_rows = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(download_one, s, m): (s, m) for s, m in tasks}
        done = 0
        total = len(futures)
        last_log = time.time()

        for future in as_completed(futures):
            sym, month, rows, status = future.result()
            done += 1
            total_rows += rows

            if status == "ok":
                results["ok"] += 1
            elif status == "skipped":
                results["skipped"] += 1
            elif status == "not_found":
                results["not_found"] += 1
            else:
                results["error"] += 1

            if time.time() - last_log > 5 or done == total:
                last_log = time.time()
                pct = done / total * 100
                logger.info(
                    f"  [{done}/{total}] {pct:.0f}% | "
                    f"ok={results['ok']} skip={results['skipped']} "
                    f"miss={results['not_found']} err={results['error']} | "
                    f"{total_rows:,} rows"
                )

    logger.info("=" * 60)
    logger.info(f"DONE! Total candle rows: {total_rows:,}")
    logger.info(f"Downloaded: {results['ok']}, Skipped: {results['skipped']}, "
                f"Not found: {results['not_found']}, Errors: {results['error']}")

    # Save manifest
    manifest = OUTPUT_DIR / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "months", "first_month", "last_month"])
        for sym in V4_SYMBOLS:
            sym_dir = OUTPUT_DIR / sym
            files = sorted(sym_dir.glob("*.csv")) if sym_dir.exists() else []
            if files:
                first = files[0].stem.split("-15m-")[1]
                last = files[-1].stem.split("-15m-")[1]
                writer.writerow([sym, len(files), first, last])
    logger.info(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
