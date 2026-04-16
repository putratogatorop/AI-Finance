"""Bulk 4h candle downloader from Binance Data Vision.

Downloads pre-packaged CSV files from data.binance.vision (static CDN, no API key).
This works from Indonesia even though api.binance.com is blocked.

Output: data/raw/4h/{SYMBOL}/{SYMBOL}-4h-{YYYY-MM}.csv
"""

import csv
import io
import logging
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
OUTPUT_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/4h")
START_YEAR = 2022
START_MONTH = 1
WORKERS = 8  # Parallel downloads

# ── Top 200+ USDT pairs (includes dead coins for survivorship bias) ──
SYMBOLS = [
    # Top 20 by market cap
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "TRXUSDT", "MATICUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT",
    "NEARUSDT", "APTUSDT", "OPUSDT", "ARBUSDT", "FILUSDT",
    # 21-50
    "ICPUSDT", "XLMUSDT", "HBARUSDT", "INJUSDT", "SUIUSDT",
    "AAVEUSDT", "MKRUSDT", "GRTUSDT", "IMXUSDT", "THETAUSDT",
    "FTMUSDT", "ALGOUSDT", "EGLDUSDT", "FLOWUSDT", "AXSUSDT",
    "SANDUSDT", "MANAUSDT", "APEUSDT", "CHZUSDT", "ENJUSDT",
    "CRVUSDT", "LRCUSDT", "SNXUSDT", "COMPUSDT", "YFIUSDT",
    "SUSHIUSDT", "1INCHUSDT", "ANKRUSDT", "KAVAUSDT", "IOTAUSDT",
    # 51-100
    "RUNEUSDT", "GALAUSDT", "ZECUSDT", "DASHUSDT", "BATUSDT",
    "ZILUSDT", "ONTUSDT", "QTUMUSDT", "WAVESUSDT", "NEOUSDT",
    "VETUSDT", "EOSUSDT", "XTZUSDT", "IOSTUSDT", "ZRXUSDT",
    "BCHUSDT", "ETCUSDT", "XMRUSDT", "SXPUSDT", "CELRUSDT",
    "RVNUSDT", "SKLUSDT", "COTIUSDT", "RSRUSDT", "STXUSDT",
    "MASKUSDT", "DYDXUSDT", "GMXUSDT", "LDOUSDT", "RNDRUSDT",
    "FETUSDT", "AGIXUSDT", "OCEANUSDT", "CFXUSDT", "ACHUSDT",
    "WOOUSDT", "MAGICUSDT", "PENDLEUSDT", "RDNTUSDT", "JOEUSDT",
    "HOOKUSDT", "HIGHUSDT", "SSVUSDT", "BLURUSDT", "IDUSDT",
    "ARBUSDT", "LQTYUSDT", "LEVERUSDT", "TLMUSDT", "AMBUSDT",
    # 101-150
    "MINAUSDT", "OGNUSDT", "PHBUSDT", "BICOUSDT", "GALUSDT",
    "XVSUSDT", "DARUSDT", "GMTUSDT", "STGUSDT", "EDUUSDT",
    "FLMUSDT", "OMUSDT", "BAKEUSDT", "ALPACAUSDT", "REEFUSDT",
    "DUSKUSDT", "LITUSDT", "TOMOUSDT", "CTKUSDT", "BELUSDT",
    "MDXUSDT", "C98USDT", "JASMYUSDT", "ARPAUSDT", "CTSIUSDT",
    "BANDUSDT", "KLAYUSDT", "STORJUSDT", "CELOUSDT", "CKBUSDT",
    "PERPUSDT", "TRUUSDT", "LPTUSDT", "AUDIOUSDT", "ROSEUSDT",
    "API3USDT", "NMRUSDT", "POWRUSDT", "SLPUSDT", "RENUSDT",
    "SUPERUSDT", "NKNUSDT", "DGBUSDT", "WAXPUSDT", "SCUSDT",
    "MTLUSDT", "SYSUSDT", "DENTUSDT", "HOTUSDT", "PEOPLEUSDT",
    # 151-200
    "LSKUSDT", "ARDRUSDT", "MDTUSDT", "POLYXUSDT", "GLMRUSDT",
    "AGLDUSDT", "RADUSDT", "BETAUSDT", "RAREUSDT", "SUNUSDT",
    "PORTOUSDT", "LAZIOUSDT", "ATAUSDT", "FORUSDT", "VOXELUSDT",
    "ALPINEUSDT", "ACMUSDT", "CITYUSDT", "SANTOS usdt",
    "TFUELUSDT", "PUNDIXUSDT", "BTRSTUSDT", "LOKAUSDT", "XECUSDT",
    "RIFUSDT", "PROSUSDT", "VIBUSDT", "WINGUSDT", "FTTUSDT",
    "LUNAUSDT", "USTCUSDT", "ANCUSDT",  # Dead coins — survivorship bias
    # Extras: meme coins, new listings
    "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT", "WIFUSDT",
    "MEMEUSDT", "TURBOUSDT", "ORDIUSDT", "TIAUSDT", "SEIUSDT",
    "JUPUSDT", "WUSDT", "STRKUSDT", "PYTHUSDT", "JTOUSDT",
    "ALTUSDT", "MANTAUSDT", "PIXELUSDT", "PORTALUSDT", "ACEUSDT",
]

# Deduplicate and clean
SYMBOLS = list(dict.fromkeys(s.strip().upper().replace(" ", "") for s in SYMBOLS if s.strip()))


def generate_months(start_year, start_month):
    """Generate YYYY-MM strings from start to now."""
    now = datetime.utcnow()
    months = []
    y, m = start_year, start_month
    while (y, m) <= (now.year, now.month):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def download_one(symbol, month):
    """Download one month of 4h candles for one symbol. Returns (symbol, month, rows, status)."""
    out_dir = OUTPUT_DIR / symbol
    out_file = out_dir / f"{symbol}-4h-{month}.csv"

    # Skip if already downloaded
    if out_file.exists() and out_file.stat().st_size > 100:
        return (symbol, month, 0, "skipped")

    url = f"{BASE_URL}/{symbol}/4h/{symbol}-4h-{month}.zip"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urlopen(req, timeout=15)
        zip_data = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                raw = f.read().decode("utf-8")

        # Parse and re-save with clean headers
        reader = csv.reader(io.StringIO(raw))
        rows = list(reader)

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore",
            ])
            writer.writerows(rows)

        return (symbol, month, len(rows), "ok")

    except HTTPError as e:
        if e.code == 404:
            return (symbol, month, 0, "not_found")
        return (symbol, month, 0, f"http_{e.code}")
    except Exception as e:
        return (symbol, month, 0, f"error: {str(e)[:50]}")


def main():
    months = generate_months(START_YEAR, START_MONTH)
    logger.info(f"Binance Data Vision Bulk Downloader")
    logger.info(f"Symbols: {len(SYMBOLS)}")
    logger.info(f"Months: {len(months)} ({months[0]} to {months[-1]})")
    logger.info(f"Max downloads: {len(SYMBOLS) * len(months):,}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Workers: {WORKERS}")
    logger.info("=" * 60)

    # Build task list
    tasks = [(sym, month) for sym in SYMBOLS for month in months]

    results = {"ok": 0, "skipped": 0, "not_found": 0, "error": 0}
    total_rows = 0
    symbols_with_data = set()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(download_one, sym, month): (sym, month) for sym, month in tasks}

        done = 0
        total = len(futures)
        last_log = time.time()

        for future in as_completed(futures):
            sym, month, rows, status = future.result()
            done += 1
            total_rows += rows

            if status == "ok":
                results["ok"] += 1
                symbols_with_data.add(sym)
            elif status == "skipped":
                results["skipped"] += 1
                if rows == 0:
                    symbols_with_data.add(sym)  # Already had data
            elif status == "not_found":
                results["not_found"] += 1
            else:
                results["error"] += 1

            # Log progress every 5 seconds
            if time.time() - last_log > 5 or done == total:
                last_log = time.time()
                pct = done / total * 100
                logger.info(
                    f"  [{done:,}/{total:,}] {pct:.0f}% | "
                    f"ok={results['ok']:,} skip={results['skipped']:,} "
                    f"miss={results['not_found']:,} err={results['error']} | "
                    f"{total_rows:,} rows | {len(symbols_with_data)} symbols"
                )

    # Final summary
    logger.info("=" * 60)
    logger.info(f"DONE!")
    logger.info(f"  Symbols with data: {len(symbols_with_data)}")
    logger.info(f"  Total candle rows: {total_rows:,}")
    logger.info(f"  Downloaded: {results['ok']:,}")
    logger.info(f"  Skipped (cached): {results['skipped']:,}")
    logger.info(f"  Not found (404): {results['not_found']:,}")
    logger.info(f"  Errors: {results['error']}")

    # List symbols with data
    logger.info(f"\nSymbols with data ({len(symbols_with_data)}):")
    for sym in sorted(symbols_with_data):
        # Count files
        sym_dir = OUTPUT_DIR / sym
        files = list(sym_dir.glob("*.csv")) if sym_dir.exists() else []
        logger.info(f"  {sym}: {len(files)} months")

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "months", "first_month", "last_month"])
        for sym in sorted(symbols_with_data):
            sym_dir = OUTPUT_DIR / sym
            files = sorted(sym_dir.glob("*.csv"))
            if files:
                first = files[0].stem.split("-4h-")[1]
                last = files[-1].stem.split("-4h-")[1]
                writer.writerow([sym, len(files), first, last])
    logger.info(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
