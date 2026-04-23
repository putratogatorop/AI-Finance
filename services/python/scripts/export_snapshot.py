"""Export a postgres + Gate.io snapshot to Parquet for backtest reproducibility.

Creates two files in data/snapshots/:
  candles_15m_<date>.parquet  — OHLCV for all pairs with timestamp < <date> 00:00 UTC
  universe_<date>.parquet     — listed Gate.io futures pairs + is_leveraged flag

Also appends one row to data/snapshots/MANIFEST.md with row counts + sha256s.

Usage: python scripts/export_snapshot.py --date 2026-04-22
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

LEVERAGED_RE = re.compile(r"[35][LS]_USDT$")


def detect_leveraged(symbol: str) -> bool:
    """Return True if `symbol` is a Gate.io leveraged token (e.g., BTC3L_USDT)."""
    return bool(LEVERAGED_RE.search(symbol))


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file (streaming)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def append_manifest_row(
    *,
    manifest_path: Path,
    date: str,
    candles_rows: int,
    candles_sha: str,
    universe_rows: int,
    universe_sha: str,
    notes: str,
) -> None:
    """Append a row to MANIFEST.md. Raise if the date is already listed."""
    if "|" in notes or "\n" in notes:
        raise ValueError(
            f"notes must not contain '|' or newlines (got: {notes!r}); "
            "those characters break the markdown table"
        )
    text = manifest_path.read_text()
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) >= 1 and parts[0] == date:
            raise ValueError(f"date {date} is already in MANIFEST")
    row = (
        f"| {date} | {candles_rows} | {candles_sha} | "
        f"{universe_rows} | {universe_sha} | {notes} |\n"
    )
    # Ensure file ends with newline before we append.
    if not text.endswith("\n"):
        text += "\n"
    manifest_path.write_text(text + row)


def _load_candles(engine, as_of: str):
    """Load all 15m candles with timestamp < {as_of} 00:00 UTC."""
    import pandas as pd
    from sqlalchemy import text as sql_text

    q = sql_text(
        """
        SELECT asset, timestamp, open, high, low, close, volume, quote_volume
        FROM asset_prices_15m
        WHERE timestamp < (:as_of)::timestamptz
        ORDER BY asset, timestamp
        """
    )
    return pd.read_sql_query(q, engine, params={"as_of": f"{as_of} 00:00:00+00:00"})


def _fetch_gateio_universe() -> list[dict]:
    """Fetch Gate.io USDT futures contracts + tickers via public REST API.

    Returns a list of contract dicts enriched with ticker fields merged in.
    The /contracts endpoint exposes is_delisting but has no 24h volume.
    The /tickers endpoint has volume_24h_quote (USDT notional) — we join on name.
    """
    import json
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    def _get(url: str) -> list[dict]:
        req = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
        try:
            resp = urlopen(req, timeout=30)
            return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            raise RuntimeError(f"failed to fetch {url}: {e}") from e

    contracts = _get("https://api.gateio.ws/api/v4/futures/usdt/contracts")
    tickers = _get("https://api.gateio.ws/api/v4/futures/usdt/tickers")

    # Build ticker lookup keyed by contract name
    ticker_by_name: dict[str, dict] = {t["contract"]: t for t in tickers if "contract" in t}

    # Merge ticker fields into each contract dict
    for c in contracts:
        name = c.get("name", "")
        t = ticker_by_name.get(name, {})
        # volume_24h_quote = USDT notional 24h volume (verified 2026-04-22 via probe)
        c["volume_24h_quote"] = t.get("volume_24h_quote", "0")

    return contracts


def _build_universe_df(candles_df, contracts: list[dict]):
    """Build universe snapshot DataFrame from candles + Gate.io contract list."""
    import pandas as pd

    # earliest candle per asset -> listed_since approximation
    listed = (candles_df.groupby("asset")["timestamp"].min()
              .reset_index().rename(columns={"timestamp": "listed_since"}))

    # Gate.io field notes (verified 2026-04-22 via probe):
    #   /contracts: name, last_price, in_delisting (bool), funding_rate, mark_price, ...
    #               NOTE: NO 24h volume on /contracts. trade_size_24h does not exist there.
    #   /tickers:   volume_24h_quote = USDT notional 24h volume
    #               (merged by _fetch_gateio_universe; volume_24h_settle == volume_24h_quote)
    rows = []
    for c in contracts:
        # Gate.io contract names are e.g. "BTC_USDT"; candles table uses "BTC"
        symbol = c.get("name", "")
        asset = symbol.replace("_USDT", "")
        listed_since = None
        match = listed[listed["asset"] == asset]
        if len(match) > 0:
            listed_since = match["listed_since"].iloc[0]
        rows.append({
            "symbol": symbol,
            "asset": asset,
            "listed_since": listed_since,
            "is_leveraged": detect_leveraged(symbol),
            "quote_volume_24h": _safe_float(c.get("volume_24h_quote")),
            "last_price": _safe_float(c.get("last_price")),
            "in_delisting": bool(c.get("in_delisting", False)),
        })
    return pd.DataFrame(rows)


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    import os

    from sqlalchemy import create_engine

    # Default snapshots dir is anchored to the repo root (parents[3] = scripts/ → python/
    # → services/ → repo). This makes `python scripts/export_snapshot.py` work from
    # services/python/ without --snapshots-dir, and writes to the repo-root data/snapshots/.
    _repo_root = Path(__file__).resolve().parents[3]
    _default_snapshots_dir = str(_repo_root / "data" / "snapshots")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Snapshot as-of date (YYYY-MM-DD, UTC)")
    parser.add_argument("--notes", default="", help="Free-form note for MANIFEST.md")
    parser.add_argument(
        "--snapshots-dir", default=_default_snapshots_dir,
        help="Directory for Parquet + MANIFEST.md (default: <repo-root>/data/snapshots)",
    )
    args = parser.parse_args()

    snapshots_dir = Path(args.snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    manifest = snapshots_dir / "MANIFEST.md"
    if not manifest.exists():
        raise FileNotFoundError(
            f"MANIFEST.md missing at {manifest}. It should have been scaffolded already."
        )

    candles_path = snapshots_dir / f"candles_15m_{args.date}.parquet"
    universe_path = snapshots_dir / f"universe_{args.date}.parquet"
    if candles_path.exists() or universe_path.exists():
        raise FileExistsError(
            f"snapshot files for {args.date} already exist — delete them manually to re-export"
        )

    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market"
    )
    engine = create_engine(db_url)
    try:
        print(f"[export_snapshot] loading candles with timestamp < {args.date} 00:00 UTC ...")
        candles_df = _load_candles(engine, args.date)
        print(f"[export_snapshot]   {len(candles_df):,} candles")
    finally:
        engine.dispose()

    print("[export_snapshot] fetching Gate.io futures universe ...")
    contracts = _fetch_gateio_universe()
    universe_df = _build_universe_df(candles_df, contracts)
    print(
        f"[export_snapshot]   {len(universe_df):,} pairs "
        f"({int(universe_df['is_leveraged'].sum()):,} leveraged)"
    )

    candles_df.to_parquet(candles_path, index=False)
    universe_df.to_parquet(universe_path, index=False)
    print(f"[export_snapshot] wrote {candles_path}")
    print(f"[export_snapshot] wrote {universe_path}")

    c_sha = sha256_file(candles_path)
    u_sha = sha256_file(universe_path)
    append_manifest_row(
        manifest_path=manifest,
        date=args.date,
        candles_rows=len(candles_df),
        candles_sha=c_sha,
        universe_rows=len(universe_df),
        universe_sha=u_sha,
        notes=args.notes or "export_snapshot",
    )
    print(f"[export_snapshot] appended manifest row for {args.date}")


if __name__ == "__main__":
    main()
