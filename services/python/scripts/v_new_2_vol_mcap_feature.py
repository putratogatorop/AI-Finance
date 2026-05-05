"""v_new_2 — vol/mcap "trending coin" feature.

Builds vol_to_mcap_24h feature (24h USD volume / market cap) for our universe.

Steps:
  1. Fetch top-500 coins from CoinGecko /coins/markets (1-2 API calls).
  2. Build map: symbol → circulating_supply.
  3. Load raw 15m candles from snapshot, resample to 4h, compute rolling 24h
     quote_volume per (symbol, timestamp).
  4. mcap_proxy[t] = current_supply × close[t]   (uses CURRENT supply — proxy,
     supply changes over time but this is good first cut).
  5. vol_to_mcap_24h = vol_24h / mcap_proxy
  6. Save as feature parquet to merge into BGM training data.

Output:
  data/v_new_1_v2/vol_mcap_feature.parquet
  Columns: symbol, timestamp, mcap_proxy_usd, vol_24h_usd, vol_to_mcap_24h
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    print("ERROR: requests not installed")
    sys.exit(1)

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                        str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))
SNAPSHOTS = ROOT / "data" / "snapshots"


def _ts(): return f"[{time.strftime('%H:%M:%S')}]"


def _fetch_cg_supply() -> dict[str, float]:
    """Fetch top-500 coins from CG /coins/markets, return symbol→supply map.
    Symbols are normalized to UPPER + 'USDT' suffix to match our perp universe."""
    print(f"{_ts()} Fetching CoinGecko market data (top 500 by mcap)...")
    out: dict[str, float] = {}
    base = "https://api.coingecko.com/api/v3/coins/markets"
    for page in [1, 2]:   # 250 per page × 2 = top 500
        for attempt in range(4):
            try:
                r = requests.get(base, params={
                    "vs_currency": "usd", "order": "market_cap_desc",
                    "per_page": 250, "page": page,
                }, timeout=30, headers={"User-Agent": "v_new_2/1.0"})
            except Exception as e:
                print(f"  page {page} attempt {attempt+1}: {e}; retrying")
                time.sleep(15); continue
            if r.status_code == 200:
                data = r.json()
                for c in data:
                    sym = c.get("symbol", "").upper()
                    supply = c.get("circulating_supply") or 0
                    if sym and supply > 0:
                        out[sym + "USDT"] = float(supply)
                print(f"  page {page}: got {len(data)} coins, total mapped: {len(out)}")
                time.sleep(3)
                break
            else:
                print(f"  page {page} status={r.status_code}; sleeping")
                time.sleep(20)
        else:
            print(f"  page {page} exhausted retries")
    return out


def _load_universe_symbols() -> set[str]:
    """Universe symbols from our top-200 membership."""
    m = pd.read_parquet(DATA_DIR / "universe_top100_membership.parquet",
                          columns=["symbol"])
    return set(m["symbol"].unique())


def _load_15m_close_volume() -> pd.DataFrame:
    """Load 15m close + quote_volume from snapshots, filter to universe symbols,
    resample to 4h."""
    print(f"{_ts()} Loading 15m candles from snapshots...")
    s1 = pd.read_parquet(SNAPSHOTS / "candles_15m_2026-04-01.parquet",
                          columns=["asset", "timestamp", "close", "quote_volume"])
    s2 = pd.read_parquet(SNAPSHOTS / "candles_15m_2026-04-27.parquet",
                          columns=["asset", "timestamp", "close", "quote_volume"])
    df = pd.concat([s1, s2], ignore_index=True)
    df = df.rename(columns={"asset": "symbol"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.drop_duplicates(subset=["symbol", "timestamp"]).sort_values(["symbol","timestamp"])
    print(f"  total 15m bars: {len(df):,}")

    # Resample to 4h: take last close, sum quote_volume
    print(f"{_ts()} Resampling 15m → 4h...")
    df = df.set_index("timestamp")
    out_parts = []
    for sym, g in df.groupby("symbol"):
        agg = g.resample("4h").agg(close=("close","last"),
                                     quote_volume=("quote_volume","sum"))
        agg["symbol"] = sym
        out_parts.append(agg.reset_index())
    out = pd.concat(out_parts, ignore_index=True)
    out = out.dropna(subset=["close"])
    print(f"  4h bars: {len(out):,}  symbols: {out['symbol'].nunique()}")
    return out


def main() -> None:
    t0 = time.time()
    print(f"{_ts()} v_new_2 vol/mcap feature build")

    universe = _load_universe_symbols()
    print(f"  universe size: {len(universe)}")

    supply_map = _fetch_cg_supply()
    in_universe = {k: v for k, v in supply_map.items() if k in universe}
    print(f"  matched to universe: {len(in_universe)} / {len(universe)}")
    # Save mapping for future use
    with open(DATA_DIR / "cg_supply_map.json", "w") as f:
        json.dump(in_universe, f, indent=2)

    df = _load_15m_close_volume()
    df = df[df["symbol"].isin(universe)].reset_index(drop=True)
    print(f"  after universe filter: {len(df):,}")

    # 24h rolling quote volume (6 4h-bars)
    print(f"{_ts()} Computing 24h rolling quote_volume + mcap_proxy...")
    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    df["vol_24h_usd"] = df.groupby("symbol")["quote_volume"].transform(
        lambda s: s.rolling(6, min_periods=1).sum()
    )
    df["circulating_supply"] = df["symbol"].map(in_universe)
    df["mcap_proxy_usd"] = df["circulating_supply"] * df["close"]
    df["vol_to_mcap_24h"] = df["vol_24h_usd"] / df["mcap_proxy_usd"]
    df.loc[~np.isfinite(df["vol_to_mcap_24h"]), "vol_to_mcap_24h"] = np.nan

    # Stats
    print(f"\n{_ts()} vol/mcap distribution:")
    v = df["vol_to_mcap_24h"].dropna()
    for q in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{q}: {np.percentile(v, q):.4f}")
    print(f"  bars with vol/mcap > 1.0:  {(v > 1.0).sum():,} ({100*(v>1.0).mean():.2f}%)")
    print(f"  bars with vol/mcap > 0.5:  {(v > 0.5).sum():,} ({100*(v>0.5).mean():.2f}%)")
    print(f"  bars with vol/mcap > 0.2:  {(v > 0.2).sum():,} ({100*(v>0.2).mean():.2f}%)")
    print(f"  bars with vol/mcap > 0.1:  {(v > 0.1).sum():,} ({100*(v>0.1).mean():.2f}%)")

    # Per-tier average
    mem = pd.read_parquet(DATA_DIR / "universe_top100_membership.parquet")
    avg_rank = mem.groupby("symbol")["rank"].mean().reset_index().rename(columns={"rank":"avg_rank"})
    def tier(r):
        if r<=25: return "top25"
        if r<=50: return "26-50"
        if r<=100: return "51-100"
        return "101-200"
    avg_rank["tier"] = avg_rank["avg_rank"].apply(tier)
    df = df.merge(avg_rank[["symbol","tier"]], on="symbol", how="left")

    print(f"\n  median vol/mcap by tier:")
    for t in ["top25","26-50","51-100","101-200"]:
        sub = df[df["tier"]==t]["vol_to_mcap_24h"].dropna()
        if len(sub)>0:
            print(f"    {t:8s}  median {sub.median():.4f}  p90 {sub.quantile(0.90):.4f}  "
                  f"p99 {sub.quantile(0.99):.4f}  >0.5: {(sub>0.5).sum():,}  >1.0: {(sub>1.0).sum():,}")

    # Save
    out = df[["symbol","timestamp","circulating_supply","close","quote_volume",
               "vol_24h_usd","mcap_proxy_usd","vol_to_mcap_24h"]]
    out_path = DATA_DIR / "vol_mcap_feature.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n  wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"{_ts()} DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
