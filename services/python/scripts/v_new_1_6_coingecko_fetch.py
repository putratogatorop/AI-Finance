"""v_new_1.6 — Fetch CoinGecko category membership.

Pulls per-category coin lists from the CoinGecko public API for the narratives
we care about (Memes, RWA, AI, DePIN, GameFi, DeFi, L1, L2, Solana eco, ETH eco).
Maps each CG symbol to our `<TICKER>USDT` perp-futures convention by inspecting
the v_new_1_v2 universe.

Output:
    services/python/data/v_new_1_v2/coingecko_categories.json
    {
        "memes":      ["DOGEUSDT", "PEPEUSDT", ...],
        "rwa":        ["ONDOUSDT", "PENDLEUSDT", ...],
        ...
    }

LOOK-AHEAD CAVEAT (acknowledged in V_NEW_1_6_PLAN-2026-05-04.md):
    CoinGecko categories don't expose history cleanly. We use CURRENT membership
    applied retrospectively. For long-standing categories (Memes, DeFi, L1) this
    is mostly clean; for newer tags (RWA c. late 2023, AI Tokens c. 2024) there
    is some retrospective tagging bias. The verdict step flags suspicious lift.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Optional

try:
    import requests
except ImportError:
    print("ERROR: requests package required. Install with: pip install requests")
    sys.exit(1)

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                    str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))

CG_BASE = "https://api.coingecko.com/api/v3"

# Map of CoinGecko category IDs → our human-readable label. Adjust if CG renames.
CATEGORIES = [
    ("meme-token",                                                    "memes"),
    ("real-world-assets-rwa",                                         "rwa"),
    ("artificial-intelligence",                                       "ai_tokens"),
    ("depin",                                                         "depin"),
    ("gaming",                                                        "gamefi"),
    ("decentralized-finance-defi",                                    "defi"),
    ("smart-contract-platform",                                       "layer1"),
    ("layer-2",                                                       "layer2"),
    ("solana-ecosystem",                                              "solana_eco"),
    ("ethereum-ecosystem",                                            "eth_eco"),
]

PER_PAGE = 250
MAX_PAGES = 4         # 4 × 250 = 1000 coins per category — well past our top-200 universe
SLEEP_BETWEEN_PAGES = 3.0
RETRY_SLEEP = 20.0
MAX_RETRIES = 4


def _ts() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


def _fetch_category(cat_id: str) -> list[str]:
    """Returns a flat list of UPPERCASE CG symbols (e.g. 'BTC', 'PEPE') for the
    category's top market-cap members."""
    out: list[str] = []
    for page in range(1, MAX_PAGES + 1):
        url = f"{CG_BASE}/coins/markets"
        params = {
            "vs_currency": "usd",
            "category": cat_id,
            "order": "market_cap_desc",
            "per_page": PER_PAGE,
            "page": page,
        }
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(url, params=params, timeout=30,
                                 headers={"User-Agent": "v_new_1_6/1.0"})
            except Exception as e:
                print(f"  page {page} attempt {attempt+1}: network error {e}; retrying")
                time.sleep(RETRY_SLEEP)
                continue
            if r.status_code == 200:
                data = r.json()
                if not data:
                    return out
                out.extend([c.get("symbol", "").upper() for c in data if c.get("symbol")])
                if len(data) < PER_PAGE:
                    return out
                break
            elif r.status_code in (429, 503):
                print(f"  page {page} attempt {attempt+1}: rate limited (status {r.status_code}); sleeping")
                time.sleep(RETRY_SLEEP)
            elif r.status_code == 404:
                print(f"  category {cat_id} returned 404 — does the slug exist?")
                return out
            else:
                print(f"  page {page} attempt {attempt+1}: status {r.status_code}; sleeping")
                time.sleep(RETRY_SLEEP)
        else:
            print(f"  page {page}: exhausted retries; returning what we have")
            return out
        time.sleep(SLEEP_BETWEEN_PAGES)
    return out


def _load_universe() -> set[str]:
    """Returns the set of <TICKER>USDT symbols in our top-200 universe so we can
    map CG symbols to our naming."""
    membership_path = DATA / "universe_top100_membership.parquet"  # legacy filename
    if not membership_path.exists():
        print(f"  WARNING: {membership_path} missing — falling back to inferring from labels")
        labels = pd.read_parquet(DATA / "labels_long.parquet", columns=["symbol"])
        return set(labels["symbol"].unique())
    m = pd.read_parquet(membership_path, columns=["symbol"])
    return set(m["symbol"].unique())


# Known CG-symbol aliases (rebrands or naming differences vs Gate.io perps).
CG_TO_PERP_ALIAS = {
    "RENDER": "RNDR",     # Render rebrand 2024; Gate retained RNDR ticker
    "BEAM": "BEAMX",      # Beam → BEAMX on some exchanges
    "PIVX": "PIVX",       # identity (placeholder)
}


def _cg_symbol_to_perp(cg_symbol: str, universe: set[str]) -> Optional[str]:
    """Map CoinGecko's symbol (e.g. 'PEPE') to our perp ('PEPEUSDT'). Try a few
    common variants. Returns None if no match in universe."""
    aliased = CG_TO_PERP_ALIAS.get(cg_symbol, cg_symbol)
    candidates = [
        f"{aliased}USDT",
        f"1000{aliased}USDT",
        f"10000{aliased}USDT",
    ]
    for c in candidates:
        if c in universe:
            return c
    return None


def main() -> None:
    print(f"{_ts()} v_new_1.6 CoinGecko category fetch — start")
    print(f"  DATA: {DATA}")
    universe = _load_universe()
    print(f"  universe size: {len(universe)} (e.g. {sorted(list(universe))[:5]})")

    out: dict[str, list[str]] = {}
    diag: dict[str, dict] = {}

    for cat_id, label in CATEGORIES:
        print(f"\n{_ts()} Fetching '{label}' (CG id={cat_id})...")
        cg_syms = _fetch_category(cat_id)
        print(f"  CG returned {len(cg_syms)} symbols")
        mapped: list[str] = []
        unmapped_sample: list[str] = []
        for s in cg_syms:
            perp = _cg_symbol_to_perp(s, universe)
            if perp is not None:
                mapped.append(perp)
            elif len(unmapped_sample) < 5:
                unmapped_sample.append(s)
        # Dedupe (keep order)
        mapped = list(dict.fromkeys(mapped))
        out[label] = mapped
        diag[label] = {
            "cg_id": cat_id,
            "cg_count": len(cg_syms),
            "mapped_count": len(mapped),
            "unmapped_sample": unmapped_sample,
        }
        print(f"  → {len(mapped)} mapped to USDT-perps in our universe "
              f"(e.g. {mapped[:5]})")

    out_path = DATA / "coingecko_categories.json"
    diag_path = DATA / "coingecko_categories_diag.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\n{_ts()} DONE")
    print(f"  wrote {out_path}")
    print(f"  wrote {diag_path}")
    print(f"\nSummary:")
    for label, syms in out.items():
        print(f"  {label:14s}: {len(syms):4d} symbols")


if __name__ == "__main__":
    main()
