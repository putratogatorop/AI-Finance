# Backtest Reproducibility Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the infrastructure that makes every new backtest byte-reproducible across agents and sessions, so "every AI gives different numbers" stops happening.

**Architecture:** Five new artifacts — a postgres→Parquet snapshot exporter, a reusable backtest template with canonical helpers (`load_snapshot`, `compute_metrics`, `write_results`), a committed MANIFEST, a protocol doc, and a CLAUDE.md pointer. A reference SMA backtest acts as a permanent self-reproducibility canary. No framework library, no superpowers skill, no CI check, no retrofit of past scripts.

**Tech Stack:** Python 3.12, pandas, numpy, pyarrow (Parquet), psycopg2/SQLAlchemy (postgres read), pytest (tests), stdlib `hashlib`/`json`/`subprocess` (sha256, metadata, git SHA).

**Spec:** `docs/superpowers/specs/2026-04-22-backtest-reproducibility-protocol-design.md`

**Working directory:** All Python commands are run from `services/python/` unless noted. All tests are `pytest tests/<name>.py -v` from that directory.

---

## Task 0: Prepare branch

**Files:** None (git metadata only)

**Context:** Current branch `feat/rl-gym-environment` is unrelated to this work (it's the RL gym branch, but the spec-and-impl we're doing is about backtest reproducibility). The spec is already committed to this branch as `b5640def`. Cleanest path: cherry-pick the spec commit onto a new branch, then continue impl there.

- [ ] **Step 1: Capture current spec commit SHA**

Run: `git log --oneline -1 docs/superpowers/specs/2026-04-22-backtest-reproducibility-protocol-design.md`

Expected: one line like `b5640def docs(spec): backtest reproducibility protocol design`

Record the SHA as `SPEC_SHA`.

- [ ] **Step 2: Create the work branch off master**

```bash
git checkout master
git pull origin master
git checkout -b feat/backtest-reproducibility
git cherry-pick <SPEC_SHA>
```

Expected: branch created, spec file appears on new branch.

- [ ] **Step 3: Verify spec file is present on new branch**

Run: `git log --oneline -1 docs/superpowers/specs/2026-04-22-backtest-reproducibility-protocol-design.md`

Expected: same commit message, on the new branch.

- [ ] **Step 4: Remove spec commit from the old branch (optional, only if you want a clean old branch)**

Skip this step if you don't care about the old branch. Otherwise:

```bash
git checkout feat/rl-gym-environment
git reset --hard HEAD~1   # DANGER: only if no other commits rely on SPEC_SHA
git checkout feat/backtest-reproducibility
```

If the spec commit is not at `HEAD` on the old branch, don't do this — leave the spec on both branches.

---

## Task 1: Scaffolding — data/snapshots/ dir, MANIFEST.md, .gitignore

**Files:**
- Create: `data/snapshots/MANIFEST.md`
- Modify: `.gitignore`

- [ ] **Step 1: Create the snapshots directory**

```bash
mkdir -p data/snapshots
```

- [ ] **Step 2: Create `data/snapshots/MANIFEST.md` with the header only**

Create file with exactly this content:

```markdown
# Backtest Data Snapshots

This file is the source of truth for which Parquet snapshots exist and the canonical sha256 of each. `export_snapshot.py` appends one row per export. Do not hand-edit.

See `docs/backtest-protocol.md` for usage.

| Date       | Candles rows | Candles sha256                                                   | Universe rows | Universe sha256                                                  | Notes |
|------------|--------------|------------------------------------------------------------------|---------------|------------------------------------------------------------------|-------|
```

- [ ] **Step 3: Edit `.gitignore` to exclude snapshot Parquet files**

Add this block at the end of `.gitignore`:

```
# Backtest data snapshots (large Parquet files — see data/snapshots/MANIFEST.md)
data/snapshots/*.parquet
```

Rationale: existing `.gitignore` has `data/*.parquet` but that only matches direct children of `data/`, not `data/snapshots/` subdir. Explicit rule is required.

- [ ] **Step 4: Verify gitignore works**

```bash
touch data/snapshots/test.parquet
git status --porcelain data/snapshots/
rm data/snapshots/test.parquet
```

Expected output of `git status`: only `data/snapshots/MANIFEST.md` listed (no `test.parquet` — it's ignored).

- [ ] **Step 5: Commit**

```bash
git add data/snapshots/MANIFEST.md .gitignore
git commit -m "feat(backtest): scaffold snapshot directory and gitignore"
```

---

## Task 2: `compute_metrics` — canonical metric calculation (TDD)

**Files:**
- Create: `services/python/scripts/backtest_template.py`
- Create: `services/python/tests/test_backtest_template.py`

**Context:** `compute_metrics(trades_df)` is the canonical Profit Factor / Win Rate / avg PnL calculation. Every new backtest uses this exact function — this is the math that stops different agents from computing "PF" different ways. It lives inline in the template (per spec: no library).

- [ ] **Step 1: Write the failing test**

Create `services/python/tests/test_backtest_template.py`:

```python
"""Tests for backtest_template.py helpers (canonical metric calc, etc)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Import helpers from the template script. Template is a runnable script,
# but its top-level helpers are also importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_template import compute_metrics  # noqa: E402


def _trades(rows):
    """Build a trades DataFrame from rows of (direction, pnl_pct, bars_held)."""
    return pd.DataFrame(
        rows, columns=["direction", "pnl_pct", "bars_held"]
    )


def test_compute_metrics_empty_returns_zeroed_shape():
    df = _trades([])
    m = compute_metrics(df)
    assert m["trades"] == 0
    assert m["wr"] == 0.0
    assert m["pf"] == 0.0
    assert m["avg_pnl_pct"] == 0.0
    assert m["total_pnl_pct"] == 0.0
    assert m["median_hold_bars"] == 0.0
    assert m["by_direction"]["long"]["trades"] == 0
    assert m["by_direction"]["short"]["trades"] == 0


def test_compute_metrics_two_winners_one_loser():
    # 2 wins of +5%, 1 loss of -5%. All longs.
    df = _trades([
        ("long", 0.05, 10),
        ("long", 0.05, 20),
        ("long", -0.05, 30),
    ])
    m = compute_metrics(df)
    assert m["trades"] == 3
    assert m["wr"] == pytest.approx(2 / 3)
    # PF = gross_profit / gross_loss = (0.05 + 0.05) / 0.05 = 2.0
    assert m["pf"] == pytest.approx(2.0)
    assert m["avg_pnl_pct"] == pytest.approx((0.05 + 0.05 - 0.05) / 3)
    assert m["total_pnl_pct"] == pytest.approx(0.05)
    assert m["median_hold_bars"] == 20.0
    assert m["by_direction"]["long"]["trades"] == 3
    assert m["by_direction"]["short"]["trades"] == 0


def test_compute_metrics_pf_infinite_when_no_losses():
    df = _trades([("long", 0.05, 5), ("long", 0.03, 5)])
    m = compute_metrics(df)
    # Convention: no losses → pf = float('inf')
    assert m["pf"] == float("inf")


def test_compute_metrics_direction_split():
    df = _trades([
        ("long", 0.05, 10),
        ("long", -0.02, 12),
        ("short", 0.04, 8),
        ("short", 0.06, 6),
    ])
    m = compute_metrics(df)
    # Long: 1 win of +5%, 1 loss of -2% → PF = 2.5, WR = 0.5
    assert m["by_direction"]["long"]["trades"] == 2
    assert m["by_direction"]["long"]["wr"] == pytest.approx(0.5)
    assert m["by_direction"]["long"]["pf"] == pytest.approx(0.05 / 0.02)
    # Short: 2 wins, no losses → PF = inf
    assert m["by_direction"]["short"]["trades"] == 2
    assert m["by_direction"]["short"]["wr"] == pytest.approx(1.0)
    assert m["by_direction"]["short"]["pf"] == float("inf")
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `cd services/python && pytest tests/test_backtest_template.py -v`

Expected: `ModuleNotFoundError: No module named 'backtest_template'` (the file doesn't exist yet).

- [ ] **Step 3: Create minimal `backtest_template.py` with `compute_metrics` only**

Create `services/python/scripts/backtest_template.py`:

```python
"""Backtest template — copy this file to start a new backtest.

Every new backtest MUST:
  - Start from this file: cp backtest_template.py backtest_<name>.py
  - Edit SNAPSHOT_DATE to a date listed in data/snapshots/MANIFEST.md
  - Fill in the strategy logic block (marked with TODO)
  - Leave load_snapshot, compute_metrics, write_results UNTOUCHED

See docs/backtest-protocol.md for the full rules.
"""
from __future__ import annotations

import pandas as pd

# --- CONSTANTS (edit SNAPSHOT_DATE when copying) ---
SNAPSHOT_DATE = "YYYY-MM-DD"  # must match a row in data/snapshots/MANIFEST.md
RANDOM_SEED = 42

# --- PARAMS (edit for your strategy) ---
# PARAM_EXAMPLE = 0.05

# --- CANONICAL HELPERS (do not edit) ---


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    """Canonical PF/WR/avg-PnL calculation.

    trades_df columns:
      - direction: 'long' or 'short'
      - pnl_pct: fractional PnL per trade (0.05 = +5%)
      - bars_held: int

    Returns dict with the keys the protocol requires. The same function is
    used by every backtest so numbers are directly comparable.
    """
    def _stats(df: pd.DataFrame) -> dict:
        n = len(df)
        if n == 0:
            return {"trades": 0, "wr": 0.0, "pf": 0.0, "avg_pnl_pct": 0.0,
                    "total_pnl_pct": 0.0, "median_hold_bars": 0.0}
        wins = df[df["pnl_pct"] > 0]["pnl_pct"].sum()
        losses = -df[df["pnl_pct"] < 0]["pnl_pct"].sum()  # positive number
        pf = float("inf") if losses == 0 else float(wins / losses)
        return {
            "trades": int(n),
            "wr": float((df["pnl_pct"] > 0).sum() / n),
            "pf": pf,
            "avg_pnl_pct": float(df["pnl_pct"].mean()),
            "total_pnl_pct": float(df["pnl_pct"].sum()),
            "median_hold_bars": float(df["bars_held"].median()),
        }

    overall = _stats(trades_df)
    overall["by_direction"] = {
        "long": _stats(trades_df[trades_df["direction"] == "long"]),
        "short": _stats(trades_df[trades_df["direction"] == "short"]),
    }
    return overall
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `cd services/python && pytest tests/test_backtest_template.py::test_compute_metrics_empty_returns_zeroed_shape tests/test_backtest_template.py::test_compute_metrics_two_winners_one_loser tests/test_backtest_template.py::test_compute_metrics_pf_infinite_when_no_losses tests/test_backtest_template.py::test_compute_metrics_direction_split -v`

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add services/python/scripts/backtest_template.py services/python/tests/test_backtest_template.py
git commit -m "feat(backtest): add compute_metrics canonical helper with tests"
```

---

## Task 3: `load_snapshot` — Parquet loader with sha256 verification (TDD)

**Files:**
- Modify: `services/python/scripts/backtest_template.py`
- Modify: `services/python/tests/test_backtest_template.py`

- [ ] **Step 1: Write the failing tests**

Append to `services/python/tests/test_backtest_template.py`:

```python


# --- load_snapshot tests ---

import hashlib
import tempfile
from pathlib import Path

from backtest_template import load_snapshot  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_snapshot(snapshots_dir: Path, date: str,
                  candles_df: pd.DataFrame, universe_df: pd.DataFrame) -> tuple[str, str]:
    """Write a fake snapshot and return (candles_sha, universe_sha)."""
    candles_path = snapshots_dir / f"candles_15m_{date}.parquet"
    universe_path = snapshots_dir / f"universe_{date}.parquet"
    candles_df.to_parquet(candles_path, index=False)
    universe_df.to_parquet(universe_path, index=False)
    return _sha256(candles_path), _sha256(universe_path)


def _write_manifest(manifest_path: Path, date: str,
                    candles_rows: int, candles_sha: str,
                    universe_rows: int, universe_sha: str, notes: str = "test") -> None:
    header = (
        "# Backtest Data Snapshots\n\n"
        "| Date       | Candles rows | Candles sha256"
        " | Universe rows | Universe sha256 | Notes |\n"
        "|------------|--------------|-----|---------------|-----|-------|\n"
    )
    row = f"| {date} | {candles_rows} | {candles_sha} | {universe_rows} | {universe_sha} | {notes} |\n"
    manifest_path.write_text(header + row)


def test_load_snapshot_returns_candles_and_universe(tmp_path):
    candles = pd.DataFrame({"asset": ["BTC"], "timestamp": [1], "open": [100.0],
                            "high": [101.0], "low": [99.0], "close": [100.5],
                            "volume": [10.0], "quote_volume": [1000.0]})
    universe = pd.DataFrame({"symbol": ["BTC_USDT"], "listed_since": ["2020-01-01"],
                             "is_leveraged": [False], "quote_volume_24h": [1e9]})
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    c_sha, u_sha = _make_snapshot(snaps, "2026-04-22", candles, universe)
    _write_manifest(snaps / "MANIFEST.md", "2026-04-22", 1, c_sha, 1, u_sha)

    c_df, u_df = load_snapshot("2026-04-22", snapshots_dir=snaps)

    assert len(c_df) == 1 and c_df["asset"].iloc[0] == "BTC"
    assert len(u_df) == 1 and u_df["symbol"].iloc[0] == "BTC_USDT"


def test_load_snapshot_raises_when_file_missing(tmp_path):
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    (snaps / "MANIFEST.md").write_text("# Backtest Data Snapshots\n")
    with pytest.raises(FileNotFoundError, match="2026-04-22"):
        load_snapshot("2026-04-22", snapshots_dir=snaps)


def test_load_snapshot_raises_on_sha256_mismatch(tmp_path):
    candles = pd.DataFrame({"asset": ["BTC"], "timestamp": [1], "open": [100.0],
                            "high": [101.0], "low": [99.0], "close": [100.5],
                            "volume": [10.0], "quote_volume": [1000.0]})
    universe = pd.DataFrame({"symbol": ["BTC_USDT"], "listed_since": ["2020-01-01"],
                             "is_leveraged": [False], "quote_volume_24h": [1e9]})
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    c_sha, u_sha = _make_snapshot(snaps, "2026-04-22", candles, universe)
    # Manifest claims a different sha256 than the file actually has.
    _write_manifest(snaps / "MANIFEST.md", "2026-04-22", 1, "0" * 64, 1, u_sha)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_snapshot("2026-04-22", snapshots_dir=snaps)


def test_load_snapshot_raises_when_date_not_in_manifest(tmp_path):
    candles = pd.DataFrame({"asset": ["BTC"], "timestamp": [1], "open": [100.0],
                            "high": [101.0], "low": [99.0], "close": [100.5],
                            "volume": [10.0], "quote_volume": [1000.0]})
    universe = pd.DataFrame({"symbol": ["BTC_USDT"], "listed_since": ["2020-01-01"],
                             "is_leveraged": [False], "quote_volume_24h": [1e9]})
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    _make_snapshot(snaps, "2026-04-22", candles, universe)
    # Manifest exists but does NOT contain this date.
    (snaps / "MANIFEST.md").write_text("# Backtest Data Snapshots\n")

    with pytest.raises(ValueError, match="not in MANIFEST"):
        load_snapshot("2026-04-22", snapshots_dir=snaps)
```

- [ ] **Step 2: Run tests — expect ImportError on `load_snapshot`**

Run: `cd services/python && pytest tests/test_backtest_template.py -v -k load_snapshot`

Expected: `ImportError: cannot import name 'load_snapshot' from 'backtest_template'`.

- [ ] **Step 3: Implement `load_snapshot` in template**

Append to `services/python/scripts/backtest_template.py` before the "CANONICAL HELPERS" section divider (or immediately after it, before `compute_metrics`). Add the imports at the top of the file with the existing `import pandas as pd` line. Final file should have this content added:

At top of file, extend the imports block:

```python
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
```

Add this function before `compute_metrics`:

```python
def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file (streaming, safe for large Parquet)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_manifest_row(manifest_text: str, date: str) -> tuple[str, str]:
    """Return (candles_sha, universe_sha) for `date` from MANIFEST.md.

    Raises ValueError if the date is not found.
    """
    for line in manifest_text.splitlines():
        # Manifest rows start with `| YYYY-MM-DD |`
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 5 or parts[0] != date:
            continue
        # Columns: Date | Candles rows | Candles sha256 | Universe rows | Universe sha256 | Notes
        return parts[2], parts[4]
    raise ValueError(f"date {date} not in MANIFEST.md")


def load_snapshot(
    date: str,
    snapshots_dir: Path | str = Path("data/snapshots"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (candles, universe) parquet files for a snapshot date.

    Verifies both files' sha256 against MANIFEST.md. Raises:
      - FileNotFoundError if either Parquet file is missing.
      - ValueError if the date is not listed in MANIFEST.md.
      - ValueError if either file's sha256 does not match the manifest.
    """
    snapshots_dir = Path(snapshots_dir)
    manifest = snapshots_dir / "MANIFEST.md"
    if not manifest.exists():
        raise FileNotFoundError(f"missing MANIFEST.md at {manifest}")
    candles_path = snapshots_dir / f"candles_15m_{date}.parquet"
    universe_path = snapshots_dir / f"universe_{date}.parquet"
    if not candles_path.exists():
        raise FileNotFoundError(
            f"snapshot candles file missing for date {date}. "
            f"Run export_snapshot.py --date {date} or pick a date from MANIFEST.md."
        )
    if not universe_path.exists():
        raise FileNotFoundError(
            f"snapshot universe file missing for date {date}. "
            f"Run export_snapshot.py --date {date} or pick a date from MANIFEST.md."
        )
    expected_candles_sha, expected_universe_sha = _parse_manifest_row(manifest.read_text(), date)
    actual_candles_sha = _sha256_file(candles_path)
    actual_universe_sha = _sha256_file(universe_path)
    if actual_candles_sha != expected_candles_sha:
        raise ValueError(
            f"sha256 mismatch for {candles_path}: expected {expected_candles_sha}, "
            f"got {actual_candles_sha}"
        )
    if actual_universe_sha != expected_universe_sha:
        raise ValueError(
            f"sha256 mismatch for {universe_path}: expected {expected_universe_sha}, "
            f"got {actual_universe_sha}"
        )
    return pd.read_parquet(candles_path), pd.read_parquet(universe_path)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `cd services/python && pytest tests/test_backtest_template.py -v -k load_snapshot`

Expected: 4 tests pass (the new ones for `load_snapshot`).

Also re-run the `compute_metrics` tests to confirm no regression:

Run: `cd services/python && pytest tests/test_backtest_template.py -v`

Expected: 8 tests pass total.

- [ ] **Step 5: Commit**

```bash
git add services/python/scripts/backtest_template.py services/python/tests/test_backtest_template.py
git commit -m "feat(backtest): add load_snapshot with sha256 verification"
```

---

## Task 4: `write_results` — standardized output writer (TDD)

**Files:**
- Modify: `services/python/scripts/backtest_template.py`
- Modify: `services/python/tests/test_backtest_template.py`

- [ ] **Step 1: Write the failing tests**

Append to `services/python/tests/test_backtest_template.py`:

```python


# --- write_results tests ---

import json
import subprocess

from backtest_template import write_results  # noqa: E402


def _stub_git_sha(monkeypatch, sha: str = "a3f1b2c4", dirty: bool = False):
    """Patch git subprocess calls used by write_results."""
    def fake_check_output(cmd, *args, **kwargs):
        if cmd[:2] == ["git", "rev-parse"]:
            return (sha + "deadbeef").encode()  # full 40-char output (sha + tail)
        if cmd[:2] == ["git", "status"]:
            return (b"M file.py\n" if dirty else b"")
        raise RuntimeError(f"unexpected git call: {cmd}")
    monkeypatch.setattr(subprocess, "check_output", fake_check_output)


def test_write_results_creates_four_files(tmp_path, monkeypatch):
    _stub_git_sha(monkeypatch)
    trades = pd.DataFrame([
        {"direction": "long", "pnl_pct": 0.05, "bars_held": 10, "symbol": "BTC_USDT"},
    ])
    metrics = {"trades": 1, "wr": 1.0, "pf": float("inf"),
               "avg_pnl_pct": 0.05, "total_pnl_pct": 0.05,
               "median_hold_bars": 10.0,
               "by_direction": {"long": {"trades": 1}, "short": {"trades": 0}}}
    params = {"SNAPSHOT_DATE": "2026-04-22", "SMA_FAST": 5, "SMA_SLOW": 20}

    results_root = tmp_path / "results"
    run_id = "test_run_a3f1b2c4_20260423T120000Z"
    write_results(
        run_id=run_id,
        trades_df=trades,
        metrics=metrics,
        params=params,
        snapshot_date="2026-04-22",
        snapshot_candles_sha256="c" * 64,
        snapshot_universe_sha256="u" * 64,
        results_root=results_root,
    )

    run_dir = results_root / run_id
    assert (run_dir / "trades.csv").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "params.json").exists()
    assert (run_dir / "run_metadata.json").exists()


def test_write_results_metrics_json_roundtrip(tmp_path, monkeypatch):
    _stub_git_sha(monkeypatch)
    trades = pd.DataFrame([{"direction": "long", "pnl_pct": 0.05, "bars_held": 10}])
    metrics = {"trades": 1, "wr": 1.0, "pf": 9999.0, "avg_pnl_pct": 0.05,
               "total_pnl_pct": 0.05, "median_hold_bars": 10.0,
               "by_direction": {"long": {"trades": 1}, "short": {"trades": 0}}}
    params = {"SNAPSHOT_DATE": "2026-04-22"}
    results_root = tmp_path / "results"
    run_id = "test_run_a3f1b2c4_20260423T120000Z"
    write_results(run_id=run_id, trades_df=trades, metrics=metrics, params=params,
                  snapshot_date="2026-04-22", snapshot_candles_sha256="c" * 64,
                  snapshot_universe_sha256="u" * 64, results_root=results_root)

    loaded = json.loads((results_root / run_id / "metrics.json").read_text())
    assert loaded == metrics


def test_write_results_metadata_captures_git_sha_and_dirty_flag(tmp_path, monkeypatch):
    _stub_git_sha(monkeypatch, sha="feedface", dirty=True)
    trades = pd.DataFrame([{"direction": "long", "pnl_pct": 0.01, "bars_held": 1}])
    metrics = {"trades": 1, "wr": 1.0, "pf": float("inf"), "avg_pnl_pct": 0.01,
               "total_pnl_pct": 0.01, "median_hold_bars": 1.0,
               "by_direction": {"long": {"trades": 1}, "short": {"trades": 0}}}
    params = {"SNAPSHOT_DATE": "2026-04-22"}
    results_root = tmp_path / "results"
    run_id = "test_run_feedface_20260423T120000Z"
    write_results(run_id=run_id, trades_df=trades, metrics=metrics, params=params,
                  snapshot_date="2026-04-22", snapshot_candles_sha256="c" * 64,
                  snapshot_universe_sha256="u" * 64, results_root=results_root)

    md = json.loads((results_root / run_id / "run_metadata.json").read_text())
    assert md["git_sha"].startswith("feedface")
    assert md["git_dirty"] is True
    assert md["snapshot_date"] == "2026-04-22"
    assert md["snapshot_candles_sha256"] == "c" * 64
    assert "python_version" in md
    assert "timestamp_utc" in md


def test_write_results_raises_when_run_dir_exists(tmp_path, monkeypatch):
    _stub_git_sha(monkeypatch)
    trades = pd.DataFrame([{"direction": "long", "pnl_pct": 0.01, "bars_held": 1}])
    metrics = {"trades": 1, "wr": 1.0, "pf": float("inf"), "avg_pnl_pct": 0.01,
               "total_pnl_pct": 0.01, "median_hold_bars": 1.0,
               "by_direction": {"long": {"trades": 1}, "short": {"trades": 0}}}
    params = {"SNAPSHOT_DATE": "2026-04-22"}
    results_root = tmp_path / "results"
    run_id = "test_run_a3f1b2c4_20260423T120000Z"
    (results_root / run_id).mkdir(parents=True)  # pre-create

    with pytest.raises(FileExistsError):
        write_results(run_id=run_id, trades_df=trades, metrics=metrics, params=params,
                      snapshot_date="2026-04-22", snapshot_candles_sha256="c" * 64,
                      snapshot_universe_sha256="u" * 64, results_root=results_root)
```

- [ ] **Step 2: Run tests — expect ImportError on `write_results`**

Run: `cd services/python && pytest tests/test_backtest_template.py -v -k write_results`

Expected: `ImportError: cannot import name 'write_results' from 'backtest_template'`.

- [ ] **Step 3: Implement `write_results`**

Add imports at the top of `services/python/scripts/backtest_template.py`:

```python
import json
import subprocess
import sys
from datetime import datetime, timezone
```

Add this function after `load_snapshot` (before `compute_metrics`):

```python
def _git_sha() -> str:
    """Return current HEAD SHA (40 chars)."""
    out = subprocess.check_output(["git", "rev-parse", "HEAD"])
    return out.decode().strip()


def _git_dirty() -> bool:
    """True if working tree has uncommitted changes."""
    out = subprocess.check_output(["git", "status", "--porcelain"])
    return bool(out.strip())


def _pip_freeze() -> list[str]:
    """Return installed packages, sorted, one per line."""
    out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    return sorted(out.decode().strip().splitlines())


def write_results(
    *,
    run_id: str,
    trades_df: pd.DataFrame,
    metrics: dict,
    params: dict,
    snapshot_date: str,
    snapshot_candles_sha256: str,
    snapshot_universe_sha256: str,
    results_root: Path | str = Path("results"),
) -> Path:
    """Write the four canonical output files to results/<run_id>/.

    Files:
      - trades.csv (one row per closed trade)
      - metrics.json (compute_metrics output)
      - params.json (the strategy params)
      - run_metadata.json (git SHA, dirty flag, snapshot shas, versions, wall time)

    Raises FileExistsError if results/<run_id>/ already exists — never overwrite.
    """
    results_root = Path(results_root)
    run_dir = results_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    trades_df.to_csv(run_dir / "trades.csv", index=False)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=str)

    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2, sort_keys=True, default=str)

    metadata = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "snapshot_date": snapshot_date,
        "snapshot_candles_sha256": snapshot_candles_sha256,
        "snapshot_universe_sha256": snapshot_universe_sha256,
        "python_version": sys.version,
        "pip_freeze": _pip_freeze(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=str)

    return run_dir
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `cd services/python && pytest tests/test_backtest_template.py -v`

Expected: 12 tests pass (4 original + 4 `load_snapshot` + 4 `write_results`).

- [ ] **Step 5: Commit**

```bash
git add services/python/scripts/backtest_template.py services/python/tests/test_backtest_template.py
git commit -m "feat(backtest): add write_results with metadata capture"
```

---

## Task 5: Wire `main()` stub + `build_run_id` helper in template

**Files:**
- Modify: `services/python/scripts/backtest_template.py`

**Context:** With helpers tested, assemble the template into a runnable skeleton. The user-facing parts are the SNAPSHOT_DATE constant, PARAMS block, and a clearly-marked strategy logic stub in `main()`.

- [ ] **Step 1: Add `build_run_id` helper and `main` skeleton**

Append to `services/python/scripts/backtest_template.py`:

```python


# --- RUN ID ---


def build_run_id(script_path: str | Path) -> str:
    """Compose the canonical run id for a backtest script invocation.

    Format: {script_stem}_{git_sha[:8]}_{YYYYMMDDTHHMMSSZ}
    """
    script_stem = Path(script_path).stem
    sha8 = _git_sha()[:8]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{script_stem}_{sha8}_{ts}"


# --- MAIN (edit this per backtest) ---


def main() -> None:
    import random

    import numpy as np

    # Seed EVERY source of randomness you touch. Add torch here if you import it.
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Warn loudly if git is dirty — reproducibility from SHA alone is lost.
    if _git_dirty():
        print(
            "WARNING: git working tree is dirty. "
            "This run cannot be exactly reproduced from git SHA alone.",
            file=sys.stderr,
        )

    candles, universe = load_snapshot(SNAPSHOT_DATE)

    # -------------------------------------------------------------------
    # TODO: STRATEGY LOGIC GOES HERE
    #
    # Your job:
    #   1. Use `candles` and `universe` to generate trades.
    #   2. Build a trades DataFrame with columns at minimum:
    #        direction ('long'|'short'), pnl_pct (float), bars_held (int)
    #      Optional extra columns (symbol, entry_time, exit_time, ...) are fine.
    #
    # Replace the line below with your actual strategy.
    # -------------------------------------------------------------------
    trades = pd.DataFrame(columns=["direction", "pnl_pct", "bars_held"])

    metrics = compute_metrics(trades)

    # Snapshot shas are read from the manifest by a convenience wrapper.
    c_sha, u_sha = _parse_manifest_row(
        Path("data/snapshots/MANIFEST.md").read_text(), SNAPSHOT_DATE
    )

    run_id = build_run_id(__file__)
    params = {k: v for k, v in globals().items()
              if k.isupper() and not k.startswith("_")}
    write_results(
        run_id=run_id,
        trades_df=trades,
        metrics=metrics,
        params=params,
        snapshot_date=SNAPSHOT_DATE,
        snapshot_candles_sha256=c_sha,
        snapshot_universe_sha256=u_sha,
    )
    print(f"run_id: {run_id}")
    print(f"metrics: {metrics}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add a test for `build_run_id`**

Append to `services/python/tests/test_backtest_template.py`:

```python


from backtest_template import build_run_id  # noqa: E402


def test_build_run_id_format(monkeypatch):
    _stub_git_sha(monkeypatch, sha="feedface")
    run_id = build_run_id("/some/path/backtest_foo.py")
    # stem_sha8_ts; ts is variable so only assert the prefix + that the ts suffix is present.
    assert run_id.startswith("backtest_foo_feedface_")
    # timestamp suffix is YYYYMMDDTHHMMSSZ (16 chars ending in Z)
    ts = run_id.split("_")[-1]
    assert len(ts) == 16
    assert ts.endswith("Z")
    assert ts[8] == "T"
```

- [ ] **Step 3: Run all template tests**

Run: `cd services/python && pytest tests/test_backtest_template.py -v`

Expected: 13 tests pass.

- [ ] **Step 4: Lint the template**

Run: `cd services/python && ruff check scripts/backtest_template.py`

Expected: "All checks passed!" or fix any issues reported.

- [ ] **Step 5: Commit**

```bash
git add services/python/scripts/backtest_template.py services/python/tests/test_backtest_template.py
git commit -m "feat(backtest): wire main() and build_run_id in template"
```

---

## Task 6: `export_snapshot.py` helpers — detect_leveraged, sha256, append_manifest (TDD)

**Files:**
- Create: `services/python/scripts/export_snapshot.py`
- Create: `services/python/tests/test_export_snapshot.py`

**Context:** Split `export_snapshot.py` into pure helpers (testable) + a `main()` that does the actual postgres + Gate.io work (integration-tested by running it). This task does the pure helpers.

- [ ] **Step 1: Write the failing tests for helpers**

Create `services/python/tests/test_export_snapshot.py`:

```python
"""Tests for export_snapshot.py pure helpers (leveraged detection, sha256, manifest append)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_snapshot import (  # noqa: E402
    append_manifest_row,
    detect_leveraged,
    sha256_file,
)


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("BTC_USDT", False),
        ("FARTCOIN5L_USDT", True),
        ("ETH3S_USDT", True),
        ("DOGE5S_USDT", True),
        ("SOL3L_USDT", True),
        ("BTC3_USDT", False),  # 3 alone isn't leverage
        ("ABC5_USDT", False),  # 5 alone isn't leverage
        ("LONG_USDT", False),  # L is a letter but no digit prefix
    ],
)
def test_detect_leveraged(symbol, expected):
    assert detect_leveraged(symbol) is expected


def test_sha256_file_deterministic(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2
    # Known SHA-256 of "hello world"
    assert h1 == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_sha256_file_changes_on_edit(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    h1 = sha256_file(p)
    p.write_bytes(b"hello WORLD")
    h2 = sha256_file(p)
    assert h1 != h2


def test_append_manifest_row_appends_to_existing_file(tmp_path):
    manifest = tmp_path / "MANIFEST.md"
    manifest.write_text(
        "# Backtest Data Snapshots\n\n"
        "| Date       | Candles rows | Candles sha256 | Universe rows | Universe sha256 | Notes |\n"
        "|------------|--------------|-----|---------------|-----|-------|\n"
    )

    append_manifest_row(
        manifest_path=manifest, date="2026-04-22",
        candles_rows=123, candles_sha="a" * 64,
        universe_rows=45, universe_sha="b" * 64,
        notes="first export",
    )

    text = manifest.read_text()
    assert "| 2026-04-22 | 123 | " + "a" * 64 + " | 45 | " + "b" * 64 + " | first export |" in text


def test_append_manifest_row_raises_on_duplicate_date(tmp_path):
    manifest = tmp_path / "MANIFEST.md"
    manifest.write_text(
        "# Backtest Data Snapshots\n\n"
        "| Date       | Candles rows | Candles sha256 | Universe rows | Universe sha256 | Notes |\n"
        "|------------|--------------|-----|---------------|-----|-------|\n"
        "| 2026-04-22 | 100 | a | 10 | b | test |\n"
    )

    with pytest.raises(ValueError, match="already in MANIFEST"):
        append_manifest_row(
            manifest_path=manifest, date="2026-04-22",
            candles_rows=123, candles_sha="a" * 64,
            universe_rows=45, universe_sha="b" * 64,
            notes="second export",
        )
```

- [ ] **Step 2: Run tests — expect ImportError**

Run: `cd services/python && pytest tests/test_export_snapshot.py -v`

Expected: `ModuleNotFoundError: No module named 'export_snapshot'`.

- [ ] **Step 3: Create minimal `export_snapshot.py` with just the helpers**

Create `services/python/scripts/export_snapshot.py`:

```python
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


LEVERAGED_RE = re.compile(r"[35][LS](?:_USDT)?$")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Snapshot as-of date (YYYY-MM-DD, UTC)")
    parser.add_argument("--notes", default="", help="Free-form note for MANIFEST.md")
    args = parser.parse_args()
    raise NotImplementedError("main() wired in Task 7")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `cd services/python && pytest tests/test_export_snapshot.py -v`

Expected: 11 tests pass (8 parametrized + 3 others).

- [ ] **Step 5: Commit**

```bash
git add services/python/scripts/export_snapshot.py services/python/tests/test_export_snapshot.py
git commit -m "feat(backtest): export_snapshot helpers (leveraged, sha256, manifest)"
```

---

## Task 7: `export_snapshot.main()` — postgres + Gate.io integration

**Files:**
- Modify: `services/python/scripts/export_snapshot.py`

**Context:** With helpers tested, wire the orchestration. Can't unit-test postgres+HTTP — covered by running it for real in Task 8.

- [ ] **Step 1: Replace the stub `main()`**

Replace the `main()` function in `services/python/scripts/export_snapshot.py` with this:

```python
def _load_candles(engine, as_of: str) -> "pd.DataFrame":
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
    """Fetch Gate.io USDT futures contracts via public REST API."""
    import json
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    url = "https://api.gateio.ws/api/v4/futures/usdt/contracts"
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    try:
        resp = urlopen(req, timeout=30)
        return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as e:
        raise RuntimeError(f"failed to fetch Gate.io universe: {e}") from e


def _build_universe_df(candles_df, contracts: list[dict]) -> "pd.DataFrame":
    """Build universe snapshot DataFrame from candles + Gate.io contract list."""
    import pandas as pd

    # earliest candle per asset -> listed_since approximation
    listed = (candles_df.groupby("asset")["timestamp"].min()
              .reset_index().rename(columns={"timestamp": "listed_since"}))

    # Gate.io /futures/usdt/contracts field names (as of 2026-04):
    #   name, last_price, trade_size_24h, funding_rate, mark_price, in_delisting, ...
    # If the API changes field names, update below — Task 8 step 2 output will flag it.
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
            "quote_volume_24h": _safe_float(c.get("trade_size_24h")),
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

    import pandas as pd  # noqa: F401 — required for _load_candles / _build_universe_df
    from sqlalchemy import create_engine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Snapshot as-of date (YYYY-MM-DD, UTC)")
    parser.add_argument("--notes", default="", help="Free-form note for MANIFEST.md")
    parser.add_argument(
        "--snapshots-dir", default="data/snapshots",
        help="Directory for Parquet + MANIFEST.md (default: data/snapshots)",
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
```

- [ ] **Step 2: Lint**

Run: `cd services/python && ruff check scripts/export_snapshot.py`

Expected: no issues.

- [ ] **Step 3: Re-run helper tests (no regression)**

Run: `cd services/python && pytest tests/test_export_snapshot.py -v`

Expected: 11 tests pass.

- [ ] **Step 4: Commit**

```bash
git add services/python/scripts/export_snapshot.py
git commit -m "feat(backtest): wire export_snapshot main() (postgres + Gate.io)"
```

---

## Task 8: Run `export_snapshot.py` to create the first real snapshot

**Files:**
- Modify: `data/snapshots/MANIFEST.md` (auto-appended)
- Create: `data/snapshots/candles_15m_<date>.parquet` (gitignored)
- Create: `data/snapshots/universe_<date>.parquet` (gitignored)

**Context:** Integration smoke test. Runs against real local postgres + Gate.io API. Pick today's date (or any date with enough historical data in postgres).

- [ ] **Step 1: Verify postgres is reachable and has data**

Run (adjust DATABASE_URL if yours differs):

```bash
cd services/python
python -c "
from sqlalchemy import create_engine, text
import os
url = os.environ.get('DATABASE_URL', 'postgresql://postgres:MySQL100%25@localhost:5432/market')
e = create_engine(url)
with e.connect() as c:
    print(c.execute(text('SELECT COUNT(*) FROM asset_prices_15m')).scalar())
    print(c.execute(text('SELECT MIN(timestamp), MAX(timestamp) FROM asset_prices_15m')).scalar())
"
```

Expected: a row count (likely millions) and a date range. If it fails, fix DB connectivity before continuing.

- [ ] **Step 2: Run the exporter for today's date**

```bash
cd services/python
python scripts/export_snapshot.py --date 2026-04-23 --notes "first snapshot"
```

Expected stdout:
```
[export_snapshot] loading candles with timestamp < 2026-04-23 00:00 UTC ...
[export_snapshot]   N candles
[export_snapshot] fetching Gate.io futures universe ...
[export_snapshot]   N pairs (M leveraged)
[export_snapshot] wrote data/snapshots/candles_15m_2026-04-23.parquet
[export_snapshot] wrote data/snapshots/universe_2026-04-23.parquet
[export_snapshot] appended manifest row for 2026-04-23
```

- [ ] **Step 3: Verify the MANIFEST.md row was added**

Run: `git diff data/snapshots/MANIFEST.md`

Expected: a new row at the bottom with today's date and two 64-char sha256s.

- [ ] **Step 4: Verify Parquet files exist but are gitignored**

Run: `git status data/snapshots/`

Expected: only `MANIFEST.md` is listed as modified — Parquet files are ignored.

Also verify the files exist:

```bash
ls -lah data/snapshots/*.parquet
```

Expected: two files, sized in 10s-100s of MB.

- [ ] **Step 5: Commit the manifest update**

```bash
git add data/snapshots/MANIFEST.md
git commit -m "data(snapshot): first snapshot 2026-04-23"
```

---

## Task 9: Reference SMA backtest — the permanent canary

**Files:**
- Create: `services/python/scripts/backtest_reference_sma.py`

**Context:** This is the smoke test that proves the protocol works. Simple 20-bar SMA crossover on BTC. If this reference EVER stops reproducing byte-identically, the plumbing is broken.

- [ ] **Step 1: Create `backtest_reference_sma.py` from the template**

Copy `services/python/scripts/backtest_template.py` → `services/python/scripts/backtest_reference_sma.py`.

Then edit three things in the new file:

1. At the top, replace the docstring with:

```python
"""Reference backtest — 20/50 SMA crossover on BTC.

This file is the permanent reproducibility canary for the backtest protocol.
If running this script twice produces different metrics.json, the plumbing
is broken. Do not edit except to rebuild the canary.
"""
```

2. Change `SNAPSHOT_DATE` to the date you exported in Task 8 (e.g., `"2026-04-23"`).

3. Replace the `# --- PARAMS ---` block and the strategy logic block inside `main()` with:

```python
# --- PARAMS ---
ASSET = "BTC"
SMA_FAST = 20
SMA_SLOW = 50
HOLD_BARS = 96  # 24h on 15m
```

Replace the strategy logic section inside `main()` (everything between the TODO banner and `metrics = compute_metrics(trades)`) with:

```python
    btc = candles[candles["asset"] == ASSET].sort_values("timestamp").reset_index(drop=True)
    if len(btc) < SMA_SLOW + 2:
        trades = pd.DataFrame(columns=["direction", "pnl_pct", "bars_held"])
    else:
        close = btc["close"].astype(float).values
        fast = pd.Series(close).rolling(SMA_FAST).mean().values
        slow = pd.Series(close).rolling(SMA_SLOW).mean().values
        # Crossover: fast moves above slow → long entry at next bar.
        rows = []
        i = SMA_SLOW
        while i < len(close) - HOLD_BARS - 1:
            if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
                entry = float(close[i + 1])
                exit_ = float(close[i + 1 + HOLD_BARS])
                rows.append({
                    "direction": "long",
                    "pnl_pct": (exit_ - entry) / entry,
                    "bars_held": HOLD_BARS,
                    "entry_idx": int(i + 1),
                })
                i += HOLD_BARS + 1
            else:
                i += 1
        trades = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["direction", "pnl_pct", "bars_held"]
        )
```

- [ ] **Step 2: Lint**

Run: `cd services/python && ruff check scripts/backtest_reference_sma.py`

Expected: no issues.

- [ ] **Step 3: Commit the reference script (not yet the results)**

```bash
git add services/python/scripts/backtest_reference_sma.py
git commit -m "feat(backtest): reference SMA backtest (reproducibility canary)"
```

---

## Task 10: Verify self-reproducibility + commit canary result

**Files:**
- Create: `results/backtest_reference_sma_<sha>_<ts>/trades.csv`
- Create: `results/backtest_reference_sma_<sha>_<ts>/metrics.json`
- Create: `results/backtest_reference_sma_<sha>_<ts>/params.json`
- Create: `results/backtest_reference_sma_<sha>_<ts>/run_metadata.json`

- [ ] **Step 1: Run the reference backtest once**

```bash
cd services/python
python scripts/backtest_reference_sma.py
```

Expected stdout ends with `run_id: backtest_reference_sma_<sha8>_<ts>` and `metrics: {...}`. No dirty-tree warning (we just committed).

Record the `run_id` printed as `RUN1`.

- [ ] **Step 2: Run it a second time**

```bash
cd services/python
python scripts/backtest_reference_sma.py
```

Record the `run_id` printed as `RUN2`. `RUN2 != RUN1` because timestamps differ — that's expected.

- [ ] **Step 3: Diff the two `metrics.json` files**

```bash
cd services/python
diff results/$RUN1/metrics.json results/$RUN2/metrics.json
```

Expected: **no output** (files byte-identical). If they differ, the plumbing has a non-determinism bug — fix before proceeding.

- [ ] **Step 4: Verify `params.json` and `trades.csv` are also identical**

```bash
cd services/python
diff results/$RUN1/params.json results/$RUN2/params.json
diff results/$RUN1/trades.csv results/$RUN2/trades.csv
```

Expected: no output for either.

- [ ] **Step 5: Verify `run_metadata.json` differs ONLY in `timestamp_utc` (and derived run_id)**

```bash
cd services/python
diff results/$RUN1/run_metadata.json results/$RUN2/run_metadata.json
```

Expected: only the `run_id` and `timestamp_utc` lines differ. Everything else (git_sha, git_dirty, pip_freeze, python_version, snapshot shas) is identical.

- [ ] **Step 6: Keep only one canary result; commit it**

```bash
cd services/python
# Delete the second run to keep results/ tidy.
rm -rf results/$RUN2
# Commit the first as the canonical canary.
git add results/$RUN1/
git commit -m "test(backtest): canary reference SMA run — protocol smoke test"
```

- [ ] **Step 7: Run the full test suite to confirm no regression**

```bash
cd services/python
pytest tests/test_backtest_template.py tests/test_export_snapshot.py -v
```

Expected: all tests pass (13 template + 11 export_snapshot = 24 tests).

---

## Task 11: `docs/backtest-protocol.md` — the rules

**Files:**
- Create: `docs/backtest-protocol.md`

- [ ] **Step 1: Create `docs/backtest-protocol.md`**

Create with this exact content:

```markdown
# Backtest Protocol

Rules for writing backtests in this project. Non-negotiable because the point is reproducibility: any Claude agent, now or in future sessions, should produce byte-identical output when running the same backtest script against the same snapshot date.

## Why this exists

Paper trading revealed that past backtests could not be reproduced — "every AI gives different numbers." Three causes: mutable postgres (candles are appended live, VPS prunes at 90 days), non-frozen universe (which pairs were listed on Gate.io futures on any given day was not captured), and re-invented metric math (different "PF" formulas across scripts). This protocol eliminates all three.

## The rules (non-negotiable)

1. **Start from the template.** New backtests MUST be created via:
   ```bash
   cp services/python/scripts/backtest_template.py services/python/scripts/backtest_<name>.py
   ```
   Do not write a backtest from scratch.

2. **Declare SNAPSHOT_DATE.** Set the constant at the top of the new script to a date listed in `data/snapshots/MANIFEST.md`. Do not dynamically compute the date.

3. **Do not read live postgres for canonical numbers.** All data comes from `load_snapshot(SNAPSHOT_DATE)`. The only script allowed to touch postgres is `export_snapshot.py`.

4. **Do not edit the canonical helpers.** `load_snapshot`, `compute_metrics`, `write_results`, and `_sha256_file` are copy-pasted (not imported) into every backtest to guarantee identical math. Do not modify them after copying.

5. **Write via `write_results()`.** Every backtest produces `results/<run_id>/{trades.csv, metrics.json, params.json, run_metadata.json}`. No custom output paths.

6. **Commit the results folder.** Published numbers must be citable — `results/<run_id>/` stays in git permanently. Do not delete. Do not edit. If the analysis is wrong, write a new run, don't rewrite an old one.

## How to refresh the snapshot

When you need a newer data cut:

```bash
cd services/python
python scripts/export_snapshot.py --date 2026-05-15 --notes "whatever"
git add ../../data/snapshots/MANIFEST.md
git commit -m "data(snapshot): 2026-05-15"
```

The `.parquet` files themselves are gitignored (they're large). They live locally and on your Google Drive backup (`My Drive/ai-finance/`). If another agent/machine needs them, they copy from Drive and `load_snapshot` verifies the sha256.

## How to reproduce a published result

1. Read `results/<run_id>/run_metadata.json`. Note the `git_sha`, `snapshot_date`, and the two snapshot sha256s.
2. `git checkout <git_sha>`.
3. Verify `data/snapshots/MANIFEST.md` lists that `snapshot_date` with matching sha256s. If the Parquet files aren't on your disk, copy them from Drive.
4. Re-run the script: `python services/python/scripts/backtest_<name>.py`.
5. Diff the new `metrics.json` against the committed one. Must be byte-identical.

## When reproducibility legitimately breaks

Two cases don't count as bugs — just write a new run:

- **Dependency upgrade** (you ran `pip install -U <package>`). The `pip_freeze` field in metadata shows the difference.
- **Param or strategy edit** (you intentionally changed something). Commit first, run second — new git_sha → new run_id → different output is expected.

In both cases: do not overwrite the old `results/<run_id>/`. Write a new one and let history speak.

## When iterating

While you're still writing the strategy block, `git_dirty: true` will appear in metadata. That's fine for iteration. But before you publish a number as "the result," commit the script first so the run has a clean SHA.

## Related files

- Template: `services/python/scripts/backtest_template.py`
- Exporter: `services/python/scripts/export_snapshot.py`
- Manifest: `data/snapshots/MANIFEST.md`
- Design: `docs/superpowers/specs/2026-04-22-backtest-reproducibility-protocol-design.md`
- Canary: `services/python/scripts/backtest_reference_sma.py` + its `results/` folder
```

- [ ] **Step 2: Commit**

```bash
git add docs/backtest-protocol.md
git commit -m "docs(backtest): backtest protocol rules"
```

---

## Task 12: `CLAUDE.md` — Backtesting section

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Read current CLAUDE.md to find the right insertion point**

Run: `cat CLAUDE.md | head -40`

Find the "## Git Workflow" section. The new "## Backtesting" section goes immediately before "## Git Workflow" (or at the end, whichever you prefer — both are acceptable).

- [ ] **Step 2: Insert new section**

Add this block before the existing "## Git Workflow" heading:

```markdown
## Backtesting

Any new backtest MUST:
- Start from `services/python/scripts/backtest_template.py` — do not write from scratch.
- Follow `docs/backtest-protocol.md` — non-negotiable rules.
- Declare `SNAPSHOT_DATE` at the top (matching a row in `data/snapshots/MANIFEST.md`) and use `load_snapshot()` — never query live postgres for canonical numbers.
- Write results via `write_results()` to `results/<run_id>/`. Commit that folder.

Canonical helpers (`load_snapshot`, `compute_metrics`, `write_results`) are copy-pasted (not imported) into each backtest to guarantee identical math across all runs. Do not edit them after copying.

Existing scripts in `services/python/learn/22042026/` are archival — treat their reported numbers as unreliable until re-run under this protocol. The permanent smoke test is `services/python/scripts/backtest_reference_sma.py`; if its output ever drifts from the committed canary in `results/`, the plumbing is broken and must be fixed before trusting any new result.

```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): add Backtesting section pointing at protocol"
```

---

## Task 13: Cross-agent reproducibility test (end-to-end)

**Files:** None — validation only.

**Context:** Final end-to-end proof. Spawn a fresh subagent (zero context), hand it only the protocol doc, tell it to re-run the canary. Compare output to the committed `results/`.

- [ ] **Step 1: Dispatch a subagent**

In a new agent conversation (or fresh Claude Code session):

> "Read `docs/backtest-protocol.md` and `services/python/scripts/backtest_reference_sma.py`. Run the reference backtest. Report the full contents of the resulting `metrics.json`."

- [ ] **Step 2: Capture the subagent's reported metrics**

Save the subagent's output to a scratch file: `/tmp/agent_metrics.json`.

- [ ] **Step 3: Diff against the committed canary metrics**

```bash
diff /tmp/agent_metrics.json services/python/results/$CANARY_RUN_ID/metrics.json
```

Expected: no output (byte-identical). The canary run_id is whatever was committed in Task 10.

If they differ → the subagent deviated from the protocol OR the template has a non-determinism bug. Either way, diagnose and fix before declaring the protocol working.

- [ ] **Step 4: Record the test result**

If the diff is clean, add a short line to `data/snapshots/MANIFEST.md` in a new "Verification log" section (this is manual, one-time):

```markdown

## Verification log

| Date       | Test                       | Result | Notes |
|------------|----------------------------|--------|-------|
| 2026-04-23 | cross-agent SMA canary     | PASS   | fresh subagent reproduced committed metrics byte-identically |
```

- [ ] **Step 5: Commit the verification entry**

```bash
git add data/snapshots/MANIFEST.md
git commit -m "test(backtest): cross-agent canary reproducibility PASS"
```

---

## Task 14: Push branch and open PR to master

**Files:** None — git + GitHub only.

**Context:** Per project CLAUDE.md, master is the branch the VPS pulls from. Feature branches merge via PR to keep commit history clean and feature-bounded.

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/backtest-reproducibility
```

- [ ] **Step 2: Open a PR to master**

```bash
gh pr create --base master --title "feat(backtest): reproducibility protocol" --body "$(cat <<'EOF'
## Summary
- Adds `export_snapshot.py` (postgres + Gate.io → Parquet) and a frozen `data/snapshots/MANIFEST.md` with sha256 per snapshot.
- Adds `backtest_template.py` with canonical `load_snapshot`, `compute_metrics`, `write_results` helpers. Every new backtest copies this file.
- Adds `backtest_reference_sma.py` — a 20/50 SMA crossover backtest that acts as the permanent reproducibility canary.
- Adds `docs/backtest-protocol.md` and a CLAUDE.md section pointing at it.
- No retrofit of past scripts in `services/python/learn/22042026/` — they stay as-is.

## Test plan
- [ ] `pytest services/python/tests/test_backtest_template.py services/python/tests/test_export_snapshot.py -v` — all green
- [ ] `python services/python/scripts/backtest_reference_sma.py` runs twice and produces byte-identical `metrics.json`
- [ ] Fresh subagent running the canary reproduces the committed `metrics.json` byte-for-byte (Task 13)

See `docs/superpowers/specs/2026-04-22-backtest-reproducibility-protocol-design.md` for the full spec.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: After review, merge**

Once tests pass in CI (or you've re-verified locally) and you're satisfied:

```bash
gh pr merge --merge --delete-branch
```

Prefer `--merge` (merge commit) over `--squash` if you want the per-task commit history preserved for future `git log`. Prefer `--squash` if you want one atomic commit in master history. Your call per project convention.

---

## Self-review checklist (for the plan author)

Before handing off for execution, verify:

1. **Spec coverage:** Every component from the spec is implemented in some task.
   - `export_snapshot.py` → Tasks 6, 7, 8
   - `backtest_template.py` → Tasks 2, 3, 4, 5
   - `MANIFEST.md` → Task 1 (scaffold), Task 8 (first row)
   - `docs/backtest-protocol.md` → Task 11
   - `CLAUDE.md` edit → Task 12
   - `.gitignore` edit → Task 1
   - Reference SMA canary → Task 9
   - Self-reproducibility test → Task 10
   - Cross-agent reproducibility test → Task 13
   - PR to master → Task 14

2. **Placeholders:** No "TBD" / "implement similar to above" / "add error handling" — all code in the plan is complete.

3. **Type consistency:** `load_snapshot` returns `(candles_df, universe_df)` throughout. `write_results` keyword args match between tasks 4, 5, 9. `compute_metrics` output keys match between tasks 2 and 9.

4. **Scope sanity:** Single implementation plan. No cross-system decomposition needed.
