"""Tests for backtest_template.py helpers (canonical metric calc, etc)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Import helpers from the template script. Template is a runnable script,
# but its top-level helpers are also importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_template import compute_metrics, load_snapshot, write_results  # noqa: E402


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


# --- load_snapshot tests ---


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
    row = (
        f"| {date} | {candles_rows} | {candles_sha} "
        f"| {universe_rows} | {universe_sha} | {notes} |\n"
    )
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


# --- write_results tests ---


def _stub_git_sha(monkeypatch, sha: str = "a3f1b2c4", dirty: bool = False):
    """Patch git subprocess calls used by write_results."""
    _real_check_output = subprocess.check_output

    def fake_check_output(cmd, *args, **kwargs):
        if cmd[:2] == ["git", "rev-parse"]:
            return (sha + "deadbeef").encode()  # full 40-char output (sha + tail)
        if cmd[:2] == ["git", "status"]:
            return (b"M file.py\n" if dirty else b"")
        # Pass through non-git calls (e.g. pip freeze) to the real implementation.
        return _real_check_output(cmd, *args, **kwargs)
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
        git_dirty=False,
        wall_time_seconds=0.0,
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
                  snapshot_universe_sha256="u" * 64, git_dirty=False,
                  wall_time_seconds=0.0, results_root=results_root)

    loaded = json.loads((results_root / run_id / "metrics.json").read_text())
    assert loaded == metrics


def test_write_results_metadata_captures_git_sha_and_dirty_flag(tmp_path, monkeypatch):
    # write_results no longer calls _git_dirty() — dirty flag comes from the kwarg.
    # We still stub git rev-parse so _git_sha() inside write_results returns "feedface...".
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
                  snapshot_universe_sha256="u" * 64, git_dirty=True,
                  wall_time_seconds=1.5, results_root=results_root)

    md = json.loads((results_root / run_id / "run_metadata.json").read_text())
    assert md["git_sha"].startswith("feedface")
    assert md["git_dirty"] is True
    assert md["wall_time_seconds"] == 1.5
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
                      snapshot_universe_sha256="u" * 64, git_dirty=False,
                      wall_time_seconds=0.0, results_root=results_root)


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
