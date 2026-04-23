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
        ("BTC3L_USDC", False),  # non-USDT quote rejected
        ("ETH5L", False),  # bare token (no USDT suffix) rejected
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


def test_append_manifest_row_rejects_pipe_in_notes(tmp_path):
    manifest = tmp_path / "MANIFEST.md"
    manifest.write_text(
        "# Backtest Data Snapshots\n\n"
        "| Date       | Candles rows | Candles sha256 | Universe rows | Universe sha256 | Notes |\n"
        "|------------|--------------|-----|---------------|-----|-------|\n"
    )
    with pytest.raises(ValueError, match="must not contain"):
        append_manifest_row(
            manifest_path=manifest, date="2026-04-22",
            candles_rows=1, candles_sha="a" * 64,
            universe_rows=1, universe_sha="b" * 64,
            notes="bad|notes",
        )
