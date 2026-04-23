# Backtest Reproducibility Protocol — Design

**Date:** 2026-04-22
**Status:** Approved, awaiting implementation plan
**Scope:** New backtests only. No retrofit of existing scripts.

---

## Problem

Different Claude Code sessions (and different agents within a session) produce different backtest numbers for what should be the same analysis. The user has lost trust in past results — e.g., `train_scanner_long_v2_ml.py` claims `WR=93.1%, PF=17.86, trades=261` at ML threshold 0.80, but subsequent runs on later dates cannot reproduce it. Paper trading (MOG, CYS, GRASS, ZEREBRO, FARTCOIN5L) further diverged from the "big movers" those backtests implied we'd catch.

Root causes of the drift:

1. **Live postgres is mutable.** Candles are appended continuously; the VPS prunes at 90 days (per `CLAUDE.md`). A backtest run today against `WHERE timestamp >= '2024-01-01'` returns a different dataset than yesterday's run of the same query.
2. **Universe is not frozen.** Which pairs were listed on Gate.io futures on a given day is not captured anywhere — so "top 100 by volume" drifts silently. Leveraged tokens (e.g., `FARTCOIN5L`) slip through because nothing marks them as such.
3. **Seeds, params, and metric formulas are re-invented per script.** Two scripts computing "Profit Factor" may sum differently (include/exclude fees, treat open positions as wins/losses).
4. **No run-level metadata.** Output CSVs lack git SHA, data snapshot reference, param dump, or wall time — so "what produced this number?" is unanswerable post-hoc.

## Goal

**Any Claude agent, now or in the future, running the same backtest script against the same declared snapshot date produces byte-identical output.** Specifically: identical `metrics.json` files.

Non-goals:
- Retrofitting existing scripts in `services/python/learn/22042026/` — treated as archival.
- Building a general experiment-tracking system (MLflow/W&B). YAGNI.
- Automating snapshot refresh cadence. Manual trigger is sufficient.

## Architecture

Five artifacts in the repo — three code/data, two docs.

```
AI-Finance/
├── data/snapshots/                          # gitignored (large files)
│   ├── candles_15m_<YYYY-MM-DD>.parquet     # OHLCV for all pairs as of snapshot date
│   ├── universe_<YYYY-MM-DD>.parquet        # listed pairs, is_leveraged flag, 24h volume
│   └── MANIFEST.md                          # COMMITTED — per-snapshot sha256 + metadata
│
├── services/python/scripts/
│   ├── export_snapshot.py                   # NEW — postgres + Gate.io → Parquet
│   └── backtest_template.py                 # NEW — skeleton for every new backtest
│
├── docs/
│   └── backtest-protocol.md                 # NEW — rules, rationale, operational guide
│
├── CLAUDE.md                                # EDITED — adds Backtesting section
└── .gitignore                               # EDITED — adds data/snapshots/*.parquet
```

Note: Parquet snapshot files are gitignored (they're large and live on Drive backup per `CLAUDE.md`). The `MANIFEST.md` IS committed — it carries the sha256 of each Parquet file so consumers can verify integrity after pulling the file from Drive. `results/<run_id>/` folders ARE committed (they're small — CSV + JSON — and the whole point is permanent, citable numbers).

### Component responsibilities

#### 1. `services/python/scripts/export_snapshot.py`

One-shot script. Arguments: `--date YYYY-MM-DD` (defaults to today UTC).

Behavior:
- Queries `asset_prices_15m` from postgres for all rows with `timestamp < <date> 00:00 UTC`. Writes to `data/snapshots/candles_15m_<date>.parquet` (columns: `asset, timestamp, open, high, low, close, volume, quote_volume`).
- Fetches Gate.io futures universe via `/futures/usdt/contracts` API. For each listed contract, records: `symbol`, `listed_since` (earliest candle in postgres for that symbol), `is_leveraged` (regex: `.*[35][LS]$`), `quote_volume_24h` (from API response). Writes to `data/snapshots/universe_<date>.parquet`.
- Computes sha256 of both files.
- Appends one row to `data/snapshots/MANIFEST.md`:
  ```
  | 2026-04-22 | 18,234,102 | a3f1... | 342 | 7b2c... | initial export |
  ```
- Exits non-zero if either file already exists (must `rm` manually to re-export — prevents accidental overwrite).

#### 2. `services/python/scripts/backtest_template.py`

Skeleton file. Every new backtest is created via `cp services/python/scripts/backtest_template.py services/python/scripts/backtest_<name>.py`, then edited.

Pre-wired sections (in order, top to bottom):

- **Header comment block** — name, date, one-line purpose.
- **Hardcoded constants at top:**
  - `SNAPSHOT_DATE = "YYYY-MM-DD"` — must be edited per run, matched against `MANIFEST.md`.
  - `RANDOM_SEED = 42` — fed to `random.seed()`, `np.random.seed()`, and `torch.manual_seed()` if ML is imported.
  - Parameter block (clearly labeled `# --- PARAMS ---`) for strategy params.
- **`load_snapshot(date: str) -> tuple[pd.DataFrame, pd.DataFrame]`** — reads the two Parquet files, asserts their sha256 matches `MANIFEST.md`. Returns `(candles_df, universe_df)`.
- **`compute_metrics(trades_df: pd.DataFrame) -> dict`** — canonical metric calculation. Required keys: `trades`, `wr`, `pf`, `avg_pnl_pct`, `total_pnl_pct`, `median_hold_bars`, `by_direction: {long: {...}, short: {...}}`. This function is copy-pasted into the template (not imported from a library) so every backtest uses the same math without requiring a package.
- **`write_results(run_id, trades_df, metrics, params)`** — writes four files to `results/<run_id>/`:
  - `trades.csv` — one row per closed trade.
  - `metrics.json` — output of `compute_metrics`.
  - `params.json` — the full param block serialized.
  - `run_metadata.json` — `git_sha`, `git_dirty` (bool), `snapshot_date`, `snapshot_candles_sha256`, `snapshot_universe_sha256`, `python_version`, `pip_freeze` (list), `wall_time_seconds`, `timestamp_utc`.
- **`main()`** — orchestrates: `load_snapshot` → TODO-marked "your entry/exit logic here" block → `compute_metrics` → `write_results`.
- **`RUN_ID`** — generated as `f"{script_stem}_{git_sha[:8]}_{ts}"` where:
  - `script_stem` = filename without `.py` (e.g., `backtest_reference_sma`).
  - `git_sha[:8]` = first 8 chars of HEAD commit SHA.
  - `ts` = current UTC time formatted `YYYYMMDDTHHMMSSZ` (e.g., `20260423T141503Z`).
  - Full example: `backtest_reference_sma_a3f1b2c4_20260423T141503Z`.

#### 3. `data/snapshots/MANIFEST.md`

Append-only markdown table. Committed. Single source of truth for "what snapshots exist and what is the canonical hash of each."

Format:

```markdown
# Snapshots

| Date       | Candles rows | Candles sha256   | Universe rows | Universe sha256  | Notes |
|------------|--------------|------------------|---------------|------------------|-------|
| 2026-04-22 | 18,234,102   | a3f1b2c4d5e6f7...| 342           | 7b2c3d4e5f6a7b...| initial export |
```

New rows are appended by `export_snapshot.py`. Humans do not hand-edit.

#### 4. `docs/backtest-protocol.md`

Committed. Non-negotiable rules + rationale + operational runbook.

Sections:
1. **Why this exists** — one paragraph on the reproducibility problem.
2. **The rules (non-negotiable):**
   - New backtests MUST start from `backtest_template.py`.
   - New backtests MUST declare `SNAPSHOT_DATE` at the top.
   - New backtests MUST NOT read live postgres — only `load_snapshot()`.
   - New backtests MUST write to `results/<run_id>/` via `write_results()`.
   - Every committed result folder is permanent — do not delete, do not rewrite.
3. **How to refresh the snapshot** — `python services/python/scripts/export_snapshot.py --date <YYYY-MM-DD>`, then `git add data/snapshots/MANIFEST.md && git commit`.
4. **How to reproduce a published result** — read the `run_metadata.json`, check out the git SHA, verify the snapshot sha256 is in `MANIFEST.md`, run the script. Output `metrics.json` must match byte-for-byte.
5. **When reproducibility legitimately breaks** — dependency updates (pip install of new version), manual params changes. In both cases, write a new run_id, do not overwrite the old one.

#### 5. `CLAUDE.md` edit

New section under Architecture (or its own top-level heading):

```markdown
## Backtesting

Any new backtest MUST:
- Start from `services/python/scripts/backtest_template.py`
- Follow `docs/backtest-protocol.md` (non-negotiable rules)
- Declare `SNAPSHOT_DATE` and use `load_snapshot()` — never query live postgres for canonical numbers
- Write results via `write_results()` to `results/<run_id>/`

Existing scripts in `services/python/learn/22042026/` are archival — treat their numbers as unreliable until re-run under this protocol.
```

## Data flow

### Writing a new backtest

1. Agent reads `CLAUDE.md` at session start — sees the Backtesting section.
2. Agent reads `docs/backtest-protocol.md`.
3. Agent reads `data/snapshots/MANIFEST.md` — picks an existing snapshot date, or runs `export_snapshot.py` first.
4. Agent copies `backtest_template.py` → `backtest_<name>.py`.
5. Agent edits: sets `SNAPSHOT_DATE`, fills param block, fills entry/exit logic block.
6. Agent runs. Script auto-writes `results/<run_id>/{trades.csv, metrics.json, params.json, run_metadata.json}`.
7. Agent commits the new backtest script AND the `results/<run_id>/` folder.

### Refreshing a snapshot

1. User (or agent) runs `python services/python/scripts/export_snapshot.py --date 2026-05-15`.
2. Script writes two Parquet files to `data/snapshots/`.
3. Script appends a row to `MANIFEST.md`.
4. User commits `MANIFEST.md` (Parquet files are gitignored; they live locally + Google Drive per `CLAUDE.md`).

### Reproducing a published result

1. Read `results/<run_id>/run_metadata.json` — extract `git_sha`, `snapshot_date`, both sha256s.
2. `git checkout <git_sha>`.
3. Verify snapshot file sha256 matches `MANIFEST.md` entry for that date. If not, the local snapshot is wrong — re-export or copy from Drive.
4. Re-run the backtest script.
5. Diff the new `metrics.json` against the committed one. Must be byte-identical.

## Error handling

Hard failures (script exits non-zero):
- Snapshot file missing → `FileNotFoundError` with message: `"Run export_snapshot.py --date {date} or pick a date from MANIFEST.md"`. No fallback to live postgres.
- Snapshot sha256 mismatch vs `MANIFEST.md` → raise — reproducibility is void if data changed.
- `results/<run_id>/` already exists → raise. Do not overwrite. User must bump run_id or clean up.
- `export_snapshot.py` invoked when output Parquet file already exists → raise.

Soft warnings (script runs, writes a flag):
- Git working tree dirty → prints a loud stderr warning **before** the backtest runs (so the user can Ctrl-C if undesired), and writes `git_dirty: true` into `run_metadata.json`. Does not hard-fail because iteration is common. Future readers of that run understand it is not exactly git-reproducible.
- `universe_df.listed_since` is an approximation — it's the earliest candle in postgres for each symbol. For pairs that were delisted and re-listed, it reflects the earliest historical candle, not the most recent listing event. Acceptable for our purposes (catching "this pair didn't exist yet on the snapshot date" is the only check we need).

Not guarded:
- Agent hand-edits the template after copying (trust + CLAUDE.md rule).
- Agent ignores the protocol and reads live postgres anyway (trust + CLAUDE.md rule).
- Different machines producing different Parquet bytes from the same postgres query — caught by sha256 when a second party tries to use the snapshot.

## Testing

1. **Sanity test — self-reproducibility.** Write `services/python/scripts/backtest_reference_sma.py` from the template — a trivial 20-bar SMA crossover on BTC_USDT. Run it twice. Diff the two `metrics.json` files. Must be byte-identical. If not, the template is broken.
2. **Cross-agent test.** Spawn a fresh subagent: "Run `backtest_reference_sma.py` and report the output `metrics.json`." Its reported metrics must match the committed `results/<run_id>/metrics.json` byte-for-byte.
3. **Snapshot integrity test.** Manually corrupt one byte of `candles_15m_<date>.parquet`. Running any backtest against that snapshot must raise a sha256 mismatch error.
4. **No unit tests on business logic.** The template contains almost no logic — it's plumbing. The reference backtest IS the smoke test.

The reference backtest stays in the repo as a permanent canary. If it ever stops reproducing, something in the plumbing drifted and must be fixed before trusting new results.

## Out of scope

Explicitly NOT included:
- Library / framework package.
- Superpowers skill for writing backtests.
- CI check or pre-commit hook enforcing protocol.
- Retrofit of `services/python/learn/22042026/`.
- Snapshot refresh automation (cron, scheduler).
- Experiment-tracking UI or database.
