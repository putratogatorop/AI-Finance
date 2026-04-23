# AI-Finance Project Guidelines

## Compute

- ML training and backtesting run **locally** (Mac) or on the **VPS** for lightweight jobs.
- Large data files (`.npz`, `.parquet`) go to Google Drive at `My Drive/ai-finance/` for backup.
- GPU training: run locally if Mac supports it, otherwise use cloud GPU as needed.

## Code Style

- Python 3.12, ruff linter (line-length=100)
- ML convention: uppercase X (X_train, X_test) allowed in `src/ml/**` and `src/rl/**`
- Use dataclasses for config, PyTorch for neural networks
- Scripts use `sys.path.insert(0, ".")` and run from `services/python/`

## Architecture

- **Next.js** — frontend + API routes (Prisma ORM)
- **Python** — ML/RL engine, data pipeline, background jobs (no FastAPI)
- **PostgreSQL** — shared data layer (Python writes, Next.js reads)
- **Docker Compose** — local deployment
- Exchange: Gate.io futures (Binance blocked in Indonesia)

## Environments

Two environments with strict separation of concerns:

### VPS (Biznet NEO Lite MM 8.8 — Jakarta, 8 vCPU / 8GB RAM / 60GB SSD)
- **Purpose: LIVE TRADING ONLY.** Scanner, paper/live executor, dashboard, Gate.io order routing.
- **Postgres retention: last 90 days max.** Older candles/signals get pruned or archived to Drive.
- **No backtesting, no ML training, no historical data loads.** Keep the VPS light — any script that scans years of data belongs on local.
- Stack: Docker Compose (Postgres + Python workers + Next.js), same as `docker-compose.yml`.
- Deploy via `git pull` + `docker compose up -d --build`.

### Local (Mac)
- **Purpose: research, backtesting, ML training, data pipeline, feature engineering.**
- Holds full historical datasets (Parquet on disk / Google Drive).
- Develops and validates strategies before promoting to VPS.

### Data flow
- Historical Parquet lives on Google Drive / local — **never on VPS**.
- VPS Postgres pulls from Gate.io live + keeps a rolling 90-day window.
- Models trained locally → committed to repo → deployed to VPS for inference.

When writing code, always ask: "does this run on VPS or local?" VPS code must stay lean and stateless w.r.t. long history.

## Backtesting

Any new backtest MUST:
- Start from `services/python/scripts/backtest_template.py` — do not write from scratch.
- Follow `docs/backtest-protocol.md` — non-negotiable rules.
- Declare `SNAPSHOT_DATE` at the top (matching a row in `data/snapshots/MANIFEST.md`) and use `load_snapshot()` — never query live postgres for canonical numbers.
- Write results via `write_results()` to `results/<run_id>/`. Commit that folder.

Canonical helpers (`load_snapshot`, `compute_metrics`, `write_results`) are copy-pasted (not imported) into each backtest to guarantee identical math across all runs. Do not edit them after copying.

Existing scripts in `services/python/learn/22042026/` are archival — treat their reported numbers as unreliable until re-run under this protocol. The permanent smoke test is `services/python/scripts/backtest_reference_sma.py`; if its output ever drifts from the committed canary in `results/`, the plumbing is broken and must be fixed before trusting any new result.

## Git Workflow

Every change must be recorded in git — no ad-hoc edits on the VPS, no "quick fixes" outside version control.

- **Work on a feature branch** (e.g., `feat/rl-gym-environment`, `fix/scanner-timeout`). Never commit directly to `master`/`main`.
- **Commit early, commit often.** Use Conventional Commits style (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`) — matches existing history.
- **Push to origin** regularly so work is backed up and visible.
- **Merge to `master` (production) via PR** — this is the branch the VPS pulls from. Only merged code reaches production.
- **VPS deploys via `git pull` on `master`** + `docker compose up -d --build`. Never `scp` files or edit directly on the server.
- **Rollback = `git checkout <previous-commit>` + redeploy**, not manual file surgery.

If a hotfix is ever needed directly on the VPS (emergency only), commit it back to a branch immediately and open a PR — nothing stays off-git.

## Trading Rules

- **Direction**: shorts only until long edge is proven via backtest. Long-side momentum entries failed ship gate 2026-04-23 (PF 0.819).
- **Per-coin HTF downtrend filter**: TBD by 24-variant backtest grid (`backtest_short_grid_v1`). Status will be set to "required" or "rejected" once grid runs.
- **Risk per trade**: hard `5% SL`. Aim ≥ `3:1 R:R` (`15% TP` baseline).
- **Sizing**: `5% of equity` per trade, `5x leverage` default → 25% notional per trade. `MAX_CONCURRENT_POSITIONS = 5` → ≤ 125% gross notional.
- **Realistic PF floor for ship**: `1.30` over `1000+` trades. The 100%/month return target is aspirational, NOT an engineering spec. Realistic monthly: 20-40% in good months, drawdowns are normal.
- **Backtest-before-deploy is non-negotiable**: any new scanner / exit / param change cites a `results/<run_id>/metrics.json` in its deploy PR.
