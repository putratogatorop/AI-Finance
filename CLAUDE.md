# AI-Finance Project Guidelines

## Compute

- User has **Colab Pro** subscription. Use it for heavy computation (ML training, large data processing, anything GPU-bound).
- **Never** suggest running training locally — always offload to Colab.
- **Colab browser UI** is the preferred workflow: open notebooks directly in Colab from GitHub. VS Code kernel connection is unreliable (upload widgets, rendering bugs).
- Large data files (`.npz`, `.parquet`) go to Google Drive at `My Drive/ai-finance/`. Notebooks copy from Drive — never use `files.upload()` widget.
- For Phase 1 training (small models): T4 GPU is sufficient. Save A100 for Phase 2 or larger workloads.
- Checkpoints save to Google Drive for crash recovery.

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
- **No backtesting, no ML training, no historical data loads.** Keep the VPS light — any script that scans years of data belongs on local/Colab.
- Stack: Docker Compose (Postgres + Python workers + Next.js), same as `docker-compose.yml`.
- Deploy via `git pull` + `docker compose up -d --build`.

### Local (Mac)
- **Purpose: research, backtesting, data pipeline, feature engineering.**
- Holds full historical datasets (Parquet on disk / Google Drive).
- Develops and validates strategies before promoting to VPS.
- ML training still offloads to Colab (see Compute section).

### Data flow
- Historical Parquet lives on Google Drive / local — **never on VPS**.
- VPS Postgres pulls from Gate.io live + keeps a rolling 90-day window.
- Models trained on Colab → artifacts saved to Drive → downloaded to VPS for inference.

When writing code, always ask: "does this run on VPS or local?" VPS code must stay lean and stateless w.r.t. long history.

## Git Workflow

Every change must be recorded in git — no ad-hoc edits on the VPS, no "quick fixes" outside version control.

- **Work on a feature branch** (e.g., `feat/rl-gym-environment`, `fix/scanner-timeout`). Never commit directly to `master`/`main`.
- **Commit early, commit often.** Use Conventional Commits style (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`) — matches existing history.
- **Push to origin** regularly so work is backed up and visible.
- **Merge to `master` (production) via PR** — this is the branch the VPS pulls from. Only merged code reaches production.
- **VPS deploys via `git pull` on `master`** + `docker compose up -d --build`. Never `scp` files or edit directly on the server.
- **Rollback = `git checkout <previous-commit>` + redeploy**, not manual file surgery.

If a hotfix is ever needed directly on the VPS (emergency only), commit it back to a branch immediately and open a PR — nothing stays off-git.
