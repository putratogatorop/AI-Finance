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
