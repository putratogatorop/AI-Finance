# AI-Finance Project Guidelines

## Strategy Roadmap (revised 2026-05-02 — paired breakout/momentum capture for big moves)

**Endgoal:** A LONG+SHORT paired breakout/momentum strategy targeting **200-1000% per year** by capturing the 10-30%+ moves that happen frequently in crypto. Both detectors run concurrently; sizing varies by conviction × regime. The current `macd_pullback_long` system is admitted as a low-return scalper and has been demonstrated to underperform BTC buy-and-hold over 4 years (96% strategy return vs ~250% buy-hold). The new direction explicitly targets per-trade magnitude, not frequency.

**Evidence base — the `big_movers` table:**
- Production DB has a `big_movers` table with **38,068 historical records** of ≥10% moves
- Empirical observation: most big moves complete within **2 hours to 3 days** (not weeks/months)
- Schema: `symbol, move_start, move_peak, start_price, peak_price, move_pct`
- This is the EVIDENCE base for what's actually catchable in crypto markets — not assumption

**Why prior research was wrong-direction (kept here so future sessions don't repeat):**
- macd_pullback_long: enters AFTER pullback consolidation, with 6× ATR fixed TP and 84-bar (14d) timeout. Most trades exit at 1-2% gains via timeout/breakeven, not at the full TP. Realized R:R is closer to 1.5:1, not 6:2.
- After fees (Gate.io 0.10% round-trip × thousands of trades) the strategy contributes 50-100pp friction over 4 years
- The 2× return over 4 years (vs BTC's ~3.5-4×) means the system is effectively charity to the exchange
- Phase L+S2a tested naive long+short pair — failed (-9,500pp deficit) because the underlying detectors are pullback-style with structural bear bleed

**New direction — paired breakout/momentum (v_new_1):**

**Why pair (both always on):**
1. The user's instinct, validated by data: specialists each tuned for their regime, both running concurrently with conviction-based sizing
2. Long specialist (`v_new_1_long`): BEAST in bull regimes, weak/dormant in bear
3. Short specialist (`v_new_1_short`): BEAST in bear regimes, weak/dormant in bull
4. Continuous regime gating via cls_score × regime_alignment × position_tier (Phase Sizing tier-learning approach)

**Strategy timeframe (calibrated to actual big_movers data):**
- 4h trigger candle for entry
- Hold horizon: 1 hour to 3 days typical (matches big_movers durations)
- 7-10 day hard timeout maximum
- NOT 60-90 day holds (that's investing territory, not trading)

**Detector candidates (TBD via Phase Breakout research):**
- N-bar high breakout (Donchian-style: e.g., 12-bar 4h high = 2-day high break)
- BB squeeze + directional breakout
- Volume-confirmed range breakout (vol z > 1.5 + price > N-bar high)
- Mirror logic for short detector (range-low breakdown, BB squeeze + bear breakout)

**Exit logic (parabolic-capture):**
- Initial stop: 5× ATR-14 (wider than current 2×, lets trade breathe through normal volatility)
- Trailing stop: max(initial_stop, highest_close_since_entry − 3× ATR)
- Hard timeout: 7-10 days (most big_movers events complete by then)
- **NO fixed TP** — capping kills the parabolic capture which is the whole point

**Labels for ML training (using big_movers table directly):**
- Positive label: signal preceded a `big_movers` event (move_start within 24-48h of signal time, same symbol) AND was directionally aligned
- Negative label: signal did NOT precede a big_movers event in the window
- This is more honest than current "did it hit 6× ATR" labels — it's labeled against ACTUAL big-move ground truth

**Foundation reuse from prior research (don't restart):**
- Phase B 2020-2026 OHLCV data: applicable as detector input
- 4-year multi-OOS framework (2020 / 2021 / 2022 / 2026 Q1): still the gold standard for validation
- Phase 0 threshold-locking protocol: prevents threshold contamination
- v2p3 architecture (LightGBM + isotonic calibration + 40-feature foundation): reusable as ML layer template for v_new_1
- Phase Sizing 10-tier conviction sizing (Scheme C): reusable on v_new_1 once base detector + classifier are validated

**Promotion gate for v_new_1:**
- 4-year multi-OOS sum_pnl > BTC buy-hold over same period (~250%)
- Median PF across 4 OOS years > 1.5
- Positive PnL in EVERY OOS year (true all-weather)
- 2022 bear year PnL > 0 (the test that pullback-long couldn't pass)
- Statistically significant (paired sign-test) over BTC buy-hold

**Strategy Roadmap research order:**
1. Phase Breakout-1 (1 day): naive Donchian-style breakout on 4h, run on 2020-2026 universe, baseline metrics
2. Phase Breakout-2 (2 days): ML filter on top using v2p3 architecture + big_movers labels
3. Phase Breakout-3 (1 day): mirror for short side (breakdown detector)
4. Phase Pair-Validation (1 day): both detectors firing concurrently, multi-OOS validation
5. Total: ~5 days research before deploy decision

**Production stays on v1 @ 0.40 macd_pullback_long until v_new_1 paired system clears the BTC buy-hold gate.** The current strategy isn't actively bad, it's just dramatically underperforming the alternative. No reason to disrupt production until v_new_1 proves itself.

---

## Agent Delegation (Opus + Sonnet pattern)

Default workflow: **Opus 4.7 max thinking is the planner/supervisor; Sonnet via the Agent tool is the executor.** This saves ~50–60% on tokens for research and analysis sessions. Empirically validated on 2026-05-02 macd_pullback_long classifier work.

**Delegate to Sonnet when:**
- The task involves >5 file reads or grep searches
- Writing analysis scripts and running them
- Multi-step research with web search
- Multiple independent audits — fan out 3–5 agents in parallel (single message, multiple Agent calls)
- Backtest re-runs, hyperparameter sweeps, comparison reports

**Do NOT delegate to Sonnet when:**
- Single-file edits (faster to do it directly)
- Continuous deep reasoning across many dependent steps (context resets between agent calls hurt)
- Tasks where Opus's judgment matters more than throughput

**Always verify Sonnet's output, especially for ML/quant work.** Sonnet has been observed to:
- Fabricate metric values (claimed top SHAP=0.330 when actual was 0.2595)
- Confuse in-sample with OOS (reported "temporal walk-forward PF=6.22" that included the training period)
- Optimize the wrong objective (locked threshold at PF-max 0.475 when PnL-max was 0.40 — the wrong threshold would have shipped to live trading)
- Claim ship gates passed when they failed by a hair (PR-AUC delta 0.0352 vs gate 0.04)

**Verification rules for ML/quant work touching the live trading path:**
- Independent re-derivation in a separate script — never trust the training script's own reported numbers
- Always include multi-threshold sensitivity (n, sum_pnl_pct, PF, WR, MDD across ≥6 thresholds)
- For threshold decisions, check absolute `sum_pnl_pct` first; PF is secondary (see `feedback_pnl_over_pf.md` in memory)
- Treat any "passed" gate within 0.01 of the threshold as failed, and demand a written waiver in the meta JSON
- Pre-2026 numbers in this repo are in-sample for nearly all classifiers — only Q1 2026 is genuine OOS today

**Parallelism rule:** when agent tasks are independent, spawn them in a single message with multiple Agent calls. Sequential spawning is wasted wall-clock.

## Compute (revised 2026-05-02 — VPS used for ML training; M2 Pro for dev only)

**Two-tier compute model:**

### Local M2 Pro Mac — light dev + analysis only
- Use for: data inspection, small backtests (<10 min), result analysis, plotting, code editing, smoke-testing scripts before VPS deploy
- Avoid for: sustained ML training, Optuna sweeps, anything > 30 min CPU-bound
- M2 Pro thermally throttles on sustained ML workloads — confirmed 2026-05-02 when Phase 3 training was killed mid-run for overheating
- libomp via DYLD_LIBRARY_PATH for LightGBM smoke tests

### VPS (Biznet NEO Lite MM 8.8) — ML training + production trading
**Updated 2026-05-02:** previous rule "VPS for live trading only, no ML training" is RELAXED. ML training is now permitted on VPS during off-peak hours, with constraints below. User prefers SSH workflow over Colab browser UI.

**Hardware:** 8 vCPU / 8GB RAM / 60GB SSD, Linux. Jakarta. SSH as `ai-finance-deploy` (NOT root). Repo at `/opt/ai-finance`.

**Constraints when training on VPS:**
- **Cap LightGBM `n_jobs=4`** (leave 4 cores for live trading services). Never set `n_jobs=-1` during production hours.
- **Monitor live trading during training**: scanner + paper executor must remain responsive. If trade decisions get delayed > 30 seconds, halt training.
- **RAM budget**: training process must stay under 4 GB resident (leave 4 GB for production services + OS). Use `pd.read_parquet(columns=...)` to load only needed columns; `gc.collect()` between training runs.
- **Run in screen/tmux** for resumability across SSH disconnects.
- **Time of day**: prefer overnight UTC (Jakarta night-day) when scanner activity is lower. Saturdays/Sundays acceptable too.
- **Disk**: training outputs (models + meta) are small (~1 MB each). Data parquets (~300 MB total) live under `/opt/ai-finance/services/python/data/v_new_1/` (gitignored, scp'd from local).

**Workflow for VPS training:**
1. Local: smoke-test the training script on Mac with `N_OPTUNA_TRIALS=1, n_estimators=10` to verify pipeline integrity
2. Local: scp data parquets to VPS (`scp services/python/data/v_new_1/*.parquet ai-finance-deploy@103.103.20.30:/opt/ai-finance/services/python/data/v_new_1/`)
3. Push code via git (training script committed to master), VPS does `git pull` to sync
4. SSH to VPS: `ssh ai-finance-deploy@103.103.20.30`
5. `cd /opt/ai-finance` and verify env / venv
6. Start training in tmux: `tmux new -s training`, then `python services/python/scripts/v_new_1_phase3_train.py 2>&1 | tee services/python/results/v_new_1_phase3/training.log`
7. Detach (`Ctrl+B, D`), close SSH. Reconnect later to check progress: `tmux attach -t training`
8. When done, scp model artifacts + meta JSONs back to Mac for analysis

**Failure mode protocol:**
- If VPS RAM hits >7 GB during training: kill the training process (`tmux send-keys -t training C-c`)
- If live scanner alerts about delayed responses: halt training immediately, investigate
- Keep production services (paper_executor, scanner) at higher priority via `nice` if needed

### Why no Colab anymore
User cancelled Colab Pro (slow, browser-UI-clunky). Free Colab + ngrok/Tailscale SSH hacks violate ToS and are unreliable. VPS-based SSH workflow matches the user's preferred operational style and existing infrastructure. Training cost on VPS = $0 incremental (already paying for the box).

## Code Style

- Python 3.12, ruff linter (line-length=100)
- ML convention: uppercase X (X_train, X_test) allowed in `src/ml/**` and `src/rl/**`
- Use dataclasses for config, PyTorch for neural networks
- Scripts use `sys.path.insert(0, ".")` and run from `services/python/`

## Architecture

- **Next.js** — frontend + API routes (Prisma ORM)
- **Python** — ML/RLwai
 engine, data pipeline, background jobs (no FastAPI)
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
