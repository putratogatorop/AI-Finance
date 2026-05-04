# v_new_1 Shadow Scoring Service — Deployment Guide

**Date:** 2026-05-02
**Status:** Deployed to VPS — observation mode only, no trading

---

## What it does

Every 4 hours at :05 past the hour, the shadow scorer:

1. Queries Postgres for the current top-100 coins by trailing 30-day USD volume
2. Fetches the last ~220 4h candles per coin (resampled from 15m DB data)
3. Computes all 44 v_new_1 features per coin (matching training exactly)
4. Scores every coin with both `v_new_1_long` and `v_new_1_short` base LightGBM models
5. Applies Platt calibration (pre-fitted, loaded from disk)
6. Fetches the current CFGI value from alternative.me API (falls back to DB)
7. Applies CFGI gates:
   - Long signals only when `CFGI > 60` (greed regime)
   - Short signals only when `CFGI < 50` (fear regime)
8. Selects top-K by Platt score: top 13 longs, top 14 shorts
9. Logs **ALL** 100×2 scores to a daily Parquet (with `action` column indicating taken/skipped)

**No actual trades are placed.** This is pure observation mode.

---

## Policy parameters (from Phase 4 best_policy.json)

| Parameter | Value |
|---|---|
| `top_k_long` | 13 |
| `top_k_short` | 14 |
| `score_threshold_long` | 0.317 |
| `score_threshold_short` | 0.468 |
| `cfgi_long_gate` | 60 (long only when CFGI > 60) |
| `cfgi_short_gate` | 50 (short only when CFGI < 50) |
| `position_size_pct` | 0.518% (for future paper trading reference) |
| `leverage` | 5× (reference only) |

---

## Files

| Path | Purpose |
|---|---|
| `services/python/scripts/v_new_1_shadow_score.py` | Main shadow scoring script (cron target) |
| `services/python/scripts/v_new_1_platt_fit.py` | One-time Platt calibration setup script |
| `services/python/models/v_new_1_long/v_new_1_long.joblib` | Long base LightGBM model |
| `services/python/models/v_new_1_long/platt_calibrator.joblib` | Long Platt calibrator |
| `services/python/models/v_new_1_long/platt_calibrator_meta.json` | Long calibration metrics |
| `services/python/models/v_new_1_short/v_new_1_short.joblib` | Short base LightGBM model |
| `services/python/models/v_new_1_short/platt_calibrator.joblib` | Short Platt calibrator |
| `services/python/models/v_new_1_short/platt_calibrator_meta.json` | Short calibration metrics |
| `services/python/results/v_new_1_shadow_log/v_new_1_signals_YYYY-MM-DD.parquet` | Daily signal logs |
| `services/python/results/v_new_1_shadow_log/cron.log` | Cron stdout/stderr |

---

## Cron entry

```bash
# v_new_1 shadow scoring — every 4h at :05, just after 4h candle close
5 */4 * * * /opt/ai-finance/services/python/.venv/bin/python3 /opt/ai-finance/services/python/scripts/v_new_1_shadow_score.py >> /opt/ai-finance/services/python/results/v_new_1_shadow_log/cron.log 2>&1
```

Install: `crontab -e` then add the line above.

---

## Monitoring

### View today's log
```python
import pandas as pd
df = pd.read_parquet("/opt/ai-finance/services/python/results/v_new_1_shadow_log/v_new_1_signals_2026-05-02.parquet")
taken = df[df["action"] == "taken"]
print(taken[["timestamp","symbol","direction","score_platt","rank"]].sort_values("score_platt", ascending=False))
```

### Tail cron log
```bash
tail -f /opt/ai-finance/services/python/results/v_new_1_shadow_log/cron.log
```

### Check last scoring run
```bash
ls -la /opt/ai-finance/services/python/results/v_new_1_shadow_log/
```

### Query all taken signals across days
```python
import pandas as pd, pathlib
logs = sorted(pathlib.Path("/opt/ai-finance/services/python/results/v_new_1_shadow_log").glob("v_new_1_signals_*.parquet"))
all_df = pd.concat([pd.read_parquet(p) for p in logs], ignore_index=True)
taken = all_df[all_df["action"] == "taken"]
print(f"Total taken signals: {len(taken)} ({taken['direction'].value_counts().to_dict()})")
```

---

## How to disable

```bash
# Remove the shadow scorer cron entry (leaves all other crons intact)
crontab -l | grep -v v_new_1_shadow_score | crontab -
```

---

## How to re-fit Platt calibrators

If the base models are retrained or updated:
```bash
cd /opt/ai-finance
services/python/.venv/bin/python3 services/python/scripts/v_new_1_platt_fit.py
```

This loads `features_full.parquet` and `labels_long/short.parquet` from
`services/python/data/v_new_1/`, fits LogisticRegression on the cal_fit slice
(2024-10-01 → 2025-12-31), and overwrites the platt_calibrator.joblib files.

---

## Reality reconciliation (after 7+ days)

After 7 days of shadow scoring, compare predictions against actual moves:

```python
import pandas as pd
from sqlalchemy import create_engine, text

# Load shadow predictions
logs = sorted(pathlib.Path("/opt/ai-finance/services/python/results/v_new_1_shadow_log").glob("*.parquet"))
shadow = pd.concat([pd.read_parquet(p) for p in logs])
taken = shadow[shadow["action"] == "taken"].copy()

# For each taken signal, check if the coin actually moved 10%+ within 7 days
engine = create_engine("postgresql://aifinance:...@localhost:5432/aifinance")
# Query asset_prices_15m for each (symbol, signal_timestamp) to compute
# forward_max / market_close_at_signal within 7 days
# long_hit = (forward_max / entry_close - 1) >= 0.10
# short_hit = (forward_min / entry_close - 1) <= -0.10
```

Key questions to answer after 2-4 weeks:
1. Top-K precision: what fraction of "taken" long signals actually moved +10% within 7d?
2. CFGI gate effectiveness: compare taken vs skipped_cfgi hit rates
3. Score discrimination: do higher platt scores have higher precision?
4. Any regime where the model consistently fails?

If precision on taken signals ≥ 0.35 (vs ~0.24 baseline for long in current market),
the model is adding value and the strategy is a candidate for paper trading promotion.

---

## Promoting to paper trading

After 2-4 weeks of satisfactory shadow performance:
1. Enable paper trading mode in `paper_executor.py` for v_new_1 signals
2. Use the same CFGI gates and top-K limits
3. Start with `position_size_pct = 0.518%` per the Phase 4 optimized policy
4. Run alongside existing v8 paper strategies for A/B comparison
5. After 4 more weeks of paper trading, evaluate for live capital allocation

---

## Resource constraints

- Script caps LightGBM inference to 2 threads (`MAX_LGBM_THREADS = 2`)
- Typical runtime: 2-4 minutes for 100 coins (well within 4h interval)
- Memory: ~500MB peak during feature computation
- Does NOT modify any existing tables or scanner signals
