# v_new_1 Phase 3 — VPS Training Runbook

**Target:** VPS Biznet `103.103.20.30`
**User:** `ai-finance-deploy`
**Estimated wall-clock:** 4-6 hours total (both directions on 4 CPU cores)
**Status when running:** training script in tmux, live trading services unaffected

---

## Pre-flight checks (on local Mac)

```bash
cd "/Users/putratogatorop/Desktop/AI Finance"

# 1. Confirm v_new_1 data parquets exist locally
ls -lh services/python/data/v_new_1/

# Should see:
# labels_long.parquet (~7.7 MB)
# labels_short.parquet (~7.6 MB)
# labels_long_oos.parquet (~8.2 MB)
# labels_short_oos.parquet (~8.2 MB)
# features_full.parquet (~263 MB)
# universe_top100_membership.parquet (~317 KB)

# 2. Smoke-test the script with N_OPTUNA_TRIALS=1 to verify no syntax errors
N_OPTUNA_TRIALS=1 LGBM_NUM_THREADS=2 \
DYLD_LIBRARY_PATH=$(services/python/.venv/bin/python3 -c "import sklearn,os; print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))") \
services/python/.venv/bin/python3 -c "
import services.python.scripts.v_new_1_phase3_train as m
print('imports OK, N_OPTUNA_TRIALS={}, LGBM_NUM_THREADS={}'.format(m.N_OPTUNA_TRIALS, m.LGBM_NUM_THREADS))
"
```

---

## Step 1 — Commit script changes to git

The training script was modified for VPS compatibility. Need to push so VPS can `git pull`.

```bash
cd "/Users/putratogatorop/Desktop/AI Finance"

git add services/python/scripts/v_new_1_phase3_train.py CLAUDE.md docs/V_NEW_1_PHASE3_VPS_RUNBOOK.md
git status

# Verify only intended changes are staged. If clean, commit:
git commit -m "feat(v_new_1): VPS-ready training script + runbook"
git push origin master
```

---

## Step 2 — SCP data parquets to VPS

The parquet files are gitignored (too big for git). Upload manually:

```bash
# Total upload: ~295 MB. On home internet, expect 5-15 minutes.

ssh ai-finance-deploy@103.103.20.30 'mkdir -p /opt/ai-finance/services/python/data/v_new_1'

scp services/python/data/v_new_1/*.parquet \
    ai-finance-deploy@103.103.20.30:/opt/ai-finance/services/python/data/v_new_1/

# Verify upload:
ssh ai-finance-deploy@103.103.20.30 'ls -lh /opt/ai-finance/services/python/data/v_new_1/'
```

---

## Step 3 — SSH to VPS and pull latest code

```bash
ssh ai-finance-deploy@103.103.20.30

# (now on VPS)
cd /opt/ai-finance
git pull origin master

# Verify the script is there with the env-var changes:
grep -n "LGBM_NUM_THREADS\|N_OPTUNA_TRIALS = int" services/python/scripts/v_new_1_phase3_train.py

# Should show the env-var lines we added.
```

---

## Step 4 — Verify Python environment on VPS

The training script needs: `lightgbm`, `optuna`, `shap`, `joblib`, `numpy`, `pandas`, `scikit-learn`. Two paths:

### Path A: Use existing Docker container's Python (preferred)

```bash
# (on VPS)
cd /opt/ai-finance

# Find the python worker container
docker compose ps | grep python

# Verify deps inside container
docker compose exec python-worker python -c "import lightgbm, optuna, shap; print('all good')"
```

If that works, use Path A for execution. If `optuna` or `shap` are missing in the container, fall back to Path B.

### Path B: Set up host venv (fallback)

```bash
# (on VPS)
cd /opt/ai-finance/services/python

# Create venv if not present
python3 -m venv .venv

# Activate + install
source .venv/bin/activate
pip install lightgbm==4.6.0 optuna==4.8.0 shap==0.51.0 joblib pandas pyarrow scikit-learn

# Verify
python -c "import lightgbm, optuna, shap; print('all good')"
```

---

## Step 5 — Start training in tmux

```bash
# (on VPS)
cd /opt/ai-finance

# Create output dir
mkdir -p services/python/results/v_new_1_phase3

# Start tmux session
tmux new -s training

# (now inside tmux)
# Run the training with VPS-safe settings
cd /opt/ai-finance

# Path A (Docker):
docker compose exec -T python-worker bash -c "
cd /opt/ai-finance && \
LGBM_NUM_THREADS=4 N_OPTUNA_TRIALS=50 \
python services/python/scripts/v_new_1_phase3_train.py 2>&1
" | tee services/python/results/v_new_1_phase3/training.log

# OR Path B (host venv):
LGBM_NUM_THREADS=4 N_OPTUNA_TRIALS=50 \
services/python/.venv/bin/python3 services/python/scripts/v_new_1_phase3_train.py 2>&1 \
| tee services/python/results/v_new_1_phase3/training.log

# Detach from tmux: Ctrl+B then D
# (training continues in background — you can close SSH safely)
```

---

## Step 6 — Monitoring

### Check training progress (from local Mac)

```bash
ssh ai-finance-deploy@103.103.20.30 'tail -50 /opt/ai-finance/services/python/results/v_new_1_phase3/training.log'

# Or stream live:
ssh ai-finance-deploy@103.103.20.30 'tail -f /opt/ai-finance/services/python/results/v_new_1_phase3/training.log'
```

### Check VPS resources during training

```bash
ssh ai-finance-deploy@103.103.20.30 'top -bn1 | head -20'

# Should see python process at <400% CPU (4 cores out of 8)
# Memory usage <4 GB (out of 8 GB)
# Live trading services should still be present and responsive
```

### Check live trading is still healthy

```bash
ssh ai-finance-deploy@103.103.20.30 'docker compose ps'
ssh ai-finance-deploy@103.103.20.30 'docker compose logs --tail=20 paper-executor'

# Look for: signal processing happening, no timeouts, recent timestamps
```

### Re-attach to tmux

```bash
ssh ai-finance-deploy@103.103.20.30
tmux attach -t training
```

---

## Step 7 — When training finishes

Training prints `🎉 Phase 3 complete` at the end (see logs). Outputs land at:

- `/opt/ai-finance/services/python/models/v_new_1_long/v_new_1_long.joblib` + `_meta.json`
- `/opt/ai-finance/services/python/models/v_new_1_short/v_new_1_short.joblib` + `_meta.json`
- `/opt/ai-finance/services/python/data/v_new_1/oos_scored_long.parquet`
- `/opt/ai-finance/services/python/data/v_new_1/oos_scored_short.parquet`

### SCP results back to local Mac for analysis

```bash
# (on local Mac)
cd "/Users/putratogatorop/Desktop/AI Finance"

mkdir -p services/python/models/v_new_1_long services/python/models/v_new_1_short

scp ai-finance-deploy@103.103.20.30:/opt/ai-finance/services/python/models/v_new_1_long/* \
    services/python/models/v_new_1_long/

scp ai-finance-deploy@103.103.20.30:/opt/ai-finance/services/python/models/v_new_1_short/* \
    services/python/models/v_new_1_short/

scp ai-finance-deploy@103.103.20.30:/opt/ai-finance/services/python/data/v_new_1/oos_scored_*.parquet \
    services/python/data/v_new_1/
```

---

## Emergency stop

If live trading gets impacted (delayed signals, OOM warnings, etc.):

```bash
ssh ai-finance-deploy@103.103.20.30
tmux attach -t training
# Inside tmux: Ctrl+C to kill training
# OR force kill:
pkill -f v_new_1_phase3_train.py
```

Live trading services should resume normal operation within seconds.

---

## Common issues

### Issue 1: `pip install` fails on VPS due to missing system libs
For `lightgbm`, may need: `sudo apt-get install libomp-dev libgomp1`. If you can't sudo as `ai-finance-deploy`, fall back to Docker (Path A).

### Issue 2: Training process gets killed by OOM
Reduce N_OPTUNA_TRIALS to 30. Or pre-load only the columns needed at each phase. Or check that no other heavy services are running.

### Issue 3: tmux not installed
`sudo apt-get install tmux` (if you have sudo). Otherwise use `nohup` + `&`:
```bash
nohup [training command] > training.log 2>&1 &
echo "PID: $!"
disown
# Now you can close SSH; training continues
```

### Issue 4: Live trading slows down
Lower `LGBM_NUM_THREADS=2` to leave more cores for trading. Wall-clock will roughly double but production stays responsive.

### Issue 5: Disk space low
Training outputs are small (~2 MB total). The data parquets (~300 MB) are the bulk. After training, optionally remove the local parquets on VPS:
```bash
rm /opt/ai-finance/services/python/data/v_new_1/*.parquet
```
(Keep the local Mac copies for re-runs.)
