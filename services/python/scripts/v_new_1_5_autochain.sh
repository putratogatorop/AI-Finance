#!/usr/bin/env bash
# Auto-chain: waits for the v_new_1 baseline retrain to finish, then runs
# the v_new_1.5 chain. Must be started AFTER `tmux training_v2` has begun.
# Run this in its own tmux session.
set -euo pipefail

REPO_ROOT="/opt/ai-finance"
LOG_DIR="${REPO_ROOT}/services/python/results/v_new_1_5_chain"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/autochain.log"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*" | tee -a "${LOG}"; }

log "=== autochain start; waiting for baseline retrain to finish ==="

# Phase 1: wait for baseline retrain process to exit.
# Poll every 2 minutes (cache-warm-friendly delay choices N/A on VPS).
while pgrep -f v_new_1_phase3_train >/dev/null 2>&1; do
    sleep 120
    log "  baseline retrain still running ($(pgrep -af v_new_1_phase3_train | head -1 | cut -c1-80)...)"
done

log "Baseline retrain process gone. Verifying outputs..."

# Verify all 6 parquets and 4 model-meta files exist.
all_ok=1
for d in long short; do
    for kind in oof cal oos; do
        path="${REPO_ROOT}/services/python/data/v_new_1/${kind}_scored_${d}.parquet"
        if [[ ! -f "${path}" ]]; then
            log "  MISSING: ${path}"
            all_ok=0
        else
            mtime=$(stat -c %Y "${path}")
            now=$(date +%s)
            age=$(( now - mtime ))
            log "  OK: ${path} (age ${age}s)"
        fi
    done
    for kind in joblib _meta.json; do
        path="${REPO_ROOT}/services/python/models/v_new_1_${d}/v_new_1_${d}${kind}"
        # Strip leading underscore for joblib path build
        if [[ "${kind}" == "joblib" ]]; then
            path="${REPO_ROOT}/services/python/models/v_new_1_${d}/v_new_1_${d}.joblib"
        fi
        if [[ ! -f "${path}" ]]; then
            log "  MISSING: ${path}"
            all_ok=0
        fi
    done
done

if [[ "${all_ok}" != "1" ]]; then
    log "Missing outputs. Aborting auto-chain (will not run v_new_1.5)."
    exit 1
fi

log "All baseline outputs present. Starting v_new_1.5 chain..."
bash "${REPO_ROOT}/services/python/scripts/v_new_1_5_run_chain.sh"
log "=== autochain complete ==="
