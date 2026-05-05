#!/usr/bin/env bash
# v_new_1.5 chain runner — to be executed on the VPS after the v_new_1 baseline
# retrain finishes. Steps:
#   1. Augment features_full.parquet with sequence features  →  features_full_v1_5.parquet
#   2. Train v_new_1.5 (V_NEW_1_5_MODE=1)                    →  v_new_1_long_v1_5, v_new_1_short_v1_5
#
# Logs to v_new_1_5_chain.log. Run inside tmux. Exits non-zero on first failure.
set -euo pipefail

REPO_ROOT="/opt/ai-finance"
PY="${REPO_ROOT}/services/python/.venv/bin/python3"
LOG_DIR="${REPO_ROOT}/services/python/results/v_new_1_5_chain"
mkdir -p "${LOG_DIR}"

LOG="${LOG_DIR}/chain.log"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*" | tee -a "${LOG}"; }

log "=== v_new_1.5 chain start ==="

# 1. Sanity check: baseline OOF/CAL parquets must exist before augment.
log "Step 1/3: validate baseline OOF/CAL/OOS parquets are present..."
for d in long short; do
    for kind in oof cal oos; do
        path="${REPO_ROOT}/services/python/data/v_new_1/${kind}_scored_${d}.parquet"
        if [[ ! -f "${path}" ]]; then
            log "MISSING: ${path}. Aborting — did the v_new_1 baseline retrain finish?"
            exit 1
        fi
        size=$(stat -c%s "${path}")
        log "  OK  ${path} (${size} bytes)"
    done
done

# 2. Augment features (~3-5 min).
log "Step 2/3: augmenting features (sequence features from OOF/CAL/OOS scores)..."
cd "${REPO_ROOT}"
"${PY}" -u services/python/scripts/v_new_1_5_augment_features.py 2>&1 | tee -a "${LOG}"
if [[ ! -f services/python/data/v_new_1/features_full_v1_5.parquet ]]; then
    log "Augment did not produce features_full_v1_5.parquet. Aborting."
    exit 1
fi
log "  features_full_v1_5.parquet ready"

# 3. Train v_new_1.5 (~2h).
log "Step 3/3: training v_new_1.5 (V_NEW_1_5_MODE=1, N_OPTUNA_TRIALS=50, threads=4)..."
V_NEW_1_5_MODE=1 N_OPTUNA_TRIALS=50 LGBM_NUM_THREADS=4 \
    "${PY}" -u services/python/scripts/v_new_1_phase3_train.py 2>&1 | tee -a "${LOG}"

log "=== v_new_1.5 chain DONE ==="
log "Artifacts:"
log "  ${REPO_ROOT}/services/python/models/v_new_1_long_v1_5/"
log "  ${REPO_ROOT}/services/python/models/v_new_1_short_v1_5/"
log "  ${REPO_ROOT}/services/python/data/v_new_1/oos_scored_{long,short}_v1_5.parquet"
log "  ${REPO_ROOT}/services/python/data/v_new_1/oof_scored_{long,short}_v1_5.parquet"
log "  ${REPO_ROOT}/services/python/data/v_new_1/cal_scored_{long,short}_v1_5.parquet"
