#!/usr/bin/env bash
# v_new_1 v2 chain: baseline retrain (CV-fixed) + augment + v_new_1.5 train.
# Operates on data/v_new_1_v2/ (top-200, 2020-05→2024-12 train, 2025-01→2025-06 cal,
# 2025_h2 + 2026_q1 OOS). Run in tmux on VPS.
set -euo pipefail

REPO_ROOT="/opt/ai-finance"
PY="${REPO_ROOT}/services/python/.venv/bin/python3"
LOG_DIR="${REPO_ROOT}/services/python/results/v_new_1_v2_chain"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/chain.log"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*" | tee -a "${LOG}"; }

# Common env for both training runs
export V_NEW_1_DATA_DIR="${REPO_ROOT}/services/python/data/v_new_1_v2"
export V_NEW_1_TRAIN_FIT_END="2024-12-31"
export V_NEW_1_CAL_FIT_START="2025-01-01"
export V_NEW_1_OOS_YEARS="2025_h2,2026_q1"
export N_OPTUNA_TRIALS="${N_OPTUNA_TRIALS:-50}"
export LGBM_NUM_THREADS="${LGBM_NUM_THREADS:-4}"

log "=== v_new_1 v2 chain start ==="
log "DATA_DIR=${V_NEW_1_DATA_DIR}"
log "TRAIN_FIT_END=${V_NEW_1_TRAIN_FIT_END}  CAL_FIT_START=${V_NEW_1_CAL_FIT_START}"
log "OOS_YEARS=${V_NEW_1_OOS_YEARS}"
log "N_OPTUNA_TRIALS=${N_OPTUNA_TRIALS}  LGBM_NUM_THREADS=${LGBM_NUM_THREADS}"

# 1. Validate input parquets present
log "Step 1/4: validate v2 input parquets..."
for f in features_full.parquet labels_long.parquet labels_short.parquet \
         labels_long_oos.parquet labels_short_oos.parquet \
         universe_top100_membership.parquet ; do
    p="${V_NEW_1_DATA_DIR}/${f}"
    if [[ ! -f "${p}" ]]; then
        log "  MISSING: ${p}. Aborting."
        exit 1
    fi
    log "  OK: ${p} ($(stat -c%s "${p}") bytes)"
done

# 2. Train v_new_1 baseline (CV-fixed)
log "Step 2/4: training v_new_1 baseline (~30 min on 8-vCPU)..."
cd "${REPO_ROOT}"
"${PY}" -u services/python/scripts/v_new_1_phase3_train.py 2>&1 | tee -a "${LOG}"
log "  baseline done"

# 3. Augment features (sequence features from OOF/CAL/OOS scores)
log "Step 3/4: augmenting features for v_new_1.5..."
"${PY}" -u services/python/scripts/v_new_1_5_augment_features.py 2>&1 | tee -a "${LOG}"
if [[ ! -f "${V_NEW_1_DATA_DIR}/features_full_v1_5.parquet" ]]; then
    log "Augment did not produce features_full_v1_5.parquet. Aborting."
    exit 1
fi
log "  augment done"

# 4. Train v_new_1.5 (sequence features layered onto BGM)
log "Step 4/4: training v_new_1.5 (V_NEW_1_5_MODE=1, ~30 min)..."
V_NEW_1_5_MODE=1 "${PY}" -u services/python/scripts/v_new_1_phase3_train.py 2>&1 | tee -a "${LOG}"

log "=== v_new_1 v2 chain DONE ==="
log "Artifacts:"
log "  ${REPO_ROOT}/services/python/models/v_new_1_long/  (CV-fixed baseline on v2 data)"
log "  ${REPO_ROOT}/services/python/models/v_new_1_short/"
log "  ${REPO_ROOT}/services/python/models/v_new_1_long_v1_5/  (sequence-feature v_new_1.5)"
log "  ${REPO_ROOT}/services/python/models/v_new_1_short_v1_5/"
log "  ${V_NEW_1_DATA_DIR}/oos_scored_*.parquet  +  oof_scored_*.parquet  +  cal_scored_*.parquet"
log "  ${V_NEW_1_DATA_DIR}/features_full_v1_5.parquet"
