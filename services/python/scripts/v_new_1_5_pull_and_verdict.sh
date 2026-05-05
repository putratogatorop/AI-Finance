#!/usr/bin/env bash
# Mac-side: polls VPS for v_new_1.5 TRAINING_DONE.marker, then SCPs the
# v1_5 model artifacts + scored parquets back to local, then runs
# v_new_1_5_verdict.py and prints the verdict path.
set -euo pipefail

REPO_ROOT="/Users/putratogatorop/Desktop/AI Finance"
VPS="ai-finance-deploy@103.103.20.30"
VPS_REPO="/opt/ai-finance"
PY="${REPO_ROOT}/services/python/.venv/bin/python3"
LOG="${REPO_ROOT}/services/python/results/v_new_1_5_chain/pull_and_verdict.log"
mkdir -p "$(dirname "${LOG}")"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*" | tee -a "${LOG}"; }

log "=== pull_and_verdict start ==="
log "Polling VPS every 3 min for v_new_1_phase3_v1_5/TRAINING_DONE.marker..."

while ! ssh "${VPS}" "test -f ${VPS_REPO}/services/python/results/v_new_1_phase3_v1_5/TRAINING_DONE.marker" 2>/dev/null; do
    sleep 180
done

log "v1_5 marker found. SCPing artifacts..."
mkdir -p "${REPO_ROOT}/services/python/models/v_new_1_long_v1_5" \
         "${REPO_ROOT}/services/python/models/v_new_1_short_v1_5" \
         "${REPO_ROOT}/services/python/data/v_new_1"

scp "${VPS}:${VPS_REPO}/services/python/models/v_new_1_long_v1_5/v_new_1_long_v1_5.joblib" \
    "${VPS}:${VPS_REPO}/services/python/models/v_new_1_long_v1_5/v_new_1_long_v1_5_meta.json" \
    "${REPO_ROOT}/services/python/models/v_new_1_long_v1_5/" 2>&1 | tee -a "${LOG}"

scp "${VPS}:${VPS_REPO}/services/python/models/v_new_1_short_v1_5/v_new_1_short_v1_5.joblib" \
    "${VPS}:${VPS_REPO}/services/python/models/v_new_1_short_v1_5/v_new_1_short_v1_5_meta.json" \
    "${REPO_ROOT}/services/python/models/v_new_1_short_v1_5/" 2>&1 | tee -a "${LOG}"

# Also pull baseline retrain meta JSONs (CV-fixed)
scp "${VPS}:${VPS_REPO}/services/python/models/v_new_1_long/v_new_1_long.joblib" \
    "${VPS}:${VPS_REPO}/services/python/models/v_new_1_long/v_new_1_long_meta.json" \
    "${REPO_ROOT}/services/python/models/v_new_1_long/" 2>&1 | tee -a "${LOG}"
scp "${VPS}:${VPS_REPO}/services/python/models/v_new_1_short/v_new_1_short.joblib" \
    "${VPS}:${VPS_REPO}/services/python/models/v_new_1_short/v_new_1_short_meta.json" \
    "${REPO_ROOT}/services/python/models/v_new_1_short/" 2>&1 | tee -a "${LOG}"

# Scored parquets (both v1_5 and baseline-retrained)
for f in oos_scored_long.parquet oos_scored_short.parquet \
         oof_scored_long.parquet oof_scored_short.parquet \
         cal_scored_long.parquet cal_scored_short.parquet \
         oos_scored_long_v1_5.parquet oos_scored_short_v1_5.parquet \
         oof_scored_long_v1_5.parquet oof_scored_short_v1_5.parquet \
         cal_scored_long_v1_5.parquet cal_scored_short_v1_5.parquet \
         features_full_v1_5.parquet ; do
    scp "${VPS}:${VPS_REPO}/services/python/data/v_new_1/${f}" \
        "${REPO_ROOT}/services/python/data/v_new_1/" 2>&1 | tee -a "${LOG}" || \
        log "  WARNING: ${f} not found on VPS"
done

log "Running verdict.py..."
DYLD_LIBRARY_PATH=$("${PY}" -c "import sklearn,os; print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))") \
    "${PY}" "${REPO_ROOT}/services/python/scripts/v_new_1_5_verdict.py" 2>&1 | tee -a "${LOG}"

today=$(date -u +%Y-%m-%d)
verdict="${REPO_ROOT}/docs/V_NEW_1_5_VERDICT-${today}.md"
log "=== DONE ==="
log "Verdict at: ${verdict}"
echo
echo "+++++ V_NEW_1_5 VERDICT +++++"
[[ -f "${verdict}" ]] && cat "${verdict}" || echo "(verdict file missing)"
