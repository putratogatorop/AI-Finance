#!/usr/bin/env bash
# Mac-side autopilot. Polls VPS for the v2 chain completion marker
# (v_new_1_short_v1_5 meta JSON), then:
#   1. SCPs v2 artifacts back to Mac
#   2. Runs v_new_1_5_verdict.py
#   3. Runs v_new_1_phase4_ceiling_find_v2.py twice (baseline + v1_5 model)
#   4. Prints full verdict trail
set -euo pipefail

REPO_ROOT="/Users/putratogatorop/Desktop/AI Finance"
VPS="ai-finance-deploy@103.103.20.30"
VPS_REPO="/opt/ai-finance"
PY="${REPO_ROOT}/services/python/.venv/bin/python3"
LOG_DIR="${REPO_ROOT}/services/python/results/v_new_1_v2_chain"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/autopilot.log"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*" | tee -a "${LOG}"; }

log "=== v2 autopilot start ==="
log "Polling VPS every 5 min for v_new_1_short_v1_5_meta.json..."

while ! ssh "${VPS}" "test -f ${VPS_REPO}/services/python/models/v_new_1_short_v1_5/v_new_1_short_v1_5_meta.json" 2>/dev/null; do
    sleep 300
done

log "v2 chain complete on VPS. Pulling artifacts..."

mkdir -p "${REPO_ROOT}/services/python/models/v_new_1_long" \
         "${REPO_ROOT}/services/python/models/v_new_1_short" \
         "${REPO_ROOT}/services/python/models/v_new_1_long_v1_5" \
         "${REPO_ROOT}/services/python/models/v_new_1_short_v1_5" \
         "${REPO_ROOT}/services/python/data/v_new_1_v2"

scp "${VPS}:${VPS_REPO}/services/python/models/v_new_1_long/"v_new_1_long{.joblib,_meta.json} \
    "${REPO_ROOT}/services/python/models/v_new_1_long/" 2>&1 | tee -a "${LOG}" || true
scp "${VPS}:${VPS_REPO}/services/python/models/v_new_1_short/"v_new_1_short{.joblib,_meta.json} \
    "${REPO_ROOT}/services/python/models/v_new_1_short/" 2>&1 | tee -a "${LOG}" || true
scp "${VPS}:${VPS_REPO}/services/python/models/v_new_1_long_v1_5/"v_new_1_long_v1_5{.joblib,_meta.json} \
    "${REPO_ROOT}/services/python/models/v_new_1_long_v1_5/" 2>&1 | tee -a "${LOG}" || true
scp "${VPS}:${VPS_REPO}/services/python/models/v_new_1_short_v1_5/"v_new_1_short_v1_5{.joblib,_meta.json} \
    "${REPO_ROOT}/services/python/models/v_new_1_short_v1_5/" 2>&1 | tee -a "${LOG}" || true

# Pull every parquet under data/v_new_1_v2/ that the local doesn't already have
for f in oos_scored_long.parquet oos_scored_short.parquet \
         oof_scored_long.parquet oof_scored_short.parquet \
         cal_scored_long.parquet cal_scored_short.parquet \
         oos_scored_long_v1_5.parquet oos_scored_short_v1_5.parquet \
         oof_scored_long_v1_5.parquet oof_scored_short_v1_5.parquet \
         cal_scored_long_v1_5.parquet cal_scored_short_v1_5.parquet \
         features_full_v1_5.parquet ; do
    scp "${VPS}:${VPS_REPO}/services/python/data/v_new_1_v2/${f}" \
        "${REPO_ROOT}/services/python/data/v_new_1_v2/" 2>&1 | tee -a "${LOG}" || \
        log "  WARNING: ${f} missing on VPS"
done

# 2. Verdict
log "Running v_new_1_5_verdict.py (baseline vs v_new_1.5 OOS comparison)..."
DYLD_LIBRARY_PATH=$("${PY}" -c "import sklearn,os; print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))") \
    "${PY}" "${REPO_ROOT}/services/python/scripts/v_new_1_5_verdict.py" 2>&1 | tee -a "${LOG}"

# 3a. Ceiling-find on baseline model
log "Running ceiling-find sweep on BASELINE (v_new_1)..."
DYLD_LIBRARY_PATH=$("${PY}" -c "import sklearn,os; print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))") \
V_NEW_1_DATA_DIR="${REPO_ROOT}/services/python/data/v_new_1_v2" \
MODEL_VARIANT="" \
N_TRIALS=200 \
    "${PY}" "${REPO_ROOT}/services/python/scripts/v_new_1_phase4_ceiling_find_v2.py" 2>&1 | tee -a "${LOG}"

# 3b. Ceiling-find on v_new_1.5 model
log "Running ceiling-find sweep on V_NEW_1.5 (sequence features)..."
DYLD_LIBRARY_PATH=$("${PY}" -c "import sklearn,os; print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))") \
V_NEW_1_DATA_DIR="${REPO_ROOT}/services/python/data/v_new_1_v2" \
MODEL_VARIANT="_v1_5" \
N_TRIALS=200 \
    "${PY}" "${REPO_ROOT}/services/python/scripts/v_new_1_phase4_ceiling_find_v2.py" 2>&1 | tee -a "${LOG}"

# 4. Print verdicts side-by-side
log "=== AUTOPILOT DONE ==="
today=$(date -u +%Y-%m-%d)
echo
echo "+++++ V_NEW_1.5 VS V_NEW_1 OOS VERDICT +++++"
[[ -f "${REPO_ROOT}/docs/V_NEW_1_5_VERDICT-${today}.md" ]] && cat "${REPO_ROOT}/docs/V_NEW_1_5_VERDICT-${today}.md" || echo "(verdict file missing)"
echo
echo "+++++ CEILING-FIND BASELINE VERDICT +++++"
[[ -f "${REPO_ROOT}/services/python/results/v_new_1_phase4_ceiling_find_v2/verdict.md" ]] && \
    cat "${REPO_ROOT}/services/python/results/v_new_1_phase4_ceiling_find_v2/verdict.md" || \
    echo "(baseline verdict missing)"
echo
echo "+++++ CEILING-FIND V_NEW_1.5 VERDICT +++++"
[[ -f "${REPO_ROOT}/services/python/results/v_new_1_phase4_ceiling_find_v2/verdict_v1_5.md" ]] && \
    cat "${REPO_ROOT}/services/python/results/v_new_1_phase4_ceiling_find_v2/verdict_v1_5.md" || \
    echo "(v1_5 verdict missing)"
