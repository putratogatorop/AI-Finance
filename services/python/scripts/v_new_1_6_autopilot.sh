#!/usr/bin/env bash
# v_new_1.6 autopilot — runs after v_new_1.5 verdict lands.
#
# Steps:
#   1. Wait for the v2 autopilot to finish (V_NEW_1_5_VERDICT-{date}.md exists)
#   2. Read v_new_1.5 verdict; decide base = v1_5 if any direction passed +5%
#      lift, else base = v1 (use v_new_1 baseline)
#   3. Build features_full_v1_6.parquet on Mac via v_new_1_6_merge_features.py
#   4. SCP features_full_v1_6.parquet to VPS
#   5. Train v_new_1_long_v1_6 + v_new_1_short_v1_6 on VPS in tmux
#   6. Wait for v_new_1_short_v1_6_meta.json
#   7. SCP v1_6 artifacts back to Mac
#   8. Run v_new_1_5_verdict.py-style comparison: v1_6 vs winning base
#   9. Run ceiling-find sweep with v1_6 model (MODEL_VARIANT=_v1_6)
#  10. Print final triple verdict (v1_5 OOS, v1_6 OOS, v1_6 ceiling)
set -euo pipefail

REPO_ROOT="/Users/putratogatorop/Desktop/AI Finance"
VPS="ai-finance-deploy@103.103.20.30"
VPS_REPO="/opt/ai-finance"
PY="${REPO_ROOT}/services/python/.venv/bin/python3"
LOG_DIR="${REPO_ROOT}/services/python/results/v_new_1_v2_chain"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/v1_6_autopilot.log"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*" | tee -a "${LOG}"; }

log "=== v_new_1.6 autopilot start ==="
log "Polling every 5 min for V_NEW_1_5_VERDICT (Mac-side, written by v2_autopilot)..."

while true; do
    found=$(find "${REPO_ROOT}/docs" -name "V_NEW_1_5_VERDICT-*.md" 2>/dev/null | head -1)
    if [[ -n "${found}" ]]; then
        log "v_new_1.5 verdict found: ${found}"
        VERDICT="${found}"
        break
    fi
    sleep 300
done

# Decide base from verdict file. The verdict markdown has two headline lines:
#   - **LONG:** {emoji} PASS|MARGINAL|FAIL ...
#   - **SHORT:** ...
# 🟢 PASS = v1_5 lift ≥ +5% on that direction. If either direction passes, use v1_5 base.
log "Reading verdict to decide v_new_1.6 base..."
if grep -E "^- \*\*(LONG|SHORT):\*\* 🟢 PASS" "${VERDICT}" >/dev/null 2>&1; then
    BASE="v1_5"
    log "  At least one direction passed +5% on v_new_1.5 → BASE = v1_5"
else
    BASE="v1"
    log "  Neither direction cleared +5% → BASE = v1 (skip sequence features)"
fi

# 3. Merge on Mac
log "Step 3/9: merging extras onto ${BASE} base..."
DYLD_LIBRARY_PATH=$("${PY}" -c "import sklearn,os; print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))") \
    "${PY}" "${REPO_ROOT}/services/python/scripts/v_new_1_6_merge_features.py" --base "${BASE}" 2>&1 | tee -a "${LOG}"

MERGED="${REPO_ROOT}/services/python/data/v_new_1_v2/features_full_v1_6.parquet"
if [[ ! -f "${MERGED}" ]]; then
    log "ERROR: merge did not produce ${MERGED}. Aborting."
    exit 1
fi

# 4. SCP to VPS
log "Step 4/9: SCPing features_full_v1_6.parquet to VPS..."
scp "${MERGED}" "${VPS}:${VPS_REPO}/services/python/data/v_new_1_v2/" 2>&1 | tee -a "${LOG}"

# 5. Launch VPS training
log "Step 5/9: launching v_new_1.6 training on VPS in tmux..."
WITH_SEQ_FLAG=$([[ "${BASE}" = "v1_5" ]] && echo "1" || echo "0")
ssh "${VPS}" "tmux new-session -d -s v1_6_train 'cd ${VPS_REPO} && \
    V_NEW_1_DATA_DIR=${VPS_REPO}/services/python/data/v_new_1_v2 \
    V_NEW_1_TRAIN_FIT_END=2024-12-31 V_NEW_1_CAL_FIT_START=2025-01-01 \
    V_NEW_1_OOS_YEARS=2025_h2,2026_q1 \
    V_NEW_1_6_MODE=1 V_NEW_1_6_WITH_SEQUENCE=${WITH_SEQ_FLAG} \
    N_OPTUNA_TRIALS=50 LGBM_NUM_THREADS=4 \
    services/python/.venv/bin/python3 -u services/python/scripts/v_new_1_phase3_train.py \
    > services/python/results/v_new_1_v2_chain/v1_6_train.log 2>&1; \
    echo V1_6_TRAIN_DONE'"
log "  v1_6_train tmux session started"

# 6. Poll for completion
log "Step 6/9: polling for v_new_1_short_v1_6_meta.json..."
while ! ssh "${VPS}" "test -f ${VPS_REPO}/services/python/models/v_new_1_short_v1_6/v_new_1_short_v1_6_meta.json" 2>/dev/null; do
    sleep 300
done
log "  v_new_1.6 training complete on VPS"

# 7. SCP back
log "Step 7/9: SCPing v1_6 artifacts back to Mac..."
mkdir -p "${REPO_ROOT}/services/python/models/v_new_1_long_v1_6" \
         "${REPO_ROOT}/services/python/models/v_new_1_short_v1_6"
scp "${VPS}:${VPS_REPO}/services/python/models/v_new_1_long_v1_6/"v_new_1_long_v1_6{.joblib,_meta.json} \
    "${REPO_ROOT}/services/python/models/v_new_1_long_v1_6/" 2>&1 | tee -a "${LOG}" || true
scp "${VPS}:${VPS_REPO}/services/python/models/v_new_1_short_v1_6/"v_new_1_short_v1_6{.joblib,_meta.json} \
    "${REPO_ROOT}/services/python/models/v_new_1_short_v1_6/" 2>&1 | tee -a "${LOG}" || true
for f in oos_scored_long_v1_6.parquet oos_scored_short_v1_6.parquet \
         oof_scored_long_v1_6.parquet oof_scored_short_v1_6.parquet \
         cal_scored_long_v1_6.parquet cal_scored_short_v1_6.parquet ; do
    scp "${VPS}:${VPS_REPO}/services/python/data/v_new_1_v2/${f}" \
        "${REPO_ROOT}/services/python/data/v_new_1_v2/" 2>&1 | tee -a "${LOG}" || \
        log "  WARNING: ${f} missing on VPS"
done

# 8. v_new_1.6 verdict comparison
log "Step 8/9: writing v_new_1.6 verdict comparison..."
DYLD_LIBRARY_PATH=$("${PY}" -c "import sklearn,os; print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))") \
    "${PY}" -c "
import json, pathlib, time
ROOT = pathlib.Path('${REPO_ROOT}')
def meta(d, s): return json.load(open(ROOT/'services/python/models'/(f'v_new_1_{d}{s}')/(f'v_new_1_{d}{s}_meta.json')))
base_long = meta('long', '_v1_5' if '${BASE}'=='v1_5' else '')
base_short = meta('short', '_v1_5' if '${BASE}'=='v1_5' else '')
new_long = meta('long', '_v1_6')
new_short = meta('short', '_v1_6')
def pr(m): return m['aggregate_oos_metrics']['median_pr_auc']
print()
print('=' * 60)
print(f'v_new_1.6 vs ${BASE} base — OOS comparison')
print('=' * 60)
print(f'  LONG:  {pr(base_long):.4f} → {pr(new_long):.4f}  '
      f'({(pr(new_long)-pr(base_long))/pr(base_long)*100:+.1f}%)')
print(f'  SHORT: {pr(base_short):.4f} → {pr(new_short):.4f}  '
      f'({(pr(new_short)-pr(base_short))/pr(base_short)*100:+.1f}%)')
# SHAP — did narrative features rank?
narratives = ['cohort_top10_momentum_7d', 'cat_memes_momentum_7d',
              'cat_rwa_momentum_7d', 'cat_ai_tokens_momentum_7d',
              'cat_depin_momentum_7d', 'cat_gamefi_momentum_7d',
              'cat_defi_momentum_7d', 'cat_layer1_momentum_7d',
              'cat_layer2_momentum_7d', 'cat_solana_eco_momentum_7d',
              'cat_eth_eco_momentum_7d', 'cat_membership_count']
for d, m in [('long', new_long), ('short', new_short)]:
    shap = m.get('shap_importance_top20', {})
    in_top = [f for f in narratives if f in shap]
    print(f'  {d.upper()}: {len(in_top)}/{len(narratives)} narrative features in top-20 SHAP')
    for f in in_top:
        print(f'    {f}: {shap[f]}')
out = ROOT / 'docs' / f\"V_NEW_1_6_VERDICT-{time.strftime('%Y-%m-%d')}.md\"
out.write_text(f\"\"\"# v_new_1.6 Verdict — Narrative & Cohort Features

Base: {'${BASE}'}

## OOS PR-AUC

| direction | base | v_new_1.6 | delta |
|---|---|---|---|
| LONG  | {pr(base_long):.4f} | {pr(new_long):.4f} | {(pr(new_long)-pr(base_long))/pr(base_long)*100:+.1f}% |
| SHORT | {pr(base_short):.4f} | {pr(new_short):.4f} | {(pr(new_short)-pr(base_short))/pr(base_short)*100:+.1f}% |

## Narrative features in SHAP top-20

LONG:  {len([f for f in narratives if f in new_long.get('shap_importance_top20',{})])} / {len(narratives)}
SHORT: {len([f for f in narratives if f in new_short.get('shap_importance_top20',{})])} / {len(narratives)}

## Decision rule

- 🟢 lift ≥+5% on either direction → narrative features carry signal; promote v_new_1.6
- 🟡 +1% to +5% → marginal; investigate (categories alone vs cohort alone)
- 🔴 ≤+1% → narrative not extractable here; stick with base
\"\"\")
print(f'  Wrote {out}')
" 2>&1 | tee -a "${LOG}"

# 9. Ceiling-find with v1_6 model
log "Step 9/9: ceiling-find sweep with v_new_1.6 model..."
DYLD_LIBRARY_PATH=$("${PY}" -c "import sklearn,os; print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))") \
V_NEW_1_DATA_DIR="${REPO_ROOT}/services/python/data/v_new_1_v2" \
MODEL_VARIANT="_v1_6" \
N_TRIALS=200 \
    "${PY}" "${REPO_ROOT}/services/python/scripts/v_new_1_phase4_ceiling_find_v2.py" 2>&1 | tee -a "${LOG}"

log "=== v_new_1.6 autopilot DONE ==="

today=$(date -u +%Y-%m-%d)
echo
echo "+++++ V_NEW_1.6 OOS VERDICT +++++"
[[ -f "${REPO_ROOT}/docs/V_NEW_1_6_VERDICT-${today}.md" ]] && \
    cat "${REPO_ROOT}/docs/V_NEW_1_6_VERDICT-${today}.md" || \
    echo "(verdict file missing)"
echo
echo "+++++ V_NEW_1.6 CEILING-FIND VERDICT +++++"
[[ -f "${REPO_ROOT}/services/python/results/v_new_1_phase4_ceiling_find_v2/verdict_v1_6.md" ]] && \
    cat "${REPO_ROOT}/services/python/results/v_new_1_phase4_ceiling_find_v2/verdict_v1_6.md" || \
    echo "(ceiling-find verdict missing)"
