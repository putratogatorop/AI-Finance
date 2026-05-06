# v_new_2 Adaptive Exit Classifier — OOS Verdict (2026-05-06)

Supervised exit classifier trained on 587,030 bar-steps (oracle ATR trail).
Uses LSTM regime probs as state features for regime-awareness.

## OOS Test Metrics

| Metric | Value |
|---|---|
| Exit Precision | 0.414 |
| Exit Recall | 0.243 |
| Exit F1 | 0.306 |
| Oracle avg exit PnL | 7.310% |
| Classifier avg exit PnL | -4.916% |

## Gate

- Precision ≥ 0.35: **PASS** (0.414)
- Recall ≥ 0.25: **FAIL** (0.243)

**Overall: HOLD**
