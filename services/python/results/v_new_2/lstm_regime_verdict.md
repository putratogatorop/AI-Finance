# v_new_2 LSTM Regime Classifier — OOS Verdict (2026-05-06)

Train < 2025-01-01  |  Val 2025-01-01–2026-01-01  |  Test (OOS) ≥ 2026-01-01

## Accuracy

| Split | Acc | Macro-F1 |
|---|---|---|
| Val  | 0.601 | 0.553 |
| Test | 0.553 | 0.321 |

## Per-class (OOS Test)

| Class | P | R | F1 | n |
|---|---|---|---|---|
| bear | 0.996 | 0.528 | 0.690 | 906 |
| neutral | 0.166 | 0.788 | 0.274 | 99 |
| bull_mania | 0.000 | 0.000 | 0.000 | 0 |

## Regime-conditional PF (OOS)

| Filter | n | PF |
|---|---|---|
| All trades     | 1005 | 4.363 |
| bull_mania     | 55 | 4.026 |
| non-bear       | 525 | 5.351 |

## Promotion gate

- Val Macro-F1 ≥ 0.40: **PASS** (0.553)
- Non-bear PF > all-trades PF (OOS): **PASS** (5.351 vs 4.363)
- Bull_mania OOS note: test period was 90% Fear (CFGI<26) — bull_mania gate deferred until next alt-mania period

**Overall: PROMOTE — integrate into scanner**
