# MACRO Track B — non-ML levers, parallel to v_new_1.5

**Date:** 2026-05-04
**Status:** Plan locked, executing alongside v_new_1.5
**Goal:** Identify and execute the 1-2 cheapest big-EV macro changes that can lift 4-year compounded return from 206% toward 300-400% **without breaking the -20% per-year DD cap**.
**Constraint reminder:** User locked **-20% max DD per year**. Higher leverage (10×) is therefore OFF the table — it would near-certainly blow that cap. Macro levers limited to those that improve return *without* expanding loss budget.

---

## Lever inventory under -20% DD lock

| Lever | Effect on return | Effect on DD | Cost (engineering) | Cost (compute) | Verdict |
|---|---|---|---|---|---|
| **B1 — Top-200 universe** (vs top-100) | + (more big-mover events) | ≈ neutral or − (better diversification) | 1-2 days (Phase 1+2 redo with extended universe) | 2× memory, 2× train time | **YES — first priority** |
| **B2 — Concentrated sizing within DD budget** | + (bet bigger on highest cls_score, smaller on borderline) | ≈ neutral if DD enforced at sizing layer | 0.5-1 day (Phase 4 policy change only) | <30 min (Optuna rerun) | **YES — runs in parallel** |
| **B3 — Higher-freq 1h candles** | ++ (catch moves earlier) | uncertain | 3-5 days (full feature recompute) | 4× data, 4× compute, RAM-tight on 8 GB | DEFER — wait for 16 GB upgrade |
| **B4 — Backfill 2017-2019** | minor (more OOS regimes) | ≈ neutral | 2-3 days (fetch + feature compute on early-crypto OHLCV) | + ~150 MB data | DEFER — pre-2017 markets have weird microstructure |
| **B5 — Variable-horizon labels** | uncertain (ML benefit) | uncertain | 1-2 days (label regen + retrain) | 2× train | DEFER — entangles with v_new_1.5 verdict |
| **B6 — Higher leverage 10×** | ++ | ++ DD (BLOWS −20% cap) | trivial | none | **OFF — DD cap forbids** |

**Active in this plan: B1 + B2.** Everything else deferred until v_new_1.5 verdict and the 16 GB RAM upgrade.

---

## Track B1 — Top-200 universe expansion

**Why first:** Crypto big-mover events are dominated by the long tail beyond the top-100. Empirical observation from `big_movers` table (38,068 records): of those, a substantial fraction occur in symbols ranked 100-200 by 30-day median dollar volume. Doubling the universe roughly doubles candidate events without doubling DD (the events are uncorrelated cross-symbol).

**Steps:**
1. Modify `v_new_1_phase1_2_pipeline.py` to widen `UNIVERSE_TOP_N` from 100 → 200.
2. Re-run Phase 1+2 on VPS to regenerate `features_full.parquet` and labels parquets with the larger universe. Estimated 1.2M → 2.4M rows. ~3-4h wall-clock at `LGBM_NUM_THREADS=4`.
3. Re-run Phase 3 with the larger universe → `v_new_1_top200_long`, `v_new_1_top200_short`. ~3h wall-clock.
4. Re-run Phase 4 policy opt → 4-yr return number. ~30 min.
5. Verdict in `docs/V_NEW_1_TOP200_VERDICT-{date}.md`. Compare 4-yr return + max-DD against top-100 baseline.

**Memory check before launch:** at 2.4M × 56-feature parquet, raw load is ~520 MB. With 8 GB RAM and live trading services running, this is tight. **Defer Phase 1+2 regeneration until 16 GB RAM upgrade lands**, OR run with column-selective loading (which the existing script already does — should fit in 6 GB peak).

**Promotion gate:** 4-yr return ≥ 250% AND max yearly DD ≥ -20% across all 4 OOS years.

---

## Track B2 — Concentrated sizing within DD budget

**Why second:** Phase 4 currently uses fixed `position_size_pct=0.518%` for every trade regardless of conviction. The cls_score distribution is right-skewed; the top decile of scores has materially higher per-trade EV. Concentrating size on those without expanding total notional preserves the DD budget but lifts return.

**Mechanism (already validated in legacy v6 ATR-sizing memory):** scale per-trade size by cls_score band:
- `cls_score ∈ [thr, p50]` → 0.5× base size
- `cls_score ∈ [p50, p90]` → 1.0× base size (current default)
- `cls_score > p90` → 1.5-2× base size
- Total open notional capped at the same ceiling as today (so DD budget unchanged)

**Steps:**
1. Add `position_size_multiplier` rules to `v_new_1_phase4_policy_opt.py` Optuna search:
   - `size_mult_low_band` ∈ [0.3, 0.7]
   - `size_mult_high_band` ∈ [1.2, 2.0]
   - Plus the existing `position_size_pct` for the mid band
2. Add a hard concurrent-notional cap that mirrors today's max simultaneous deployment.
3. Rerun Phase 4 Optuna (100 trials) with these added knobs. Same DD gate (-20% per year).
4. Verdict in `docs/V_NEW_1_CONCENTRATED_SIZING_VERDICT-{date}.md`.

**Compute budget:** runs entirely on Mac. <30 min for Optuna 100 trials.

**Promotion gate:** 4-yr return ≥ 250% AND max yearly DD ≥ -20% AND no individual OOS year worse than baseline by more than 3pp.

---

## Sequencing with v_new_1.5

```
Day 0 (today)        v_new_1.5 patch + scripts (Mac)        ──┐
                     B2 concentrated sizing scripts (Mac)     │
                                                              │
Day 1                VPS rerun #1: v_new_1 with OOF (2h)      │
                     v_new_1.5 augment features (Mac)         │
                     B2 Optuna rerun (Mac, 30 min)            │
                                                              │
Day 1-2              VPS rerun #2: v_new_1.5 train (2h)       │
                     B2 verdict committed                     │
                                                              │
Day 2                v_new_1.5 verdict committed              │
                                                              │
                     === decision point ===                   │
                                                              │
Day 3+               If v_new_1.5 lift ≥ +5%:                 │
                       Phase 4 rerun w/ v_new_1.5 scores      │
                       Then: B1 top-200 (after 16 GB upgrade) │
                     Else:                                    │
                       Skip LSTM. Go straight to B1 top-200.  │
```

B2 (concentrated sizing) runs entirely in parallel and doesn't block anything else.

---

## What we're NOT doing in this track

- 10× leverage. Forbidden by the -20% DD cap.
- 1h candles. Wait for 16 GB RAM.
- 2017-2019 backfill. Low-EV vs the engineering cost.
- Options / leveraged ETFs / on-chain flows. Outside scope of this re-plan.
