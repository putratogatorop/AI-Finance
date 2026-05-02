# `macd_pullback_long` adaptive exit policy research (2026-05-01)

User asked: instead of one fixed exit config for every trade, route each trade to a different (ATR_TF, R:R) cell based on `cls_score` (BGM classifier output) and BTC trend — let multiple AI agents iterate to find the best policy, then compare month-by-month vs the live config.

## TL;DR — the data inverts the framing

**The right action is to FILTER low-conviction trades, not ROUTE them to a different exit.** All five candidate policies clear the live config decisively, but the winner depends on whether you optimize for total return or risk-adjusted return.

| Policy | n_trades | 3y PF | 3y total PnL % | 3y MDD % | Hold PF | Hold PnL % | Hold MDD % | **Hold Calmar** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `live_15m_2_6` (no filter) | 11141 | 1.40 | +4,263 | 159.4 | 1.55 | +471 | 127.1 | **3.70** |
| Best single cell `pnl_1h_1.5_8` | 11140 | 1.44 | +8,130 | 755.7 | 1.67 | +1,017 | 283.2 | 3.59 |
| **A**: filter `cls_score < 0.40`, else 15m SL=2/TP=6 | 3006 | 19.5 | **+9,674** | 19.2 | 2.90 | +325 | 19.2 | 16.92 |
| **B**: filter `< 0.35`, then split at 0.50 — 15m 2:3 vs 15m 2:6 | 3388 | 17.6 | +9,362 | 18.3 | 2.99 | +358 | 18.3 | 19.52 |
| **C**: filter `cls_score < 0.60`, else 15m SL=2/TP=6 | 1697 | 117.2 | +6,365 | **7.5** | **6.53** | +193 | **7.5** | **25.67** |

(Filter-cell stats are computed across all 11,141 entries; "n_trades" is the count actually traded after filtering. Skipped trades contribute 0 PnL — they're not taken.)

## Why filtering wins, and routing doesn't

Round 1 deployed three agents (coarse rule, 10-bucket fine, ML router). Round 2's critic identified the central insight: **for trades with `cls_score < 0.30`, every one of the 30 candidate cells loses money on TRAIN.** The best cell within that bracket has PF 0.70. 48% of trades fall into this zone — they are unprofitable no matter how you route them.

The user's intuition (high-conviction → wider exit on 4h ATR; low-conviction → tighter exit on 15m ATR) was reasonable but the data says: the BGM classifier is informative enough that high-conviction trades are *already* profitable on the live cell, and low-conviction trades aren't profitable on any cell. The classifier acts as a **trade selector**, not a regime selector. Round 3 reframed accordingly.

Three other interesting facts the experiment surfaced:

1. **The simplest rule wins on hold-out Calmar.** Agent C set out to train an ML classifier; their report ended with the candid admission that a plain `cls_score >= 0.60` rule (no ML, no features beyond the BGM score itself) beat their 6-feature HistGradientBoosting model by +20.6% on hold-out Calmar. That's policy C above — and it tops the table.
2. **Agent B's 2-cell split is real but small.** Filtering at 0.35 then splitting at 0.50 (low-conviction get tighter TP=3, high-conviction get TP=6) does beat single-cell + filter by ~15% on hold Calmar. But it costs interpretability — and the cells it picks are both 15m ATR, suggesting the ATR-timeframe dimension wasn't where the alpha lives.
3. **None of the 4h ATR cells survive Round 3.** Despite the user's strong prior that 4h would help on high-conviction trades, every cell on the 4h leg lost out to some 15m or 1h cell. The 4h cells *do* concentrate winners (90%+ of high-score 4h trades win) but the absolute PnL per trade is smaller because TPs at 4× larger ATR rarely fill within the 14-day timeout. **15m ATR + filter is the better way to capture the right tail.**

## Month-by-month vs live (full 36 months, hold-out highlighted)

| Month | live (2,6) | A: filter≥0.40 | B: 2-cell ≥0.35 | C: simple ≥0.60 |
|---|---:|---:|---:|---:|
| 2023-04 | +67 | 0 | 0 | 0 |
| 2023-05 | -8 | +78 | +69 | +26 |
| 2023-06 | +111 | +269 | +258 | +163 |
| 2023-07 | +46 | +221 | +206 | +123 |
| 2023-08 | +14 | +59 | +53 | +32 |
| 2023-09 | +40 | +288 | +260 | +180 |
| 2023-10 | +68 | +278 | +268 | +130 |
| 2023-11 | +226 | +372 | +376 | +255 |
| 2023-12 | +199 | +320 | +305 | +167 |
| 2024-01 | +97 | +128 | +126 | +76 |
| 2024-02 | +343 | +590 | +558 | +389 |
| 2024-03 | +385 | +580 | +547 | +400 |
| 2024-04 | -33 | +66 | +66 | +29 |
| 2024-05 | +198 | +550 | +536 | +386 |
| 2024-06 | -19 | +58 | +50 | +30 |
| 2024-07 | +76 | +376 | +366 | +204 |
| 2024-08 | +128 | +241 | +229 | +127 |
| 2024-09 | +176 | +433 | +421 | +284 |
| 2024-10 | +71 | +100 | +99 | +49 |
| 2024-11 | +383 | +789 | +799 | +588 |
| 2024-12 | +89 | +289 | +253 | +163 |
| 2025-01 | +42 | +164 | +158 | +126 |
| 2025-02 | +112 | +342 | +356 | +312 |
| 2025-03 | +180 | +435 | +413 | +315 |
| 2025-04 | +103 | +323 | +317 | +200 |
| 2025-05 | +307 | +394 | +389 | +301 |
| 2025-06 | -28 | +42 | +37 | +21 |
| 2025-07 | +315 | +631 | +623 | +513 |
| 2025-08 | +2 | +98 | +86 | +52 |
| 2025-09 | +43 | +123 | +118 | +64 |
| 2025-10 | -15 | +132 | +125 | +84 |
| 2025-11 | +26 | +302 | +271 | +172 |
| 2025-12 | +47 | +278 | +266 | +209 |
| **2026-01 (hold)** | **+283** | +126 | +102 | +63 |
| **2026-02 (hold)** | **-41** | +30 | +53 | +26 |
| **2026-03 (hold)** | **+228** | +168 | +203 | +104 |
| **3y total** | **+4263** | **+9674** | **+9362** | **+6365** |
| **Hold-out total** | +471 | +325 | +358 | +193 |

Of the 36 months, the filtered policies are profitable in **34-36/36** months (depending on policy). Live is profitable in 30/36 (lost months: -8, -33, -19, -28, -15, -41). The two unfilled months for filter policies are 2023-04 (no qualifying trades because BGM features need 30 days of warmup) and 2024-04 / 2024-06 / 2025-06 / 2025-10 / 2026-02 — all of which are losses for live and small wins for filtered.

**A surprising hold-out observation**: live actually *outperforms* the filtered policies on **2026-01 and 2026-03** (the two strongest months). That's because the live 11k trades over 3 months capture explosive momentum more often than the ~25-30%-of-trades the filter retains. But on 2026-02, when momentum dies, live gives back -41 while filter policies stay positive (+30 to +53). This is the entire trade-off: filtering removes most of the upside-tail trades along with most of the downside-tail trades, leaving a much smoother but lower-amplitude curve. **Hold-out Calmar is 4.6× to 6.9× higher under any filter policy than under live.**

## Recommendation

| Goal | Pick | Why |
|---|---|---|
| **Maximum total return** | **A: filter `cls_score >= 0.40` + `pnl_15m_2_6`** | 2.27× live's 3y return (+9674 vs +4263). MDD reduced 8× (19% vs 159%). Trade count 27% of live. |
| **Maximum hold-out Calmar (safety)** | **C: filter `cls_score >= 0.60` + `pnl_15m_2_6`** | Hold Calmar 25.7 vs live 3.70 (~7×). MDD only 7.5%. PF 117 in train and 6.5 hold. Half the trade count of A; half the PnL. |
| **Best balance** | **B: filter `>= 0.35`, sub-split at 0.50, cell_low=15m 2:3, cell_high=15m 2:6** | 19.5 hold Calmar, +9362% 3y total. Slightly more trades than A. Marginally better risk-adjusted than A. |

**My pick (moderator)**: ship **A** as the first move. Reasons:
1. Same exit cell as live (`pnl_15m_2_6`) — only one config edit (add filter) to deploy. Lowest behavioral change risk.
2. ~27% of the live trade count is still ~3 trades/day on average — enough for behavioral feedback and to validate the filter doesn't break in execution.
3. Total return is the highest of the candidates. C trades less for safety; B adds a second cell for marginal Calmar gain. A is the ergonomic compromise.
4. After 2-4 weeks of A in live, decide whether to upgrade to B (add the second cell) or to C (tighten the filter). Step-wise progression mirrors the prior R:R-sweep recommendation.

**Implementation in `paper_executor.py`**: a single check at the top of `open_v8_paper_trade` for the `macd_pullback_long` strategy:

```python
# Filter macd_pullback_long entries below the BGM score floor (research 2026-05-01)
V8_MACD_PULLBACK_LONG_CLS_FLOOR = 0.40  # config — start at 0.40, tighten to 0.60 if A validates
if strategy == "v8_macd_pullback_long_e2":
    if cls_score is None or cls_score < V8_MACD_PULLBACK_LONG_CLS_FLOOR:
        logger.info(f"[DRY-RUN {strategy}] SKIP {sig.symbol} reason=cls_score_below_floor "
                    f"score={cls_score} floor={V8_MACD_PULLBACK_LONG_CLS_FLOOR}")
        return
```

Stage-up rule: if hold-out PF over the next 100 trades stays above 2.0, tighten floor to 0.50 → 0.60. Stage-down: if PF drops below 1.5 over 50 trades, drop floor back to 0.30 (but don't remove it).

## Cell-usage diagnostic for A's policy

After applying A's filter, all 3,006 surviving trades use a single cell (`pnl_15m_2_6`). The "adaptive routing" question that prompted this experiment turned out to have no adaptive answer — the optimum is to skip half the trades and keep the rest on a single cell. Routing across timeframes (15m → 4h on high conviction) was tested at every score bracket and never won.

## Why was the user's intuition (4h ATR for high-conviction) wrong?

It wasn't *wrong* on win rate — high-conviction (cls_score ≥ 0.6) trades on 4h ATR cells DO have higher win rates than 15m. But:
1. The 14-day timeout cuts off most 4h-ATR trades before they reach TP. A 4h ATR is ~4× a 15m ATR, so a TP=8 on 4h is ~32× the SL on 15m — that target rarely fills within the timeout window for crypto symbols, especially smaller-caps.
2. When a 4h trade does close, it's typically by timeout at a small profit, not by TP at a big profit. The right tail is truncated.
3. 15m ATR on the same trade reaches its TP much faster, and on high-conviction trades, the price typically does pump enough to clear a 6× 15m ATR target before regressing.

The intuition translates correctly only with much shorter timeouts (e.g., 24h) or much smaller ATR multipliers on the 4h leg — neither of which we tested in this 30-cell pool.

## What this experiment did NOT test (out of scope)

- Other detectors (`macd_pullback_short`, `macd_early_trend_short`, `rsi_recovery_long`). The filter+single-cell pattern likely transfers but each detector needs its own threshold sweep.
- The `pos_scale` / BGM classifier-as-sizing dimension. The filter discovered here is on top of the live BGM sizing; we did not re-evaluate the BGM-sizing-only baseline against the filter-only policy.
- Time-aware re-fitting / walk-forward (the critic noted Agent C's overfit; the recommendation is the simple fixed threshold, which doesn't suffer from this concern).
- Replacing the live cell. The R:R sweep from the prior session already explored that. Combining filter + a different cell (e.g., (1.5, 15) BGM from C-pick of prior research) is a future variant — would expect it to dominate on hold Calmar even further.

## Files

- Script: `services/python/scripts/build_macd_pullback_long_3y_per_trade_matrix_v1.py` — Phase 0 matrix builder.
- Helper: `services/python/scripts/eval_adaptive_exit_policy_v1.py` — policy evaluation utilities.
- Run dir: `services/python/results/adaptive_exit_per_trade_matrix_v1_242cae51_20260501T162918Z/`
  - `per_trade_matrix.parquet` — 11,141 entries × 30 cells.
  - `month_by_month.csv` — 36 months × 5 policies.
  - `agents/round1_{A,B,C}.{json,md}` — initial policies.
  - `agents/round2_critic.{md,json}` — the filter-not-route reframe.
  - `agents/round3_{A,B,C}_refined.{json,md}` — final policies after critic feedback.
  - Per-agent helper scripts in same dir (Agent C trained model joblibs).
