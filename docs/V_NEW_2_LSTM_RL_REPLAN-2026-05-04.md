# v_new_2 — LSTM/RL Layer on top of v_new_1 — Re-planning

**Date:** 2026-05-04
**Status:** Design proposal awaiting user confirmation
**Goal:** Push 4-year compounded return from current 206% (v_new_1 + Phase 4 policy) toward 400-1000% per 4 years using sequence/policy modeling on top of v_new_1 BGM scores.

---

## TL;DR

Three flavors of "LSTM/RL on top of v_new_1" are technically possible. In honesty-order:

1. **LSTM-style sequence features INSIDE the BGM** (no actual LSTM) — cheap, CPU-friendly, +5-15% expected lift, 1-2 days work
2. **Small CPU LSTM as a stacker on v_new_1 scores** — moderate cost, marginal expected lift, 3-5 days work, M2 Pro/VPS-friendly with PyTorch
3. **RL policy network replacing Phase 4 Optuna** — heavy work, uncertain payoff without GPU, 2-3 weeks, would benefit from rented GPU (~$10-20)

The hard truth: **getting to 400-1000% per 4 years requires the macro path (better data, more leverage, aggressive sizing) more than ML sophistication.** A neural net on top of the same 600K rows + 4h candles + 7-day horizon has a structural ceiling. I want to be honest about that before we burn weeks on it.

---

## Where v_new_1 currently sits

| | |
|---|---|
| Architecture | LightGBM (BGM), 44 features, Platt-calibrated, both directions |
| Training data | 2023-04 to 2025-12 (3 years, ~600K rows after universe filter) |
| OOS validation | 4 years (2020 / 2021 / 2022 / 2026 Q1) |
| Phase 3 median OOS PR-AUC | 0.442 long, 0.488 short |
| Phase 4 best policy | Top-K filter + CFGI gates + 7-day timeout, NO SL/TP |
| Phase 4 4-year compounded | 206% (vs BTC buy-hold ~250%, vs current prod 96%) |
| Ceiling on this architecture | Estimated 250-350% with feature/sizing tweaks |

To hit user's 200-1000%/year stated target, need either:
- **5-10× more capital deployed per trade** (current 0.5% × 5× lev = 2.5% notional) — risk explodes
- **Different/higher-frequency data** (1m candles, on-chain flows)
- **Different strategy class** (parabolic capture, options, leveraged ETFs)
- **NOT just a smarter classifier on the same data**

So whatever we add (LSTM/RL/stacking) needs to be paired with one of those macro changes for transformational returns.

---

## Architecture proposals

### Option A — Sequence features inside BGM (no actual LSTM)

**What it is:** Add lagged v_new_1 scores and rolling statistics as features to the existing BGM training:
- `score_long_t-1` (4h ago), `score_long_t-2` (8h ago), `score_long_t-6` (1d ago)
- `score_long_rolling_mean_3d`, `score_long_rolling_std_3d`
- `score_long_change_1d` (momentum of the score itself)
- Same for short

Tree boosters capture sequence patterns through splits when given lagged features. This is "sequence modeling without LSTM."

**Cost:** ~1-2 days. Just Phase 1+2+3 redo with augmented features.

**Expected lift:** +5-15% over current (so 220-240% 4-yr).

**Pros:**
- M2 Pro/VPS friendly, no GPU
- Reuses entire existing pipeline
- Tree boosters naturally handle the lagged features
- Low risk

**Cons:**
- Same model class — BGM has known ceiling
- Lift bounded; won't unlock 400%+ alone

### Option B — Small CPU LSTM as score-stacker

**What it is:** Train a small LSTM (1-2 layers, 64 hidden units) that takes a sequence of v_new_1 scores + key features for one coin, outputs refined probability:

```
Input shape: (batch, sequence_length=24, features=10)
  features = [score_long, score_short, cfgi, btc_score, parkinson_vol, atr_pct, vol_z, breadth_diff, hour_sin, hour_cos]
  sequence_length = 24 × 4h = 4 days context

Architecture:
  LSTM(input=10, hidden=64, layers=1, dropout=0.2)
  → Linear(64, 32)
  → ReLU
  → Linear(32, 2)  # P(big_long), P(big_short)
  → Sigmoid

Training:
  Loss: BCE on labels
  Optimizer: AdamW
  Scheduler: cosine
  Batch: 256
  Epochs: 20-30 with early stopping
  Compute: ~3-5 hours on M2 Pro CPU (PyTorch), or ~30-60 min on rented T4 GPU
```

**Cost:** 3-5 days research (build, train, validate, integrate).

**Expected lift:** +15-30% over current (so 240-270% 4-yr). HIGH UNCERTAINTY.

**Pros:**
- Captures genuine sequential dependencies BGM misses
- Small enough to train on CPU overnight
- Can stack on top of BGM without retraining BGM

**Cons:**
- 600K rows × 24 sequence length is a lot of memory; needs careful batching
- LSTM training on M2 Pro = 3-5 hours and HEAT (per CLAUDE.md: M2 Pro thermal throttles)
- Better on rented GPU (~$2-5 on Vast.ai T4)
- Marginal lift on small data

### Option C — RL policy network replacing Phase 4

**What it is:** Replace Phase 4's static Optuna search with an RL agent that learns the policy directly from simulated PnL feedback.

```
Environment:
  State: portfolio state, current scored coins (top 30 by score), CFGI, time-of-day
  Action: continuous size for each potential entry, plus binary "exit existing position"
  Reward: realized PnL of closed positions, with risk-adjusted shaping
  Episode: 1 trading year, with multi-year curriculum

Algorithm: PPO (proximal policy optimization)
  - Better than DDPG for our discrete-ish action space
  - Stable on small data
Training: 1000-5000 episodes (each = 1 year of simulation)
Compute: 10-50 hours wall clock on CPU; 2-5 hours on T4 GPU
```

**Cost:** 2-3 weeks research.

**Expected lift:** ±30%. RL is famously fickle on small data. Could be transformational OR catastrophic.

**Pros:**
- Directly optimizes PnL, not surrogate metrics
- Can learn dynamic position sizing, exit timing
- Captures portfolio-level interaction effects
- Where most paper-trading research has gone in last 3 years

**Cons:**
- RL is sample-hungry; 600K rows insufficient for full convergence
- High variance in training outcomes (3 random seeds → 3 different policies)
- Complex debugging (sparse rewards, credit assignment)
- Per CLAUDE.md, GPU-needing architectures are "impractical" — we'd need to rent
- The Phase 4 Optuna result (206%) might already be near-optimal for this data; RL would just rediscover it

### Option D — TabPFN/TFT (transformer alternatives)

**What it is:** Use a pretrained foundation model (TabPFN for small-data tabular) or Temporal Fusion Transformer.

**Cost:** TabPFN is fast (no training, just inference). TFT requires GPU + 1 week.

**Expected lift:** Unknown — TabPFN has been benchmarked vs LightGBM on small tabular and wins ~50% of time. TFT wins on time-series but needs more data.

**Recommend:** SKIP for now — exotic, GPU-dependent.

---

## What would actually move the needle (not ML alone)

If user genuinely wants 400-1000% per 4 years:

1. **Higher leverage** (5× → 10× or 15×): doubles returns, doubles drawdown. Phase 4 with 10× would probably hit 400%+ but break the -20% DD constraint. **Need to relax DD constraint to gain returns.**

2. **Wider universe** (100 coins → 200-300): more big-mover events to capture, more diversification, more compute.

3. **Higher-frequency data** (4h → 1h or 15m): catches moves earlier, more signals. Requires recomputing all features at higher freq, much bigger data, longer training.

4. **Aggressive concentrated sizing** (0.5% × 5× → 2-3% × 10× on top conviction): bet bigger when score is very high. RL would naturally learn this if reward shaped correctly.

5. **Different label window** (7 days → variable, e.g., "exit at peak then reverse"): would require detecting peaks in real-time, not just predicting forward returns.

6. **More OOS depth** (~3 years → 6-7 years): re-run Phase B-style backfill of 2017-2019 data, train on more history, get more diverse regime samples.

---

## Recommended phased approach (if you want to proceed)

**Phase v_new_1.5 (1-2 days, low risk) — Sequence-feature BGM**
- Add lagged score + rolling features to v_new_1 training
- Re-run Phase 3+4
- Target: 220-250% 4-year return, validate sequence features actually help

**Phase v_new_2 (3-5 days, moderate risk) — CPU LSTM stacker**
- Only if v_new_1.5 shows sequence signal helps
- Train small LSTM on M2 Pro overnight (or rent T4 for $2-5)
- Stacks on top of v_new_1.5 BGM scores
- Target: 250-300% 4-year return

**Phase v_new_3 (2-3 weeks, high risk) — RL policy**
- Only if v_new_1.5 + v_new_2 gives meaningful lift
- PPO agent learns position sizing + exit timing
- GPU rental ~$10-20 if needed
- Target: 300-450% 4-year return — uncertain

**Phase MACRO (parallel to above) — non-ML changes**
- Test 10× leverage with relaxed DD constraint (-30% allowed)
- Test top-200 universe instead of top-100
- Optional: backfill 2017-2019 data for more OOS depth

The ML stack alone won't get us to 1000%. The ML stack + leverage/sizing/universe changes might.

---

## My honest recommendation

**Start with Phase v_new_1.5 (sequence features in BGM). It's 1-2 days, M2 Pro friendly, low risk, and tells us if sequence patterns help AT ALL.**

If yes → proceed to v_new_2 LSTM
If no → LSTM/RL won't help either; pivot to MACRO changes (leverage, universe)

This is the cheap diagnostic before committing to the expensive neural-net research.

For the user's 200-1000%/year target specifically: realistically the ML side maxes out around 300% per 4 years on this data architecture. To go beyond, we need MACRO changes (leverage 10×, universe expansion, OR different data/strategy entirely). I'd rather be honest about this than promise 1000% returns from LSTM/RL alone.

---

## Open decisions for user

1. **Compute reality check**: do you want to stay on M2 Pro/VPS only, or are you OK renting GPU on Vast.ai for one-off training runs (~$2-5 per LSTM session, ~$10-20 per RL session)?

2. **Risk tolerance for higher returns**: Phase 4 had hard -20% max DD constraint per year. To get 400%+ returns, that almost certainly needs to be relaxed to -30% or -40%. OK with that?

3. **Phased acceptance**: start with v_new_1.5 (cheap diagnostic) and build up, OR jump straight to v_new_2 LSTM?

4. **MACRO changes priority**: do you want to test 10× leverage in Phase 4 first (1 day, big lift potential) BEFORE starting LSTM/RL research? It's a faster bigger lever.

Until we agree on these, I won't spawn anything.
