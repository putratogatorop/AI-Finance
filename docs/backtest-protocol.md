# Backtest Protocol

Rules for writing backtests in this project. Non-negotiable because the point is reproducibility: any Claude agent, now or in future sessions, should produce byte-identical output when running the same backtest script against the same snapshot date.

## Why this exists

Paper trading revealed that past backtests could not be reproduced — "every AI gives different numbers." Three causes: mutable postgres (candles are appended live, VPS prunes at 90 days), non-frozen universe (which pairs were listed on Gate.io futures on any given day was not captured), and re-invented metric math (different "PF" formulas across scripts). This protocol eliminates all three.

## The rules (non-negotiable)

1. **Start from the template.** New backtests MUST be created via:
   ```bash
   cp services/python/scripts/backtest_template.py services/python/scripts/backtest_<name>.py
   ```
   Do not write a backtest from scratch.

2. **Declare SNAPSHOT_DATE.** Set the constant at the top of the new script to a date listed in `data/snapshots/MANIFEST.md`. Do not dynamically compute the date.

3. **Do not read live postgres for canonical numbers.** All data comes from `load_snapshot(SNAPSHOT_DATE)`. The only script allowed to touch postgres is `export_snapshot.py`.

4. **Do not edit the canonical helpers.** `load_snapshot`, `compute_metrics`, `write_results`, and `_sha256_file` are copy-pasted (not imported) into every backtest to guarantee identical math. Do not modify them after copying.

5. **Write via `write_results()`.** Every backtest produces `results/<run_id>/{trades.csv, metrics.json, params.json, run_metadata.json}`. No custom output paths.

6. **Commit the results folder.** Published numbers must be citable — `results/<run_id>/` stays in git permanently. Do not delete. Do not edit. If the analysis is wrong, write a new run, don't rewrite an old one.

## How to refresh the snapshot

When you need a newer data cut:

```bash
cd services/python
python scripts/export_snapshot.py --date 2026-05-15 --notes "whatever"
git add ../../data/snapshots/MANIFEST.md
git commit -m "data(snapshot): 2026-05-15"
```

The `.parquet` files themselves are gitignored (they're large). They live locally and on your Google Drive backup (`My Drive/ai-finance/`). If another agent/machine needs them, they copy from Drive and `load_snapshot` verifies the sha256.

## How to reproduce a published result

1. Read `results/<run_id>/run_metadata.json`. Note the `git_sha`, `snapshot_date`, and the two snapshot sha256s.
2. `git checkout <git_sha>`.
3. Verify `data/snapshots/MANIFEST.md` lists that `snapshot_date` with matching sha256s. If the Parquet files aren't on your disk, copy them from Drive.
4. Re-run the script: `python services/python/scripts/backtest_<name>.py`.
5. Diff the new `metrics.json` against the committed one. Must be byte-identical.

## When reproducibility legitimately breaks

Two cases don't count as bugs — just write a new run:

- **Dependency upgrade** (you ran `pip install -U <package>`). The `pip_freeze` field in metadata shows the difference.
- **Param or strategy edit** (you intentionally changed something). Commit first, run second — new git_sha → new run_id → different output is expected.

In both cases: do not overwrite the old `results/<run_id>/`. Write a new one and let history speak.

## When iterating

While you're still writing the strategy block, `git_dirty: true` will appear in metadata. That's fine for iteration. But before you publish a number as "the result," commit the script first so the run has a clean SHA.

## Predictor artifact contract

When a backtest gates trades by a trained model, both the model and its
threshold must be reproducible from artifacts checked into git. The
contract for any predictor-gated backtest:

1. **Model artifact** lives at `services/python/models/<name>_v<N>.joblib`.
   It is a callable supporting `predict_proba(X) -> array of shape (n, 2)`,
   typically a `sklearn.pipeline.Pipeline`. Inputs are floats in the order
   declared by the meta JSON (next).
2. **Meta JSON** lives at `services/python/models/<name>_v<N>_meta.json` and
   contains at minimum: `feature_names` (ordered list), `label_thresholds`
   (the labeler's positive-class definition), `train_auc`, `oot_auc`,
   `cv_aucs`, `trained_on_n`, `snapshot_date`. Anyone reproducing the run
   must match this metadata to be sure they are using the same model.
3. **Threshold is NOT stored in the joblib.** Instead, the backtest script
   declares a *quantile* (e.g. "67th percentile of train-period predicted
   probabilities") and re-derives the resulting cutoff at run-time from
   training data. This makes threshold drift impossible — given identical
   training inputs, the cutoff is deterministic.
4. **Feature implementation** lives under `services/python/src/ml/<area>/features.py`
   and MUST satisfy a "no-lookahead" unit test that permutes future bars
   and asserts feature output is unchanged.
5. **Label implementation** lives under `services/python/src/ml/<area>/labels.py`
   and is allowed to use future bars (it produces the training target).
   Features and labels MUST live in separate modules so a reviewer can
   verify the no-lookahead boundary at file granularity.
6. The protocol-compliant backtest writes the chosen threshold value into
   `results/<run_id>/metrics.json` so the actual cutoff used at runtime is
   visible in the audit trail.

Example: `services/python/scripts/backtest_strat_b_with_predictor.py` +
`docs/continuation-predictor-v1.md`.

## Continuous sizing artifact contract

When a backtest applies *continuous position sizing* on top of a binary
predictor (e.g., a BTC-trend-score sizing curve), the additional rules:

7. **Sizing function lives in `src/ml/<area>/sizing.py`.** It must be a
   pure function of `(scalar_score, direction, config_object)` that
   returns a non-negative scalar. Sign-locked by direction: when the score
   contradicts the trade direction, the function MUST return 0.0 (the
   "no trades against the trend" rule).
8. **Sizing config is JSON-serialized to `services/python/models/<name>_v<N>_sizing.json`**
   alongside the predictor's `joblib`. Anyone running the backtest must
   load this exact config — backtest scripts MUST NOT silently use defaults
   that diverge from the saved JSON.
9. **The simulator multiplies `pnl_pct` by the sizing scale** to produce
   `sized_pnl`. Trades with `pos_scale == 0` are excluded from "trades"
   counts in metrics (they didn't happen). Walk-forward and Monte Carlo
   computations use the `sized_pnl` column.
10. **Sensitivity sweep over sizing knobs is reportable, never selectable.**
    Knobs (cap, floor, slope, intercept) are locked at training time. A
    Phase-6-style sensitivity run perturbs them ±20% to confirm the ship
    gate verdict is robust, but the final shipping config must be the
    pre-locked one — no post-hoc OOT tuning.

Example: `services/python/scripts/backtest_bigmover_combined_with_ml.py` +
`docs/bigmover-combined-v1.md` (continuous sign-locked sizing on the BTC
trend score).

## Related files

- Template: `services/python/scripts/backtest_template.py`
- Exporter: `services/python/scripts/export_snapshot.py`
- Manifest: `data/snapshots/MANIFEST.md`
- Design: `docs/superpowers/specs/2026-04-22-backtest-reproducibility-protocol-design.md`
- Canary: `services/python/scripts/backtest_reference_sma.py` + its `results/` folder
