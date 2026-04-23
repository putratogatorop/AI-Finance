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

## Related files

- Template: `services/python/scripts/backtest_template.py`
- Exporter: `services/python/scripts/export_snapshot.py`
- Manifest: `data/snapshots/MANIFEST.md`
- Design: `docs/superpowers/specs/2026-04-22-backtest-reproducibility-protocol-design.md`
- Canary: `services/python/scripts/backtest_reference_sma.py` + its `results/` folder
