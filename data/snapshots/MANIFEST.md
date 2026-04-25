# Backtest Data Snapshots

This file is the source of truth for which Parquet snapshots exist and the canonical sha256 of each. `export_snapshot.py` appends one row per export. Do not hand-edit.

See `docs/backtest-protocol.md` for usage.

| Date       | Candles rows | Candles sha256                                                   | Universe rows | Universe sha256                                                  | Notes |
|------------|--------------|------------------------------------------------------------------|---------------|------------------------------------------------------------------|-------|
| 2026-04-23 | 20310838 | 7a7a3bd40a166d15a3c37229ff336b94ab14076a981bf8b5e7787687eb87a0d8 | 667 | ff5ecdb01d61b46f90dee4d1c6fa83e5c33e14a28a473a7275a82d9d04d97ce2 | first snapshot |

## Verification log

| Date       | Test                       | Result | Notes |
|------------|----------------------------|--------|-------|
| 2026-04-23 | self-reproducibility (SMA) | PASS   | two runs of `backtest_reference_sma.py` produced byte-identical metrics.json (708 trades, PF 1.214) |
| 2026-04-23 | cross-agent (SMA)          | PASS   | fresh subagent with only protocol+script reproduced committed metrics.json byte-for-byte |
| 2026-04-24 | 20552035 | 9a21151142bcf8e022e2cce23b929e7c15198fd2f54d19e6d8836d290f3f29ac | 667 | e273d296f6213149966f4c168c508cb197668991f0e06acf22243be75040d9bd | post-backfill: +32 pairs incl CHIP HOLO GUN; 21 futures-only still missing |
| 2026-04-01 | 19021078 | ee67165ca0ac41ad5320303029509cdc68e28ded700d875eb4b8cec5aeb62946 | 672 | 6cda0add4249195789ed48a2ea5998fddfc3c75a94f903a177c9863310067f6d | local full-history export on fresh Mac |
