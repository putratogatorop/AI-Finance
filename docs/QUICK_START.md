# Quick Start — Dual-Direction Scanner + Paper Trading

## Prerequisites
- PostgreSQL running on localhost:5432 (database: `market`)
- Node.js + npm installed
- Python 3.12 with dependencies installed

## Start Everything — One Command

```bash
./start_all.sh
```

Launches short scanner, long scanner, paper executor, and dashboard in parallel.
Logs go to `services/python/logs/*.log`. Open `http://localhost:3000/paper`.
`Ctrl+C` stops everything cleanly.

---

## Or manually (4 terminals)

### Terminal 1 — Short Scanner (ML ≥ 0.90)
```bash
cd services/python
python scripts/live_scanner_v2.py
```
- Polls Gate.io every 60s, scans ~2000 USDT pairs
- Bearish regime gate (BTC < 20d EMA AND breadth < 60%)
- 3-phase entry: dump → bounce → rejection
- Writes to `scanner_signals_v2` table
- Logs: `logs/live_scanner_v2.log`

### Terminal 2 — Long Scanner (ML ≥ 0.80)
```bash
cd services/python
python scripts/live_scanner_long_v2.py
```
- Bullish regime gate (BTC > 20d EMA AND breadth > 40%)
- 3-phase entry: pump → pullback → reclaim
- Writes to `scanner_signals_long_v2` table
- Logs: `logs/live_scanner_long_v2.log`

Opposite regime gates mean only one scanner fires at a time — whichever regime we're in.

### Terminal 3 — Paper Executor (DRY-RUN)
```bash
cd services/python
python scripts/paper_executor.py
```
- Hard DRY_RUN — no real orders placed to Gate.io
- Polls both signal tables every 60s, logs order it WOULD place
- Tracks TP/SL/timeout against live prices, records PnL
- Writes to `paper_trades` table
- Config: $100 starting equity, 10% position, 1× leverage, TP 5% / SL 5% / 48h timeout
- Kill-switch: halts if day PnL ≤ -5%; max 3 open positions
- Logs: `logs/paper_executor.log`

### Terminal 4 — Dashboard
```bash
cd services/nextjs
npm run dev
```
Open `http://localhost:3000`:
- `/scanner` — dual-direction KPIs + tournament leaderboard + live signals
- `/scanner-long` — Long v2 + ML details (WR 93.1%, PF 17.86)
- `/scanner-short` — Short v2 + ML details (WR 81.6%, PF 5.05)
- `/paper` — paper trading equity curve, open positions, closed trades
- `/tournament` — backtest leaderboard

## Current Models (backtest OOS)

| Model | Direction | Trades | WR | PF | Avg |
|---|---|---|---|---|---|
| LONG v2 + ML ≥ 0.80 | long | 261 | 93.1% | 17.86 | +4.35% |
| SHORT v2 + ML ≥ 0.90 | short | 1,132 | 81.6% | 5.05 | +3.11% |

Live degradation expected — plan for PF ~half of backtest.

## Best Hours
Check dashboard at **20:00–02:00 WIB** (13:00–19:00 UTC) for most signals.

## Go-Live Path

1. **NOW:** run the 4 terminals above. Paper trade for 3–7 days.
2. **Validation:** once ~50 paper trades closed, compare `/paper` stats vs backtest. WR within 10 points = OK.
3. **Phase 2 ($50 real):** build `live_executor.py` (signed Gate.io API, Telegram alerts). Flip DRY_RUN off.
4. **Phase 3 ($200 → $600):** scale after 20+ real trades match paper.

## Stop
`Ctrl+C` in the `start_all.sh` terminal stops everything. If manual: `Ctrl+C` in each terminal.

bash start_all.sh

