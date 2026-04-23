#!/bin/bash
# Launch everything in one command with ordered boot:
#   1. Ingest daemon — waits for "Backfill complete" before proceeding (fills gap first)
#   2. Scanners (short + long) — read from DB
#   3. Paper executor (DRY-RUN)
#   4. Dashboard
#
# Ctrl+C stops all processes cleanly.
# Usage: ./start_all.sh

set -e
cd "$(dirname "$0")"
mkdir -p services/python/logs

# Load .env so child processes (Python scripts, Next.js) see DATABASE_URL etc.
# The Python scripts only read os.environ and do not parse .env themselves.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "[start_all] loaded .env"
fi

# Detect Python launcher (Windows Git Bash usually has `py`, not `python`)
if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v py >/dev/null 2>&1; then
  PY=py
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "ERROR: no python/py/python3 on PATH" >&2
  exit 1
fi
echo "[start_all] using python launcher: $PY"

pids=()

cleanup() {
  echo ""
  echo "[start_all] stopping all processes..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  echo "[start_all] done."
}
trap cleanup EXIT INT TERM

echo "================================================================"
echo "  AI-Finance — dual scanner + surge scanner + paper executor + UI"
echo "  Boot order: ingest (backfill first) -> scanners -> paper -> UI"
echo "================================================================"

# 1. Ingest daemon — MUST complete backfill before scanners start
echo ""
echo "[1/6] ingest daemon -> logs/ingest_candles.log"
echo "      Filling data gap from last DB bar to now..."
INGEST_LOG="services/python/logs/ingest_candles.log"
: > "$INGEST_LOG"  # truncate so grep works cleanly

(cd services/python && $PY scripts/ingest_candles.py) \
  > services/python/logs/ingest_candles.stdout.log 2>&1 &
INGEST_PID=$!
pids+=($INGEST_PID)

# Wait for "Entering live loop" line — fires after bootstrap + gap scan + trailing
# backfill all complete. Bootstrap alone can take 20+ min when the DB is missing
# assets, so give a 30-min budget.
READY_PATTERN='Entering live loop'
echo "      Waiting for ingest to reach live loop (timeout 30 min)..."
for i in $(seq 1 1800); do
  if grep -q "$READY_PATTERN" "$INGEST_LOG" 2>/dev/null; then
    echo "      ✓ Ingest ready (bootstrap/backfill done, live loop entered)."
    break
  fi
  # Fail fast if ingest process died
  if ! kill -0 "$INGEST_PID" 2>/dev/null; then
    echo "      ✗ Ingest process died. Last output:"
    tail -n 20 services/python/logs/ingest_candles.stdout.log | sed 's/^/        /'
    exit 1
  fi
  sleep 1
  if [ $((i % 30)) -eq 0 ]; then
    echo "      ... still syncing (${i}s elapsed)"
  fi
done

if ! grep -q "$READY_PATTERN" "$INGEST_LOG" 2>/dev/null; then
  echo "      ✗ Ingest did NOT reach live loop in 30 min. Aborting."
  exit 1
fi

echo ""
echo "[2/6] short scanner  -> logs/live_scanner_v2.log"
(cd services/python && $PY scripts/live_scanner_v2.py) \
  > services/python/logs/live_scanner_v2.stdout.log 2>&1 &
pids+=($!)

echo "[3/6] long scanner   -> logs/live_scanner_long_v2.log"
(cd services/python && $PY scripts/live_scanner_long_v2.py) \
  > services/python/logs/live_scanner_long_v2.stdout.log 2>&1 &
pids+=($!)

echo "[4/6] surge scanner  -> logs/live_scanner_surge_v1.log"
(cd services/python && $PY scripts/live_scanner_surge_v1.py) \
  > services/python/logs/live_scanner_surge_v1.stdout.log 2>&1 &
SURGE_PID=$!
pids+=($SURGE_PID)

echo "[5/6] paper executor -> logs/paper_executor.log  (DRY-RUN)"
(cd services/python && $PY scripts/paper_executor.py) \
  > services/python/logs/paper_executor.stdout.log 2>&1 &
pids+=($!)

echo "[6/6] dashboard      -> http://localhost:3000"
echo ""
echo "================================================================"
echo "  Open:  http://localhost:3000/paper"
echo "  Stop:  Ctrl+C"
echo "================================================================"
echo ""

(cd services/nextjs && npm run dev) &
pids+=($!)

wait -n
