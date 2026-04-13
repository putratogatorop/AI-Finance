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
echo "  AI-Finance — dual scanner + paper executor + UI"
echo "  Boot order: ingest (backfill first) -> scanners -> paper -> UI"
echo "================================================================"

# 1. Ingest daemon — MUST complete backfill before scanners start
echo ""
echo "[1/5] ingest daemon -> logs/ingest_candles.log"
echo "      Filling data gap from last DB bar to now..."
INGEST_LOG="services/python/logs/ingest_candles.log"
: > "$INGEST_LOG"  # truncate so grep works cleanly

(cd services/python && python scripts/ingest_candles.py) \
  > services/python/logs/ingest_candles.stdout.log 2>&1 &
pids+=($!)

# Wait for "Backfill complete" line (timeout 15 min)
echo "      Waiting for backfill to complete (timeout 15 min)..."
for i in $(seq 1 900); do
  if grep -q "Backfill complete" "$INGEST_LOG" 2>/dev/null; then
    echo "      ✓ Backfill complete. DB is current."
    break
  fi
  sleep 1
  if [ $((i % 30)) -eq 0 ]; then
    echo "      ... still backfilling (${i}s elapsed)"
  fi
done

if ! grep -q "Backfill complete" "$INGEST_LOG" 2>/dev/null; then
  echo "      ✗ Backfill did NOT complete in 15 min. Aborting."
  exit 1
fi

echo ""
echo "[2/5] short scanner  -> logs/live_scanner_v2.log"
(cd services/python && python scripts/live_scanner_v2.py) \
  > services/python/logs/live_scanner_v2.stdout.log 2>&1 &
pids+=($!)

echo "[3/5] long scanner   -> logs/live_scanner_long_v2.log"
(cd services/python && python scripts/live_scanner_long_v2.py) \
  > services/python/logs/live_scanner_long_v2.stdout.log 2>&1 &
pids+=($!)

echo "[4/5] paper executor -> logs/paper_executor.log  (DRY-RUN)"
(cd services/python && python scripts/paper_executor.py) \
  > services/python/logs/paper_executor.stdout.log 2>&1 &
pids+=($!)

echo "[5/5] dashboard      -> http://localhost:3000"
echo ""
echo "================================================================"
echo "  Open:  http://localhost:3000/paper"
echo "  Stop:  Ctrl+C"
echo "================================================================"
echo ""

(cd services/nextjs && npm run dev) &
pids+=($!)

wait -n
