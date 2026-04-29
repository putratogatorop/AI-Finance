#!/usr/bin/env bash
# Service health monitor — runs on VPS via cron every 5 minutes.
# Checks each expected container is Up; sends Telegram alert if any are down.
#
# Setup:
#   1. Create a Telegram bot: message @BotFather → /newbot → get token
#   2. Get your chat ID: message @userinfobot → get your ID
#   3. Set in /opt/ai-finance/.env:
#        TELEGRAM_BOT_TOKEN=<your-token>
#        TELEGRAM_CHAT_ID=<your-chat-id>
#
# Cron: */5 * * * * /opt/ai-finance/scripts/monitor_services.sh >> /var/log/monitor.log 2>&1

set -euo pipefail

source /opt/ai-finance/.env 2>/dev/null || true

TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

EXPECTED_SERVICES=(
    "ai-finance-postgres-1"
    "ai-finance-ingest-1"
    "ai-finance-paper-1"
    "ai-finance-scanner-short-1"
    "ai-finance-scanner-long-1"
    "ai-finance-scanner-d1e-1"
    "ai-finance-scanner-v8-macd-pullback-short-1"
    "ai-finance-scanner-v8-macd-early-trend-short-1"
    "ai-finance-scanner-v8-macd-pullback-long-1"
    "ai-finance-scanner-v8-rsi-recovery-long-1"
)

send_telegram() {
    local msg="$1"
    if [[ -z "$TELEGRAM_BOT_TOKEN" || -z "$TELEGRAM_CHAT_ID" ]]; then
        echo "[monitor] Telegram not configured — skipping alert"
        return
    fi
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        -d "text=${msg}" \
        -d "parse_mode=HTML" > /dev/null
}

DOWN=()
for svc in "${EXPECTED_SERVICES[@]}"; do
    status=$(docker inspect --format='{{.State.Status}}' "$svc" 2>/dev/null || echo "missing")
    if [[ "$status" != "running" ]]; then
        DOWN+=("$svc ($status)")
    fi
done

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

if [[ ${#DOWN[@]} -eq 0 ]]; then
    echo "[$TS] OK — all ${#EXPECTED_SERVICES[@]} services running"
else
    MSG="🚨 <b>VPS ALERT</b> ${TS}%0A${#DOWN[@]} service(s) down:%0A"
    for d in "${DOWN[@]}"; do
        MSG+="• ${d}%0A"
        echo "[$TS] DOWN: $d"
    done
    MSG+="SSH: ai-finance-deploy@103.103.20.30"
    send_telegram "$MSG"
fi
