#!/usr/bin/env bash
# Daily Postgres backup — runs on VPS via cron.
# Dumps the aifinance DB, keeps last 7 days locally.
# Cron: 0 2 * * * /opt/ai-finance/scripts/backup_postgres.sh >> /var/log/pg-backup.log 2>&1

set -euo pipefail

BACKUP_DIR="/opt/backups/postgres"
COMPOSE_FILE="/opt/ai-finance/docker-compose.prod.yml"
DB_NAME="${POSTGRES_DB:-aifinance}"
DB_USER="${POSTGRES_USER:-aifinance}"
KEEP_DAYS=7
DATE=$(date +%Y%m%d_%H%M%S)
OUTFILE="$BACKUP_DIR/pg_${DB_NAME}_${DATE}.sql.gz"

mkdir -p "$BACKUP_DIR"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting backup → $OUTFILE"

docker compose -f "$COMPOSE_FILE" exec -T postgres \
    pg_dump -U "$DB_USER" "$DB_NAME" | gzip > "$OUTFILE"

SIZE=$(du -sh "$OUTFILE" | cut -f1)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Done — $SIZE"

# Prune backups older than KEEP_DAYS
find "$BACKUP_DIR" -name "pg_${DB_NAME}_*.sql.gz" -mtime +"$KEEP_DAYS" -delete
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Pruned files older than ${KEEP_DAYS}d"
