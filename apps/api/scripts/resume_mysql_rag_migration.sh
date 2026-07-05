#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
SQLITE_PATH="${SQLITE_PATH:-$ROOT_DIR/data/processed/eureka_rag.sqlite3}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DOCUMENT_BATCH_SIZE="${DOCUMENT_BATCH_SIZE:-25}"
CHUNK_BATCH_SIZE="${CHUNK_BATCH_SIZE:-50}"
PROGRESS_EVERY_DOCUMENTS="${PROGRESS_EVERY_DOCUMENTS:-25}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-5}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

TOTAL_DOCUMENTS="$("$PYTHON_BIN" - <<'PY'
import sqlite3
from pathlib import Path

sqlite_path = Path("data/processed/eureka_rag.sqlite3")
connection = sqlite3.connect(sqlite_path)
try:
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    print(cursor.fetchone()[0])
finally:
    connection.close()
PY
)"

echo "Target SQLite documents: $TOTAL_DOCUMENTS"

while true; do
  read -r CURRENT_DOCUMENTS MAX_DOCUMENT_ID <<EOF
$("$PYTHON_BIN" - <<'PY'
import os
from dotenv import load_dotenv
import pymysql

load_dotenv(".env")
connection = pymysql.connect(
    host=os.getenv("ALITIGATOR_RAG_MYSQL_HOST"),
    port=int(os.getenv("ALITIGATOR_RAG_MYSQL_PORT", "3306")),
    user=os.getenv("ALITIGATOR_RAG_MYSQL_USER"),
    password=os.getenv("ALITIGATOR_RAG_MYSQL_PASSWORD"),
    database=os.getenv("ALITIGATOR_RAG_MYSQL_DATABASE"),
    charset="utf8mb4",
    ssl={},
    cursorclass=pymysql.cursors.DictCursor,
)
try:
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) AS c, COALESCE(MAX(document_id), '') AS max_id FROM rag_documents")
        row = cursor.fetchone()
        print(f"{row['c']} {row['max_id']}")
finally:
    connection.close()
PY
)
EOF

  echo "MySQL progress: documents=$CURRENT_DOCUMENTS max_document_id=${MAX_DOCUMENT_ID:-<none>}"

  if [[ "$CURRENT_DOCUMENTS" -ge "$TOTAL_DOCUMENTS" ]]; then
    echo "Migration complete."
    exit 0
  fi

  if "$PYTHON_BIN" scripts/migrate_sqlite_rag_to_mysql.py \
    --env-file "$ENV_FILE" \
    --sqlite-path "$SQLITE_PATH" \
    --document-batch-size "$DOCUMENT_BATCH_SIZE" \
    --chunk-batch-size "$CHUNK_BATCH_SIZE" \
    --progress-every-documents "$PROGRESS_EVERY_DOCUMENTS" \
    --start-after-document-id "$MAX_DOCUMENT_ID"
  then
    continue
  fi

  echo "Migration attempt failed, retrying in ${RETRY_DELAY_SECONDS}s..."
  sleep "$RETRY_DELAY_SECONDS"
done
