#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CHAT_CASES_PATH="${CHAT_CASES_PATH:-data/processed/rag_eval_cases.ksef.json}"
LAW_CASES_PATH="${LAW_CASES_PATH:-data/laws/processed/rag_law_eval_cases.vat.ksef.json}"
REPORT_DIR="${REPORT_DIR:-data/processed/rag_eval_reports}"
CHAT_REPORT="${CHAT_REPORT:-$REPORT_DIR/ksef_chat_retrieval.json}"
LAW_REPORT="${LAW_REPORT:-$REPORT_DIR/ksef_law_retrieval.json}"
LIMIT="${LIMIT:-8}"
LAW_LIMIT="${LAW_LIMIT:-6}"

mkdir -p "$REPORT_DIR"

env ALITIGATOR_RAG_CROSS_ENCODER_ENABLED="${ALITIGATOR_RAG_CROSS_ENCODER_ENABLED:-false}" \
  .venv/bin/python -B -m app.rag_ksef_eval \
    --cases "$CHAT_CASES_PATH" \
    --limit "$LIMIT" \
    --report "$CHAT_REPORT" \
    --fail-on-miss

env ALITIGATOR_RAG_CROSS_ENCODER_ENABLED="${ALITIGATOR_RAG_CROSS_ENCODER_ENABLED:-false}" \
  .venv/bin/python -B -m app.rag_law_eval \
    --cases "$LAW_CASES_PATH" \
    --tax-domain VAT \
    --limit "$LAW_LIMIT" \
    --report "$LAW_REPORT"

echo "KSeF chat report: $CHAT_REPORT"
echo "KSeF law report: $LAW_REPORT"
