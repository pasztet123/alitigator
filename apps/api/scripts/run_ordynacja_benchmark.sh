#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CASES_PATH="${CASES_PATH:-data/laws/processed/rag_law_eval_cases.ordynacja.seed.json}"
EXCLUDE_PATH="${EXCLUDE_PATH:-data/laws/processed/rag_law_eval_cases.ordynacja.holdout.json}"
REPORT_DIR="${REPORT_DIR:-/private/tmp/law-ordynacja-benchmark}"
FINAL_REPORT="${FINAL_REPORT:-data/processed/rag_eval_reports/law_ordynacja_seed_cross_encoder_v1.json}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LIMIT="${LIMIT:-6}"
TAX_DOMAIN="${TAX_DOMAIN:-ORDYNACJA}"

CASES_PATH="$CASES_PATH" \
EXCLUDE_PATH="$EXCLUDE_PATH" \
REPORT_DIR="$REPORT_DIR" \
FINAL_REPORT="$FINAL_REPORT" \
TAX_DOMAIN="$TAX_DOMAIN" \
BATCH_SIZE="$BATCH_SIZE" \
LIMIT="$LIMIT" \
scripts/run_law_benchmark.sh
