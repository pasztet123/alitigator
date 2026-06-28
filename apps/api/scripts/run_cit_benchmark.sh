#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CASES_PATH="${CASES_PATH:-data/laws/processed/rag_law_eval_cases.cit.seed.json}"
EXCLUDE_PATH="${EXCLUDE_PATH:-/private/tmp/rag-empty-cases.json}"
REPORT_DIR="${REPORT_DIR:-/private/tmp/law-cit-benchmark}"
FINAL_REPORT="${FINAL_REPORT:-data/processed/rag_eval_reports/law_cit_seed_cross_encoder_v1.json}"
LIMIT="${LIMIT:-6}"

if [[ ! -f "$EXCLUDE_PATH" ]]; then
  printf '[]\n' > "$EXCLUDE_PATH"
fi

env TAX_DOMAIN=CIT CASES_PATH="$CASES_PATH" EXCLUDE_PATH="$EXCLUDE_PATH" REPORT_DIR="$REPORT_DIR" FINAL_REPORT="$FINAL_REPORT" LIMIT="$LIMIT" ./scripts/run_law_benchmark.sh
