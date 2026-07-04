#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CASES_PATH="${CASES_PATH:?CASES_PATH is required}"
EXCLUDE_PATH="${EXCLUDE_PATH:?EXCLUDE_PATH is required}"
REPORT_DIR="${REPORT_DIR:?REPORT_DIR is required}"
FINAL_REPORT="${FINAL_REPORT:?FINAL_REPORT is required}"
TAX_DOMAIN="${TAX_DOMAIN:?TAX_DOMAIN is required}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LIMIT="${LIMIT:-6}"
DEVICE="${ALITIGATOR_RAG_CROSS_ENCODER_DEVICE:-cpu}"
CANDIDATE_LIMIT="${ALITIGATOR_RAG_CROSS_ENCODER_CANDIDATE_LIMIT:-12}"

mkdir -p "$REPORT_DIR"

TOTAL_CASES="$(python3 -c "import json; from pathlib import Path; cases=json.loads(Path('$CASES_PATH').read_text(encoding='utf-8')); exclude={case['id'] for case in json.loads(Path('$EXCLUDE_PATH').read_text(encoding='utf-8'))}; print(sum(1 for case in cases if case['id'] not in exclude))")"

echo "Benchmark cases: $TOTAL_CASES"
echo "Reports dir: $REPORT_DIR"

# Prebuild the local statute index once to avoid concurrent SQLite writes when
# worker processes start in parallel and all try to auto-reindex together.
.venv/bin/python -B -c "from app.rag import ensure_local_index_ready; ensure_local_index_ready()"

for ((start=0; start<TOTAL_CASES; start+=BATCH_SIZE)); do
  end=$((start + BATCH_SIZE - 1))
  if (( end >= TOTAL_CASES )); then
    end=$((TOTAL_CASES - 1))
  fi
  echo "Running offsets $start-$end"
  for ((i=start; i<=end; i++)); do
    report_path="$REPORT_DIR/$i.json"
    if [[ -f "$report_path" ]]; then
      echo "  offset $i already computed"
      continue
    fi
    (
      env \
        ALITIGATOR_RAG_CROSS_ENCODER_DEVICE="$DEVICE" \
        ALITIGATOR_RAG_CROSS_ENCODER_CANDIDATE_LIMIT="$CANDIDATE_LIMIT" \
        .venv/bin/python -B -m app.rag_law_eval \
          --cases "$CASES_PATH" \
          --exclude-cases "$EXCLUDE_PATH" \
          --offset "$i" \
          --max-cases 1 \
          --tax-domain "$TAX_DOMAIN" \
          --limit "$LIMIT" \
          --report "$report_path" \
          > "$REPORT_DIR/$i.log" 2>&1
    ) &
  done
  wait
done

python3 scripts/merge_law_eval_reports.py \
  --reports-dir "$REPORT_DIR" \
  --expected-count "$TOTAL_CASES" \
  --output "$FINAL_REPORT" \
  --cases "$CASES_PATH" \
  --exclude-cases "$EXCLUDE_PATH"

echo "Merged report: $FINAL_REPORT"
