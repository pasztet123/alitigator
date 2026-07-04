#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PDF_PATH="${PDF_PATH:-data/laws/raw/local_taxes_act_DU_2025_707.pdf}"
OUTPUT_PATH="${OUTPUT_PATH:-data/laws/processed/local_taxes_act_DU_2025_707.jsonl}"

python3 app/law_chunk.py \
  "$PDF_PATH" \
  "$OUTPUT_PATH" \
  --source-url "https://api.sejm.gov.pl/eli/acts/DU/2025/707/text.pdf" \
  --law-id "ustawa-o-podatkach-i-oplatach-lokalnych-2025-707" \
  --short-title "o podatkach i opłatach lokalnych" \
  --act-title "Ustawa z dnia 12 stycznia 1991 r. o podatkach i opłatach lokalnych" \
  --publication "Dz.U. 2025 poz. 707" \
  --legal-state-date "2025-05-16" \
  --published-date "2025-06-02" \
  --tax-tag "NIERUCHOMOŚCI"
