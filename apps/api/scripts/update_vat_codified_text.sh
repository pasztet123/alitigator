#!/usr/bin/env bash
set -euo pipefail

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT_DATE="${SNAPSHOT_DATE:-2026-05-05}"
SOURCE_URL="${SOURCE_URL:-https://eli.gov.pl/api/acts/DU/2025/775/text/U/D20250775Lj.pdf}"
RAW_PATH="${API_DIR}/data/laws/raw/vat_act_DU_2025_775_codified_${SNAPSHOT_DATE}.pdf"
OUTPUT_PATH="${API_DIR}/data/laws/processed/vat_act_DU_2025_775_codified_${SNAPSHOT_DATE}.jsonl"
PYTHON="${PYTHON:-${API_DIR}/.venv/bin/python}"

curl -fL "${SOURCE_URL}" -o "${RAW_PATH}"

PYTHONPATH="${API_DIR}" "${PYTHON}" "${API_DIR}/app/law_chunk.py" \
  "${RAW_PATH}" \
  "${OUTPUT_PATH}" \
  --target-chars 6000 \
  --source-url "${SOURCE_URL}" \
  --law-id "ustawa-o-podatku-od-towarow-i-uslug-2025-775-ujednolicony-${SNAPSHOT_DATE}" \
  --short-title "o podatku od towarów i usług" \
  --act-title "Ustawa z dnia 11 marca 2004 r. o podatku od towarów i usług" \
  --publication "Dz.U. 2025 poz. 775 ze zm." \
  --legal-state-date "${SNAPSHOT_DATE}" \
  --published-date "${SNAPSHOT_DATE}" \
  --tax-tag "VAT" \
  --source-subtype "codified_text"

echo "Downloaded: ${RAW_PATH}"
echo "Processed: ${OUTPUT_PATH}"
