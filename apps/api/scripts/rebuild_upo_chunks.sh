#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_PATH="${OUTPUT_PATH:-data/laws/processed/tax_treaties_core.jsonl}"
MANIFEST_PATH="${MANIFEST_PATH:-data/laws/processed/tax_treaties_core_manifest.json}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" app/treaty_chunk.py \
  "$OUTPUT_PATH" \
  --manifest "$MANIFEST_PATH"
