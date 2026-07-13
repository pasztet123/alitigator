"""Generate read-only corpus audit artifacts; never mutates an index."""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.rag import get_rag_config
from app.rag_diagnostics import collect_corpus_health, inventory_local_corpus


def main() -> None:
    output = Path("artifacts/corpus_audit")
    output.mkdir(parents=True, exist_ok=True)
    config = get_rag_config()
    inventory = inventory_local_corpus(config)
    health = collect_corpus_health(config)
    (output / "local_corpus_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "backend_inventory.json").write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = ["# Local corpus inventory", "", "| Source | Valid records | Unique docs | Types |", "|---|---:|---:|---|"]
    for source in inventory["sources"]:
        rows.append("| {path} | {valid_json_count} | {unique_document_count} | {source_type_distribution} |".format(**{**source, "valid_json_count": source.get("valid_json_count", 0), "unique_document_count": source.get("unique_document_count", 0), "source_type_distribution": ", ".join("%s=%s" % item for item in source.get("source_type_distribution", {}).items())}))
    rows.extend(["", "## Active backend", "", "```json", json.dumps(health, ensure_ascii=False, indent=2), "```"])
    (output / "local_corpus_inventory.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
