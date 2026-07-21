"""Best-effort persistent cache for question-independent document cards."""
from __future__ import annotations

import json
import os
from typing import Any

DOCUMENT_CARD_CACHE_VERSION = "mysql_document_card_cache_v1"


def _enabled() -> bool:
    return os.getenv("LEGAL_DOCUMENT_CARD_MYSQL_CACHE_ENABLED", "true").casefold() not in {"0", "false", "no"}


def load_document_card_payload(document_id: str, extractor_version: str, dictionary_version: str, content_hash: str) -> dict[str, Any] | None:
    if not _enabled(): return None
    try:
        from app.mysql_rag import mysql_connection
        with mysql_connection() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT card_json FROM legal_document_cards WHERE document_id=%s AND extractor_version=%s AND dictionary_version=%s AND content_sha256=%s", (document_id, extractor_version, dictionary_version, content_hash))
            row = cursor.fetchone()
            return json.loads(str(row["card_json"])) if row else None
    except Exception:
        return None


def save_document_card_payload(document_id: str, extractor_version: str, dictionary_version: str, content_hash: str, payload: dict[str, Any]) -> bool:
    if not _enabled(): return False
    try:
        from app.mysql_rag import mysql_connection
        with mysql_connection() as connection, connection.cursor() as cursor:
            cursor.execute("""INSERT INTO legal_document_cards (document_id, extractor_version, dictionary_version, content_sha256, card_json)
                VALUES (%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE card_json=VALUES(card_json), created_at=CURRENT_TIMESTAMP""", (document_id, extractor_version, dictionary_version, content_hash, json.dumps(payload, ensure_ascii=False)))
            connection.commit()
        return True
    except Exception:
        return False
