from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate local RAG corpus into configured MariaDB/MySQL storage.")
    parser.add_argument("--env-file", default="apps/api/.env")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path("apps/api").resolve()))
    load_dotenv(args.env_file)

    from app.mysql_rag import ensure_schema, mysql_connection
    from app.rag import reindex_corpus

    print("Connecting and ensuring schema...", flush=True)
    with mysql_connection() as connection:
        ensure_schema(connection)
        if args.verify_only:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) AS count FROM rag_documents")
                documents = cursor.fetchone()["count"]
                cursor.execute("SELECT COUNT(*) AS count FROM rag_chunks")
                chunks = cursor.fetchone()["count"]
            print({"documents": int(documents), "chunks": int(chunks)}, flush=True)
            return 0

    print("Starting reindex...", flush=True)
    result = reindex_corpus(limit=args.limit, force=args.force)
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
