"""Single, safe-to-log description of the configured RAG runtime.

This module deliberately contains no database imports.  It is used by request
handlers, indexing jobs and diagnostics so a deployment cannot accidentally
read from one backend while reporting another.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


_BACKENDS = {"sqlite", "mysql", "mariadb", "supabase"}


@dataclass(frozen=True)
class CorpusSource:
    source_id: str
    path: Path
    expected_source_types: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class RagRuntimeInfo:
    read_backend: str
    write_backend: str
    fallback_backend: Optional[str]
    supabase_search_enabled: bool
    supabase_sync_enabled: bool
    mysql_enabled: bool


def _enabled(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _backend(value: str, *, name: str) -> str:
    normalized = value.strip().lower()
    if normalized not in _BACKENDS:
        raise RuntimeError("%s must be one of %s" % (name, ", ".join(sorted(_BACKENDS))))
    return "mysql" if normalized == "mariadb" else normalized


def resolve_rag_runtime() -> RagRuntimeInfo:
    """Resolve all backend routing from explicit env vars, with no fallback."""
    legacy = os.getenv("ALITIGATOR_RAG_BACKEND", "sqlite")
    read_backend = _backend(os.getenv("RAG_READ_BACKEND", legacy), name="RAG_READ_BACKEND")
    fallback_raw = os.getenv("RAG_FALLBACK_BACKEND", "").strip()
    fallback_backend = _backend(fallback_raw, name="RAG_FALLBACK_BACKEND") if fallback_raw else None
    if fallback_backend == read_backend:
        raise RuntimeError("RAG_FALLBACK_BACKEND must differ from RAG_READ_BACKEND")
    sync_enabled = _enabled(os.getenv("ALITIGATOR_RAG_SUPABASE_SYNC", "false"))
    write_backend = _backend(os.getenv("RAG_WRITE_BACKEND", "supabase" if sync_enabled else read_backend), name="RAG_WRITE_BACKEND")
    mysql_enabled = all(os.getenv(name) for name in (
        "ALITIGATOR_RAG_MYSQL_HOST", "ALITIGATOR_RAG_MYSQL_DATABASE",
        "ALITIGATOR_RAG_MYSQL_USER", "ALITIGATOR_RAG_MYSQL_PASSWORD",
    ))
    return RagRuntimeInfo(
        read_backend=read_backend,
        write_backend=write_backend,
        fallback_backend=fallback_backend,
        supabase_search_enabled=_enabled(os.getenv("ALITIGATOR_RAG_USE_SUPABASE", "true")),
        supabase_sync_enabled=sync_enabled,
        mysql_enabled=mysql_enabled,
    )


def iter_configured_corpus_sources(config: Any) -> list[CorpusSource]:
    """Return every configured corpus once, preserving deterministic order."""
    paths = [Path(config.processed_path), *(Path(path) for path in config.additional_source_paths)]
    seen: set[Path] = set()
    result: list[CorpusSource] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        lower_name = path.name.lower()
        expected = ("judgment",) if "judgment" in lower_name or "cbosa" in lower_name else (("interpretation",) if path == Path(config.processed_path) else ("statute",))
        result.append(CorpusSource(source_id=path.stem, path=path, expected_source_types=expected))
    return result


def corpus_manifest_hash(sources: Iterable[CorpusSource]) -> str:
    """Hash paths and source bytes; suitable for resumable, idempotent sync state."""
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source.path).encode("utf-8"))
        if source.path.exists():
            with source.path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()
