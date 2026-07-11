from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence, runtime_checkable


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_EMBEDDING_DIMENSIONS = 3072
DEFAULT_EMBEDDING_SCHEMA_VERSION = "legal-rag-v2"
DEFAULT_CHUNKER_VERSION = "legal-provision-v2"

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider-neutral, asynchronous embedding contract.

    The offline hash implementation below satisfies this protocol, but is never
    selected implicitly. Production callers must construct an actual provider.
    """

    model: str
    dimensions: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class OpenAIEmbeddingProvider:
    """Embeddings backed by the official asynchronous OpenAI SDK."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        client: Any = None,
        api_key: Optional[str] = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Embedding model cannot be empty")
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        self.model = model.strip()
        self.dimensions = dimensions
        self._client = client
        self._api_key = api_key

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - depends on deployment extras
                raise RuntimeError(
                    "OpenAIEmbeddingProvider requires the official `openai` package"
                ) from exc
            kwargs = {"api_key": self._api_key} if self._api_key else {}
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        inputs = [str(text) for text in texts]
        if not inputs:
            return []
        if any(not text.strip() for text in inputs):
            raise ValueError("Embedding inputs cannot be empty")

        response = await self._get_client().embeddings.create(
            model=self.model,
            input=inputs,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        raw_data = response.data if hasattr(response, "data") else response["data"]
        ordered = sorted(
            raw_data,
            key=lambda item: int(
                item.index if hasattr(item, "index") else item.get("index", 0)
            ),
        )
        vectors = [
            [
                float(value)
                for value in (
                    item.embedding
                    if hasattr(item, "embedding")
                    else item["embedding"]
                )
            ]
            for item in ordered
        ]
        if len(vectors) != len(inputs):
            raise RuntimeError(
                f"Embedding provider returned {len(vectors)} vectors for {len(inputs)} inputs"
            )
        if any(len(vector) != self.dimensions for vector in vectors):
            returned = sorted({len(vector) for vector in vectors})
            raise RuntimeError(
                f"Embedding dimensions mismatch: configured={self.dimensions}, returned={returned}"
            )
        return vectors

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Compatibility alias for callers that prefer an explicit method name."""

        return await self.embed(texts)


class OfflineHashEmbeddingProvider:
    """Deterministic, low-quality embedding for explicit offline use only.

    Instantiating this class is the opt-in. No production component falls back
    to it automatically when a network provider fails.
    """

    trace_marker = "explicit_offline_hash_embedding"

    def __init__(self, *, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        self.model = "offline-hash-embedding-v1"
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(str(text)) for text in texts]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed(texts)

    def _embed_one(self, text: str) -> list[float]:
        normalized = " ".join(text.casefold().split())
        vector = [0.0] * self.dimensions
        if not normalized:
            return vector

        tokens = _TOKEN_RE.findall(normalized)
        features: list[str] = [f"tok:{token}" for token in tokens]
        features.extend(
            f"tri:{normalized[index:index + 3]}"
            for index in range(max(0, len(normalized) - 2))
        )
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


@dataclass(frozen=True)
class EmbeddingInput:
    item_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return self.item_id


@dataclass(frozen=True)
class EmbeddingHit:
    item_id: str
    text: str
    score: float
    metadata: dict[str, Any]
    model: str
    dimensions: int
    schema_version: str
    content_hash: str
    chunker_version: str
    created_at: str

    @property
    def chunk_id(self) -> str:
        return self.item_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "text": self.text,
            "score": self.score,
            "metadata": dict(self.metadata),
            "model": self.model,
            "dimensions": self.dimensions,
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "chunker_version": self.chunker_version,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class IndexingReport:
    total: int
    indexed: int
    skipped: int
    metadata_updated: int
    batches_committed: int
    model: str
    dimensions: int
    schema_version: str
    chunker_version: str


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(
            f"Cannot compare vectors with different dimensions: {len(left)} != {len(right)}"
        )
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class VersionedEmbeddingIndex:
    """Small backend-independent SQLite vector index.

    Every content revision remains addressable, while exactly one revision per
    item/configuration is marked current. Commits happen per embedding batch, so
    an interrupted indexing job resumes without recomputing completed batches.
    """

    def __init__(
        self,
        path: str | Path,
        provider: EmbeddingProvider,
        *,
        schema_version: str = DEFAULT_EMBEDDING_SCHEMA_VERSION,
        chunker_version: str = DEFAULT_CHUNKER_VERSION,
    ) -> None:
        if not schema_version.strip() or not chunker_version.strip():
            raise ValueError("Schema and chunker versions cannot be empty")
        self.path = str(path) if str(path) == ":memory:" else str(Path(path).expanduser())
        self.provider = provider
        self.schema_version = schema_version.strip()
        self.chunker_version = chunker_version.strip()
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS embedding_vectors (
                item_id TEXT NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                schema_version TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                chunker_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                is_current INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (
                    item_id, model, dimensions, schema_version,
                    content_hash, chunker_version
                )
            );
            CREATE INDEX IF NOT EXISTS idx_embedding_vectors_current
                ON embedding_vectors (
                    model, dimensions, schema_version, chunker_version, is_current
                );
            CREATE INDEX IF NOT EXISTS idx_embedding_vectors_item
                ON embedding_vectors (item_id, is_current);
            """
        )
        self._connection.commit()

    async def index(
        self,
        items: Iterable[EmbeddingInput | Mapping[str, Any] | Any],
        *,
        batch_size: int = 64,
    ) -> IndexingReport:
        return await self.index_records(items, batch_size=batch_size)

    async def index_records(
        self,
        items: Iterable[EmbeddingInput | Mapping[str, Any] | Any],
        *,
        batch_size: int = 64,
    ) -> IndexingReport:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        normalized = _normalize_inputs(items)
        total = len(normalized)
        indexed = 0
        skipped = 0
        metadata_updated = 0
        batches_committed = 0

        async with self._lock:
            pending: list[tuple[EmbeddingInput, str]] = []
            for item in normalized:
                digest = content_hash(item.text)
                existing = self._connection.execute(
                    """
                    SELECT metadata_json FROM embedding_vectors
                    WHERE item_id = ? AND model = ? AND dimensions = ?
                      AND schema_version = ? AND content_hash = ?
                      AND chunker_version = ?
                    """,
                    (
                        item.item_id,
                        self.provider.model,
                        self.provider.dimensions,
                        self.schema_version,
                        digest,
                        self.chunker_version,
                    ),
                ).fetchone()
                if existing is None:
                    pending.append((item, digest))
                    continue

                metadata_json = _stable_json(dict(item.metadata))
                # Re-indexing an older, already stored content revision makes
                # that revision current again without recomputing its vector.
                self._connection.execute(
                    """
                    UPDATE embedding_vectors SET is_current = CASE
                        WHEN content_hash = ? THEN 1 ELSE 0 END
                    WHERE item_id = ? AND model = ? AND dimensions = ?
                      AND schema_version = ? AND chunker_version = ?
                    """,
                    (
                        digest,
                        item.item_id,
                        self.provider.model,
                        self.provider.dimensions,
                        self.schema_version,
                        self.chunker_version,
                    ),
                )
                if str(existing["metadata_json"]) != metadata_json:
                    self._connection.execute(
                        """
                        UPDATE embedding_vectors SET metadata_json = ?, is_current = 1
                        WHERE item_id = ? AND model = ? AND dimensions = ?
                          AND schema_version = ? AND content_hash = ?
                          AND chunker_version = ?
                        """,
                        (
                            metadata_json,
                            item.item_id,
                            self.provider.model,
                            self.provider.dimensions,
                            self.schema_version,
                            digest,
                            self.chunker_version,
                        ),
                    )
                    metadata_updated += 1
                skipped += 1
            self._connection.commit()

            for offset in range(0, len(pending), batch_size):
                batch = pending[offset : offset + batch_size]
                vectors = await self.provider.embed([item.text for item, _ in batch])
                if len(vectors) != len(batch):
                    raise RuntimeError(
                        f"Embedding provider returned {len(vectors)} vectors for {len(batch)} inputs"
                    )
                now = datetime.now(timezone.utc).isoformat()
                with self._connection:
                    for (item, digest), vector in zip(batch, vectors):
                        if len(vector) != self.provider.dimensions:
                            raise RuntimeError(
                                f"Embedding for {item.item_id} has {len(vector)} dimensions; "
                                f"expected {self.provider.dimensions}"
                            )
                        self._connection.execute(
                            """
                            UPDATE embedding_vectors SET is_current = 0
                            WHERE item_id = ? AND model = ? AND dimensions = ?
                              AND schema_version = ? AND chunker_version = ?
                            """,
                            (
                                item.item_id,
                                self.provider.model,
                                self.provider.dimensions,
                                self.schema_version,
                                self.chunker_version,
                            ),
                        )
                        self._connection.execute(
                            """
                            INSERT INTO embedding_vectors (
                                item_id, model, dimensions, schema_version,
                                content_hash, chunker_version, created_at, text,
                                metadata_json, embedding_json, is_current
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                            ON CONFLICT (
                                item_id, model, dimensions, schema_version,
                                content_hash, chunker_version
                            ) DO UPDATE SET
                                created_at = excluded.created_at,
                                text = excluded.text,
                                metadata_json = excluded.metadata_json,
                                embedding_json = excluded.embedding_json,
                                is_current = 1
                            """,
                            (
                                item.item_id,
                                self.provider.model,
                                self.provider.dimensions,
                                self.schema_version,
                                digest,
                                self.chunker_version,
                                now,
                                item.text,
                                _stable_json(dict(item.metadata)),
                                _stable_json([float(value) for value in vector]),
                            ),
                        )
                indexed += len(batch)
                batches_committed += 1

        return IndexingReport(
            total=total,
            indexed=indexed,
            skipped=skipped,
            metadata_updated=metadata_updated,
            batches_committed=batches_committed,
            model=self.provider.model,
            dimensions=self.provider.dimensions,
            schema_version=self.schema_version,
            chunker_version=self.chunker_version,
        )

    async def query(
        self,
        text: str,
        *,
        limit: int = 10,
        metadata_filters: Optional[Mapping[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> list[EmbeddingHit]:
        if not text.strip():
            raise ValueError("Query text cannot be empty")
        vectors = await self.provider.embed([text])
        if len(vectors) != 1:
            raise RuntimeError("Embedding provider must return exactly one query vector")
        return self.query_by_vector(
            vectors[0],
            limit=limit,
            metadata_filters=metadata_filters,
            min_score=min_score,
        )

    def query_by_vector(
        self,
        vector: Sequence[float],
        *,
        limit: int = 10,
        metadata_filters: Optional[Mapping[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> list[EmbeddingHit]:
        if limit <= 0:
            return []
        if len(vector) != self.provider.dimensions:
            raise ValueError(
                f"Query vector has {len(vector)} dimensions; expected {self.provider.dimensions}"
            )
        rows = self._connection.execute(
            """
            SELECT item_id, text, metadata_json, embedding_json, model,
                   dimensions, schema_version, content_hash, chunker_version,
                   created_at
            FROM embedding_vectors
            WHERE model = ? AND dimensions = ? AND schema_version = ?
              AND chunker_version = ? AND is_current = 1
            """,
            (
                self.provider.model,
                self.provider.dimensions,
                self.schema_version,
                self.chunker_version,
            ),
        ).fetchall()
        hits: list[EmbeddingHit] = []
        for row in rows:
            metadata = _load_json_object(str(row["metadata_json"]))
            if not _metadata_matches(metadata, metadata_filters or {}):
                continue
            stored_vector = [float(value) for value in json.loads(row["embedding_json"])]
            score = cosine_similarity(vector, stored_vector)
            if min_score is not None and score < min_score:
                continue
            hits.append(
                EmbeddingHit(
                    item_id=str(row["item_id"]),
                    text=str(row["text"]),
                    score=score,
                    metadata=metadata,
                    model=str(row["model"]),
                    dimensions=int(row["dimensions"]),
                    schema_version=str(row["schema_version"]),
                    content_hash=str(row["content_hash"]),
                    chunker_version=str(row["chunker_version"]),
                    created_at=str(row["created_at"]),
                )
            )
        hits.sort(key=lambda item: (-item.score, item.item_id))
        return hits[:limit]

    def count(self, *, current_only: bool = True) -> int:
        where = " WHERE is_current = 1" if current_only else ""
        row = self._connection.execute(
            f"SELECT COUNT(*) AS count FROM embedding_vectors{where}"
        ).fetchone()
        return int(row["count"] if row is not None else 0)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "VersionedEmbeddingIndex":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _normalize_inputs(
    items: Iterable[EmbeddingInput | Mapping[str, Any] | Any],
) -> list[EmbeddingInput]:
    by_id: dict[str, EmbeddingInput] = {}
    order: list[str] = []
    for raw in items:
        if isinstance(raw, EmbeddingInput):
            item = raw
        elif isinstance(raw, Mapping):
            item_id = raw.get("item_id") or raw.get("chunk_id") or raw.get("id")
            text = raw.get("text")
            if text is None:
                text = raw.get("chunk_text")
            metadata = raw.get("metadata") or {}
            item = EmbeddingInput(str(item_id or ""), str(text or ""), metadata)
        else:
            item_id = (
                getattr(raw, "item_id", None)
                or getattr(raw, "chunk_id", None)
                or getattr(raw, "id", None)
            )
            text = getattr(raw, "text", None)
            if text is None:
                text = getattr(raw, "chunk_text", None)
            metadata = getattr(raw, "metadata", {}) or {}
            item = EmbeddingInput(str(item_id or ""), str(text or ""), metadata)
        if not item.item_id.strip():
            raise ValueError("Every embedding input requires an item_id")
        if not item.text.strip():
            raise ValueError(f"Embedding input {item.item_id!r} has empty text")
        if not isinstance(item.metadata, Mapping):
            raise TypeError(f"Embedding metadata for {item.item_id!r} must be a mapping")
        if item.item_id not in by_id:
            order.append(item.item_id)
        by_id[item.item_id] = item
    return [by_id[item_id] for item_id in order]


def _metadata_matches(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, expected in filters.items():
        actual: Any = metadata
        for part in str(key).split("."):
            if not isinstance(actual, Mapping) or part not in actual:
                return False
            actual = actual[part]
        if isinstance(expected, (set, frozenset, list, tuple)):
            expected_values = set(expected)
            if isinstance(actual, (set, frozenset, list, tuple)):
                if not expected_values.intersection(actual):
                    return False
            elif actual not in expected_values:
                return False
        elif isinstance(actual, (set, frozenset, list, tuple)):
            if expected not in actual:
                return False
        elif actual != expected:
            return False
    return True


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_json_object(value: str) -> dict[str, Any]:
    try:
        loaded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}
