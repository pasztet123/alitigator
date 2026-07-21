"""Safe, atomic per-request trace persistence for legal RAG v2."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional
from uuid import uuid4

from pydantic import BaseModel

from .schemas import FallbackTrace


DEFAULT_TRACE_ROOT = Path("artifacts/model_rag_model")

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "runtime.json",
    "request.json",
    "model_config.json",
    "legal_research_plan.json",
    "research_plan.json",
    "clarification.json",
    "fallback_trace.json",
    "planner_fallback.json",
    "institution_matches.json",
    "institution_planner_conflicts.json",
    "institution_final_locks.json",
    "institution_queries.json",
    "query_understanding.json",
    "query_families.json",
    "query_family_results.json",
    "document_cards.json",
    "institution_gate_results.json",
    "relevance_results.json",
    "institution_filter_rejections.json",
    "primary_queries.json",
    "primary_candidates.json",
    "authority_queries.json",
    "authority_candidates.json",
    "authority_cards.json",
    "first_pass_reranking.json",
    "legal_rules.json",
    "wrong_neighbor_rejections.json",
    "evidence_bindings.json",
    "missing_evidence_requests.json",
    "second_pass_queries.json",
    "second_pass_candidates.json",
    "backreferences.json",
    "reranking.json",
    "provision_graph.json",
    "evidence_bundles.json",
    "issue_coverage.json",
    "provision_lineage.json",
    "authority_lineage.json",
    "claims.json",
    "calculations.json",
    "answer_plan.json",
    "writer_payload.json",
    "writer_output.json",
    "final_answer.txt",
    "validation.json",
    "timings.json",
    "token_usage.json",
    "costs.json",
    "metrics.json",
)

JSON_ARTIFACTS = frozenset(name for name in REQUIRED_ARTIFACTS if name.endswith(".json"))
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_KEY = re.compile(r"(?:api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token)", re.I)
_OPENAI_KEY = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b")


class TracePathError(ValueError):
    pass


def _redact_string(value: str) -> str:
    return _OPENAI_KEY.sub("[REDACTED_API_KEY]", value)


def _jsonable(value: Any, *, key: Optional[str] = None) -> Any:
    if key is not None and _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(item_key): _jsonable(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, str):
        return _redact_string(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not trace-serializable")


class TraceWriter:
    """Persist one run below ``artifacts/legal_rag_v2/<run_id>``.

    Artifact names are allow-listed and each replacement is atomic. Temporary
    files are created in the destination directory with user-only permissions,
    so a crash cannot expose a half-written JSON document.
    """

    required_artifacts = REQUIRED_ARTIFACTS

    def __init__(
        self,
        run_id: Optional[str] = None,
        *,
        root: str | Path = DEFAULT_TRACE_ROOT,
    ) -> None:
        self.run_id = run_id or uuid4().hex
        if (
            not _SAFE_RUN_ID.fullmatch(self.run_id)
            or self.run_id in {".", ".."}
            or ".." in self.run_id.split("/")
        ):
            raise TracePathError("run_id contains unsafe path characters")

        root_path = Path(root).expanduser()
        root_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root = root_path.resolve()
        candidate = self.root / self.run_id
        candidate.mkdir(parents=False, exist_ok=True, mode=0o700)
        self.run_dir = candidate.resolve()
        try:
            self.run_dir.relative_to(self.root)
        except ValueError as exc:
            raise TracePathError("run directory escapes the trace root") from exc
        if not self.run_dir.is_dir():
            raise TracePathError("run trace path is not a directory")

    def path_for(self, artifact_name: str) -> Path:
        if artifact_name not in REQUIRED_ARTIFACTS:
            raise TracePathError(f"unsupported trace artifact: {artifact_name!r}")
        path = self.run_dir / artifact_name
        # Artifact names are fixed basenames. This second check protects future
        # changes to the allow-list from accidentally enabling traversal.
        if path.parent.resolve() != self.run_dir:
            raise TracePathError("artifact path escapes the run directory")
        if path.is_symlink():
            raise TracePathError("trace artifacts cannot be symbolic links")
        return path

    def write_json(self, artifact_name: str, payload: Any) -> Path:
        if artifact_name not in JSON_ARTIFACTS:
            raise TracePathError(f"{artifact_name!r} is not a JSON artifact")
        serialized = json.dumps(
            _jsonable(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        return self._atomic_write(self.path_for(artifact_name), serialized + "\n")

    def write_text(self, artifact_name: str, text: str) -> Path:
        if artifact_name != "final_answer.txt":
            raise TracePathError(f"{artifact_name!r} is not a text artifact")
        return self._atomic_write(self.path_for(artifact_name), _redact_string(str(text)))

    def write(self, artifact_name: str, payload: Any) -> Path:
        if artifact_name.endswith(".json"):
            return self.write_json(artifact_name, payload)
        return self.write_text(artifact_name, str(payload))

    def write_stage(self, stage: str, payload: Any) -> Path:
        """Write by stage name (``claims``) or exact artifact filename."""

        artifact_name = stage if stage in REQUIRED_ARTIFACTS else (
            "final_answer.txt" if stage == "final_answer" else f"{stage}.json"
        )
        return self.write(artifact_name, payload)

    def initialize_required(self) -> None:
        """Create parseable placeholders for all stages not written yet."""

        list_artifacts = {
            "institution_planner_conflicts.json",
            "institution_final_locks.json",
            "institution_queries.json",
            "institution_filter_rejections.json",
            "primary_queries.json",
            "primary_candidates.json",
            "authority_queries.json",
            "authority_candidates.json",
            "authority_cards.json",
            "first_pass_reranking.json",
            "legal_rules.json",
            "wrong_neighbor_rejections.json",
            "evidence_bindings.json",
            "missing_evidence_requests.json",
            "second_pass_queries.json",
            "second_pass_candidates.json",
            "backreferences.json",
            "reranking.json",
            "evidence_bundles.json",
            "issue_coverage.json",
            "provision_lineage.json",
            "authority_lineage.json",
            "claims.json",
            "calculations.json",
            "validation.json",
            "metrics.json",
            "token_usage.json",
        }
        for artifact_name in REQUIRED_ARTIFACTS:
            path = self.path_for(artifact_name)
            if path.exists():
                continue
            if artifact_name == "final_answer.txt":
                self.write_text(artifact_name, "")
            elif artifact_name == "fallback_trace.json":
                self.write_json(artifact_name, FallbackTrace())
            elif artifact_name in list_artifacts:
                self.write_json(artifact_name, [])
            else:
                self.write_json(artifact_name, {})

    # Friendly aliases for pipeline/bootstrap code.
    initialize = initialize_required
    ensure_required_artifacts = initialize_required

    @staticmethod
    def _atomic_write(destination: Path, content: str) -> Path:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            text=True,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            # Persist the directory entry too where the platform supports it.
            try:
                directory_fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
            return destination
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary.unlink(missing_ok=True)
            raise
