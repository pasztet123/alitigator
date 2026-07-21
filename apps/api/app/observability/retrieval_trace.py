"""Stable trace and cache identity helpers shared by the v2 retrieval flow."""
from __future__ import annotations

import hashlib
import json
from typing import Any

RETRIEVAL_TRACE_VERSION = "retrieval_trace_v2"


def build_cache_identity(*, normalized_question: str, retrieval_profile: str, pipeline_version: str, dictionary_version: str, query_analyzer_version: str, document_extractor_version: str, validator_version: str, query_builder_version: str) -> dict[str, str]:
    payload = {"normalized_question": normalized_question, "retrieval_profile": retrieval_profile, "pipeline_version": pipeline_version, "dictionary_version": dictionary_version, "query_analyzer_version": query_analyzer_version, "document_extractor_version": document_extractor_version, "validator_version": validator_version, "query_builder_version": query_builder_version}
    payload["key"] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return payload


def candidate_funnel(*, raw_chunks: int, raw_documents: int, deduplicated: int, after_institution_gate: int, after_relevance_validation: int) -> dict[str, int]:
    return {"raw_chunks": raw_chunks, "raw_documents": raw_documents, "deduplicated": deduplicated, "after_institution_gate": after_institution_gate, "after_relevance_validation": after_relevance_validation}
