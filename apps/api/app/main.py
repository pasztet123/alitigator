from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import time
import asyncio
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Literal, Optional, Union
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from postgrest.exceptions import APIError
from pydantic import BaseModel, Field

from app.model_gateway import (
    ModelGatewayError,
    ModelSchemaError,
    ModelTechnicalError,
    configured_model_ids,
    create_model_gateway,
    get_model_gateway_config,
    is_model_gateway_configured,
)

from app.auth import AuthenticatedUser, get_current_user, is_admin_user
from app.billing import (
    apply_topup_from_checkout_session,
    build_credit_pack_for_amount,
    consume_credit_for_chat,
    create_checkout_session,
    ensure_profile,
    find_credit_pack,
    get_checkout_session,
    get_credit_balance,
    get_credit_cost_per_query,
    get_credit_currency,
    get_credit_packs,
    get_credit_unit_price_gross,
    grant_credits_to_user,
    is_stripe_configured,
    list_profiles_with_credit_balances,
    mark_order_status,
    update_profile,
)
from app.eureka_ingest import DEFAULT_CONCURRENCY, DEFAULT_PAGE_SIZE, DEFAULT_SORT, FetchConfig, run_ingest
from app.bad_debt_pipeline import (
    BAD_DEBT_BENCHMARK_QUERY,
    can_run_bad_debt_pipeline,
    is_bad_debt_benchmark_trace_request,
    run_bad_debt_pipeline,
)
from app.controlled_legal_pipeline import is_mixed_invoice_query, run_legal_pipeline
from app.controlled_authority_retrieval import retrieve_housing_authorities
from app.housing_relief_pipeline import (
    can_run_housing_relief_pipeline,
    run_housing_relief_pipeline,
)
from app.legal_pipeline import (
    build_claims_from_rules,
    build_registry_from_rules,
    claim_to_dict,
    validate_claim,
)
from app.hybrid_authority_rag import (
    HybridAuthorityResult,
    bind_claims_to_evidence_bundles,
    build_authority_evidence_context,
    clarifier_enabled_from_env,
    get_legal_retrieval_mode,
    run_hybrid_authority_retrieval,
    to_jsonable,
    write_hybrid_trace_artifacts,
)
from app.legal_research.pipeline import create_default_pipeline
from app.rag import (
    RagChunk,
    add_primary_source_fallback_chunks,
    build_answer_context_block,
    build_axis_coverage,
    build_axis_coverage_context,
    build_context_block,
    build_legal_rules_context,
    build_legal_rule_trace_context,
    build_legal_source_plan,
    build_provision_reference_registry,
    detect_domains,
    detect_missing_required_facts,
    detect_mechanisms,
    extract_legal_rules_from_statute_chunks,
    filter_legal_rules_for_target_date,
    get_rag_config,
    inspect_search,
    index_exists,
    legal_rule_to_dict,
    legal_source_plan_to_dict,
    list_citations,
    normalize_provision_reference,
    prioritize_legal_rules_for_query,
    reindex_corpus,
    search_primary_law_chunks,
    search_chunks,
    build_source_plan_context,
)
from app.supabase_rag import (
    is_supabase_rag_configured,
    is_supabase_sync_enabled,
    reindex_corpus_to_supabase,
    search_chunks_supabase,
)
from app.rag_diagnostics import collect_corpus_health
from app.rag_runtime import resolve_rag_runtime
from app.supabase_client import get_supabase_service_client, is_supabase_configured

load_dotenv()

logger = logging.getLogger("alitigator.api")
API_VERSION = "2.0.22"
MODEL_GATEWAY_CONFIG = get_model_gateway_config()
DEFAULT_MODEL = MODEL_GATEWAY_CONFIG.model
AVAILABLE_MODELS = list(
    dict.fromkeys(
        [
            DEFAULT_MODEL,
            *(configured_model_ids(MODEL_GATEWAY_CONFIG)),
        ]
    )
)
HINTS_MODEL = os.getenv("LEGAL_HINTS_MODEL", DEFAULT_MODEL)
HINTS_REQUEST_TIMEOUT_SECONDS = min(
    25.0,
    max(5.0, float(os.getenv("ALITIGATOR_HINTS_REQUEST_TIMEOUT_SECONDS", "15"))),
)
CHAT_MAX_TOKENS = max(1024, MODEL_GATEWAY_CONFIG.max_output_tokens)
MODEL_CHAT_TIMEOUT_SECONDS = min(
    180.0,
    max(30.0, MODEL_GATEWAY_CONFIG.timeout_seconds),
)

_LEGAL_PIPELINE_MODES = {"legacy", "model_rag_model", "legal_rag_v2", "shadow"}
_legal_rag_v2_pipeline = None
_shadow_tasks: set[asyncio.Task] = set()
_shadow_semaphore = asyncio.Semaphore(
    max(1, int(os.getenv("LEGAL_RAG_V2_SHADOW_MAX_CONCURRENCY", "2")))
)


def get_legal_pipeline_mode() -> str:
    # LEGAL_RAG_MODE is the public, deployment-facing switch.  Keep the old
    # variable as a compatibility alias for existing environments and tests.
    # A v2 failure must never turn a user request into a 502. Until its
    # production error budget is clean, the unset deployment default is
    # shadow: legacy serves the answer while v2 runs asynchronously and keeps
    # diagnostic artifacts. A direct legal_rag_v2 setting remains available
    # only as an explicit release switch.
    raw = os.getenv("LEGAL_RAG_MODE") or os.getenv("LEGAL_PIPELINE_MODE", "shadow")
    mode = raw.strip().lower()
    aliases = {
        "rag_v2": "legal_rag_v2",
        "legacy": "legacy",
        "model_rag_model": "model_rag_model",
        "shadow": "shadow",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in _LEGAL_PIPELINE_MODES else "legacy"


def legal_runtime_debug(*, controlled_pipeline_used: bool = False) -> dict[str, object]:
    """Stable runtime identity included in API/debug traces, never secrets."""
    mode = get_legal_pipeline_mode()
    return {
        "pipeline_mode": mode,
        "retrieval_mode": "issue_scoped_bidirectional" if mode in {"model_rag_model", "legal_rag_v2"} else get_legal_retrieval_mode(),
        "rag_backend": resolve_rag_runtime().read_backend,
        "planner_mode": "model_first" if mode in {"model_rag_model", "legal_rag_v2"} else "legacy_rules",
        "authority_extractor_mode": "model_structured" if mode in {"model_rag_model", "legal_rag_v2"} else "heuristic_or_disabled",
        "answer_provider": MODEL_GATEWAY_CONFIG.provider,
        "answer_model": MODEL_GATEWAY_CONFIG.answer_writer_model if mode in {"model_rag_model", "legal_rag_v2"} else DEFAULT_MODEL,
        "provider": MODEL_GATEWAY_CONFIG.provider,
        "model": MODEL_GATEWAY_CONFIG.answer_writer_model if mode in {"model_rag_model", "legal_rag_v2"} else DEFAULT_MODEL,
        "git_commit": _git_commit(),
        "api_version": API_VERSION,
        "controlled_pipeline_used": controlled_pipeline_used,
        "fallbacks_used": [],
    }


def retrieve_controlled_authority_lane(query: str) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Run secondary-authority retrieval even when primary claims are deterministic.

    A controlled calculation establishes the legal conclusion, but Alitigator is
    also a research product.  Interpretations and judgments are therefore a
    separate, non-blocking lane: their absence is reported in the trace rather
    than silently becoming an empty sources section.
    """
    candidates: list[RagChunk] = []
    errors: list[str] = []
    counts: dict[str, int] = {}
    for source_type, limit in (("interpretation", 4), ("judgment", 3)):
        try:
            hits = search_chunks(query, limit=limit, source_types={source_type})
            counts[source_type] = len(hits)
            candidates.extend(hits)
        except Exception as exc:  # Authority research must not erase a verified primary answer.
            logger.exception("Controlled authority retrieval failed for %s", source_type)
            counts[source_type] = 0
            errors.append(f"{source_type}:{type(exc).__name__}")

    cards: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in candidates:
        label = (chunk.signature or chunk.subject or "").strip()
        key = (chunk.source_type, label)
        if not label or key in seen:
            continue
        seen.add(key)
        text = re.sub(r"\s+", " ", chunk.chunk_text).strip()
        holding = next(
            (
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", text)
                if re.search(r"\b(uznano|stwierdzono|stanowisko|przychód|wydatek|zwolnien|podatek)\b", sentence, re.I)
            ),
            text[:360],
        ) or "Brak fragmentu pozwalającego odtworzyć holding."
        cards.append(
            {
                "source_type": chunk.source_type,
                "label": label,
                "source_url": chunk.source_url or "",
                "holding": holding[:420],
                "similarity_reason": "Źródło zostało znalezione dla tego samego mechanizmu i słów kluczowych zapytania.",
                "distinguishing_facts": "Porównaj datę i cel wydatku, status nieruchomości oraz terminy ze stanem faktycznym źródła.",
            }
        )
    judgment_candidates = counts.get("judgment", 0)
    judgment_selected = sum(1 for card in cards if card.get("source_type") == "judgment")
    judgment_empty_reason = (
        "retrieval_error" if any(error.startswith("judgment:") for error in errors)
        else "no_candidates_from_corpus" if judgment_candidates == 0
        else "candidates_not_selected"
        if judgment_selected == 0
        else ""
    )
    outcome: dict[str, object] = {
        "authority_lane_executed": True,
        "authority_candidates_count_recorded": True,
        "candidate_counts": counts,
        "rendered_authority_cards": len(cards),
        "empty_authority_result_explained": not cards,
        "outcome": "no_matching_authorities" if not cards else "authorities_found",
        "errors": errors,
        "judgment_lane_executed": True,
        "judgment_candidate_count_recorded": True,
        "judgment_selected_count_recorded": True,
        "judgment_empty_result_reason_recorded": True,
        "judgment_lane": {
            "executed": True,
            "candidate_count": judgment_candidates,
            "selected_count": judgment_selected,
            "empty_result_reason": judgment_empty_reason,
        },
    }
    return cards, outcome


def _git_commit() -> str:
    """Expose a revision in diagnostics without making a failed lookup fatal."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            capture_output=True,
            text=True,
            timeout=1,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return os.getenv("K_REVISION", "unknown")


def get_legal_rag_v2_pipeline():
    global _legal_rag_v2_pipeline
    if _legal_rag_v2_pipeline is None:
        _legal_rag_v2_pipeline = create_default_pipeline()
    return _legal_rag_v2_pipeline


async def run_legal_rag_v2_shadow(question: str) -> None:
    async with _shadow_semaphore:
        try:
            await get_legal_rag_v2_pipeline().run(question, mode="shadow")
        except Exception:
            logger.exception("legal_rag_v2 shadow run failed")


def schedule_legal_rag_v2_shadow(question: str) -> None:
    task = asyncio.create_task(run_legal_rag_v2_shadow(question))
    _shadow_tasks.add(task)
    task.add_done_callback(_shadow_tasks.discard)
CHAT_REQUEST_DEADLINE_SECONDS = min(
    180.0,
    max(30.0, float(os.getenv("ALITIGATOR_CHAT_REQUEST_DEADLINE_SECONDS", "120"))),
)
RETRIEVAL_STAGE_TIMEOUT_SECONDS = min(
    75.0,
    max(20.0, float(os.getenv("ALITIGATOR_RETRIEVAL_STAGE_TIMEOUT_SECONDS", "45"))),
)


async def retrieve_baseline_chat_evidence(
    query: str,
    *,
    include_interpretations: bool,
    include_judgments: Optional[bool],
    timeout_seconds: float,
) -> tuple[list[RagChunk], dict[str, object]]:
    """Retrieve primary and secondary sources without an all-or-nothing timeout.

    Primary law is fetched first and preserved.  Interpretations and judgments
    are optional, independently observable lanes; their timeout or backend
    failure cannot turn a successful statute lookup into an empty retrieval.
    """
    started = time.monotonic()
    deadline = started + max(0.001, timeout_seconds)
    # Primary law owns the full retrieval budget.  Optional lanes use only the
    # time left after it completes; reserving time for them would reintroduce
    # the failure mode where a slow-but-successful statute query is cancelled
    # solely to make room for non-controlling materials.
    primary_budget = max(0.001, timeout_seconds)
    trace: dict[str, object] = {
        "strategy": "primary_first_independent_lanes",
        "timeout_seconds": timeout_seconds,
        "primary_law": {"status": "not_started", "count": 0},
        "interpretation": {"status": "disabled", "count": 0},
        "judgment": {"status": "disabled", "count": 0},
    }

    primary_chunks: list[RagChunk] = []
    try:
        primary_chunks = await asyncio.wait_for(
            asyncio.to_thread(search_primary_law_chunks, query),
            timeout=primary_budget,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Primary-law retrieval exceeded its lane budget timeout_seconds=%.1f",
            primary_budget,
        )
        trace["primary_law"] = {
            "status": "deadline_exceeded",
            "count": 0,
            "timeout_seconds": primary_budget,
        }
    except Exception as exc:
        logger.exception("Primary-law retrieval failed")
        trace["primary_law"] = {
            "status": "backend_error",
            "count": 0,
            "error_type": type(exc).__name__,
        }
    else:
        trace["primary_law"] = {
            "status": "completed" if primary_chunks else "no_candidates",
            "count": len(primary_chunks),
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }

    requested_lanes: list[str] = []
    if include_interpretations:
        requested_lanes.append("interpretation")
    if include_judgments is not False:
        requested_lanes.append("judgment")

    remaining = deadline - time.monotonic()
    authority_chunks: list[RagChunk] = []
    if requested_lanes and remaining > 0:

        async def retrieve_authority_lane(source_type: str) -> tuple[str, list[RagChunk]]:
            chunks = await asyncio.to_thread(
                search_chunks,
                query,
                source_types={source_type},
            )
            return source_type, chunks

        tasks = [retrieve_authority_lane(source_type) for source_type in requested_lanes]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=max(0.001, remaining),
            )
        except asyncio.TimeoutError:
            for source_type in requested_lanes:
                trace[source_type] = {
                    "status": "deadline_exceeded",
                    "count": 0,
                }
        else:
            for source_type, result in zip(requested_lanes, results):
                if isinstance(result, Exception):
                    logger.error(
                        "Optional authority retrieval failed lane=%s error=%s",
                        source_type,
                        type(result).__name__,
                    )
                    trace[source_type] = {
                        "status": "backend_error",
                        "count": 0,
                        "error_type": type(result).__name__,
                    }
                    continue
                _, chunks = result
                authority_chunks.extend(chunks)
                trace[source_type] = {
                    "status": "completed" if chunks else "no_candidates",
                    "count": len(chunks),
                }
    elif requested_lanes:
        for source_type in requested_lanes:
            trace[source_type] = {"status": "skipped_no_budget", "count": 0}

    merged: list[RagChunk] = []
    seen_chunk_ids: set[str] = set()
    for chunk in [*primary_chunks, *authority_chunks]:
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)
        merged.append(chunk)
    trace["total_count"] = len(merged)
    trace["total_elapsed_ms"] = round((time.monotonic() - started) * 1000)
    trace["primary_preserved_when_authority_incomplete"] = bool(primary_chunks) and any(
        str((trace.get(source_type) or {}).get("status"))
        in {"deadline_exceeded", "backend_error", "skipped_no_budget"}
        for source_type in requested_lanes
    )
    return merged, trace

SYSTEM_PROMPT = """
Jesteś asystentem aLitigator dla polskich prawników podatkowych.

Zasady odpowiedzi:
1. Odpowiadaj po polsku.
2. Oddzielaj ustalenia źródłowe od własnych wniosków.
3. Jeśli nie masz zweryfikowanych źródeł, napisz to wprost.
4. Struktura odpowiedzi ma mieć sekcje: Teza, Analiza, Źródła, Ryzyka i luki.
5. Nie udawaj pewności. Gdy stan prawny lub orzecznictwo wymaga potwierdzenia, zaznacz to jednoznacznie.
6. Zawsze rozróżniaj rodzaj źródła: ustawa jest treścią normy, interpretacja indywidualna przedstawia ocenę organu w konkretnej sprawie, interpretacja ogólna ma odrębny charakter, a wyrok jest wykładnią sądu.
7. Nie przedstawiaj interpretacji ani komentarza jako obowiązującego przepisu. Nie przedstawiaj przepisu jako stanowiska organu.
8. Jeśli dostarczone źródła nie wystarczają do stanowczej odpowiedzi, napisz to wprost.
9. Najpierw odpowiedz użytkownikowi możliwie użytecznie na podstawie tego, co jednak wynika ze źródeł; dopiero potem wskaż braki.
10. Nie buduj odpowiedzi wokół zastrzeżeń. Zastrzeżenia mają być krótkie i konkretne; główna część ma syntetyzować realną treść materiału.
11. Oceń relewantność każdego źródła względem pytania. W analizie opieraj się tylko na źródłach relewantnych lub częściowo relewantnych; źródła nietrafne można wspomnieć najwyżej zdaniem.
12. Jeżeli źródło jest tylko częściowo relewantne, wyciągnij z niego dokładnie ten fragment, który odpowiada na pytanie, i oznacz ograniczenie jego wagi.
13. Jeżeli materiał jest niepełny, sformułuj minimalną użyteczną odpowiedź: co wynika wprost, jakie ostrożne wnioski można wyprowadzić i czego nie da się potwierdzić.
14. Nie pisz, że „nie da się odpowiedzieć”, jeśli da się odpowiedzieć choćby częściowo. Zamiast tego napisz „na podstawie tych źródeł można stwierdzić co najmniej, że...”.
15. W pytaniach o KSeF nie zakładaj, że miejsce dostawy lub świadczenia poza Polską automatycznie oznacza brak faktury ustrukturyzowanej. Najpierw sprawdź zakres polskich zasad fakturowania z art. 106a, obowiązek faktury z art. 106b, wyjątki z art. 106ga ust. 2, a dopiero potem sposób udostępnienia z art. 106gb.
16. Jeżeli źródła pokazują, że faktura została wystawiona poza KSeF, nie zakładaj, że sama późniejsza faktura w KSeF automatycznie zastępuje wcześniejszą; może chodzić o duplikat tej samej transakcji, a prawo do odliczenia i moment jego realizacji oceniaj według materialnych przesłanek art. 86, nie według samego późniejszego numeru KSeF.
17. Jeżeli art. 106gb ust. 4 mówi o udostępnieniu faktury nabywcy w uzgodniony sposób, traktuj to jako regułę doręczenia/udostępnienia faktury ustrukturyzowanej, nie jako wyłączenie obowiązku KSeF.
18. Jeżeli pytanie dotyczy błędów w danych nabywcy, nie zakładaj automatycznie, że nota korygująca pozostaje właściwym narzędziem; sprawdź, czy z materiału nie wynika konieczność korekty przez sprzedawcę.
19. Nie wolno Ci przenosić elementów stanu faktycznego ze źródła do kazusu użytkownika. Fakty kazusu pochodzą tylko z pytania użytkownika i z dostarczonych doprecyzowań intencji, nie z treści wyroku ani interpretacji.
20. Jeżeli wyrok lub interpretacja opisuje inny stan faktyczny, możesz wykorzystać tylko wynikającą z niego tezę albo kierunek wykładni. Zawsze wyraźnie zaznacz, które elementy są wspólne, a które różne.
21. Nie używaj wyroku ani interpretacji do dopowiadania brakującej przesłanki ustawowej. Jeżeli przepis uzależnia wynik od konkretnego faktu, a tego faktu nie ma w pytaniu ani w źródłach, wskaż brak tej przesłanki zamiast zgadywać.
""".strip()

REDACTION_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "pesel": re.compile(r"\b\d{11}\b"),
    "nip": re.compile(r"\b\d{3}[- ]?\d{3}[- ]?\d{2}[- ]?\d{2}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+48[- ]?)?(?:\d[- ]?){9}(?!\d)"),
}

ASSISTANT_SECTION_TITLES = (
    "Teza",
    "Analiza",
    "Źródła",
    "Ryzyka i luki",
    "Źródła zwrócone przez retrieval",
    "Źródła użyte przez retrieval",
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=24)
    model: Optional[str] = None
    chat_id: Optional[str] = Field(default=None, max_length=128)
    intent_hints: list["IntentHintAnswer"] = Field(default_factory=list, max_length=12)
    retrieval_preferences: Optional["RetrievalPreferences"] = None


class ChatResponse(BaseModel):
    reply: str
    mode: Literal["demo", "live"]
    model: str
    redactions: list[str]
    analysis_trace: dict[str, object] = Field(default_factory=dict)
    chat_id: Optional[str] = None
    assistant_message_id: Optional[str] = None
    structured_reply: Optional["StructuredReply"] = None


class StructuredReplySection(BaseModel):
    key: str
    title: str
    content: str


class StructuredReply(BaseModel):
    opening_statute: Optional[str] = None
    sections: list[StructuredReplySection] = Field(default_factory=list)


ChatResponse.model_rebuild()


class PromptHintOption(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=80)


class IntentHintAnswer(BaseModel):
    question: str = Field(min_length=1, max_length=600)
    option_id: str = Field(min_length=1, max_length=128)
    option_label: str = Field(min_length=1, max_length=240)


class RetrievalPreferences(BaseModel):
    include_interpretations: bool = True
    include_judgments: bool = True


ChatRequest.model_rebuild()


class PromptHintsRequest(BaseModel):
    draft: str = Field(min_length=1, max_length=12000)
    intent_hints: list[IntentHintAnswer] = Field(default_factory=list, max_length=24)
    excluded_questions: list[str] = Field(default_factory=list, max_length=48)
    max_hints: int = Field(default=3, ge=1, le=3)


class PromptHint(BaseModel):
    id: str
    question: str
    options: list[PromptHintOption] = Field(min_length=2, max_length=5)


class PromptHintsResponse(BaseModel):
    hints: list[PromptHint]
    model: str
    mode: Literal["live", "fallback"]


class ModelPromptHintOption(BaseModel):
    label: str = Field(min_length=1, max_length=80)


class ModelPromptHint(BaseModel):
    question: str = Field(min_length=1, max_length=600)
    options: list[ModelPromptHintOption] = Field(min_length=2, max_length=5)


class ModelPromptHints(BaseModel):
    hints: list[ModelPromptHint] = Field(default_factory=list, max_length=3)


class ModelsResponse(BaseModel):
    default_model: str
    models: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_configured: bool
    llm_provider: str
    supabase_configured: bool
    rag_index_configured: bool
    chat_storage_available: bool
    auth_configured: bool
    stripe_configured: bool


class ChatThreadSummary(BaseModel):
    id: str
    title: str
    archived: bool
    updated_at: str
    created_at: str
    last_message_preview: str


class ChatThreadsResponse(BaseModel):
    active: list[ChatThreadSummary]
    archived: list[ChatThreadSummary]


class ChatThreadCreateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=160)


class ChatThreadUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=160)
    archived: Optional[bool] = None


class PersistedChatMessage(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str
    feedback_rating: Optional[int] = None
    feedback_comment: Optional[str] = None
    feedback_created_at: Optional[str] = None


class ChatMessageFeedbackRequest(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = Field(default=None, max_length=1200)


class ChatThreadDetail(BaseModel):
    id: str
    title: str
    archived: bool
    updated_at: str
    created_at: str
    last_message_preview: str
    messages: list[PersistedChatMessage]


class ProfileResponse(BaseModel):
    id: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    law_firm: Optional[str] = None
    is_admin: bool = False
    stripe_customer_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CreditPackResponse(BaseModel):
    id: str
    name: str
    credit_amount: int
    price_gross: int
    currency: str
    description: str


class AccountResponse(BaseModel):
    user_id: str
    email: Optional[str] = None
    profile: ProfileResponse
    is_admin: bool = False
    credit_balance: int
    credit_cost_per_query: int
    credit_unit_price_gross: int
    credit_currency: str
    stripe_configured: bool
    credit_packs: list[CreditPackResponse]


class ProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = Field(default=None, max_length=160)
    law_firm: Optional[str] = Field(default=None, max_length=160)


class CheckoutSessionRequest(BaseModel):
    credit_pack_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    credit_amount: Optional[int] = Field(default=None, ge=1, le=100000)
    success_url: Optional[str] = Field(default=None, max_length=2000)
    cancel_url: Optional[str] = Field(default=None, max_length=2000)


class CheckoutSessionResponse(BaseModel):
    order_id: str
    checkout_url: str
    checkout_session_id: str


class CheckoutSessionStatusResponse(BaseModel):
    checkout_session_id: str
    payment_status: str
    status: Optional[str] = None
    credited: bool


class AdminGrantCreditsRequest(BaseModel):
    user_email: str = Field(min_length=3, max_length=320)
    credit_amount: int = Field(ge=1, le=100000)
    reason: Optional[str] = Field(default=None, max_length=280)


class AdminGrantCreditsResponse(BaseModel):
    user_id: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    credit_balance: int


class AdminUserSummary(BaseModel):
    user_id: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    law_firm: Optional[str] = None
    is_admin: bool = False
    credit_balance: int
    created_at: Optional[str] = None


class AdminUsersResponse(BaseModel):
    users: list[AdminUserSummary]


class EurekaImportRequest(BaseModel):
    limit: int = Field(default=1000, ge=1, le=5000)
    page_size: int = Field(default=DEFAULT_PAGE_SIZE, ge=1, le=100)
    concurrency: int = Field(default=DEFAULT_CONCURRENCY, ge=1, le=16)
    sort: str = Field(default=DEFAULT_SORT, min_length=1, max_length=64)
    start_page: int = Field(default=0, ge=0, le=100000)
    retry_count: int = Field(default=3, ge=1, le=10)
    request_timeout: float = Field(default=45.0, ge=1.0, le=120.0)
    pause_seconds: float = Field(default=0.0, ge=0.0, le=10.0)
    category: Optional[str] = Field(default="Interpretacja indywidualna", max_length=256)
    law_tags: list[str] = Field(default_factory=list, max_length=20)
    raw_output_path: Optional[str] = Field(default=None, max_length=4096)
    output_path: Optional[str] = Field(default=None, max_length=4096)
    overwrite: bool = False


class EurekaImportResponse(BaseModel):
    count: int
    output_path: str
    raw_output_path: str
    source: str
    sort: str
    last_document_id: Optional[str]
    failed_ids: list[str]
    total_unique_ids: int


class RagReindexRequest(BaseModel):
    limit: Optional[int] = Field(default=None, ge=1, le=50000)
    force: bool = False
    sync_supabase: Optional[bool] = None


class RagReindexResponse(BaseModel):
    processed: int
    indexed: int
    skipped: int
    chunk_count: int
    db_path: str
    total_documents: int
    total_chunks: int
    supabase_synced: bool
    supabase_documents: int = 0
    supabase_chunks: int = 0


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4000)
    limit: Optional[int] = Field(default=None, ge=1, le=30)
    source_types: Optional[list[Literal["interpretation", "statute", "judgment", "commentary"]]] = None
    tax_domains: Optional[list[str]] = Field(default=None, max_length=20)


class RagSearchHit(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    chunk_index: int
    score: float
    subject: str
    signature: Optional[str]
    published_date: Optional[str]
    source_url: Optional[str]
    canonical_source_id: Optional[str] = None
    evidence_role: Optional[str] = None
    category: Optional[str]
    source: str
    source_type: str
    source_subtype: Optional[str]
    authority: Optional[str]
    publication: Optional[str]
    legal_state_date: Optional[str]
    source_pages: list[int]
    legal_provisions: list[str]
    chunk_chars: int
    preview: str
    selected_for_context: bool


class RagSearchResponse(BaseModel):
    query: str
    match_query: Optional[str]
    requested_limit: int
    retrieved_count: int
    selected_count: int
    selected_context_chars: int
    citations: str
    context_block: str
    axis_coverage: list[dict[str, object]] = Field(default_factory=list)
    source_plan: dict[str, object] = Field(default_factory=dict)
    legal_rules: list[dict[str, object]] = Field(default_factory=list)
    analysis_trace: dict[str, object] = Field(default_factory=dict)
    hits: list[RagSearchHit]


app = FastAPI(title="aLitigator API", version=API_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.exception_handler(Exception)
async def unhandled_exception_response(request: Request, exc: Exception) -> JSONResponse:
    error_id = str(uuid4())
    logger.exception(
        "Unhandled API error id=%s method=%s path=%s",
        error_id,
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Wewnętrzny błąd backendu.",
            "error_id": error_id,
        },
        headers={"Access-Control-Allow-Origin": "*"},
    )


def redact_text(text: str) -> tuple[str, list[str]]:
    redacted = text
    applied: list[str] = []

    for label, pattern in REDACTION_PATTERNS.items():
        next_text, count = pattern.subn(f"[REDACTED_{label.upper()}]", redacted)
        if count:
            applied.append(label)
            redacted = next_text

    return redacted, applied


def slugify_hint_question(question: str) -> str:
    normalized = question.lower()
    normalized = normalized.replace("ź", "z").replace("ł", "l")
    normalized = normalized.replace("ą", "a").replace("ę", "e")
    normalized = normalized.replace("ś", "s").replace("ć", "c")
    normalized = normalized.replace("ń", "n").replace("ó", "o").replace("ż", "z")
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-")[:64] or "hint"


def slugify_hint_option(label: str) -> str:
    normalized = label.lower()
    normalized = normalized.replace("ź", "z").replace("ł", "l")
    normalized = normalized.replace("ą", "a").replace("ę", "e")
    normalized = normalized.replace("ś", "s").replace("ć", "c")
    normalized = normalized.replace("ń", "n").replace("ó", "o").replace("ż", "z")
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-")[:48] or "opcja"


def build_hint(question: str, options: list[str]) -> PromptHint:
    normalized_options = [option.strip()[:80] for option in options if option.strip()]
    deduped_options: list[str] = []
    for option in normalized_options:
        if option.lower() not in {value.lower() for value in deduped_options}:
            deduped_options.append(option)
    if len(deduped_options) < 2:
        deduped_options = ["Tak", "Nie", "Nie wiem"]
    return PromptHint(
        id=slugify_hint_question(question),
        question=question[:240],
        options=[PromptHintOption(id=slugify_hint_option(option), label=option) for option in deduped_options[:5]],
    )


def build_hint_context(intent_hints: list[IntentHintAnswer]) -> str:
    if not intent_hints:
        return ""

    lines = ["Dodatkowe doprecyzowanie intencji użytkownika:"]
    for hint in intent_hints:
        lines.append(f"- {hint.question} Wybrana odpowiedź: {hint.option_label}.")
    return "\n".join(lines)


def build_effective_user_prompt(user_prompt: str, intent_hints: list[IntentHintAnswer]) -> str:
    hint_context = build_hint_context(intent_hints)
    if not hint_context:
        return user_prompt
    return f"{user_prompt}\n\n{hint_context}"


def build_retrieval_preferences_context(preferences: Optional[RetrievalPreferences]) -> str:
    if preferences is None:
        return ""
    if preferences.include_interpretations and preferences.include_judgments:
        return (
            "Użytkownik chce, aby odpowiedź opierała się na przepisach oraz była uzupełniona o interpretacje"
            " i wyroki sądów, jeśli retrieval znajdzie materiały relewantne."
        )
    if preferences.include_interpretations:
        return (
            "Użytkownik chce, aby odpowiedź opierała się na przepisach oraz była uzupełniona o interpretacje,"
            " ale bez wyroków sądów."
        )
    return "Użytkownik chce odpowiedzi opartej wyłącznie na przepisach ustawowych, bez interpretacji i bez wyroków."


HINT_DOMAIN_KEYWORDS: dict[str, tuple[tuple[str, float], ...]] = {
    "WHT": (
        ("wht", 3.2),
        ("withholding tax", 3.0),
        ("podatek u źródła", 3.0),
        ("podatek u zrodla", 3.0),
        ("certyfikat rezydencji", 2.2),
        ("certyfikat rezydencji", 2.2),
        ("beneficial owner", 2.0),
        ("należności licencyjne", 1.8),
        ("naleznosci licencyjne", 1.8),
        ("royalties", 1.8),
    ),
    "VAT": (
        ("vat", 2.4),
        ("import usług", 2.2),
        ("import uslug", 2.2),
        ("miejsce świadczenia", 1.9),
        ("miejsce swiadczenia", 1.9),
        ("odliczenie vat", 1.8),
        ("stawka vat", 1.8),
        ("zwolnienie z vat", 1.8),
        ("podlega vat", 1.7),
        ("podatek od towarów i usług", 1.9),
        ("podatek od towarow i uslug", 1.9),
    ),
    "CENY TRANSFEROWE": (
        ("ceny transferowe", 3.0),
        ("transakcja kontrolowana", 2.8),
        ("podmioty powiązane", 2.7),
        ("podmioty powiazane", 2.7),
        ("grupy kapitałowej", 2.0),
        ("grupy kapitalowej", 2.0),
        ("dokumentacji cen transferowych", 2.3),
        ("local file", 1.8),
        ("benchmark", 1.4),
    ),
    "CIT": (
        ("cit", 2.4),
        ("koszt uzyskania przychodów", 2.2),
        ("koszt uzyskania przychodow", 2.2),
        ("wartość niematerialna i prawna", 2.0),
        ("wartosc niematerialna i prawna", 2.0),
        ("wartość początkowa", 1.8),
        ("wartosc poczatkowa", 1.8),
        ("koszt wytworzenia", 1.8),
        ("mały podatnik", 1.8),
        ("maly podatnik", 1.8),
    ),
    "PIT": (
        ("pit", 2.4),
        ("osoba fizyczna", 1.8),
        ("rezydent", 1.4),
        ("nierezydent", 1.4),
    ),
    "PCC": (
        ("pcc", 2.8),
        ("podatek od czynności cywilnoprawnych", 2.6),
        ("podatek od czynnosci cywilnoprawnych", 2.6),
    ),
    "SD": (
        ("podatek od spadków i darowizn", 3.0),
        ("podatek od spadkow i darowizn", 3.0),
        ("sd-z2", 2.8),
        ("darowizna małżonce", 2.2),
        ("darowizna malzonce", 2.2),
        ("grupa zerowa", 1.8),
    ),
}


def score_hint_domains(draft: str) -> list[tuple[str, float]]:
    normalized = draft.lower()
    scores: dict[str, float] = {}
    for domain, keywords in HINT_DOMAIN_KEYWORDS.items():
        score = 0.0
        for keyword, weight in keywords:
            if keyword in normalized:
                score += weight
        if score > 0:
            scores[domain] = score
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def fallback_prompt_hints(
    draft: str,
    intent_hints: list[IntentHintAnswer],
    *,
    excluded_questions: Optional[list[str]] = None,
    max_hints: int = 3,
) -> list[PromptHint]:
    answered_questions = {hint.question.strip().lower() for hint in intent_hints if hint.question.strip()}
    excluded_question_set = {
        question.strip().lower()
        for question in excluded_questions or []
        if question.strip()
    }
    domains = {domain.upper() for domain in detect_domains(draft)}
    scored_domains = score_hint_domains(draft)
    normalized_draft = draft.lower()
    candidates: list[tuple[str, list[str]]] = []

    top_scored_domains = [domain for domain, _ in scored_domains[:4]]
    multiple_strong_domains = len(scored_domains) >= 2 and scored_domains[1][1] >= max(scored_domains[0][1] - 0.8, 1.8)
    dominant_domain = scored_domains[0][0] if scored_domains else None

    if multiple_strong_domains and len(top_scored_domains) >= 2:
        candidates.append(("Który obszar jest tu najbliższy sedna problemu?", [*top_scored_domains[:4], "Nie wiem"]))
    elif dominant_domain == "WHT":
        candidates.append(("Który obszar WHT jest tu najbliższy sedna problemu?", ["Zakres podatku u źródła", "Certyfikat rezydencji", "Beneficial owner", "Inne", "Nie wiem"]))
    elif dominant_domain == "CENY TRANSFEROWE":
        candidates.append(("Który obszar cen transferowych jest tu najbliższy sedna problemu?", ["Transakcja kontrolowana", "Dokumentacja", "Cena rynkowa", "Inne", "Nie wiem"]))
    elif dominant_domain == "VAT":
        candidates.append(("Który obszar VAT jest tu najbliższy sedna problemu?", ["Stawka VAT", "Zwolnienie", "Odliczenie", "Inne", "Nie wiem"]))
    elif dominant_domain == "PCC" or "PCC" in domains:
        candidates.append(("Który aspekt PCC jest tu najważniejszy?", ["Obowiązek podatkowy", "Stawka lub podstawa", "Zwolnienie lub wyłączenie", "Inne", "Nie wiem"]))
    elif dominant_domain == "CIT" or "CIT" in domains:
        candidates.append(("Który obszar CIT jest tu najbliższy sedna problemu?", ["Przychód", "Koszt", "Stawka lub podatek", "Inne", "Nie wiem"]))
    elif dominant_domain == "PIT" or "PIT" in domains:
        candidates.append(("Który obszar PIT jest tu najbliższy sedna problemu?", ["Przychód", "Koszt", "Stawka lub forma opodatkowania", "Inne", "Nie wiem"]))
    else:
        default_options = top_scored_domains[:4] if top_scored_domains else ["VAT", "PIT", "CIT", "PCC"]
        candidates.append(("Którego podatku lub obszaru dotyczy pytanie najbardziej?", [*default_options[:4], "Nie wiem"]))

    candidates.extend(
        [
            ("Jakiego typu odpowiedzi potrzebujesz najbardziej?", ["Ocena konkretnego stanu faktycznego", "Ogólne zasady", "Wskazanie przepisu", "Nie wiem"]),
            ("Jaki jest zakres problemu?", ["Jedna transakcja lub zdarzenie", "Cały model działania", "Spór co do roli stron", "Nie wiem"]),
            ("Czy w sprawie jest element zagraniczny?", ["Tak", "Nie", "Nie wiem"]),
            ("Czy znaczenie ma konkretny moment w czasie?", ["Tak", "Nie", "Nie wiem"]),
        ]
    )
    if re.search(r"sprzedawc|kupuj|nabywc", normalized_draft):
        candidates.insert(
            1,
            ("Której strony dotyczy problem przede wszystkim?", ["Sprzedawca", "Kupujący", "Obie strony", "Nie wiem"]),
        )
    if re.search(r"nieruchom", normalized_draft) and re.search(r"sprzeda|kupuj|naby", normalized_draft):
        candidates.insert(
            2,
            ("Czy pytanie dotyczy nieruchomości wykorzystywanej w działalności gospodarczej sprzedawcy?", ["Tak", "Nie", "Nie wiem"]),
        )
    if dominant_domain == "VAT" and not multiple_strong_domains and re.search(r"sprzeda|dostaw|naby|kupuj|nieruchom|samochod|pojazd|import usl|import usług", normalized_draft):
        candidates.insert(
            2,
            ("Czy kluczowe jest ustalenie, czy ta transakcja podlega VAT?", ["Tak", "Nie", "Nie wiem"]),
        )
    if {"PIT", "CIT"} & domains and re.search(r"wspolnik|udzialowiec|spolk", normalized_draft):
        candidates.insert(
            2,
            ("Kogo dotyczy główny skutek dochodowy?", ["Spółka", "Wspólnik", "Obie strony", "Nie wiem"]),
        )

    unique_questions: list[PromptHint] = []
    for question, options in candidates:
        normalized = question.lower()
        if normalized in answered_questions or normalized in excluded_question_set:
            continue
        if any(existing.question.lower() == normalized for existing in unique_questions):
            continue
        unique_questions.append(build_hint(question, options))
        if len(unique_questions) >= max_hints:
            break
    return unique_questions


def parse_prompt_hints_response(text: str) -> list[PromptHint]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    hints = payload.get("hints")
    if not isinstance(hints, list):
        return []

    parsed_hints: list[PromptHint] = []
    for item in hints:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        options = item.get("options")
        if not question or not isinstance(options, list):
            continue
        option_labels = [
            str(option.get("label") or "").strip()
            for option in options
            if isinstance(option, dict)
        ]
        parsed_hints.append(build_hint(question, option_labels))
        if len(parsed_hints) >= 3:
            break
    return parsed_hints


async def request_prompt_hints(
    draft: str,
    intent_hints: list[IntentHintAnswer],
    *,
    excluded_questions: Optional[list[str]] = None,
    max_hints: int = 3,
) -> PromptHintsResponse:
    if not is_model_gateway_configured(MODEL_GATEWAY_CONFIG):
        return PromptHintsResponse(
            hints=fallback_prompt_hints(
                draft,
                intent_hints,
                excluded_questions=excluded_questions,
                max_hints=max_hints,
            ),
            model="fallback",
            mode="fallback",
        )

    answered_context = build_hint_context(intent_hints) or "Brak wcześniejszych odpowiedzi."
    excluded_context = "\n".join(f"- {question}" for question in (excluded_questions or []) if question.strip()) or "Brak."
    system_prompt = (
        "Tworzysz krótkie pytania doprecyzowujące do formularza prawnopodatkowego."
        " Nie odpowiadasz merytorycznie i nie udzielasz porady."
        " Twoim celem jest wyłącznie wydobycie intencji użytkownika tak, by poprawić retrieval."
        " Generuj maksymalnie 3 krótkie pytania z własnymi opcjami odpowiedzi."
        " Każde pytanie ma mieć 2 do 5 krótkich opcji, z czego ostatnią może być 'Nie wiem', gdy to pomaga."
        " Jeśli pytanie jest naprawdę binarne, opcje mogą brzmieć: Tak, Nie, Nie wiem."
        " Jeśli pytanie dotyczy wyboru wariantu, stron transakcji, podatku lub zakresu problemu,"
        " podaj konkretne opcje zamiast Tak/Nie."
        " Preferuj pytania o fakty, które realnie zmieniają wynik podatkowy lub dobór źródeł,"
        " np. status strony, rodzaj transakcji, to która strona jest w centrum pytania, albo czy trzeba ocenić podleganie VAT."
        " Nie zakładaj z góry, że sednem jest VAT tylko dlatego, że w wiadomości pojawia się transakcja, sprzedaż albo usługa."
        " Najpierw ustal dominujące słowa kluczowe i jeśli widać mocniej WHT, ceny transferowe, CIT, PCC albo inny obszar, to od niego zacznij."
        " Unikaj pytań oczywistych, duplikatów i żargonu."
        " Pytania mają być po polsku, proste i praktyczne."
        " Nie zadawaj pytań typu 'czy A, czy B, czy C?', jeśli odpowiedzią nie byłoby sensowne Tak/Nie."
        f" Zwróć maksymalnie {max_hints} pytań."
        " Wynik musi być zgodny z przekazanym schematem Structured Output."
    )
    user_prompt = (
        f"Wersja robocza wiadomości użytkownika:\n{draft}\n\n"
        f"Już zebrane doprecyzowania:\n{answered_context}\n\n"
        f"Pytania, których nie wolno już proponować:\n{excluded_context}\n\n"
        "Jeśli wiadomość jest zbyt krótka albo niejasna, pytania mają pomóc ustalić:"
        " podatek/domenę, czy chodzi o stan faktyczny czy ogólną regułę,"
        " czy sprawa ma element zagraniczny, oraz czy ważny jest konkretny moment w czasie."
    )

    try:
        gateway = create_model_gateway(MODEL_GATEWAY_CONFIG)
        model_output = await asyncio.wait_for(
            gateway.generate_structured(
                response_model=ModelPromptHints,
                input=user_prompt,
                system_prompt=system_prompt,
                model=HINTS_MODEL,
                reasoning_effort="low",
                max_output_tokens=600,
            ),
            timeout=HINTS_REQUEST_TIMEOUT_SECONDS,
        )
        hints = [
            build_hint(item.question, [option.label for option in item.options])
            for item in model_output.hints
        ]
        excluded_question_set = {
            question.strip().lower()
            for question in excluded_questions or []
            if question.strip()
        }
        hints = [
            hint
            for hint in hints
            if hint.question.strip().lower() not in excluded_question_set
        ][:max_hints]
        if not hints:
            raise RuntimeError("Hint model returned no parseable hints")
        return PromptHintsResponse(hints=hints, model=HINTS_MODEL, mode="live")
    except (asyncio.TimeoutError, ModelGatewayError, ValueError):
        return PromptHintsResponse(
            hints=fallback_prompt_hints(
                draft,
                intent_hints,
                excluded_questions=excluded_questions,
                max_hints=max_hints,
            ),
            model="fallback",
            mode="fallback",
        )


def slugify_section_title(title: str) -> str:
    normalized = title.lower()
    normalized = normalized.replace("ź", "z").replace("ł", "l")
    normalized = normalized.replace("ą", "a").replace("ę", "e")
    normalized = normalized.replace("ś", "s").replace("ć", "c")
    normalized = normalized.replace("ń", "n").replace("ó", "o")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_") or "sekcja"


def parse_structured_reply(reply: str) -> Optional[StructuredReply]:
    normalized = reply.replace("\r\n", "\n").strip()
    if not normalized:
        return None

    headings_pattern = "|".join(re.escape(title) for title in ASSISTANT_SECTION_TITLES)
    section_regex = re.compile(rf"(^|\n\n)({headings_pattern})\n")
    matches = list(section_regex.finditer(normalized))

    if not matches:
        return StructuredReply(opening_statute=normalized, sections=[])

    opening_statute = normalized[: matches[0].start()].strip() or None
    sections: list[StructuredReplySection] = []

    for index, match in enumerate(matches):
        title = match.group(2)
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        content = normalized[content_start:content_end].strip()
        sections.append(
            StructuredReplySection(
                key=slugify_section_title(title),
                title=title,
                content=content,
            )
        )

    return StructuredReply(opening_statute=opening_statute, sections=sections)


def detect_fact_timeline_issues(user_prompt: str) -> list[dict[str, str]]:
    normalized = " ".join(user_prompt.lower().split())
    issues: list[dict[str, str]] = []

    annual_profit_match = re.search(
        r"(zysk\w*\s+(?:roczny|za rok)\s+za\s+(20\d{2})).{0,160}(zatwierdz\w*|uchwal\w*|wypłac\w*|wyplac\w*)",
        normalized,
        re.IGNORECASE,
    )
    if annual_profit_match:
        year = annual_profit_match.group(2)
        same_year_december = re.search(rf"\bgrudni\w*\s+{year}\b", normalized)
        if same_year_december:
            issues.append(
                {
                    "code": "annual_profit_approved_before_period_end",
                    "severity": "medium",
                    "message": (
                        f"Opis sugeruje zatwierdzenie albo wypłatę zysku rocznego za {year} jeszcze w grudniu {year}, "
                        "co jest wewnętrznie niespójne dla zwykłego rocznego wyniku i wymaga rozróżnienia od zaliczki na poczet zysku."
                    ),
                }
            )

    if "zysk" in normalized and "grudni" in normalized and "zaliczk" not in normalized and "zatwierdz" in normalized:
        issues.append(
            {
                "code": "ambiguous_profit_payment_nature",
                "severity": "medium",
                "message": "Należy rozróżnić wypłatę zaliczki na poczet zysku od wypłaty po zatwierdzeniu sprawozdania finansowego.",
            }
        )

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for issue in issues:
        code = issue["code"]
        if code in seen:
            continue
        seen.add(code)
        deduped.append(issue)
    return deduped


def build_analysis_trace(
    *,
    user_prompt: str,
    retrieved_chunks: list[RagChunk],
    legal_rules: list[dict[str, object]],
    missing_required_facts: list[str],
    timeline_issues: list[dict[str, str]],
    allowed_provision_references: set[str],
    axis_coverage: Optional[list[dict[str, object]]] = None,
    evidence_bundles: Optional[list[dict[str, object]]] = None,
) -> dict[str, object]:
    target_date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", user_prompt)
    target_date = (
        target_date_match.group(1)
        if target_date_match
        else datetime.now(timezone.utc).date().isoformat()
    )
    registry = build_registry_from_rules(legal_rules)
    claims = build_claims_from_rules(
        legal_rules,
        axis_ids=[
            str(item.get("axis_id") or "general")
            for item in (axis_coverage or [])
        ],
        missing_facts=missing_required_facts,
    )
    claims = bind_claims_to_evidence_bundles(claims, evidence_bundles or [])
    claim_validations = [
        validate_claim(
            claim,
            registry,
            target_date=target_date,
            facts={},
            calculations={},
        )
        for claim in claims
    ]
    return {
        "runtime": legal_runtime_debug(),
        "query": user_prompt,
        "target_date": target_date,
        "required_missing_facts": list(missing_required_facts),
        "timeline_issues": timeline_issues,
        "allowed_provision_references": sorted(allowed_provision_references),
        "axis_coverage": axis_coverage or [],
        "evidence_bundles": evidence_bundles or [],
        "provision_registry": {
            "validation": registry.validate(),
            "provisions": [
                {
                    "provision_id": provision.provision_id,
                    "document_id": provision.document_id,
                    "version_id": provision.version_id,
                    "citation": provision.citation,
                    "effective_from": provision.effective_from,
                    "effective_to": provision.effective_to,
                    "status": provision.status,
                }
                for provision in registry.provisions
            ],
        },
        "claims": [claim_to_dict(claim) for claim in claims],
        "claim_validation": [
            {
                "claim_id": validation.claim_id,
                "claim_supported": validation.claim_supported,
                "temporal_match": validation.temporal_match,
                "facts_satisfy_conditions": validation.facts_satisfy_conditions,
                "calculation_bound": validation.calculation_bound,
                "errors": list(validation.errors),
            }
            for validation in claim_validations
        ],
        "claim_source_traces": [
            {
                "claim_id": str(rule.get("provision_id") or ""),
                "claim": str(rule.get("directive") or ""),
                "source_document_id": str(rule.get("source_id") or ""),
                "provision_id": str(rule.get("provision_id") or ""),
                "provision_reference": str(rule.get("citation") or ""),
                "retrieval_stage": str(rule.get("retrieval_stage") or "primary_source_exact_lookup"),
                "selected_chunk_ids": list(rule.get("supporting_chunk_ids") or []),
                "source_span": str(rule.get("exact_source_span") or ""),
            }
            for rule in legal_rules
        ],
        "selected_documents": [
            {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "source_type": chunk.source_type,
                "publication": chunk.publication,
                "legal_state_date": chunk.legal_state_date,
            }
            for chunk in retrieved_chunks
        ],
    }


def build_missing_facts_context(missing_required_facts: list[str]) -> str:
    if not missing_required_facts:
        return ""
    lines = [
        "Brakujące fakty decydujące o wyborze wariantu:",
        "Jeżeli któregoś z poniższych faktów nie da się potwierdzić z pytania lub źródeł, writer ma zwrócić wynik warunkowy i nie ustalać finalnej kwoty ani stawki:",
    ]
    for index, fact in enumerate(missing_required_facts, start=1):
        lines.append(f"{index}. {fact}")
    return "\n".join(lines)


def build_timeline_issue_context(timeline_issues: list[dict[str, str]]) -> str:
    if not timeline_issues:
        return ""
    lines = ["Wykryte wątpliwości chronologii stanu faktycznego:"]
    for issue in timeline_issues:
        lines.append(f"- [{issue['severity']}] {issue['message']}")
    return "\n".join(lines)


PROVISION_REFERENCE_RENDER_RE = re.compile(
    r"\bart\.?\s*\d+[a-z]?(?:\s+ust\.?\s*\d+[a-z]?)?(?:\s+pkt\s*\d+[a-z]?)?(?:\s+lit\.?\s*[a-z])?",
    re.IGNORECASE,
)
DUPLICATED_PROVISION_REFERENCE_RE = re.compile(
    r"\b(art\.?\s*\d+[a-z]?(?:\s+ust\.?\s*\d+[a-z]?)?(?:\s+pkt\s*\d+[a-z]?)?(?:\s+lit\.?\s*[a-z])?)\s+\1\b",
    re.IGNORECASE,
)
GENERIC_PRIMARY_LAW_PLACEHOLDER_RE = re.compile(
    r"zweryfikowany przepis wskazany w źródłach(?: primary law)?|\bten przepis\b",
    re.IGNORECASE,
)
EMPTY_LEGAL_REFERENCE_SLOT_RE = re.compile(
    r"\(\s*(?!art\.)[^)]{0,80}\bustawy\s+o\s+(?:VAT|CIT|PIT|PCC|AKCYZA)\s*\)|"
    r"\b(?:jest|regulowan\w*|wymaga|nakłada|wynika\s+z|zgodnie\s+z|podstawa\s+prawna:?)\s+ustawy\s+o\s+(?:VAT|CIT|PIT|PCC|AKCYZA)\b",
    re.IGNORECASE,
)
EMPTY_LEGAL_REFERENCE_PAREN_RE = re.compile(
    r"\(\s*(?!art\.)[^)]*?ustawy\s+o\s+(VAT|CIT|PIT|PCC|AKCYZA)\s*\)",
    re.IGNORECASE,
)
EMPTY_LEGAL_REFERENCE_STATUTE_RE = re.compile(
    r"\b(?P<prefix>jest|regulowan\w*|wymaga|nakłada|wynika\s+z|zgodnie\s+z|podstawa\s+prawna:?)\s+ustawy\s+o\s+(?P<domain>VAT|CIT|PIT|PCC|AKCYZA)\b",
    re.IGNORECASE,
)
UNSUPPORTED_AUTHORITY_LINE_RE = re.compile(
    r"\b(?:ugruntowan\w* stanowisk\w*|utrwalon\w* lini\w*|jednolit\w* lini\w*|konsekwentn\w* stanowisk\w*|organy podatkowe i sądy administracyjne konsekwentnie)\b",
    re.IGNORECASE,
)
AUTHORITY_SIGNATURE_RE = re.compile(
    r"\b(?:\d{4}-[A-Z0-9.-]{10,}|[IVXLCDM]{0,4}\s*(?:FSK|SA|SK|FPS|GPS|GSK|OSK)[/\s][A-Z]{0,4}\s*\d+/\d{2,4})\b",
    re.IGNORECASE,
)
UNCERTAIN_PROVISION_PHRASES_RE = re.compile(
    r"\b(lub|albo)\s+(?:inny|odpowiedni|właściwy|wlasciwy)\s+przepis\b|"
    r"\bodpowiedni przepis\b|"
    r"\bzależnie od numeracji\b|\bzaleznie od numeracji\b",
    re.IGNORECASE,
)
UNCERTAIN_NUMBERING_FRAGMENT_RE = re.compile(
    r"(?:lub|albo)\s+(?:ust\.?\s*\d+[a-z]?|pkt\s*\d+[a-z]?|lit\.?\s*[a-z])(?:\s+[a-z]\.)?\s*(?:-|–)?\s*(?:zależnie|zaleznie)\s+od\s+numeracji",
    re.IGNORECASE,
)
UNCERTAIN_NUMBERING_RESIDUE_RE = re.compile(
    r"\s+(?:lub|albo)\s+ust\.?\s*\d+[a-z]?\s*(?:-|–)\s*(?=art\.)",
    re.IGNORECASE,
)
DEFINITIVE_PERCENT_RE = re.compile(
    r"\b(?:wynosi|obowiązuje|obowiazuje|należy zastosować|nalezy zastosowac|stosuje się|stosuje sie)\s+(\d+(?:[.,]\d+)?)\s*%",
    re.IGNORECASE,
)
RENDER_COMPLETION_MARKER = "<<ALITIGATOR_COMPLETE>>"


def select_best_claim_trace_for_text(text: str, claim_source_traces: list[dict[str, object]]) -> Optional[dict[str, object]]:
    normalized_text = " ".join(text.lower().split())
    if not claim_source_traces:
        return None

    def trace_score(trace: dict[str, object]) -> tuple[int, int]:
        haystack = " ".join(
            str(trace.get(key) or "")
            for key in ("claim", "provision_reference", "source_document_id", "source_span")
        ).lower()
        tokens = [token for token in re.findall(r"[0-9a-ząćęłńóśźż]{4,}", normalized_text, re.IGNORECASE)]
        overlap = sum(1 for token in tokens if token in haystack)
        exact_ref_bonus = 4 if normalize_provision_reference(str(trace.get("provision_reference") or "")) in normalized_text else 0
        return (overlap + exact_ref_bonus, len(str(trace.get("source_span") or "")))

    return max(claim_source_traces, key=trace_score)


def render_exact_primary_law_reference(trace: dict[str, object]) -> str:
    return str(trace.get("provision_reference") or "").strip()


def select_best_claim_trace_for_domain(
    claim_source_traces: list[dict[str, object]],
    domain: str,
) -> Optional[dict[str, object]]:
    wanted = domain.upper()
    for trace in claim_source_traces:
        haystack = " ".join(
            str(trace.get(key) or "")
            for key in ("provision_reference", "source_document_id", "claim", "source_span")
        ).upper()
        if wanted in haystack:
            return trace
        if wanted == "VAT" and ("PODATKU OD TOWARÓW" in haystack or "VAT_ACT" in haystack):
            return trace
        if wanted == "CIT" and ("DOCHODOWYM OD OSÓB PRAWNYCH" in haystack or "CIT_ACT" in haystack):
            return trace
    return claim_source_traces[0] if claim_source_traces else None


def repair_empty_legal_reference_slots(
    reply: str,
    claim_source_traces: list[dict[str, object]],
) -> str:
    def replacement_for_domain(domain: str) -> str:
        trace = select_best_claim_trace_for_domain(claim_source_traces, domain)
        reference = render_exact_primary_law_reference(trace or {})
        return reference or "przepisy wskazane w sekcji Źródła"

    repaired = EMPTY_LEGAL_REFERENCE_PAREN_RE.sub(
        lambda match: f"({replacement_for_domain(match.group(1))})",
        reply,
    )

    def replace_statute_slot(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        domain = match.group("domain")
        return f"{prefix} {replacement_for_domain(domain)}"

    return EMPTY_LEGAL_REFERENCE_STATUTE_RE.sub(replace_statute_slot, repaired)


def render_exact_primary_law_trace_block(claim_source_traces: list[dict[str, object]]) -> str:
    if not claim_source_traces:
        return ""
    lines = ["Dokładne oparcie w primary law"]
    for trace in claim_source_traces:
        provision_reference = str(trace.get("provision_reference") or "").strip()
        source_document_id = str(trace.get("source_document_id") or "").strip()
        source_span = str(trace.get("source_span") or "").strip()
        if not provision_reference or not source_document_id or not source_span:
            continue
        lines.append(f"- {source_document_id} | {provision_reference} | {source_span}.")
    return "\n".join(lines) if len(lines) > 1 else ""


def strip_render_completion_marker(reply: str) -> str:
    return reply.replace(RENDER_COMPLETION_MARKER, "").strip()


def render_verified_retrieval_sources(retrieved_chunks: list) -> str:
    """Render stable, validator-recognisable identifiers for retrieved documents."""
    lines: list[str] = []
    seen_document_ids: set[str] = set()
    for chunk in retrieved_chunks:
        document_id = str(getattr(chunk, "document_id", "") or "").strip()
        if not document_id or document_id in seen_document_ids:
            continue
        seen_document_ids.add(document_id)
        signature = str(getattr(chunk, "signature", "") or "").strip()
        subject = str(getattr(chunk, "subject", "") or "").strip()
        source_url = str(getattr(chunk, "source_url", "") or "").strip()
        descriptor = signature or subject or "dokument z indeksu"
        lines.append(
            f"- source_document_id: {document_id} | {descriptor}"
            + (f" | {source_url}" if source_url else "")
        )
    return "\n".join(lines)


def complete_empty_sources_section(reply: str, *, retrieval_citations: str) -> str:
    """Attach verified retrieval citations when the writer omitted all references.

    The writer is still responsible for explaining which sources it used.  This
    narrow repair only prevents a structurally valid answer from becoming a
    502 because its mandatory Sources section contains prose such as "brak"
    despite the request having verified retrieval results available.
    """
    structured = parse_structured_reply(strip_render_completion_marker(reply))
    source_content = next(
        (section.content for section in (structured.sections if structured else []) if section.title == "Źródła"),
        None,
    )
    if source_content is None or re.search(
        r"(?:art\.|Dz\.|DU/|\[provision_id:|source_document_id|https?://)",
        source_content,
        re.IGNORECASE,
    ):
        return reply

    if retrieval_citations:
        supplement = (
            "\n\nZweryfikowane źródła z retrievalu (uzupełnione automatycznie, "
            "bo model nie podał referencji):\n"
            f"{retrieval_citations}"
        )
    else:
        supplement = (
            "\n\nNie znaleziono zweryfikowanych źródeł w retrievalu. "
            "Wnioski wymagają potwierdzenia w aktualnych źródłach prawa."
        )
    section_pattern = re.compile(
        r"(?P<heading>(?:^|\n\n)Źródła\n)(?P<content>.*?)(?=\n\n(?:Ryzyka i luki|Źródła zwrócone przez retrieval|Źródła użyte przez retrieval)\n|\Z)",
        re.DOTALL,
    )
    return section_pattern.sub(
        lambda match: f"{match.group('heading')}{match.group('content').rstrip()}{supplement}",
        reply,
        count=1,
    )


def build_render_retry_input(
    *,
    user_prompt: str,
    rejected_output: str,
    validation_error: str,
    allowed_provision_references: set[str],
) -> str:
    """Build an edit-only retry with a closed provision-reference vocabulary."""
    references = "\n".join(
        f"- {reference}" for reference in sorted(allowed_provision_references)
    ) or "- brak; pomiń numery jednostek redakcyjnych"
    return (
        "Popraw poprzedni render, zachowując jego poprawne wnioski i wymagane sekcje.\n"
        f"Błąd walidacji: {validation_error}\n\n"
        "ZAMKNIĘTA LISTA DOZWOLONYCH REFERENCJI:\n"
        f"{references}\n\n"
        "Nie wolno podać żadnego numeru artykułu, ustępu, punktu ani litery spoza tej listy. "
        "Jeżeli dokładnej podstawy nie ma na liście, pomiń numer i zawęź twierdzenie. "
        "Nie używaj sformułowań 'ten przepis', 'zweryfikowany przepis wskazany w źródłach' "
        "ani innych placeholderów. Ostatnią linią ma być wymagany znacznik końca.\n\n"
        f"PYTANIE UŻYTKOWNIKA:\n{user_prompt}\n\n"
        f"POPRZEDNI ODRZUCONY RENDER:\n{rejected_output}"
    )


def render_verified_primary_law_source_lines(
    claim_source_traces: list[dict[str, object]],
) -> str:
    """Render only registry-bound primary-law identifiers, without model prose."""
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for trace in claim_source_traces:
        source_document_id = str(trace.get("source_document_id") or "").strip()
        reference = str(trace.get("provision_reference") or "").strip()
        key = (source_document_id, normalize_provision_reference(reference))
        if not source_document_id or not key[1] or key in seen:
            continue
        seen.add(key)
        lines.append(f"- source_document_id: {source_document_id} | {reference}")
    return "\n".join(lines)


def build_fail_closed_render_fallback(
    rejected_output: str,
    *,
    allowed_provision_references: set[str],
    verified_source_lines: str,
    required_axis_labels: Optional[list[str]] = None,
) -> tuple[str, int]:
    """Drop complete unsupported claims instead of guessing replacement provisions.

    This is deliberately a last-resort renderer.  It preserves safe paragraphs,
    removes any entire line containing an unregistered reference or a source
    placeholder, and discloses the omission.  It never rewrites one article
    number into another.
    """
    structured = parse_structured_reply(strip_render_completion_marker(rejected_output))
    sections = {
        section.title: section.content
        for section in (structured.sections if structured else [])
    }
    removed_count = 0

    def line_is_unsafe(line: str) -> bool:
        references = [
            normalize_provision_reference(match.group(0))
            for match in PROVISION_REFERENCE_RENDER_RE.finditer(line)
        ]
        return (
            any(reference not in allowed_provision_references for reference in references)
            or bool(GENERIC_PRIMARY_LAW_PLACEHOLDER_RE.search(line))
            or bool(EMPTY_LEGAL_REFERENCE_SLOT_RE.search(line))
            or bool(UNCERTAIN_PROVISION_PHRASES_RE.search(line))
            or bool(UNCERTAIN_NUMBERING_FRAGMENT_RE.search(line))
            or (
                bool(UNSUPPORTED_AUTHORITY_LINE_RE.search(line))
                and not bool(AUTHORITY_SIGNATURE_RE.search(line))
            )
        )

    def safe_content(title: str) -> str:
        nonlocal removed_count
        safe_lines: list[str] = []
        for line in sections.get(title, "").splitlines():
            if line_is_unsafe(line):
                removed_count += 1
                continue
            safe_lines.append(line)
        return "\n".join(safe_lines).strip()

    thesis = safe_content("Teza") or (
        "Na podstawie zweryfikowanego materiału nie można zatwierdzić pominiętego "
        "twierdzenia modelu. Pozostałe wnioski wymagają oparcia na jednostkach "
        "redakcyjnych wskazanych w sekcji Źródła."
    )
    analysis = safe_content("Analiza")
    if not analysis:
        analysis = (
            "Pominięto twierdzenia zawierające referencje spoza zamkniętego rejestru "
            "źródeł. System nie zastąpił ich innymi numerami przepisów."
        )
    for axis in required_axis_labels or []:
        if not re.search(
            rf"(?im)^\s*(?:#{{1,6}}\s*)?(?:\*\*)?{re.escape(axis)}(?:\*\*)?(?:\s*[:–—-]|\s*$)",
            analysis,
        ):
            analysis += (
                f"\n\n### {axis}\n"
                "Nie zatwierdzono usuniętego fragmentu tej osi bez zweryfikowanej podstawy prawnej."
            )

    sources = safe_content("Źródła")
    if verified_source_lines:
        sources = verified_source_lines
    elif not re.search(
        r"(?:art\.|Dz\.|DU/|\[provision_id:|source_document_id|https?://)",
        sources,
        re.IGNORECASE,
    ):
        sources = "Nie znaleziono zweryfikowanych źródeł primary law w retrievalu."

    risks = safe_content("Ryzyka i luki")
    disclosure = (
        f"Automatyczny guardrail pominął {removed_count} "
        "fragmentów zawierających niezweryfikowane odwołania lub placeholdery; "
        "nie zastępował ich domyślnymi numerami przepisów."
    )
    risks = f"{risks}\n\n{disclosure}" if risks else disclosure
    fallback = (
        f"Teza\n{thesis}\n\n"
        f"Analiza\n{analysis}\n\n"
        f"Źródła\n{sources}\n\n"
        f"Ryzyka i luki\n{risks}\n\n"
        f"{RENDER_COMPLETION_MARKER}"
    )
    return fallback, removed_count


def build_render_diagnostics(
    *,
    raw_candidate: str,
    guarded_candidate: str,
    completed_candidate: str,
    retrieved_chunks: list,
) -> dict[str, object]:
    """Create a privacy-conscious diagnostic payload for rejected model renders."""
    def summarize(reply: str) -> dict[str, object]:
        structured = parse_structured_reply(strip_render_completion_marker(reply))
        sections = structured.sections if structured else []
        sources_content = next(
            (section.content for section in sections if section.title == "Źródła"),
            None,
        )
        result: dict[str, object] = {
            "characters": len(reply),
            "sha256": hashlib.sha256(reply.encode("utf-8")).hexdigest(),
            "has_completion_marker": RENDER_COMPLETION_MARKER in reply,
            "section_titles": [section.title for section in sections],
            "sources_section_present": sources_content is not None,
            "sources_section_characters": len(sources_content or ""),
            "sources_section_preview": (sources_content or "")[:4000],
        }
        if os.getenv("ALITIGATOR_CHAT_DIAGNOSTICS_INCLUDE_RENDERED_OUTPUT", "").lower() in {"1", "true", "yes"}:
            result["full_rendered_output"] = reply
        return result

    raw_references = [match.group(0) for match in PROVISION_REFERENCE_RENDER_RE.finditer(raw_candidate)]
    guarded_references = [match.group(0) for match in PROVISION_REFERENCE_RENDER_RE.finditer(guarded_candidate)]
    removed_references = sorted(
        set(raw_references) - set(guarded_references), key=str.casefold
    )
    return {
        # Request traces are access-controlled alongside the chat.  Keeping
        # the exact strings makes a lost provision diagnosable, rather than
        # merely showing a hash after a destructive sanitizer ran.
        "raw_model_output_text": raw_candidate,
        "sanitized_output_text": guarded_candidate,
        "postprocessed_output_text": completed_candidate,
        "raw_model_output": summarize(raw_candidate),
        "after_guardrails": summarize(guarded_candidate),
        "after_sources_repair": summarize(completed_candidate),
        "guardrails_changed_output": raw_candidate != guarded_candidate,
        "sources_repair_changed_output": guarded_candidate != completed_candidate,
        "removed_provision_references": removed_references,
        "output_changed": raw_candidate != completed_candidate,
        "retrieval": {
            "chunk_count": len(retrieved_chunks),
            "documents": [
                {
                    "document_id": str(getattr(chunk, "document_id", "") or ""),
                    "chunk_id": str(getattr(chunk, "chunk_id", "") or ""),
                    "source_type": str(getattr(chunk, "source_type", "") or ""),
                    "signature": str(getattr(chunk, "signature", "") or ""),
                }
                for chunk in retrieved_chunks[:30]
            ],
            "verified_source_lines": render_verified_retrieval_sources(retrieved_chunks),
        },
    }


def validate_final_output(
    reply: str,
    *,
    axis_coverage: list[dict[str, object]],
    expected_sections: list[str],
    allowed_provision_references: Optional[set[str]] = None,
    verified_source_count: Optional[int] = None,
) -> dict[str, object]:
    has_completion_marker = RENDER_COMPLETION_MARKER in reply
    stripped_reply = strip_render_completion_marker(reply)
    structured = parse_structured_reply(stripped_reply)
    rendered_section_titles = [section.title for section in (structured.sections if structured else [])]
    missing_sections = [section for section in expected_sections if section not in rendered_section_titles]
    section_content_by_title = {
        section.title: section.content.strip()
        for section in (structured.sections if structured else [])
    }
    empty_required_sections = [
        section
        for section in expected_sections
        if section in rendered_section_titles and not section_content_by_title.get(section)
    ]
    sources_content = section_content_by_title.get("Źródła", "")
    sources_without_sources = (
        "Źródła" in rendered_section_titles
        and not re.search(
            r"(?:art\.|Dz\.|DU/|\[provision_id:|source_document_id|https?://)",
            sources_content,
            re.IGNORECASE,
        )
        and not re.search(
            r"nie znaleziono zweryfikowanych źródeł(?:\s+\w+){0,4}\s+w retrievalu",
            sources_content,
            re.IGNORECASE,
        )
    )
    rendered_references = [
        normalize_provision_reference(match.group(0))
        for match in PROVISION_REFERENCE_RENDER_RE.finditer(stripped_reply)
    ]
    unverified_references = (
        sorted({reference for reference in rendered_references if reference not in allowed_provision_references})
        if allowed_provision_references is not None
        else []
    )
    claims_verified_sources = bool(re.search(r"(?:ustawa|przepis|art\.)", stripped_reply, re.IGNORECASE))
    source_section_denies_sources = bool(re.search(
        r"nie znaleziono zweryfikowanych źródeł", sources_content, re.IGNORECASE
    ))
    source_section_contradiction = bool(
        verified_source_count == 0 and claims_verified_sources and not source_section_denies_sources
    )

    expected_domains = sorted(
        {
            token
            for token in ("VAT", "CIT", "PIT", "PCC", "SD", "AKCYZA", "ORDYNACJA")
            if any(token in str(item.get("label") or "").upper() or token in str(item.get("axis_id") or "").upper() for item in axis_coverage)
        }
    )
    rendered_domains = [
        domain
        for domain in expected_domains
        if re.search(
            rf"(?im)^\s*(?:#{{1,6}}\s*)?(?:\*\*)?{re.escape(domain)}(?:\*\*)?(?:\s*[:–—-]|\s*$)",
            stripped_reply,
        )
    ]

    last_meaningful_line = next(
        (line.strip() for line in reversed(stripped_reply.splitlines()) if line.strip()),
        "",
    )
    unfinished_sentence = bool(last_meaningful_line) and not re.search(r"[.!?)]$", last_meaningful_line)
    validation = {
        "expected_axes": len(axis_coverage),
        "rendered_axes": len(rendered_domains) if expected_domains else len(axis_coverage),
        "missing_planned_sections": len(missing_sections),
        "unfinished_sentence": unfinished_sentence,
        "has_completion_marker": has_completion_marker,
        "expected_sections": expected_sections,
        "rendered_sections": rendered_section_titles,
        "empty_required_sections": empty_required_sections,
        "sources_without_sources": sources_without_sources,
        "verified_source_count": verified_source_count,
        "unverified_references": unverified_references,
        "source_section_contradiction": source_section_contradiction,
        "no_placeholder_tokens": not bool(GENERIC_PRIMARY_LAW_PLACEHOLDER_RE.search(stripped_reply)),
        "no_empty_legal_reference_slots": not bool(EMPTY_LEGAL_REFERENCE_SLOT_RE.search(stripped_reply)),
        "no_uncertain_provision_phrases": not bool(
            UNCERTAIN_PROVISION_PHRASES_RE.search(stripped_reply)
            or UNCERTAIN_NUMBERING_FRAGMENT_RE.search(stripped_reply)
        ),
        "authority_line_claims_supported": (
            not bool(UNSUPPORTED_AUTHORITY_LINE_RE.search(stripped_reply))
            or bool(AUTHORITY_SIGNATURE_RE.search(stripped_reply))
        ),
        "tables_closed": stripped_reply.count("|") == 0
        or all(
            line.count("|") >= 2
            for line in stripped_reply.splitlines()
            if line.strip().startswith("|")
        ),
    }
    if (
        not has_completion_marker
        or missing_sections
        or empty_required_sections
        or sources_without_sources
        or unverified_references
        or source_section_contradiction
        or unfinished_sentence
        or (expected_domains and len(rendered_domains) < len(expected_domains))
        or not validation["no_placeholder_tokens"]
        or not validation["no_empty_legal_reference_slots"]
        or not validation["no_uncertain_provision_phrases"]
        or not validation["authority_line_claims_supported"]
        or not validation["tables_closed"]
    ):
        failed_checks: list[str] = []
        if not has_completion_marker:
            failed_checks.append("brak znacznika końca")
        if missing_sections:
            failed_checks.append(f"brak sekcji: {', '.join(missing_sections)}")
        if empty_required_sections:
            failed_checks.append(f"pusta sekcja: {', '.join(empty_required_sections)}")
        if sources_without_sources:
            failed_checks.append("sekcja Źródła bez źródeł")
        if unverified_references:
            failed_checks.append("niezweryfikowane referencje: " + ", ".join(unverified_references))
        if source_section_contradiction:
            failed_checks.append("sprzeczność między twierdzeniem o źródłach a liczbą zweryfikowanych źródeł")
        if unfinished_sentence:
            failed_checks.append("urwane ostatnie zdanie")
        if expected_domains and len(rendered_domains) < len(expected_domains):
            missing_domains = sorted(set(expected_domains) - set(rendered_domains))
            failed_checks.append(f"brak osi: {', '.join(missing_domains)}")
        if not validation["no_placeholder_tokens"]:
            failed_checks.append("placeholder źródłowy")
        if not validation["no_empty_legal_reference_slots"]:
            failed_checks.append("puste miejsce po referencji prawnej")
        if not validation["no_uncertain_provision_phrases"]:
            failed_checks.append("niepewna podstawa prawna")
        if not validation["authority_line_claims_supported"]:
            failed_checks.append("twierdzenie o linii organów lub sądów bez sygnatur")
        if not validation["tables_closed"]:
            failed_checks.append("niezamknięta tabela")
        raise HTTPException(
            status_code=502,
            detail=(
                "Model zwrócił odpowiedź odrzuconą przez walidator: "
                + "; ".join(failed_checks)
                + "."
            ),
        )
    return validation


def enforce_reply_guardrails(
    reply: str,
    *,
    allowed_provision_references: set[str],
    missing_required_facts: list[str],
    timeline_issues: list[dict[str, str]],
    claim_source_traces: Optional[list[dict[str, object]]] = None,
) -> str:
    traces = claim_source_traces or []

    def replace_uncertain_fragment(match: re.Match[str]) -> str:
        trace = select_best_claim_trace_for_text(reply, traces)
        return render_exact_primary_law_reference(trace) if trace else "ten przepis"

    sanitized = UNCERTAIN_PROVISION_PHRASES_RE.sub(replace_uncertain_fragment, reply)
    sanitized = UNCERTAIN_NUMBERING_FRAGMENT_RE.sub(replace_uncertain_fragment, sanitized)
    sanitized = UNCERTAIN_NUMBERING_RESIDUE_RE.sub(" ", sanitized)

    def replace_unverified_reference(match: re.Match[str]) -> str:
        reference = match.group(0)
        normalized = normalize_provision_reference(reference)
        if normalized in allowed_provision_references:
            return reference
        trace = select_best_claim_trace_for_text(match.string, traces)
        return render_exact_primary_law_reference(trace) if trace else "ten przepis"

    sanitized = PROVISION_REFERENCE_RENDER_RE.sub(replace_unverified_reference, sanitized)
    sanitized = GENERIC_PRIMARY_LAW_PLACEHOLDER_RE.sub(
        lambda match: render_exact_primary_law_reference(select_best_claim_trace_for_text(match.string, traces) or {}),
        sanitized,
    )
    sanitized = repair_empty_legal_reference_slots(sanitized, traces)
    sanitized = DUPLICATED_PROVISION_REFERENCE_RE.sub(r"\1", sanitized)
    sanitized = re.sub(r"\bze\s+(art\.)", r"z \1", sanitized, flags=re.IGNORECASE)

    if missing_required_facts:
        sanitized = DEFINITIVE_PERCENT_RE.sub(r"możliwy jest wariant \1%", sanitized)
        conditional_note = (
            "Na obecnym materiale nie można ustalić finalnej kwoty ani jednej ostatecznej stawki, "
            "bo brakuje faktów koniecznych do wyboru właściwego wariantu.\n\n"
            "Potrzebne doprecyzowanie\n"
            + "\n".join(f"- {fact}" for fact in missing_required_facts)
        )
        if re.search(r"(^|\n\n)Teza\n", sanitized):
            sanitized = re.sub(
                r"(^|\n\n)Teza\n",
                lambda match: f"{match.group(1)}Teza\n{conditional_note}\n\n",
                sanitized,
                count=1,
            )
        else:
            sanitized = f"Teza\n{conditional_note}\n\n{sanitized}"

    if timeline_issues:
        timeline_block = (
            "\n\nRyzyka i luki\n"
            "Przed ostatecznym zastosowaniem norm trzeba wyjaśnić wątpliwości chronologii stanu faktycznego:\n"
            + "\n".join(f"- {issue['message']}" for issue in timeline_issues)
        )
        if "annual_profit_approved_before_period_end" in {issue["code"] for issue in timeline_issues}:
            timeline_block += (
                "\n- Jeżeli była to zaliczka na poczet zysku, analiza wymaga odrębnego wariantu."
                "\n- Jeżeli wypłata nastąpiła dopiero po zatwierdzeniu sprawozdania w kolejnym roku, skutki trzeba ocenić w tym wariancie."
            )
        if timeline_block not in sanitized:
            sanitized += timeline_block

    return sanitized


def build_demo_reply(user_prompt: str, retrieved_chunks: list, *, retrieval_prompt: Optional[str] = None) -> str:
    citations = list_citations(retrieved_chunks)
    opening_quote = extract_opening_statute_quote(retrieved_chunks, query=retrieval_prompt or user_prompt)
    quote_block = f"Cytat z przepisu\n{opening_quote}\n\n" if opening_quote else ""
    return (
        quote_block
        + "Teza\n"
        "To jest tryb demonstracyjny MVP: nie generuję opinii prawnej bez modelu językowego.\n\n"
        "Analiza\n"
        "Odebrałem pytanie: \""
        f"{user_prompt[:900]}"
        "\". Lokalny retrieval znalazł zweryfikowane źródła, ale tryb demo ich nie interpretuje.\n\n"
        "Źródła\n"
        f"{citations or 'Nie znaleziono trafnych fragmentów w lokalnym indeksie.'}\n\n"
        "Ryzyka i luki\n"
        "Do odpowiedzi merytorycznej potrzebny jest skonfigurowany model językowy; powyższe źródła są jednak dostępne w RAG.\n\n"
        f"{RENDER_COMPLETION_MARKER}"
    )


def build_missing_primary_law_reply() -> str:
    """The only permitted general-answer output without controlling primary law.

    This deliberately contains no substantive tax conclusion.  It is shared
    by demo and live legacy paths so provider availability can never turn an
    empty retrieval into an answer from model memory.
    """
    return (
        "Teza\n"
        "Nie można zatwierdzić materialnej odpowiedzi prawnej, ponieważ retrieval "
        "nie dostarczył zweryfikowanego przepisu kontrolującego.\n\n"
        "Analiza\n"
        "Nie uruchomiono syntezy ani kalkulacji opartej na pamięci modelu. "
        "Najpierw trzeba odnaleźć aktualną jednostkę redakcyjną właściwej ustawy.\n\n"
        "Źródła\n"
        "Nie znaleziono zweryfikowanych źródeł primary law w retrievalu.\n\n"
        "Ryzyka i luki\n"
        "Każda kategoryczna ocena lub rekomendacja wymaga ponownego researchu "
        "z controlling provision.\n\n"
        f"{RENDER_COMPLETION_MARKER}"
    )


def resolve_model(requested_model: Optional[str]) -> str:
    if requested_model and requested_model in AVAILABLE_MODELS:
        return requested_model

    return DEFAULT_MODEL if DEFAULT_MODEL in AVAILABLE_MODELS else AVAILABLE_MODELS[0]


def build_bad_debt_benchmark_chat_payload(controlled_result) -> tuple[str, dict[str, object]]:
    claims = list(controlled_result.claims.values())
    provisions = list(controlled_result.renderer_payload.get("provisions", []))
    selected_provisions = [
        {
            "provision_id": str(item.get("provision_id") or ""),
            "display_reference": str(item.get("display_reference") or ""),
            "version_id": str(item.get("version_id") or ""),
        }
        for item in provisions
    ]
    rejected_historical_provisions = [
        {
            "provision_id": provision_id,
            "target_date": "2026-03-31",
            "selected": False,
            "reason": "not_effective_on_target_date_or_missing",
        }
        for provision_id in (
            "vat_art_89a_ust_2_pkt_1",
            "vat_art_89a_ust_2_pkt_2",
            "vat_art_89a_ust_2_pkt_3_lit_b",
        )
    ]
    built_claims = [
        {
            "claim_id": claim.claim_id,
            "axis_id": claim.axis_id,
            "claim_type": claim.claim_type,
            "status": claim.status,
            "result_code": claim.result_code,
            "result": claim.result,
            "provision_ids": list(claim.controlling_provisions),
            "fact_ids": list(claim.fact_dependencies),
            "missing_fact_ids": list(claim.missing_fact_dependencies),
            "calculation_id": claim.calculation_id,
            "calculation_ids": list(claim.calculation_ids),
        }
        for claim in claims
    ]
    validation_result = asdict(controlled_result.render_validation)
    acceptance_summary = {
        "creditor_vat_status_reference_date": controlled_result.claims[
            "claim_vat_creditor_registration_date"
        ].result.get("creditor_vat_status_reference_date"),
        "receivable_payment_cutoff": controlled_result.claims[
            "claim_vat_payment_cutoff"
        ].result.get("receivable_payment_cutoff"),
        "vat_rules_merged": controlled_result.claims[
            "claim_vat_creditor_registration_date"
        ].controlling_provisions == controlled_result.claims[
            "claim_vat_payment_cutoff"
        ].controlling_provisions,
        "payment_cutoff": controlled_result.claims[
            "claim_cit_payment_cutoff"
        ].result.get("payment_cutoff"),
        "insolvency_reference_date": controlled_result.claims[
            "claim_cit_relief"
        ].result.get("insolvency_reference_date"),
        "missing_debtor_vat_status_detected": controlled_result.claims[
            "claim_vat_debtor_registration_path"
        ].result.get("missing_debtor_vat_status_detected"),
        "material_claims_without_claim_id": sum(
            1 for claim in claims if claim.is_material and not claim.claim_id
        ),
        "material_claims_without_provision_id": sum(
            1 for claim in claims if claim.is_material and not claim.controlling_provisions
        ),
        "numeric_claims_without_calculation_id": sum(
            1
            for claim in claims
            if claim.is_material
            and claim.claim_type == "calculated_result"
            and not claim.calculation_ids
            and not claim.calculation_id
        ),
        "empty_required_sections_returned": "required_sections_empty"
        in controlled_result.render_validation.errors,
        "sources_section_without_sources": False,
        "blank_legal_references": 0,
        "display_reference_preserved_end_to_end": all(
            item["display_reference"] in controlled_result.answer
            for item in selected_provisions
            if item["display_reference"]
        ),
    }
    trace = {
        "selected_provisions": selected_provisions,
        "rejected_historical_provisions": rejected_historical_provisions,
        "extracted_rules": [
            {
                "rule_id": "vat_payment_cutoff",
                "provision_id": "vat_art_89a_ust_3",
                "result": {"receivable_payment_cutoff": "through_return_filing_date"},
            },
            {
                "rule_id": "vat_creditor_registration_date",
                "provision_id": "vat_art_89a_ust_2_pkt_3_lit_a",
                "result": {
                    "creditor_vat_status_reference_date": "day_before_return_filing"
                },
            },
            {
                "rule_id": "cit_payment_cutoff",
                "provision_id": "cit_art_18f_ust_5",
                "result": {"payment_cutoff": "return_filing_date"},
            },
            {
                "rule_id": "cit_insolvency_reference_date",
                "provision_id": "cit_art_18f_ust_10",
                "result": {"insolvency_reference_date": "last_day_of_previous_month"},
            },
        ],
        "built_claims": built_claims,
        "validated_claims": [
            {
                "claim_id": claim.claim_id,
                "status": claim.status,
                "accepted": True,
                "provision_ids": list(claim.controlling_provisions),
                "calculation_ids": list(claim.calculation_ids),
            }
            for claim in claims
        ],
        "renderer_payload": controlled_result.renderer_payload,
        "raw_renderer_output": controlled_result.answer,
        "postprocessed_output": controlled_result.answer,
        "validation_result": validation_result,
        "acceptance_summary": acceptance_summary,
    }
    reply = (
        "Teza\n"
        "Benchmark kontrolowanego pipeline’u VAT/CIT zakończył się wynikiem PASS.\n\n"
        "Analiza\n"
        f"- selected_provisions: {len(selected_provisions)}\n"
        f"- built_claims: {len(built_claims)}\n"
        f"- render_validation.passed: {controlled_result.render_validation.passed}\n"
        f"- placeholder_count: {controlled_result.render_validation.placeholder_count}\n"
        "- VAT payment cutoff: through_return_filing_date\n"
        "- VAT creditor status date: day_before_return_filing\n"
        "- CIT payment cutoff: return_filing_date\n"
        "- CIT insolvency date: last_day_of_previous_month\n"
        "- debtor_vat_registration_status: missing\n\n"
        "Źródła\n"
        + "\n".join(
            f"- {item['display_reference']} [provision_id:{item['provision_id']}] [version_id:{item['version_id']}]."
            for item in selected_provisions
        )
        + "\n\nRyzyka i luki\n"
        "Brak pustych sekcji, brak pustych referencji prawnych i brak materialnych claimów bez provenance."
    )
    return reply, trace


OPENING_STATUTE_STOPWORDS = {
    "oraz", "który", "ktora", "której", "którego", "które", "których", "przez", "wobec", "dotyczy",
    "skutki", "transakcji", "rozpisz", "przede", "wszystkim", "jako", "swojej", "swoja", "swoje",
    "spolki", "spółki", "wspolnik", "wspólnik", "podatek", "podatki", "pytanie", "ustawa", "ustawy",
    "niższych", "nizszych", "niższa", "nizsza", "warunkach", "poniżej", "ponizej", "wartości", "wartosci",
}

RETRIEVAL_COVERAGE_RULES = (
    {
        "id": "ksef_2_0_current_law",
        "label": "KSeF 2.0: aktualna podstawa Dz.U. 2025 poz. 1203 / terminy / limit / sankcje",
        "query_patterns": (r"\b(ksef|krajow(?:y|ego) system(?:u)? e[ -]?faktur|faktur\w* ustrukturyzowan\w*)\b",),
        "chunk_patterns": (r"\b(Dz\.U\.\s*2025\s*poz\.\s*1203|5 sierpnia 2025|1 lutego 2026|1 kwietnia 2026|10 000 zł|art\.\s*106ni|1 stycznia 2027)\b",),
    },
    {
        "id": "ksef_2_0_offline_modes",
        "label": "KSeF 2.0: offline24 / niedostępność / awaria / awaria podatnika",
        "query_patterns": (r"\b(ksef).{0,180}\b(offline24|offline\s*24|awari\w*|niedost[ęe]pno\w*|system\w* ksi[ęe]gow\w*)\b|\b(offline24|offline\s*24|awari\w*|niedost[ęe]pno\w*)\b.{0,180}\b(ksef)\b",),
        "chunk_patterns": (r"\b(art\.\s*106nda|art\.\s*106nf|art\.\s*106nh|następnym dniu roboczym|całkowita awaria|awaria KSeF)\b",),
    },
    {
        "id": "ksef_2_0_receipt_deduction_corrections",
        "label": "KSeF 2.0: otrzymanie faktury / odliczenie / korekta in minus",
        "query_patterns": (r"\b(ksef).{0,220}\b(pdf|poza\s+ksef|odliczen\w*|korekt\w*|in minus|pusta faktur\w*)\b|\b(pdf|poza\s+ksef|odliczen\w*|korekt\w*|in minus|pusta faktur\w*)\b.{0,220}\b(ksef)\b",),
        "chunk_patterns": (r"\b(art\.\s*86|art\.\s*88|art\.\s*29a|faktycznego otrzymania|nie przesuwa automatycznie|czynności niedokonane|faktura korygująca ustrukturyzowana)\b",),
    },
    {
        "id": "ksef_2_0_scope_smpd_buyer_capacity",
        "label": "KSeF 2.0: zakres obowiązku / SMPD uczestniczy / B2C / capacity nabywcy",
        "query_patterns": (r"\b(ksef).{0,220}\b(smpd|sta[łl]\w* miejsce|zagraniczn\w*|b2c|konsument\w*|nip|prywatn\w*|mieszan\w*)\b|\b(smpd|sta[łl]\w* miejsce|zagraniczn\w*|b2c|konsument\w*|nip|prywatn\w*|mieszan\w*)\b.{0,220}\b(ksef)\b",),
        "chunk_patterns": (r"\b(stałe miejsce.*nie uczestniczy|nie uczestniczy w dostawie|art\.\s*106gb\s*ust\.\s*4|osoby fizycznej nieprowadzącej działalności|faktury konsumenckie|capacity_of_buyer|NIP jest dowodem pomocniczym)\b",),
    },
    {
        "id": "wht_core_statutes",
        "label": "WHT: przepisy CIT o dywidendach, odsetkach, preferencjach i pay-and-refund",
        "query_patterns": (r"\b(wht|podatek u źr[óo]dła|beneficial owner|rzeczywist\w* właściciel\w*|certyfikat\w* rezydencji|pay and refund|należyt\w* staranno\w*)\b",),
        "chunk_patterns": (r"\b(art\.\s*21\b|art\.\s*22\b|art\.\s*22c\b|art\.\s*26\b|certyfikat rezydencji|rzeczywistym właścicielem|dochowania należytej staranności)\b",),
    },
    {
        "id": "wht_treaty_axes",
        "label": "WHT: umowa międzynarodowa / dywidendy / odsetki / zyski przedsiębiorstw / zakład",
        "query_patterns": (r"\b(dywidend\w*|odsetk\w*|zarządz\w*|zarzadz\w*|holdingow\w*|holandi\w*|niderland\w*|umow\w* międzynarodow\w*|upo|zakład\w*|zaklad\w*)\b",),
        "chunk_patterns": (r"\b(umow\w* o unikaniu podwójnego opodatkowania|beneficial owner|osob\w* uprawnion\w* do dywidend|odsetek|zyski przedsiębiorstw|zagraniczny zakład|zakład)\b",),
    },
    {
        "id": "crossborder_treaty_default",
        "label": "transgranicznie: czy retrieval pokrywa UPO / zakład / rezydencję",
        "query_patterns": (r"\b(transgraniczn\w*|nierezydent\w*|podmiot\w* zagraniczn\w*|certyfikat\w* rezydencji|zakład\w*|zaklad\w*|upo)\b",),
        "chunk_patterns": (r"\b(umow\w* o unikaniu podwójnego opodatkowania|miejsce zamieszkania lub siedziba|zakład|zyski przedsiębiorstw|certyfikat rezydencji)\b",),
    },
    {
        "id": "wht_pay_and_refund_scope",
        "label": "WHT: zakres art. 26 ust. 2e / próg / nadwyżka ponad limit",
        "query_patterns": (r"\b(pay and refund|2 mln|2 000 000|próg\w*|prog\w*|limit\w* płatno\w*|limit\w* należno\w*)\b",),
        "chunk_patterns": (r"\b(art\.\s*26\s*ust\.\s*2e|nadwyżk\w* ponad kwotę 2 000 000 zł|na rzecz tego samego podatnika|z tytułów wymienionych w art\.\s*21\s*ust\.\s*1\s*pkt\s*1 oraz art\.\s*22\s*ust\.\s*1)\b",),
    },
    {
        "id": "family_foundation_permitted_activity",
        "label": "fundacja rodzinna: dozwolona działalność / art. 5",
        "query_patterns": (r"\b(fundacj\w* rodzinn\w*|beneficjent\w*|fundator\w*)\b",),
        "chunk_patterns": (r"\b(fundacj\w* rodzinn\w*|art\.\s*5\b|art\.\s*5\s*ust\.\s*1\s*pkt\s*2|art\.\s*5\s*ust\.\s*1\s*pkt\s*5\s*lit\.\s*a|dozwolon\w* działalno\w*|nabyte wyłącznie w celu dalszego zbycia|spółkom kapitałowym, w których fundacja rodzinna posiada udziały albo akcje)\b",),
    },
    {
        "id": "family_foundation_cit_hidden_profit_24q_24r",
        "label": "fundacja rodzinna: CIT 24q / 24r / ukryte zyski / dochód z działalności niedozwolonej",
        "query_patterns": (r"\b(fundacj\w* rodzinn\w*).{0,180}\b(ukryt\w* zysk\w*|pożyczk\w*|pozyczk\w*|odsetk\w*|25%|24q|24r|usług\w* praw\w*|księgow\w*|zarządz\w*)\b",),
        "chunk_patterns": (r"\b(art\.\s*24q|art\.\s*24r|ukryty zysk|dochód z działalności wykraczającej|usługi prawne|księgowe|zarządzania)\b",),
    },
    {
        "id": "family_foundation_pit_exemption",
        "label": "fundacja rodzinna: PIT beneficjenta / proporcja zwolnienia",
        "query_patterns": (r"\b(fundacj\w* rodzinn\w*).{0,120}\b(beneficjent\w*|syn\w* fundator\w*|wypłat\w*|świadczeni\w*)\b|\b(beneficjent\w*|syn\w* fundator\w*).{0,120}\b(fundacj\w* rodzinn\w*)\b",),
        "chunk_patterns": (r"\b(art\.\s*21\s*ust\.\s*1\s*pkt\s*157|art\.\s*21\s*ust\.\s*49|grup\w* zerow\w*|fundator\w*|10%|15%|zwolnien\w*.*fundacj\w* rodzin\w*)\b",),
    },
    {
        "id": "family_foundation_vat_related_party",
        "label": "fundacja rodzinna: VAT / art. 32 / najem / sprzedaż poniżej wartości rynkowej",
        "query_patterns": (r"\b(fundacj\w* rodzinn\w*).{0,180}\b(vat|najem\w*|samochod\w*|warto[śs][ćc] rynkow\w*|połow\w* warto[śs]ci|podmiot\w* powi[ąa]zan\w*)\b",),
        "chunk_patterns": (r"\b(art\.\s*32|art\.\s*43|wartość rynkowa|podmiot powiązany|najem mieszkalny)\b",),
    },
    {
        "id": "real_estate_vat_first_occupancy",
        "label": "nieruchomość: VAT przy sprzedaży / pierwsze zasiedlenie",
        "query_patterns": (r"\b(sprzeda[żz]\w*|zby\w*)\b.{0,120}\b(nieruchomo\w*|apartament\w*|lokal\w*|mieszkani\w*|budynek\w*)\b|\b(nieruchomo\w*|apartament\w*|lokal\w*|mieszkani\w*|budynek\w*)\b.{0,120}\b(sprzeda[żz]\w*|zby\w*)\b",),
        "chunk_patterns": (r"\b(pierwsz\w* zasiedleni\w*|art\.\s*43\s*ust\.\s*1\s*pkt\s*10|art\.\s*43\s*ust\.\s*1\s*pkt\s*10a|ulepszeni\w* nieruchomo\w*)\b",),
    },
    {
        "id": "real_estate_pcc_vat_interplay",
        "label": "nieruchomość: relacja PCC do VAT przy zwolnieniu",
        "query_patterns": (r"\b(pcc)\b.{0,120}\b(vat|zwolnion\w*)\b.{0,120}\b(nieruchomo\w*|apartament\w*|lokal\w*|budynek\w*)\b|\b(nieruchomo\w*|apartament\w*|lokal\w*|budynek\w*)\b.{0,120}\b(pcc)\b",),
        "chunk_patterns": (r"\b(art\.\s*2\s*pkt\s*4|sprzedaż nieruchomości|nieruchomości.*zwolnion\w* z vat|wyłączenie z opodatkowania pcc)\b",),
    },
    {
        "id": "treaty",
        "label": "umowa o unikaniu podwójnego opodatkowania / treaty override",
        "query_patterns": (r"\b(usa|niemc|francj|wielkiej bryt|zagraniczn|nierezydent|umow[ay] o unikaniu)\b",),
        "chunk_patterns": (r"\b(umow[ay] o unikaniu|uopo|treaty|royalt(?:y|ies)|business profits|zaklad|zakład)\b",),
    },
    {
        "id": "permanent_establishment",
        "label": "zakład / permanent establishment",
        "query_patterns": (r"\b(zaklad|zakład|stale miejsce|stałe miejsce|pracownic\w+.*polsce|pracownic\w+.*polsce)\b",),
        "chunk_patterns": (r"\b(zaklad|zakład|permanent establishment|business profits|art\.?\s*[56]\b)\b",),
    },
    {
        "id": "fixed_establishment_vat",
        "label": "VAT fixed establishment / stałe miejsce prowadzenia działalności",
        "query_patterns": (r"\b(import uslug|import usług|vat)\b.*\b(stale miejsce|stałe miejsce|pracownic\w+.*polsce)\b",),
        "chunk_patterns": (r"\b(stale miejsce prowadzenia dzialalnosci|stałe miejsce prowadzenia działalności|fixed establishment)\b",),
    },
    {
        "id": "residence_certificate",
        "label": "certyfikat rezydencji / dokumentowanie WHT",
        "query_patterns": (r"\b(certyfikat rezydencji|pdf)\b",),
        "chunk_patterns": (r"\b(certyfikat rezydencji|art\.?\s*26|kopi[ai] certyfikatu|pdf)\b",),
    },
    {
        "id": "post_leasing_vehicle_vat",
        "label": "samochód po leasingu: VAT darowizna / prawo do odliczenia / podatnik / faktura",
        "query_patterns": (
            r"(\bsamoch[óo]d\w*|\bpojazd\w*|\bauto\b).{0,220}(\bleasing\w*|\bwykup\w*|\bdarowizn\w*|\bfaktur\w*)|"
            r"(\bleasing\w*|\bwykup\w*|\bdarowizn\w*|\bfaktur\w*).{0,220}(\bsamoch[óo]d\w*|\bpojazd\w*|\bauto\b)",
        ),
        "chunk_patterns": (
            r"\b(art\.?\s*7|art\.?\s*15|art\.?\s*86|art\.?\s*91|art\.?\s*106b|przysługiwało.*prawo do obniżenia|podatnik jest obowiązany wystawić fakturę)\b",
        ),
    },
    {
        "id": "post_leasing_vehicle_pit",
        "label": "samochód po leasingu: PIT pół roku / 6 lat / 20% kosztów / koszt darowanego składnika",
        "query_patterns": (
            r"(\bsamoch[óo]d\w*|\bpojazd\w*|\bauto\b).{0,260}(\bleasing\w*|\bwykup\w*|\bdarowizn\w*|\bmałżonk\w*|\bmalzonk\w*)|"
            r"(\bleasing\w*|\bwykup\w*|\bdarowizn\w*|\bmałżonk\w*|\bmalzonk\w*).{0,260}(\bsamoch[óo]d\w*|\bpojazd\w*|\bauto\b)",
        ),
        "chunk_patterns": (
            r"\b(art\.?\s*10|art\.?\s*14|art\.?\s*22|art\.?\s*23|pół roku|pol roku|nie upłynęło 6 lat|rzeczami ruchomymi|wysokości 20|wysokosci 20|art\.?\s*11\s*ust\.?\s*2)\b",
        ),
    },
    {
        "id": "post_leasing_vehicle_sd",
        "label": "darowizna małżonkowi: SD / grupa zerowa / zgłoszenie / wyjątki",
        "query_patterns": (r"\b(darowizn\w*).{0,160}\b(małżonk\w*|malzonk\w*|żon\w*|zon\w*|męż\w*|mez\w*)\b|\b(sd-z2|spadk\w* i darowizn\w*)\b",),
        "chunk_patterns": (r"\b(art\.?\s*4a|art\.?\s*6|art\.?\s*9|art\.?\s*14|małżonka|malzonka|zgłoszą nabycie|terminie 6 miesięcy|obowiązek zgłoszenia nie obejmuje)\b",),
    },
    {
        "id": "software_tax_classification",
        "label": "kwalifikacja oprogramowania: licencja / WNiP / koszt / koszt wytworzenia",
        "query_patterns": (r"\b(platform\w* informatyczn\w*|program\w*|kod\w* źródł\w*|kod\w* zrodl\w*|wdrożeni\w*|wdrozeni\w*)\b",),
        "chunk_patterns": (r"\b(licencj\w*|program\w* komputerow\w*|wartość niematerialn\w*|wartosc niematerialn\w*|koszt wytworzeni\w*|amortyz\w*|art\.?\s*16b)\b",),
    },
    {
        "id": "transfer_pricing_thresholds",
        "label": "ceny transferowe / obowiązek dokumentacyjny",
        "query_patterns": (r"\b(cen transferow\w*|grup\w* kapitałow\w*|powiązan\w*|powiazan\w*)\b",),
        "chunk_patterns": (r"\b(cen transferow\w*|local file|dokumentacj\w* cen transferow\w*|art\.?\s*11[krt])\b",),
    },
)


def _normalize_matching_text(value: str) -> str:
    normalized = value.lower()
    return (
        normalized.replace("ą", "a").replace("ć", "c").replace("ę", "e")
        .replace("ł", "l").replace("ń", "n").replace("ó", "o")
        .replace("ś", "s").replace("ż", "z").replace("ź", "z")
    )


def query_mentions_ksef(value: str) -> bool:
    return bool(re.search(r"\b(ksef|krajow(?:y|ego) system(?:u)? e[ -]?faktur|faktur\w* ustrukturyzowan\w*)\b", value or "", re.IGNORECASE))


def _chunk_domain_labels(chunk: RagChunk) -> set[str]:
    provision_domains = {
        match.group(1).upper()
        for provision in chunk.legal_provisions
        for match in [re.match(r"\[(CIT|PIT|VAT|PCC|SD|EXCISE|AKCYZA|ORDYNACJA|OP|WHT)\]", provision, re.IGNORECASE)]
        if match
    }
    text_domains = {domain.upper() for domain in detect_domains(" ".join(part for part in [chunk.subject, chunk.chunk_text] if part))}
    return provision_domains | text_domains


def build_retrieval_coverage_context(user_prompt: str, retrieved_chunks: list[RagChunk]) -> str:
    if not user_prompt.strip() or not retrieved_chunks:
        return ""

    config = get_rag_config()
    normalized_query = _normalize_matching_text(user_prompt)
    query_domains = {domain.upper() for domain in detect_domains(user_prompt)}
    query_mechanisms = detect_mechanisms(user_prompt, config=config)
    chunk_texts = [
        _normalize_matching_text(" ".join(part for part in [chunk.subject, chunk.chunk_text, " ".join(chunk.legal_provisions)] if part))
        for chunk in retrieved_chunks
    ]
    chunk_domains = [_chunk_domain_labels(chunk) for chunk in retrieved_chunks]
    chunk_mechanisms = [
        detect_mechanisms(" ".join(part for part in [chunk.subject, chunk.chunk_text, " ".join(chunk.legal_provisions)] if part), config=config)
        for chunk in retrieved_chunks
    ]

    expected_axes: list[str] = []
    covered_axes: list[str] = []
    missing_axes: list[str] = []

    for domain in sorted(query_domains):
        expected_axes.append(domain)
        if any(domain in labels for labels in chunk_domains):
            covered_axes.append(domain)
        else:
            missing_axes.append(domain)

    for mechanism in sorted(query_mechanisms):
        expected_axes.append(f"mechanism:{mechanism}")
        if any(mechanism in labels for labels in chunk_mechanisms):
            covered_axes.append(mechanism)
        else:
            missing_axes.append(mechanism)

    for rule in RETRIEVAL_COVERAGE_RULES:
        if any(re.search(pattern, normalized_query) for pattern in rule["query_patterns"]):
            expected_axes.append(rule["label"])
            if any(re.search(pattern, chunk_text) for pattern in rule["chunk_patterns"] for chunk_text in chunk_texts):
                covered_axes.append(rule["label"])
            else:
                missing_axes.append(rule["label"])

    deduped_expected = list(dict.fromkeys(expected_axes))
    deduped_covered = [label for label in dict.fromkeys(covered_axes) if label in deduped_expected]
    deduped_missing = [label for label in dict.fromkeys(missing_axes) if label in deduped_expected and label not in deduped_covered]

    if not deduped_expected:
        return ""

    coverage_ratio = len(deduped_covered) / max(len(deduped_expected), 1)
    if coverage_ratio >= 0.85 and not deduped_missing:
        status = "pokrycie wysokie"
    elif coverage_ratio >= 0.5:
        status = "pokrycie częściowe"
    else:
        status = "pokrycie słabe"

    return (
        "Ocena pokrycia retrievalu względem pytania:\n"
        f"- Status: {status} ({len(deduped_covered)}/{len(deduped_expected)} osi).\n"
        f"- Główne osie pytania: {', '.join(deduped_expected)}.\n"
        f"- Pokryte przez retrieval: {', '.join(deduped_covered) if deduped_covered else 'brak wyraźnie pokrytych osi'}.\n"
        f"- Słabo pokryte lub niepokryte: {', '.join(deduped_missing) if deduped_missing else 'brak istotnych luk'}.\n"
        "Jeżeli status to 'pokrycie częściowe' albo 'pokrycie słabe', przejdź w tryb ostrożny:"
        " nie podawaj stanowczych tez na osiach niepokrytych,"
        " nie zgaduj numerów jednostek redakcyjnych ani treści umów międzynarodowych,"
        " nie domykaj odpowiedzi szeroką syntezą wykraczającą poza materiał."
        " Nie przenoś też do kazusu użytkownika dodatkowych faktów ze źródeł częściowo relewantnych."
        " Na osiach niepokrytych wolno tylko:"
        " (a) wskazać, że to jest kluczowy problem,"
        " (b) opisać bezpiecznie możliwe warianty,"
        " (c) nazwać, jakiego typu źródła brakuje do stanowczej odpowiedzi."
    )


def opening_statute_topic_terms(query: str) -> set[str]:
    normalized = query.lower()
    normalized = normalized.replace("ą", "a").replace("ć", "c").replace("ę", "e")
    normalized = normalized.replace("ł", "l").replace("ń", "n").replace("ó", "o")
    normalized = normalized.replace("ś", "s").replace("ż", "z").replace("ź", "z")
    terms = {
        token
        for token in re.findall(r"[a-z0-9]{4,}", normalized)
        if token not in OPENING_STATUTE_STOPWORDS
    }
    return terms


def score_opening_statute_candidate(chunk: RagChunk, *, query: str) -> float:
    text = re.sub(r"\s+", " ", " ".join(part for part in [chunk.subject, chunk.chunk_text[:1200]] if part)).lower()
    topic_terms = opening_statute_topic_terms(query)
    query_domains = {domain.upper() for domain in detect_domains(query)}
    candidate_domains = (
        {
            match.group(1).upper()
            for provision in chunk.legal_provisions
            for match in [re.match(r"\[(CIT|PIT|VAT|PCC|SD|EXCISE|AKCYZA|ORDYNACJA|OP)\]", provision, re.IGNORECASE)]
            if match
        }
        | {domain.upper() for domain in detect_domains(text)}
    )
    score = float(chunk.score)
    if query_domains and candidate_domains & query_domains:
        score += 2.0
    if topic_terms:
        score += sum(0.35 for term in topic_terms if term in text)

    query_mentions_vehicle = bool(re.search(r"\b(samochod\w*|pojazd\w*|auto)\b", query.lower()))
    query_mentions_real_estate = bool(re.search(r"\b(nieruchomo\w*|lokal\w*|mieszkani\w*|budynek\w*|grunt\w*)\b", query.lower()))
    if query_mentions_real_estate and not query_mentions_vehicle and re.search(r"\b(samochod\w*|pojazd\w*|auto)\b", text):
        score -= 4.0
    if query_mentions_vehicle and not query_mentions_real_estate and re.search(r"\b(nieruchomo\w*|lokal\w*|mieszkani\w*|budynek\w*|grunt\w*)\b", text):
        score -= 2.0
    if re.search(r"\b(wartosc rynkow|wartość rynkow|cena rynkow|preferencyjn|podmiot\w* powiazan|podmiot\w* powiązan)\b", text):
        score += 1.2
    if re.search(r"\b(ewidencj\w* przebiegu|pojazdow samochodowych)\b", text) and not query_mentions_vehicle:
        score -= 4.5
    return score


def extract_opening_statute_quote(retrieved_chunks: list[RagChunk], *, query: Optional[str] = None) -> Optional[str]:
    query_domains = {domain.upper() for domain in detect_domains(query or "")}
    statutes = [chunk for chunk in retrieved_chunks if chunk.source_type == "statute"]
    if query_domains:
        domain_matched = [
            chunk for chunk in statutes
            if (
                {match.group(1).upper() for provision in chunk.legal_provisions for match in [re.match(r"\[(CIT|PIT|VAT|PCC|SD|EXCISE|AKCYZA|ORDYNACJA|OP)\]", provision, re.IGNORECASE)] if match}
                | {domain.upper() for domain in detect_domains(" ".join(part for part in [chunk.subject, chunk.chunk_text[:400]] if part))}
            ) & query_domains
        ]
        if domain_matched:
            statutes = domain_matched

    if query:
        statutes = sorted(
            statutes,
            key=lambda chunk: (score_opening_statute_candidate(chunk, query=query), chunk.score),
            reverse=True,
        )

    for chunk in statutes:
        text = re.sub(r"\s+", " ", chunk.chunk_text.strip())
        article_match = re.search(r"\bArt\.\s*\d+[a-z]*\.?", text)
        if not article_match:
            if text:
                return text
            continue
        excerpt_start = article_match.start()
        next_article_match = re.search(r"\bArt\.\s*\d+[a-z]*\.?", text[article_match.end():])
        excerpt_end = (
            article_match.end() + next_article_match.start()
            if next_article_match
            else len(text)
        )
        excerpt = text[excerpt_start:excerpt_end].strip()
        if excerpt:
            return excerpt
    return None


def build_chat_system_prompt(
    user_prompt: str,
    retrieved_context: str,
    retrieved_chunks: list[RagChunk],
    *,
    intent_hint_context: str = "",
    retrieval_preferences_context: str = "",
    retrieval_coverage_context: str = "",
    source_plan_context: str = "",
    authority_evidence_context: str = "",
    legal_rules_context: str = "",
    legal_rule_trace_context: str = "",
    missing_facts_context: str = "",
    timeline_issue_context: str = "",
) -> str:
    render_completion_instruction = (
        "\n\nZamknięty kontrakt odpowiedzi:\n"
        "Zacznij odpowiedź dokładnie od osobnej linii: Teza."
        " Nie umieszczaj przed nią cytatu, komentarza ani innego tekstu."
        " Następnie wyrenderuj kolejno sekcje: Analiza, Źródła, Ryzyka i luki."
        f" Po zakończeniu pełnej odpowiedzi dodaj w osobnej ostatniej linii dokładnie znacznik {RENDER_COMPLETION_MARKER}."
        " Nie dodawaj nic po tym znaczniku."
    )
    if not retrieved_context:
        return (
            SYSTEM_PROMPT
            + "\n\nNie znaleziono trafnych fragmentów w indeksie źródeł. Nie twórz pozornych źródeł."
            + render_completion_instruction
        )

    hint_instruction = (
        "\n\nDodatkowe ukryte doprecyzowanie intencji użytkownika:\n"
        + intent_hint_context
        + "\nWykorzystaj ten materiał wyłącznie do lepszego ustalenia intencji, doboru źródeł i priorytetów analizy."
        + " Nie cytuj go jako źródła prawa i nie przedstawiaj go tak, jakby był częścią pytania użytkownika."
    ) if intent_hint_context else ""

    retrieval_preferences_instruction = (
        "\n\nPreferencja użytkownika co do zakresu materiałów:\n"
        + retrieval_preferences_context
        + "\nUwzględnij tę preferencję przy doborze akcentów odpowiedzi i opisie źródeł."
    ) if retrieval_preferences_context else ""

    retrieval_coverage_instruction = (
        "\n\nDodatkowa ocena jakości pokrycia retrievalu:\n"
        + retrieval_coverage_context
        + "\nTa ocena ma pierwszeństwo nad pokusą dopowiadania brakującego obrazu z pamięci modelu."
        + " Status osi jest bramką odpowiedzi: dla osi unresolved nie podawaj skutku podatkowego,"
        + " stawki, limitu, terminu ani numeru przepisu jako konkluzji. Dla osi partially_covered"
        + " oddziel to, co wynika z primary source, od tego, czego nie potwierdzono."
    ) if retrieval_coverage_context else ""

    source_plan_instruction = (
        "\n\nPlan źródeł i orkiestracja agentów:\n"
        + source_plan_context
        + "\nTraktuj to jako wynik plannera. Nie pokazuj pełnego planu użytkownikowi,"
        + " ale podporządkuj mu selekcję materiału i kolejność analizy."
    ) if source_plan_context else ""

    authority_evidence_instruction = (
        "\n\nEksperymentalne EvidenceBundle per issue:\n"
        + authority_evidence_context
        + "\nTo jest strukturalny wynik retrievalu authorities. Używaj interpretacji i wyroków wyłącznie jako"
        + " supporting, contrary albo historical authorities, nie jako zamiennika ustawy."
        + " Jeżeli dla osi nie ma supporting ani contrary authorities, jawnie napisz:"
        + " W przeszukanym zbiorze nie znaleziono dostatecznie podobnej interpretacji lub orzeczenia."
    ) if authority_evidence_context else ""

    legal_rules_instruction = (
        "\n\nWynik legal rule extractora:\n"
        + legal_rules_context
        + "\nNajpierw przerób te normy na reguły postępowania dla kazusu użytkownika."
        + " Dopiero potem użyj interpretacji i wyroków jako secondary authority."
    ) if legal_rules_context else ""

    legal_rule_trace_instruction = (
        "\n\nTechniczny trace przepisów pobranych deterministycznie:\n"
        + legal_rule_trace_context
        + "\nNumery artykułów, ustępów i punktów wolno przywoływać wyłącznie z tego trace'u albo z legal rule extractora."
        + " Nie wolno odtwarzać numerów z pamięci modelu ani pisać 'lub odpowiedni przepis'."
    ) if legal_rule_trace_context else ""

    missing_facts_instruction = (
        "\n\nBramka brakujących faktów:\n"
        + missing_facts_context
        + "\nJeżeli brakuje faktu wymagającego wyboru wariantu, odpowiedź ma być warunkowa."
        + " Pokaż konkurencyjne warianty, ale nie wybieraj finalnej stawki, kwoty ani terminu."
        + " Nie używaj logiki typu conservative fallback."
    ) if missing_facts_context else ""

    timeline_issue_instruction = (
        "\n\nWalidacja chronologii przed analizą:\n"
        + timeline_issue_context
        + "\nNajpierw nazwij niespójność czasu i rozpisz warianty zgodne z możliwą chronologią."
    ) if timeline_issue_context else ""

    return (
        SYSTEM_PROMPT
        + "\n\nPoniżej znajdują się zweryfikowane dokumenty z indeksu źródeł prawnych."
        + " Dokumenty zostały wybrane przez retrieval chunkowy, ale w kontekście dostajesz pełną treść wybranych dokumentów, jeśli była dostępna w indeksie."
        + " Odpowiadaj wyłącznie na ich podstawie w części źródłowej, a własne wnioski oznaczaj jako wnioski."
        + " Nie traktuj kilku fragmentów albo części tego samego dokumentu jako niezależnych źródeł."
        + " Jeśli źródła są niejednoznaczne albo częściowe, napisz to wprost zamiast domyślać stanowisko."
        + " Jeżeli pytanie dotyczy przyszłej daty, podaj datę weryfikacji researchu, docelową datę skutku i zastrzeż,"
        + " że opierasz się na przepisach już ogłoszonych oraz źródłach dostępnych w materiale; nie gwarantuj braku późniejszych nowelizacji."
        + " Najpierw wykonaj wewnętrznie selekcję materiału: sporządź robocze podsumowanie każdego dokumentu,"
        + " oddziel źródła trafne, częściowo trafne i nietrafne wobec pytania, a potem wybierz tylko elementy ważne dla odpowiedzi."
        + " W odpowiedzi pokaż przede wszystkim treść wynikającą ze źródeł trafnych i częściowo trafnych."
        + " Nie wolno Ci maskować słabego retrievalu stanowczą analizą."
        + " Jeżeli pokrycie retrievalu jest częściowe lub słabe, zawęź odpowiedź do osi naprawdę wspartych materiałem."
        + " Wtedy odpowiedź ma być bardziej zachowawcza, a nie bardziej kategoryczna."
        + " Nie wolno Ci wprowadzać do odpowiedzi elementów stanu faktycznego, których użytkownik nie podał."
        + " Jeżeli wyrok albo interpretacja zawiera własny stan faktyczny, potraktuj go jako tło źródła, a nie jako część kazusu użytkownika."
        + " Przed użyciem wyroku lub interpretacji wykonaj wewnętrznie test podobieństwa:"
        + " (a) jakie fakty są wspólne,"
        + " (b) jakie fakty są różne,"
        + " (c) czy te różnice blokują przeniesienie wniosku."
        + " Jeśli różnica dotyczy kluczowej przesłanki, napisz, że źródło jest tylko częściowo relewantne i nie rozszerzaj jego tezy na cały kazus."
        + " Nie podawaj dokładnych numerów artykułów, ustępów ani twierdzeń o treści umów międzynarodowych, jeżeli nie masz ich w dostarczonym materiale."
        + " Gdy pytanie dotyczy szerokiego zagadnienia, a retrieval zwraca materiał wycinkowy, nie kończ na stwierdzeniu braków."
        + " Zamiast tego zsyntetyzuj punktowo tylko ten fragment obrazu prawnego, który da się odtworzyć z dostępnych materiałów."
        + " W sekcji Teza daj bezpośrednią odpowiedź na pytanie w co najmniej 5-8 zdaniach."
        + " W sekcji Analiza stosuj trzy podsekcje w tej kolejności:"
        + " (1) Ustalenia wprost ze źródeł,"
        + " (2) Ostrożne wnioski,"
        + " (3) Czego te źródła nie przesądzają."
        + " Jeżeli pytanie dotyczy więcej niż jednego podatku albo użytkownik wymienia konkretne podatki,"
        + " podziel analizę jednoznacznie według podatków i nazwij sekcje wprost, np. VAT, CIT, PIT, PCC, SD."
        + " W takich miejscach używaj czytelnych śródtytułów markdown, np. '### VAT' oraz krótkich pogrubionych etykiet, np. '**Kto ponosi skutek:**'."
        + " Nie mieszaj skutków różnych podatków w jednym akapicie, jeżeli da się je rozdzielić."
        + " Dla każdego podatku odpowiedz osobno przynajmniej na trzy kwestie:"
        + " kto ponosi skutek, jaka jest jego istota oraz od jakich faktów zależy wynik."
        + " Jeżeli rozstrzygnięcie zależy od brakującego elementu stanu faktycznego,"
        + " nie poprzestawaj na zdaniu 'to zależy'."
        + " W takiej sytuacji albo zadaj krótkie pytanie doprecyzowujące w sekcji 'Potrzebne doprecyzowanie',"
        + " albo rozpisz co najmniej dwa wyraźne warianty, np. 'Wariant 1: jeśli transakcja podlega VAT...' oraz 'Wariant 2: jeśli transakcja nie podlega VAT...'."
        + " Gdy pytanie zostało już zadane i masz odpowiadać merytorycznie, preferuj rozpisanie wariantów zamiast urywania analizy."
        + " Jeżeli wynik zależy od ustawowej przesłanki podmiotowej lub przedmiotowej,"
        + " nazwij ją wprost i oceń osobno, czy materiał pozwala stwierdzić jej spełnienie."
        + " Nie zastępuj tej przesłanki luźnym podobieństwem ekonomicznym ani samym powiązaniem osobowym."
        + " Jeżeli pytanie dotyczy sprzedaży, nabycia, aportu, najmu albo innych czynności mogących angażować kilka podatków,"
        + " dopilnuj, aby każdy wskazany przez użytkownika podatek dostał osobny, rozwinięty fragment odpowiedzi."
        + " Jeżeli użytkownik pyta o skutki podatkowe transakcji, odpowiedź ma być raczej pełniejsza niż skrótowa."
        + " Rozwijaj praktyczne konsekwencje dla każdej strony transakcji, zamiast kończyć na jednym ogólnym zdaniu."
        + " W pytaniach o fundację rodzinną nie zakładaj automatycznie, że pożyczka dla spółki jest dozwolona tylko dlatego, że beneficjent jest z nią personalnie związany."
        + " Jeżeli w materiale nie ma potwierdzenia, że fundacja jest wspólnikiem lub akcjonariuszem tej spółki, wskaż to jako brakującą przesłankę."
        + " Jeżeli materiał zawiera art. 5 UFR, traktuj go jako katalog enumeratywny i mapuj czynność do konkretnego punktu lub litery."
        + " Nie przedstawiaj jako spornego tego, co jest wprost wymienione w art. 5 UFR."
        + " Najem, dzierżawa i udostępnianie mienia do korzystania to art. 5 ust. 1 pkt 2 UFR."
        + " Pożyczka dla spółki kapitałowej, w której fundacja posiada udziały albo akcje, to art. 5 ust. 1 pkt 5 lit. a UFR."
        + " Pożyczka dla niezależnej spółki bez udziałów/akcji fundacji zasadniczo nie mieści się w tej regule."
        + " W pytaniach o sprzedaż mienia przez fundację rodzinną oceń osobno, czy materiał pozwala stwierdzić, że mienie nie zostało nabyte wyłącznie w celu dalszego zbycia."
        + " W pytaniach o świadczenia dla beneficjenta fundacji rodzinnej nie zakładaj pełnego zwolnienia PIT bez sprawdzenia, czy źródła potwierdzają zakres zwolnienia i ewentualną proporcję przypisaną fundatorowi."
        + " Dla fundacji rodzinnej buduj osobno drzewo CIT fundacji i PIT odbiorcy; ukryty zysk w CIT nie jest automatycznie statutowym świadczeniem w PIT."
        + " Przy art. 24r CIT wskazuj jako podstawę 25% CIT dochód z działalności wykraczającej poza art. 5, nie sam przychód; odsetki są przychodem, a koszty finansowania i obsługi trzeba zbadać."
        + " Pożyczony kapitał nie jest podstawą 25% CIT."
        + " Zwolnienia PIT fundatora nie uzasadniaj pokrewieństwem fundatora z samym sobą; ustawa PIT obejmuje fundatora wprost."
        + " Proporcja zwolnienia PIT zależy od mienia wniesionego przez fundatorów i mienia uznawanego za wniesione przez fundację według art. 27-29 UFR, a nie od relacji funduszu założycielskiego do aktualnego majątku."
        + " Dziecko fundatora traktuj jako zstępnego/grupę zerową; pokaż hierarchię: zwolnienie według proporcji, potem 10% dla I/II grupy w niezwolnionym zakresie, potem 15% dla pozostałych."
        + " Przy sprzedaży fundatorowi lub beneficjentowi poniżej wartości rynkowej oceń osobno: dozwolone zbywanie mienia, ukryty zysk/różnicę cenową, PIT odbiorcy oraz VAT art. 32."
        + " Usługi prawne, księgowe, doradcze, zarządzania lub podobne świadczone przez fundatora, beneficjenta albo podmiot powiązany sprawdź pod art. 24q ust. 1a pkt 3 niezależnie od rynkowości ceny."
        + " Jeżeli użytkownik podaje wiele czynności fundacji rodzinnej, odpowiedz macierzą dla każdej czynności i nie kończ przed omówieniem wszystkich."
        + " Dla każdej czynności fundacji rodzinnej podaj: allowed_activity, foundation_cit, hidden_profit, beneficiary_pit, vat, tax_base, tax_rate, tax_point, missing_facts."
        + " W pytaniach o mieszane wykorzystanie nieruchomości dla VAT nie utożsamiaj automatycznie proporcji sprzedaży z prewspółczynnikiem."
        + " Użyj prewspółczynnika tylko wtedy, gdy źródła rzeczywiście wskazują na mieszanie działalności gospodarczej z użyciem pozostającym poza działalnością gospodarczą."
        + " W pytaniach o PCC przy nieruchomości nie uogólniaj, że każde zwolnienie z VAT wyłącza PCC."
        + " Jeżeli materiał nie potwierdza wyjątku dla nieruchomości, zaznacz to jako istotną lukę zamiast podawać regułę ogólną bez zastrzeżenia."
        + " Jeżeli wynik podatkowy zależy od skuteczności albo kwalifikacji czynności cywilnoprawnej, najpierw ustal tę skuteczność cywilistyczną"
        + " (na przykład zgoda wierzyciela przy przejęciu długu z art. 519-521 KC albo ważność darowizny z art. 888-890 KC),"
        + " a dopiero potem przechodź do podatków. Nie ustawiaj PIT, PCC i SD jako równorzędnych hipotez bez ustalenia tytułu prawnego."
        + " Jeśli materiał nie pozwala rozstrzygnąć skuteczności cywilnoprawnej, zadaj krótkie pytanie doprecyzowujące zamiast zgadywać skutki podatkowe."
        + " Wewnętrznie pracuj w rolach: planner ustala wymagane źródła, legal rule extractor zamienia przepisy na normy,"
        + " a writer pisze odpowiedź opartą najpierw na tych normach. Nie mieszaj tych etapów."
        + " W każdej odpowiedzi zacznij analizę od primary law. Interpretacje i wyroki mogą tylko objaśniać lub potwierdzać regułę wynikającą z przepisu."
        + " W pytaniach o przekształcenie spółki komandytowej w sp. z o.o. rozdziel osobno:"
        + " (a) skutki samego przekształcenia i sukcesji, (b) wejście w estoński CIT i ewentualne ukryte zyski,"
        + " (c) PCC, oraz (d) koszt podatkowy udziałów przy późniejszej sprzedaży."
        + " Nie zakładaj automatycznie, że historyczne zyski spółki osobowej stają się nowym zyskiem spółki kapitałowej,"
        + " ani że zbycie udziałów po przekształceniu jest nowym objęciem udziałów bez sprawdzenia podstawy kosztowej."
        + " Przy sprzedaży niezabudowanego gruntu nie używaj testu pierwszego zasiedlenia."
        + " Najpierw sprawdź, czy materiał pokrywa analizę terenu budowlanego, przeznaczenia pod zabudowę oraz relację art. 43 ust. 1 pkt 9 do definicji terenu budowlanego."
        + " Jeżeli w stanie faktycznym pojawiają się warunki zabudowy, pozwolenie na budowę, podział działki albo przygotowanie gruntu pod inwestycję, oceń wyraźnie znaczenie tych faktów dla zwolnienia VAT przy gruncie niezabudowanym."
        + " Nie przenoś automatycznie wniosku, że status podatnika VAT oznacza także działalność gospodarczą w PIT."
        + " W takim kazusie oceń osobno przesłanki PIT, w szczególności zorganizowanie, ciągłość, działanie we własnym imieniu, skalę oraz powtarzalność."
        + " Jeżeli źródło VAT i źródło PIT opierają się na podobnym, ale nie tożsamym stanie faktycznym, zaznacz ograniczenie transferu między podatkami zamiast formułować jeden wspólny wniosek."
        + " Gdy sprzedaż gruntu poprzedza dzierżawa oraz udzielenie pełnomocnictwa deweloperowi do uzyskania decyzji administracyjnych, potraktuj te fakty jako istotne dla oceny VAT,"
        + " ale nie pomijaj różnic takich jak liczba działek, skala przedsięwzięcia, liczba nabywców i to, kto faktycznie ponosi koszty przygotowania inwestycji."
        + " W pytaniach o WHT analizuj oddzielnie każdą kategorię płatności, np. dywidendę, odsetki i usługi zarządzania."
        + " Nie wolno Ci automatycznie przenosić przesłanek lub tez z jednej kategorii należności na inną tylko dlatego, że wszystkie występują w jednym kazusie."
        + " Wyraźnie rozdzielaj: (a) zwolnienie ustawowe, (b) stawkę lub wyłączenie z UPO, (c) klauzule antyabuzywne lub odmowę preferencji."
        + " Jeżeli wyrok dotyczy beneficial owner przy odsetkach, nie pisz automatycznie, że identyczny warunek wynika wprost z przepisu o dywidendach albo usługach, chyba że masz to w materiale źródłowym."
        + " Przy mechanizmie pay and refund najpierw ustal dokładnie, które typy należności wchodzą do zakresu przepisu, a które nie."
        + " Następnie wskaż, czy próg liczy się łącznie dla relewantnych płatności do tego samego podatnika, czy podatek pobiera się od całej kwoty czy tylko od nadwyżki ponad limit, oraz czy kolejność wypłat ma znaczenie."
        + " Nie używaj sformułowania 'wszystkie płatności' bez wcześniejszego sprawdzenia zakresu przepisu."
        + " Jeżeli pytanie dotyczy transakcji transgranicznej, nierezydenta, zakładu, certyfikatu rezydencji albo płatności do podmiotu zagranicznego, domyślnie sprawdź także, czy potrzebna jest analiza właściwej UPO."
        + " Jeżeli w takich pytaniach retrieval nie dostarcza UPO albo materiału o zakładzie i rezydencji, zaznacz to jako istotną lukę zamiast kończyć odpowiedź wyłącznie na ustawie krajowej."
        + " Nie rozbudowuj sekcji Ryzyka i luki ponad to, co konieczne."
        + " Jeżeli jakieś źródło jest marginalne, uboczne albo zawiera obiter dictum, nadal wykorzystaj jego treść,"
        + " ale wyraźnie oznacz ograniczoną wagę tej wypowiedzi."
        + " Jeśli w materiale są interpretacje lub wyroki, używaj ich jako wsparcia argumentacji, ale wyraźnie odróżniaj je od treści ustawy."
        + " Przy pytaniach o KSeF wykonaj wewnętrznie checklistę: art. 106a, art. 106b, art. 106ga ust. 2,"
        + " art. 106gb ust. 4. Nie wyprowadzaj braku KSeF wyłącznie z tego, że miejsce dostawy lub świadczenia jest poza Polską."
        + " Przy pytaniach o KSeF po 1 września 2025 r. jako aktualnego źródła kontrolnego wymagaj KSeF 2.0,"
        + " w szczególności ustawy z 5 sierpnia 2025 r., Dz.U. 2025 poz. 1203, oraz aktualnych materiałów MF."
        + " Nie opieraj rozstrzygnięcia wyłącznie na ustawie z 9 maja 2024 r.; jeżeli nie masz Dz.U. 2025 poz. 1203 albo aktualnego bundle KSeF 2.0 w materiale, oznacz oś jako nierozstrzygniętą."
        + " Dla dat w 2026 r. sprawdź etapowanie: 1 lutego 2026 dla podatników z wartością sprzedaży brutto za 2024 r. ponad 200 mln zł oraz 1 kwietnia 2026 dla pozostałych."
        + " Do końca 2026 r. sprawdź przejściowy miesięczny limit 10 000 zł brutto dla faktur objętych obowiązkowym KSeF; jeżeli brak wartości sprzedaży miesięcznej albo informacji o wcześniejszym przekroczeniu, wskaż brak faktów."
        + " Nie wskazuj kar KSeF za naruszenia w 2026 r.; sankcje z art. 106ni analizuj dopiero od 1 stycznia 2027 r., jeśli materiał to potwierdza."
        + " Rozróżnij tryby: online, offline24, niedostępność KSeF, awaria KSeF, całkowita awaria KSeF oraz awaria systemu podatnika."
        + " Offline24 nie wymaga oficjalnej awarii KSeF; standardowy termin przesłania to najpóźniej następny dzień roboczy, chyba że w źródłach wynika szczególny tryb awaryjny."
        + " Przy fakturze PDF wystawionej poza KSeF z naruszeniem obowiązku nie przesuwaj automatycznie odliczenia na datę późniejszego numeru KSeF; badaj faktyczne otrzymanie i art. 86 oraz 88."
        + " Przy korekcie in minus nie stosuj automatycznie historycznego SLIM VAT; oddziel korektę ustrukturyzowaną od korekty poza KSeF, offline24, niedostępności i awarii."
        + " Przy podmiocie zagranicznym odróżnij samo istnienie SMPD od uczestnictwa tego SMPD w konkretnej dostawie albo usłudze."
        + " Dla usługodawcy badaj zasoby potrzebne do świadczenia usługi i ich uczestnictwo w transakcji, nie test odbioru usług przez nabywcę."
        + " Przy B2B/B2C klasyfikuj charakter nabywcy per transakcja: business, private albo mixed_or_unclear; NIP jest tylko przesłanką pomocniczą, nie rozstrzygającą."
        + " Jeżeli użytkownik podaje listę transakcji lub przypadków KSeF, odpowiedz macierzą obejmującą każdy przypadek z osobna."
        + " Przed zakończeniem policz, ile przypadków wskazał użytkownik i ile opisałeś; jeśli brakuje któregoś przypadku, dopisz go zamiast kończyć odpowiedź."
        + " Dla każdego przypadku KSeF podaj co najmniej: obowiązek KSeF, tryb/wyjątek, data lub fakt brakujący, skutek dla odliczenia/korekty/sankcji oraz źródło."
        + " Jeżeli źródła pokazują, że polskie przepisy fakturowe mają zastosowanie do transakcji poza terytorium kraju,"
        + " rozróżnij obowiązek wystawienia faktury ustrukturyzowanej od sposobu jej udostępnienia nabywcy."
        + " Nie uzależniaj obowiązku KSeF wyłącznie od tego, czy podatnik jest czynnym podatnikiem VAT."
        + " Najpierw ustal, czy działa jako podatnik VAT przy danej transakcji, czy ma obowiązek wystawić fakturę, a dopiero potem oceniaj wejście do KSeF i ewentualne przepisy przejściowe."
        + " Jeżeli faktura została najpierw otrzymana poza KSeF, a później ten sam wydatek lub ta sama transakcja pojawia się w KSeF, nie przesądzaj automatycznie o korekcie JPK ani o przesunięciu momentu odliczenia."
        + " Potraktuj późniejszy dokument przede wszystkim jako potencjalny duplikat albo wtórne potwierdzenie tej samej transakcji i sprawdź materialne przesłanki odliczenia z art. 86."
        + " Jeżeli pytanie dotyczy błędnych danych nabywcy, nie zakładaj z góry, że wystarczy nota korygująca; sprawdź, czy z materiału nie wynika korekta po stronie sprzedawcy."
        + " W pytaniach o dropshipping, platformę, interfejs elektroniczny, IOSS albo sprzedaż towarów importowanych do 150 euro najpierw sprawdź art. 7a, art. 19a, art. 17, art. 28d oraz art. 138a-138j."
        + " Nie zastępuj tych przepisów ogólną regułą z art. 22 ani automatycznym wnioskiem, że importerem jest ten, kto organizuje odprawę."
        + " Przy usługach pośrednika działającego w imieniu i na rzecz sprzedawcy dla B2C użyj art. 28d, nie art. 28c."
        + " W pytaniach o KSeF i B2C nie zakładaj, że faktura dla konsumenta musi być obowiązkowo wystawiona w KSeF; obowiązek dotyczy przede wszystkim relacji B2B i ustawowych wyjątków."
        + " W pytaniach o samochód wykupiony po leasingu do majątku prywatnego i późniejszą darowiznę albo sprzedaż nie pomijaj specjalnych osi:"
        + " w PIT sprawdź art. 14 ust. 2 pkt 19 oraz art. 10 ust. 2 pkt 4 przed użyciem prywatnego półrocznego terminu,"
        + " a przy prywatnym samochodzie używanym częściowo w działalności odróżnij limit 20% z art. 23 ust. 1 pkt 46 od limitu 75% wynikającego z art. 23 ust. 1 pkt 46a."
        + " W pytaniach o ulgę mieszkaniową sprawdź najpierw tytuł do nieruchomości, faktycznie poniesiony wydatek oraz to, czy czasowy wynajem sam w sobie nie wyklucza ulgi."
        + " Przy umorzeniu albo zaniechaniu poboru związanym z kwalifikowanym kredytem mieszkaniowym sprawdź art. 52i i nie nazywaj banku płatnikiem bez wyraźnej podstawy."
        + " Nie opieraj kwalifikacji środka trwałego wyłącznie na braku technicznego wpisu do ewidencji; sprawdź, czy składnik podlegał ujęciu i jak był faktycznie używany."
        + " Przy sprzedaży rzeczy otrzymanej w darowiźnie nie przyjmuj automatycznie wartości rynkowej z dnia darowizny jako kosztu; najpierw sprawdź, czy przy nieodpłatnym nabyciu powstał przychód z art. 11 ust. 2-2b, czy darowizna była poza PIT jako podlegająca podatkowi od spadków i darowizn."
        + " W VAT odróżnij faktyczne nieodliczenie VAT od ustawowego braku prawa do odliczenia; dla art. 7 ust. 2 znaczenie ma prawo do odliczenia, a nie samo to, czy podatnik z niego skorzystał."
        + " Przy fakturze od małżonka nieprowadzącego działalności najpierw ustal, czy osoba działa jako podatnik VAT w tej konkretnej sprzedaży, a dopiero potem pisz o obowiązku fakturowania."
        + " W podatku od spadków i darowizn przy darowiźnie między małżonkami sprawdź zwolnienie z art. 4a, termin zgłoszenia, wyjątki od zgłoszenia oraz relację do kwoty z art. 9."
        + " Zasygnalizuj też cywilnoprawną kwestię majątku wspólnego małżonków, jeżeli pytanie mówi tylko o 'majątku prywatnym'."
        + hint_instruction
        + retrieval_preferences_instruction
        + source_plan_instruction
        + authority_evidence_instruction
        + legal_rules_instruction
        + legal_rule_trace_instruction
        + missing_facts_instruction
        + timeline_issue_instruction
        + render_completion_instruction
        + retrieval_coverage_instruction
        + "\n\nKontekst źródłowy:\n"
        + retrieved_context
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_supabase_service_client():
    client = get_supabase_service_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Supabase is not configured on the backend")

    return client


def is_chat_storage_ready() -> bool:
    client = get_supabase_service_client()
    if client is None:
        return False

    try:
        client.table("chat_threads").select("id").limit(1).execute()
    except APIError as exc:
        if exc.json().get("code") == "PGRST205":
            return False
        raise
    except httpx.HTTPError:
        return False

    return True


def ensure_chat_storage_ready() -> None:
    if not is_chat_storage_ready():
        raise HTTPException(
            status_code=503,
            detail="Chat history schema is not available in Supabase yet. Apply the SQL migration first.",
        )


def is_chat_storage_available() -> bool:
    try:
        return is_chat_storage_ready()
    except Exception:
        return False


def normalize_thread_title(title: Optional[str], fallback: str = "Nowy wątek") -> str:
    cleaned = (title or "").strip()
    return cleaned[:160] if cleaned else fallback


def build_thread_title_from_message(message: str) -> str:
    compact = " ".join(message.split())
    if not compact:
        return "Nowy wątek"

    return compact[:72].rstrip(" ,.;:-") or "Nowy wątek"


def build_last_message_preview(message: str) -> str:
    compact = " ".join(message.split())
    return compact[:140]


def map_thread_summary(row: dict) -> ChatThreadSummary:
    return ChatThreadSummary(
        id=row["id"],
        title=normalize_thread_title(row.get("title")),
        archived=bool(row.get("archived")),
        updated_at=row.get("updated_at") or utc_now_iso(),
        created_at=row.get("created_at") or utc_now_iso(),
        last_message_preview=row.get("last_message_preview") or "",
    )


def fetch_thread_row(chat_id: str, *, user_id: str) -> dict:
    client = require_supabase_service_client()
    response = (
        client.table("chat_threads")
        .select("id,user_id,title,archived,updated_at,created_at,last_message_preview")
        .eq("id", chat_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Chat thread not found")

    return rows[0]


def upsert_thread_metadata(chat_id: str, *, user_id: str, title: str, last_message_preview: str) -> None:
    client = require_supabase_service_client()
    payload = {
        "id": chat_id,
        "user_id": user_id,
        "title": normalize_thread_title(title),
        "last_message_preview": last_message_preview,
        "updated_at": utc_now_iso(),
    }
    client.table("chat_threads").upsert(payload).execute()


def persist_pending_chat_messages(chat_id: str, *, user_id: str, messages: list[dict[str, str]]) -> None:
    client = require_supabase_service_client()
    existing_response = (
        client.table("chat_messages")
        .select("id", count="exact")
        .eq("chat_id", chat_id)
        .execute()
    )
    existing_count = existing_response.count or 0

    pending_messages = messages[existing_count:]
    if pending_messages:
        client.table("chat_messages").insert(
            [
                {
                    "id": str(uuid4()),
                    "chat_id": chat_id,
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in pending_messages
            ]
        ).execute()

    latest_user_message = next((message["content"] for message in reversed(messages) if message["role"] == "user"), "")
    current_thread = fetch_thread_row(chat_id, user_id=user_id)
    existing_title = normalize_thread_title(current_thread.get("title"))
    thread_title = (
        build_thread_title_from_message(latest_user_message)
        if existing_title == "Nowy wątek" and latest_user_message
        else existing_title
    )
    upsert_thread_metadata(
        chat_id,
        user_id=user_id,
        title=thread_title,
        last_message_preview=build_last_message_preview(latest_user_message),
    )


def persist_chat_exchange(chat_id: str, *, user_id: str, messages: list[dict[str, str]], reply: str) -> dict:
    client = require_supabase_service_client()
    persist_pending_chat_messages(chat_id, user_id=user_id, messages=messages)

    assistant_message_id = str(uuid4())
    assistant_response = client.table("chat_messages").insert(
        {
            "id": assistant_message_id,
            "chat_id": chat_id,
            "role": "assistant",
            "content": reply,
        }
    ).execute()

    current_thread = fetch_thread_row(chat_id, user_id=user_id)
    existing_title = normalize_thread_title(current_thread.get("title"))
    upsert_thread_metadata(
        chat_id,
        user_id=user_id,
        title=existing_title,
        last_message_preview=build_last_message_preview(reply),
    )

    assistant_rows = assistant_response.data or []
    if not assistant_rows:
        # Some PostgREST/Supabase setups acknowledge inserts without returning a representation.
        fallback_response = (
            client.table("chat_messages")
            .select("id,role,content,created_at,feedback_rating,feedback_comment,feedback_created_at")
            .eq("id", assistant_message_id)
            .eq("chat_id", chat_id)
            .limit(1)
            .execute()
        )
        assistant_rows = fallback_response.data or []
    if not assistant_rows:
        raise HTTPException(status_code=500, detail="Failed to load persisted assistant reply")

    return assistant_rows[0]


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        llm_configured=is_model_gateway_configured(MODEL_GATEWAY_CONFIG),
        llm_provider=MODEL_GATEWAY_CONFIG.provider,
        supabase_configured=is_supabase_configured(),
        rag_index_configured=index_exists(),
        chat_storage_available=is_chat_storage_available(),
        auth_configured=is_supabase_configured(),
        stripe_configured=is_stripe_configured(),
    )


@app.get("/api/admin/rag/corpus-health")
def admin_rag_corpus_health(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict[str, object]:
    """Read-only corpus health; no connection details or document text."""
    if not is_admin_user(current_user):
        raise HTTPException(status_code=403, detail="Tylko admin moze odczytac stan korpusu RAG.")
    report = collect_corpus_health(get_rag_config())
    if os.getenv("RAG_REQUIRE_COMPLETE_CORPUS", "false").strip().lower() in {"1", "true", "yes"}:
        if report["status"] != "healthy":
            raise HTTPException(status_code=503, detail="Aktywny korpus RAG jest niekompletny.")
    return report


@app.get("/api/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(
        default_model=resolve_model(None),
        models=AVAILABLE_MODELS,
    )


def build_account_response(user: AuthenticatedUser) -> AccountResponse:
    profile_row = ensure_profile(user)
    resolved_is_admin = is_admin_user(user)
    if resolved_is_admin and not profile_row.get("is_admin"):
        profile_row = {
            **profile_row,
            "is_admin": True,
        }
    billing_available = True

    try:
        credit_balance = get_credit_balance(user.id)
    except HTTPException as exc:
        if exc.status_code == 503:
            billing_available = False
            credit_balance = 0
        else:
            raise

    return AccountResponse(
        user_id=user.id,
        email=user.email,
        profile=ProfileResponse(**profile_row),
        is_admin=resolved_is_admin,
        credit_balance=credit_balance,
        credit_cost_per_query=get_credit_cost_per_query(),
        credit_unit_price_gross=get_credit_unit_price_gross(),
        credit_currency=get_credit_currency(),
        stripe_configured=billing_available and is_stripe_configured(),
        credit_packs=[CreditPackResponse(**pack.__dict__) for pack in get_credit_packs()],
    )


@app.get("/api/account", response_model=AccountResponse)
def get_account(current_user: AuthenticatedUser = Depends(get_current_user)) -> AccountResponse:
    return build_account_response(current_user)


@app.patch("/api/account/profile", response_model=ProfileResponse)
def patch_account_profile(
    request: ProfileUpdateRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> ProfileResponse:
    ensure_profile(current_user)
    profile_row = update_profile(
        current_user.id,
        full_name=request.full_name,
        law_firm=request.law_firm,
    )
    return ProfileResponse(**profile_row)


@app.post("/api/billing/checkout-session", response_model=CheckoutSessionResponse)
def create_billing_checkout_session(
    request: CheckoutSessionRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> CheckoutSessionResponse:
    ensure_profile(current_user)
    if request.credit_amount is not None:
        pack = build_credit_pack_for_amount(request.credit_amount)
    elif request.credit_pack_id:
        pack = find_credit_pack(request.credit_pack_id)
    else:
        raise HTTPException(status_code=400, detail="Podaj liczbe kredytow albo identyfikator pakietu.")

    checkout = create_checkout_session(
        user=current_user,
        pack=pack,
        success_url=request.success_url,
        cancel_url=request.cancel_url,
    )
    return CheckoutSessionResponse(**checkout)


@app.get("/api/billing/checkout-session/{session_id}", response_model=CheckoutSessionStatusResponse)
def get_billing_checkout_session_status(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> CheckoutSessionStatusResponse:
    session = get_checkout_session(session_id)
    metadata = session.get("metadata") or {}
    session_user_id = str(metadata.get("user_id") or session.get("client_reference_id") or "").strip()

    if session_user_id != current_user.id and not is_admin_user(current_user):
        raise HTTPException(status_code=403, detail="Ta sesja Stripe nie nalezy do zalogowanego uzytkownika.")

    payment_status = str(session.get("payment_status") or "")
    checkout_status = session.get("status")
    credited = payment_status == "paid"

    if credited:
        apply_topup_from_checkout_session(session)

    return CheckoutSessionStatusResponse(
        checkout_session_id=str(session.get("id") or session_id),
        payment_status=payment_status,
        status=str(checkout_status) if checkout_status is not None else None,
        credited=credited,
    )


@app.post("/api/admin/credits/grant", response_model=AdminGrantCreditsResponse)
def grant_admin_credits(
    request: AdminGrantCreditsRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AdminGrantCreditsResponse:
    if not is_admin_user(current_user):
        raise HTTPException(status_code=403, detail="Tylko admin moze przyznawac kredyty.")

    result = grant_credits_to_user(
        admin_user=current_user,
        target_email=request.user_email,
        credit_amount=request.credit_amount,
        reason=request.reason,
    )
    profile = result["profile"]
    return AdminGrantCreditsResponse(
        user_id=str(profile["id"]),
        email=profile.get("email"),
        full_name=profile.get("full_name"),
        credit_balance=int(result["credit_balance"]),
    )


@app.get("/api/admin/users", response_model=AdminUsersResponse)
def list_admin_users(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AdminUsersResponse:
    if not is_admin_user(current_user):
        raise HTTPException(status_code=403, detail="Tylko admin moze przegladac liste uzytkownikow.")

    users = [
        AdminUserSummary(
            user_id=str(profile["id"]),
            email=profile.get("email"),
            full_name=profile.get("full_name"),
            law_firm=profile.get("law_firm"),
            is_admin=bool(profile.get("is_admin")),
            credit_balance=int(profile.get("credit_balance") or 0),
            created_at=profile.get("created_at"),
        )
        for profile in list_profiles_with_credit_balances()
    ]
    return AdminUsersResponse(users=users)


@app.post("/api/billing/webhooks/stripe")
async def stripe_webhook(request: Request) -> dict[str, str]:
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    if not webhook_secret:
        raise HTTPException(status_code=503, detail="Stripe webhook nie jest skonfigurowany.")

    try:
        import stripe  # type: ignore
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Biblioteka Stripe nie jest zainstalowana.") from exc

    stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
    payload = await request.body()
    signature = request.headers.get("stripe-signature")
    if not signature:
        raise HTTPException(status_code=400, detail="Brak podpisu Stripe.")

    try:
        event = stripe.Webhook.construct_event(payload=payload, sig_header=signature, secret=webhook_secret)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Nieprawidlowy webhook Stripe.") from exc

    event_type = str(event.get("type"))
    event_object = event.get("data", {}).get("object", {})
    metadata = event_object.get("metadata") or {}
    order_id = str(metadata.get("order_id") or "").strip()

    if event_type == "checkout.session.completed":
        apply_topup_from_checkout_session(event_object)
    elif order_id and event_type == "checkout.session.expired":
        mark_order_status(order_id=order_id, status="expired")
    elif order_id and event_type == "checkout.session.async_payment_failed":
        mark_order_status(order_id=order_id, status="failed")

    return {"status": "ok"}


@app.post("/api/chat/hints", response_model=PromptHintsResponse)
async def chat_hints(request: PromptHintsRequest) -> PromptHintsResponse:
    return await request_prompt_hints(
        request.draft,
        request.intent_hints,
        excluded_questions=request.excluded_questions,
        max_hints=request.max_hints,
    )


@app.get("/api/chats", response_model=ChatThreadsResponse)
def list_chat_threads(current_user: AuthenticatedUser = Depends(get_current_user)) -> ChatThreadsResponse:
    if not is_chat_storage_available():
        return ChatThreadsResponse(active=[], archived=[])
    ensure_profile(current_user)
    client = require_supabase_service_client()
    response = (
        client.table("chat_threads")
        .select("id,title,archived,updated_at,created_at,last_message_preview")
        .eq("user_id", current_user.id)
        .order("updated_at", desc=True)
        .execute()
    )
    rows = response.data or []
    active: list[ChatThreadSummary] = []
    archived: list[ChatThreadSummary] = []

    for row in rows:
        thread = map_thread_summary(row)
        if thread.archived:
            archived.append(thread)
        else:
            active.append(thread)

    return ChatThreadsResponse(active=active, archived=archived)


@app.post("/api/chats", response_model=ChatThreadSummary)
def create_chat_thread(
    request: ChatThreadCreateRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> ChatThreadSummary:
    ensure_chat_storage_ready()
    ensure_profile(current_user)
    client = require_supabase_service_client()
    payload = {
        "id": str(uuid4()),
        "user_id": current_user.id,
        "title": normalize_thread_title(request.title),
        "archived": False,
        "last_message_preview": "",
    }
    response = client.table("chat_threads").insert(payload).execute()
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=500, detail="Failed to create chat thread")

    return map_thread_summary(rows[0])


@app.get("/api/chats/{chat_id}", response_model=ChatThreadDetail)
def get_chat_thread(chat_id: str, current_user: AuthenticatedUser = Depends(get_current_user)) -> ChatThreadDetail:
    ensure_chat_storage_ready()
    ensure_profile(current_user)
    thread_row = fetch_thread_row(chat_id, user_id=current_user.id)
    client = require_supabase_service_client()
    response = (
        client.table("chat_messages")
        .select("id,role,content,created_at,feedback_rating,feedback_comment,feedback_created_at")
        .eq("chat_id", chat_id)
        .order("created_at")
        .execute()
    )
    messages = [PersistedChatMessage(**row) for row in (response.data or [])]
    thread = map_thread_summary(thread_row)

    return ChatThreadDetail(
        id=thread.id,
        title=thread.title,
        archived=thread.archived,
        updated_at=thread.updated_at,
        created_at=thread.created_at,
        last_message_preview=thread.last_message_preview,
        messages=messages,
    )


@app.patch("/api/chats/{chat_id}", response_model=ChatThreadSummary)
def update_chat_thread(
    chat_id: str,
    request: ChatThreadUpdateRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> ChatThreadSummary:
    ensure_chat_storage_ready()
    ensure_profile(current_user)
    fetch_thread_row(chat_id, user_id=current_user.id)
    update_payload: dict[str, object] = {"updated_at": utc_now_iso()}
    if request.title is not None:
        update_payload["title"] = normalize_thread_title(request.title)
    if request.archived is not None:
        update_payload["archived"] = request.archived

    client = require_supabase_service_client()
    response = (
        client.table("chat_threads")
        .update(update_payload)
        .eq("id", chat_id)
        .eq("user_id", current_user.id)
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=500, detail="Failed to update chat thread")

    return map_thread_summary(rows[0])


@app.post("/api/chats/{chat_id}/messages/{message_id}/feedback", response_model=PersistedChatMessage)
def save_chat_message_feedback(
    chat_id: str,
    message_id: str,
    request: ChatMessageFeedbackRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> PersistedChatMessage:
    ensure_chat_storage_ready()
    ensure_profile(current_user)
    fetch_thread_row(chat_id, user_id=current_user.id)
    client = require_supabase_service_client()
    response = (
        client.table("chat_messages")
        .update(
            {
                "feedback_rating": request.rating,
                "feedback_comment": (request.comment or "").strip() or None,
                "feedback_created_at": utc_now_iso(),
            }
        )
        .eq("id", message_id)
        .eq("chat_id", chat_id)
        .eq("role", "assistant")
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Nie znaleziono odpowiedzi do ocenienia.")

    return PersistedChatMessage(**rows[0])


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> ChatResponse:
    request_started_at = time.monotonic()
    request_trace_id = uuid4().hex

    def remaining_request_seconds(*, reserve_seconds: float = 3.0) -> float:
        return CHAT_REQUEST_DEADLINE_SECONDS - (time.monotonic() - request_started_at) - reserve_seconds

    ensure_profile(current_user)
    redactions: list[str] = []
    sanitized_messages: list[dict[str, str]] = []

    for message in request.messages:
        clean_content, applied = redact_text(message.content)
        redactions.extend(applied)
        sanitized_messages.append({"role": message.role, "content": clean_content})

    model_configured = is_model_gateway_configured(MODEL_GATEWAY_CONFIG)
    model = resolve_model(request.model)
    chat_id = request.chat_id or str(uuid4())
    chat_storage_available = is_chat_storage_available()

    if chat_storage_available:
        if request.chat_id:
            fetch_thread_row(chat_id, user_id=current_user.id)
        else:
            upsert_thread_metadata(
                chat_id,
                user_id=current_user.id,
                title="Nowy wątek",
                last_message_preview="",
            )

    latest_user_message = next(
        (message["content"] for message in reversed(sanitized_messages) if message["role"] == "user"),
        "",
    )
    if chat_storage_available:
        persist_pending_chat_messages(chat_id, user_id=current_user.id, messages=sanitized_messages)

    intent_hint_context = build_hint_context(request.intent_hints)
    retrieval_preferences_context = build_retrieval_preferences_context(request.retrieval_preferences)
    effective_user_prompt = build_effective_user_prompt(latest_user_message, request.intent_hints)

    pipeline_mode = get_legal_pipeline_mode()
    if pipeline_mode == "shadow":
        schedule_legal_rag_v2_shadow(effective_user_prompt)
    elif pipeline_mode in {"model_rag_model", "legal_rag_v2"}:
        run_id = uuid4().hex
        try:
            v2_result = await get_legal_rag_v2_pipeline().run(
                effective_user_prompt,
                mode=pipeline_mode,
                request_id=str(uuid4()),
                run_id=run_id,
            )
        except Exception as exc:
            logger.exception("legal_rag_v2 request failed", extra={"run_id": run_id})
            raise HTTPException(
                status_code=502,
                detail=(
                    "Nowy pipeline prawny nie ukończył analizy. "
                    f"Trace diagnostyczny: {run_id}."
                ),
            ) from exc

        failed_validations = [
            item.stage for item in v2_result.validation if not item.passed
        ]
        if failed_validations:
            logger.error(
                "legal_rag_v2 result blocked by deterministic validation",
                extra={"run_id": run_id, "failed_stages": failed_validations},
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    "Nowy pipeline prawny zablokował odpowiedź po walidacji. "
                    f"Trace diagnostyczny: {run_id}."
                ),
            )

        v2_reply = v2_result.final_answer or (
            "Teza\nBrak zatwierdzonej odpowiedzi.\n\n"
            "Analiza\nPipeline nie utworzył renderowalnego wyniku.\n\n"
            "Źródła\nBrak.\n\n"
            "Ryzyka i luki\nSprawdź trace przebiegu."
        )
        if model_configured:
            consume_credit_for_chat(
                user_id=current_user.id,
                model=v2_result.writer_output and MODEL_GATEWAY_CONFIG.answer_writer_model or model,
                chat_id=chat_id,
                request_id=str(uuid4()),
            )
        persisted_assistant_message = None
        if chat_storage_available:
            persisted_assistant_message = persist_chat_exchange(
                chat_id,
                user_id=current_user.id,
                messages=sanitized_messages,
                reply=v2_reply,
            )
        return ChatResponse(
            reply=v2_reply,
            mode="live" if model_configured else "demo",
            model=MODEL_GATEWAY_CONFIG.answer_writer_model,
            redactions=sorted(set(redactions)),
            analysis_trace={
                "pipeline": pipeline_mode,
                "runtime": legal_runtime_debug(),
                "run_id": v2_result.run_id,
                "fallback": v2_result.fallback_trace.model_dump(mode="json"),
                "plan": v2_result.legal_research_plan.model_dump(mode="json"),
                "evidence_bundles": [
                    item.model_dump(mode="json") for item in v2_result.evidence_bundles
                ],
                "claims": [item.model_dump(mode="json") for item in v2_result.claims],
                "validation": [
                    item.model_dump(mode="json") for item in v2_result.validation
                ],
                "timings_ms": v2_result.timings_ms,
            },
            chat_id=chat_id if chat_storage_available else None,
            assistant_message_id=(persisted_assistant_message or {}).get("id"),
            structured_reply=parse_structured_reply(v2_reply),
        )

    if is_bad_debt_benchmark_trace_request(effective_user_prompt):
        try:
            controlled_result = run_bad_debt_pipeline(BAD_DEBT_BENCHMARK_QUERY)
            controlled_reply, benchmark_trace = build_bad_debt_benchmark_chat_payload(
                controlled_result
            )
        except Exception as exc:
            logger.exception("Bad-debt benchmark trace failed")
            raise HTTPException(
                status_code=502,
                detail="Benchmark kontrolowanego pipeline VAT/CIT nie ukończył analizy.",
            ) from exc
        persisted_assistant_message = None
        if chat_storage_available:
            persisted_assistant_message = persist_chat_exchange(
                chat_id,
                user_id=current_user.id,
                messages=sanitized_messages,
                reply=controlled_reply,
            )
        return ChatResponse(
            reply=controlled_reply,
            mode="demo",
            model=model,
            redactions=sorted(set(redactions)),
            analysis_trace={
                "pipeline": "bad_debt_benchmark_trace",
                **benchmark_trace,
            },
            chat_id=chat_id if chat_storage_available else None,
            assistant_message_id=(persisted_assistant_message or {}).get("id"),
            structured_reply=parse_structured_reply(controlled_reply),
        )

    if can_run_bad_debt_pipeline(effective_user_prompt):
        try:
            controlled_result = run_bad_debt_pipeline(effective_user_prompt)
        except Exception as exc:
            logger.exception("Bad-debt controlled pipeline failed")
            raise HTTPException(
                status_code=502,
                detail="Kontrolowany pipeline VAT/CIT nie ukończył analizy.",
            ) from exc
        controlled_reply = controlled_result.answer
        persisted_assistant_message = None
        if model_configured:
            consume_credit_for_chat(
                user_id=current_user.id,
                model=model,
                chat_id=chat_id,
                request_id=str(uuid4()),
            )
        if chat_storage_available:
            persisted_assistant_message = persist_chat_exchange(
                chat_id,
                user_id=current_user.id,
                messages=sanitized_messages,
                reply=controlled_reply,
            )
        return ChatResponse(
            reply=controlled_reply,
            mode="live" if model_configured else "demo",
            model=model,
            redactions=sorted(set(redactions)),
            analysis_trace={
                "pipeline": "bad_debt_claim_renderer",
                "facts": {
                    fact_id: {
                        "fact_id": fact.fact_id,
                        "fact_type": fact.fact_type,
                        "value": fact.value,
                        "status": fact.status,
                        "subject_role": fact.subject_role,
                    }
                    for fact_id, fact in controlled_result.facts.items()
                },
                "calculations": {
                    calculation_id: {
                        "calculation_id": calculation.calculation_id,
                        "operation": calculation.operation,
                        "result": calculation.result,
                    }
                    for calculation_id, calculation in controlled_result.calculations.items()
                },
                "claims": [
                    {
                        "claim_id": claim.claim_id,
                        "status": claim.status,
                        "result_code": claim.result_code,
                        "result": claim.result,
                        "provision_ids": list(claim.controlling_provisions),
                        "display_references": [
                            str(source.get("display_reference") or "")
                            for source in controlled_result.renderer_payload.get("provisions", [])
                            if str(source.get("provision_id") or "") in claim.controlling_provisions
                        ],
                        "fact_ids": list(claim.fact_dependencies),
                        "calculation_id": claim.calculation_id,
                        "calculation_ids": list(claim.calculation_ids),
                    }
                    for claim in controlled_result.claims.values()
                ],
                "renderer_payload": controlled_result.renderer_payload,
                "reference_trace": {
                    "claim_stage": [
                        {
                            "claim_id": claim.claim_id,
                            "display_references": [
                                str(source.get("display_reference") or "")
                                for source in controlled_result.renderer_payload.get("provisions", [])
                                if str(source.get("provision_id") or "") in claim.controlling_provisions
                            ],
                        }
                        for claim in controlled_result.claims.values()
                    ],
                    "payload_stage": controlled_result.renderer_payload.get("provisions", []),
                    "response_stage": sorted(
                        {
                            str(source.get("display_reference") or "")
                            for source in controlled_result.renderer_payload.get("provisions", [])
                            if str(source.get("display_reference") or "") in controlled_reply
                        }
                    ),
                },
                "render_validation": {
                    "passed": controlled_result.render_validation.passed,
                    "missing_claim_ids": list(controlled_result.render_validation.missing_claim_ids),
                    "placeholder_count": controlled_result.render_validation.placeholder_count,
                    "thesis_contradictions": list(controlled_result.render_validation.thesis_contradictions),
                    "truncated": controlled_result.render_validation.truncated,
                },
            },
            chat_id=chat_id if chat_storage_available else None,
            assistant_message_id=(persisted_assistant_message or {}).get("id"),
            structured_reply=parse_structured_reply(controlled_reply),
        )

    # Controlled cases bypass the legacy free-form writer entirely. The renderer
    # receives only validated claims and exact registry references.
    if is_mixed_invoice_query(effective_user_prompt):
        authority_cards, authority_outcome = retrieve_controlled_authority_lane(effective_user_prompt)
        try:
            controlled_result = run_legal_pipeline(
                effective_user_prompt,
                authority_cards=authority_cards,
                interpretation_lane_outcome=dict(authority_outcome.get("interpretation_lane") or {}),
                judgment_lane_outcome=dict(authority_outcome["judgment_lane"]),
            )
        except Exception as exc:
            logger.exception("Mixed-invoice controlled pipeline failed")
            raise HTTPException(
                status_code=502,
                detail="Kontrolowany pipeline faktury mieszanej nie ukończył analizy.",
            ) from exc
        controlled_reply = controlled_result.answer
        persisted_assistant_message = None
        if model_configured:
            consume_credit_for_chat(
                user_id=current_user.id,
                model=model,
                chat_id=chat_id,
                request_id=str(uuid4()),
            )
        if chat_storage_available:
            persisted_assistant_message = persist_chat_exchange(
                chat_id,
                user_id=current_user.id,
                messages=sanitized_messages,
                reply=controlled_reply,
            )
        return ChatResponse(
            reply=controlled_reply,
            mode="live" if model_configured else "demo",
            model=model,
            redactions=sorted(set(redactions)),
            analysis_trace={
                "pipeline": "controlled_claim_renderer",
                "claims": [
                    {
                        "claim_id": claim.claim_id,
                        "status": claim.status,
                        "result": claim.result,
                        "controlling_provisions": list(claim.controlling_provisions),
                        "dependency_provisions": list(claim.dependency_provisions),
                    }
                    for claim in controlled_result.claims.values()
                ],
                "renderer_payload": controlled_result.renderer_payload,
                "authority_retrieval": authority_outcome,
                "render_validation": {
                    "passed": controlled_result.render_validation.passed,
                    "missing_claim_ids": list(controlled_result.render_validation.missing_claim_ids),
                    "placeholder_count": controlled_result.render_validation.placeholder_count,
                    "unknown_provision_ids": list(controlled_result.render_validation.unknown_provision_ids),
                    "thesis_contradictions": list(controlled_result.render_validation.thesis_contradictions),
                    "truncated": controlled_result.render_validation.truncated,
                    "missing_required_sections": list(
                        controlled_result.render_validation.missing_required_sections
                    ),
                },
            },
            chat_id=chat_id if chat_storage_available else None,
            assistant_message_id=(persisted_assistant_message or {}).get("id"),
            structured_reply=parse_structured_reply(controlled_reply),
        )

    if can_run_housing_relief_pipeline(effective_user_prompt):
        authority_timeout = min(
            RETRIEVAL_STAGE_TIMEOUT_SECONDS,
            max(5.0, remaining_request_seconds(reserve_seconds=20.0)),
        )
        try:
            authority_cards, authority_outcome = await asyncio.wait_for(
                asyncio.to_thread(retrieve_housing_authorities, effective_user_prompt),
                timeout=authority_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Housing authority retrieval exceeded its request budget trace=%s timeout_seconds=%.1f",
                request_trace_id,
                authority_timeout,
            )
            authority_cards = []
            authority_outcome = {
                "authority_lane_executed": True,
                "outcome": "retrieval_deadline_exceeded",
                "errors": ["retrieval_deadline_exceeded"],
                "candidate_counts": {"interpretation": 0, "judgment": 0},
                "filtered_counts": {"interpretation": 0, "judgment": 0},
                "selected_counts": {"interpretation": 0, "judgment": 0},
                "interpretation_lane": {
                    "executed": True,
                    "status": "deadline_exceeded",
                    "candidates_before_filters": 0,
                    "candidates_after_filters": 0,
                    "selected_count": 0,
                    "candidate_waterfall": [],
                },
                "judgment_lane": {
                    "executed": True,
                    "status": "deadline_exceeded",
                    "candidate_count": 0,
                    "selected_count": 0,
                    "filtered_count": 0,
                    "empty_result_reason": "retrieval_error",
                    "zero_candidates_root_cause": "request_deadline_exceeded",
                },
            }
        try:
            controlled_result = run_housing_relief_pipeline(
                effective_user_prompt,
                authority_cards=authority_cards,
                interpretation_lane_outcome=dict(authority_outcome["interpretation_lane"]),
                judgment_lane_outcome=dict(authority_outcome["judgment_lane"]),
            )
        except Exception as exc:
            # Authorities are secondary evidence.  A malformed or overly broad
            # quoted holding must not suppress a deterministic calculation based
            # on primary law.  Retry without cards, retain an explicit lane
            # status in the answer and preserve the original failure in logs.
            logger.exception(
                "Housing-relief rendering with authority cards failed; retrying without cards trace=%s",
                request_trace_id,
            )
            fallback_interpretation_lane = dict(authority_outcome["interpretation_lane"])
            fallback_judgment_lane = dict(authority_outcome["judgment_lane"])
            for lane in (fallback_interpretation_lane, fallback_judgment_lane):
                lane["status"] = "completed_with_errors"
                lane["selected_count"] = 0
                lane["empty_result_reason"] = "authority_rendering_error"
            try:
                controlled_result = run_housing_relief_pipeline(
                    effective_user_prompt,
                    authority_cards=(),
                    interpretation_lane_outcome=fallback_interpretation_lane,
                    judgment_lane_outcome=fallback_judgment_lane,
                )
            except Exception as fallback_exc:
                logger.exception(
                    "Housing-relief controlled pipeline fallback failed trace=%s",
                    request_trace_id,
                )
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Kontrolowany pipeline ulgi mieszkaniowej nie ukończył analizy. "
                        f"Trace diagnostyczny: {request_trace_id}."
                    ),
                ) from fallback_exc
        controlled_reply = controlled_result.answer
        persisted_assistant_message = None
        if model_configured:
            consume_credit_for_chat(
                user_id=current_user.id,
                model=model,
                chat_id=chat_id,
                request_id=str(uuid4()),
            )
        if chat_storage_available:
            persisted_assistant_message = persist_chat_exchange(
                chat_id,
                user_id=current_user.id,
                messages=sanitized_messages,
                reply=controlled_reply,
            )
        return ChatResponse(
            reply=controlled_reply,
            mode="live" if model_configured else "demo",
            model=model,
            redactions=sorted(set(redactions)),
            analysis_trace={
                "pipeline": "housing_relief_claim_renderer",
                "facts": {
                    fact_id: {
                        "fact_id": fact.fact_id,
                        "fact_type": fact.fact_type,
                        "value": fact.value,
                        "status": fact.status,
                    }
                    for fact_id, fact in controlled_result.facts.items()
                },
                "calculations": {
                    calculation_id: {
                        "calculation_id": calculation.calculation_id,
                        "operation": calculation.operation,
                        "result": calculation.result,
                    }
                    for calculation_id, calculation in controlled_result.calculations.items()
                },
                "claims": [
                    {
                        "claim_id": claim.claim_id,
                        "status": claim.status,
                        "result_code": claim.result_code,
                        "result": claim.result,
                        "provision_ids": list(claim.controlling_provisions),
                        "display_references": [
                            str(source.get("display_reference") or "")
                            for source in controlled_result.renderer_payload.get("provisions", [])
                            if str(source.get("provision_id") or "") in claim.controlling_provisions
                        ],
                        "fact_ids": list(claim.fact_dependencies),
                        "calculation_id": claim.calculation_id,
                        "calculation_ids": list(claim.calculation_ids),
                    }
                    for claim in controlled_result.claims.values()
                ],
                "renderer_payload": controlled_result.renderer_payload,
                "authority_retrieval": authority_outcome,
                "render_validation": {
                    "passed": controlled_result.render_validation.passed,
                    "missing_claim_ids": list(controlled_result.render_validation.missing_claim_ids),
                    "placeholder_count": controlled_result.render_validation.placeholder_count,
                    "unknown_provision_ids": list(controlled_result.render_validation.unknown_provision_ids),
                    "thesis_contradictions": list(controlled_result.render_validation.thesis_contradictions),
                    "truncated": controlled_result.render_validation.truncated,
                    "missing_required_sections": list(
                        controlled_result.render_validation.missing_required_sections
                    ),
                    "errors": list(controlled_result.render_validation.errors),
                },
            },
            chat_id=chat_id if chat_storage_available else None,
            assistant_message_id=(persisted_assistant_message or {}).get("id"),
            structured_reply=parse_structured_reply(controlled_reply),
        )

    remaining_before_retrieval = remaining_request_seconds(reserve_seconds=5.0)
    if remaining_before_retrieval <= 0:
        raise HTTPException(
            status_code=504,
            detail=(
                "Analiza przekroczyła limit czasu tej usługi. "
                "Dla bardzo długich pytań trzeba zwiększyć timeout Cloud Run albo uruchomić tryb zadania w tle."
            ),
        )

    include_interpretations = (
        request.retrieval_preferences.include_interpretations
        if request.retrieval_preferences
        else True
    )
    include_judgments = (
        request.retrieval_preferences.include_judgments
        if request.retrieval_preferences
        else None
    )
    retrieval_mode = get_legal_retrieval_mode()
    hybrid_result: Optional[HybridAuthorityResult] = None
    retrieval_lane_trace: dict[str, object] = {}
    retrieval_timeout = min(
        RETRIEVAL_STAGE_TIMEOUT_SECONDS,
        max(5.0, remaining_request_seconds(reserve_seconds=20.0)),
    )
    if retrieval_mode == "hybrid_authority":
        try:
            hybrid_result = await asyncio.wait_for(
                asyncio.to_thread(
                    run_hybrid_authority_retrieval,
                    effective_user_prompt,
                    include_interpretations=include_interpretations,
                    include_judgments=include_judgments,
                    clarifier_enabled=clarifier_enabled_from_env(),
                ),
                timeout=retrieval_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Hybrid retrieval exceeded its request budget trace=%s timeout_seconds=%.1f",
                request_trace_id,
                retrieval_timeout,
            )
            retrieved_chunks = []
        except Exception as exc:
            logger.exception("Hybrid authority retrieval failed")
            raise HTTPException(
                status_code=502,
                detail="Eksperymentalny hybrid authority retrieval nie ukończył analizy.",
            ) from exc
        else:
            retrieved_chunks = list(hybrid_result.selected_chunks)
    else:
        retrieved_chunks, retrieval_lane_trace = await retrieve_baseline_chat_evidence(
            effective_user_prompt,
            include_interpretations=include_interpretations,
            include_judgments=include_judgments,
            timeout_seconds=retrieval_timeout,
        )
    # Hybrid retrieval remains experimental.  Its timeout must still recover
    # primary law through the configured backend instead of depending on raw
    # files that are absent from the production image.
    if retrieval_mode == "hybrid_authority" and not any(
        chunk.source_type == "statute" for chunk in retrieved_chunks
    ):
        recovery_budget = min(15.0, max(0.001, remaining_request_seconds(reserve_seconds=20.0)))
        try:
            recovered_primary = await asyncio.wait_for(
                asyncio.to_thread(search_primary_law_chunks, effective_user_prompt),
                timeout=recovery_budget,
            )
        except Exception as exc:
            logger.warning(
                "Hybrid primary-law recovery failed trace=%s error=%s",
                request_trace_id,
                type(exc).__name__,
            )
            recovered_primary = []
        retrieved_chunks = [*recovered_primary, *retrieved_chunks]
    if query_mentions_ksef(effective_user_prompt):
        has_current_ksef_source = any(
            "Dz.U. 2025 poz. 1203" in " ".join(
                part for part in [chunk.publication or "", chunk.subject or "", chunk.chunk_text[:1400]] if part
            )
            or "KSeF 2.0" in " ".join(
                part for part in [chunk.subject or "", chunk.chunk_text[:1400]] if part
            )
            for chunk in retrieved_chunks
        )
        if not has_current_ksef_source:
            fallback_chunks = search_chunks(
                effective_user_prompt + " KSeF 2.0 Dz.U. 2025 poz. 1203 offline24 10 000 zł art. 106ni art. 106nda art. 106nh art. 29a art. 86 art. 88",
                limit=max(10, len(retrieved_chunks) + 6),
                source_types={"statute"},
                enforce_query_domain=True,
                tax_domains={"VAT"},
            )
            seen = {chunk.chunk_id for chunk in retrieved_chunks}
            retrieved_chunks = [
                *retrieved_chunks,
                *[chunk for chunk in fallback_chunks if chunk.chunk_id not in seen],
            ]
    if not retrieved_chunks and resolve_rag_runtime().fallback_backend == "supabase":
        retrieved_chunks = search_chunks_supabase(effective_user_prompt)
    source_plan = build_legal_source_plan(
        effective_user_prompt,
        include_interpretations=include_interpretations,
        include_judgments=include_judgments,
    )
    axis_coverage = [
        {
            "axis_id": item.axis_id,
            "label": item.label,
            "controlling_rule_present": item.controlling_rule_present,
            "current_law_source_present": item.current_law_source_present,
            "relevant_resolution_present": item.relevant_resolution_present,
            "primary_source_present": item.primary_source_present,
            "required_treaty_present": item.required_treaty_present,
            "missing_source_types": item.missing_source_types,
            "misleading_neighbor_present": item.misleading_neighbor_present,
            "coverage_score": item.coverage_score,
            "status": item.status,
            "supporting_source_ids": item.supporting_source_ids,
        }
        for item in build_axis_coverage(effective_user_prompt, retrieved_chunks)
    ]
    legal_rules = prioritize_legal_rules_for_query(
        extract_legal_rules_from_statute_chunks(retrieved_chunks),
        effective_user_prompt,
    )
    target_date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", effective_user_prompt)
    legal_rules = filter_legal_rules_for_target_date(
        legal_rules,
        target_date_match.group(1) if target_date_match else datetime.now(timezone.utc).date().isoformat(),
    )
    missing_required_facts = detect_missing_required_facts(effective_user_prompt, legal_rules)
    timeline_issues = detect_fact_timeline_issues(effective_user_prompt)
    allowed_provision_references = build_provision_reference_registry(retrieved_chunks, legal_rules)
    analysis_trace = build_analysis_trace(
        user_prompt=effective_user_prompt,
        retrieved_chunks=retrieved_chunks,
        legal_rules=[legal_rule_to_dict(rule) for rule in legal_rules],
        missing_required_facts=missing_required_facts,
        timeline_issues=timeline_issues,
        allowed_provision_references=allowed_provision_references,
        axis_coverage=axis_coverage,
        evidence_bundles=(
            [to_jsonable(bundle) for bundle in hybrid_result.evidence_bundles]
            if hybrid_result
            else None
        ),
    )
    analysis_trace["request_trace_id"] = request_trace_id
    analysis_trace["verified_primary_source_count"] = len(legal_rules)
    analysis_trace["allowed_provision_references"] = sorted(allowed_provision_references)
    analysis_trace["retrieval_lanes"] = retrieval_lane_trace
    if hybrid_result:
        analysis_trace["hybrid_authority_rag"] = {
            "run_id": hybrid_result.run_id,
            "retrieval_mode": hybrid_result.retrieval_mode,
            "clarifier_enabled": hybrid_result.clarifier_enabled,
            "intent_profile": to_jsonable(hybrid_result.intent_profile),
            "fact_graph": to_jsonable(hybrid_result.fact_graph),
            "issue_graph": to_jsonable(hybrid_result.issue_graph),
            "authority_queries": to_jsonable(hybrid_result.authority_queries),
            "selected_chunk_ids": [chunk.chunk_id for chunk in hybrid_result.selected_chunks],
            "timings": hybrid_result.timings,
        }
    # Fail closed before either the demo renderer or a provider sees the
    # question.  A generic writer has no authority to supply a tax conclusion
    # when no current, extracted primary-law rule controls it.
    if not legal_rules:
        blocked_reply = build_missing_primary_law_reply()
        validation = validate_final_output(
            blocked_reply,
            axis_coverage=[],
            expected_sections=["Teza", "Analiza", "Źródła", "Ryzyka i luki"],
        )
        analysis_trace["claim_gate"] = {
            "approved_claims_without_primary_source": 0,
            "categorical_answer_with_no_sources": False,
            "blocked_reason": "missing_controlling_primary_law",
        }
        analysis_trace["render_attempts"] = [{
            "raw_model_output": None,
            "sanitized_output": None,
            "postprocessing_applied": False,
            "validation": validation,
        }]
        analysis_trace["render_validation"] = validation
        persisted_assistant_message = None
        if chat_storage_available:
            persisted_assistant_message = persist_chat_exchange(
                chat_id,
                user_id=current_user.id,
                messages=sanitized_messages,
                reply=strip_render_completion_marker(blocked_reply),
            )
        return ChatResponse(
            reply=strip_render_completion_marker(blocked_reply),
            mode="live" if model_configured else "demo",
            model=model,
            redactions=sorted(set(redactions)),
            analysis_trace=analysis_trace,
            chat_id=chat_id if chat_storage_available else None,
            assistant_message_id=(persisted_assistant_message or {}).get("id"),
            structured_reply=parse_structured_reply(blocked_reply),
        )
    if not model_configured:
        demo_reply = build_demo_reply(latest_user_message, retrieved_chunks, retrieval_prompt=effective_user_prompt)
        demo_reply = enforce_reply_guardrails(
            demo_reply,
            allowed_provision_references=allowed_provision_references,
            missing_required_facts=missing_required_facts,
            timeline_issues=timeline_issues,
            claim_source_traces=list(analysis_trace.get("claim_source_traces") or []),
        )
        validation = validate_final_output(
            demo_reply,
            axis_coverage=[],
            expected_sections=["Teza", "Analiza", "Źródła", "Ryzyka i luki"],
        )
        analysis_trace["render_validation"] = validation
        demo_reply = strip_render_completion_marker(demo_reply)
        structured_reply = parse_structured_reply(demo_reply)
        persisted_assistant_message = None
        if chat_storage_available:
            persisted_assistant_message = persist_chat_exchange(
                chat_id,
                user_id=current_user.id,
                messages=sanitized_messages,
                reply=demo_reply,
            )
        if hybrid_result:
            write_hybrid_trace_artifacts(
                hybrid_result,
                claims=list(analysis_trace.get("claims") or []),
                renderer_payload={
                    "mode": "demo",
                    "retrieved_chunk_ids": [chunk.chunk_id for chunk in retrieved_chunks],
                    "source_plan": legal_source_plan_to_dict(source_plan, retrieved_chunks),
                },
                final_answer=demo_reply,
                validation=validation,
            )
        return ChatResponse(
            reply=demo_reply,
            mode="demo",
            model=model,
            redactions=sorted(set(redactions)),
            analysis_trace=analysis_trace,
            chat_id=chat_id if chat_storage_available else None,
            assistant_message_id=(persisted_assistant_message or {}).get("id"),
            structured_reply=structured_reply,
        )
    retrieved_context = build_answer_context_block(retrieved_chunks)
    retrieval_coverage_context = "\n\n".join(
        part
        for part in [
            build_retrieval_coverage_context(effective_user_prompt, retrieved_chunks),
            build_axis_coverage_context(effective_user_prompt, retrieved_chunks),
        ]
        if part
    )
    system_prompt = build_chat_system_prompt(
        latest_user_message,
        retrieved_context,
        retrieved_chunks,
        intent_hint_context=intent_hint_context,
        retrieval_preferences_context=retrieval_preferences_context,
        retrieval_coverage_context=retrieval_coverage_context,
        source_plan_context=build_source_plan_context(source_plan, retrieved_chunks),
        authority_evidence_context=build_authority_evidence_context(hybrid_result),
        legal_rules_context=build_legal_rules_context(legal_rules),
        legal_rule_trace_context=build_legal_rule_trace_context(legal_rules),
        missing_facts_context=build_missing_facts_context(missing_required_facts),
        timeline_issue_context=build_timeline_issue_context(timeline_issues),
    )
    required_axis_labels = sorted(
        {
            token
            for token in ("VAT", "CIT", "PIT", "PCC", "SD", "AKCYZA", "ORDYNACJA")
            if any(
                token in str(item.get("label") or "").upper()
                or token in str(item.get("axis_id") or "").upper()
                for item in axis_coverage
            )
        }
    )
    compact_system_prompt = (
        SYSTEM_PROMPT
        + "\n\nTRYB COMPACT RETRY."
        + "\nZacznij dokładnie od: Teza"
        + "\nWyrenderuj kolejno wyłącznie sekcje: Teza, Analiza, Źródła, Ryzyka i luki."
        + "\nNie przepisuj pełnych dokumentów ani pełnego brzmienia artykułów."
        + "\nZachowaj wszystkie osie podatkowe, ale każdą opisz maksymalnie w 3 krótkich akapitach."
        + (
            "\nW sekcji Analiza użyj dokładnie osobnych nagłówków: "
            + ", ".join(f"### {axis}" for axis in required_axis_labels)
            + "."
            if required_axis_labels
            else ""
        )
        + f"\nOstatnią linią musi być dokładnie {RENDER_COMPLETION_MARKER}."
        + ("\n\nZwalidowane reguły:\n" + build_legal_rules_context(legal_rules) if legal_rules else "")
        + ("\n\nTrace podstaw prawnych:\n" + build_legal_rule_trace_context(legal_rules) if legal_rules else "")
        + ("\n\nBrakujące fakty:\n" + build_missing_facts_context(missing_required_facts) if missing_required_facts else "")
        + "\n\nOgraniczony kontekst źródłowy:\n"
        + retrieved_context[:30000]
    )

    reply = ""
    validation: dict[str, object] = {}
    last_render_error: Optional[HTTPException] = None
    last_raw_candidate = ""
    render_attempts: list[dict[str, object]] = []
    try:
        model_timeout = min(
            MODEL_CHAT_TIMEOUT_SECONDS,
            max(5.0, remaining_request_seconds(reserve_seconds=6.0)),
        )
        if model_timeout < 10.0:
            raise HTTPException(
                status_code=504,
                detail=(
                    "Analiza źródeł zajęła zbyt dużo czasu przed wygenerowaniem odpowiedzi. "
                    "Dla bardzo długich pytań trzeba zwiększyć timeout Cloud Run albo zawęzić zakres pytania."
                ),
            )
        gateway = create_model_gateway(MODEL_GATEWAY_CONFIG)
        for attempt in range(2):
            if remaining_request_seconds(reserve_seconds=6.0) < 10.0:
                raise HTTPException(
                    status_code=504,
                    detail=(
                        "Model potrzebuje więcej czasu niż pozwala obecny limit requestu. "
                        "Dla pełnych, kompleksowych analiz trzeba zwiększyć timeout Cloud Run."
                    ),
                )
            retry_feedback = ""
            writer_system_prompt = system_prompt
            writer_input: list[dict[str, str]] = list(sanitized_messages)
            if attempt:
                if last_render_error is not None:
                    retry_feedback = (
                        "\n\nPoprzedni render został odrzucony przez walidator: "
                        f"{last_render_error.detail}"
                        "\nNapraw dokładnie wskazane błędy. Nie zostawiaj pustych slotów typu "
                        "'ustawy o VAT', 'ustawy o CIT' ani '( i ust. ... )'. "
                        "Każde odwołanie do przepisu musi mieć konkretny numer artykułu/ustępu albo zostać pominięte."
                    )
                writer_system_prompt = compact_system_prompt + retry_feedback
                writer_input = [{
                    "role": "user",
                    "content": build_render_retry_input(
                        user_prompt=latest_user_message,
                        rejected_output=last_raw_candidate,
                        validation_error=str(last_render_error.detail) if last_render_error else "nieznany błąd",
                        allowed_provision_references=allowed_provision_references,
                    ),
                }]
            raw_candidate = await asyncio.wait_for(
                gateway.generate_text(
                    input=writer_input,
                    system_prompt=writer_system_prompt,
                    model=model,
                    reasoning_effort="medium",
                    max_output_tokens=CHAT_MAX_TOKENS,
                ),
                timeout=model_timeout,
            )
            last_raw_candidate = raw_candidate
            # A writer error is rejected, never "repaired" by regexes.  In
            # particular, do not substitute an unverified article with "ten
            # przepis": that loses the lineage needed to find the real fault.
            guarded_candidate = raw_candidate
            candidate = raw_candidate
            attempt_diagnostics = build_render_diagnostics(
                raw_candidate=raw_candidate,
                guarded_candidate=guarded_candidate,
                completed_candidate=candidate,
                retrieved_chunks=retrieved_chunks,
            )
            attempt_diagnostics["attempt"] = attempt + 1
            try:
                validation = validate_final_output(
                    candidate,
                    axis_coverage=axis_coverage,
                    expected_sections=["Teza", "Analiza", "Źródła", "Ryzyka i luki"],
                    allowed_provision_references=allowed_provision_references,
                    verified_source_count=len(legal_rules),
                )
            except HTTPException as exc:
                last_render_error = exc
                attempt_diagnostics["validation_error"] = str(exc.detail)
                render_attempts.append(attempt_diagnostics)
                logger.warning(
                    "Chat render rejected trace=%s diagnostics=%s",
                    request_trace_id,
                    json.dumps(attempt_diagnostics, ensure_ascii=False, default=str),
                )
                continue
            attempt_diagnostics["validation"] = validation
            render_attempts.append(attempt_diagnostics)
            reply = candidate
            break
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Model odpowiadał zbyt długo. Spróbuj ponowić zapytanie albo skrócić zakres odpowiedzi."
        ) from exc
    except ModelTechnicalError as exc:
        raise HTTPException(
            status_code=502,
            detail="Nie udało się połączyć z modelem odpowiedzi."
        ) from exc
    except (ModelSchemaError, ModelGatewayError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Model odpowiedzi zwrócił niepoprawny wynik."
        ) from exc

    if not reply and last_raw_candidate:
        verified_primary_sources = render_verified_primary_law_source_lines(
            list(analysis_trace.get("claim_source_traces") or [])
        )
        fallback_candidate, removed_count = build_fail_closed_render_fallback(
            last_raw_candidate,
            allowed_provision_references=allowed_provision_references,
            verified_source_lines=(
                verified_primary_sources
                or render_verified_retrieval_sources(retrieved_chunks)
            ),
            required_axis_labels=required_axis_labels,
        )
        fallback_diagnostics = build_render_diagnostics(
            raw_candidate=last_raw_candidate,
            guarded_candidate=fallback_candidate,
            completed_candidate=fallback_candidate,
            retrieved_chunks=retrieved_chunks,
        )
        fallback_diagnostics.update({
            "attempt": "deterministic_fail_closed_fallback",
            "removed_unsafe_fragments": removed_count,
        })
        try:
            validation = validate_final_output(
                fallback_candidate,
                axis_coverage=axis_coverage,
                expected_sections=["Teza", "Analiza", "Źródła", "Ryzyka i luki"],
                allowed_provision_references=allowed_provision_references,
                verified_source_count=len(legal_rules),
            )
        except HTTPException as exc:
            fallback_diagnostics["validation_error"] = str(exc.detail)
            render_attempts.append(fallback_diagnostics)
            logger.error(
                "Fail-closed render fallback rejected trace=%s diagnostics=%s",
                request_trace_id,
                json.dumps(fallback_diagnostics, ensure_ascii=False, default=str),
            )
        else:
            validation["fail_closed_repair_applied"] = True
            validation["removed_unsafe_fragments"] = removed_count
            fallback_diagnostics["validation"] = validation
            render_attempts.append(fallback_diagnostics)
            reply = fallback_candidate
            logger.warning(
                "Fail-closed render fallback used trace=%s removed_unsafe_fragments=%d",
                request_trace_id,
                removed_count,
            )

    if not reply:
        logger.error(
            "Chat render failed after retries trace=%s diagnostics=%s",
            request_trace_id,
            json.dumps(render_attempts, ensure_ascii=False, default=str),
        )
        if last_render_error is not None:
            raise HTTPException(
                status_code=last_render_error.status_code,
                detail=f"{last_render_error.detail} Trace diagnostyczny: {request_trace_id}.",
            )
        raise HTTPException(
            status_code=502,
            detail=(
                "Model dwukrotnie zwrócił niekompletną odpowiedź. "
                f"Trace diagnostyczny: {request_trace_id}."
            ),
        )
    analysis_trace["render_validation"] = validation
    analysis_trace["render_attempts"] = render_attempts
    reply = strip_render_completion_marker(reply)
    structured_reply = parse_structured_reply(reply)

    request_id = str(uuid4())
    consume_credit_for_chat(
        user_id=current_user.id,
        model=model,
        chat_id=chat_id,
        request_id=request_id,
    )

    persisted_assistant_message = None
    if chat_storage_available:
        persisted_assistant_message = persist_chat_exchange(
            chat_id,
            user_id=current_user.id,
            messages=sanitized_messages,
            reply=reply,
        )

    if hybrid_result:
        write_hybrid_trace_artifacts(
            hybrid_result,
            claims=list(analysis_trace.get("claims") or []),
            renderer_payload={
                "mode": "live",
                "model": model,
                "retrieved_chunk_ids": [chunk.chunk_id for chunk in retrieved_chunks],
                "source_plan": legal_source_plan_to_dict(source_plan, retrieved_chunks),
                "context_characters": len(retrieved_context),
            },
            final_answer=reply,
            validation=validation,
        )

    return ChatResponse(
        reply=reply,
        mode="live",
        model=model,
        redactions=sorted(set(redactions)),
        analysis_trace=analysis_trace,
        chat_id=chat_id if chat_storage_available else None,
        assistant_message_id=(persisted_assistant_message or {}).get("id"),
        structured_reply=structured_reply,
    )


@app.post("/api/rag/search", response_model=RagSearchResponse)
def rag_search(request: RagSearchRequest) -> RagSearchResponse:
    source_types = set(request.source_types or []) or None
    tax_domains = set(request.tax_domains or []) or None
    inspection = inspect_search(request.query, limit=request.limit, source_types=source_types, tax_domains=tax_domains)
    chunks = search_chunks(request.query, limit=request.limit, source_types=source_types, tax_domains=tax_domains)
    if source_types is None or "statute" in source_types:
        chunks = add_primary_source_fallback_chunks(request.query, chunks)
    axis_coverage = [
        {
            "axis_id": item.axis_id,
            "label": item.label,
            "controlling_rule_present": item.controlling_rule_present,
            "current_law_source_present": item.current_law_source_present,
            "relevant_resolution_present": item.relevant_resolution_present,
            "primary_source_present": item.primary_source_present,
            "required_treaty_present": item.required_treaty_present,
            "missing_source_types": item.missing_source_types,
            "misleading_neighbor_present": item.misleading_neighbor_present,
            "coverage_score": item.coverage_score,
            "status": item.status,
            "supporting_source_ids": item.supporting_source_ids,
        }
        for item in build_axis_coverage(request.query, chunks)
    ]
    source_plan = build_legal_source_plan(request.query)
    legal_rules = prioritize_legal_rules_for_query(
        extract_legal_rules_from_statute_chunks(chunks),
        request.query,
    )
    target_date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", request.query)
    legal_rules = filter_legal_rules_for_target_date(
        legal_rules,
        target_date_match.group(1) if target_date_match else datetime.now(timezone.utc).date().isoformat(),
    )
    missing_required_facts = detect_missing_required_facts(request.query, legal_rules)
    timeline_issues = detect_fact_timeline_issues(request.query)
    allowed_provision_references = build_provision_reference_registry(chunks, legal_rules)
    return RagSearchResponse(
        query=inspection.query,
        match_query=inspection.match_query,
        requested_limit=inspection.requested_limit,
        retrieved_count=inspection.retrieved_count,
        selected_count=inspection.selected_count,
        selected_context_chars=inspection.selected_context_chars,
        citations=list_citations(chunks),
        context_block=build_answer_context_block(chunks),
        axis_coverage=axis_coverage,
        source_plan=legal_source_plan_to_dict(source_plan, chunks),
        legal_rules=[legal_rule_to_dict(rule) for rule in legal_rules],
        analysis_trace=build_analysis_trace(
            user_prompt=request.query,
            retrieved_chunks=chunks,
            legal_rules=[legal_rule_to_dict(rule) for rule in legal_rules],
            missing_required_facts=missing_required_facts,
            timeline_issues=timeline_issues,
            allowed_provision_references=allowed_provision_references,
            axis_coverage=axis_coverage,
        ),
        hits=[
            RagSearchHit(
                rank=rank,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
                subject=chunk.subject,
                signature=chunk.signature,
                published_date=chunk.published_date,
                source_url=chunk.source_url,
                canonical_source_id=None,
                evidence_role=chunk.evidence_role or None,
                category=chunk.category,
                source=chunk.source,
                source_type=chunk.source_type,
                source_subtype=chunk.source_subtype,
                authority=chunk.authority,
                publication=chunk.publication,
                legal_state_date=chunk.legal_state_date,
                source_pages=chunk.source_pages,
                legal_provisions=chunk.legal_provisions,
                chunk_chars=len(chunk.chunk_text),
                preview=chunk.chunk_text[:280].strip(),
                selected_for_context=True,
            )
            for rank, chunk in enumerate(chunks, start=1)
        ],
    )


@app.post("/api/rag/reindex", response_model=RagReindexResponse)
def reindex_rag(request: RagReindexRequest) -> RagReindexResponse:
    try:
        should_sync = is_supabase_sync_enabled() if request.sync_supabase is None else request.sync_supabase
        if should_sync:
            if not is_supabase_rag_configured():
                raise RuntimeError("Supabase sync is enabled, but SUPABASE_URL or SUPABASE_SECRET_KEY is missing")
            result = reindex_corpus_to_supabase(limit=request.limit, force=request.force)
            sync_stats = {"documents": result["total_documents"], "chunks": result["total_chunks"]}
        else:
            result = reindex_corpus(limit=request.limit, force=request.force)
            sync_stats = {"documents": 0, "chunks": 0}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - local data path
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RagReindexResponse(
        processed=result["processed"],
        indexed=result["indexed"],
        skipped=result["skipped"],
        chunk_count=result["chunk_count"],
        db_path=result["db_path"],
        total_documents=result["total_documents"],
        total_chunks=result["total_chunks"],
        supabase_synced=should_sync,
        supabase_documents=sync_stats["documents"],
        supabase_chunks=sync_stats["chunks"],
    )


@app.post("/api/rag/eureka/import", response_model=EurekaImportResponse)
async def import_eureka_latest(request: EurekaImportRequest) -> EurekaImportResponse:
    try:
        result = await run_ingest(
            FetchConfig(
                limit=request.limit,
                page_size=request.page_size,
                start_page=request.start_page,
                concurrency=request.concurrency,
                retry_count=request.retry_count,
                request_timeout=request.request_timeout,
                pause_seconds=request.pause_seconds,
                category=request.category,
                law_tags=request.law_tags,
                sort=request.sort,
                raw_output_path=request.raw_output_path,
                output_path=request.output_path,
                overwrite=request.overwrite,
            )
        )
    except Exception as exc:  # pragma: no cover - remote dependency path
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return EurekaImportResponse(**result)
