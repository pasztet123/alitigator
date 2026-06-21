from __future__ import annotations

import os
import re
from typing import Literal, Optional, Union

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.eureka_ingest import DEFAULT_CONCURRENCY, DEFAULT_PAGE_SIZE, DEFAULT_SORT, FetchConfig, run_ingest
from app.rag import (
    build_context_block,
    inspect_search,
    index_exists,
    list_citations,
    reindex_corpus,
    search_chunks,
)
from app.supabase_rag import (
    is_supabase_rag_configured,
    is_supabase_sync_enabled,
    reindex_corpus_to_supabase,
    search_chunks_supabase,
)
from app.supabase_client import is_supabase_configured

load_dotenv()

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
AVAILABLE_MODELS = [
    model.strip()
    for model in os.getenv(
        "ANTHROPIC_MODELS",
        "claude-opus-4-8,claude-sonnet-4-6,claude-haiku-4-5-20251001",
    ).split(",")
    if model.strip()
]
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALITIGATOR_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
ALLOWED_ORIGIN_REGEX = os.getenv(
    "ALITIGATOR_ALLOWED_ORIGIN_REGEX",
    r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
)

SYSTEM_PROMPT = """
Jesteś asystentem aLitigator dla polskich prawników podatkowych.

Zasady odpowiedzi:
1. Odpowiadaj po polsku.
2. Oddzielaj ustalenia źródłowe od własnych wniosków.
3. Jeśli nie masz zweryfikowanych źródeł, napisz to wprost.
4. Struktura odpowiedzi ma mieć sekcje: Teza, Analiza, Źródła, Ryzyka i luki.
5. Nie udawaj pewności. Gdy stan prawny lub orzecznictwo wymaga potwierdzenia, zaznacz to jednoznacznie.
6. Jeżeli dostałeś fragmenty źródłowe z indeksu interpretacji, opieraj odpowiedź wyłącznie na nich i cytuj tylko te źródła.
7. Jeśli dostarczone źródła nie wystarczają do stanowczej odpowiedzi, napisz to wprost.
""".strip()

REDACTION_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "pesel": re.compile(r"\b\d{11}\b"),
    "nip": re.compile(r"\b\d{3}[- ]?\d{3}[- ]?\d{2}[- ]?\d{2}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+48[- ]?)?(?:\d[- ]?){9}(?!\d)"),
}


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=24)
    model: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    mode: Literal["demo", "live"]
    model: str
    redactions: list[str]


class ModelsResponse(BaseModel):
    default_model: str
    models: list[str]


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
    category: Optional[str]
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
    hits: list[RagSearchHit]


app = FastAPI(title="aLitigator API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
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


def build_demo_reply(user_prompt: str) -> str:
    return (
        "Teza\n"
        "To jest tryb demonstracyjny MVP. Backend działa poprawnie, ale klucz Anthropic nie został jeszcze podany w środowisku.\n\n"
        "Analiza\n"
        "Odebrałem pytanie: \""
        f"{user_prompt[:900]}"
        "\". Na tym etapie mogę już obsłużyć bezpieczny przepływ przez backend, przygotować historię rozmowy i maskować podstawowe dane wrażliwe.\n\n"
        "Źródła\n"
        "Jeżeli indeks RAG nie został jeszcze zbudowany albo klucz modelu nie jest dostępny, nie podaję pozornych cytowań.\n\n"
        "Ryzyka i luki\n"
        "Brakuje jeszcze uwierzytelniania, kredytów, retencji w Supabase i warstwy źródeł do researchu podatkowego."
    )


def extract_text_from_anthropic(payload: dict) -> str:
    parts: list[str] = []

    for block in payload.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))

    return "\n".join(part for part in parts if part).strip()


def resolve_model(requested_model: Optional[str]) -> str:
    if requested_model and requested_model in AVAILABLE_MODELS:
        return requested_model

    return DEFAULT_MODEL if DEFAULT_MODEL in AVAILABLE_MODELS else AVAILABLE_MODELS[0]


def build_chat_system_prompt(retrieved_context: str) -> str:
    if not retrieved_context:
        return SYSTEM_PROMPT + "\n\nNie znaleziono trafnych fragmentów w indeksie interpretacji. Nie twórz pozornych źródeł."

    return (
        SYSTEM_PROMPT
        + "\n\nPoniżej znajdują się zweryfikowane fragmenty z indeksu interpretacji indywidualnych."
        + " Odpowiadaj wyłącznie na ich podstawie w części źródłowej, a własne wnioski oznaczaj jako wnioski."
        + " Nie traktuj powtarzających się fragmentów z tego samego dokumentu jako niezależnych źródeł."
        + " Jeśli źródła są niejednoznaczne albo częściowe, napisz to wprost zamiast domyślać stanowisko."
        + "\n\nKontekst źródłowy:\n"
        + retrieved_context
    )


@app.get("/api/health")
def health() -> dict[str, Union[str, bool]]:
    return {
        "status": "ok",
        "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "supabase_configured": is_supabase_configured(),
        "rag_index_configured": index_exists(),
    }


@app.get("/api/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(
        default_model=resolve_model(None),
        models=AVAILABLE_MODELS,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    redactions: list[str] = []
    sanitized_messages: list[dict[str, str]] = []

    for message in request.messages:
        clean_content, applied = redact_text(message.content)
        redactions.extend(applied)
        sanitized_messages.append({"role": message.role, "content": clean_content})

    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = resolve_model(request.model)
    latest_user_message = next(
        (message["content"] for message in reversed(sanitized_messages) if message["role"] == "user"),
        "",
    )

    if not api_key:
        return ChatResponse(
            reply=build_demo_reply(latest_user_message),
            mode="demo",
            model=model,
            redactions=sorted(set(redactions)),
        )

    retrieved_chunks = search_chunks_supabase(latest_user_message) or search_chunks(latest_user_message)
    retrieved_context = build_context_block(retrieved_chunks)
    system_prompt = build_chat_system_prompt(retrieved_context)

    payload = {
        "model": model,
        "max_tokens": 1400,
        "temperature": 0.15,
        "system": system_prompt,
        "messages": [
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
            for message in sanitized_messages
        ],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    reply = extract_text_from_anthropic(response.json())
    if not reply:
        raise HTTPException(status_code=502, detail="Anthropic returned an empty response")

    if retrieved_chunks:
        reply = f"{reply}\n\nŹródła użyte przez retrieval\n{list_citations(retrieved_chunks)}"

    return ChatResponse(
        reply=reply,
        mode="live",
        model=model,
        redactions=sorted(set(redactions)),
    )


@app.post("/api/rag/search", response_model=RagSearchResponse)
def rag_search(request: RagSearchRequest) -> RagSearchResponse:
    inspection = inspect_search(request.query, limit=request.limit)
    chunks = search_chunks(request.query, limit=request.limit)
    return RagSearchResponse(
        query=inspection.query,
        match_query=inspection.match_query,
        requested_limit=inspection.requested_limit,
        retrieved_count=inspection.retrieved_count,
        selected_count=inspection.selected_count,
        selected_context_chars=inspection.selected_context_chars,
        citations=list_citations(chunks),
        context_block=build_context_block(chunks),
        hits=[RagSearchHit(**hit) for hit in inspection.hits],
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
