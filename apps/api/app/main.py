from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Literal, Optional, Union
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from postgrest.exceptions import APIError
from pydantic import BaseModel, Field

from app.eureka_ingest import DEFAULT_CONCURRENCY, DEFAULT_PAGE_SIZE, DEFAULT_SORT, FetchConfig, run_ingest
from app.rag import (
    RagChunk,
    build_context_block,
    inspect_search,
    index_exists,
    list_citations,
    reindex_corpus,
    search_chat_chunks,
    search_chunks,
)
from app.supabase_rag import (
    is_supabase_rag_configured,
    is_supabase_sync_enabled,
    reindex_corpus_to_supabase,
    search_chunks_supabase,
)
from app.supabase_client import get_supabase_service_client, is_supabase_configured

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
6. Zawsze rozróżniaj rodzaj źródła: ustawa jest treścią normy, interpretacja indywidualna przedstawia ocenę organu w konkretnej sprawie, interpretacja ogólna ma odrębny charakter, a wyrok jest wykładnią sądu.
7. Nie przedstawiaj interpretacji ani komentarza jako obowiązującego przepisu. Nie przedstawiaj przepisu jako stanowiska organu.
8. Jeśli dostarczone źródła nie wystarczają do stanowczej odpowiedzi, napisz to wprost.
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
    chat_id: Optional[str] = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    reply: str
    mode: Literal["demo", "live"]
    model: str
    redactions: list[str]
    chat_id: Optional[str] = None


class ModelsResponse(BaseModel):
    default_model: str
    models: list[str]


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


class ChatThreadDetail(BaseModel):
    id: str
    title: str
    archived: bool
    updated_at: str
    created_at: str
    last_message_preview: str
    messages: list[PersistedChatMessage]


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


def build_demo_reply(user_prompt: str, retrieved_chunks: list) -> str:
    citations = list_citations(retrieved_chunks)
    opening_quote = extract_opening_statute_quote(retrieved_chunks)
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
        "Do odpowiedzi merytorycznej potrzebny jest skonfigurowany model językowy; powyższe źródła są jednak dostępne w RAG."
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


def extract_opening_statute_quote(retrieved_chunks: list[RagChunk], *, max_chars: int = 420) -> Optional[str]:
    for chunk in retrieved_chunks:
        if chunk.source_type != "statute":
            continue
        text = re.sub(r"\s+", " ", chunk.chunk_text.strip())
        article_match = re.search(r"Art\.\s*\d+[a-z]*\.?", text)
        excerpt_start = article_match.start() if article_match else 0
        excerpt = text[excerpt_start:].strip()
        if len(excerpt) > max_chars:
            excerpt = excerpt[: max_chars - 1].rstrip() + "…"
        if excerpt:
            return excerpt
    return None


def build_chat_system_prompt(retrieved_context: str, retrieved_chunks: list[RagChunk]) -> str:
    if not retrieved_context:
        return SYSTEM_PROMPT + "\n\nNie znaleziono trafnych fragmentów w indeksie źródeł. Nie twórz pozornych źródeł."

    opening_quote = extract_opening_statute_quote(retrieved_chunks)
    opening_instruction = ""
    if opening_quote:
        opening_instruction = (
            " Zacznij odpowiedź od krótkiego cytatu z przepisu ustawy, bez żadnego wstępu,"
            " bez nagłówka i bez parafrazy przed cytatem. Pierwszy akapit ma być cytatem z ustawy,"
            " najlepiej z najbardziej trafnego przepisu. Użyj tego fragmentu jako punktu wyjścia: \""
            + opening_quote
            + "\". Dopiero po cytacie przejdź do tezy i analizy."
        )

    return (
        SYSTEM_PROMPT
        + "\n\nPoniżej znajdują się zweryfikowane fragmenty z indeksu źródeł prawnych."
        + " Odpowiadaj wyłącznie na ich podstawie w części źródłowej, a własne wnioski oznaczaj jako wnioski."
        + " Nie traktuj powtarzających się fragmentów z tego samego dokumentu jako niezależnych źródeł."
        + " Jeśli źródła są niejednoznaczne albo częściowe, napisz to wprost zamiast domyślać stanowisko."
        + opening_instruction
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

    return True


def ensure_chat_storage_ready() -> None:
    if not is_chat_storage_ready():
        raise HTTPException(
            status_code=503,
            detail="Chat history schema is not available in Supabase yet. Apply the SQL migration first.",
        )


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


def fetch_thread_row(chat_id: str) -> dict:
    client = require_supabase_service_client()
    response = (
        client.table("chat_threads")
        .select("id,title,archived,updated_at,created_at,last_message_preview")
        .eq("id", chat_id)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Chat thread not found")

    return rows[0]


def upsert_thread_metadata(chat_id: str, *, title: str, last_message_preview: str) -> None:
    client = require_supabase_service_client()
    payload = {
        "id": chat_id,
        "title": normalize_thread_title(title),
        "last_message_preview": last_message_preview,
        "updated_at": utc_now_iso(),
    }
    client.table("chat_threads").upsert(payload).execute()


def persist_chat_exchange(chat_id: str, messages: list[dict[str, str]], reply: str) -> None:
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

    client.table("chat_messages").insert(
        {
            "id": str(uuid4()),
            "chat_id": chat_id,
            "role": "assistant",
            "content": reply,
        }
    ).execute()

    latest_user_message = next((message["content"] for message in reversed(messages) if message["role"] == "user"), "")
    current_thread = fetch_thread_row(chat_id)
    existing_title = normalize_thread_title(current_thread.get("title"))
    thread_title = (
        build_thread_title_from_message(latest_user_message)
        if existing_title == "Nowy wątek" and latest_user_message
        else existing_title
    )
    upsert_thread_metadata(
        chat_id,
        title=thread_title,
        last_message_preview=build_last_message_preview(reply),
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


@app.get("/api/chats", response_model=ChatThreadsResponse)
def list_chat_threads() -> ChatThreadsResponse:
    ensure_chat_storage_ready()
    client = require_supabase_service_client()
    response = (
        client.table("chat_threads")
        .select("id,title,archived,updated_at,created_at,last_message_preview")
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
def create_chat_thread(request: ChatThreadCreateRequest) -> ChatThreadSummary:
    ensure_chat_storage_ready()
    client = require_supabase_service_client()
    payload = {
        "id": str(uuid4()),
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
def get_chat_thread(chat_id: str) -> ChatThreadDetail:
    ensure_chat_storage_ready()
    thread_row = fetch_thread_row(chat_id)
    client = require_supabase_service_client()
    response = (
        client.table("chat_messages")
        .select("id,role,content,created_at")
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
def update_chat_thread(chat_id: str, request: ChatThreadUpdateRequest) -> ChatThreadSummary:
    ensure_chat_storage_ready()
    fetch_thread_row(chat_id)
    update_payload: dict[str, object] = {"updated_at": utc_now_iso()}
    if request.title is not None:
        update_payload["title"] = normalize_thread_title(request.title)
    if request.archived is not None:
        update_payload["archived"] = request.archived

    client = require_supabase_service_client()
    response = client.table("chat_threads").update(update_payload).eq("id", chat_id).execute()
    rows = response.data or []
    if not rows:
        raise HTTPException(status_code=500, detail="Failed to update chat thread")

    return map_thread_summary(rows[0])


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
    chat_id = request.chat_id or str(uuid4())

    if is_chat_storage_ready():
        if request.chat_id:
            fetch_thread_row(chat_id)
        else:
            upsert_thread_metadata(chat_id, title="Nowy wątek", last_message_preview="")

    latest_user_message = next(
        (message["content"] for message in reversed(sanitized_messages) if message["role"] == "user"),
        "",
    )

    retrieved_chunks = search_chat_chunks(latest_user_message) or search_chunks_supabase(latest_user_message)
    if not api_key:
        demo_reply = build_demo_reply(latest_user_message, retrieved_chunks)
        if is_chat_storage_ready():
            persist_chat_exchange(chat_id, sanitized_messages, demo_reply)
        return ChatResponse(
            reply=demo_reply,
            mode="demo",
            model=model,
            redactions=sorted(set(redactions)),
            chat_id=chat_id if is_chat_storage_ready() else None,
        )

    retrieved_context = build_context_block(retrieved_chunks)
    system_prompt = build_chat_system_prompt(retrieved_context, retrieved_chunks)

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

    if is_chat_storage_ready():
        persist_chat_exchange(chat_id, sanitized_messages, reply)

    return ChatResponse(
        reply=reply,
        mode="live",
        model=model,
        redactions=sorted(set(redactions)),
        chat_id=chat_id if is_chat_storage_ready() else None,
    )


@app.post("/api/rag/search", response_model=RagSearchResponse)
def rag_search(request: RagSearchRequest) -> RagSearchResponse:
    source_types = set(request.source_types or []) or None
    inspection = inspect_search(request.query, limit=request.limit, source_types=source_types)
    chunks = search_chunks(request.query, limit=request.limit, source_types=source_types)
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
