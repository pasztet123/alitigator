from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Literal, Optional, Union
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from postgrest.exceptions import APIError
from pydantic import BaseModel, Field

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
from app.rag import (
    RagChunk,
    build_answer_context_block,
    build_context_block,
    detect_domains,
    detect_mechanisms,
    get_rag_config,
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
HINTS_MODEL = os.getenv("ANTHROPIC_HINTS_MODEL", "claude-haiku-4-5-20251001")
CHAT_MAX_TOKENS = max(1024, int(os.getenv("ANTHROPIC_CHAT_MAX_TOKENS", "6000")))
ANTHROPIC_CHAT_TIMEOUT_SECONDS = max(30.0, float(os.getenv("ANTHROPIC_CHAT_TIMEOUT_SECONDS", "180")))

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
    question: str = Field(min_length=1, max_length=240)
    option_id: str = Field(min_length=1, max_length=64)
    option_label: str = Field(min_length=1, max_length=80)


class RetrievalPreferences(BaseModel):
    include_interpretations: bool = True
    include_judgments: bool = True


ChatRequest.model_rebuild()


class PromptHintsRequest(BaseModel):
    draft: str = Field(min_length=3, max_length=4000)
    intent_hints: list[IntentHintAnswer] = Field(default_factory=list, max_length=12)
    excluded_questions: list[str] = Field(default_factory=list, max_length=24)
    max_hints: int = Field(default=3, ge=1, le=3)


class PromptHint(BaseModel):
    id: str
    question: str
    options: list[PromptHintOption] = Field(min_length=2, max_length=5)


class PromptHintsResponse(BaseModel):
    hints: list[PromptHint]
    model: str
    mode: Literal["live", "fallback"]


class ModelsResponse(BaseModel):
    default_model: str
    models: list[str]


class HealthResponse(BaseModel):
    status: str
    anthropic_configured: bool
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
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
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
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
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
        " Zwróć wyłącznie JSON w postaci {\"hints\":[{\"question\":\"...\",\"options\":[{\"label\":\"...\"}]}]}."
    )
    user_prompt = (
        f"Wersja robocza wiadomości użytkownika:\n{draft}\n\n"
        f"Już zebrane doprecyzowania:\n{answered_context}\n\n"
        f"Pytania, których nie wolno już proponować:\n{excluded_context}\n\n"
        "Jeśli wiadomość jest zbyt krótka albo niejasna, pytania mają pomóc ustalić:"
        " podatek/domenę, czy chodzi o stan faktyczny czy ogólną regułę,"
        " czy sprawa ma element zagraniczny, oraz czy ważny jest konkretny moment w czasie."
    )

    payload = {
        "model": HINTS_MODEL,
        "max_tokens": 220,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            }
        ],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(response.text)
        text = extract_text_from_anthropic(response.json())
        hints = parse_prompt_hints_response(text)
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
    except Exception:
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
        "chunk_patterns": (r"\b(fundacj\w* rodzinn\w*|art\.\s*5\b|dozwolon\w* działalno\w*|nabyte wyłącznie w celu dalszego zbycia|spółkom kapitałowym, w których fundacja rodzinna posiada udziały albo akcje)\b",),
    },
    {
        "id": "family_foundation_pit_exemption",
        "label": "fundacja rodzinna: PIT beneficjenta / proporcja zwolnienia",
        "query_patterns": (r"\b(fundacj\w* rodzinn\w*).{0,120}\b(beneficjent\w*|syn\w* fundator\w*|wypłat\w*|świadczeni\w*)\b|\b(beneficjent\w*|syn\w* fundator\w*).{0,120}\b(fundacj\w* rodzinn\w*)\b",),
        "chunk_patterns": (r"\b(art\.\s*21\s*ust\.\s*1\s*pkt\s*157|art\.\s*21\s*ust\.\s*49|grup\w* zerow\w*|zwolnien\w*.*fundacj\w* rodzin\w*)\b",),
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
) -> str:
    if not retrieved_context:
        return SYSTEM_PROMPT + "\n\nNie znaleziono trafnych fragmentów w indeksie źródeł. Nie twórz pozornych źródeł."

    opening_quote = extract_opening_statute_quote(retrieved_chunks, query=user_prompt)
    opening_instruction = ""
    if opening_quote:
        opening_instruction = (
            " Zacznij odpowiedź od pełnego brzmienia jednego najbardziej trafnego przepisu ustawy,"
            " bez żadnego wstępu, bez nagłówka i bez parafrazy przed cytatem."
            " Nie skracaj przepisu, nie urywaj go wielokropkiem i nie podawaj tylko fragmentu jednostki redakcyjnej."
            " Pierwszy akapit ma być pełnym przepisem albo pełną jednostką redakcyjną zaczynającą się od 'Art.'."
            " Użyj tego przepisu jako punktu wyjścia: \""
            + opening_quote
            + "\". Dopiero po cytacie przejdź do tezy i analizy."
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
    ) if retrieval_coverage_context else ""

    return (
        SYSTEM_PROMPT
        + "\n\nPoniżej znajdują się zweryfikowane dokumenty z indeksu źródeł prawnych."
        + " Dokumenty zostały wybrane przez retrieval chunkowy, ale w kontekście dostajesz pełną treść wybranych dokumentów, jeśli była dostępna w indeksie."
        + " Odpowiadaj wyłącznie na ich podstawie w części źródłowej, a własne wnioski oznaczaj jako wnioski."
        + " Nie traktuj kilku fragmentów albo części tego samego dokumentu jako niezależnych źródeł."
        + " Jeśli źródła są niejednoznaczne albo częściowe, napisz to wprost zamiast domyślać stanowisko."
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
        + " W pytaniach o sprzedaż mienia przez fundację rodzinną oceń osobno, czy materiał pozwala stwierdzić, że mienie nie zostało nabyte wyłącznie w celu dalszego zbycia."
        + " W pytaniach o świadczenia dla beneficjenta fundacji rodzinnej nie zakładaj pełnego zwolnienia PIT bez sprawdzenia, czy źródła potwierdzają zakres zwolnienia i ewentualną proporcję przypisaną fundatorowi."
        + " W pytaniach o mieszane wykorzystanie nieruchomości dla VAT nie utożsamiaj automatycznie proporcji sprzedaży z prewspółczynnikiem."
        + " Użyj prewspółczynnika tylko wtedy, gdy źródła rzeczywiście wskazują na mieszanie działalności gospodarczej z użyciem pozostającym poza działalnością gospodarczą."
        + " W pytaniach o PCC przy nieruchomości nie uogólniaj, że każde zwolnienie z VAT wyłącza PCC."
        + " Jeżeli materiał nie potwierdza wyjątku dla nieruchomości, zaznacz to jako istotną lukę zamiast podawać regułę ogólną bez zastrzeżenia."
        + " Jeżeli wynik podatkowy zależy od skuteczności albo kwalifikacji czynności cywilnoprawnej, najpierw ustal tę skuteczność cywilistyczną"
        + " (na przykład zgoda wierzyciela przy przejęciu długu z art. 519-521 KC albo ważność darowizny z art. 888-890 KC),"
        + " a dopiero potem przechodź do podatków. Nie ustawiaj PIT, PCC i SD jako równorzędnych hipotez bez ustalenia tytułu prawnego."
        + " Jeśli materiał nie pozwala rozstrzygnąć skuteczności cywilnoprawnej, zadaj krótkie pytanie doprecyzowujące zamiast zgadywać skutki podatkowe."
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
        + opening_instruction
        + hint_instruction
        + retrieval_preferences_instruction
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
        anthropic_configured=bool(os.getenv("ANTHROPIC_API_KEY")),
        supabase_configured=is_supabase_configured(),
        rag_index_configured=index_exists(),
        chat_storage_available=is_chat_storage_available(),
        auth_configured=is_supabase_configured(),
        stripe_configured=is_stripe_configured(),
    )


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
    ensure_profile(current_user)
    redactions: list[str] = []
    sanitized_messages: list[dict[str, str]] = []

    for message in request.messages:
        clean_content, applied = redact_text(message.content)
        redactions.extend(applied)
        sanitized_messages.append({"role": message.role, "content": clean_content})

    api_key = os.getenv("ANTHROPIC_API_KEY")
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

    retrieved_chunks = search_chat_chunks(
        effective_user_prompt,
        include_interpretations=(request.retrieval_preferences.include_interpretations if request.retrieval_preferences else True),
        include_judgments=(request.retrieval_preferences.include_judgments if request.retrieval_preferences else None),
    )
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
    if not retrieved_chunks and os.getenv("ALITIGATOR_RAG_BACKEND", "sqlite").strip().lower() == "sqlite":
        retrieved_chunks = search_chunks_supabase(effective_user_prompt)
    if not api_key:
        demo_reply = build_demo_reply(latest_user_message, retrieved_chunks, retrieval_prompt=effective_user_prompt)
        structured_reply = parse_structured_reply(demo_reply)
        persisted_assistant_message = None
        if chat_storage_available:
            persisted_assistant_message = persist_chat_exchange(
                chat_id,
                user_id=current_user.id,
                messages=sanitized_messages,
                reply=demo_reply,
            )
        return ChatResponse(
            reply=demo_reply,
            mode="demo",
            model=model,
            redactions=sorted(set(redactions)),
            chat_id=chat_id if chat_storage_available else None,
            assistant_message_id=(persisted_assistant_message or {}).get("id"),
            structured_reply=structured_reply,
        )

    retrieved_context = build_answer_context_block(retrieved_chunks)
    retrieval_coverage_context = build_retrieval_coverage_context(effective_user_prompt, retrieved_chunks)
    system_prompt = build_chat_system_prompt(
        latest_user_message,
        retrieved_context,
        retrieved_chunks,
        intent_hint_context=intent_hint_context,
        retrieval_preferences_context=retrieval_preferences_context,
        retrieval_coverage_context=retrieval_coverage_context,
    )

    payload = {
        "model": model,
        "max_tokens": CHAT_MAX_TOKENS,
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

    try:
        async with httpx.AsyncClient(timeout=ANTHROPIC_CHAT_TIMEOUT_SECONDS) as client:
            response = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
    except httpx.ReadTimeout as exc:
        raise HTTPException(
            status_code=504,
            detail="Model odpowiadał zbyt długo. Spróbuj ponowić zapytanie albo skrócić zakres odpowiedzi."
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Nie udało się połączyć z modelem odpowiedzi."
        ) from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    reply = extract_text_from_anthropic(response.json())
    if not reply:
        raise HTTPException(status_code=502, detail="Anthropic returned an empty response")

    if retrieved_chunks:
        reply = (
            f"{reply}\n\n"
            "Źródła zwrócone przez retrieval "
            "(nie wszystkie muszą być relewantne ani wykorzystane w analizie)\n"
            f"{list_citations(retrieved_chunks)}"
        )

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

    return ChatResponse(
        reply=reply,
        mode="live",
        model=model,
        redactions=sorted(set(redactions)),
        chat_id=chat_id if chat_storage_available else None,
        assistant_message_id=(persisted_assistant_message or {}).get("id"),
        structured_reply=structured_reply,
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
        context_block=build_answer_context_block(chunks),
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
