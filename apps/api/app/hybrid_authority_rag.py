from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from uuid import uuid4

import httpx

from app.legal_pipeline import LegalClaim
from app.rag import (
    RagChunk,
    build_legal_source_plan,
    chunk_canonical_source_id,
    classify_chunk_evidence_role,
    decompose_query_into_legal_axes,
    detect_domains,
    extract_article_key_from_text,
    extract_legal_rules_from_statute_chunks,
    extract_statute_target_from_text,
    filter_legal_rules_for_target_date,
    get_rag_config,
    inspect_search,
    legal_rule_to_dict,
    legal_source_plan_to_dict,
    normalize_whitespace,
    prioritize_legal_rules_for_query,
    resolve_statute_tax_domains,
    search_chat_chunks,
    select_diverse_chunks,
)


AUTHORITY_CARD_EXTRACTOR_PROMPT_VERSION = "authority_card_extractor_v1"
DEFAULT_AUTHORITY_CARD_SCHEMA_VERSION = "v1"
DEFAULT_AUTHORITY_RERANKER_VERSION = "v1"
DEFAULT_CLARIFIER_MODEL = "claude-haiku-4-5-20251001"


@dataclass(frozen=True)
class HybridAuthorityConfig:
    primary_limit_per_issue: int = 6
    authority_candidate_limit_per_query: int = 10
    authority_selected_limit_per_issue: int = 4
    contrary_limit_per_issue: int = 2
    historical_limit_per_issue: int = 1
    rrf_k: int = 60
    min_authority_score: float = 0.38
    artifact_root: Path = Path("artifacts/hybrid_rag_experiment")
    authority_card_schema_version: str = DEFAULT_AUTHORITY_CARD_SCHEMA_VERSION
    authority_reranker_version: str = DEFAULT_AUTHORITY_RERANKER_VERSION
    authority_card_cache_enabled: bool = True
    fast_sql_primary_candidates: bool = False
    fast_sql_authority_candidates: bool = False
    extractor_model: str = "heuristic_v1"
    clarifier_model: str = DEFAULT_CLARIFIER_MODEL


@dataclass(frozen=True)
class LegalResearchIntent:
    answer_mode: str
    needs_normative_answer: bool
    needs_interpretations: bool
    needs_case_law: bool
    needs_conflict_analysis: bool
    needs_calculations: bool
    needs_clarification: bool
    primary_law_weight: float
    authority_weight: float
    requested_document_types: tuple[str, ...]


@dataclass(frozen=True)
class ClarifierQuestion:
    id: str
    question: str
    reason: str


@dataclass(frozen=True)
class ClarifierResult:
    should_ask: bool
    questions: tuple[ClarifierQuestion, ...] = ()
    missing_dimensions: tuple[str, ...] = ()
    retrieval_assumptions_if_unanswered: tuple[str, ...] = ()
    intent_profile: dict[str, Any] = field(default_factory=dict)
    answers_used: dict[str, str] = field(default_factory=dict)
    augmented_query: str = ""
    model: str = "heuristic"
    mode: str = "disabled"


@dataclass(frozen=True)
class FactGraph:
    entities: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    transactions: tuple[str, ...] = ()
    payments: tuple[str, ...] = ()
    dates: tuple[str, ...] = ()
    jurisdictions: tuple[str, ...] = ()
    relationships: tuple[str, ...] = ()
    known_facts: tuple[str, ...] = ()
    missing_facts: tuple[str, ...] = ()


@dataclass(frozen=True)
class IssueNode:
    issue_id: str
    label: str
    query: str
    tax: str = ""
    mechanism: str = ""
    priority: str = "medium"
    contrast: str = ""
    source_types: tuple[str, ...] = ()
    preferred_targets: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class PrimaryLaneResult:
    issue_id: str
    target_date: str
    queries: tuple[dict[str, str], ...]
    controlling_provisions: tuple[RagChunk, ...] = ()
    dependency_provisions: tuple[RagChunk, ...] = ()
    exception_provisions: tuple[RagChunk, ...] = ()
    historical_provisions_rejected: tuple[dict[str, Any], ...] = ()
    inspections: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class AuthorityCard:
    document_id: str
    signature: str
    document_type: str
    authority: str
    court: str
    date: str
    tax: str
    target_law_period: dict[str, Optional[str]]
    facts: tuple[str, ...]
    issues: tuple[str, ...]
    cited_provisions: tuple[str, ...]
    taxpayer_position: Optional[str]
    authority_holding: Optional[str]
    court_holding: Optional[str]
    outcome: Optional[str]
    result_for_taxpayer: Optional[str]
    legal_reasoning_summary: Optional[str]
    distinguishing_facts: tuple[str, ...]
    temporal_status: str
    source_spans: dict[str, Any]
    source_chunk_id: str = ""
    source_canonical_id: str = ""


@dataclass(frozen=True)
class RerankScore:
    score: float
    dimensions: dict[str, float]
    positive_reasons: tuple[str, ...] = ()
    negative_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoredAuthority:
    issue_id: str
    card: AuthorityCard
    chunk: RagChunk
    candidate_rank: int
    family_scores: dict[str, float]
    filter_status: str
    filter_reasons: tuple[str, ...]
    rerank: RerankScore


@dataclass(frozen=True)
class AuthorityLine:
    issue_id: str
    position_id: str
    holding_summary: str
    supporting_documents: tuple[AuthorityCard, ...] = ()
    contrary_documents: tuple[AuthorityCard, ...] = ()
    historical_documents: tuple[AuthorityCard, ...] = ()
    dominance: str = "unknown"
    distinguishing_facts: tuple[str, ...] = ()
    temporal_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceBundle:
    issue_id: str
    controlling_provisions: tuple[dict[str, Any], ...] = ()
    dependency_provisions: tuple[dict[str, Any], ...] = ()
    supporting_authorities: tuple[dict[str, Any], ...] = ()
    contrary_authorities: tuple[dict[str, Any], ...] = ()
    historical_authorities: tuple[dict[str, Any], ...] = ()
    missing_source_requirements: tuple[str, ...] = ()
    retrieval_confidence: float = 0.0


@dataclass(frozen=True)
class HybridAuthorityResult:
    request_id: str
    run_id: str
    retrieval_mode: str
    clarifier_enabled: bool
    query: str
    retrieval_query: str
    target_date: str
    intent_profile: LegalResearchIntent
    clarifier: ClarifierResult
    fact_graph: FactGraph
    issue_graph: tuple[IssueNode, ...]
    primary_results: tuple[PrimaryLaneResult, ...]
    authority_queries: tuple[dict[str, Any], ...]
    candidate_documents: tuple[dict[str, Any], ...]
    filtered_documents: tuple[dict[str, Any], ...]
    reranked_documents: tuple[dict[str, Any], ...]
    authority_cards: tuple[AuthorityCard, ...]
    authority_lines: tuple[AuthorityLine, ...]
    evidence_bundles: tuple[EvidenceBundle, ...]
    selected_chunks: tuple[RagChunk, ...]
    timings: dict[str, int]
    token_usage: dict[str, int] = field(default_factory=dict)


def get_legal_retrieval_mode() -> str:
    mode = os.getenv("LEGAL_RETRIEVAL_MODE", "baseline").strip().lower()
    return mode if mode in {"baseline", "hybrid_authority"} else "baseline"


def clarifier_enabled_from_env() -> bool:
    return os.getenv("ENABLE_LEGAL_CLARIFIER", "false").strip().lower() in {"1", "true", "yes", "on"}


def get_hybrid_authority_config() -> HybridAuthorityConfig:
    return HybridAuthorityConfig(
        primary_limit_per_issue=max(1, int(os.getenv("HYBRID_RAG_PRIMARY_LIMIT_PER_ISSUE", "6"))),
        authority_candidate_limit_per_query=max(3, int(os.getenv("HYBRID_RAG_AUTHORITY_CANDIDATE_LIMIT", "10"))),
        authority_selected_limit_per_issue=max(1, int(os.getenv("HYBRID_RAG_SUPPORTING_LIMIT", "4"))),
        contrary_limit_per_issue=max(0, int(os.getenv("HYBRID_RAG_CONTRARY_LIMIT", "2"))),
        historical_limit_per_issue=max(0, int(os.getenv("HYBRID_RAG_HISTORICAL_LIMIT", "1"))),
        min_authority_score=float(os.getenv("HYBRID_RAG_MIN_AUTHORITY_SCORE", "0.38")),
        artifact_root=Path(os.getenv("HYBRID_RAG_ARTIFACT_ROOT", "artifacts/hybrid_rag_experiment")),
        authority_card_schema_version=os.getenv("AUTHORITY_CARD_SCHEMA_VERSION", DEFAULT_AUTHORITY_CARD_SCHEMA_VERSION),
        authority_reranker_version=os.getenv("AUTHORITY_RERANKER_VERSION", DEFAULT_AUTHORITY_RERANKER_VERSION),
        authority_card_cache_enabled=os.getenv("AUTHORITY_CARD_CACHE_ENABLED", "true").strip().lower() not in {"0", "false", "no"},
        fast_sql_primary_candidates=os.getenv("HYBRID_RAG_FAST_SQL_PRIMARY", "false").strip().lower() in {"1", "true", "yes"},
        fast_sql_authority_candidates=os.getenv("HYBRID_RAG_FAST_SQL_AUTHORITY", "false").strip().lower() in {"1", "true", "yes"},
        extractor_model=os.getenv("AUTHORITY_CARD_EXTRACTOR_MODEL", "heuristic_v1"),
        clarifier_model=os.getenv("ANTHROPIC_CLARIFIER_MODEL", DEFAULT_CLARIFIER_MODEL),
    )


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = normalize_whitespace(str(value or "")).strip()
        key = clean.lower()
        if not clean or key in seen:
            continue
        seen.add(key)
        deduped.append(clean)
    return tuple(deduped)


def _target_date_from_query(query: str) -> str:
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", query)
    if match:
        return match.group(1)
    year_match = re.search(r"\b(20\d{2})\b", query)
    if year_match:
        return f"{year_match.group(1)}-12-31"
    return datetime.now(timezone.utc).date().isoformat()


def classify_legal_research_intent(query: str) -> LegalResearchIntent:
    lowered = normalize_whitespace(query).lower()
    authority_terms = (
        "znajdź interpretacje",
        "znajdz interpretacje",
        "interpretacje i wyroki",
        "orzecznictwo",
        "linia organ",
        "linia interpretacyjna",
        "wyroki",
        "nsa",
        "wsa",
        "praktyka organ",
    )
    rule_terms = (
        "jaki przepis",
        "podstawa prawna",
        "aktualna reguła",
        "aktualna regula",
        "co mówi ustawa",
        "co mowi ustawa",
        "brzmienie art.",
    )
    calculation_terms = ("ile wynosi", "oblicz", "wylicz", "kwota", "limit", "stawka")

    if any(term in lowered for term in authority_terms):
        mode = "authority_research"
        primary_weight = 0.35
        authority_weight = 0.65
    elif any(term in lowered for term in rule_terms) and not re.search(r"\b(kazus|stan faktyczny|transakcj|rozlicz|ryzyk)\w*", lowered):
        mode = "rule_first"
        primary_weight = 0.72
        authority_weight = 0.28
    else:
        mode = "mixed_analysis"
        primary_weight = 0.50
        authority_weight = 0.50

    requested_document_types = ["individual_interpretation", "general_interpretation"]
    if mode == "authority_research" or re.search(r"\b(wyrok\w*|orzecze|nsa|wsa|sąd|sad)\b", lowered):
        requested_document_types.extend(["wsa_judgment", "nsa_judgment"])
    elif mode == "mixed_analysis":
        requested_document_types.extend(["wsa_judgment", "nsa_judgment"])

    needs_conflict = mode == "authority_research" or bool(
        re.search(r"\b(rozbież|rozbiezn|przeciw|ryzyk|linia|spór|spor)\w*", lowered)
    )
    needs_clarification = _query_needs_clarification(lowered)
    return LegalResearchIntent(
        answer_mode=mode,
        needs_normative_answer=True,
        needs_interpretations=True,
        needs_case_law=True,
        needs_conflict_analysis=needs_conflict,
        needs_calculations=any(term in lowered for term in calculation_terms),
        needs_clarification=needs_clarification,
        primary_law_weight=primary_weight,
        authority_weight=authority_weight,
        requested_document_types=tuple(dict.fromkeys(requested_document_types)),
    )


def _query_needs_clarification(lowered_query: str) -> bool:
    if len(lowered_query.split()) < 5:
        return True
    if re.search(r"\bwht|podatek u źródła|podatek u zrodla\b", lowered_query) and not re.search(
        r"\b(odset|dywidend|licencyj|zarządz|zarzadz|doradcz|usług|uslug)\w*", lowered_query
    ):
        return True
    if re.search(r"\bsprzedaż|sprzedaz|zbycie\b", lowered_query) and not re.search(
        r"\b(nieruchomo|udział|udzial|akcj|samoch|towar|przedsiębior|przedsiebior)\w*", lowered_query
    ):
        return True
    if re.search(r"\btransakcj\w*\b", lowered_query) and not detect_domains(lowered_query):
        return True
    return False


def build_fact_graph(query: str, clarifier_answers: Optional[dict[str, str]] = None) -> FactGraph:
    lowered = normalize_whitespace(query).lower()
    entities = _dedupe(re.findall(r"\b[A-ZŁŚŻŹĆŃÓ][\wŁŚŻŹĆŃÓąęółśżźćń-]{2,}\b", query))
    roles = _dedupe(
        role
        for pattern, role in (
            (r"\bpodatnik\w*", "taxpayer"),
            (r"\bpłatnik\w*|\bplatnik\w*", "withholding_agent"),
            (r"\bbeneficjent\w*", "beneficiary"),
            (r"\bfundator\w*", "founder"),
            (r"\bsprzedawc\w*|\bzbywc\w*", "seller"),
            (r"\bnabywc\w*|\bkupując\w*|\bkupujac\w*", "buyer"),
            (r"\bspółk\w*|\bspolk\w*", "company"),
            (r"\borgan\w*", "tax_authority"),
            (r"\bnierezydent\w*", "non_resident"),
        )
        if re.search(pattern, lowered)
    )
    transactions = _dedupe(
        label
        for pattern, label in (
            (r"\bsprzedaż|sprzedaz|zbycie\b", "sale"),
            (r"\bnajem|dzierżaw|dzierzaw\b", "lease"),
            (r"\bdarowizn\w*", "gift"),
            (r"\baport\w*", "in_kind_contribution"),
            (r"\bprzekształc|przeksztalc\b", "transformation"),
            (r"\bleasing\w*", "leasing"),
            (r"\bspłat\w* kredyt|\bsplat\w* kredyt", "loan_repayment"),
            (r"\bpożyczk|pozyczk\b", "loan"),
        )
        if re.search(pattern, lowered)
    )
    payments = _dedupe(
        label
        for pattern, label in (
            (r"\bodsetk\w*", "interest"),
            (r"\bdywidend\w*", "dividend"),
            (r"\blicencyjn\w*|royalt", "royalty"),
            (r"\bdoradcz\w*", "advisory_services"),
            (r"\bzarządz|zarzadz", "management_services"),
            (r"\bświadczen\w*|swiadczen\w*", "benefit"),
            (r"\bczynsz\w*", "rent"),
            (r"\bkredyt\w*", "credit"),
        )
        if re.search(pattern, lowered)
    )
    jurisdictions = _dedupe(
        label
        for pattern, label in (
            (r"\bpolsk\w*|\bPL\b", "PL"),
            (r"\bniemc\w*|\bgerman", "DE"),
            (r"\bhiszpani\w*|\bspain", "ES"),
            (r"\bczech\w*", "CZ"),
            (r"\busa|stany zjednoczone", "US"),
            (r"\bholandi\w*|\bniderland", "NL"),
            (r"\bfrancj\w*", "FR"),
        )
        if re.search(pattern, lowered, re.IGNORECASE)
    )
    relationships = _dedupe(
        label
        for pattern, label in (
            (r"\bpowiązan\w*|powiazan\w*", "related_parties"),
            (r"\bmałżonk|malzonk\w*", "spouses"),
            (r"\bdzieck\w*|zstępn\w*|zstepn\w*", "descendant"),
            (r"\bwspólnik\w*|wspolnik\w*", "shareholder"),
        )
        if re.search(pattern, lowered)
    )
    dates = _dedupe(re.findall(r"\b20\d{2}(?:-\d{2}-\d{2})?\b", query))
    known_facts = _dedupe([*roles, *transactions, *payments, *jurisdictions, *relationships])
    missing_facts = tuple(clarifier_answers or ())
    return FactGraph(
        entities=entities,
        roles=roles,
        transactions=transactions,
        payments=payments,
        dates=dates,
        jurisdictions=jurisdictions,
        relationships=relationships,
        known_facts=known_facts,
        missing_facts=missing_facts,
    )


def build_issue_graph(query: str, intent: LegalResearchIntent, fact_graph: FactGraph) -> tuple[IssueNode, ...]:
    axes = decompose_query_into_legal_axes(query)
    issues: list[IssueNode] = []
    for axis in axes:
        tax = sorted(axis.tax_domains or [])[:1]
        mechanism = _mechanism_from_axis(axis.axis_id, axis.label)
        issues.append(
            IssueNode(
                issue_id=axis.axis_id,
                label=axis.label,
                query=axis.query or query,
                tax=tax[0] if tax else "",
                mechanism=mechanism,
                priority="high" if axis.preferred_targets or intent.answer_mode != "rule_first" else "medium",
                contrast=_contrast_hint(axis.axis_id, fact_graph),
                source_types=tuple(sorted(axis.source_types or ())),
                preferred_targets=axis.preferred_targets,
            )
        )

    if not issues:
        domains = tuple(sorted(resolve_statute_tax_domains(query) or detect_domains(query)))
        if not domains and "WHT" in (item.upper() for item in fact_graph.payments):
            domains = ("CIT",)
        if not domains:
            domains = ("TAX",)
        for domain in domains:
            issues.append(
                IssueNode(
                    issue_id=f"{domain.lower()}_general_tax_issue",
                    label=f"{domain}: general tax issue",
                    query=query,
                    tax=domain,
                    mechanism=fact_graph.transactions[0] if fact_graph.transactions else "",
                    priority="high",
                    contrast=_contrast_hint("", fact_graph),
                    source_types=("statute", "interpretation", "judgment"),
                )
            )

    return tuple(_dedupe_issues(issues))


def _dedupe_issues(issues: Iterable[IssueNode]) -> list[IssueNode]:
    deduped: list[IssueNode] = []
    seen: set[str] = set()
    for issue in issues:
        if issue.issue_id in seen:
            continue
        seen.add(issue.issue_id)
        deduped.append(issue)
    return deduped


def _mechanism_from_axis(axis_id: str, label: str) -> str:
    text = f"{axis_id} {label}".lower()
    for token in (
        "bad_debt",
        "housing_relief",
        "ksef",
        "wht",
        "beneficial_owner",
        "family_foundation",
        "estonian_cit",
        "fixed_establishment",
        "vehicle",
        "real_estate",
    ):
        if token in text:
            return token
    return ""


def _contrast_hint(axis_id: str, fact_graph: FactGraph) -> str:
    values = set(fact_graph.transactions) | set(fact_graph.payments)
    if "loan_repayment" in values or "credit" in values:
        return "credit_on_sold_property_vs_credit_on_new_property"
    if "interest" in values:
        return "interest_vs_advisory_or_management_services"
    if "advisory_services" in values or "management_services" in values:
        return "services_vs_interest_or_royalties"
    if "beneficiary" in fact_graph.roles or "founder" in fact_graph.roles:
        return "founder_vs_beneficiary_benefit"
    if "vehicle" in axis_id:
        return "private_car_vs_business_car"
    if "spolka_komandytowa" in axis_id or "komandyt" in axis_id:
        return "pre_cit_vs_post_cit_limited_partnership"
    return ""


def build_retrieval_clarification(
    query: str,
    intent: LegalResearchIntent,
    *,
    enabled: bool,
    fixture_answers: Optional[dict[str, str]] = None,
    config: Optional[HybridAuthorityConfig] = None,
) -> ClarifierResult:
    config = config or get_hybrid_authority_config()
    if not enabled:
        return ClarifierResult(
            should_ask=False,
            augmented_query=query,
            intent_profile=to_jsonable(intent),
            mode="disabled",
        )
    if fixture_answers:
        answer_text = "; ".join(f"{key}: {value}" for key, value in sorted(fixture_answers.items()))
        return ClarifierResult(
            should_ask=False,
            answers_used=dict(fixture_answers),
            augmented_query=f"{query}\n\nDoprecyzowania z fixture: {answer_text}",
            intent_profile=to_jsonable(intent),
            model="fixture",
            mode="fixture",
        )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        model_result = _request_claude_clarifier(query, intent, api_key=api_key, config=config)
        if model_result is not None:
            return model_result

    questions = _heuristic_clarifier_questions(query, intent)
    return ClarifierResult(
        should_ask=bool(questions),
        questions=tuple(questions),
        missing_dimensions=tuple(question.id for question in questions),
        augmented_query=query,
        intent_profile=to_jsonable(intent),
        model="heuristic",
        mode="questions_only",
    )


def _request_claude_clarifier(
    query: str,
    intent: LegalResearchIntent,
    *,
    api_key: str,
    config: HybridAuthorityConfig,
) -> Optional[ClarifierResult]:
    system_prompt = (
        "Jestes modulem doprecyzowania retrievalu podatkowego. Nie odpowiadasz na problem prawny. "
        "Wykryj tylko brakujace fakty, ktore realnie zmieniaja wybor przepisow, interpretacji lub orzeczen. "
        "Zadaj maksymalnie trzy krotkie pytania. Nie pytaj o informacje inferowalne z pytania. "
        "Zwroc wylacznie JSON: {\"should_ask\":bool,\"questions\":[{\"id\":\"...\",\"question\":\"...\",\"reason\":\"...\"}],"
        "\"missing_dimensions\":[],\"retrieval_assumptions_if_unanswered\":[],\"intent_profile\":{}}."
    )
    user_prompt = (
        f"Pytanie uzytkownika:\n{query}\n\n"
        f"Profil intencji:\n{json.dumps(to_jsonable(intent), ensure_ascii=False)}"
    )
    payload = {
        "model": config.clarifier_model,
        "max_tokens": 550,
        "temperature": 0,
        "system": system_prompt,
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages"), headers=headers, json=payload)
        if response.status_code >= 400:
            return None
        text = _extract_text_from_anthropic(response.json())
        parsed = json.loads(_extract_json_object(text))
        questions = tuple(
            ClarifierQuestion(
                id=str(item.get("id") or f"q{index}").strip()[:80],
                question=str(item.get("question") or "").strip()[:240],
                reason=str(item.get("reason") or "").strip()[:400],
            )
            for index, item in enumerate(parsed.get("questions") or [], start=1)
            if isinstance(item, dict) and str(item.get("question") or "").strip()
        )[:3]
        return ClarifierResult(
            should_ask=bool(parsed.get("should_ask") and questions),
            questions=questions,
            missing_dimensions=tuple(str(item) for item in parsed.get("missing_dimensions") or [] if str(item).strip()),
            retrieval_assumptions_if_unanswered=tuple(
                str(item) for item in parsed.get("retrieval_assumptions_if_unanswered") or [] if str(item).strip()
            ),
            intent_profile=dict(parsed.get("intent_profile") or {}),
            augmented_query=query,
            model=config.clarifier_model,
            mode="claude_questions_only",
        )
    except Exception:
        return None


def _extract_text_from_anthropic(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in payload.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "\n".join(parts).strip()


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object in clarifier response")
    return text[start : end + 1]


def _heuristic_clarifier_questions(query: str, intent: LegalResearchIntent) -> list[ClarifierQuestion]:
    lowered = normalize_whitespace(query).lower()
    questions: list[ClarifierQuestion] = []
    if re.search(r"\bwht|podatek u źródła|podatek u zrodla\b", lowered) and not re.search(
        r"\b(odset|dywidend|licencyj|doradcz|zarządz|zarzadz|usług|uslug)\w*", lowered
    ):
        questions.append(
            ClarifierQuestion(
                id="payment_type",
                question="Jakiego rodzaju płatności dotyczy problem?",
                reason="Rodzaj płatności zmienia przepisy WHT i zestaw relewantnych interpretacji oraz wyroków.",
            )
        )
    if re.search(r"\btransakcj|sprzedaż|sprzedaz|zbycie\b", lowered) and not re.search(r"\b20\d{2}\b", lowered):
        questions.append(
            ClarifierQuestion(
                id="transaction_date",
                question="Jakiej daty lub roku dotyczy transakcja?",
                reason="Data wpływa na wersję przepisów i ocenę aktualności authorities.",
            )
        )
    if intent.answer_mode != "authority_research" and not detect_domains(lowered):
        questions.append(
            ClarifierQuestion(
                id="tax_domain",
                question="Którego podatku dotyczy problem?",
                reason="Domena podatku zawęża primary law i ogranicza nietrafne authorities.",
            )
        )
    return questions[:3]


def build_primary_queries(issue: IssueNode, query: str, fact_graph: FactGraph) -> tuple[dict[str, str], ...]:
    targets = " ".join(f"art. {article}" for _domain, article in issue.preferred_targets)
    facts = " ".join([*fact_graph.transactions, *fact_graph.payments, *fact_graph.roles])
    queries = [
        {"family": "natural_language", "query": issue.query or query},
        {"family": "issue_signature", "query": normalize_whitespace(f"{issue.label} {issue.tax} {issue.mechanism} {facts} {targets}")},
    ]
    if targets:
        queries.append({"family": "explicit_or_preferred_provision", "query": normalize_whitespace(f"{issue.tax} {targets} {issue.label}")})
    return tuple(query_item for query_item in queries if query_item["query"])


def run_primary_lane(issue: IssueNode, query: str, fact_graph: FactGraph, *, config: HybridAuthorityConfig) -> PrimaryLaneResult:
    target_date = _target_date_from_query(query)
    chunks: list[RagChunk] = []
    inspections: list[dict[str, Any]] = []
    for query_item in build_primary_queries(issue, query, fact_graph):
        if config.fast_sql_primary_candidates:
            found_chunks, hits = _fast_sql_authority_candidates(
                query_item["query"],
                issue=issue,
                source_types={"statute"},
                limit=config.primary_limit_per_issue,
            )
            retrieved_count = len(found_chunks)
            match_query = "fast_sql_primary"
        else:
            inspection = inspect_search(
                query_item["query"],
                limit=config.primary_limit_per_issue,
                source_types={"statute"},
                enforce_query_domain=bool(issue.tax),
                tax_domains={issue.tax} if issue.tax and issue.tax != "TAX" else None,
            )
            found_chunks = inspection.chunks
            hits = inspection.hits[: config.primary_limit_per_issue]
            retrieved_count = inspection.retrieved_count
            match_query = inspection.match_query
        inspections.append(
            {
                "family": query_item["family"],
                "query": query_item["query"],
                "match_query": match_query,
                "retrieved_count": retrieved_count,
                "hits": hits,
            }
        )
        chunks.extend(found_chunks)
    deduped = _dedupe_chunks(chunks)
    controlling = tuple(chunk for chunk in deduped if _chunk_is_temporally_current(chunk, target_date))
    historical = tuple(
        {"document_id": chunk.document_id, "signature": chunk.signature, "date": chunk.published_date or chunk.legal_state_date}
        for chunk in deduped
        if chunk not in controlling
    )
    return PrimaryLaneResult(
        issue_id=issue.issue_id,
        target_date=target_date,
        queries=build_primary_queries(issue, query, fact_graph),
        controlling_provisions=controlling[: config.primary_limit_per_issue],
        dependency_provisions=(),
        exception_provisions=(),
        historical_provisions_rejected=historical,
        inspections=tuple(inspections),
    )


def build_authority_queries(
    issue: IssueNode,
    query: str,
    fact_graph: FactGraph,
    primary_result: PrimaryLaneResult,
) -> tuple[dict[str, Any], ...]:
    facts = " ".join([*fact_graph.transactions, *fact_graph.payments, *fact_graph.roles, *fact_graph.relationships])
    provision_terms = _provision_terms_from_chunks(primary_result.controlling_provisions)
    queries: list[dict[str, Any]] = [
        {"issue_id": issue.issue_id, "family": "natural_language", "query": query, "weight": 1.0},
        {
            "issue_id": issue.issue_id,
            "family": "issue_signature",
            "query": normalize_whitespace(f"{issue.label} {issue.tax} {issue.mechanism} {facts}"),
            "weight": 1.15,
        },
    ]
    if provision_terms:
        queries.append(
            {
                "issue_id": issue.issue_id,
                "family": "provision_anchored",
                "query": normalize_whitespace(f"{' '.join(provision_terms[:8])} {facts} {issue.label}"),
                "weight": 1.35,
            }
        )
    contrast_query = _factual_contrast_query(issue, fact_graph)
    if contrast_query:
        queries.append(
            {
                "issue_id": issue.issue_id,
                "family": "factual_contrast",
                "query": contrast_query,
                "weight": 0.9,
            }
        )
    return tuple(item for item in queries if item["query"])


def _provision_terms_from_chunks(chunks: Iterable[RagChunk]) -> list[str]:
    terms: list[str] = []
    for chunk in chunks:
        for provision in chunk.legal_provisions:
            article = extract_article_key_from_text(provision)
            if article:
                terms.append(f"art. {article}")
            else:
                terms.append(provision)
    return list(dict.fromkeys(terms))


def _factual_contrast_query(issue: IssueNode, fact_graph: FactGraph) -> str:
    if issue.contrast == "credit_on_sold_property_vs_credit_on_new_property":
        return "spłata kredytu zaciągniętego na zbywaną nieruchomość art. 21 ust. 30a nie kredyt na nową nieruchomość"
    if issue.contrast == "interest_vs_advisory_or_management_services":
        return "podatek u źródła odsetki beneficial owner nie usługi doradcze zarządzania"
    if issue.contrast == "services_vs_interest_or_royalties":
        return "podatek u źródła usługi doradcze zarządzania nie odsetki nie należności licencyjne"
    if issue.contrast == "founder_vs_beneficiary_benefit":
        return "fundacja rodzinna świadczenie dla fundatora beneficjenta różnice PIT CIT ukryte zyski"
    if "private_car_vs_business_car" == issue.contrast:
        return "samochód prywatny działalność gospodarcza limit 20% 75% różnice"
    return " ".join([*fact_graph.transactions, *fact_graph.payments, issue.label]).strip()


def run_hybrid_authority_retrieval(
    query: str,
    *,
    include_interpretations: bool = True,
    include_judgments: Optional[bool] = None,
    clarifier_enabled: bool = False,
    clarification_fixture: Optional[dict[str, str]] = None,
    request_id: Optional[str] = None,
    config: Optional[HybridAuthorityConfig] = None,
) -> HybridAuthorityResult:
    config = config or get_hybrid_authority_config()
    timings: dict[str, int] = {}
    started = time.perf_counter()
    request_id = request_id or str(uuid4())
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{request_id[:8]}"

    intent = classify_legal_research_intent(query)
    clarifier = build_retrieval_clarification(
        query,
        intent,
        enabled=clarifier_enabled,
        fixture_answers=clarification_fixture,
        config=config,
    )
    retrieval_query = clarifier.augmented_query or query
    fact_graph = build_fact_graph(retrieval_query, clarifier.answers_used)
    issue_graph = build_issue_graph(retrieval_query, intent, fact_graph)
    timings["planning_ms"] = _elapsed_ms(started)

    primary_start = time.perf_counter()
    primary_results = tuple(
        run_primary_lane(issue, retrieval_query, fact_graph, config=config)
        for issue in issue_graph
    )
    timings["primary_retrieval_ms"] = _elapsed_ms(primary_start)

    authority_start = time.perf_counter()
    all_authority_queries: list[dict[str, Any]] = []
    candidate_documents: list[dict[str, Any]] = []
    scored_by_issue: list[ScoredAuthority] = []
    card_by_source: dict[str, AuthorityCard] = {}

    for issue, primary_result in zip(issue_graph, primary_results):
        authority_queries = list(build_authority_queries(issue, retrieval_query, fact_graph, primary_result))
        all_authority_queries.extend(authority_queries)
        candidates = _generate_authority_candidates(
            issue,
            authority_queries,
            include_interpretations=include_interpretations,
            include_judgments=include_judgments,
            config=config,
        )
        candidate_documents.extend(candidates["candidate_documents"])

        first_pass_chunks = [item["chunk"] for item in candidates["ranked_candidates"][:8]]
        first_cards = [
            extract_authority_card(chunk, target_date=primary_result.target_date, config=config)
            for chunk in first_pass_chunks
        ]
        for backref_query in _build_authority_backref_queries(issue, first_cards):
            all_authority_queries.append(backref_query)
            extra = _generate_authority_candidates(
                issue,
                [backref_query],
                include_interpretations=include_interpretations,
                include_judgments=include_judgments,
                config=config,
            )
            candidate_documents.extend(extra["candidate_documents"])
            candidates["ranked_candidates"].extend(extra["ranked_candidates"])

        for rank, candidate in enumerate(candidates["ranked_candidates"], start=1):
            chunk = candidate["chunk"]
            card = extract_authority_card(chunk, target_date=primary_result.target_date, config=config)
            card_by_source[card.source_canonical_id or chunk_canonical_source_id(chunk)] = card
            filter_status, filter_reasons = prefilter_authority_card(card, issue, intent, fact_graph, primary_result)
            rerank = score_authority_card(
                card,
                issue=issue,
                fact_graph=fact_graph,
                primary_result=primary_result,
                family_score=float(candidate["fusion_score"]),
                filter_status=filter_status,
            )
            scored_by_issue.append(
                ScoredAuthority(
                    issue_id=issue.issue_id,
                    card=card,
                    chunk=chunk,
                    candidate_rank=rank,
                    family_scores=dict(candidate["family_scores"]),
                    filter_status=filter_status,
                    filter_reasons=tuple(filter_reasons),
                    rerank=rerank,
                )
            )
    timings["authority_retrieval_ms"] = _elapsed_ms(authority_start)

    rerank_start = time.perf_counter()
    scored_by_issue = _dedupe_scored_authorities(scored_by_issue)
    scored_by_issue.sort(key=lambda item: (item.issue_id, -item.rerank.score, item.candidate_rank))
    authority_lines = build_authority_lines(scored_by_issue, config=config)
    evidence_bundles = build_evidence_bundles(
        issue_graph,
        primary_results,
        scored_by_issue,
        config=config,
    )
    selected_chunks = _select_hybrid_chunks(primary_results, scored_by_issue, evidence_bundles)
    timings["reranking_ms"] = _elapsed_ms(rerank_start)
    timings["total_ms"] = _elapsed_ms(started)

    return HybridAuthorityResult(
        request_id=request_id,
        run_id=run_id,
        retrieval_mode="hybrid_authority",
        clarifier_enabled=clarifier_enabled,
        query=query,
        retrieval_query=retrieval_query,
        target_date=_target_date_from_query(retrieval_query),
        intent_profile=intent,
        clarifier=clarifier,
        fact_graph=fact_graph,
        issue_graph=issue_graph,
        primary_results=primary_results,
        authority_queries=tuple(all_authority_queries),
        candidate_documents=tuple(candidate_documents),
        filtered_documents=tuple(_filtered_trace(scored_by_issue)),
        reranked_documents=tuple(_reranked_trace(scored_by_issue)),
        authority_cards=tuple(card_by_source.values()),
        authority_lines=authority_lines,
        evidence_bundles=evidence_bundles,
        selected_chunks=tuple(selected_chunks),
        timings=timings,
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _generate_authority_candidates(
    issue: IssueNode,
    authority_queries: list[dict[str, Any]],
    *,
    include_interpretations: bool,
    include_judgments: Optional[bool],
    config: HybridAuthorityConfig,
) -> dict[str, Any]:
    source_types: set[str] = set()
    if include_interpretations:
        source_types.add("interpretation")
    if include_judgments is not False:
        source_types.add("judgment")
    if not source_types:
        return {"ranked_candidates": [], "candidate_documents": []}

    by_source: dict[str, dict[str, Any]] = {}
    candidate_documents: list[dict[str, Any]] = []
    for query_item in authority_queries:
        if config.fast_sql_authority_candidates:
            chunks, hits = _fast_sql_authority_candidates(
                str(query_item["query"]),
                issue=issue,
                source_types=source_types,
                limit=config.authority_candidate_limit_per_query,
            )
        else:
            inspection = inspect_search(
                str(query_item["query"]),
                limit=config.authority_candidate_limit_per_query,
                source_types=source_types,
                enforce_query_domain=bool(issue.tax and issue.tax != "TAX"),
                tax_domains={issue.tax} if issue.tax and issue.tax != "TAX" else None,
            )
            chunks = inspection.chunks
            hits = inspection.hits
        weight = float(query_item.get("weight") or 1.0)
        family = str(query_item.get("family") or "unknown")
        for rank, chunk in enumerate(chunks, start=1):
            source_id = chunk_canonical_source_id(chunk)
            entry = by_source.setdefault(
                source_id,
                {
                    "chunk": chunk,
                    "fusion_score": 0.0,
                    "best_rank": rank,
                    "family_scores": {},
                },
            )
            entry["fusion_score"] += weight / (config.rrf_k + rank)
            entry["best_rank"] = min(int(entry["best_rank"]), rank)
            entry["family_scores"][family] = entry["family_scores"].get(family, 0.0) + weight / (config.rrf_k + rank)
        candidate_documents.extend(
            {
                "issue_id": issue.issue_id,
                "query_family": family,
                "query": query_item["query"],
                "rank": hit.get("rank"),
                "document_id": hit.get("document_id"),
                "signature": hit.get("signature"),
                "source_type": hit.get("source_type"),
                "subject": hit.get("subject"),
                "score": hit.get("score"),
            }
            for hit in hits
        )
    ranked = sorted(by_source.values(), key=lambda item: (-float(item["fusion_score"]), int(item["best_rank"])))
    return {"ranked_candidates": ranked, "candidate_documents": candidate_documents}


def _fast_sql_authority_candidates(
    query: str,
    *,
    issue: IssueNode,
    source_types: set[str],
    limit: int,
) -> tuple[list[RagChunk], list[dict[str, Any]]]:
    config = get_rag_config()
    if not config.db_path.exists():
        return [], []
    terms = [term for term in _terms(query) if len(term) >= 4][:8]
    if not terms:
        terms = [query[:40]]
    type_placeholders = ", ".join("?" for _ in source_types)
    values: list[Any] = [*sorted(source_types)]
    domain_clause = ""
    if issue.tax and issue.tax != "TAX":
        domain_clause = " AND (UPPER(d.tax_domain) = ? OR d.legal_provisions_json LIKE ?)"
        values.extend([issue.tax.upper(), f"%[{issue.tax.upper()}]%"])
    like_clauses: list[str] = []
    for term in terms:
        like_clauses.append(
            "(d.subject LIKE ? OR d.question_text LIKE ? OR d.facts_text LIKE ? OR d.legal_provisions_json LIKE ? OR c.chunk_text LIKE ?)"
        )
        like_value = f"%{term}%"
        values.extend([like_value, like_value, like_value, like_value, like_value])
    values.append(max(limit * 8, limit))
    sql = f"""
        SELECT
            c.chunk_id, c.document_id, c.chunk_index, c.chunk_text,
            d.subject, d.signature, d.published_date, d.source_url, d.category,
            d.legal_provisions_json, d.source, d.source_type, d.source_subtype,
            d.authority, d.publication, d.legal_state_date, d.source_pages_json,
            d.tax_domain
        FROM chunks c
        JOIN documents d ON d.document_id = c.document_id
        WHERE d.source_type IN ({type_placeholders})
          AND c.chunk_index = 0
          {domain_clause}
          AND ({" OR ".join(like_clauses)})
        ORDER BY d.published_date DESC, c.chunk_index ASC
        LIMIT ?
    """
    connection = sqlite3.connect(config.db_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(sql, tuple(values)).fetchall()
    finally:
        connection.close()
    scored_rows = sorted(
        rows,
        key=lambda row: (
            -_fast_sql_row_score(row, terms, issue),
            str(row["published_date"] or ""),
            int(row["chunk_index"]),
        ),
    )
    chunks = [_row_to_rag_chunk(row, score=_fast_sql_row_score(row, terms, issue)) for row in scored_rows[:limit]]
    hits = [
        {
            "rank": rank,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "score": chunk.score,
            "canonical_source_id": chunk_canonical_source_id(chunk),
            "evidence_role": classify_chunk_evidence_role(chunk),
            "subject": chunk.subject,
            "signature": chunk.signature,
            "published_date": chunk.published_date,
            "source_url": chunk.source_url,
            "category": chunk.category,
            "source": chunk.source,
            "source_type": chunk.source_type,
            "source_subtype": chunk.source_subtype,
            "authority": chunk.authority,
            "publication": chunk.publication,
            "legal_state_date": chunk.legal_state_date,
            "source_pages": chunk.source_pages,
            "legal_provisions": chunk.legal_provisions,
            "chunk_chars": len(chunk.chunk_text),
            "preview": chunk.chunk_text[:280],
            "selected_for_context": True,
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]
    return chunks, hits


def _fast_sql_row_score(row: sqlite3.Row, terms: list[str], issue: IssueNode) -> float:
    haystack = normalize_whitespace(
        " ".join(
            [
                str(row["subject"] or ""),
                str(row["signature"] or ""),
                str(row["legal_provisions_json"] or ""),
                str(row["chunk_text"] or "")[:1600],
            ]
        )
    ).lower()
    score = 0.0
    for term in terms:
        if term.lower() in haystack:
            score += 1.0
    if issue.tax and issue.tax.upper() in str(row["tax_domain"] if "tax_domain" in row.keys() else "").upper():
        score += 0.5
    if str(row["source_type"] or "") == "judgment":
        score += 0.15
    return score


def _row_to_rag_chunk(row: sqlite3.Row, *, score: float) -> RagChunk:
    return RagChunk(
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["document_id"]),
        chunk_index=int(row["chunk_index"]),
        score=float(score),
        chunk_text=str(row["chunk_text"] or ""),
        subject=str(row["subject"] or ""),
        signature=str(row["signature"] or "") or None,
        published_date=str(row["published_date"] or "") or None,
        source_url=str(row["source_url"] or "") or None,
        category=str(row["category"] or "") or None,
        source=str(row["source"] or ""),
        source_type=str(row["source_type"] or "interpretation"),
        source_subtype=str(row["source_subtype"] or "") or None,
        authority=str(row["authority"] or "") or None,
        publication=str(row["publication"] or "") or None,
        legal_state_date=str(row["legal_state_date"] or "") or None,
        source_pages=[int(value) for value in json.loads(row["source_pages_json"] or "[]")],
        legal_provisions=[str(value) for value in json.loads(row["legal_provisions_json"] or "[]")],
    )


def _build_authority_backref_queries(issue: IssueNode, cards: list[AuthorityCard]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for card in cards:
        for provision in card.cited_provisions[:4]:
            article = extract_article_key_from_text(provision)
            if not article:
                continue
            query = normalize_whitespace(f"art. {article} {issue.label} {card.tax} {card.signature}")
            if query.lower() in seen:
                continue
            seen.add(query.lower())
            queries.append(
                {
                    "issue_id": issue.issue_id,
                    "family": "authority_back_reference",
                    "query": query,
                    "weight": 0.75,
                }
            )
            if len(queries) >= 4:
                return queries
    return queries


def extract_authority_card(
    chunk: RagChunk,
    *,
    target_date: str,
    config: Optional[HybridAuthorityConfig] = None,
) -> AuthorityCard:
    config = config or get_hybrid_authority_config()
    cache_key = _authority_card_cache_key(chunk, config)
    cache_path = config.artifact_root / "authority_card_cache" / f"{cache_key}.json"
    if config.authority_card_cache_enabled and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return AuthorityCard(**payload)
        except Exception:
            pass

    text = normalize_whitespace(chunk.chunk_text)
    taxpayer_position, taxpayer_span = _extract_taxpayer_position(text)
    authority_holding, authority_span = _extract_authority_holding(text, chunk.source_type)
    court_holding, court_span = _extract_court_holding(text, chunk.source_type)
    outcome, result_for_taxpayer = _infer_outcome(text, authority_holding, court_holding)
    document_type = _authority_document_type(chunk)
    cited_provisions = _extract_cited_provisions(chunk, text)
    facts = _extract_fact_snippets(text, chunk)
    issues = _dedupe([chunk.subject, *chunk.legal_provisions[:4]])
    temporal_status = _temporal_status(chunk, target_date)
    source_spans = {
        "chunk_text": {
            "chunk_id": chunk.chunk_id,
            "start": 0,
            "end": len(chunk.chunk_text),
        }
    }
    if taxpayer_span:
        source_spans["taxpayer_position"] = {"chunk_id": chunk.chunk_id, **taxpayer_span}
    if authority_span:
        source_spans["authority_holding"] = {"chunk_id": chunk.chunk_id, **authority_span}
    if court_span:
        source_spans["court_holding"] = {"chunk_id": chunk.chunk_id, **court_span}

    card = AuthorityCard(
        document_id=chunk.document_id,
        signature=chunk.signature or "",
        document_type=document_type,
        authority=chunk.authority or _authority_from_type(document_type),
        court=chunk.authority or ("Naczelny Sąd Administracyjny" if document_type == "nsa_judgment" else ""),
        date=(chunk.published_date or chunk.legal_state_date or "")[:10],
        tax=_infer_card_tax(chunk),
        target_law_period={"from": None, "to": None},
        facts=facts,
        issues=issues,
        cited_provisions=cited_provisions,
        taxpayer_position=taxpayer_position,
        authority_holding=authority_holding,
        court_holding=court_holding,
        outcome=outcome,
        result_for_taxpayer=result_for_taxpayer,
        legal_reasoning_summary=_extract_reasoning_summary(text, authority_holding or court_holding),
        distinguishing_facts=_extract_distinguishing_facts(text),
        temporal_status=temporal_status,
        source_spans=source_spans,
        source_chunk_id=chunk.chunk_id,
        source_canonical_id=chunk_canonical_source_id(chunk),
    )
    if config.authority_card_cache_enabled:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(to_jsonable(card), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return card


def _authority_card_cache_key(chunk: RagChunk, config: HybridAuthorityConfig) -> str:
    document_hash = hashlib.sha256(
        "\n".join([chunk.document_id, chunk.signature or "", chunk.subject, chunk.chunk_text]).encode("utf-8")
    ).hexdigest()
    raw = "|".join(
        [
            document_hash,
            config.extractor_model,
            AUTHORITY_CARD_EXTRACTOR_PROMPT_VERSION,
            config.authority_card_schema_version,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _authority_document_type(chunk: RagChunk) -> str:
    source_type = str(chunk.source_type or "").lower()
    subtype = str(chunk.source_subtype or "").lower()
    if source_type == "interpretation":
        if "general" in subtype:
            return "general_interpretation"
        return "individual_interpretation"
    if source_type == "judgment":
        if "wsa" in subtype or re.search(r"\bWSA\b", chunk.subject or ""):
            return "wsa_judgment"
        if "nsa" in subtype or "Naczelny Sąd Administracyjny" in (chunk.authority or ""):
            return "nsa_judgment"
        return "judgment"
    return source_type or "other"


def _authority_from_type(document_type: str) -> str:
    if document_type == "individual_interpretation":
        return "Dyrektor Krajowej Informacji Skarbowej"
    if document_type == "general_interpretation":
        return "Minister Finansów"
    if document_type == "nsa_judgment":
        return "Naczelny Sąd Administracyjny"
    if document_type == "wsa_judgment":
        return "Wojewódzki Sąd Administracyjny"
    return ""


def _infer_card_tax(chunk: RagChunk) -> str:
    if chunk.legal_provisions:
        for provision in chunk.legal_provisions:
            match = re.search(r"\[([A-ZŚĆŹŻÓŁŃ]+)\]", provision)
            if match:
                value = match.group(1)
                return "AKCYZA" if value in {"AKC"} else value
    return ""


def _extract_taxpayer_position(text: str) -> tuple[Optional[str], Optional[dict[str, int]]]:
    return _extract_first_span(
        text,
        [
            r"(Państwa stanowisko w sprawie.*?)(?=Ocena stanowiska|Uzasadnienie interpretacji|Organ|W świetle|$)",
            r"(Stanowisko Wnioskodawcy.*?)(?=Ocena stanowiska|Uzasadnienie|Organ|$)",
            r"(Zdaniem Wnioskodawcy.*?)(?=Ocena stanowiska|Uzasadnienie|Organ|$)",
        ],
    )


def _extract_authority_holding(text: str, source_type: str) -> tuple[Optional[str], Optional[dict[str, int]]]:
    if source_type != "interpretation":
        return None, None
    return _extract_first_span(
        text,
        [
            r"(Ocena stanowiska.*?)(?=Uzasadnienie interpretacji|Pouczenie|Funkcję ochronną|$)",
            r"(stanowisko.*?jest prawidłowe.*?)(?=Uzasadnienie|Pouczenie|$)",
            r"(stanowisko.*?jest nieprawidłowe.*?)(?=Uzasadnienie|Pouczenie|$)",
            r"(Organ stwierdza.*?)(?=Pouczenie|$)",
        ],
    )


def _extract_court_holding(text: str, source_type: str) -> tuple[Optional[str], Optional[dict[str, int]]]:
    if source_type != "judgment":
        return None, None
    return _extract_first_span(
        text,
        [
            r"((?:Naczelny Sąd Administracyjny|NSA|Sąd).*?(?:oddala|uchyla|uwzględnia|zasądza|orzeka).*?)(?=Uzasadnienie|Z tych względów|$)",
            r"(Z tych względów.*?)(?=$)",
        ],
    )


def _extract_first_span(text: str, patterns: list[str]) -> tuple[Optional[str], Optional[dict[str, int]]]:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        value = normalize_whitespace(match.group(1))[:900]
        if value:
            return value, {"start": match.start(1), "end": match.end(1)}
    return None, None


def _infer_outcome(
    text: str,
    authority_holding: Optional[str],
    court_holding: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    lowered = normalize_whitespace(" ".join(part for part in [authority_holding or "", court_holding or "", text[:1200]] if part)).lower()
    if re.search(r"stanowisko.*jest nieprawidłowe|uznaje.*za nieprawidłowe", lowered):
        return "taxpayer_position_rejected", "unfavorable"
    if re.search(r"stanowisko.*jest prawidłowe|uznaje.*za prawidłowe", lowered):
        return "taxpayer_position_accepted", "favorable"
    if re.search(r"\buchyla\b", lowered):
        return "interpretation_or_decision_quashed", "favorable_or_mixed"
    if re.search(r"\boddala\b", lowered):
        return "complaint_dismissed", "unfavorable_or_mixed"
    return None, None


def _extract_cited_provisions(chunk: RagChunk, text: str) -> tuple[str, ...]:
    provisions = list(chunk.legal_provisions)
    for match in re.finditer(r"\bart\.\s*\d+[a-z]?(?:\s*ust\.\s*\d+[a-z]?)?(?:\s*pkt\s*\d+[a-z]?)?", text, re.IGNORECASE):
        provisions.append(match.group(0))
    return _dedupe(provisions)[:16]


def _extract_fact_snippets(text: str, chunk: RagChunk) -> tuple[str, ...]:
    snippets: list[str] = []
    for marker in ("stan faktyczny", "zdarzenie przyszłe", "opis sprawy", "wnioskodawca wskazał"):
        match = re.search(marker + r".{0,650}", text, re.IGNORECASE | re.DOTALL)
        if match:
            snippets.append(normalize_whitespace(match.group(0))[:500])
    if not snippets and chunk.subject:
        snippets.append(chunk.subject)
    return _dedupe(snippets)[:4]


def _extract_reasoning_summary(text: str, holding: Optional[str]) -> Optional[str]:
    if holding:
        return holding[:500]
    role = classify_chunk_evidence_role(
        RagChunk(
            chunk_id="tmp",
            document_id="tmp",
            chunk_index=0,
            score=0,
            chunk_text=text,
            subject="",
            signature=None,
            published_date=None,
            source_url=None,
            category=None,
        )
    )
    if role in {"authority_assessment", "operative_conclusion", "reasoning"}:
        return text[:500]
    return None


def _extract_distinguishing_facts(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    facts: list[str] = []
    for pattern, label in (
        (r"kredyt.*zbywan", "credit secured on or used for the sold property"),
        (r"kredyt.*now\w* nieruchomo", "credit for a new property"),
        (r"odsetk", "interest payment"),
        (r"usług\w* doradcz|uslug\w* doradcz", "advisory services"),
        (r"fundator", "founder role"),
        (r"beneficjent", "beneficiary role"),
        (r"samochód prywatny|samochod prywatny", "private car"),
        (r"działalnoś\w* gospodarcz|dzialalnos\w* gospodarcz", "business use"),
    ):
        if re.search(pattern, lowered):
            facts.append(label)
    return _dedupe(facts)


def _temporal_status(chunk: RagChunk, target_date: str) -> str:
    source_date = _parse_date(chunk.legal_state_date or chunk.published_date or "")
    target = _parse_date(target_date)
    if source_date is None or target is None:
        return "uncertain"
    if source_date.year < target.year - 7:
        return "historical"
    return "current"


def _parse_date(value: str) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def prefilter_authority_card(
    card: AuthorityCard,
    issue: IssueNode,
    intent: LegalResearchIntent,
    fact_graph: FactGraph,
    primary_result: PrimaryLaneResult,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    status = "kept"
    requested = set(intent.requested_document_types)
    if card.document_type not in requested and card.document_type not in {"judgment", "other"}:
        reasons.append("document_type_not_requested")
    if issue.tax and card.tax and issue.tax != "TAX" and card.tax.upper() != issue.tax.upper():
        reasons.append("tax_domain_mismatch")
        status = "penalized"
    if card.temporal_status == "historical":
        reasons.append("historical_document")
        status = "penalized"
    if _is_pre_material_amendment_authority(card, issue):
        reasons.append("pre_material_amendment")
        status = "penalized"
    if not (card.authority_holding or card.court_holding):
        reasons.append("holding_not_found")
        status = "penalized" if status == "kept" else status
    if _is_wrong_neighbor(card, issue, fact_graph):
        reasons.append("wrong_neighbor_candidate")
        status = "penalized"
    return status, reasons


def score_authority_card(
    card: AuthorityCard,
    *,
    issue: IssueNode,
    fact_graph: FactGraph,
    primary_result: PrimaryLaneResult,
    family_score: float,
    filter_status: str,
) -> RerankScore:
    dimensions: dict[str, float] = {}
    positives: list[str] = []
    negatives: list[str] = []

    tax_match = 1.0 if not issue.tax or issue.tax == "TAX" or not card.tax or card.tax.upper() == issue.tax.upper() else 0.0
    dimensions["tax_match"] = tax_match
    (positives if tax_match else negatives).append("same tax domain" if tax_match else "different tax domain")

    issue_terms = set(_terms(f"{issue.label} {issue.mechanism} {issue.query}"))
    card_terms = set(_terms(" ".join([*card.issues, *card.facts, card.legal_reasoning_summary or ""])))
    overlap = len(issue_terms & card_terms) / max(len(issue_terms), 1)
    dimensions["issue_match"] = min(1.0, overlap * 2.5)
    if dimensions["issue_match"] >= 0.35:
        positives.append("same issue vocabulary")

    fact_terms = set(_terms(" ".join([*fact_graph.transactions, *fact_graph.payments, *fact_graph.roles, *fact_graph.relationships])))
    material_overlap = len(fact_terms & card_terms) / max(len(fact_terms), 1) if fact_terms else 0.4
    dimensions["material_fact_match"] = min(1.0, material_overlap * 2.0)
    if dimensions["material_fact_match"] >= 0.45:
        positives.append("same material facts")

    provision_terms = set(_article_terms(_provision_terms_from_chunks(primary_result.controlling_provisions)))
    card_provisions = set(_article_terms(card.cited_provisions))
    provision_match = len(provision_terms & card_provisions) / max(len(provision_terms), 1) if provision_terms else 0.0
    if issue.preferred_targets and not provision_terms:
        preferred = {article for _domain, article in issue.preferred_targets}
        provision_match = len(preferred & card_provisions) / max(len(preferred), 1)
    current_special_rule_match = 1.0 if _card_cites_current_special_rule(card, issue) else 0.0
    if current_special_rule_match:
        provision_match = max(provision_match, 1.0)
        positives.append("cites current special rule")
    dimensions["provision_match"] = min(1.0, provision_match)
    dimensions["current_special_rule_match"] = current_special_rule_match
    if provision_match:
        positives.append("same provision")

    dimensions["taxpayer_role_match"] = 1.0 if not fact_graph.roles or any(role in " ".join(card.facts).lower() for role in fact_graph.roles) else 0.4
    dimensions["transaction_match"] = 1.0 if not fact_graph.transactions or any(item in " ".join(card.facts).lower() for item in fact_graph.transactions) else 0.35
    dimensions["payment_type_match"] = 1.0 if not fact_graph.payments or any(item in " ".join(card.facts).lower() for item in fact_graph.payments) else 0.35
    dimensions["mechanism_match"] = 1.0 if issue.mechanism and issue.mechanism in " ".join(card.issues).lower() else dimensions["issue_match"]
    dimensions["temporal_match"] = 1.0 if card.temporal_status == "current" else (0.45 if card.temporal_status == "uncertain" else 0.15)
    if card.temporal_status == "historical":
        negatives.append("historical document")
    if _is_pre_material_amendment_authority(card, issue):
        negatives.append("pre-material-amendment authority")
    dimensions["exception_match"] = 0.5
    dimensions["holding_relevance"] = 1.0 if (card.authority_holding or card.court_holding) else 0.2
    if dimensions["holding_relevance"] >= 1.0:
        positives.append("holding extracted")
    else:
        negatives.append("holding not extracted")
    dimensions["document_authority_weight"] = _document_authority_weight(card)
    wrong_neighbor_penalty = 0.0
    if _is_wrong_neighbor(card, issue, fact_graph):
        wrong_neighbor_penalty = 0.35
        negatives.append("wrong neighbor")
    if filter_status == "penalized":
        wrong_neighbor_penalty = max(wrong_neighbor_penalty, 0.12)
    dimensions["wrong_neighbor_penalty"] = wrong_neighbor_penalty

    weighted = (
        family_score * 5.0
        + dimensions["tax_match"] * 0.13
        + dimensions["issue_match"] * 0.17
        + dimensions["material_fact_match"] * 0.16
        + dimensions["provision_match"] * 0.20
        + dimensions["payment_type_match"] * 0.08
        + dimensions["transaction_match"] * 0.07
        + dimensions["temporal_match"] * 0.08
        + dimensions["holding_relevance"] * 0.08
        + dimensions["document_authority_weight"] * 0.05
        + dimensions["current_special_rule_match"] * 0.08
        - wrong_neighbor_penalty
    )
    score = round(max(0.0, min(1.0, weighted)), 4)
    return RerankScore(score=score, dimensions=dimensions, positive_reasons=_dedupe(positives), negative_reasons=_dedupe(negatives))


def _terms(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-ząćęłńóśźż0-9]{3,}", normalize_whitespace(text).lower())
        if token not in {"oraz", "jest", "dla", "przez", "ust", "art", "podatek", "podatku", "ustawa"}
    ]


def _article_terms(values: Iterable[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        article = extract_article_key_from_text(str(value))
        if article:
            terms.append(article)
    return terms


def _document_authority_weight(card: AuthorityCard) -> float:
    if card.document_type == "nsa_judgment":
        return 1.0
    if card.document_type == "wsa_judgment":
        return 0.78
    if card.document_type == "general_interpretation":
        return 0.72
    if card.document_type == "individual_interpretation":
        return 0.55
    return 0.3


def _is_wrong_neighbor(card: AuthorityCard, issue: IssueNode, fact_graph: FactGraph) -> bool:
    text = normalize_whitespace(" ".join([*card.facts, *card.issues, card.legal_reasoning_summary or ""])).lower()
    if issue.contrast == "credit_on_sold_property_vs_credit_on_new_property":
        wants_sold = re.search(r"zbywan|sprzedawan", " ".join(fact_graph.known_facts).lower() + " " + issue.query.lower())
        mentions_new = re.search(r"now\w* nieruchomo|nabywan\w* lokal", text)
        mentions_sold = re.search(r"zbywan|sprzedawan", text)
        return bool(wants_sold and mentions_new and not mentions_sold)
    if issue.contrast == "interest_vs_advisory_or_management_services":
        return bool(re.search(r"usług\w* doradcz|uslug\w* doradcz|zarządz|zarzadz", text) and not re.search(r"odset", text))
    if issue.contrast == "services_vs_interest_or_royalties":
        return bool(re.search(r"odset|licencyjn|royalt", text) and not re.search(r"doradcz|zarządz|zarzadz|usług|uslug", text))
    if issue.contrast == "founder_vs_beneficiary_benefit":
        asks_founder = "founder" in fact_graph.roles
        asks_beneficiary = "beneficiary" in fact_graph.roles
        return bool((asks_founder and "beneficjent" in text and "fundator" not in text) or (asks_beneficiary and "fundator" in text and "beneficjent" not in text))
    if issue.contrast == "private_car_vs_business_car":
        asks_private = "prywat" in issue.query.lower()
        return bool(asks_private and "firmow" in text and "prywat" not in text)
    if issue.contrast == "pre_cit_vs_post_cit_limited_partnership":
        return bool(re.search(r"przed.*objęci|przed.*objeci", text) and re.search(r"po.*objęci|po.*objeci", issue.query.lower()))
    return False


def _card_cites_current_special_rule(card: AuthorityCard, issue: IssueNode) -> bool:
    if issue.contrast != "credit_on_sold_property_vs_credit_on_new_property":
        return False
    provisions_text = " ".join(card.cited_provisions).lower()
    return bool(
        re.search(r"art\.?\s*21[^.;,\n]{0,80}ust\.?\s*30a", provisions_text)
        or re.search(r"art\.\s*21-ust\.\s*30a", provisions_text)
    )


def _is_pre_material_amendment_authority(card: AuthorityCard, issue: IssueNode) -> bool:
    if issue.contrast != "credit_on_sold_property_vs_credit_on_new_property":
        return False
    parsed = _parse_date(card.date)
    if parsed is None:
        return False
    return parsed < date(2022, 1, 1) and not _card_cites_current_special_rule(card, issue)


def _authority_effective_temporal_status(item: ScoredAuthority) -> str:
    if item.card.temporal_status == "historical" or "pre_material_amendment" in item.filter_reasons:
        return "historical"
    return item.card.temporal_status


def _required_primary_bundle_for_issue(issue: IssueNode) -> tuple[str, ...]:
    if issue.contrast == "credit_on_sold_property_vs_credit_on_new_property":
        return (
            "pit_art_21_ust_25_pkt_2",
            "pit_art_21_ust_30",
            "pit_art_21_ust_30a",
        )
    return ()


def _missing_required_primary_bundle(
    issue: IssueNode,
    primary: Optional[PrimaryLaneResult],
) -> tuple[str, ...]:
    required = set(_required_primary_bundle_for_issue(issue))
    if not required:
        return ()
    present: set[str] = set()
    for chunk in (
        *((primary.controlling_provisions if primary else ()) or ()),
        *((primary.dependency_provisions if primary else ()) or ()),
        *((primary.exception_provisions if primary else ()) or ()),
    ):
        present.update(_primary_bundle_keys_from_chunk(chunk))
    return tuple(sorted(required - present))


def _primary_bundle_keys_from_chunk(chunk: RagChunk) -> set[str]:
    text = " ".join(
        [
            chunk.subject or "",
            chunk.signature or "",
            chunk.chunk_text or "",
            *chunk.legal_provisions,
        ]
    ).lower()
    keys: set[str] = set()
    if re.search(r"art\.?\s*21[^.;,\n]{0,100}ust\.?\s*25[^.;,\n]{0,60}pkt\s*2|art\.\s*21-ust\.\s*25-pkt\s*2", text):
        keys.add("pit_art_21_ust_25_pkt_2")
    if re.search(r"art\.?\s*21[^.;,\n]{0,100}ust\.?\s*30a|art\.\s*21-ust\.\s*30a", text):
        keys.add("pit_art_21_ust_30a")
    if re.search(r"art\.?\s*21[^.;,\n]{0,100}ust\.?\s*30(?!a)|art\.\s*21-ust\.\s*30(?!a)", text):
        keys.add("pit_art_21_ust_30")
    return keys


def build_authority_lines(
    scored_authorities: list[ScoredAuthority],
    *,
    config: HybridAuthorityConfig,
) -> tuple[AuthorityLine, ...]:
    by_issue: dict[str, list[ScoredAuthority]] = {}
    for item in scored_authorities:
        if item.rerank.score < config.min_authority_score and item.filter_status != "kept":
            continue
        by_issue.setdefault(item.issue_id, []).append(item)

    lines: list[AuthorityLine] = []
    for issue_id, items in by_issue.items():
        current = [item for item in items if _authority_effective_temporal_status(item) != "historical"]
        historical = [item for item in items if _authority_effective_temporal_status(item) == "historical"]
        supporting = [
            item for item in current
            if item.card.result_for_taxpayer not in {"unfavorable", "unfavorable_or_mixed"}
        ]
        contrary = [
            item for item in current
            if item.card.result_for_taxpayer in {"unfavorable", "unfavorable_or_mixed"}
        ]
        dominance = "unknown"
        if len(supporting) >= 3 and not contrary:
            dominance = "dominant"
        elif supporting and contrary:
            dominance = "mixed"
        elif len(supporting) == 1:
            dominance = "isolated"
        holding_summary = _summarize_line(supporting, contrary)
        lines.append(
            AuthorityLine(
                issue_id=issue_id,
                position_id=f"{issue_id}:main",
                holding_summary=holding_summary,
                supporting_documents=tuple(item.card for item in supporting[: config.authority_selected_limit_per_issue]),
                contrary_documents=tuple(item.card for item in contrary[: config.contrary_limit_per_issue]),
                historical_documents=tuple(item.card for item in historical[: config.historical_limit_per_issue]),
                dominance=dominance,
                distinguishing_facts=_dedupe(
                    fact
                    for item in [*supporting, *contrary]
                    for fact in item.card.distinguishing_facts
                ),
                temporal_notes=tuple(
                    f"{item.card.signature or item.card.document_id}: {item.card.temporal_status}"
                    for item in historical[: config.historical_limit_per_issue]
                ),
            )
        )
    return tuple(lines)


def _summarize_line(supporting: list[ScoredAuthority], contrary: list[ScoredAuthority]) -> str:
    if supporting and contrary:
        return "Wyniki authority są mieszane; bundle zawiera dokumenty wspierające i przeciwne."
    if supporting:
        best = supporting[0].card.authority_holding or supporting[0].card.court_holding or supporting[0].card.legal_reasoning_summary
        return (best or "Znaleziono dokumenty wspierające tę oś.")[:500]
    if contrary:
        best = contrary[0].card.authority_holding or contrary[0].card.court_holding or contrary[0].card.legal_reasoning_summary
        return (best or "Znaleziono dokumenty przeciwne dla tej osi.")[:500]
    return "Nie znaleziono dostatecznie podobnej linii authorities."


def build_evidence_bundles(
    issues: tuple[IssueNode, ...],
    primary_results: tuple[PrimaryLaneResult, ...],
    scored_authorities: list[ScoredAuthority],
    *,
    config: HybridAuthorityConfig,
) -> tuple[EvidenceBundle, ...]:
    scored_by_issue: dict[str, list[ScoredAuthority]] = {}
    for item in scored_authorities:
        scored_by_issue.setdefault(item.issue_id, []).append(item)
    primary_by_issue = {item.issue_id: item for item in primary_results}
    bundles: list[EvidenceBundle] = []
    for issue in issues:
        primary = primary_by_issue.get(issue.issue_id)
        issue_scores = sorted(scored_by_issue.get(issue.issue_id, []), key=lambda item: -item.rerank.score)
        missing_primary_bundle = _missing_required_primary_bundle(issue, primary)
        current = [
            item for item in issue_scores
            if _authority_effective_temporal_status(item) != "historical" and item.rerank.score >= config.min_authority_score
        ]
        supporting = [
            item for item in current
            if item.card.result_for_taxpayer not in {"unfavorable", "unfavorable_or_mixed"}
        ][: config.authority_selected_limit_per_issue]
        contrary = [
            item for item in current
            if item.card.result_for_taxpayer in {"unfavorable", "unfavorable_or_mixed"}
        ][: config.contrary_limit_per_issue]
        historical = [
            item for item in issue_scores
            if _authority_effective_temporal_status(item) == "historical"
        ][: config.historical_limit_per_issue]
        missing: list[str] = []
        if not primary or not primary.controlling_provisions:
            missing.append("controlling_primary_law")
        if missing_primary_bundle:
            missing.append("primary_bundle_incomplete:" + ",".join(missing_primary_bundle))
            supporting = []
            contrary = []
        if not supporting and not contrary:
            missing.append("supporting_authority")
        confidence = _bundle_confidence(primary, supporting, contrary)
        bundles.append(
            EvidenceBundle(
                issue_id=issue.issue_id,
                controlling_provisions=tuple(_chunk_source_ref(chunk) for chunk in (primary.controlling_provisions if primary else ())[:4]),
                dependency_provisions=tuple(_chunk_source_ref(chunk) for chunk in (primary.dependency_provisions if primary else ())[:4]),
                supporting_authorities=tuple(_authority_ref(item) for item in supporting),
                contrary_authorities=tuple(_authority_ref(item) for item in contrary),
                historical_authorities=tuple(_authority_ref(item) for item in historical),
                missing_source_requirements=tuple(missing),
                retrieval_confidence=confidence,
            )
        )
    return tuple(bundles)


def _bundle_confidence(
    primary: Optional[PrimaryLaneResult],
    supporting: list[ScoredAuthority],
    contrary: list[ScoredAuthority],
) -> float:
    score = 0.0
    if primary and primary.controlling_provisions:
        score += 0.45
    if supporting:
        score += min(0.4, 0.12 * len(supporting) + max(item.rerank.score for item in supporting) * 0.18)
    if contrary:
        score += 0.08
    return round(min(score, 1.0), 3)


def _authority_ref(item: ScoredAuthority) -> dict[str, Any]:
    card = item.card
    return {
        "document_id": card.document_id,
        "signature": card.signature,
        "document_type": card.document_type,
        "authority": card.authority or card.court,
        "date": card.date,
        "tax": card.tax,
        "score": item.rerank.score,
        "positive_reasons": list(item.rerank.positive_reasons),
        "negative_reasons": list(item.rerank.negative_reasons),
        "holding": card.authority_holding or card.court_holding,
        "outcome": card.outcome,
        "result_for_taxpayer": card.result_for_taxpayer,
        "temporal_status": _authority_effective_temporal_status(item),
        "applicable_law_period": card.target_law_period,
        "source_spans": card.source_spans,
    }


def _chunk_source_ref(chunk: RagChunk) -> dict[str, Any]:
    return {
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "signature": chunk.signature,
        "source_type": chunk.source_type,
        "source_subtype": chunk.source_subtype,
        "authority": chunk.authority,
        "publication": chunk.publication,
        "legal_state_date": chunk.legal_state_date,
        "legal_provisions": list(chunk.legal_provisions),
        "source_spans": {
            "chunk_text": {
                "chunk_id": chunk.chunk_id,
                "start": 0,
                "end": len(chunk.chunk_text),
            }
        },
    }


def bind_claims_to_evidence_bundles(
    claims: Iterable[LegalClaim],
    evidence_bundles: Iterable[EvidenceBundle | dict[str, Any]],
) -> list[LegalClaim]:
    bundles = [_bundle_to_dict(bundle) for bundle in evidence_bundles]
    if not bundles:
        return list(claims)
    bound: list[LegalClaim] = []
    for claim in claims:
        matching = _matching_bundle_for_claim(claim, bundles)
        if matching is None:
            bound.append(claim)
            continue
        bound.append(
            replace(
                claim,
                supporting_authorities=tuple(matching.get("supporting_authorities") or ()),
                contrary_authorities=tuple(matching.get("contrary_authorities") or ()),
                historical_authorities=tuple(matching.get("historical_authorities") or ()),
                authority_confidence=float(matching.get("retrieval_confidence") or 0.0),
            )
        )
    return bound


def _bundle_to_dict(bundle: EvidenceBundle | dict[str, Any]) -> dict[str, Any]:
    return to_jsonable(bundle) if not isinstance(bundle, dict) else bundle


def _matching_bundle_for_claim(claim: LegalClaim, bundles: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    claim_axis = claim.axis_id.lower()
    for bundle in bundles:
        issue_id = str(bundle.get("issue_id") or "").lower()
        if issue_id and (issue_id == claim_axis or issue_id in claim_axis or claim_axis in issue_id):
            return bundle
    claim_tax = claim.axis_id.split("_", 1)[0].upper()
    for bundle in bundles:
        if any(
            claim_tax and claim_tax in str(source.get("tax") or "").upper()
            for source in bundle.get("supporting_authorities") or []
        ):
            return bundle
    return bundles[0] if len(bundles) == 1 else None


def build_authority_evidence_context(result: Optional[HybridAuthorityResult]) -> str:
    if result is None:
        return ""
    lines = [
        "Eksperymentalny EvidenceBundle authority RAG:",
        f"- retrieval_mode: {result.retrieval_mode}",
        f"- answer_mode: {result.intent_profile.answer_mode}",
        f"- primary_law_weight: {result.intent_profile.primary_law_weight:.2f}",
        f"- authority_weight: {result.intent_profile.authority_weight:.2f}",
    ]
    if result.intent_profile.answer_mode == "authority_research":
        lines.append("- renderer emphasis: krótka podstawa normatywna, potem linie interpretacyjne/orzecznicze, dokumenty przeciwne i sygnatury.")
    elif result.intent_profile.answer_mode == "rule_first":
        lines.append("- renderer emphasis: odpowiedź i przepisy, potem krótka praktyka organów i sądów jeśli bundle ją zawiera.")
    else:
        lines.append("- renderer emphasis: odpowiedź, przepisy, praktyka organów, orzecznictwo, rozbieżności i dokumenty do dalszego researchu.")
    for bundle in result.evidence_bundles:
        lines.append(
            f"- issue={bundle.issue_id} confidence={bundle.retrieval_confidence:.2f} "
            f"primary={len(bundle.controlling_provisions)} supporting={len(bundle.supporting_authorities)} "
            f"contrary={len(bundle.contrary_authorities)} historical={len(bundle.historical_authorities)}"
        )
        if not bundle.supporting_authorities and not bundle.contrary_authorities:
            lines.append("  W przeszukanym zbiorze nie znaleziono dostatecznie podobnej interpretacji lub orzeczenia.")
        for source in [*bundle.supporting_authorities, *bundle.contrary_authorities, *bundle.historical_authorities]:
            lines.append(
                "  "
                f"{source.get('document_type')} {source.get('signature') or source.get('document_id')} "
                f"score={float(source.get('score') or 0):.2f} temporal={source.get('temporal_status')}"
            )
    return "\n".join(lines)


def write_hybrid_trace_artifacts(
    result: HybridAuthorityResult,
    *,
    claims: Optional[list[dict[str, Any]]] = None,
    renderer_payload: Optional[dict[str, Any]] = None,
    final_answer: str = "",
    validation: Optional[dict[str, Any]] = None,
) -> Path:
    run_dir = get_hybrid_authority_config().artifact_root / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "summary.json": {
            "request_id": result.request_id,
            "run_id": result.run_id,
            "retrieval_mode": result.retrieval_mode,
            "clarifier_enabled": result.clarifier_enabled,
            "query": result.query,
            "retrieval_query": result.retrieval_query,
            "selected_chunk_count": len(result.selected_chunks),
            "timings": result.timings,
        },
        "intent.json": result.intent_profile,
        "fact_graph.json": result.fact_graph,
        "issue_graph.json": result.issue_graph,
        "primary_retrieval.json": result.primary_results,
        "authority_retrieval.json": {
            "authority_queries": result.authority_queries,
            "candidate_documents": result.candidate_documents,
            "filtered_documents": result.filtered_documents,
        },
        "reranking.json": result.reranked_documents,
        "authority_cards.json": result.authority_cards,
        "evidence_bundles.json": result.evidence_bundles,
        "claims.json": claims or [],
        "renderer_payload.json": renderer_payload or {},
        "validation.json": validation or {},
    }
    for filename, payload in payloads.items():
        (run_dir / filename).write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "final_answer.txt").write_text(final_answer or "", encoding="utf-8")
    return run_dir


def baseline_retrieval_for_comparison(
    query: str,
    *,
    limit: Optional[int] = None,
    include_interpretations: bool = True,
    include_judgments: Optional[bool] = None,
) -> tuple[RagChunk, ...]:
    chunks = search_chat_chunks(
        query,
        limit=limit,
        include_interpretations=include_interpretations,
        include_judgments=include_judgments,
    )
    return tuple(select_diverse_chunks(chunks))


def build_structured_claim_inputs(query: str, chunks: list[RagChunk]) -> dict[str, Any]:
    target_date = _target_date_from_query(query)
    legal_rules = prioritize_legal_rules_for_query(
        extract_legal_rules_from_statute_chunks(chunks),
        query,
    )
    legal_rules = filter_legal_rules_for_target_date(legal_rules, target_date)
    return {
        "target_date": target_date,
        "legal_rules": [legal_rule_to_dict(rule) for rule in legal_rules],
        "source_plan": legal_source_plan_to_dict(build_legal_source_plan(query), chunks),
    }


def _chunk_is_temporally_current(chunk: RagChunk, target_date: str) -> bool:
    status = _temporal_status(chunk, target_date)
    return status != "historical"


def _dedupe_chunks(chunks: Iterable[RagChunk]) -> tuple[RagChunk, ...]:
    deduped: list[RagChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        key = chunk_canonical_source_id(chunk)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return tuple(deduped)


def _dedupe_scored_authorities(items: list[ScoredAuthority]) -> list[ScoredAuthority]:
    best: dict[tuple[str, str], ScoredAuthority] = {}
    for item in items:
        key = (item.issue_id, item.card.source_canonical_id or item.card.document_id)
        current = best.get(key)
        if current is None or item.rerank.score > current.rerank.score:
            best[key] = item
    return list(best.values())


def _filtered_trace(items: list[ScoredAuthority]) -> list[dict[str, Any]]:
    return [
        {
            "issue_id": item.issue_id,
            "document_id": item.card.document_id,
            "signature": item.card.signature,
            "document_type": item.card.document_type,
            "filter_status": item.filter_status,
            "filter_reasons": list(item.filter_reasons),
        }
        for item in items
    ]


def _reranked_trace(items: list[ScoredAuthority]) -> list[dict[str, Any]]:
    return [
        {
            "issue_id": item.issue_id,
            "rank": index,
            "document_id": item.card.document_id,
            "signature": item.card.signature,
            "document_type": item.card.document_type,
            "score": item.rerank.score,
            "dimensions": item.rerank.dimensions,
            "positive_reasons": list(item.rerank.positive_reasons),
            "negative_reasons": list(item.rerank.negative_reasons),
        }
        for index, item in enumerate(sorted(items, key=lambda value: (value.issue_id, -value.rerank.score)), start=1)
    ]


def _select_hybrid_chunks(
    primary_results: tuple[PrimaryLaneResult, ...],
    scored_authorities: list[ScoredAuthority],
    evidence_bundles: tuple[EvidenceBundle, ...],
) -> list[RagChunk]:
    selected: list[RagChunk] = []
    for primary in primary_results:
        selected.extend(primary.controlling_provisions[:4])
    selected_authority_ids = {
        str(source.get("document_id") or "")
        for bundle in evidence_bundles
        for source in [*bundle.supporting_authorities, *bundle.contrary_authorities, *bundle.historical_authorities]
    }
    for item in sorted(scored_authorities, key=lambda value: -value.rerank.score):
        if item.card.document_id in selected_authority_ids:
            selected.append(item.chunk)
    return list(_dedupe_chunks(selected))


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(to_jsonable(item) for item in value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value
