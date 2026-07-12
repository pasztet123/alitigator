"""Issue-scoped, fail-closed secondary-authority retrieval for controlled cases.

The controlled answer pipeline already knows the legal issues and controlling
provisions.  This module deliberately does not use a broad ``housing relief``
pool: each authority is retrieved, scored and bound against one issue and one
or more claims.  A document that cannot supply a complete holding from a
recognised section is evidence-free and therefore never rendered.
"""
from __future__ import annotations

import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from app.rag import RagChunk, RagDocumentContext, fetch_document_contexts, get_rag_config, search_chunks


Search = Callable[..., list[RagChunk]]
ContextFetcher = Callable[..., list[RagDocumentContext]]
_API_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_JUDGMENT_CORPUS = _API_DIR / "data" / "processed" / "cbosa_nsa_fsk_judgments.jsonl"


@dataclass(frozen=True)
class ControlledAuthorityIssue:
    issue_id: str
    label: str
    query_suffix: str
    required_provision: str
    claim_ids: tuple[str, ...]
    transaction_type: str
    event_type: str
    required_terms: tuple[str, ...]
    distinguishing_fact: str


HOUSING_AUTHORITY_ISSUES: tuple[ControlledAuthorityIssue, ...] = (
    ControlledAuthorityIssue(
        issue_id="credit_on_sold_property",
        label="spłata kredytu dotyczącego sprzedanego mieszkania",
        query_suffix=(
            "art. 21 ust. 30a ustawy PIT spłata kredytu zaciągniętego na zakup "
            "sprzedanej nieruchomości ze środków ze sprzedaży"
        ),
        required_provision="21.30a",
        claim_ids=("claim_credit_scope",),
        transaction_type="credit_repayment",
        event_type="repayment_from_sale_proceeds",
        required_terms=("kredyt", "spłat", "sprzed"),
        distinguishing_fact="czy kredyt został zaciągnięty na sprzedane mieszkanie przed sprzedażą",
    ),
    ControlledAuthorityIssue(
        issue_id="developer_ownership_deadline",
        label="nabycie własności od dewelopera przed terminem ulgi",
        query_suffix=(
            "art. 21 ust. 25a ustawy PIT deweloper nabycie własności przed upływem "
            "trzech lat od końca roku sprzedaży"
        ),
        required_provision="21.25a",
        claim_ids=("claim_developer_deadline",),
        transaction_type="developer_acquisition",
        event_type="ownership_transfer_deadline",
        required_terms=("deweloper", "własno", "naby"),
        distinguishing_fact="data aktu przenoszącego własność względem ustawowego terminu",
    ),
    ControlledAuthorityIssue(
        issue_id="exemption_formula",
        label="proporcja zwolnienia dochodu z wydatkami mieszkaniowymi",
        query_suffix=(
            "art. 21 ust. 1 pkt 131 ustawy PIT dochód zwolniony iloczyn dochodu "
            "i udział wydatków mieszkaniowych w przychodzie"
        ),
        required_provision="21.1.131",
        claim_ids=("claim_formula", "claim_expense_not_income"),
        transaction_type="housing_relief_calculation",
        event_type="exemption_formula",
        required_terms=("dochód", "wydatk", "przych"),
        distinguishing_fact="relacja kwalifikowanych wydatków do przychodu ze sprzedaży",
    ),
    ControlledAuthorityIssue(
        issue_id="five_year_rule",
        label="pięcioletni termin dla sprzedaży nieruchomości",
        query_suffix=(
            "art. 10 ust. 1 pkt 8 ustawy PIT odpłatne zbycie nieruchomości "
            "przed upływem pięciu lat od końca roku nabycia"
        ),
        required_provision="10.1.8",
        claim_ids=("claim_sale_tax_regime",),
        transaction_type="real_estate_sale",
        event_type="five_year_sale_rule",
        required_terms=("sprzeda", "nieruchomo", "pięciu"),
        distinguishing_fact="rok nabycia i rok odpłatnego zbycia nieruchomości",
    ),
)


def retrieve_housing_authorities(
    query: str,
    *,
    search: Search = search_chunks,
    context_fetcher: ContextFetcher = fetch_document_contexts,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Retrieve only high-relevance cards and preserve complete selection trace."""
    cards: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []
    errors: list[str] = []
    candidate_counts = {"interpretation": 0, "judgment": 0}
    filtered_counts = {"interpretation": 0, "judgment": 0}
    selected_counts = {"interpretation": 0, "judgment": 0}
    seen: set[tuple[str, str, str]] = set()

    jobs: list[tuple[ControlledAuthorityIssue, str, int, str]] = []
    for issue in HOUSING_AUTHORITY_ISSUES:
        authority_query = _issue_query(query, issue)
        for source_type, limit in (("interpretation", 4), ("judgment", 3)):
            queries.append(
                {
                    "issue_id": issue.issue_id,
                    "issue_label": issue.label,
                    "source_type": source_type,
                    "query": authority_query,
                    "required_provision": issue.required_provision,
                    "transaction_type": issue.transaction_type,
                    "event_type": issue.event_type,
                }
            )
            jobs.append((issue, source_type, limit, authority_query))

    # Search each issue-source pair independently.  Parallel reads keep the
    # stricter retrieval design from multiplying user-visible latency.
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as executor:
        futures = {
            executor.submit(search, authority_query, limit=limit, source_types={source_type}): (issue, source_type)
            for issue, source_type, limit, authority_query in jobs
        }
        for future in as_completed(futures):
            issue, source_type = futures[future]
            try:
                candidates = future.result()
            except Exception as exc:  # Secondary research must not suppress primary law.
                errors.append(f"{issue.issue_id}:{source_type}:{type(exc).__name__}")
                continue
            candidate_counts[source_type] += len(candidates)
            for chunk in _hydrate_document_contexts(candidates, context_fetcher=context_fetcher):
                selection = _select_candidate(chunk, issue)
                if selection is None:
                    filtered_counts[source_type] += 1
                    continue
                key = (issue.issue_id, chunk.source_type, chunk.signature or chunk.document_id)
                if key in seen:
                    continue
                seen.add(key)
                cards.append(selection)
                selected_counts[source_type] += 1

    cards.sort(key=lambda item: (-float(item["authority_score"]), str(item["issue_id"]), str(item["label"])))
    judgment_audit = audit_judgment_corpus(
        candidate_count=candidate_counts["judgment"],
        selected_count=selected_counts["judgment"],
        filtered_count=filtered_counts["judgment"],
        errors=errors,
    )
    outcome: dict[str, object] = {
        "authority_lane_executed": True,
        "authority_queries_per_issue": True,
        "generic_housing_relief_pool_reused_for_all_claims": False,
        "authority_provision_match_scored": True,
        "authority_queries": queries,
        "candidate_counts": candidate_counts,
        "filtered_counts": filtered_counts,
        "selected_counts": selected_counts,
        "rendered_authority_cards": len(cards),
        "empty_high_quality_result_supported": True,
        "outcome": "no_high_quality_authorities" if not cards else "high_quality_authorities_found",
        "errors": errors,
        "judgment_lane": judgment_audit,
        "judgment_lane_executed": True,
        "judgment_candidate_count_recorded": True,
        "judgment_selected_count_recorded": True,
        "judgment_empty_result_reason_recorded": True,
        "judgment_corpus_count_recorded": True,
        "judgment_indexed_count_recorded": True,
    }
    return cards, outcome


def _hydrate_document_contexts(
    candidates: list[RagChunk],
    *,
    context_fetcher: ContextFetcher,
) -> list[RagChunk]:
    """Score holdings from the complete authority document, not a seed chunk."""
    if not candidates:
        return []
    unique_ids = list(dict.fromkeys(chunk.document_id for chunk in candidates if chunk.document_id))
    try:
        contexts = context_fetcher(unique_ids, seed_chunks=candidates)
    except Exception:
        contexts = []
    by_document = {context.document_id: context for context in contexts}
    hydrated: list[RagChunk] = []
    for chunk in candidates:
        context = by_document.get(chunk.document_id)
        if context is None or not context.text.strip():
            hydrated.append(chunk)
            continue
        hydrated.append(
            RagChunk(
                chunk_id=f"{context.document_id}:document_context",
                document_id=context.document_id,
                chunk_index=0,
                score=chunk.score,
                chunk_text=context.text,
                subject=context.subject,
                signature=context.signature,
                published_date=context.published_date,
                source_url=context.source_url,
                category=context.category,
                source=context.source,
                source_type=context.source_type,
                source_subtype=context.source_subtype,
                authority=context.authority,
                publication=context.publication,
                legal_state_date=context.legal_state_date,
                source_pages=context.source_pages,
                legal_provisions=context.legal_provisions,
            )
        )
    return hydrated


def _issue_query(query: str, issue: ControlledAuthorityIssue) -> str:
    return re.sub(r"\s+", " ", f"{issue.query_suffix}. Stan faktyczny: {query}").strip()


def _select_candidate(
    chunk: RagChunk,
    issue: ControlledAuthorityIssue,
) -> dict[str, object] | None:
    source_type = str(chunk.source_type or "").lower()
    if source_type not in {"interpretation", "judgment"}:
        return None
    text = _source_text(chunk)
    transaction_type, event_type = _classify_transaction(text)
    if _is_wrong_neighbor(transaction_type, event_type, issue):
        return None
    holding, holding_span, holding_section = _extract_complete_holding(chunk)
    if holding is None or holding_span is None or not holding_section:
        return None
    provision_score = _provision_match_score(text, issue.required_provision)
    issue_score = _issue_match_score(text, issue)
    material_fact_score = _material_fact_match_score(text, issue)
    holding_relevance = _holding_relevance_score(holding, issue)
    # An authority must earn every leg of the relevance test.  In particular a
    # document on a neighbouring mortgage topic cannot pass merely on keywords.
    if (
        provision_score < 1.0
        or issue_score < 0.34
        or material_fact_score < 0.34
        or holding_relevance < 0.5
    ):
        return None
    score = round(
        provision_score * 0.35
        + issue_score * 0.25
        + material_fact_score * 0.22
        + holding_relevance * 0.18,
        4,
    )
    binding_reason = (
        f"Holding dotyczy issue „{issue.label}”, wskazuje art. {issue.required_provision} "
        "i odpowiada materialnym faktom tego claimu."
    )
    return {
        "source_type": source_type,
        "label": (chunk.signature or chunk.subject or chunk.document_id).strip(),
        "date": (chunk.published_date or chunk.legal_state_date or "")[:10],
        "source_url": chunk.source_url or "",
        "issue_id": issue.issue_id,
        "issue_label": issue.label,
        "holding": holding,
        "holding_section": holding_section,
        "holding_source_span": {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "scope": "document_context" if chunk.chunk_id.endswith(":document_context") else "chunk",
            **holding_span,
        },
        "holding_complete_sentence": True,
        "outcome": _outcome_from_holding(holding),
        "transaction_type": transaction_type,
        "event_type": event_type,
        "provision_match_score": provision_score,
        "issue_match_score": issue_score,
        "material_fact_match_score": material_fact_score,
        "holding_relevance_score": holding_relevance,
        "authority_score": score,
        "similarity_reason": (
            f"Zgodny przepis i mechanizm: {issue.label}."
        ),
        "distinguishing_facts": issue.distinguishing_fact,
        "claim_bindings": [
            {"claim_id": claim_id, "score": score, "reason": binding_reason}
            for claim_id in issue.claim_ids
        ],
    }


def _source_text(chunk: RagChunk) -> str:
    return " ".join(
        part
        for part in (
            chunk.subject,
            chunk.signature or "",
            " ".join(chunk.legal_provisions),
            chunk.chunk_text,
        )
        if part
    )


def _provision_match_score(text: str, required: str) -> float:
    normalized = re.sub(r"\s+", " ", text.lower())
    patterns = {
        "21.30a": r"art\.?\s*21[^.\n]{0,90}ust\.?\s*30a|art\.\s*21-ust\.\s*30a",
        "21.25a": r"art\.?\s*21[^.\n]{0,90}ust\.?\s*25a|art\.\s*21-ust\.\s*25a",
        "21.1.131": r"art\.?\s*21[^.\n]{0,90}ust\.?\s*1[^.\n]{0,45}pkt\s*131|art\.\s*21-ust\.\s*1-pkt\.\s*131",
        "10.1.8": r"art\.?\s*10[^.\n]{0,90}ust\.?\s*1[^.\n]{0,45}pkt\s*8|art\.\s*10-ust\.\s*1-pkt\.\s*8",
    }
    return 1.0 if re.search(patterns[required], normalized, re.IGNORECASE) else 0.0


def _issue_match_score(text: str, issue: ControlledAuthorityIssue) -> float:
    lowered = text.lower()
    hits = sum(1 for term in issue.required_terms if term in lowered)
    return round(hits / len(issue.required_terms), 4)


def _material_fact_match_score(text: str, issue: ControlledAuthorityIssue) -> float:
    lowered = text.lower()
    if issue.event_type == "repayment_from_sale_proceeds":
        hits = ["kredyt" in lowered, bool(re.search(r"spłat|spłaci", lowered)), bool(re.search(r"zbywan|sprzed", lowered))]
    elif issue.event_type == "ownership_transfer_deadline":
        hits = ["deweloper" in lowered, "własno" in lowered, bool(re.search(r"termin|trzech lat", lowered))]
    elif issue.event_type == "exemption_formula":
        hits = ["dochód" in lowered, "wydatk" in lowered, "przych" in lowered]
    else:
        hits = [bool(re.search(r"sprzeda|zby", lowered)), "nieruchomo" in lowered, bool(re.search(r"pięciu|5 lat", lowered))]
    return round(sum(hits) / len(hits), 4)


def _holding_relevance_score(holding: str, issue: ControlledAuthorityIssue) -> float:
    return _issue_match_score(holding, issue) if holding else 0.0


def _classify_transaction(text: str) -> tuple[str, str]:
    lowered = text.lower()
    if re.search(r"umorzen|umorz", lowered):
        return "mortgage_remission", "remission"
    if re.search(r"zwolnieni\w* z dług|zwolnieni\w* z długu", lowered):
        return "debt_release", "debt_release"
    if re.search(r"ugoda.*bank|bank.*ugod", lowered):
        return "bank_settlement", "bank_settlement"
    if re.search(r"przewalutowan|walut", lowered):
        return "currency_conversion", "currency_conversion"
    if re.search(r"darowizn|otrzymał.*nieodpłat", lowered):
        return "gifted_property", "gift"
    if re.search(r"spłat|spłaci", lowered) and "kredyt" in lowered:
        return "credit_repayment", "repayment_from_sale_proceeds"
    if "deweloper" in lowered:
        return "developer_acquisition", "ownership_transfer_deadline"
    return "other", "other"


def _is_wrong_neighbor(
    transaction_type: str,
    event_type: str,
    issue: ControlledAuthorityIssue,
) -> bool:
    if issue.event_type != "repayment_from_sale_proceeds":
        return False
    return transaction_type in {
        "mortgage_remission",
        "debt_release",
        "bank_settlement",
        "currency_conversion",
        "gifted_property",
    } or event_type in {"remission", "debt_release", "bank_settlement", "currency_conversion", "gift"}


def _extract_complete_holding(chunk: RagChunk) -> tuple[str | None, dict[str, int] | None, str | None]:
    text = chunk.chunk_text
    source_type = str(chunk.source_type or "").lower()
    sections: list[tuple[str, int, int]] = []
    if source_type == "interpretation":
        sections.extend(_sections_after_headers(text, ("Ocena stanowiska", "Ocena Państwa stanowiska"), "assessment"))
        sections.extend(_sections_after_headers(text, ("Stanowisko.*?jest (?:prawidłowe|nieprawidłowe)",), "assessment"))
    elif source_type == "judgment":
        sections.extend(_sections_after_headers(text, ("Sentencja", "Z tych względów"), "operative"))
        sections.extend(_sections_after_headers(text, ("(?:NSA|Naczelny Sąd Administracyjny|WSA|Sąd).*?(?:oddala|uchyla|uwzględnia|orzeka)",), "operative"))
    for section_name, start, end in sections:
        segment = text[start:end]
        for sentence, offset_start, offset_end in _complete_sentences(segment):
            if _holding_sentence_is_relevant(sentence):
                return sentence, {"start": start + offset_start, "end": start + offset_end}, section_name
    return None, None, None


def _sections_after_headers(text: str, headers: Iterable[str], section: str) -> list[tuple[str, int, int]]:
    found: list[tuple[str, int, int]] = []
    for header in headers:
        for match in re.finditer(header, text, re.IGNORECASE | re.DOTALL):
            start = match.start()
            next_header = re.search(
                r"(?:Uzasadnienie(?:\s+interpretacji)?|Pouczenie|Funkcj[aię]|Stan faktyczny|Opis zdarzenia|Skarga)",
                text[match.end() :],
                re.IGNORECASE,
            )
            end = match.end() + next_header.start() if next_header else min(len(text), match.end() + 1600)
            found.append((section, start, end))
    return found


def _complete_sentences(text: str) -> Iterable[tuple[str, int, int]]:
    for match in re.finditer(r"[^.!?]{20,}[.!?](?=\s|$)", text, re.DOTALL):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        if value and len(value) <= 600 and value[-1] in ".!?":
            yield value, match.start(), match.end()


def _holding_sentence_is_relevant(sentence: str) -> bool:
    lowered = sentence.lower()
    return bool(
        re.search(r"stanowisko|uznano|stwierdza|należy|przysługuje|nie przysługuje|oddala|uchyla|zwoln", lowered)
    )


def _outcome_from_holding(holding: str) -> str:
    lowered = holding.lower()
    if re.search(r"nieprawidłowe|nie przysługuje|oddala", lowered):
        return "niekorzystny"
    if re.search(r"prawidłowe|przysługuje|uchyla|zwoln", lowered):
        return "korzystny_lub_potwierdzający"
    return "nierozstrzygnięty"


@lru_cache(maxsize=4)
def _count_jsonl_records(path_value: str, mtime_ns: int) -> int:
    path = Path(path_value)
    with path.open("rb") as source:
        return sum(1 for _ in source)


def _local_judgment_corpus_count() -> tuple[int, bool]:
    configured = Path(os.getenv("ALITIGATOR_JUDGMENT_CORPUS_PATH", str(_DEFAULT_JUDGMENT_CORPUS)))
    try:
        stat = configured.stat()
    except OSError:
        return 0, False
    return _count_jsonl_records(str(configured), stat.st_mtime_ns), True


def _active_judgment_index_count() -> tuple[int, str | None]:
    backend = os.getenv("ALITIGATOR_RAG_BACKEND", "sqlite").strip().lower()
    try:
        if backend in {"mysql", "mariadb"}:
            from app.mysql_rag import get_mysql_target, mysql_connection

            documents_table, _ = get_mysql_target()
            with mysql_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) AS count FROM `{documents_table}` WHERE source_type = %s", ("judgment",))
                    return int((cursor.fetchone() or {}).get("count") or 0), None
        db_path = get_rag_config().db_path
        if not db_path.exists():
            return 0, "active_sqlite_index_missing"
        with sqlite3.connect(db_path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM documents WHERE source_type = ?", ("judgment",)).fetchone()
        return int(row[0] if row else 0), None
    except Exception as exc:
        return 0, f"index_count_error:{type(exc).__name__}"


def audit_judgment_corpus(
    *,
    candidate_count: int,
    selected_count: int,
    filtered_count: int,
    errors: list[str],
) -> dict[str, object]:
    corpus_count, corpus_available = _local_judgment_corpus_count()
    indexed_count, index_error = _active_judgment_index_count()
    if any(":judgment:" in error for error in errors):
        root_cause = "judgment_retrieval_error"
    elif indexed_count == 0 and corpus_available:
        root_cause = "judgment_corpus_not_indexed_in_active_backend"
    elif indexed_count == 0:
        root_cause = "active_backend_has_no_judgment_corpus"
    elif candidate_count == 0:
        root_cause = "no_issue_matching_judgment_candidates"
    elif selected_count == 0 and filtered_count:
        root_cause = "judgment_candidates_failed_quality_filters"
    else:
        root_cause = ""
    empty_reason = (
        "retrieval_error" if root_cause == "judgment_retrieval_error"
        else "no_candidates_from_corpus" if candidate_count == 0
        else "candidates_not_selected" if selected_count == 0
        else ""
    )
    return {
        "executed": True,
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "filtered_count": filtered_count,
        "empty_result_reason": empty_reason,
        "corpus_count": corpus_count,
        "corpus_available": corpus_available,
        "indexed_count": indexed_count,
        "index_count_error": index_error,
        "zero_candidates_root_cause": root_cause,
    }
