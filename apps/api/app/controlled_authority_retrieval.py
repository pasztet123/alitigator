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
_HOUSING_CREDIT_SPECIAL_RULE_START_YEAR = 2022


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
    judgment_query_suffix: str | None = None


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
        judgment_query_suffix=(
            "art. 21 ust. 1 pkt 131 oraz art. 21 ust. 25 pkt 2 ustawy PIT "
            "spłata kredytu zaciągniętego na nabycie następnie sprzedanej "
            "nieruchomości ze środków z jej sprzedaży"
        ),
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
    waterfalls: dict[str, list[dict[str, object]]] = {"interpretation": [], "judgment": []}
    seen: set[tuple[str, str, str]] = set()

    jobs: list[tuple[ControlledAuthorityIssue, str, int, str]] = []
    for issue in HOUSING_AUTHORITY_ISSUES:
        # Use the compact, provision-anchored issue query as the single recall
        # query.  The former pair of broad factual and provision-first queries
        # doubled expensive MySQL work without improving the later factual
        # selection stage, which already scores the full document text.
        for source_type, limit in (("interpretation", 8), ("judgment", 6)):
            authority_query = (
                issue.judgment_query_suffix
                if source_type == "judgment" and issue.judgment_query_suffix
                else issue.query_suffix
            )
            queries.append(
                {
                    "issue_id": issue.issue_id,
                    "issue_label": issue.label,
                    "source_type": source_type,
                    "query": authority_query,
                    "required_provision": issue.required_provision,
                    "transaction_type": issue.transaction_type,
                    "event_type": issue.event_type,
                    "query_variant": "provision_anchored",
                }
            )
            jobs.append((issue, source_type, limit, authority_query))

    search_results: list[tuple[ControlledAuthorityIssue, str, list[RagChunk]]] = []
    # Search each issue-source pair independently.  Exact citation lookups are
    # cheap; the bounded parallelism also contains the legacy FULLTEXT fallback
    # needed for authorities whose citation metadata is unknown.
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as executor:
        futures = {
            executor.submit(search, authority_query, limit=limit, source_types={source_type}): (issue, source_type)
            for issue, source_type, limit, authority_query in jobs
        }
        for future in as_completed(futures):
            issue, source_type = futures[future]
            try:
                candidates = _dedupe_candidate_documents(future.result())
            except Exception as exc:  # Secondary research must not suppress primary law.
                errors.append(f"{issue.issue_id}:{source_type}:{type(exc).__name__}")
                continue
            candidate_counts[source_type] += len(candidates)
            search_results.append((issue, source_type, candidates))

    # Hydrate every unique authority in one database round-trip.  The previous
    # per-query hydration repeated large document reads eight times and consumed
    # a material part of the request budget even after search had completed.
    seed_chunks = [chunk for _, _, candidates in search_results for chunk in candidates]
    hydrated_chunks = _hydrate_document_contexts(seed_chunks, context_fetcher=context_fetcher)
    hydrated_by_document = {chunk.document_id: chunk for chunk in hydrated_chunks}

    for issue, source_type, candidates in search_results:
        for seed_chunk in candidates:
            chunk = hydrated_by_document.get(seed_chunk.document_id, seed_chunk)
            selection, waterfall = _select_candidate_with_trace(chunk, issue)
            waterfalls[source_type].append(waterfall)
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
    quality_pass_counts = dict(selected_counts)
    cards = _limit_authority_cards(cards, per_issue_source=1)
    selected_counts = {
        source_type: sum(1 for card in cards if card.get("source_type") == source_type)
        for source_type in ("interpretation", "judgment")
    }
    judgment_audit = audit_judgment_corpus(
        candidate_count=candidate_counts["judgment"],
        selected_count=selected_counts["judgment"],
        filtered_count=filtered_counts["judgment"],
        errors=errors,
    )
    interpretation_errors = [error for error in errors if ":interpretation:" in error]
    interpretation_empty_reason = (
        "retrieval_error"
        if interpretation_errors and candidate_counts["interpretation"] == 0
        else "no_candidates_from_corpus"
        if candidate_counts["interpretation"] == 0
        else "candidates_not_selected"
        if selected_counts["interpretation"] == 0
        else ""
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
        "quality_pass_counts": quality_pass_counts,
        "rendered_authority_cards": len(cards),
        "empty_high_quality_result_supported": True,
        "outcome": "no_high_quality_authorities" if not cards else "high_quality_authorities_found",
        "errors": errors,
        "judgment_lane": judgment_audit,
        "interpretation_lane": {
            "executed": True,
            "status": "completed_with_errors" if interpretation_errors else "completed",
            "candidates_before_filters": candidate_counts["interpretation"],
            "candidates_after_filters": candidate_counts["interpretation"] - filtered_counts["interpretation"],
            "selected_count": selected_counts["interpretation"],
            "empty_result_reason": interpretation_empty_reason,
            "candidate_waterfall": waterfalls["interpretation"],
        },
        "judgment_filter_waterfall": waterfalls["judgment"],
        "interpretation_lane_executed": True,
        "interpretation_candidates_before_filters_recorded": True,
        "interpretation_candidates_after_filters_recorded": True,
        "interpretation_selected_count_recorded": True,
        "judgment_lane_executed": True,
        "judgment_candidate_count_recorded": True,
        "judgment_selected_count_recorded": True,
        "judgment_empty_result_reason_recorded": True,
        "judgment_corpus_count_recorded": True,
        "judgment_indexed_count_recorded": True,
    }
    return cards, outcome


def _dedupe_candidate_documents(candidates: list[RagChunk]) -> list[RagChunk]:
    """Keep the strongest seed chunk once per authority document."""
    deduped: list[RagChunk] = []
    seen: set[str] = set()
    for chunk in candidates:
        document_id = str(chunk.document_id or "").strip()
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        deduped.append(chunk)
    return deduped


def _limit_authority_cards(
    cards: list[dict[str, object]],
    *,
    per_issue_source: int,
) -> list[dict[str, object]]:
    """Keep the answer useful without rendering a wall of near-duplicates."""
    selected: list[dict[str, object]] = []
    counts: dict[tuple[str, str], int] = {}
    for card in cards:
        key = (str(card.get("issue_id") or ""), str(card.get("source_type") or ""))
        if counts.get(key, 0) >= per_issue_source:
            continue
        counts[key] = counts.get(key, 0) + 1
        selected.append(card)
    return selected


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
    if source_type not in {"interpretation", "general_interpretation", "judgment"}:
        return None
    text = _source_text(chunk)
    holding, holding_span, holding_section = _extract_complete_holding(chunk, issue)
    if holding is None or holding_span is None or not holding_section:
        return None
    transaction_type, event_type = _classify_transaction(text, holding=holding, issue=issue)
    if _is_wrong_neighbor(transaction_type, event_type, issue):
        return None
    provision_score = _provision_match_score(text, issue.required_provision)
    provision_status = _provision_match_status(chunk, provision_score)
    historical_evidence = _historical_authority_evidence(chunk, issue)
    issue_score = _issue_match_score(text, issue)
    material_fact_score = _material_fact_match_score(text, issue)
    holding_relevance = _holding_relevance_score(holding, issue)
    provision_allowed, authority_status, effective_provision_score = _authority_provision_gate(
        source_type=source_type,
        issue=issue,
        provision_status=provision_status,
        historical_evidence=historical_evidence,
        issue_score=issue_score,
        material_fact_score=material_fact_score,
        holding_relevance=holding_relevance,
    )
    # An authority must earn every leg of the relevance test.  In particular a
    # document on a neighbouring mortgage topic cannot pass merely on keywords.
    if (
        not provision_allowed
        or issue_score < 0.34
        or material_fact_score < 0.34
        or holding_relevance < 0.34
    ):
        return None
    score = round(
        effective_provision_score * 0.35
        + issue_score * 0.25
        + material_fact_score * 0.22
        + holding_relevance * 0.18,
        4,
    )
    if authority_status == "historical_authority":
        binding_reason = (
            f"Historyczne rozstrzygnięcie dotyczy mechanizmu „{issue.label}”, ale nie stosuje "
            f"obecnego art. {issue.required_provision}; wyjaśnia tło zmiany i nie zastępuje aktualnej ustawy."
        )
    else:
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
        "provision_match_score": None if provision_status == "unknown" else provision_score,
        "provision_match_status": provision_status,
        "authority_status": authority_status,
        "temporal_status": "historical" if authority_status == "historical_authority" else (
            "uncertain" if authority_status == "provision_metadata_unknown" else "current"
        ),
        "current_law_support": authority_status == "current_authority",
        "historical_basis": historical_evidence or "",
        "issue_match_score": issue_score,
        "material_fact_match_score": material_fact_score,
        "holding_relevance_score": holding_relevance,
        "authority_score": score,
        "similarity_reason": (
            f"Historycznie zgodny mechanizm: {issue.label}; aktualną podstawą jest art. {issue.required_provision}."
            if authority_status == "historical_authority"
            else f"Zgodny przepis i mechanizm: {issue.label}."
        ),
        "distinguishing_facts": (
            f"Źródło historyczne — nie przedstawia obecnego art. {issue.required_provision}; {issue.distinguishing_fact}."
            if authority_status == "historical_authority"
            else issue.distinguishing_fact
        ),
        "claim_bindings": [
            {"claim_id": claim_id, "score": score, "reason": binding_reason}
            for claim_id in issue.claim_ids
        ],
    }


def _select_candidate_with_trace(
    chunk: RagChunk, issue: ControlledAuthorityIssue,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Keep every quality decision observable; unknown metadata is not mismatch."""
    text = _source_text(chunk)
    holding, _, holding_section = _extract_complete_holding(chunk, issue)
    transaction_type, event_type = _classify_transaction(text, holding=holding or "", issue=issue)
    provision_score = _provision_match_score(text, issue.required_provision)
    provision_status = _provision_match_status(chunk, provision_score)
    historical_evidence = _historical_authority_evidence(chunk, issue)
    scores: dict[str, float | None] = {
        "provision": None if provision_status == "unknown" else provision_score,
        "issue": _issue_match_score(text, issue),
        "material_fact": _material_fact_match_score(text, issue),
        "holding": _holding_relevance_score(holding or "", issue),
    }
    provision_allowed, authority_status, _ = _authority_provision_gate(
        source_type=str(chunk.source_type or "").lower(),
        issue=issue,
        provision_status=provision_status,
        historical_evidence=historical_evidence,
        issue_score=float(scores["issue"] or 0.0),
        material_fact_score=float(scores["material_fact"] or 0.0),
        holding_relevance=float(scores["holding"] or 0.0),
    )
    metadata = {
        name: "match" if value else "unknown"
        for name, value in {
            "signature": chunk.signature, "authority": chunk.authority,
            "published_date": chunk.published_date, "legal_provisions": chunk.legal_provisions,
            "source_subtype": chunk.source_subtype, "source_url": chunk.source_url,
            "full_text": chunk.chunk_text,
        }.items()
    }
    thresholds = {"provision": 1.0, "issue": .34, "material_fact": .34, "holding": .34}
    rejection = ""
    if _is_wrong_neighbor(transaction_type, event_type, issue): rejection = "wrong_neighbor"
    elif holding is None or not holding_section: rejection = "missing_holding"
    elif not provision_allowed: rejection = (
        "provision_metadata_unknown_insufficient_match"
        if provision_status == "unknown"
        else "provision_mismatch"
    )
    else:
        rejection = next(
            (
                key
                for key, threshold in thresholds.items()
                if key != "provision" and float(scores[key] or 0.0) < threshold
            ),
            "",
        )
    trace = {"document_id": chunk.document_id, "source_type": chunk.source_type,
             "issue_id": issue.issue_id, "issue_label": issue.label,
             "scores": scores, "thresholds": thresholds, "metadata": metadata,
             "provision_match_status": provision_status,
             "authority_status": authority_status,
             "historical_basis": historical_evidence,
             "first_rejection_reason": rejection or None,
             "result": "selected" if not rejection else "rejected"}
    return (_select_candidate(chunk, issue) if not rejection else None), trace


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


def _provision_match_status(chunk: RagChunk, provision_score: float) -> str:
    if provision_score >= 1.0:
        return "match"
    # No declared provisions means the legal-unit comparison is unknown, not a
    # proven negative.  Strong factual/holding evidence may still admit a
    # judgment as uncertain secondary authority.
    if not tuple(value for value in chunk.legal_provisions if str(value).strip()):
        return "unknown"
    return "mismatch"


def _authority_provision_gate(
    *,
    source_type: str,
    issue: ControlledAuthorityIssue,
    provision_status: str,
    historical_evidence: str | None,
    issue_score: float,
    material_fact_score: float,
    holding_relevance: float,
) -> tuple[bool, str, float]:
    if provision_status == "match":
        return True, "current_authority", 1.0

    strong_mechanism_match = min(issue_score, material_fact_score, holding_relevance) >= 0.67
    if source_type == "judgment" and strong_mechanism_match:
        # Article 21(30a) codified a mechanism litigated under the earlier legal
        # state.  Such a judgment may explain the genesis of the current rule,
        # but must never be presented as proof of current law.
        if issue.issue_id == "credit_on_sold_property" and historical_evidence:
            return True, "historical_authority", 0.45
        if provision_status == "unknown":
            return True, "provision_metadata_unknown", 0.35
    return False, "rejected_authority", 0.0


def _historical_authority_evidence(
    chunk: RagChunk,
    issue: ControlledAuthorityIssue,
) -> str | None:
    """Return explicit proof that the dispute uses the pre-30a legal state."""
    if str(chunk.source_type or "").lower() != "judgment" or issue.issue_id != "credit_on_sold_property":
        return None
    for label, value in (
        ("legal_state_date", chunk.legal_state_date or ""),
        ("publication_date", chunk.published_date or ""),
    ):
        year_match = re.search(r"\b(20\d{2})\b", value)
        if year_match and int(year_match.group(1)) < _HOUSING_CREDIT_SPECIAL_RULE_START_YEAR:
            return f"{label}:{year_match.group(1)}"

    # A judgment can be published years after the material tax period.  Limit
    # detection to the case-identification/opening part of the document, rather
    # than treating a citation of an older judgment deep in the reasoning as the
    # legal state of the case at hand.
    opening = re.sub(r"\s+", " ", chunk.chunk_text[:8000]).lower()
    material_period_patterns = (
        r"(?:podatk\w*|zobowiązani\w*)[^.]{0,180}\bza(?:\s+rok)?\s+(20\d{2})",
        r"odpłatn\w* zbyci\w*[^.]{0,180}(?:w roku|w dniu|w)\s+(?:\d{1,2}[.-]\d{1,2}[.-])?(20\d{2})",
        r"sprzedaż\w*[^.]{0,140}(?:w roku|w dniu|w)\s+(?:\d{1,2}[.-]\d{1,2}[.-])?(20\d{2})",
    )
    years = [
        int(match.group(1))
        for pattern in material_period_patterns
        for match in re.finditer(pattern, opening, re.IGNORECASE)
    ]
    historical_years = [year for year in years if year < _HOUSING_CREDIT_SPECIAL_RULE_START_YEAR]
    if historical_years:
        return f"material_tax_period:{max(historical_years)}"
    return None


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
    if not holding:
        return 0.0
    issue_score = _issue_match_score(holding, issue)
    if issue.issue_id == "credit_on_sold_property" and not _credit_on_sold_property_relation(holding):
        return 0.0
    return issue_score


def _credit_on_sold_property_relation(text: str) -> bool:
    """Require the credit to relate to the sold asset, not merely co-occur."""
    normalized = re.sub(r"\s+", " ", text.lower())
    return bool(
        re.search(
            r"spłat\w* kredyt\w*.{0,180}(?:na (?:uprzednie )?(?:nabycie|zakup|wybudowanie|wytworzenie))"
            r".{0,180}(?:następnie )?(?:sprzedan|zbywan)",
            normalized,
        )
        or re.search(
            r"spłat\w* kredyt\w*.{0,180}(?:na (?:uprzednie )?(?:nabycie|zakup|wybudowanie|wytworzenie))"
            r".{0,180}(?:tego|tej|ww\.) (?:lokalu|mieszkania|nieruchomości|budynku)",
            normalized,
        )
        or re.search(
            r"(?:sprzedaż|sprzedaży|zbycie|zbycia).{0,360}(?:na spłat\w*|spłat\w*).{0,100}kredyt\w*"
            r".{0,220}(?:nabycie|zakup|wybudowanie|wytworzenie).{0,100}(?:tego|tej|sprzedan|zbywan|ww\.)",
            normalized,
        )
    )


def _classify_transaction(
    text: str,
    *,
    holding: str = "",
    issue: ControlledAuthorityIssue | None = None,
) -> tuple[str, str]:
    # The selected holding is the highest-signal window.  Whole-document prose
    # routinely mentions gifts, currency conversion or debt relief only as
    # background; treating one incidental word as the transaction type rejected
    # directly relevant interpretations.
    focus = re.sub(r"\s+", " ", holding).lower()
    lowered = text.lower()
    credit_focus = focus or lowered
    if (
        "kredyt" in credit_focus
        and re.search(r"spłat|spłaci", credit_focus)
        and re.search(r"zbywan|sprzed", credit_focus)
    ):
        return "credit_repayment", "repayment_from_sale_proceeds"
    if issue and issue.event_type == "ownership_transfer_deadline" and (
        "deweloper" in credit_focus and re.search(r"własno|naby", credit_focus)
    ):
        return "developer_acquisition", "ownership_transfer_deadline"
    negative_focus = focus or lowered[:6000]
    if re.search(r"umorzen|umorz", negative_focus):
        return "mortgage_remission", "remission"
    if re.search(r"zwolnieni\w* z dług|zwolnieni\w* z długu", negative_focus):
        return "debt_release", "debt_release"
    if re.search(r"ugoda.{0,160}bank|bank.{0,160}ugod", negative_focus):
        return "bank_settlement", "bank_settlement"
    if re.search(r"przewalutowan|kredyt\w* walut", negative_focus):
        return "currency_conversion", "currency_conversion"
    if re.search(r"darowizn|otrzymał.{0,160}nieodpłat", negative_focus):
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


def _extract_complete_holding(
    chunk: RagChunk,
    issue: ControlledAuthorityIssue,
) -> tuple[str | None, dict[str, int] | None, str | None]:
    text = chunk.chunk_text
    source_type = str(chunk.source_type or "").lower()
    sections: list[tuple[str, int, int]] = []
    if source_type in {"interpretation", "general_interpretation"}:
        interpretation_stops = (
            "Dodatkowe informacje",
            "Informacja o zakresie rozstrzygnięcia",
            "Funkcja ochronna",
            "Pouczenie",
            "Skarga do sądu",
        )
        sections.extend(
            _sections_after_headers(
                text,
                ("Ocena stanowiska", "Ocena Państwa stanowiska"),
                "assessment_reasoning",
                stop_headers=interpretation_stops,
            )
        )
        sections.extend(
            _sections_after_headers(
                text,
                (r"Stanowisko[^.\n]{0,300}?jest (?:prawidłowe|nieprawidłowe)",),
                "assessment_reasoning",
                stop_headers=interpretation_stops,
            )
        )
    elif source_type == "judgment":
        sections.extend(
            _sections_after_headers(
                text,
                ("Naczelny Sąd Administracyjny zważył", "Sąd zważył", "Uzasadnienie"),
                "judicial_reasoning",
                stop_headers=("Pouczenie",),
            )
        )
        sections.extend(
            _sections_after_headers(
                text,
                ("Sentencja", "Z tych względów"),
                "operative",
                stop_headers=("Uzasadnienie",),
            )
        )

    candidates: list[tuple[tuple[float, ...], str, dict[str, int], str]] = []
    seen_spans: set[tuple[int, int]] = set()
    search_sections = [*sections, (
        "assessment_reasoning"
        if source_type in {"interpretation", "general_interpretation"}
        else "judicial_reasoning",
        0,
        len(text),
    )]
    for section_name, start, end in search_sections:
        segment = text[start:end]
        for sentence, offset_start, offset_end in _complete_sentences(segment):
            absolute_span = (start + offset_start, start + offset_end)
            if absolute_span in seen_spans:
                continue
            seen_spans.add(absolute_span)
            rank = _holding_candidate_rank(
                sentence,
                issue=issue,
                section_name=section_name,
                position=absolute_span[0],
                source_type=source_type,
            )
            if rank is None:
                continue
            candidates.append(
                (rank, sentence, {"start": absolute_span[0], "end": absolute_span[1]}, section_name)
            )
    if not candidates:
        return None, None, None
    _, sentence, span, section_name = max(candidates, key=lambda item: item[0])
    return sentence, span, section_name


def _sections_after_headers(
    text: str,
    headers: Iterable[str],
    section: str,
    *,
    stop_headers: Iterable[str] = (),
) -> list[tuple[str, int, int]]:
    found: list[tuple[str, int, int]] = []
    for header in headers:
        for match in re.finditer(header, text, re.IGNORECASE | re.DOTALL):
            start = match.start()
            stop_pattern = "|".join(f"(?:{value})" for value in stop_headers)
            next_header = (
                re.search(stop_pattern, text[match.end() :], re.IGNORECASE)
                if stop_pattern
                else None
            )
            end = match.end() + next_header.start() if next_header else len(text)
            found.append((section, start, end))
    return found


def _complete_sentences(text: str) -> Iterable[tuple[str, int, int]]:
    # Protect common legal abbreviations before locating sentence boundaries;
    # otherwise a conclusion ending with "art. 21 ..." is truncated at "art.".
    protected = list(text)
    abbreviation_re = re.compile(
        r"\b(?:art|ust|pkt|lit|poz|dz|nr|sygn|ww|tj|t\.\s*j|r|"
        r"p\.\s*p\.\s*s\.\s*a|u\.\s*p\.\s*d\.\s*o\.\s*f|o\.\s*p)\.",
        re.IGNORECASE,
    )
    for abbreviation in abbreviation_re.finditer(text):
        for index in range(abbreviation.start(), abbreviation.end()):
            if protected[index] == ".":
                protected[index] = "․"
    protected_text = "".join(protected)
    sentence_start = 0
    for boundary in re.finditer(r"[.!?](?=\s|$)", protected_text):
        raw_start = sentence_start
        raw_end = boundary.end()
        sentence_start = raw_end
        while raw_start < raw_end and text[raw_start].isspace():
            raw_start += 1
        value = re.sub(r"\s+", " ", text[raw_start:raw_end]).strip()
        if value and 20 <= len(value) <= 1400 and value[-1] in ".!?":
            yield value, raw_start, raw_end


def _holding_candidate_rank(
    sentence: str,
    *,
    issue: ControlledAuthorityIssue,
    section_name: str,
    position: int,
    source_type: str,
) -> tuple[float, ...] | None:
    if not _holding_sentence_is_relevant(sentence):
        return None
    lowered = sentence.lower()
    if "?" in sentence or re.match(r"\s*(?:czy|wątpliwości|pytani[ea])\b", lowered):
        return None
    if re.search(r"\b(?:zdaniem|według)\s+(?:pani|pana|wnioskodawc|skarżąc)|\bwnioskodawc\w* uważa", lowered):
        return None
    issue_score = _holding_relevance_score(sentence, issue)
    material_score = _material_fact_match_score(sentence, issue)
    strong_conclusion = 1.0 if re.search(
        r"uprawnia|nie uprawnia|stanowi|nie stanowi|nie jest (?:wydatkiem|kosztem)|"
        r"jest wydatkiem|przysługuje|nie przysługuje|należało uznać|prawidłowy wniosek|"
        r"nie można (?:podzielić|uznać)",
        lowered,
    ) else 0.0
    section_score = 1.0 if section_name in {"assessment_reasoning", "judicial_reasoning"} else 0.5
    source_specific = 1.0 if (
        source_type in {"interpretation", "general_interpretation"}
        and re.search(r"uprawnia|stanowisko|zwoln", lowered)
    ) or (
        source_type == "judgment" and re.search(r"sąd|należy|nie jest|stanowi|prawidłow", lowered)
    ) else 0.0
    return (
        issue_score,
        material_score,
        strong_conclusion,
        section_score,
        source_specific,
        min(position / max(len(sentence), 1), 1_000_000.0),
    )


def _holding_sentence_is_relevant(sentence: str) -> bool:
    lowered = sentence.lower()
    return bool(
        re.search(
            r"stanowisko|uznano|stwierdza|należy|przysługuje|nie przysługuje|"
            r"uprawnia|nie uprawnia|stanowi|nie stanowi|nie jest|prawidłow|"
            r"oddala|uchyla|zwoln",
            lowered,
        )
    )


def _outcome_from_holding(holding: str) -> str:
    lowered = holding.lower()
    if re.search(r"nieprawidłowe|nie przysługuje|nie uprawnia|nie stanowi|nie jest wydatkiem|oddala", lowered):
        return "niekorzystny"
    if re.search(r"prawidłowe|przysługuje|uprawnia|stanowi|jest wydatkiem|uchyla|zwoln", lowered):
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


def _judgment_metadata_coverage() -> dict[str, object]:
    """Diagnostic-only completeness report; empty metadata is never mismatch."""
    backend = os.getenv("ALITIGATOR_RAG_BACKEND", "sqlite").strip().lower()
    fields = ("signature", "authority", "published_date", "legal_provisions", "source_subtype", "source_url")
    try:
        if backend in {"mysql", "mariadb"}:
            from app.mysql_rag import get_mysql_target, mysql_connection
            table, _ = get_mysql_target()
            with mysql_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT COUNT(*) total, "
                        "SUM(signature IS NOT NULL AND signature <> '') signature, "
                        "SUM(authority <> '') authority, SUM(published_date IS NOT NULL AND published_date <> '') published_date, "
                        "SUM(legal_provisions_json <> '[]') legal_provisions, SUM(source_subtype <> '') source_subtype, "
                        "SUM(source_url IS NOT NULL AND source_url <> '') source_url "
                        "FROM `%s` WHERE source_type = 'judgment'" % table
                    )
                    row = cursor.fetchone() or {}
        else:
            with sqlite3.connect(get_rag_config().db_path) as connection:
                row = connection.execute("SELECT COUNT(*) total, SUM(signature <> '') signature, SUM(authority <> '') authority, SUM(published_date <> '') published_date, SUM(legal_provisions_json <> '[]') legal_provisions, SUM(source_subtype <> '') source_subtype, SUM(source_url <> '') source_url FROM documents WHERE source_type='judgment'").fetchone()
                row = dict(zip(("total", *fields), row))
        total = int(row.get("total") or 0)
        return {"total": total, "fields": {field: int(row.get(field) or 0) for field in fields}, "missing_metadata_is_unknown": True}
    except Exception as exc:
        return {"error": type(exc).__name__, "missing_metadata_is_unknown": True}


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
        "status": "completed_with_errors" if any(":judgment:" in error for error in errors) else "completed",
        "candidate_count": candidate_count,
        "selected_count": selected_count,
        "filtered_count": filtered_count,
        "empty_result_reason": empty_reason,
        "local_source_document_count": corpus_count,
        "corpus_available": corpus_available,
        "backend_document_count": indexed_count,
        # Compatibility aliases; presentation uses the explicit names above.
        "corpus_count": corpus_count,
        "indexed_count": indexed_count,
        "index_count_error": index_error,
        "zero_candidates_root_cause": root_cause,
        "metadata_coverage": _judgment_metadata_coverage(),
    }
