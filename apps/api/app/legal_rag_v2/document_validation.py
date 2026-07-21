"""Question/document comparison built from independently extracted cards."""
from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

from app.legal_concepts import ConceptMatcher
from app.legal_concepts.normalizer import normalize_text
from app.query_understanding.deterministic_extractor import build_question_card as build_generic_question_card
from app.document_understanding.cache import load_document_card_payload, save_document_card_payload

DOCUMENT_CARD_VERSION = "document_card_v3"
VALIDATOR_VERSION = "relevance_validator_v2.1"
_PROVISION_RE = re.compile(r"\b(?:CIT|PIT|VAT)?\s*art\.\s*\d+[a-z]*(?:\s+ust\.\s*\d+[a-z]*)?(?:\s+pkt\s*\d+)?", re.I)
_HEADINGS_RE = re.compile(r"(?im)^\s*(Pytanie|Stanowisko wnioskodawcy|Ocena stanowiska|Uzasadnienie|Podsumowanie|Reasumuj[aą]c|W konsekwencji|Maj[aą]c na uwadze|Zatem|Tym samym)\s*[:\-]?\s*$")
_CACHE: "OrderedDict[str, DocumentCard]" = OrderedDict()
_CACHE_LIMIT = 2_000


def _values(value: object) -> list[str]:
    if isinstance(value, str): return [value]
    if isinstance(value, (list, tuple, set, frozenset)): return [str(item) for item in value if str(item).strip()]
    return []


def _unique(values: list[str]) -> tuple[str, ...]: return tuple(dict.fromkeys(item for item in values if item))


def _article(value: str) -> str:
    match = re.search(r"\bart\.\s*(\d+[a-z]*)", value, re.I)
    return match.group(1).casefold() if match else ""


@dataclass(frozen=True)
class QuestionCard:
    question_id: str = ""
    tax_domains: tuple[str, ...] = ()
    locked_institutions: tuple[str, ...] = ()
    primary_mechanism: str = ""
    secondary_mechanisms: tuple[str, ...] = ()
    provision_hints: tuple[str, ...] = ()
    taxpayer_role: str = ""
    counterparty_role: str = ""
    payment_direction: str = ""
    payment_type: str = ""
    transaction_type: str = ""
    contract_type: str = ""
    product_or_service: str = ""
    material_concepts: tuple[str, ...] = ()
    negative_concepts: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentEvidence:
    institution_id: str
    evidence_type: str
    value: str
    source: str = "document"
    start: int = -1
    end: int = -1


@dataclass(frozen=True)
class DocumentSections:
    question_section: str = ""
    position_result: str = ""
    conclusion_paragraphs: tuple[str, ...] = ()
    keyword_windows: tuple[str, ...] = ()

    def compact_text(self, *, title: str, limit: int = 6_000) -> str:
        return "\n\n".join(item for item in (title, self.question_section, self.position_result, *self.conclusion_paragraphs, *self.keyword_windows) if item)[:limit]


@dataclass(frozen=True)
class DocumentCard:
    document_id: str
    signature: str = ""
    document_type: str = ""
    title: str = ""
    tax_domains: tuple[str, ...] = ()
    detected_institutions: tuple[str, ...] = ()
    detected_mechanisms: tuple[str, ...] = ()
    cited_provisions: tuple[str, ...] = ()
    taxpayer_roles: tuple[str, ...] = ()
    counterparty_roles: tuple[str, ...] = ()
    payment_directions: tuple[str, ...] = ()
    payment_types: tuple[str, ...] = ()
    transaction_types: tuple[str, ...] = ()
    contract_types: tuple[str, ...] = ()
    products_or_services: tuple[str, ...] = ()
    material_concepts: tuple[str, ...] = ()
    question_presented_in_document: str = ""
    result_for_taxpayer: str = ""
    evidence: tuple[DocumentEvidence, ...] = ()
    extractor_version: str = DOCUMENT_CARD_VERSION

    @property
    def payment_direction(self) -> str | None: return self.payment_directions[0] if self.payment_directions else None
    def evidence_for(self, concept_id: str) -> tuple[DocumentEvidence, ...]: return tuple(item for item in self.evidence if item.institution_id == concept_id)
    def to_dict(self) -> dict[str, object]:
        evidence = [item.__dict__ for item in self.evidence]
        return {"document_id": self.document_id, "signature": self.signature, "document_type": self.document_type, "title": self.title, "tax_domains": list(self.tax_domains), "detected_institutions": list(self.detected_institutions), "detected_mechanisms": list(self.detected_mechanisms), "cited_provisions": list(self.cited_provisions), "taxpayer_roles": list(self.taxpayer_roles), "counterparty_roles": list(self.counterparty_roles), "payment_directions": list(self.payment_directions), "payment_types": list(self.payment_types), "transaction_types": list(self.transaction_types), "contract_types": list(self.contract_types), "products_or_services": list(self.products_or_services), "material_concepts": list(self.material_concepts), "question_presented_in_document": self.question_presented_in_document, "result_for_taxpayer": self.result_for_taxpayer, "evidence": evidence, "evidence_spans": evidence, "extractor_version": self.extractor_version}


@dataclass(frozen=True)
class DocumentValidation:
    passed: bool
    relation: str
    reject: bool
    reason: str
    matched_institutions: tuple[str, ...] = ()
    axes: Mapping[str, bool] | None = None


def build_question_card(*, question: str, issue: Any) -> QuestionCard:
    card, _ = build_generic_question_card(question)
    locks = list(dict.fromkeys([*card.locked_institutions, *getattr(issue, "locked_institution_ids", ())]))
    hints = [item.citation for item in card.verified_provision_hints]
    return QuestionCard(
        question_id=str(getattr(issue, "issue_id", "")), tax_domains=tuple(dict.fromkeys([*card.tax_domains, *getattr(issue, "tax_domains", ())])),
        locked_institutions=tuple(locks), primary_mechanism=locks[0] if locks else str(getattr(issue, "legal_mechanism", "")),
        provision_hints=tuple(dict.fromkeys([*hints, *getattr(issue, "possible_provision_concepts", ())])),
        taxpayer_role=card.taxpayer_roles[0] if card.taxpayer_roles else "", counterparty_role=card.counterparty_roles[0] if card.counterparty_roles else "",
        payment_direction=card.payment_direction or "", payment_type=card.payment_types[0] if card.payment_types else "",
        transaction_type=card.transaction_types[0] if card.transaction_types else "", contract_type=card.contract_types[0] if card.contract_types else "",
        product_or_service=card.products_or_services[0] if card.products_or_services else "", material_concepts=tuple(card.material_facts), negative_concepts=tuple(card.negative_concepts),
    )


def extract_document_sections(text: str, *, keywords: tuple[str, ...] = ()) -> DocumentSections:
    raw = str(text or "")
    headings = list(_HEADINGS_RE.finditer(raw)); sections: dict[str, str] = {}
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
        sections[normalize_text(match.group(1)).normalized] = raw[match.end():end].strip()[:2_000]
    conclusions = tuple(raw[match.start():match.start() + 900].strip() for marker in ("Reasumując", "W konsekwencji", "Zatem", "Tym samym") for match in re.finditer(re.escape(marker), raw, re.I))
    windows = tuple(raw[max(0, match.start() - 350):match.end() + 650].strip() for word in keywords[:24] for match in [re.search(re.escape(word), raw, re.I)] if match)
    return DocumentSections(sections.get("pytanie", ""), sections.get("ocena stanowiska", "") or sections.get("stanowisko wnioskodawcy", ""), _unique(list(conclusions))[:4], _unique(list(windows))[:6])


def _cache_key(candidate: Any, title: str, body: str, provisions: tuple[str, ...], dictionary_version: str) -> str:
    document_id = str(candidate.document_id or candidate.candidate_id)
    content_hash = hashlib.sha256("\x1f".join((title, body, *provisions)).encode()).hexdigest()[:16]
    return f"document_card:{document_id}:{DOCUMENT_CARD_VERSION}:{dictionary_version}:{content_hash}"


def _card_from_payload(payload: Mapping[str, Any]) -> DocumentCard:
    evidence = tuple(DocumentEvidence(**item) for item in payload.get("evidence", []))
    return DocumentCard(
        document_id=str(payload["document_id"]), signature=str(payload.get("signature") or ""), document_type=str(payload.get("document_type") or ""), title=str(payload.get("title") or ""),
        tax_domains=tuple(payload.get("tax_domains") or ()), detected_institutions=tuple(payload.get("detected_institutions") or ()), detected_mechanisms=tuple(payload.get("detected_mechanisms") or ()), cited_provisions=tuple(payload.get("cited_provisions") or ()),
        taxpayer_roles=tuple(payload.get("taxpayer_roles") or ()), counterparty_roles=tuple(payload.get("counterparty_roles") or ()), payment_directions=tuple(payload.get("payment_directions") or ()), payment_types=tuple(payload.get("payment_types") or ()), transaction_types=tuple(payload.get("transaction_types") or ()), contract_types=tuple(payload.get("contract_types") or ()), products_or_services=tuple(payload.get("products_or_services") or ()), material_concepts=tuple(payload.get("material_concepts") or ()), question_presented_in_document=str(payload.get("question_presented_in_document") or ""), result_for_taxpayer=str(payload.get("result_for_taxpayer") or ""), evidence=evidence, extractor_version=str(payload.get("extractor_version") or DOCUMENT_CARD_VERSION),
    )


def build_document_card(document: Any, *, matcher: ConceptMatcher | None = None) -> DocumentCard:
    """Does not accept a question or plan; every concept needs document evidence."""
    matcher = matcher if isinstance(matcher, ConceptMatcher) else ConceptMatcher()
    metadata: Mapping[str, Any] = document.metadata
    title = str(metadata.get("subject") or metadata.get("title") or "")
    body = str(document.text or "")
    metadata_provisions = tuple(_values(metadata.get("legal_provisions") or metadata.get("provisions")))
    key = _cache_key(document, title, body, metadata_provisions, matcher.dictionary.version)
    cached = _CACHE.get(key)
    if cached: _CACHE.move_to_end(key); return cached
    content_hash = key.rsplit(":", 1)[-1]
    persistent = load_document_card_payload(str(getattr(document, "document_id", None) or getattr(document, "candidate_id", "")), DOCUMENT_CARD_VERSION, matcher.dictionary.version, content_hash)
    if persistent:
        card = _card_from_payload(persistent); _CACHE[key] = card
        return card
    sections = extract_document_sections(body)
    scope = sections.compact_text(title=title)
    provisions = _unique([*metadata_provisions, *[item.group(0) for item in _PROVISION_RE.finditer(body)]])
    # A card remains question-independent but must inspect the complete
    # hydrated document when available; section excerpts only add structure.
    source_text = " ".join((title, body[:24_000], scope, *provisions))
    matches = matcher.match(source_text)
    groups: dict[str, list[str]] = {}
    evidence: list[DocumentEvidence] = []
    for match in matches.matches:
        definition = matcher.dictionary.by_id[match.concept_id]
        # Shadow terminology can describe facts and contracts, but only active
        # legal institutions may participate in a hard institution gate.
        if definition.status != "active" and match.concept_type == "legal_institution": continue
        groups.setdefault(match.concept_type, []).append(match.concept_id)
        start = source_text.casefold().find(match.matched_text.casefold())
        evidence.append(DocumentEvidence(match.concept_id, match.match_type, match.matched_text, "document", start, start + len(match.matched_text) if start >= 0 else -1))
    domains = _values(metadata.get("tax_domains"))
    if not domains: domains = [domain for match in matches.matches for domain in matcher.dictionary.by_id[match.concept_id].tax_domains]
    roles = _unique(groups.get("entity_role", []))
    directions = _unique([item.concept_id for item in matches.matches if matcher.dictionary.by_id[item.concept_id].semantic_role == "payment_direction"]) if len(roles) >= 2 else ()
    card = DocumentCard(
        document_id=str(getattr(document, "document_id", None) or getattr(document, "candidate_id", "")), signature=str(metadata.get("signature") or ""), document_type=str(getattr(document, "source_type", "")), title=title,
        tax_domains=_unique([item.upper() for item in domains]), detected_institutions=_unique(groups.get("legal_institution", [])),
        detected_mechanisms=_unique([*groups.get("legal_institution", []), *groups.get("legal_mechanism", [])]), cited_provisions=provisions,
        taxpayer_roles=roles[:1], counterparty_roles=roles[1:], payment_directions=directions, payment_types=_unique(groups.get("payment_type", [])),
        transaction_types=_unique(groups.get("transaction_type", [])), contract_types=_unique(groups.get("contract_type", [])), products_or_services=_unique(groups.get("product_or_service", [])),
        material_concepts=_unique(groups.get("factual_concept", [])), question_presented_in_document=sections.question_section, result_for_taxpayer=sections.position_result, evidence=tuple(evidence),
    )
    _CACHE[key] = card; _CACHE.move_to_end(key)
    save_document_card_payload(card.document_id, DOCUMENT_CARD_VERSION, matcher.dictionary.version, content_hash, card.to_dict())
    while len(_CACHE) > _CACHE_LIMIT: _CACHE.popitem(last=False)
    return card


def evaluate_document_relevance(question: QuestionCard, document: DocumentCard) -> DocumentValidation:
    required = set(question.locked_institutions); detected = set(document.detected_institutions); matched = tuple(sorted(required & detected))
    same_domain = not question.tax_domains or bool(set(question.tax_domains) & set(document.tax_domains))
    same_mechanism = bool(question.primary_mechanism and question.primary_mechanism in document.detected_mechanisms)
    question_articles = {_article(value) for value in question.provision_hints if _article(value)}; document_articles = {_article(value) for value in document.cited_provisions if _article(value)}
    same_provision = not question_articles or bool(question_articles & document_articles)
    same_payment = not question.payment_type or question.payment_type in document.payment_types
    same_transaction = not question.transaction_type or question.transaction_type in document.transaction_types
    same_contract = not question.contract_type or question.contract_type in document.contract_types
    same_product = not question.product_or_service or question.product_or_service in document.products_or_services
    same_direction = not question.payment_direction or question.payment_direction in document.payment_directions
    same_role = not question.taxpayer_role or question.taxpayer_role in document.taxpayer_roles
    axes = {"same_tax_domain": same_domain, "same_institution": bool(matched), "same_mechanism": same_mechanism, "same_provision_family": same_provision, "same_taxpayer_role": same_role, "same_counterparty_role": not question.counterparty_role or question.counterparty_role in document.counterparty_roles, "same_payment_direction": same_direction, "same_payment_type": same_payment, "same_transaction_type": same_transaction, "same_contract_type": same_contract, "same_product_or_service": same_product}
    if required and not matched: return DocumentValidation(False, "irrelevant", True, "missing_document_institution_evidence", axes=axes)
    material_match = all((same_payment, same_transaction, same_contract, same_product))
    if matched and same_mechanism and same_domain and material_match: return DocumentValidation(True, "direct", False, "document_institution_and_material_evidence", matched, axes)
    core_fact_match = any((
        bool(question.payment_type and same_payment),
        bool(question.transaction_type and same_transaction),
        bool(question.product_or_service and same_product),
    ))
    if matched and same_mechanism and same_domain and same_provision and core_fact_match:
        return DocumentValidation(True, "strong_analogy", False, "document_institution_with_core_factual_overlap", matched, axes)
    if matched: return DocumentValidation(False, "different_mechanism" if not same_mechanism else "context_only", True, "document_material_evidence_insufficient", matched, axes)
    return DocumentValidation(False, "irrelevant", True, "document_validation_failed", axes=axes)
