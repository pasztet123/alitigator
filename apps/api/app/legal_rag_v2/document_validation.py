"""Question/document separation for deterministic authority relevance checks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.legal_institutions import InstitutionMatcher
from app.legal_institutions.normalizer import normalize_polish, phrase_present


DOCUMENT_CARD_VERSION = "deterministic_document_card_v1"
_ARTICLE_RE = re.compile(r"\bart\.\s*(\d+[a-z]*)", re.IGNORECASE)


def _values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _article(value: str) -> str:
    match = _ARTICLE_RE.search(normalize_polish(value).normalized)
    return match.group(1) if match else ""


@dataclass(frozen=True)
class QuestionCard:
    tax_domains: tuple[str, ...] = ()
    locked_institutions: tuple[str, ...] = ()
    primary_mechanism: str = ""
    provision_hints: tuple[str, ...] = ()
    payment_type: str = ""
    transaction_type: str = ""
    contract_type: str = ""
    material_concepts: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentEvidence:
    institution_id: str
    evidence_type: str
    value: str


@dataclass(frozen=True)
class DocumentCard:
    document_id: str
    signature: str = ""
    tax_domains: tuple[str, ...] = ()
    detected_institutions: tuple[str, ...] = ()
    detected_mechanisms: tuple[str, ...] = ()
    cited_provisions: tuple[str, ...] = ()
    payment_types: tuple[str, ...] = ()
    transaction_types: tuple[str, ...] = ()
    contract_types: tuple[str, ...] = ()
    material_concepts: tuple[str, ...] = ()
    evidence: tuple[DocumentEvidence, ...] = ()
    classification_source: str = DOCUMENT_CARD_VERSION

    def evidence_for(self, institution_id: str) -> tuple[DocumentEvidence, ...]:
        return tuple(item for item in self.evidence if item.institution_id == institution_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "signature": self.signature,
            "tax_domains": list(self.tax_domains),
            "detected_institutions": list(self.detected_institutions),
            "detected_mechanisms": list(self.detected_mechanisms),
            "cited_provisions": list(self.cited_provisions),
            "payment_types": list(self.payment_types),
            "transaction_types": list(self.transaction_types),
            "contract_types": list(self.contract_types),
            "material_concepts": list(self.material_concepts),
            "evidence": [item.__dict__ for item in self.evidence],
            "classification_source": self.classification_source,
        }


@dataclass(frozen=True)
class DocumentValidation:
    passed: bool
    relation: str
    reject: bool
    reason: str
    matched_institutions: tuple[str, ...] = ()


def build_question_card(*, question: str, issue: Any) -> QuestionCard:
    normalized = normalize_polish(question).normalized
    payment_type = "saas_access_fee" if "saas" in normalized else ""
    transaction_type = "software_access" if any(value in normalized for value in ("saas", "program komputerowy", "oprogramowan")) else ""
    contract_type = "eula" if "eula" in normalized else ""
    concepts = tuple(value for value in ("saas", "eula", "program komputerowy", "należności licencyjne") if value in normalized)
    return QuestionCard(
        tax_domains=tuple(str(item).upper() for item in issue.tax_domains),
        locked_institutions=tuple(issue.locked_institution_ids),
        primary_mechanism=str(issue.legal_mechanism),
        provision_hints=tuple(issue.possible_provision_concepts),
        payment_type=payment_type,
        transaction_type=transaction_type,
        contract_type=contract_type,
        material_concepts=concepts,
    )


def build_document_card(candidate: Any, *, matcher: InstitutionMatcher) -> DocumentCard:
    """Classify solely from candidate text and metadata; never from a question."""

    metadata: Mapping[str, Any] = candidate.metadata
    title = str(metadata.get("subject") or metadata.get("title") or "")
    provisions = tuple(_values(metadata.get("legal_provisions") or metadata.get("provisions")))
    body = str(candidate.text)
    normalized_title = normalize_polish(title)
    normalized_body = normalize_polish(body)
    normalized = normalize_polish(" ".join((title, body, *provisions)))
    domains = tuple(str(item).upper() for item in _values(metadata.get("tax_domains")))
    if not domains:
        domains = tuple(value for value in ("CIT", "PIT", "VAT") if phrase_present(value, normalized))
    evidence: list[DocumentEvidence] = []
    institutions: list[str] = []
    mechanisms: list[str] = []
    for definition in matcher.dictionary.institutions:
        if definition.status != "active":
            continue
        found: list[DocumentEvidence] = []
        # Document mode deliberately excludes material_concepts from the
        # recognition threshold: words such as payer or non-resident occur in
        # background facts and cannot alone label a whole interpretation.
        for phrase in (
            definition.canonical_name,
            *definition.exact_aliases,
            *definition.lemma_aliases,
            *definition.statutory_phrases,
        ):
            if phrase_present(phrase, normalized_title):
                found.append(DocumentEvidence(definition.institution_id, "title_phrase", phrase))
            if phrase_present(phrase, normalized_body):
                found.append(DocumentEvidence(definition.institution_id, "text_phrase", phrase))
        for hint in definition.provision_hints:
            article = _article(hint)
            if phrase_present(hint, normalized) or (
                article and article in {_article(value) for value in provisions}
                and (not definition.tax_domains or set(domains).intersection(definition.tax_domains))
            ):
                found.append(DocumentEvidence(definition.institution_id, "provision", hint))
        if found:
            institutions.append(definition.institution_id)
            mechanisms.extend(definition.legal_mechanisms or (definition.institution_id,))
            evidence.extend(found)

    # A document can be independently classified even where the corresponding
    # catalogue entry is intentionally shadow.  These coarse mechanisms are
    # only negative safeguards and are never inherited from the question.
    if re.search(r"ulga mieszkaniow|sprzedaz.*nieruchom|lokal mieszkal", normalized.normalized):
        mechanisms.append("housing_relief")
    if re.search(r"rehabilitacyj|niepelnosprawn", normalized.normalized):
        mechanisms.append("rehabilitation_relief")
    if re.search(r"leasing.{0,60}(samoch|pojazd)|samochod.{0,60}leasing", normalized.normalized):
        mechanisms.append("vehicle_lease_cost")
    payment_types = tuple(value for value in ("saas_access_fee", "interest", "royalty") if (
        (value == "saas_access_fee" and "saas" in normalized.normalized)
        or (value == "interest" and "odsetk" in normalized.normalized)
        or (value == "royalty" and "naleznosci licencyjn" in normalized.normalized)
    ))
    transaction_types = tuple(value for value in ("software_access", "debt_to_capital", "foreign_tax_credit") if (
        (value == "software_access" and any(term in normalized.normalized for term in ("saas", "oprogramowan", "program komputerowy")))
        or (value == "debt_to_capital" and re.search(r"konwersj.{0,40}(dlug|wierzyteln).{0,80}kapital", normalized.normalized))
        or (value == "foreign_tax_credit" and re.search(r"odliczen.{0,40}zagraniczn.{0,40}podat", normalized.normalized))
    ))
    contract_types = ("eula",) if "eula" in normalized.normalized else ()
    return DocumentCard(
        document_id=str(candidate.document_id),
        signature=str(metadata.get("signature") or ""),
        tax_domains=tuple(dict.fromkeys(domains)),
        detected_institutions=tuple(dict.fromkeys(institutions)),
        detected_mechanisms=tuple(dict.fromkeys(mechanisms)),
        cited_provisions=provisions,
        payment_types=payment_types,
        transaction_types=transaction_types,
        contract_types=contract_types,
        material_concepts=tuple(
            item.value
            for item in evidence
            if item.evidence_type in {"title_phrase", "text_phrase"}
        ),
        evidence=tuple(dict.fromkeys(evidence)),
    )


def evaluate_document_relevance(question: QuestionCard, document: DocumentCard) -> DocumentValidation:
    required = set(question.locked_institutions)
    detected = set(document.detected_institutions)
    matched = tuple(sorted(required.intersection(detected)))
    if required and not matched:
        return DocumentValidation(False, "irrelevant", True, "missing_document_institution_evidence")
    same_domain = not question.tax_domains or not document.tax_domains or bool(set(question.tax_domains).intersection(document.tax_domains))
    same_payment = not question.payment_type or question.payment_type in document.payment_types
    same_transaction = not question.transaction_type or question.transaction_type in document.transaction_types
    if required and matched and same_domain and same_payment and same_transaction:
        return DocumentValidation(
            True,
            "direct",
            False,
            "document_institution_and_material_evidence",
            matched,
        )
    if required and matched:
        return DocumentValidation(
            False,
            "context_only",
            True,
            "document_mechanism_without_matching_transaction",
            matched,
        )
    return DocumentValidation(False, "irrelevant", True, "document_validation_failed")
