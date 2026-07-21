"""Independent, evidence-backed question/document relevance cards."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

from app.legal_institutions import InstitutionMatcher
from app.legal_institutions.normalizer import normalize_polish, phrase_present


DOCUMENT_CARD_VERSION = "deterministic_document_card_v2"
_ARTICLE_RE = re.compile(r"\bart\.\s*(\d+[a-z]*)", re.IGNORECASE)
_ACT_RE = re.compile(r"\b(CIT|PIT|VAT|ORDYNACJA|AKCYZA)\b", re.IGNORECASE)
_PROVISION_RE = re.compile(r"\b(?:CIT|PIT|VAT)?\s*art\.\s*\d+[a-z]*(?:\s+ust\.\s*\d+[a-z]*)?(?:\s+pkt\s*\d+)?", re.IGNORECASE)
_HEADINGS_RE = re.compile(
    r"(?im)^\s*(Pytanie|Stanowisko wnioskodawcy|Ocena stanowiska|"
    r"Uzasadnienie|Podsumowanie|Reasumuj[aą]c|W konsekwencji|"
    r"Maj[aą]c na uwadze|Zatem|Tym samym)\s*[:\-]?\s*$"
)
_CARD_CACHE: "OrderedDict[str, DocumentCard]" = OrderedDict()
_CARD_CACHE_LIMIT = 2_000


def _values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _article(value: str) -> str:
    match = _ARTICLE_RE.search(normalize_polish(value).normalized)
    return match.group(1) if match else ""


def _act(value: str) -> str:
    match = _ACT_RE.search(str(value or ""))
    return match.group(1).upper() if match else ""


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _fold(value: str) -> str:
    """Accent-insensitive text for deterministic, non-dictionary rules."""

    decomposed = unicodedata.normalize("NFD", str(value or "")).replace("ł", "l").replace("Ł", "L")
    return "".join(character for character in decomposed if unicodedata.category(character) != "Mn").casefold()


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
    material_concepts: tuple[str, ...] = ()
    negative_concepts: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentEvidence:
    institution_id: str
    evidence_type: str
    value: str
    source: str = ""
    start: int = -1
    end: int = -1


@dataclass(frozen=True)
class DocumentSections:
    question_section: str = ""
    position_result: str = ""
    conclusion_paragraphs: tuple[str, ...] = ()
    keyword_windows: tuple[str, ...] = ()

    def compact_text(self, *, title: str, limit: int = 6_000) -> str:
        parts = [title, self.question_section, self.position_result, *self.conclusion_paragraphs, *self.keyword_windows]
        return "\n\n".join(part.strip() for part in parts if part.strip())[:limit]


@dataclass(frozen=True)
class DocumentCard:
    document_id: str
    signature: str = ""
    tax_domains: tuple[str, ...] = ()
    detected_institutions: tuple[str, ...] = ()
    detected_mechanisms: tuple[str, ...] = ()
    cited_provisions: tuple[str, ...] = ()
    taxpayer_roles: tuple[str, ...] = ()
    counterparty_roles: tuple[str, ...] = ()
    payment_direction: str | None = None
    payment_types: tuple[str, ...] = ()
    transaction_types: tuple[str, ...] = ()
    contract_types: tuple[str, ...] = ()
    material_concepts: tuple[str, ...] = ()
    evidence: tuple[DocumentEvidence, ...] = ()
    classification_source: str = DOCUMENT_CARD_VERSION

    def evidence_for(self, institution_id: str) -> tuple[DocumentEvidence, ...]:
        return tuple(item for item in self.evidence if item.institution_id == institution_id)

    def to_dict(self) -> dict[str, object]:
        evidence = [item.__dict__ for item in self.evidence]
        return {
            "document_id": self.document_id,
            "signature": self.signature,
            "tax_domains": list(self.tax_domains),
            "detected_institutions": list(self.detected_institutions),
            "detected_mechanisms": list(self.detected_mechanisms),
            "cited_provisions": list(self.cited_provisions),
            "taxpayer_roles": list(self.taxpayer_roles),
            "counterparty_roles": list(self.counterparty_roles),
            "payment_direction": self.payment_direction,
            "payment_types": list(self.payment_types),
            "transaction_types": list(self.transaction_types),
            "contract_types": list(self.contract_types),
            "material_concepts": list(self.material_concepts),
            "evidence": evidence,
            "evidence_spans": evidence,
            "classification_source": self.classification_source,
        }


@dataclass(frozen=True)
class DocumentValidation:
    passed: bool
    relation: str
    reject: bool
    reason: str
    matched_institutions: tuple[str, ...] = ()
    axes: Mapping[str, bool] | None = None


def build_question_card(*, question: str, issue: Any) -> QuestionCard:
    normalized = _fold(normalize_polish(question).normalized)
    is_wht = "podatek u zrodla" in normalized or "wht" in normalized
    is_saas = "saas" in normalized
    is_eula = "eula" in normalized
    payment_type = "saas_access_fee" if is_saas else ""
    transaction_type = "software_access" if any(value in normalized for value in ("saas", "program komputerowy", "oprogramowan")) else ""
    concepts = tuple(value for value in (
        "saas", "eula", "program komputerowy", "naleznosci licencyjne",
        "uzytkownik koncowy", "prawo do zwielokrotniania", "prawo do modyfikacji", "prawo do dystrybucji",
    ) if value in normalized)
    return QuestionCard(
        question_id=str(getattr(issue, "issue_id", "")),
        tax_domains=tuple(str(item).upper() for item in issue.tax_domains),
        locked_institutions=tuple(issue.locked_institution_ids),
        primary_mechanism=str(issue.legal_mechanism),
        secondary_mechanisms=("software_license_wht",) if is_wht and is_saas else (),
        provision_hints=tuple(issue.possible_provision_concepts),
        taxpayer_role="polish_payer_company" if is_wht and "polska spolka" in normalized else "",
        counterparty_role="foreign_service_provider" if is_wht and any(term in normalized for term in ("zagraniczn", "nierezydent")) else "",
        payment_direction="poland_to_foreign_recipient" if is_wht and any(term in normalized for term in ("zagraniczn", "nierezydent")) else "",
        payment_type=payment_type,
        transaction_type=transaction_type,
        contract_type="eula" if is_eula else "",
        material_concepts=concepts,
        negative_concepts=tuple(str(item) for item in getattr(issue, "negative_fact_constraints", ()) if str(item)),
    )


def extract_document_sections(text: str, *, keywords: tuple[str, ...] = ()) -> DocumentSections:
    """Return a bounded deterministic evidence pack from a complete document."""

    raw = str(text or "")
    headings = list(_HEADINGS_RE.finditer(raw))
    sections: dict[str, str] = {}
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
        sections[normalize_polish(match.group(1)).normalized] = raw[match.end():end].strip()[:2_000]
    conclusions: list[str] = []
    for marker in ("Reasumując", "W konsekwencji", "Mając na uwadze", "Zatem", "Tym samym"):
        for match in re.finditer(re.escape(marker), raw, re.IGNORECASE):
            conclusions.append(raw[match.start():match.start() + 900].strip())
    windows: list[str] = []
    for keyword in keywords[:24]:
        match = re.search(re.escape(keyword), raw, re.IGNORECASE)
        if match:
            windows.append(raw[max(0, match.start() - 350):match.end() + 650].strip())
    return DocumentSections(
        question_section=sections.get("pytanie", ""),
        position_result=sections.get("ocena stanowiska", "") or sections.get("stanowisko wnioskodawcy", ""),
        conclusion_paragraphs=_unique(conclusions)[:4],
        keyword_windows=_unique(windows)[:6],
    )


def _cache_key(candidate: Any, *, title: str, body: str, provisions: tuple[str, ...]) -> str:
    document_id = str(candidate.document_id or candidate.candidate_id)
    fingerprint = hashlib.sha256("\x1f".join((title, body, *provisions)).encode("utf-8")).hexdigest()[:16]
    return f"document_card:{document_id}:{DOCUMENT_CARD_VERSION}:{fingerprint}"


def _evidence(institution_id: str, evidence_type: str, value: str, source: str, raw: str) -> DocumentEvidence:
    match = re.search(re.escape(value), raw, re.IGNORECASE)
    return DocumentEvidence(
        institution_id=institution_id,
        evidence_type=evidence_type,
        value=value,
        source=source,
        start=match.start() if match else -1,
        end=match.end() if match else -1,
    )


def _has_strong_withholding_tax_evidence(
    *,
    title: str,
    scope: str,
    question_section: str,
    provisions: tuple[str, ...],
) -> bool:
    """Reject incidental WHT citations in otherwise unrelated documents.

    A long interpretation can quote art. 26 CIT or mention WHT in background.
    For document mode this is weaker than the query-mode lock: it needs either
    an editorial WHT subject or a payment-to-foreign-recipient context.
    """

    normalized_title = _fold(normalize_polish(title).normalized)
    normalized_scope = _fold(normalize_polish(scope).normalized)
    normalized_question = _fold(normalize_polish(question_section).normalized)
    title_wht = "podatek u zrodla" in normalized_title
    question_wht = "podatek u zrodla" in normalized_question
    section_wht = "podatek u zrodla" in normalized_scope
    has_foreign = any(term in normalized_scope for term in ("nierezydent", "zagraniczn", "upo"))
    has_payment = any(term in normalized_scope for term in ("odsetk", "naleznosci licencyjn", "wynagrodzen", "wyplac", "oplata", "uslug niematerialn"))
    has_payer = "platnik" in normalized_scope
    cited_articles = {_article(value) for value in provisions}
    has_core_provision = bool(cited_articles.intersection({"21", "22"}))
    has_payer_provision = "26" in cited_articles
    return (
        title_wht
        or question_wht
        or (section_wht and has_foreign and has_payment)
        or (has_core_provision and has_foreign and has_payment)
        or (has_payer_provision and has_payer and has_foreign)
    )


def build_document_card(candidate: Any, *, matcher: InstitutionMatcher) -> DocumentCard:
    """Classify solely from candidate document text and metadata, never a question."""

    metadata: Mapping[str, Any] = candidate.metadata
    title = str(metadata.get("subject") or metadata.get("title") or "")
    metadata_provisions = tuple(_values(metadata.get("legal_provisions") or metadata.get("provisions")))
    body = str(candidate.text or "")
    cache_key = _cache_key(candidate, title=title, body=body, provisions=metadata_provisions)
    cached = _CARD_CACHE.get(cache_key)
    if cached is not None:
        _CARD_CACHE.move_to_end(cache_key)
        return cached

    sections = extract_document_sections(body)
    scope = sections.compact_text(title=title)
    normalized_title = normalize_polish(title)
    normalized_scope = normalize_polish(scope)
    normalized_full = normalize_polish(" ".join((title, body, *metadata_provisions)))
    provisions = _unique([*metadata_provisions, *[match.group(0) for match in _PROVISION_RE.finditer(body)]])
    domains = tuple(str(item).upper() for item in _values(metadata.get("tax_domains")))
    if not domains:
        domains = tuple(value for value in ("CIT", "PIT", "VAT") if phrase_present(value, normalized_full))

    evidence: list[DocumentEvidence] = []
    institutions: list[str] = []
    mechanisms: list[str] = []
    for definition in matcher.dictionary.institutions:
        if definition.status != "active":
            continue
        found: list[DocumentEvidence] = []
        for phrase in (
            definition.canonical_name,
            *definition.exact_aliases,
            *definition.lemma_aliases,
            *definition.statutory_phrases,
        ):
            if phrase_present(phrase, normalized_title):
                found.append(_evidence(definition.institution_id, "phrase", phrase, "title", title))
            elif phrase_present(phrase, normalized_scope):
                found.append(_evidence(definition.institution_id, "phrase", phrase, "relevant_sections", scope))
        for hint in definition.provision_hints:
            article = _article(hint)
            hint_act = _act(hint)
            matching_provision = any(
                article
                and article == _article(provision)
                and (
                    not hint_act
                    or _act(provision) == hint_act
                    or (not _act(provision) and hint_act in domains)
                )
                for provision in provisions
            )
            exact_hint_act_present = not hint_act or bool(
                re.search(rf"\b{re.escape(hint_act)}\b", " ".join((title, body, *provisions)), re.IGNORECASE)
            )
            if (phrase_present(hint, normalized_full) and exact_hint_act_present) or (
                matching_provision
                and (not definition.tax_domains or set(domains).intersection(definition.tax_domains))
            ):
                found.append(_evidence(definition.institution_id, "provision", hint, "cited_provisions", " ".join(provisions)))
        if definition.institution_id == "withholding_tax" and not _has_strong_withholding_tax_evidence(
            title=title,
            scope=scope,
            question_section=sections.question_section,
            provisions=provisions,
        ):
            found = []
        if found:
            institutions.append(definition.institution_id)
            mechanisms.extend(definition.legal_mechanisms or (definition.institution_id,))
            evidence.extend(found)

    # These independent broad mechanisms are negative safeguards for shadow
    # entries; they are never copied from a question and do not grant a lock.
    if re.search(r"ulga mieszkaniow|sprzedaz.*nieruchom|lokal mieszkal", normalized_full.normalized):
        mechanisms.append("housing_relief")
    if re.search(r"rehabilitacyj|niepelnosprawn", normalized_full.normalized):
        mechanisms.append("rehabilitation_relief")
    if re.search(r"leasing.{0,60}(samoch|pojazd)|samochod.{0,60}leasing", normalized_full.normalized):
        mechanisms.append("vehicle_lease_cost")

    flat = _fold(normalized_full.normalized)
    payment_types = tuple(value for value in ("saas_access_fee", "interest", "royalty") if (
        (value == "saas_access_fee" and "saas" in flat)
        or (value == "interest" and "odsetk" in flat)
        or (value == "royalty" and "naleznosci licencyjn" in flat)
    ))
    transaction_types = tuple(value for value in ("software_access", "debt_to_capital", "foreign_tax_credit") if (
        (value == "software_access" and any(term in flat for term in ("saas", "oprogramowan", "program komputerowy")))
        or (value == "debt_to_capital" and re.search(r"konwersj.{0,40}(dlug|wierzyteln).{0,80}kapital", flat))
        or (value == "foreign_tax_credit" and re.search(r"odliczen.{0,40}zagraniczn.{0,40}podat", flat))
    ))
    taxpayer_roles = ("polish_payer_company",) if "platnik" in flat and "polsk" in flat else ()
    counterparty_roles = ("foreign_service_provider",) if any(term in flat for term in ("nierezydent", "zagraniczn")) else ()
    direction = "poland_to_foreign_recipient" if taxpayer_roles and counterparty_roles else None
    card = DocumentCard(
        document_id=str(candidate.document_id),
        signature=str(metadata.get("signature") or ""),
        tax_domains=_unique(list(domains)),
        detected_institutions=_unique(institutions),
        detected_mechanisms=_unique(mechanisms),
        cited_provisions=provisions,
        taxpayer_roles=taxpayer_roles,
        counterparty_roles=counterparty_roles,
        payment_direction=direction,
        payment_types=payment_types,
        transaction_types=transaction_types,
        contract_types=("eula",) if "eula" in flat else (),
        material_concepts=_unique([item.value for item in evidence if item.evidence_type == "phrase"]),
        evidence=tuple(dict.fromkeys(evidence)),
    )
    _CARD_CACHE[cache_key] = card
    _CARD_CACHE.move_to_end(cache_key)
    while len(_CARD_CACHE) > _CARD_CACHE_LIMIT:
        _CARD_CACHE.popitem(last=False)
    return card


def evaluate_document_relevance(question: QuestionCard, document: DocumentCard) -> DocumentValidation:
    required = set(question.locked_institutions)
    detected = set(document.detected_institutions)
    matched = tuple(sorted(required.intersection(detected)))
    same_domain = not question.tax_domains or not document.tax_domains or bool(set(question.tax_domains).intersection(document.tax_domains))
    same_mechanism = not question.primary_mechanism or question.primary_mechanism in document.detected_mechanisms
    question_articles = {_article(value) for value in question.provision_hints if _article(value)}
    document_articles = {_article(value) for value in document.cited_provisions if _article(value)}
    same_provision = not question_articles or bool(question_articles.intersection(document_articles))
    same_payment = not question.payment_type or question.payment_type in document.payment_types
    same_transaction = not question.transaction_type or question.transaction_type in document.transaction_types
    same_direction = not question.payment_direction or not document.payment_direction or question.payment_direction == document.payment_direction
    same_role = not question.taxpayer_role or not document.taxpayer_roles or question.taxpayer_role in document.taxpayer_roles
    axes = {
        "same_tax_domain": same_domain,
        "same_legal_mechanism": same_mechanism,
        "same_provision_family": same_provision,
        "same_taxpayer_role": same_role,
        "same_payment_direction": same_direction,
        "same_payment_type": same_payment,
        "same_transaction_type": same_transaction,
    }
    if required and not matched:
        return DocumentValidation(False, "irrelevant", True, "missing_document_institution_evidence", axes=axes)
    if required and matched and all((same_domain, same_mechanism, same_payment, same_transaction, same_direction, same_role)):
        return DocumentValidation(True, "direct", False, "document_institution_and_material_evidence", matched, axes)
    if required and matched:
        return DocumentValidation(False, "context_only", True, "document_mechanism_without_matching_transaction", matched, axes)
    return DocumentValidation(False, "irrelevant", True, "document_validation_failed", axes=axes)
