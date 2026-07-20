"""Deterministic, data-agnostic research signals for Polish tax authorities.

This module deliberately does not contain document identifiers or question
templates.  It turns a question into safe lexical anchors and a small legal
research hypothesis, then evaluates a candidate against that hypothesis.  The
same code is used by the isolated interpretation search and by V2 authority
retrieval, so both profiles expose comparable relevance diagnostics.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+", re.UNICODE)
PROVISION_RE = re.compile(
    r"\bart\.?\s*(\d+[a-z]{0,3})(?:\s*(?:ust\.?|§|pkt)\s*(\d+[a-z]{0,3}))?",
    re.IGNORECASE,
)

# These are linguistic or legal-boilerplate classes, not tax topics.  They
# prevent a generic prose form from becoming a precision query by itself.
STOPWORDS = frozenset(
    {
        "czy", "moze", "mogą", "moga", "jest", "byc", "być", "oraz", "albo",
        "jezeli", "jeśli", "dla", "przy", "bez", "w", "z", "na", "do", "od",
        "przedsiebiorca", "przedsiębiorca", "podatnik", "dzialalnosc", "działalność",
        "gospodarcza", "gospodarczego", "koszt", "koszty", "uzyskania", "przychodu",
        "wydatek", "wydatki", "zaliczyc", "stanowic", "podatkowy", "podatkowe",
        "firmy", "firma", "klienta", "klient", "zwiazku", "związku", "realizacja",
        "wylacznie", "wyłącznie", "innym", "jednoczesnie", "jednocześnie",
        "tam", "tutaj", "gdzie", "ktorym", "którym",
    }
)
FORBIDDEN_PREFIX_STEMS = frozenset(
    {
        # Locative forms are especially unsafe with a prefix FTS index:
        # "mieście" must never become "miesci*" and match "mieści się".
        "miesci", "miejs", "miejsc",
        # Short grammatical and legal-boilerplate stems are context, not facts.
        "koszt", "uzysk", "przychod", "podat", "dzialaln", "gospodarcz",
        "przedsiebiorc", "wydatk", "stanow", "moz", "moga", "ktor", "tych",
    }
)
CONTEXT_PREFIXES = (
    "przedsiebior", "podatni", "dzialal", "gospodar", "koszt", "uzysk", "przychod",
    "wydatk", "podatek", "zalicz", "stanow", "moz", "moga", "formal", "charakter",
)
TAX_INTENT_PREFIXES = ("koszt", "uzysk", "przychod", "podat", "pit", "cit", "vat", "wht")
TAX_RESEARCH_MECHANISMS = frozenset(
    {
        "business_expense",
        "business_accommodation_expense",
        "business_education_expense",
        "cash_payment_cost_exclusion",
        "mixed_use_vehicle_vat",
        "ksef_invoice_tax_evidence",
    }
)


def normalize(value: str) -> str:
    folded = value.casefold().translate(str.maketrans({"ł": "l"}))
    return "".join(
        character
        for character in unicodedata.normalize("NFD", folded)
        if unicodedata.category(character) != "Mn"
    )


def _stem(value: str) -> str:
    """Use only conservative inflection trimming; preserve locative endings."""

    for suffix in (
        "owego", "owych", "owymi", "owej", "owemu", "aniem", "enia", "aniu",
        "nego", "nych", "nymi", "ami", "ach", "ego", "emu", "ymi", "owa",
        "owe", "owy", "ych", "nia", "nie", "nym", "nej", "owi", "om",
        "a", "e", "i", "o", "u", "y",
    ):
        # Do not reduce *-cie*: it is a common Polish locative/verb boundary
        # and its shortened prefix collides with unrelated legal prose.
        if value.endswith(suffix) and len(value) - len(suffix) >= 5:
            return value[: -len(suffix)]
    return value


@dataclass(frozen=True)
class AnchorSet:
    high_precision: tuple[str, ...] = ()
    medium_precision: tuple[str, ...] = ()
    context_only: tuple[str, ...] = ()
    forbidden_prefix: tuple[str, ...] = ()

    @property
    def retrieval_terms(self) -> tuple[str, ...]:
        return (*self.high_precision, *self.medium_precision)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "high_precision": list(self.high_precision),
            "medium_precision": list(self.medium_precision),
            "context_only": list(self.context_only),
            "forbidden_prefix": list(self.forbidden_prefix),
        }


def build_anchors(question: str) -> AnchorSet:
    high: list[str] = []
    medium: list[str] = []
    context: list[str] = []
    forbidden: list[str] = []
    seen: set[str] = set()

    raw_tokens = TOKEN_RE.findall(question)
    for raw in raw_tokens:
        token = normalize(raw)
        identifier = bool(re.search(r"\d", raw)) or raw.isupper()
        if len(token) < 3 and not identifier:
            continue
        stem = _stem(token)
        if stem in seen:
            continue
        seen.add(stem)
        if stem in FORBIDDEN_PREFIX_STEMS or raw.casefold().endswith("ście"):
            forbidden.append(stem)
        elif token in STOPWORDS or stem.startswith(CONTEXT_PREFIXES):
            context.append(stem)
        elif identifier or len(stem) >= 5:
            high.append(stem)
        else:
            medium.append(stem)

    # Preserve legal identifiers which tokenisation may split (VAT-26, art. 22p).
    compact = normalize(re.sub(r"[^0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]", "", question))
    for value in re.findall(r"\b(?:vat|pit|cit)\s*-?\s*\d+[a-z]*\b", question, re.I):
        stem = normalize(re.sub(r"\W", "", value))
        if stem and stem not in seen:
            seen.add(stem)
            high.append(stem)
    if "vat26" in compact and "vat26" not in seen:
        high.append("vat26")

    return AnchorSet(tuple(high), tuple(medium), tuple(context), tuple(forbidden))


@dataclass(frozen=True)
class ResearchUnderstanding:
    tax_domain: str = ""
    legal_mechanism: str = "business_expense"
    candidate_provisions: tuple[str, ...] = ()
    material_concepts: tuple[str, ...] = ()
    negative_concepts: tuple[str, ...] = ()
    anchors: AnchorSet = field(default_factory=AnchorSet)

    def to_dict(self) -> dict[str, object]:
        return {
            "tax_domain": self.tax_domain,
            "legal_mechanism": self.legal_mechanism,
            "candidate_provisions": list(self.candidate_provisions),
            "material_concepts": list(self.material_concepts),
            "negative_concepts": list(self.negative_concepts),
            "anchors": self.anchors.to_dict(),
        }


def research_understanding_from_fields(
    *,
    tax_domain: str,
    legal_mechanism: str,
    candidate_provisions: Sequence[str],
    material_concepts: Sequence[str],
    negative_concepts: Sequence[str] = (),
) -> ResearchUnderstanding:
    """Adapt a V2 issue without allowing a second, divergent taxonomy."""

    concepts = tuple(dict.fromkeys(normalize(item) for item in material_concepts if str(item).strip()))
    anchors = AnchorSet(high_precision=concepts)
    return ResearchUnderstanding(
        tax_domain=str(tax_domain or "").upper(),
        legal_mechanism=str(legal_mechanism or "business_expense"),
        candidate_provisions=tuple(str(item) for item in candidate_provisions if str(item).strip()),
        material_concepts=concepts,
        negative_concepts=tuple(str(item) for item in negative_concepts if str(item).strip()),
        anchors=anchors,
    )


def understand_tax_research_question(question: str) -> ResearchUnderstanding:
    """Build a bounded research hypothesis from facts, not document IDs.

    The rules identify statutory mechanisms, not scenarios.  They are applied
    to any wording containing the relevant factual/legal signals and remain a
    retriever hint, never a legal conclusion.
    """

    normalized = normalize(question)
    anchors = build_anchors(question)
    tax_domain = "VAT" if re.search(r"\bvat\b|odlicz", normalized) else "PIT"
    mechanism = "business_expense"
    provisions: list[str] = ["PIT art. 22 ust. 1"] if tax_domain == "PIT" else []
    negatives: list[str] = []

    if re.search(r"gotowk|rachunk|platnosc.{0,40}(?:15[,. ]?000|18000)|rat", normalized):
        mechanism = "cash_payment_cost_exclusion"
        tax_domain = "PIT"
        provisions = ["PIT art. 22p"]
    elif re.search(r"(?:50\s*%|polow).{0,80}(?:vat|odlicz)|vat.{0,80}(?:paliw|samochod|pojazd)|vat26", normalized):
        mechanism = "mixed_use_vehicle_vat"
        tax_domain = "VAT"
        provisions = ["VAT art. 86a"]
    elif re.search(r"wynaj|najem|zakwaterow", normalized) and re.search(r"mieszkan|lokal|hotel", normalized):
        mechanism = "business_accommodation_expense"
    elif re.search(r"studi|kurs|szkolen|kwalifik|jezyk", normalized):
        mechanism = "business_education_expense"
    elif "ksef" in normalized:
        mechanism = "ksef_invoice_tax_evidence"

    if mechanism.endswith("expense") or mechanism == "business_expense":
        # These are mutually exclusive tax mechanisms in the corpus.  They are
        # used only as penalties after a candidate itself declares them.
        negatives.extend(("ulga rehabilitacyjna", "ulga na powrót", "rezydencja podatkowa"))
    if tax_domain == "VAT":
        negatives.extend(("ulga na powrót", "rezydencja podatkowa"))

    return ResearchUnderstanding(
        tax_domain=tax_domain,
        legal_mechanism=mechanism,
        candidate_provisions=tuple(provisions),
        material_concepts=anchors.retrieval_terms,
        negative_concepts=tuple(dict.fromkeys(negatives)),
        anchors=anchors,
    )


def candidate_boolean_queries(understanding: ResearchUnderstanding, *, max_queries: int = 8) -> list[str]:
    """Build stable, selective Boolean FTS probes from safe user concepts."""

    high_terms = list(understanding.anchors.high_precision)
    # A short listed object (for example "pies") can be decisive only beside
    # a concrete action.  It remains medium precision, but may participate in
    # a two-term query; it never becomes a solo precision anchor.
    medium_terms = [
        value for value in understanding.anchors.medium_precision if value not in high_terms
    ]
    terms = [*high_terms, *medium_terms]
    provision_budget = len(understanding.candidate_provisions)
    bridge_budget = min(2, len(terms))
    pair_budget = max(1, max_queries - provision_budget - bridge_budget)
    pairs: list[tuple[str, str]] = []
    if terms:
        # A short concrete object ("pies") is useful beside the leading
        # action ("zakup"), but never alone.  Reserve these pairs before the
        # less selective list of other modifiers.
        pairs.extend((terms[0], right) for right in medium_terms)
        pairs.extend((terms[0], right) for right in high_terms[1:])
    if len(pairs) < pair_budget:
        pairs.extend(zip(terms[1:], terms[2:]))
    pairs = pairs[:pair_budget]
    queries = [f"+{left}* +{right}*" for left, right in pairs if left and right]
    # A generic tax-outcome term can be a *second* required word.  It never
    # becomes an anchor on its own, but safely bridges ordinary inflections
    # such as "wynajem" in a question and "najmu" in an editorial subject.
    for term in terms[:bridge_budget]:
        queries.append(f"+{term}* +koszt*")
    for provision in understanding.candidate_provisions:
        article = next((match.group(1) for match in PROVISION_RE.finditer(normalize(provision))), "")
        domain = normalize(provision.split()[0]) if provision.split() else ""
        if domain and article:
            queries.append(f"+{domain}* +{article}*")
    return list(dict.fromkeys(query for query in queries if query))[:max_queries]


def _tokens(value: str) -> set[str]:
    return {normalize(token) for token in TOKEN_RE.findall(value) if len(normalize(token)) >= 3}


def _text_matches(terms: Sequence[str], text: str) -> tuple[int, float]:
    return _token_matches(terms, _tokens(text))


def _token_matches(terms: Sequence[str], text_tokens: set[str]) -> tuple[int, float]:
    if not terms:
        return 0, 0.0
    matched = sum(1 for term in terms if term in text_tokens or any(token.startswith(term) for token in text_tokens))
    return matched, matched / len(terms)


def _document_mechanism(text: str, provisions: Iterable[str]) -> str:
    value = normalize(" ".join((text, *provisions)))
    if re.search(r"ulg\w{0,4}\s+na\s+powrot", value) or "rezyden" in value:
        return "return_relief_or_residency"
    if re.search(r"ulg\w{0,4}\s+rehabilitacyj", value):
        return "rehabilitation_relief"
    if re.search(r"(?:zwolnien|ulg\w{0,4}).{0,120}(?:cel.{0,20}mieszkan|mieszkan|odplatn.{0,30}zbyci|sprzedaz.{0,30}nieruchom)", value) or re.search(r"(?:odplatn.{0,30}zbyci|sprzedaz.{0,30}nieruchom)", value):
        return "housing_relief"
    if re.search(r"(?:najm|wynaj).{0,60}(?:prywatn|ryczalt|opodatkow)|(?:prywatn|ryczalt|opodatkow).{0,60}(?:najm|wynaj)", value):
        return "rental_income_relief"
    if re.search(r"amortyz|srod.{0,12}trwal", value):
        return "fixed_asset_or_depreciation"
    if re.search(r"art\.?\s*86a|odlicz.{0,50}(?:paliw|samochod|pojazd)", value):
        return "mixed_use_vehicle_vat"
    if re.search(r"art\.?\s*22p|platnosc.{0,80}(?:rachunk|gotowk)|gotowk.{0,80}platn", value):
        return "cash_payment_cost_exclusion"
    if "ksef" in value:
        return "ksef_invoice_tax_evidence"
    if re.search(r"wynaj|najm|zakwaterow", value) and re.search(r"mieszkan|lokal|hotel", value) and re.search(r"koszt|wydatek|dzialaln|firm\w*|kontrakt|sluzbow", value):
        return "business_accommodation_expense"
    if re.search(r"studi|kurs|szkolen|kwalifik|jezyk", value):
        return "business_education_expense"
    if re.search(r"koszt.{0,80}(?:uzyskan|przychod)|zaliczen.{0,80}koszt", value):
        return "business_expense"
    return "unknown"


def _provision_specificity(provisions: Sequence[str]) -> float:
    return 1.0 if any(re.search(r"art\.\s*(?:22p|86a|23\s+ust\.\s*1\s+pkt)", value, re.I) for value in provisions) else 0.45


@dataclass(frozen=True)
class CandidateAssessment:
    relation: str
    reject: bool
    reason: str
    document_mechanism: str
    material_differences: tuple[str, ...]
    score: float
    components: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "relation": self.relation,
            "reject": self.reject,
            "reason": self.reason,
            "document_mechanism": self.document_mechanism,
            "material_differences": list(self.material_differences),
            "score": self.score,
            "components": dict(self.components),
        }


def assess_candidate(
    understanding: ResearchUnderstanding,
    *,
    subject: str,
    text: str,
    provisions: Sequence[str] = (),
    tax_domain: str = "",
) -> CandidateAssessment:
    """Classify a document and calculate a bounded 0--100 research score."""

    title_tokens = _tokens(subject)
    body_tokens = _tokens(text)
    title_matches, title_ratio = _token_matches(understanding.material_concepts, title_tokens)
    body_matches, body_ratio = _token_matches(understanding.material_concepts, body_tokens)
    material_matches, _ = _token_matches(understanding.material_concepts, title_tokens | body_tokens)
    # A complete interpretation contains quotations, examples and boilerplate.
    # Its editor-supplied subject is therefore the primary mechanism signal;
    # full text can confirm an otherwise generic subject but cannot turn a
    # word collision in a different title into a direct match.
    title_mechanism = _document_mechanism(subject, provisions)
    body_mechanism = _document_mechanism(text, provisions)
    document_mechanism = title_mechanism if title_mechanism != "unknown" else body_mechanism
    expected = understanding.legal_mechanism
    same_mechanism = document_mechanism == expected or (
        expected == "business_expense" and document_mechanism in {
            "business_accommodation_expense", "business_education_expense", "business_expense"
        }
    )
    title_confirms_mechanism = title_mechanism == expected or (
        expected == "business_expense" and title_mechanism in {
            "business_accommodation_expense", "business_education_expense", "business_expense"
        }
    )
    orthogonal = document_mechanism in {
        "return_relief_or_residency", "rehabilitation_relief", "housing_relief", "rental_income_relief",
        "fixed_asset_or_depreciation",
    } and expected not in {document_mechanism, "unknown"}
    provision_text = normalize(" ".join(provisions) + " " + text)
    if not tax_domain:
        if re.search(r"\bvat\b|art\.?\s*86a", provision_text):
            tax_domain = "VAT"
        elif re.search(r"\bpit\b|art\.?\s*22(?:p|\b)", provision_text):
            tax_domain = "PIT"
        elif "cit" in provision_text:
            tax_domain = "CIT"
    provision_matches = sum(
        normalize(match.group(1)) in provision_text
        for provision in understanding.candidate_provisions
        for match in PROVISION_RE.finditer(normalize(provision))
    )
    provision_score = min(20.0, 20.0 * provision_matches / max(1, len(understanding.candidate_provisions)))
    if provision_matches:
        provision_score *= _provision_specificity(understanding.candidate_provisions)
    tax_score = 12.0 if not tax_domain or not understanding.tax_domain or tax_domain.upper() == understanding.tax_domain else 0.0
    mechanism_score = (
        25.0 if title_confirms_mechanism
        else 12.0 if same_mechanism
        else 7.0 if document_mechanism == "unknown"
        else 0.0
    )
    transaction_score = 18.0 * body_ratio
    title_score = 10.0 * title_ratio
    body_score = 15.0 * body_ratio
    penalty = -55.0 if orthogonal else (-18.0 if document_mechanism not in {expected, "unknown", "business_expense"} else 0.0)
    score = max(0.0, min(100.0, title_score + body_score + provision_score + mechanism_score + transaction_score + tax_score + penalty))

    differences: list[str] = []
    if orthogonal:
        differences.append("wrong_legal_mechanism")
    if tax_score == 0.0:
        differences.append("different_tax_domain")
    if material_matches == 0:
        differences.append("different_expense_or_transaction")
    if tax_score == 0.0 or orthogonal or (not same_mechanism and material_matches < 2):
        relation, reject, reason = "different_mechanism", True, "Brak zgodności mechanizmu prawnego."
    elif same_mechanism and not title_confirms_mechanism:
        relation, reject, reason = "context_only", False, "Mechanizm wynika tylko z pełnej treści, a nie z przedmiotu interpretacji."
    elif same_mechanism and material_matches >= 2:
        relation, reject, reason = "direct", False, "Zgodny mechanizm i materialne elementy stanu faktycznego."
    elif same_mechanism and material_matches:
        relation, reject, reason = (
            ("strong_analogy", False, "Zgodny mechanizm i wspólny przedmiot wydatku w tytule dokumentu.")
            if title_matches
            else ("context_only", False, "Zgodny mechanizm, ale zbyt mało wspólnych elementów stanu faktycznego.")
        )
    elif expected == "business_expense" and material_matches >= 2:
        relation, reject, reason = "strong_analogy", False, "Zbliżony mechanizm lub rodzaj wydatku."
    elif body_matches:
        relation, reject, reason = "context_only", False, "Wspólny kontekst, bez pełnej zgodności mechanizmu."
    else:
        relation, reject, reason = "irrelevant", True, "Brak materialnej zbieżności z pytaniem."
    return CandidateAssessment(
        relation=relation,
        reject=reject,
        reason=reason,
        document_mechanism=document_mechanism,
        material_differences=tuple(differences),
        score=round(score, 2),
        components={
            "title_match": round(title_score, 2),
            "body_match": round(body_score, 2),
            "provision_match": round(provision_score, 2),
            "mechanism_match": round(mechanism_score, 2),
            "expense_or_transaction_match": round(transaction_score, 2),
            "tax_domain_match": round(tax_score, 2),
            "negative_match_penalty": round(penalty, 2),
        },
    )
