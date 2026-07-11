from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Iterable, Literal, Optional


ClaimStatus = Literal[
    "supported",
    "approved",
    "conditional_missing_fact",
    "blocked",
    "blocked_missing_provision_reference",
]
ProvisionStatus = Literal["active", "repealed", "unknown"]
RuleRelationship = Literal["general_rule", "special_extension", "exception", "notwithstanding", "peer"]


class TemporalConflictError(RuntimeError):
    pass


def _date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    candidate = value.strip()[:10]
    try:
        return date.fromisoformat(candidate)
    except ValueError:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None


def normalize_reference(value: str) -> str:
    normalized = " ".join(value.lower().replace("artykuł", "art.").split())
    normalized = re.sub(r"\bart\s*\.?", "art.", normalized)
    normalized = re.sub(r"\s*,\s*", " ", normalized)
    return normalized.strip(" .;:,")


@dataclass(frozen=True)
class LegalDocumentVersion:
    document_id: str
    version_id: str
    document_type: str
    title: str
    citation: str
    jurisdiction: str
    effective_from: str
    effective_to: Optional[str] = None
    publication_date: Optional[str] = None
    is_consolidated_text: bool = False

    def is_effective_on(self, target_date: date) -> bool:
        start = _date(self.effective_from)
        end = _date(self.effective_to)
        return start is not None and start <= target_date and (end is None or target_date <= end)


@dataclass(frozen=True)
class ProvisionRecord:
    provision_id: str
    document_id: str
    version_id: str
    citation: str
    article: str
    paragraph: Optional[str]
    point: Optional[str]
    letter: Optional[str]
    text: str
    effective_from: str
    effective_to: Optional[str]
    status: ProvisionStatus
    source_document_id: str
    source_chunk_ids: tuple[str, ...]
    source_span_start: int = 0
    source_span_end: int = 0
    references: tuple[str, ...] = ()
    amends: Optional[str] = None
    repealed_by: Optional[str] = None
    display_reference: str = ""
    tax_domain: str = ""
    taxpayer_role: str = ""
    legal_mechanism: str = ""
    entailed_result_codes: tuple[str, ...] = ()
    rule_relationship: RuleRelationship = "peer"
    related_provisions: tuple[str, ...] = ()
    special_rule_provisions: tuple[str, ...] = ()
    exception_provisions: tuple[str, ...] = ()
    general_rule_provisions: tuple[str, ...] = ()

    def is_effective_on(self, target_date: date) -> bool:
        start = _date(self.effective_from)
        end = _date(self.effective_to)
        return (
            self.status == "active"
            and start is not None
            and start <= target_date
            and (end is None or target_date <= end)
        )


class ProvisionRegistry:
    """In-memory exact-lookup view; persistence is provided by the RAG database."""

    def __init__(
        self,
        documents: Iterable[LegalDocumentVersion] = (),
        provisions: Iterable[ProvisionRecord] = (),
    ) -> None:
        self._documents = {(item.document_id, item.version_id): item for item in documents}
        self._provisions: dict[str, list[ProvisionRecord]] = {}
        self._by_reference: dict[tuple[str, str], list[ProvisionRecord]] = {}
        self._by_article: dict[tuple[str, str], list[ProvisionRecord]] = {}
        for item in provisions:
            self._provisions.setdefault(item.provision_id, []).append(item)
            self._by_reference.setdefault(
                (item.document_id, normalize_reference(item.citation)), []
            ).append(item)
            self._by_article.setdefault((item.document_id, item.article), []).append(item)

    @property
    def provisions(self) -> tuple[ProvisionRecord, ...]:
        return tuple(
            item for versions in self._provisions.values() for item in versions
        )

    def get(self, provision_id: str, target_date: str) -> Optional[ProvisionRecord]:
        parsed = _date(target_date)
        if parsed is None:
            return None
        effective = [
            item
            for item in self._provisions.get(provision_id, [])
            if item.is_effective_on(parsed)
        ]
        if len(effective) > 1:
            raise TemporalConflictError(
                f"Multiple active versions for {provision_id} on {target_date}"
            )
        return effective[0] if effective else None

    def exact_lookup(
        self, document_id: str, citation: str, target_date: str
    ) -> Optional[ProvisionRecord]:
        parsed = _date(target_date)
        if parsed is None:
            return None
        candidates = self._by_reference.get(
            (document_id, normalize_reference(citation)), []
        )
        effective = [item for item in candidates if item.is_effective_on(parsed)]
        if len(effective) > 1:
            raise TemporalConflictError(
                f"Multiple active versions for {document_id} {citation} on {target_date}"
            )
        effective.sort(key=lambda item: item.effective_from, reverse=True)
        return effective[0] if effective else None

    def validate(self) -> dict[str, int]:
        return {
            "provisions_without_version": sum(
                1 for item in self.provisions if not item.version_id
            ),
            "provisions_without_effective_dates": sum(
                1 for item in self.provisions if not _date(item.effective_from)
            ),
        }

    def resolve_applicable_provisions(
        self,
        provision_ids: Iterable[str],
        target_date: str,
    ) -> tuple[ProvisionRecord, ...]:
        resolved: list[ProvisionRecord] = []
        seen: set[str] = set()
        queue = [item for item in provision_ids if item]
        while queue:
            provision_id = queue.pop(0)
            if provision_id in seen:
                continue
            seen.add(provision_id)
            record = self.get(provision_id, target_date)
            if record is None:
                continue
            resolved.append(record)
            queue.extend(self._neighboring_special_rules(record, target_date))
            queue.extend(record.related_provisions)
            queue.extend(record.special_rule_provisions)
            queue.extend(record.exception_provisions)
            queue.extend(record.general_rule_provisions)
        return tuple(resolved)

    def _neighboring_special_rules(
        self,
        record: ProvisionRecord,
        target_date: str,
    ) -> tuple[str, ...]:
        if not record.paragraph:
            return ()
        article_records = self._by_article.get((record.document_id, record.article), [])
        next_key = _next_paragraph_key(record.paragraph)
        if next_key is None:
            return ()
        neighbors = [
            item.provision_id
            for item in article_records
            if item.paragraph == next_key and self.get(item.provision_id, target_date) is not None
        ]
        return tuple(neighbors)


@dataclass(frozen=True)
class SourceRequirementPlan:
    axis_id: str
    target_date: str
    mandatory_documents: tuple[str, ...]
    expected_mechanisms: tuple[str, ...]
    optional_source_types: tuple[str, ...] = ("interpretation", "judgment")


@dataclass(frozen=True)
class FactRecord:
    fact_id: str
    fact_type: str
    value: object
    status: Literal["known", "missing", "conflicting"] = "known"
    source: str = "user_question"
    confidence: float = 1.0
    date: Optional[str] = None
    subject_role: str = "case"


def build_bad_debt_status_facts() -> dict[str, FactRecord]:
    """Keep registration and insolvency as independent, never inferred facts."""
    return {
        "debtor_vat_registration_status": FactRecord(
            fact_id="debtor_vat_registration_status",
            fact_type="vat_registration_status",
            value=None,
            status="missing",
            subject_role="debtor",
        ),
        "debtor_insolvency_status": FactRecord(
            fact_id="debtor_insolvency_status",
            fact_type="insolvency_or_restructuring_status",
            value=None,
            status="missing",
            subject_role="debtor",
        ),
    }


@dataclass(frozen=True)
class RuleFactBinding:
    rule_id: str
    tax_axis: str
    provision_ids: tuple[str, ...]
    required_fact_ids: tuple[str, ...]
    effective_from: str
    effective_to: Optional[str] = None

    def is_effective_on(self, target_date: str) -> bool:
        parsed = _date(target_date)
        start = _date(self.effective_from)
        end = _date(self.effective_to)
        return (
            parsed is not None
            and start is not None
            and start <= parsed
            and (end is None or parsed <= end)
        )


def build_bad_debt_rule_bindings(target_date: str) -> tuple[RuleFactBinding, ...]:
    rules = (
        RuleFactBinding(
            rule_id="vat_bad_debt_historical_debtor_status",
            tax_axis="VAT",
            provision_ids=(
                "vat_art_89a_ust_2_pkt_1",
                "vat_art_89a_ust_2_pkt_2",
                "vat_art_89a_ust_2_pkt_3_lit_b",
            ),
            required_fact_ids=("debtor_insolvency_status",),
            effective_from="2013-01-01",
            effective_to="2021-09-30",
        ),
        RuleFactBinding(
            rule_id="vat_bad_debt_current_creditor_status_date",
            tax_axis="VAT",
            provision_ids=("vat_art_89a_ust_2_pkt_3_lit_a",),
            required_fact_ids=("creditor_vat_registration_status",),
            effective_from="2021-10-01",
        ),
        RuleFactBinding(
            rule_id="vat_bad_debt_current_payment_cutoff",
            tax_axis="VAT",
            provision_ids=("vat_art_89a_ust_3",),
            required_fact_ids=("receivable_payment_status_through_return_filing",),
            effective_from="2021-10-01",
        ),
        RuleFactBinding(
            rule_id="vat_bad_debt_current_debtor_registration_path",
            tax_axis="VAT",
            provision_ids=("vat_art_89a_ust_2", "vat_art_89a_ust_2a"),
            required_fact_ids=("debtor_vat_registration_status",),
            effective_from="2021-10-01",
        ),
        RuleFactBinding(
            rule_id="cit_bad_debt_payment_cutoff",
            tax_axis="CIT",
            provision_ids=("cit_art_18f_ust_5",),
            required_fact_ids=("receivable_payment_status_on_return_filing_date",),
            effective_from="2020-01-01",
        ),
        RuleFactBinding(
            rule_id="cit_bad_debt_debtor_status",
            tax_axis="CIT",
            provision_ids=("cit_art_18f_ust_10",),
            required_fact_ids=("debtor_insolvency_status",),
            effective_from="2020-01-01",
        ),
    )
    return tuple(rule for rule in rules if rule.is_effective_on(target_date))


@dataclass(frozen=True)
class CalculationRecord:
    calculation_id: str
    operation: str
    inputs: dict[str, object]
    result: object


@dataclass(frozen=True)
class LegalClaim:
    claim_id: str
    axis_id: str
    claim_type: str
    text: str
    source_provisions: tuple[str, ...]
    fact_dependencies: tuple[str, ...] = ()
    missing_fact_dependencies: tuple[str, ...] = ()
    calculation_id: Optional[str] = None
    calculation_ids: tuple[str, ...] = ()
    status: ClaimStatus = "supported"
    is_material: bool = True
    result: dict[str, object] = field(default_factory=dict)
    controlling_provisions: tuple[str, ...] = ()
    dependency_provisions: tuple[str, ...] = ()
    version_id: str = ""
    result_code: str = ""
    taxpayer_role: str = ""
    legal_mechanism: str = ""
    provenance: tuple[dict[str, object], ...] = ()
    fact_subject_roles: dict[str, str] = field(default_factory=dict)
    supporting_authorities: tuple[dict[str, object], ...] = ()
    contrary_authorities: tuple[dict[str, object], ...] = ()
    historical_authorities: tuple[dict[str, object], ...] = ()
    authority_confidence: float = 0.0


@dataclass(frozen=True)
class ClaimValidation:
    claim_id: str
    claim_supported: bool
    temporal_match: bool
    facts_satisfy_conditions: bool
    calculation_bound: bool
    applicable_provisions: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


MANDATORY_MECHANISM_SOURCE_BUNDLES: dict[str, tuple[str, ...]] = {
    "housing_relief_credit_repayment": (
        "pit_art_21_ust_25_pkt_2",
        "pit_art_21_ust_30",
        "pit_art_21_ust_30a",
    ),
}


RULE_RELATIONSHIP_PRECEDENCE: dict[RuleRelationship, int] = {
    "general_rule": 0,
    "peer": 1,
    "special_extension": 3,
    "exception": 4,
    "notwithstanding": 5,
}


def _paragraph_sort_key(value: Optional[str]) -> tuple[int, str]:
    if not value:
        return (-1, "")
    match = re.fullmatch(r"(\d+)([a-z]?)", value.strip().lower())
    if not match:
        return (-1, value.strip().lower())
    return (int(match.group(1)), match.group(2))


def _next_paragraph_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    match = re.fullmatch(r"(\d+)([a-z]?)", value.strip().lower())
    if not match:
        return None
    number = match.group(1)
    suffix = match.group(2)
    if not suffix:
        return f"{number}a"
    if suffix == "z":
        return str(int(number) + 1)
    return f"{number}{chr(ord(suffix) + 1)}"


def _specificity_score(record: ProvisionRecord) -> tuple[int, int, int, int]:
    return (
        RULE_RELATIONSHIP_PRECEDENCE.get(record.rule_relationship, 1),
        1 if record.letter else 0,
        1 if record.point else 0,
        1 if record.paragraph else 0,
    )


def _provisions_conflict(
    left: ProvisionRecord,
    right: ProvisionRecord,
) -> bool:
    if left.document_id != right.document_id:
        return False
    if left.article == right.article:
        return True
    linked = {
        *left.related_provisions,
        *left.special_rule_provisions,
        *left.exception_provisions,
        *left.general_rule_provisions,
    }
    reverse_linked = {
        *right.related_provisions,
        *right.special_rule_provisions,
        *right.exception_provisions,
        *right.general_rule_provisions,
    }
    return right.provision_id in linked or left.provision_id in reverse_linked


def validate_claim(
    claim: LegalClaim,
    registry: ProvisionRegistry,
    *,
    target_date: str,
    facts: dict[str, FactRecord],
    calculations: dict[str, CalculationRecord],
) -> ClaimValidation:
    errors: list[str] = []
    controlling_ids = claim.controlling_provisions or claim.source_provisions
    applicable = registry.resolve_applicable_provisions(
        (*controlling_ids, *claim.dependency_provisions),
        target_date,
    )
    applicable_by_id = {item.provision_id: item for item in applicable}
    resolved = [applicable_by_id.get(provision_id) for provision_id in controlling_ids]
    temporal_match = bool(resolved) and all(item is not None for item in resolved)
    if claim.is_material and not controlling_ids:
        errors.append("material_claim_without_source")
    if controlling_ids and not temporal_match:
        errors.append("provision_missing_or_not_effective")
    if any(item is not None and not item.display_reference for item in resolved):
        errors.append("missing_provision_display_reference")
    claim_domain = claim.axis_id.split("_", 1)[0].upper()
    if any(
        item is not None
        and item.tax_domain
        and claim_domain in {"VAT", "CIT", "PIT", "PCC"}
        and item.tax_domain.upper() != claim_domain
        for item in resolved
    ):
        errors.append("source_tax_domain_mismatch")
    if any(
        item is not None
        and item.taxpayer_role
        and claim.taxpayer_role
        and item.taxpayer_role != claim.taxpayer_role
        for item in resolved
    ):
        errors.append("source_taxpayer_role_mismatch")
    if any(
        item is not None
        and item.legal_mechanism
        and claim.legal_mechanism
        and item.legal_mechanism != claim.legal_mechanism
        for item in applicable
    ):
        errors.append("source_mechanism_mismatch")
    if claim.result_code:
        supporting = [
            item
            for item in applicable
            if item.entailed_result_codes
            and claim.result_code in item.entailed_result_codes
        ]
        all_mismatching = [
            item
            for item in applicable
            if item.entailed_result_codes
            and claim.result_code not in item.entailed_result_codes
        ]
        conflicting = [
            item
            for item in all_mismatching
            if any(
                _provisions_conflict(item, supported_item)
                for supported_item in supporting
            )
        ]
        if all_mismatching and not supporting:
            errors.append("source_does_not_entail_claim")
        elif supporting and conflicting:
            best_support = max(supporting, key=_specificity_score)
            best_conflict = max(conflicting, key=_specificity_score)
            if _specificity_score(best_support) <= _specificity_score(best_conflict):
                errors.append("unresolved_rule_conflict")
        if (
            claim.legal_mechanism == "housing_relief_credit_repayment"
            and claim.result_code == "credit_on_sold_property_disqualified"
            and "pit_art_21_ust_30a" in applicable_by_id
        ):
            errors.append("credit_repayment_disqualification_blocked")
    if claim.legal_mechanism in MANDATORY_MECHANISM_SOURCE_BUNDLES:
        required_bundle = set(MANDATORY_MECHANISM_SOURCE_BUNDLES[claim.legal_mechanism])
        if not required_bundle.issubset(applicable_by_id):
            errors.append("incomplete_mechanism_source_bundle")
    dependency_only = (
        bool(claim.dependency_provisions)
        and not claim.controlling_provisions
        and not claim.source_provisions
    )
    if dependency_only:
        errors.append("dependency_used_as_controlling_source")

    missing_facts = [
        fact_id
        for fact_id in claim.fact_dependencies
        if fact_id not in facts or facts[fact_id].status != "known"
    ]
    if missing_facts:
        errors.append("missing_fact_dependency")
    subject_role_mismatches = [
        fact_id
        for fact_id, expected_role in claim.fact_subject_roles.items()
        if (
            fact_id in facts
            and expected_role
            and facts[fact_id].subject_role
            and facts[fact_id].subject_role != expected_role
        )
    ]
    if subject_role_mismatches:
        errors.append("subject_role_mismatch")

    numeric = bool(re.search(r"\b\d+(?:[.,]\d+)?(?:\s*%|\s*zł)?\b", claim.text))
    calculation_bound = (
        not numeric
        or bool(claim.calculation_id and claim.calculation_id in calculations)
        or bool(
            claim.calculation_ids
            and all(item in calculations for item in claim.calculation_ids)
        )
        or any(fact_id in facts for fact_id in claim.fact_dependencies)
    )
    if numeric and not calculation_bound:
        errors.append("numeric_claim_without_calculation_or_fact")

    return ClaimValidation(
        claim_id=claim.claim_id,
        claim_supported=not errors and claim.status in {"supported", "approved"},
        temporal_match=temporal_match,
        facts_satisfy_conditions=not missing_facts,
        calculation_bound=calculation_bound,
        applicable_provisions=tuple(sorted(applicable_by_id)),
        errors=tuple(errors),
    )


def claim_to_dict(claim: LegalClaim) -> dict[str, object]:
    return asdict(claim)


@dataclass(frozen=True)
class AnswerSection:
    section_id: str
    title: str
    required_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerPlan:
    sections: tuple[AnswerSection, ...]
    allowed_claim_ids: tuple[str, ...]

    def validate_claim_set(self, claims: Iterable[LegalClaim]) -> list[str]:
        available = {claim.claim_id for claim in claims if claim.status != "blocked"}
        required = {
            claim_id
            for section in self.sections
            for claim_id in section.required_claim_ids
        }
        return sorted(required - available)


@dataclass
class PipelineTrace:
    target_date: str
    source_requirements: list[dict[str, object]] = field(default_factory=list)
    provisions: list[dict[str, object]] = field(default_factory=list)
    claims: list[dict[str, object]] = field(default_factory=list)
    claim_validation: list[dict[str, object]] = field(default_factory=list)
    registry_validation: dict[str, int] = field(default_factory=dict)


def build_registry_from_rules(
    rules: Iterable[dict[str, object]],
) -> ProvisionRegistry:
    documents: dict[tuple[str, str], LegalDocumentVersion] = {}
    provisions: list[ProvisionRecord] = []
    for rule in rules:
        document_id = str(rule.get("source_id") or "").strip()
        provision_id = str(rule.get("provision_id") or "").strip()
        citation = str(rule.get("citation") or "").strip()
        effective_from = str(
            rule.get("effective_from")
            or rule.get("legal_state_date")
            or rule.get("publication")
            or ""
        ).strip()[:10]
        if not document_id or not provision_id or not citation:
            continue
        version_id = f"{document_id}@{effective_from or 'undated'}"
        status: ProvisionStatus = (
            "repealed"
            if str(rule.get("rule_type") or "") == "repealed"
            else ("active" if _date(effective_from) else "unknown")
        )
        documents[(document_id, version_id)] = LegalDocumentVersion(
            document_id=document_id,
            version_id=version_id,
            document_type="statute",
            title=str(rule.get("act_title") or "Akt prawny"),
            citation=str(rule.get("publication") or ""),
            jurisdiction="PL",
            effective_from=effective_from,
            publication_date=str(rule.get("publication") or "") or None,
            is_consolidated_text=True,
        )
        text = str(rule.get("exact_source_span") or rule.get("directive") or "")
        provisions.append(
            ProvisionRecord(
                provision_id=provision_id,
                document_id=document_id,
                version_id=version_id,
                citation=citation,
                article=str(rule.get("article_key") or ""),
                paragraph=str(rule.get("paragraph") or "") or None,
                point=str(rule.get("point") or "") or None,
                letter=str(rule.get("letter") or "") or None,
                text=text,
                effective_from=effective_from,
                effective_to=str(rule.get("effective_to") or "") or None,
                status=status,
                source_document_id=document_id,
                source_chunk_ids=tuple(
                    str(item) for item in rule.get("supporting_chunk_ids") or []
                ),
                source_span_end=len(text),
                references=tuple(
                    str(item) for item in rule.get("definition_dependencies") or []
                ),
                display_reference=str(
                    rule.get("display_reference") or citation
                ).strip(),
                tax_domain=str(rule.get("tax_domain") or ""),
                taxpayer_role=str(rule.get("taxpayer_role") or ""),
                legal_mechanism=str(rule.get("legal_mechanism") or ""),
                entailed_result_codes=tuple(
                    str(item) for item in rule.get("entailed_result_codes") or []
                ),
                rule_relationship=str(
                    rule.get("rule_relationship") or "peer"
                ),
                related_provisions=tuple(
                    str(item) for item in rule.get("related_provisions") or []
                ),
                special_rule_provisions=tuple(
                    str(item) for item in rule.get("special_rule_provisions") or []
                ),
                exception_provisions=tuple(
                    str(item) for item in rule.get("exception_provisions") or []
                ),
                general_rule_provisions=tuple(
                    str(item) for item in rule.get("general_rule_provisions") or []
                ),
            )
        )
    return ProvisionRegistry(documents.values(), provisions)


def build_claims_from_rules(
    rules: Iterable[dict[str, object]],
    *,
    axis_ids: Iterable[str],
    missing_facts: Iterable[str],
) -> list[LegalClaim]:
    axes = tuple(axis_ids) or ("general",)
    missing = {item.strip() for item in missing_facts if item.strip()}
    claims: list[LegalClaim] = []
    for index, rule in enumerate(rules):
        required = tuple(str(item) for item in rule.get("required_facts") or [])
        absent = tuple(item for item in required if item in missing)
        provision_id = str(rule.get("provision_id") or "")
        status: ClaimStatus
        if not provision_id:
            status = "blocked"
        elif absent:
            status = "conditional_missing_fact"
        else:
            status = "supported"
        claims.append(
            LegalClaim(
                claim_id=f"claim:{provision_id or index}",
                axis_id=axes[index % len(axes)],
                claim_type="legal_rule",
                text=str(rule.get("directive") or ""),
                source_provisions=(provision_id,) if provision_id else (),
                fact_dependencies=required,
                missing_fact_dependencies=absent,
                status=status,
            )
        )
    return claims
