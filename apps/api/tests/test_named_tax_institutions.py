from __future__ import annotations

import asyncio

from app.legal_institutions import InstitutionMatcher, merge_locked_institutions
from app.legal_institutions.dictionary import load_default_dictionary
from app.legal_institutions.evaluate import evaluate_dictionary_cases
from app.legal_rag_v2.retrieval import (
    LegalRetriever,
    RetrievalCandidate,
    RetrievalConfig,
    reciprocal_rank_fusion,
)
from app.legal_rag_v2.schemas import Clarification, LegalIssue, LegalResearchPlan, ResearchIntent
from tests.fixtures.named_institution_cases import COLLISION_CASES, E2E_CASES, negative_cases, positive_cases


def _plan() -> LegalResearchPlan:
    return LegalResearchPlan(
        user_query="test",
        intent=ResearchIntent(mode="mixed_analysis"),
        issues=[
            LegalIssue(
                issue_id="issue",
                label="Zagadnienie podatkowe",
                tax_domains=[],
                legal_mechanism="general_tax_analysis",
                requested_source_types=["interpretation"],
            )
        ],
        clarification=Clarification(),
        confidence=0.8,
    )


def test_dictionary_is_versioned_large_and_has_stage_a_activation() -> None:
    dictionary = load_default_dictionary()
    assert dictionary.version == "pl_tax_institutions_v1"
    assert len(dictionary.institutions) >= 120
    assert sum(item.status == "active" for item in dictionary.institutions) >= 50
    assert all(item.canonical_name for item in dictionary.institutions)


def test_metrics_report_covers_required_deterministic_quality_signals() -> None:
    report = evaluate_dictionary_cases(positive_cases(), negative_cases())
    assert report["positive_recognition_coverage"] == 1.0
    assert report["positive_lock_contract_coverage"] == 1.0
    assert report["negative_no_lock_rate"] == 1.0
    assert report["direct_gate"]["rejection_reason"] == "missing_locked_institution_markers"


def test_generated_fixture_has_at_least_150_positive_and_150_negative_cases() -> None:
    matcher = InstitutionMatcher()
    positives = positive_cases()
    negatives = negative_cases()
    assert len(positives) >= 150
    assert len(negatives) >= 150
    for case in positives:
        ids = {item.institution_id for item in matcher.match(str(case["question"])).matches}
        assert case["institution_id"] in ids, case
    for case in negatives:
        assert not [item for item in matcher.match(str(case["question"])).matches if item.locked], case


def test_required_aliases_are_deterministic_and_shadow_entries_do_not_lock() -> None:
    matcher = InstitutionMatcher()
    cases = {
        "IP-Box": ("ip_box", True),
        "B+R": ("research_and_development_relief", True),
        "estoński CIT": ("estonian_cit", True),
        "MPP": ("split_payment", True),
        "biała lista": ("white_list", True),
        "pay&refund": ("pay_and_refund", True),
        "beneficial owner": ("beneficial_owner", True),
        "KSeF": ("ksef_mandatory", True),
        "TPR": ("tpr_information", True),
        "APA": ("apa", True),
        "WIA": ("binding_excise_information", True),
        "WIT": ("binding_tariff_information", True),
        "PE podatkowy": ("permanent_establishment", True),
        "FE w VAT": ("fixed_establishment", True),
        "ulga CSR": ("csr_sponsorship_relief", False),
        "CFC": ("cfc_taxation", False),
        "WIS": ("binding_rate_information", True),
        "GAAR": ("ga_ar", True),
    }
    for phrase, (institution_id, locked) in cases.items():
        match = next(item for item in matcher.match(f"Czy stosuje się {phrase}?").matches if item.institution_id == institution_id)
        assert match.locked is locked


def test_collision_suite_does_not_create_false_locks() -> None:
    matcher = InstitutionMatcher()
    for name, question, expected in COLLISION_CASES:
        actual = {item.institution_id for item in matcher.match(question).matches if item.locked}
        assert actual == expected, name


def test_e2e_locks_are_merged_into_candidate_channels_and_preserve_provisions() -> None:
    matcher = InstitutionMatcher()
    for case in E2E_CASES:
        result = matcher.match(case["question"])
        plan = merge_locked_institutions(_plan(), result, dictionary=matcher.dictionary)
        lock_ids = {item.institution_id for item in plan.deterministic_institutions}
        assert case["institution_id"] in lock_ids, case["name"]
        issue = next(issue for issue in plan.issues if case["institution_id"] in issue.locked_institution_ids)
        families = {item.family for item in issue.query_families if item.origin == "deterministic"}
        assert "named_institution_canonical" in families
        assert "named_institution_provision" in families
        assert set(case["expected_provisions"]).intersection(issue.possible_provision_concepts)


def test_conflicting_model_hypothesis_cannot_replace_deterministic_lock() -> None:
    matcher = InstitutionMatcher()
    model_plan = _plan().model_copy(
        update={
            "model_inferred_institutions": ["minimum_cit"],
            "issues": [
                _plan().issues[0].model_copy(
                    update={
                        "legal_mechanism": "general_tax_analysis",
                        "model_inferred_institution_ids": ["minimum_cit"],
                    }
                )
            ],
        }
    )
    merged = merge_locked_institutions(
        model_plan,
        matcher.match("Czy IP-Box obejmuje dochód programisty?"),
        dictionary=matcher.dictionary,
    )
    assert merged.issues[0].legal_mechanism == "ip_box"
    assert merged.issues[0].locked_institution_ids == ["ip_box"]
    assert merged.institution_conflicts
    assert {item.resolution for item in merged.institution_conflicts} == {"keep_deterministic"}


class _MarkerBackend:
    async def search(self, query, *, limit, source_types, metadata_filters):
        return [
            RetrievalCandidate(
                candidate_id="direct",
                document_id="direct-doc",
                text="Interpretacja dotycząca IP Box oraz kwalifikowanego prawa własności intelektualnej.",
                source_type="interpretation",
                metadata={"subject": "IP Box", "tax_domains": ["PIT"], "legal_provisions": ["PIT art. 30ca"]},
            ),
            RetrievalCandidate(
                candidate_id="wrong-neighbour",
                document_id="wrong-doc",
                text="Interpretacja dotycząca kosztów reprezentacji w działalności gospodarczej.",
                source_type="interpretation",
                metadata={"subject": "Koszty reprezentacji", "tax_domains": ["PIT"], "legal_provisions": ["PIT art. 22"]},
            ),
        ]


def test_direct_authority_gate_rejects_documents_without_locked_markers() -> None:
    matcher = InstitutionMatcher()
    plan = merge_locked_institutions(
        _plan(), matcher.match("Czy IP Box obejmuje dochód programisty?"), dictionary=matcher.dictionary
    )
    result = asyncio.run(
        LegalRetriever(
            _MarkerBackend(),
            primary_enabled=False,
            config=RetrievalConfig(selected_limit_per_issue=6),
            institution_matcher=matcher,
        ).retrieve(plan)
    )
    candidates = result.authorities[0].candidates
    assert [item.candidate_id for item in candidates] == ["direct"]
    rejections = [item for item in result.trace if item.get("event") == "institution_filter_rejection"]
    assert rejections and rejections[0]["reason"] == "missing_locked_institution_markers"
    assert rejections[0]["institution_ids"] == ["ip_box"]


def test_named_institution_channels_have_bounded_higher_rrf_priority() -> None:
    generic = RetrievalCandidate(candidate_id="generic", text="ogólny koszt", source_type="interpretation")
    named = RetrievalCandidate(candidate_id="named", text="IP Box", source_type="interpretation")
    fused = reciprocal_rank_fusion(
        [
            ("lexical:interpretation", "natural_language", [generic]),
            ("lexical:interpretation", "named_institution_canonical", [named]),
        ]
    )
    assert fused[0].candidate_id == "named"


def test_document_marker_accepts_verified_article_with_corpus_specific_act_prefix() -> None:
    matcher = InstitutionMatcher()
    definition = matcher.dictionary.by_id["bad_debt_relief_vat"]
    markers = matcher.document_markers(
        definition,
        text="Prawo do skorzystania z ulgi na złe długi.",
        metadata={
            "tax_domains": ["VAT"],
            "legal_provisions": ["[VAT] art. 89a ust. 1"],
        },
    )
    assert "VAT art. 89a" in markers
    wrong_domain = matcher.document_markers(
        definition,
        text="Ogólny dokument.",
        metadata={"tax_domains": ["CIT"], "legal_provisions": ["[CIT] art. 89a"]},
    )
    assert "VAT art. 89a" not in wrong_domain
