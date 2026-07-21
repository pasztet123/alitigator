from app.legal_institutions import InstitutionMatcher
from app.legal_rag_v2.document_validation import (
    QuestionCard,
    build_document_card,
    build_question_card,
    evaluate_document_relevance,
)
from app.legal_rag_v2.retrieval import RetrievalCandidate
from app.legal_rag_v2.retrieval import LegalRetriever, RetrievalConfig, _families_for_lane
from app.legal_rag_v2.schemas import Clarification, LegalIssue, LegalResearchPlan, ResearchIntent


WHT_QUESTION = (
    "Czy polska spółka musi pobrać podatek u źródła od opłaty za dostęp "
    "do zagranicznego SaaS, jeżeli umowa to EULA?"
)


def _question_card() -> QuestionCard:
    issue = LegalIssue(
        issue_id="wht",
        label="WHT SaaS",
        tax_domains=["CIT"],
        legal_mechanism="withholding_tax",
        locked_institution_ids=["withholding_tax"],
        possible_provision_concepts=["CIT art. 21", "CIT art. 26"],
    )
    return build_question_card(question=WHT_QUESTION, issue=issue)


def _candidate(
    *,
    title: str,
    text: str,
    provisions=(),
    domains=("PIT",),
    candidate_id="candidate",
    document_id="document",
    signature="TEST",
) -> RetrievalCandidate:
    return RetrievalCandidate(
        candidate_id=candidate_id,
        document_id=document_id,
        text=text,
        source_type="interpretation",
        metadata={
            "signature": signature,
            "subject": title,
            "tax_domains": list(domains),
            "legal_provisions": list(provisions),
        },
    )


def test_document_card_does_not_inherit_question_mechanism() -> None:
    card = build_document_card(
        _candidate(
            title="Ulga rehabilitacyjna na używanie samochodu",
            text="Dokument dotyczy odliczenia wydatków rehabilitacyjnych w podatku PIT.",
        ),
        matcher=InstitutionMatcher(),
    )

    assert "withholding_tax" not in card.detected_mechanisms
    assert "withholding_tax" not in card.detected_institutions
    assert "rehabilitation_relief" in card.detected_mechanisms


def test_irrelevant_relief_is_rejected_for_wht_saas_question() -> None:
    card = build_document_card(
        _candidate(
            title="Ulga mieszkaniowa",
            text="Sprzedaż lokalu mieszkalnego i zwolnienie z podatku dochodowego.",
        ),
        matcher=InstitutionMatcher(),
    )
    result = evaluate_document_relevance(_question_card(), card)

    assert "withholding_tax" not in card.detected_institutions
    assert result.passed is False
    assert result.relation == "irrelevant"
    assert result.reason == "missing_document_institution_evidence"


def test_other_wht_transaction_is_context_not_direct_for_saas_eula() -> None:
    card = build_document_card(
        _candidate(
            title="Podatek u źródła od odsetek i konwersji długu na kapitał",
            text=(
                "Analiza podatku u źródła od odsetek wypłacanych do Indonezji, "
                "konwersji długu na kapitał oraz odliczenia zagranicznego podatku."
            ),
            provisions=["CIT art. 20"],
            domains=("CIT",),
        ),
        matcher=InstitutionMatcher(),
    )
    result = evaluate_document_relevance(_question_card(), card)

    assert "withholding_tax" in card.detected_institutions
    assert result.passed is False
    assert result.relation == "context_only"
    assert result.matched_institutions == ("withholding_tax",)


def test_saas_wht_document_requires_its_own_evidence() -> None:
    card = build_document_card(
        _candidate(
            title="Podatek u źródła od opłat za dostęp do oprogramowania SaaS",
            text="Polski płatnik dokonuje opłaty za SaaS na podstawie EULA.",
            provisions=["CIT art. 21 ust. 1", "CIT art. 26"],
            domains=("CIT",),
        ),
        matcher=InstitutionMatcher(),
    )
    result = evaluate_document_relevance(_question_card(), card)

    assert "withholding_tax" in card.detected_institutions
    assert card.evidence_for("withholding_tax")
    assert result.passed is True
    assert result.relation == "direct"
    assert result.matched_institutions == ("withholding_tax",)


class _WhtCandidateBackend:
    async def search(self, query, *, limit, source_types, metadata_filters):
        return [
            _candidate(
                title="Ulga mieszkaniowa po sprzedaży lokalu",
                text="Zwolnienie mieszkaniowe w PIT.",
                candidate_id="housing",
                document_id="housing",
            ),
            _candidate(
                title="Podatek u źródła od odsetek i konwersji długu",
                text="Podatek u źródła od odsetek, konwersji długu i kapitału.",
                provisions=["CIT art. 20"],
                domains=("CIT",),
                candidate_id="wht-context",
                document_id="wht-context",
                signature="WHT-CONTEXT",
            ),
            RetrievalCandidate(
                candidate_id="wht-saas",
                document_id="wht-saas",
                text="Polski płatnik płaci za dostęp do oprogramowania SaaS na podstawie EULA.",
                source_type="interpretation",
                metadata={
                    "signature": "WHT-SAAS",
                    "subject": "Podatek u źródła od opłaty za dostęp do oprogramowania SaaS",
                    "tax_domains": ["CIT"],
                    "legal_provisions": ["CIT art. 21", "CIT art. 26"],
                },
            ),
        ]


def _wht_plan() -> LegalResearchPlan:
    return LegalResearchPlan(
        user_query=WHT_QUESTION,
        intent=ResearchIntent(mode="mixed_analysis"),
        issues=[
            LegalIssue(
                issue_id="wht",
                label="WHT SaaS",
                tax_domains=["CIT"],
                legal_mechanism="withholding_tax",
                locked_institution_ids=["withholding_tax"],
                possible_provision_concepts=["CIT art. 21", "CIT art. 26"],
                requested_source_types=["interpretation"],
            )
        ],
        clarification=Clarification(),
        confidence=0.9,
    )


def test_authority_retrieval_keeps_only_direct_wht_saas_document_cards() -> None:
    import asyncio

    result = asyncio.run(
        LegalRetriever(
            _WhtCandidateBackend(),
            primary_enabled=False,
            config=RetrievalConfig(selected_limit_per_issue=6),
            institution_matcher=InstitutionMatcher(),
        ).retrieve(_wht_plan())
    )

    candidates = result.authorities[0].candidates
    assert [candidate.document_id for candidate in candidates] == ["wht-saas"]
    validation = candidates[0].metadata["document_validation"]
    assert validation["relation"] == "direct"
    assert validation["institution_gate_passed"] is True
    rejections = {
        item["candidate_signature"]["document_id"]: item
        for item in result.trace
        if item.get("event") == "institution_filter_rejection"
    }
    assert rejections["housing"]["reason"] == "missing_document_institution_evidence"
    assert rejections["wht-context"]["relation"] == "context_only"


def test_locked_authority_query_preserves_user_distinguishing_terms() -> None:
    families = _families_for_lane(_wht_plan(), _wht_plan().issues[0], "authority")

    assert any(
        family.family == "user_terminology"
        and family.query == "SaaS EULA"
        and family.origin == "user"
        for family in families
    )
