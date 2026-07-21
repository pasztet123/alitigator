from app.legal_institutions import InstitutionMatcher
from app.legal_rag_v2.document_validation import (
    QuestionCard,
    build_document_card,
    build_question_card,
    extract_document_sections,
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


def test_document_card_uses_relevant_sections_and_caches_by_document_content() -> None:
    document = _candidate(
        title="Interpretacja podatkowa",
        text=(
            "Pytanie\nCzy płatnik ma pobrać podatek?\n\n"
            "Ocena stanowiska\nAnalizowane są należności licencyjne.\n\n"
            "Reasumując\nPodatek u źródła wynika z art. 21 CIT."
        ),
        provisions=["CIT art. 21"],
        domains=("CIT",),
        document_id="sectioned-document",
    )

    sections = extract_document_sections(document.text, keywords=("podatek u źródła",))
    first = build_document_card(document, matcher=InstitutionMatcher())
    second = build_document_card(document, matcher=InstitutionMatcher())

    assert sections.question_section.startswith("Czy płatnik")
    assert "withholding_tax" in first.detected_institutions
    assert first is second
    assert all(item.source and item.start >= -1 for item in first.evidence_for("withholding_tax"))


def test_hydrated_document_is_validated_instead_of_seed_chunk() -> None:
    class HydratingBackend:
        async def search(self, query, *, limit, source_types, metadata_filters):
            return [
                _candidate(
                    title="Interpretacja CIT",
                    text="Pierwszy chunk nie zawiera pełnej analizy.",
                    provisions=["CIT art. 21", "CIT art. 26"],
                    domains=("CIT",),
                    candidate_id="hydrated",
                    document_id="hydrated",
                )
            ]

        async def hydrate_document(self, candidate):
            return RetrievalCandidate(
                candidate_id=candidate.candidate_id,
                document_id=candidate.document_id,
                chunk_id="document:hydrated:full",
                source_type=candidate.source_type,
                text=(
                    "Pytanie\nCzy polski płatnik musi pobrać podatek u źródła?\n\n"
                    "Ocena stanowiska\nOpłata za dostęp do oprogramowania SaaS na podstawie EULA."
                ),
                metadata=candidate.metadata,
            )

    import asyncio

    result = asyncio.run(
        LegalRetriever(
            HydratingBackend(),
            primary_enabled=False,
            config=RetrievalConfig(selected_limit_per_issue=6),
            institution_matcher=InstitutionMatcher(),
        ).retrieve(_wht_plan())
    )

    candidate = result.authorities[0].candidates[0]
    assert candidate.metadata["document_validation_hydrated"] is True
    assert candidate.metadata["document_validation"]["relation"] == "direct"


def test_irrelevant_regression_documents_do_not_inherit_wht() -> None:
    fixtures = {
        "0115-KDIT1.4011.321.2026.1.MK": ("Ulga mieszkaniowa", "Sprzedaż lokalu mieszkalnego w PIT."),
        "0115-KDIT2.4011.79.2026.2.MD": ("Ulga rehabilitacyjna", "Odliczenie wydatków rehabilitacyjnych."),
        "0113-KDIPT2-3.4011.179.2022.12.GG": ("Leasing samochodu", "Koszty leasingu samochodu osobowego."),
        "0115-KDIT3.4011.779.2025.1.DP": ("Wkład w spółce komandytowej", "Obniżenie wkładu wspólnika."),
        "0115-KDIT3.4011.295.2026.2.PS": ("Ulga B+R", "Odliczenie kosztów działalności badawczo-rozwojowej."),
    }
    for signature, (title, text) in fixtures.items():
        card = build_document_card(
            _candidate(title=title, text=text, signature=signature, document_id=signature),
            matcher=InstitutionMatcher(),
        )
        result = evaluate_document_relevance(_question_card(), card)
        assert "withholding_tax" not in card.detected_institutions, signature
        assert result.passed is False and result.reject is True, signature
        assert result.relation == "irrelevant", signature


def test_sponsorship_and_expansion_documents_need_their_own_institution_evidence() -> None:
    matcher = InstitutionMatcher()
    cases = (
        (
            "csr_sponsorship_relief",
            "Ulga sponsoringowa",
            "Dodatkowe odliczenie wydatków na podstawie art. 18ee CIT.",
            ["CIT art. 18ee"],
        ),
        (
            "expansion_relief",
            "Ulga na ekspansję",
            "Odliczenie kosztów zwiększenia przychodów na podstawie art. 18eb CIT.",
            ["CIT art. 18eb"],
        ),
    )
    for institution_id, title, text, provisions in cases:
        card = build_document_card(
            _candidate(title=title, text=text, provisions=provisions, domains=("CIT",), document_id=institution_id),
            matcher=matcher,
        )
        question = QuestionCard(
            tax_domains=("CIT",),
            locked_institutions=(institution_id,),
            primary_mechanism=institution_id,
            provision_hints=tuple(provisions),
        )
        result = evaluate_document_relevance(question, card)
        assert institution_id in card.detected_institutions
        assert card.evidence_for(institution_id)
        assert result.relation == "direct"


def test_studying_child_document_is_not_direct_for_postgraduate_business_studies() -> None:
    question = QuestionCard(
        tax_domains=("PIT",),
        primary_mechanism="business_expense",
        transaction_type="professional_education",
    )
    document = build_document_card(
        _candidate(
            title="Ulga prorodzinna na studiujące dziecko",
            text="Dokument dotyczy samotnego rodzica i dziecka studiującego.",
            domains=("PIT",),
        ),
        matcher=InstitutionMatcher(),
    )

    result = evaluate_document_relevance(question, document)
    assert result.passed is False
    assert result.reject is True
    assert result.relation == "irrelevant"
