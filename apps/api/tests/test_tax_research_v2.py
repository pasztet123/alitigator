from app.legal_rag_v2.pipeline import _enrich_research_plan
from app.legal_rag_v2.retrieval import RetrievalCandidate, TransparentLegalReranker
from app.legal_rag_v2.schemas import Clarification, LegalIssue, LegalResearchPlan, ResearchIntent


def _plan() -> LegalResearchPlan:
    return LegalResearchPlan(
        intent=ResearchIntent(mode="mixed_analysis"),
        issues=[
            LegalIssue(
                issue_id="general",
                label="Ogólne zagadnienie podatkowe",
                tax_domains=["PIT"],
                legal_mechanism="general_tax_analysis",
                requested_source_types=["interpretation"],
            )
        ],
        clarification=Clarification(),
        confidence=0.5,
    )


def test_v2_enrichment_sends_business_rent_mechanism_and_negatives_to_authorities() -> None:
    question = (
        "Czy przedsiębiorca może zaliczyć do kosztów wydatki na wynajem mieszkania "
        "w innym mieście w związku z realizacją kontraktu?"
    )
    issue = _enrich_research_plan(_plan(), question).issues[0]

    assert issue.legal_mechanism == "business_accommodation_expense"
    assert "ulga na powrót" in issue.negative_fact_constraints
    authority_queries = {item.query for item in issue.query_families if item.lane == "authority"}
    assert any("wynajem" in query for query in authority_queries)
    assert "PIT art. 22 ust. 1" in authority_queries


def test_v2_enrichment_recognises_special_cash_and_vehicle_provisions() -> None:
    cash_question = "Czy zapłata gotówką w ratach za transakcję 18 000 zł pozwala zachować koszt?"
    vehicle_question = "Czy można odliczyć 50% VAT od paliwa do samochodu bez VAT-26?"

    cash_issue = _enrich_research_plan(_plan(), cash_question).issues[0]
    vehicle_issue = _enrich_research_plan(_plan(), vehicle_question).issues[0]

    assert cash_issue.legal_mechanism == "cash_payment_cost_exclusion"
    assert "PIT art. 22p" in cash_issue.possible_provision_concepts
    assert vehicle_issue.legal_mechanism == "mixed_use_vehicle_vat"
    assert "VAT art. 86a" in vehicle_issue.possible_provision_concepts


def test_v2_reranker_rejects_return_relief_as_wrong_neighbour() -> None:
    question = "Czy wynajem mieszkania przy kontrakcie może być kosztem działalności?"
    issue = _enrich_research_plan(_plan(), question).issues[0]
    reranker = TransparentLegalReranker()
    direct = RetrievalCandidate(
        candidate_id="direct",
        text="Koszty najmu mieszkania ponoszonego podczas realizacji kontraktu dla klienta.",
        source_type="interpretation",
        metadata={
            "subject": "Koszty najmu mieszkania przy realizacji kontraktu",
            "tax_domains": ["PIT"],
            "legal_provisions": ["PIT art. 22 ust. 1"],
        },
    )
    return_relief = RetrievalCandidate(
        candidate_id="return-relief",
        text="Podatnik korzysta z ulgi na powrót po zmianie rezydencji podatkowej.",
        source_type="interpretation",
        metadata={
            "subject": "Możliwość skorzystania z ulgi na powrót",
            "tax_domains": ["PIT"],
            "legal_provisions": ["PIT art. 21 ust. 1 pkt 152"],
        },
    )

    direct_score = reranker.score(issue, direct, target_date=None)
    wrong_score = reranker.score(issue, return_relief, target_date=None)

    assert direct_score.final_score > wrong_score.final_score
    assert any(reason.startswith("wrong_legal_mechanism:") for reason in wrong_score.negative_reasons)
