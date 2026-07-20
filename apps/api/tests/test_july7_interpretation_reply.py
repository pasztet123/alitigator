from app.main import build_july7_interpretations_reply
from app.legacy_july7.rag import RagChunk
from app.legacy_interpretations import TaxResearchDocument, TaxResearchSearchResult
from app.tax_research import CandidateAssessment, ResearchUnderstanding


def test_july7_interpretation_reply_keeps_full_document_without_truncation() -> None:
    full_text = "Początek.\n\n" + ("Pełna treść interpretacji. " * 80) + "\n\nKoniec."
    chunk = RagChunk(
        chunk_id="interpretation-1:0",
        document_id="interpretation-1",
        chunk_index=0,
        score=99.0,
        chunk_text=full_text,
        subject="Implanty zębowe",
        signature="0112-KDIL2-2.4011.8.2026.3.MM",
        published_date="2026-03-04",
        source_url="https://example.test/interpretation-1",
        category="Interpretacja indywidualna",
    )

    reply = build_july7_interpretations_reply([chunk])

    assert full_text in reply
    assert "### 1. 0112-KDIL2-2.4011.8.2026.3.MM" in reply
    assert "**Pełna treść interpretacji**" in reply


def test_july7_interpretation_reply_separates_wrong_legal_mechanisms() -> None:
    direct_chunk = RagChunk(
        chunk_id="direct:0", document_id="direct", chunk_index=0, score=80.0,
        chunk_text="Koszt najmu mieszkania przy kontrakcie.", subject="Koszt najmu mieszkania",
        signature="DIRECT", published_date="2026-01-01", source_url="https://example.test/direct",
        category="Interpretacja indywidualna",
    )
    wrong_chunk = RagChunk(
        chunk_id="wrong:0", document_id="wrong", chunk_index=0, score=0.0,
        chunk_text="Ulga na powrót po zmianie rezydencji.", subject="Ulga na powrót",
        signature="RETURN-RELIEF", published_date="2026-01-01", source_url="https://example.test/wrong",
        category="Interpretacja indywidualna",
    )
    direct_assessment = CandidateAssessment(
        relation="direct", reject=False, reason="Zgodny mechanizm.",
        document_mechanism="business_accommodation_expense", material_differences=(), score=80.0,
        components={},
    )
    wrong_assessment = CandidateAssessment(
        relation="different_mechanism", reject=True, reason="Inny mechanizm.",
        document_mechanism="return_relief_or_residency", material_differences=("wrong_legal_mechanism",), score=0.0,
        components={},
    )
    result = TaxResearchSearchResult(
        question="Pytanie", understanding=ResearchUnderstanding(), database_queries=(), candidate_counts={},
        candidates_before_rerank=(), candidate_document_ids=(), reranker_scores=(), validation_results=(),
        documents=(
            TaxResearchDocument(direct_chunk, direct_assessment),
            TaxResearchDocument(wrong_chunk, wrong_assessment),
        ),
    )

    reply = build_july7_interpretations_reply(result)

    assert "## Bezpośrednio relewantne" in reply
    assert "## Inny mechanizm podatkowy" in reply
    assert reply.index("DIRECT") < reply.index("RETURN-RELIEF")
