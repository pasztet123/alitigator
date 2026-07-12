from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.legal_rag_v2.authority import AuthorityExtractionResult
from app.legal_rag_v2.pipeline import (
    ClaimSet,
    LegalRagV2Config,
    LegalRagV2Pipeline,
    _build_evidence_bundles,
    _build_provision_graph,
)
from app.legal_rag_v2.planner import LegalQueryPlanner
from app.legal_rag_v2.retrieval import (
    LegalRetriever,
    LaneResult,
    LegalRetrievalResult,
    RetrievalCandidate,
    RetrievalConfig,
)
from app.legal_rag_v2.schemas import (
    AuthorityCard,
    AuthoritySourceSpans,
    Clarification,
    DocumentSourceSpan,
    Fact,
    LegalClaim,
    LegalIssue,
    LegalResearchPlan,
    QueryFamily,
    ResearchIntent,
    SourceSpan,
    WriterAnalysisSection,
    WriterOutput,
    WriterSource,
)


QUESTION = "Czy sprzedaż udziału podlega PIT?"


def research_plan() -> LegalResearchPlan:
    start = QUESTION.index("sprzedaż")
    return LegalResearchPlan(
        intent=ResearchIntent(
            mode="mixed_analysis",
            needs_normative_answer=True,
            needs_interpretations=True,
            needs_case_law=True,
            needs_conflict_analysis=True,
            needs_calculations=False,
        ),
        facts=[
            Fact(
                fact_id="fact_sale",
                subject="podatnik",
                role="seller",
                predicate="sells",
                value="sprzedaż",
                status="explicit",
                source_span=SourceSpan(
                    start=start,
                    end=start + len("sprzedaż"),
                    quote="sprzedaż",
                ),
            )
        ],
        issues=[
            LegalIssue(
                issue_id="pit_sale",
                label="Skutek sprzedaży w PIT",
                tax_domains=["PIT"],
                legal_mechanism="share_sale",
                requested_source_types=["statute", "interpretation", "judgment"],
                query_families=[
                    QueryFamily(
                        family="natural_language",
                        query=QUESTION,
                        lane="both",
                        origin="model",
                    )
                ],
            )
        ],
        clarification=Clarification(),
        confidence=0.9,
    )


class FakeGateway:
    async def generate_structured(self, *, response_model, **kwargs):
        if response_model is LegalResearchPlan:
            return research_plan()
        if response_model is ClaimSet:
            source_text = "Treść normy podatkowej dotyczącej sprzedaży udziału."
            return ClaimSet(
                claims=[
                    LegalClaim(
                        claim_id="claim_sale",
                        issue_id="pit_sale",
                        claim_type="application",
                        text="Regułę ustawową należy zastosować do wskazanej sprzedaży.",
                        status="approved",
                        result="Sprzedaż wymaga rozliczenia według pobranej reguły ustawowej.",
                        controlling_provision_ids=["pit_art_10"],
                        fact_dependencies=["fact_sale"],
                        source_spans=[
                            DocumentSourceSpan(
                                start=0,
                                end=len(source_text),
                                quote=source_text,
                                source_id="law-chunk",
                                document_id="law-document",
                                chunk_id="law-chunk",
                            )
                        ],
                        confidence=0.8,
                    )
                ]
            )
        if response_model is WriterOutput:
            return WriterOutput(
                thesis="Sprzedaż wymaga rozliczenia zgodnie z zatwierdzonym claimem.",
                analysis_sections=[
                    WriterAnalysisSection(
                        section_id="analysis_pit_sale",
                        title="Skutek sprzedaży w PIT",
                        content="Zastosowanie wynika z claimu opartego na pobranym przepisie.",
                        claim_ids_used=["claim_sale"],
                    )
                ],
                sources=[
                    WriterSource(
                        source_id="pit_art_10",
                        label="Przepis",
                        citation="art. 10",
                        claim_ids=["claim_sale"],
                    )
                ],
                risks_and_gaps=["Brak danych o dacie nabycia."],
                claim_ids_used=["claim_sale"],
            )
        raise AssertionError(f"Unexpected response model: {response_model}")

    async def generate_text(self, **kwargs):
        raise AssertionError("V2 must use structured outputs")


class FakeBackend:
    trace_marker = "fake_policy_free_backend"

    async def search(self, query, *, limit, source_types, metadata_filters):
        if "statute" in source_types:
            return [
                RetrievalCandidate(
                    candidate_id="law-chunk",
                    document_id="law-document",
                    chunk_id="law-chunk",
                    text="Treść normy podatkowej dotyczącej sprzedaży udziału.",
                    source_type="statute",
                    metadata={
                        "provision_id": "pit_art_10",
                        "legal_provisions": ["art. 10"],
                        "tax_domains": ["PIT"],
                        "legal_state_date": "2025-01-01",
                    },
                )
            ]
        return [
            RetrievalCandidate(
                candidate_id="authority-chunk",
                document_id="authority-document",
                chunk_id="authority-chunk",
                text="Organ uznał stanowisko podatnika za prawidłowe.",
                source_type="interpretation",
                metadata={
                    "signature": "TEST-SIGNATURE",
                    "tax_domains": ["PIT"],
                    "published_date": "2025-02-01",
                },
            )
        ]


class FakeAuthorityExtractor:
    async def extract(self, candidate):
        start = candidate.text.index("Organ")
        span = DocumentSourceSpan(
            start=start,
            end=len(candidate.text),
            quote=candidate.text[start:],
            source_id="authority_document",
            document_id=candidate.document_id,
            chunk_id=candidate.chunk_id,
        )
        return AuthorityExtractionResult(
            card=AuthorityCard(
                document_id=candidate.document_id,
                signature="TEST-SIGNATURE",
                document_type="interpretation",
                authority="organ podatkowy",
                date="2025-02-01",
                tax_domains=["PIT"],
                authority_holding=candidate.text,
                source_spans=AuthoritySourceSpans(authority_holding=[span]),
                extraction_confidence=0.9,
            ),
            trace={"extractor": "fake_model", "fallback_used": False},
        )


class LegalRagV2PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_special_rule_is_controlling_and_general_rule_is_dependency_across_chunks(self) -> None:
        shared = {
            "legal_provisions": ["art. 21"],
            "act_title": "Ustawa PIT",
            "publication": "Dz.U. test",
        }
        retrieval = LegalRetrievalResult(
            primary_law=(
                LaneResult(
                    issue_id="pit_sale",
                    lane="primary_law",
                    query_families=(),
                    candidates=(
                        RetrievalCandidate(
                            "special", "Art. 21.\n30a. Wydatki, o których mowa w ust. 25, obejmują także spłatę kredytu.",
                            "statute", document_id="pit-art21-part9", chunk_id="special", metadata=shared,
                        ),
                        RetrievalCandidate(
                            "general", "Art. 21.\n25. Reguła wydatków mieszkaniowych.",
                            "statute", document_id="pit-art21-part8", chunk_id="general", metadata=shared,
                        ),
                    ),
                    candidate_count_before_rerank=2,
                ),
            ),
            authorities=(),
            trace=(),
        )
        graph, _, references = _build_provision_graph(retrieval)
        bundles = _build_evidence_bundles(research_plan(), retrieval, {}, graph, references)

        self.assertIn("art. 21 ust. 30a", [item.citation for item in bundles[0].controlling_provisions])
        self.assertIn("art. 21 ust. 25", [item.citation for item in bundles[0].dependency_provisions])

    async def test_future_provision_is_not_controlling_for_historical_target_date(self) -> None:
        plan = research_plan().model_copy(update={"target_date": "2020-01-01"})
        candidate = RetrievalCandidate(
            candidate_id="future-law",
            document_id="law-document",
            chunk_id="future-law",
            text="Art. 10. Przepis obowiązujący od 2021 roku.",
            source_type="statute",
            metadata={
                "effective_from": "2021-01-01",
                "legal_provisions": ["art. 10"],
            },
        )
        retrieval = LegalRetrievalResult(
            primary_law=(
                LaneResult(
                    issue_id="pit_sale",
                    lane="primary_law",
                    query_families=(),
                    candidates=(candidate,),
                    candidate_count_before_rerank=1,
                ),
            ),
            authorities=(),
            trace=(),
        )

        graph, graph_schema, references = _build_provision_graph(
            retrieval,
            target_date=plan.target_date,
        )
        bundles = _build_evidence_bundles(plan, retrieval, {}, graph, references)

        self.assertTrue(graph_schema.provisions)
        self.assertTrue(all(item.status == "historical" for item in graph_schema.provisions))
        self.assertEqual(bundles[0].controlling_provisions, [])
        self.assertIn("primary_law", bundles[0].missing_sources)

    async def test_every_request_uses_one_evidence_gated_pipeline(self) -> None:
        gateway = FakeGateway()
        planner = LegalQueryPlanner(gateway, model="gpt-5.6-terra")
        retriever = LegalRetriever(
            FakeBackend(),
            config=RetrievalConfig(selected_limit_per_issue=4),
        )
        with tempfile.TemporaryDirectory() as directory:
            pipeline = LegalRagV2Pipeline(
                gateway=gateway,
                planner=planner,
                retriever=retriever,
                authority_extractor=FakeAuthorityExtractor(),
                config=LegalRagV2Config(
                    artifact_root=Path(directory),
                    allow_legacy_fallback=False,
                ),
            )
            result = await pipeline.run(QUESTION, run_id="test-run")

            self.assertEqual(result.mode, "legal_rag_v2")
            self.assertEqual(result.legal_research_plan.issues[0].issue_id, "pit_sale")
            self.assertEqual(result.claims[0].status, "approved")
            self.assertTrue(all(item.passed for item in result.validation))
            self.assertIn("Teza\n", result.final_answer or "")
            self.assertIn("Ryzyka i luki\n", result.final_answer or "")
            self.assertEqual(
                {path.name for path in (Path(directory) / "test-run").iterdir()},
                set(pipeline.trace_factory("test-run").required_artifacts),
            )


if __name__ == "__main__":
    unittest.main()
