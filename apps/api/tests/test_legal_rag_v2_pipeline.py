from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app.legal_rag_v2.pipeline as pipeline_module
from app.legal_rag_v2.authority import AuthorityExtractionResult
from app.model_gateway import (
    ModelFallbackError,
    ModelProviderRequestError,
    ModelRateLimitError,
    ModelRequestError,
)
from app.legal_rag_v2.pipeline import (
    ClaimSet,
    LegalRagV2Config,
    LegalRagV2Pipeline,
    _build_evidence_bundles,
    _build_provision_graph,
    _git_commit,
    validate_writer_output,
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
    EvidenceBundle,
    Fact,
    LegalClaim,
    LegalIssue,
    LegalResearchPlan,
    ProvisionReference,
    QueryFamily,
    ResearchIntent,
    SourceSpan,
    ValidationRecord,
    WriterAnalysisSection,
    WriterOutput,
    WriterSource,
)


QUESTION = "Czy sprzedaż udziału podlega PIT?"


class RuntimeDiagnosticTests(unittest.TestCase):
    def test_git_commit_uses_safe_fallback_in_cloud_run_layout(self) -> None:
        with (
            patch.object(pipeline_module, "__file__", "/app/app/legal_rag_v2/pipeline.py"),
            patch.dict("os.environ", {"K_REVISION": "cloud-run-revision"}, clear=False),
        ):
            self.assertEqual(_git_commit(), "cloud-run-revision")


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

    async def test_invalid_model_render_is_replaced_only_by_revalidated_fallback(self) -> None:
        gateway = FakeGateway()
        planner = LegalQueryPlanner(gateway, model="gpt-5.6-terra")
        retriever = LegalRetriever(FakeBackend(), config=RetrievalConfig(selected_limit_per_issue=4))
        original_validate = pipeline_module.validate_rendered_answer
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return ValidationRecord(
                    stage="post_render_validation",
                    passed=False,
                    errors=["simulated_model_render_integrity_failure"],
                )
            return original_validate(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            pipeline = LegalRagV2Pipeline(
                gateway=gateway,
                planner=planner,
                retriever=retriever,
                authority_extractor=FakeAuthorityExtractor(),
                config=LegalRagV2Config(artifact_root=Path(directory), allow_legacy_fallback=False),
            )
            with patch.object(pipeline_module, "validate_rendered_answer", side_effect=fail_once):
                result = await pipeline.run(QUESTION, run_id="revalidated-render")

        final_validation = result.validation[-1]
        self.assertTrue(final_validation.passed)
        self.assertIn("deterministic_render_revalidated", final_validation.warnings)
        self.assertEqual(2, calls)

    async def test_large_synthesis_is_split_and_recovers_every_issue(self) -> None:
        issues = [
            LegalIssue(
                issue_id=f"issue_{index}",
                label=f"Zagadnienie {index}",
                legal_mechanism="general",
            )
            for index in range(5)
        ]
        plan = LegalResearchPlan(
            user_query="Wielowątkowy kazus podatkowy.",
            intent=ResearchIntent(mode="mixed_analysis"),
            issues=issues,
            clarification=Clarification(),
            confidence=0.9,
        )
        bundles = []
        for index, issue in enumerate(issues):
            provision = ProvisionReference(
                provision_id=f"provision_{index}",
                document_id=f"document_{index}",
                citation=f"art. {index + 1}",
                status="active",
                source_span=DocumentSourceSpan(
                    start=0,
                    end=10,
                    document_id=f"document_{index}",
                ),
            )
            bundles.append(
                EvidenceBundle(
                    issue_id=issue.issue_id,
                    controlling_provisions=[provision],
                    coverage_status="complete",
                )
            )

        class SizeRejectingGateway(FakeGateway):
            async def generate_structured(self, *, response_model, input, **kwargs):
                if response_model is not ClaimSet:
                    return await super().generate_structured(
                        response_model=response_model, input=input, **kwargs
                    )
                payload = pipeline_module.json.loads(input)
                batch_issues = payload["plan"]["issues"]
                if len(batch_issues) > 1:
                    raise ModelRequestError("payload too large")
                issue_id = batch_issues[0]["issue_id"]
                bundle = next(
                    item for item in bundles if item.issue_id == issue_id
                )
                provision = bundle.controlling_provisions[0]
                return ClaimSet(
                    claims=[
                        LegalClaim(
                            claim_id=f"claim_{issue_id}",
                            issue_id=issue_id,
                            claim_type="application",
                            text="Zastosowanie przepisu wymaga oceny faktów.",
                            status="conditional_missing_fact",
                            result="Oś została przeanalizowana niezależnie.",
                            controlling_provision_ids=[provision.provision_id],
                            source_spans=[provision.source_span],
                            confidence=0.8,
                        )
                    ]
                )

        gateway = SizeRejectingGateway()
        pipeline = LegalRagV2Pipeline(
            gateway=gateway,
            planner=LegalQueryPlanner(gateway),
            retriever=LegalRetriever(FakeBackend()),
            config=LegalRagV2Config(),
        )
        claims, validation = await pipeline._synthesize_and_validate_claims(
            plan.user_query,
            plan,
            bundles,
            [],
        )

        self.assertTrue(validation.passed)
        self.assertEqual({item.issue_id for item in issues}, {item.issue_id for item in claims})
        self.assertTrue(all(item.status == "conditional_missing_fact" for item in claims))
        self.assertTrue(any("batch_split_after:ModelRequestError" in item for item in validation.warnings))

    async def test_rate_limit_failure_does_not_fan_out_into_per_issue_calls(self) -> None:
        issues = [
            LegalIssue(
                issue_id=f"limited_{index}",
                label=f"Zagadnienie {index}",
                legal_mechanism="general",
            )
            for index in range(10)
        ]
        plan = LegalResearchPlan(
            user_query="Wielowątkowy kazus podatkowy.",
            intent=ResearchIntent(mode="mixed_analysis"),
            issues=issues,
            clarification=Clarification(),
            confidence=0.9,
        )

        class RateLimitedGateway(FakeGateway):
            claim_calls = 0

            async def generate_structured(self, *, response_model, **kwargs):
                if response_model is ClaimSet:
                    self.claim_calls += 1
                    raise ModelFallbackError(
                        ModelRateLimitError("OpenAI rate limit exceeded"),
                        ModelProviderRequestError(
                            "Anthropic",
                            status_code=400,
                            category="billing",
                            error_code="invalid_request_error",
                        ),
                    )
                return await super().generate_structured(
                    response_model=response_model, **kwargs
                )

        gateway = RateLimitedGateway()
        pipeline = LegalRagV2Pipeline(
            gateway=gateway,
            planner=LegalQueryPlanner(gateway),
            retriever=LegalRetriever(FakeBackend()),
            config=LegalRagV2Config(),
        )

        claims, validation = await pipeline._synthesize_and_validate_claims(
            plan.user_query,
            plan,
            [],
            [],
        )

        self.assertEqual(gateway.claim_calls, 1)
        self.assertEqual(len(claims), len(issues))
        self.assertEqual(len(validation.errors), len(issues))
        self.assertFalse(any("batch_split_after" in item for item in validation.warnings))

    async def test_family_synthesis_repairs_missing_verified_claim_coverage(self) -> None:
        question = "Fundacja rodzinna wypłaca świadczenie beneficjentowi."
        issue = LegalIssue(
            issue_id="family_foundation_cit_hidden_profit",
            label="fundacja rodzinna: CIT 24q / ukryte zyski / świadczenia",
            tax_domains=["CIT", "UFR"],
            legal_mechanism="family_foundation",
        )
        plan = LegalResearchPlan(
            user_query=question,
            intent=ResearchIntent(mode="mixed_analysis"),
            issues=[issue],
            clarification=Clarification(),
            confidence=0.9,
        )
        ufr = ProvisionReference(
            provision_id="ufr-2-2",
            document_id="pl-ustawa-o-fundacji-rodzinnej-art-2",
            citation="art. 2 ust. 2",
            text="Świadczenie oznacza składniki majątkowe przekazane beneficjentowi.",
            status="active",
            source_span=DocumentSourceSpan(
                start=0,
                end=66,
                document_id="pl-ustawa-o-fundacji-rodzinnej-art-2",
            ),
        )
        cit = ProvisionReference(
            provision_id="cit-24q-1",
            document_id="pl-ustawa-o-podatku-dochodowym-od-osob-prawnych-art-24q",
            citation="art. 24q ust. 1",
            text="Podatek dochodowy od świadczenia wynosi 15% podstawy opodatkowania.",
            status="active",
            source_span=DocumentSourceSpan(
                start=0,
                end=69,
                document_id="pl-ustawa-o-podatku-dochodowym-od-osob-prawnych-art-24q",
            ),
        )
        bundle = EvidenceBundle(
            issue_id=issue.issue_id,
            controlling_provisions=[ufr],
            dependency_provisions=[cit],
            coverage_status="complete",
        )

        class CoverageRepairGateway(FakeGateway):
            claim_calls = 0

            async def generate_structured(self, *, response_model, input, **kwargs):
                if response_model is not ClaimSet:
                    return await super().generate_structured(
                        response_model=response_model, input=input, **kwargs
                    )
                self.claim_calls += 1
                payload = pipeline_module.json.loads(input)
                if "completion_request" not in payload:
                    provision = ufr
                    claim_id = "definition_only"
                    text = "Świadczenie obejmuje składniki przekazane beneficjentowi."
                else:
                    provision = cit
                    claim_id = "completed_tax_charge"
                    text = "Art. 24q ust. 1 ustanawia podatek w wysokości 15%."
                return ClaimSet(
                    claims=[
                        LegalClaim(
                            claim_id=claim_id,
                            issue_id=issue.issue_id,
                            claim_type="normative_rule",
                            text=text,
                            status="approved",
                            result="Reguła wynika z podanej jednostki.",
                            controlling_provision_ids=[provision.provision_id],
                            source_spans=[provision.source_span],
                            confidence=0.9,
                        )
                    ]
                )

        gateway = CoverageRepairGateway()
        pipeline = LegalRagV2Pipeline(
            gateway=gateway,
            planner=LegalQueryPlanner(gateway),
            retriever=LegalRetriever(FakeBackend()),
            config=LegalRagV2Config(),
        )

        claims, validation = await pipeline._synthesize_and_validate_claims(
            question, plan, [bundle], []
        )

        self.assertTrue(validation.passed)
        self.assertEqual(2, gateway.claim_calls)
        self.assertEqual(
            {ufr.provision_id, cit.provision_id},
            {provision_id for claim in claims for provision_id in claim.controlling_provision_ids},
        )
        self.assertTrue(any("claim_coverage_repair_applied" in item for item in validation.warnings))

    def test_writer_integrity_accepts_multiple_verified_citations_for_one_source_id(self) -> None:
        article_21 = ProvisionReference(
            provision_id="cit-source",
            document_id="cit-act",
            citation="art. 21 ust. 1 pkt 1",
            status="active",
        )
        article_26 = ProvisionReference(
            provision_id="cit-source",
            document_id="cit-act",
            citation="art. 26 ust. 2e",
            status="active",
        )
        output = WriterOutput(
            thesis="Wniosek.",
            sources=[
                WriterSource(
                    source_id="cit-source",
                    label="Ustawa o CIT",
                    citation="art. 21 ust. 1 pkt 1",
                ),
                WriterSource(
                    source_id="cit-source",
                    label="Ustawa o CIT",
                    citation="art. 26 ust. 2e",
                ),
            ],
        )
        bundles = [
            EvidenceBundle(
                issue_id="pit_sale",
                controlling_provisions=[article_21, article_26],
                coverage_status="complete",
            )
        ]

        self.assertEqual(
            [],
            validate_writer_output(output, answer_plan=pipeline_module.AnswerPlan(), bundles=bundles),
        )

    def test_claim_validator_blocks_unbound_model_provision_labels(self) -> None:
        provision = ProvisionReference(
            provision_id="pit-art-10",
            document_id="pit-act",
            citation="art. 10 ust. 1",
            status="active",
            source_span=DocumentSourceSpan(start=0, end=1, document_id="pit-act"),
        )
        claim = LegalClaim(
            claim_id="approved-claim",
            issue_id="pit_sale",
            claim_type="application",
            text="Art. 30e ust. 1 ma zastosowanie do sprzedaży.",
            result="Zastosuj art. 30e ust. 1 po weryfikacji faktów.",
            status="approved",
            controlling_provision_ids=[provision.provision_id],
            source_spans=[provision.source_span],
            confidence=0.8,
        )
        plan = research_plan()
        bundles = [
            EvidenceBundle(
                issue_id="pit_sale",
                controlling_provisions=[provision],
                coverage_status="complete",
            )
        ]
        validated, errors, _ = pipeline_module.validate_claims(
            [claim],
            plan=plan,
            bundles=bundles,
            calculations=[],
        )
        self.assertTrue(any("unbound_textual_provision_reference" in item for item in errors))
        self.assertEqual("blocked_invalid_provision", validated[0].status)

        answer_plan = pipeline_module._build_answer_plan(plan, validated, [])
        output = pipeline_module._deterministic_writer_output(
            {
                "validated_claims": validated,
                "legal_research_plan": plan,
                "evidence_bundles": bundles,
                "answer_plan": answer_plan,
            }
        )

        self.assertIn("Brak materialnej konkluzji", output.thesis)
        self.assertNotIn("właściwy przepis", output.thesis)
        self.assertNotIn("art. 30e", output.thesis.casefold())
        self.assertEqual([], validate_writer_output(output, answer_plan=answer_plan, bundles=bundles))

    def test_whole_article_binding_cannot_support_invented_point(self) -> None:
        provision = ProvisionReference(
            provision_id="cit-art-11n",
            document_id="pl-ustawa-o-podatku-dochodowym-od-osob-prawnych-art-11n",
            citation="art. 11n",
            status="active",
            source_span=DocumentSourceSpan(
                start=0,
                end=10,
                document_id="pl-ustawa-o-podatku-dochodowym-od-osob-prawnych-art-11n",
            ),
        )
        claim = LegalClaim(
            claim_id="invented-child",
            issue_id="pit_sale",
            claim_type="normative_rule",
            text="Zwolnienie wynika z art. 11n pkt 1.",
            result="Należy zastosować punkt pierwszy.",
            status="approved",
            controlling_provision_ids=[provision.provision_id],
            source_spans=[provision.source_span],
            confidence=0.8,
        )
        bundle = EvidenceBundle(
            issue_id="pit_sale",
            controlling_provisions=[provision],
            coverage_status="complete",
        )

        validated, errors, _ = pipeline_module.validate_claims(
            [claim], plan=research_plan(), bundles=[bundle], calculations=[]
        )

        self.assertEqual("blocked_invalid_provision", validated[0].status)
        self.assertTrue(any("unbound_textual_provision_reference" in item for item in errors))

    def test_claim_validator_auto_binds_unique_exact_retrieved_point(self) -> None:
        article = ProvisionReference(
            provision_id="cit-art-11n",
            document_id="cit-art-11n-document",
            citation="art. 11n",
            status="active",
            source_span=DocumentSourceSpan(
                start=0,
                end=10,
                document_id="cit-art-11n-document",
            ),
        )
        point = ProvisionReference(
            provision_id="cit-art-11n-point-1",
            document_id="cit-art-11n-document",
            citation="art. 11n pkt 1",
            text="Warunki zwolnienia dokumentacyjnego określone w punkcie 1.",
            status="active",
            source_span=DocumentSourceSpan(
                start=11,
                end=68,
                document_id="cit-art-11n-document",
            ),
        )
        claim = LegalClaim(
            claim_id="exact-child-in-bundle",
            issue_id="pit_sale",
            claim_type="normative_rule",
            text="Zwolnienie dokumentacyjne wynika z art. 11n pkt 1.",
            result="Należy zbadać wszystkie warunki punktu pierwszego.",
            status="approved",
            controlling_provision_ids=[article.provision_id],
            source_spans=[article.source_span],
            confidence=0.9,
        )
        bundle = EvidenceBundle(
            issue_id="pit_sale",
            controlling_provisions=[article, point],
            coverage_status="complete",
        )

        validated, errors, warnings = pipeline_module.validate_claims(
            [claim], plan=research_plan(), bundles=[bundle], calculations=[]
        )

        self.assertEqual([], errors)
        self.assertEqual("approved", validated[0].status)
        self.assertEqual(
            [article.provision_id, point.provision_id],
            validated[0].controlling_provision_ids,
        )
        self.assertIn(point.source_span, validated[0].source_spans)
        self.assertTrue(any("auto_bound_exact_provision:art. 11n pkt 1" in item for item in warnings))

    def test_claim_validator_does_not_auto_bind_ambiguous_exact_reference(self) -> None:
        bound_article = ProvisionReference(
            provision_id="first-act-art-6",
            document_id="first-act",
            citation="art. 6",
            status="active",
            source_span=DocumentSourceSpan(start=2, end=3, document_id="first-act"),
        )
        first = ProvisionReference(
            provision_id="first-act-art-5",
            document_id="first-act",
            citation="art. 5 ust. 1",
            status="active",
            source_span=DocumentSourceSpan(start=0, end=1, document_id="first-act"),
        )
        second = ProvisionReference(
            provision_id="second-act-art-5",
            document_id="second-act",
            citation="art. 5 ust. 1",
            status="active",
            source_span=DocumentSourceSpan(start=0, end=1, document_id="second-act"),
        )
        claim = LegalClaim(
            claim_id="ambiguous-exact-reference",
            issue_id="pit_sale",
            claim_type="normative_rule",
            text="Reguła wynika z art. 5 ust. 1.",
            result="Należy zastosować wskazaną regułę.",
            status="approved",
            controlling_provision_ids=[bound_article.provision_id],
            source_spans=[bound_article.source_span],
            confidence=0.8,
        )
        bundle = EvidenceBundle(
            issue_id="pit_sale",
            controlling_provisions=[bound_article, first, second],
            coverage_status="complete",
        )

        validated, errors, warnings = pipeline_module.validate_claims(
            [claim], plan=research_plan(), bundles=[bundle], calculations=[]
        )

        self.assertEqual("blocked_invalid_provision", validated[0].status)
        self.assertTrue(any("unbound_textual_provision_reference" in item for item in errors))
        self.assertFalse(any("auto_bound_exact_provision" in item for item in warnings))

    def test_claim_validator_auto_binds_both_ends_of_retrieved_provision_range(self) -> None:
        article_12_5 = ProvisionReference(
            provision_id="cit-art-12-5",
            document_id="pl-ustawa-o-podatku-dochodowym-od-osob-prawnych-art-12",
            citation="art. 12 ust. 5",
            status="active",
            source_span=DocumentSourceSpan(
                start=0,
                end=1,
                document_id="pl-ustawa-o-podatku-dochodowym-od-osob-prawnych-art-12",
            ),
        )
        article_12_6a = article_12_5.model_copy(
            update={"provision_id": "cit-art-12-6a", "citation": "art. 12 ust. 6a"}
        )
        bundle = EvidenceBundle(
            issue_id="pit_sale",
            controlling_provisions=[article_12_5, article_12_6a],
            coverage_status="complete",
        )
        claim = LegalClaim(
            claim_id="range-claim",
            issue_id="pit_sale",
            claim_type="normative_rule",
            text="Wartość ustala się zgodnie z art. 12 ust. 5–6a.",
            result="Należy zastosować wskazany przedział jednostek.",
            status="approved",
            controlling_provision_ids=[article_12_5.provision_id],
            source_spans=[article_12_5.source_span],
            confidence=0.9,
        )

        approved, errors, warnings = pipeline_module.validate_claims(
            [claim], plan=research_plan(), bundles=[bundle], calculations=[]
        )
        self.assertEqual([], errors)
        self.assertEqual("approved", approved[0].status)
        self.assertEqual(
            [article_12_5.provision_id, article_12_6a.provision_id],
            approved[0].controlling_provision_ids,
        )
        self.assertTrue(any("auto_bound_exact_provision:art. 12 ust. 6a" in item for item in warnings))

        complete_claim = claim.model_copy(
            update={
                "controlling_provision_ids": [
                    article_12_5.provision_id,
                    article_12_6a.provision_id,
                ],
                "source_spans": [article_12_5.source_span, article_12_6a.source_span],
            }
        )
        approved, errors, _ = pipeline_module.validate_claims(
            [complete_claim], plan=research_plan(), bundles=[bundle], calculations=[]
        )
        self.assertEqual([], errors)
        self.assertEqual("approved", approved[0].status)

    def test_claim_validator_rejects_denial_of_rate_present_in_bound_law(self) -> None:
        provision = ProvisionReference(
            provision_id="cit-art-24q-1",
            document_id="pl-ustawa-o-podatku-dochodowym-od-osob-prawnych-art-24q",
            citation="art. 24q ust. 1",
            text="Podatek dochodowy wynosi 15 % podstawy opodatkowania.",
            status="active",
            source_span=DocumentSourceSpan(
                start=0,
                end=58,
                document_id="pl-ustawa-o-podatku-dochodowym-od-osob-prawnych-art-24q",
            ),
        )
        plan = research_plan()
        claim = LegalClaim(
            claim_id="rate-denial",
            issue_id="pit_sale",
            claim_type="normative_rule",
            text="Z art. 24q ust. 1 nie wynika jednak stawka podatku.",
            result="Stawka pozostaje nieustalona.",
            status="approved",
            controlling_provision_ids=[provision.provision_id],
            source_spans=[provision.source_span],
            confidence=0.9,
        )
        bundle = EvidenceBundle(
            issue_id="pit_sale",
            controlling_provisions=[provision],
            coverage_status="complete",
        )

        validated, errors, _ = pipeline_module.validate_claims(
            [claim], plan=plan, bundles=[bundle], calculations=[]
        )

        self.assertEqual("blocked_invalid_provision", validated[0].status)
        self.assertTrue(any("claim_denies_rate" in item for item in errors))

    def test_deterministic_risks_do_not_expose_internal_dependency_ids(self) -> None:
        plan = research_plan()
        bundle = EvidenceBundle(
            issue_id="pit_sale",
            missing_sources=["required_primary:pit_internal_dependency_id"],
            coverage_status="partial",
        )
        answer_plan = pipeline_module._build_answer_plan(plan, [], [])

        output = pipeline_module._deterministic_writer_output(
            {
                "validated_claims": [],
                "legal_research_plan": plan,
                "evidence_bundles": [bundle],
                "answer_plan": answer_plan,
            }
        )

        rendered_risks = " ".join(output.risks_and_gaps)
        self.assertIn("kompletnego zestawu przepisów", rendered_risks)
        self.assertNotIn("required_primary", rendered_risks)
        self.assertNotIn("internal_dependency_id", rendered_risks)


if __name__ == "__main__":
    unittest.main()
