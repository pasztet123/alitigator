from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from app.hybrid_authority_rag import (
    AuthorityCard,
    EvidenceBundle,
    HybridAuthorityConfig,
    PrimaryLaneResult,
    bind_claims_to_evidence_bundles,
    build_authority_evidence_context,
    build_authority_queries,
    build_evidence_bundles,
    build_fact_graph,
    build_issue_graph,
    build_primary_queries,
    build_retrieval_clarification,
    classify_legal_research_intent,
    extract_authority_card,
    get_legal_retrieval_mode,
    prefilter_authority_card,
    run_hybrid_authority_retrieval,
    score_authority_card,
)
from app.legal_pipeline import LegalClaim
from app.rag import RagChunk, RetrievalInspection


def chunk(
    *,
    chunk_id: str = "c1",
    document_id: str = "d1",
    source_type: str = "interpretation",
    source_subtype: str = "individual",
    signature: str = "0111-KDIB1.4010.1.2026.1.AA",
    subject: str = "WHT od odsetek",
    text: str = "",
    tax: str = "CIT",
    provisions: list[str] | None = None,
    published_date: str = "2026-01-15",
) -> RagChunk:
    return RagChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        score=1.0,
        chunk_text=text
        or (
            "Państwa stanowisko w sprawie jest takie, że odsetki korzystają ze zwolnienia. "
            "Ocena stanowiska: Państwa stanowisko jest prawidłowe. Uzasadnienie interpretacji."
        ),
        subject=subject,
        signature=signature,
        published_date=published_date,
        source_url="https://example.test/doc",
        category=None,
        source_type=source_type,
        source_subtype=source_subtype,
        authority="Naczelny Sąd Administracyjny" if source_type == "judgment" else "Dyrektor KIS",
        legal_state_date=published_date[:10],
        legal_provisions=provisions or [f"[{tax}] Ustawa o CIT-art. 21-ust. 1"],
    )


def inspection(chunks: list[RagChunk]) -> RetrievalInspection:
    return RetrievalInspection(
        query="q",
        match_query="q",
        requested_limit=10,
        retrieved_count=len(chunks),
        selected_count=len(chunks),
        selected_context_chars=sum(len(item.chunk_text) for item in chunks),
        hits=[
            {
                "rank": rank,
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "chunk_index": item.chunk_index,
                "score": item.score,
                "canonical_source_id": item.document_id,
                "evidence_role": "authority_assessment",
                "subject": item.subject,
                "signature": item.signature,
                "published_date": item.published_date,
                "source_url": item.source_url,
                "category": item.category,
                "source": item.source,
                "source_type": item.source_type,
                "source_subtype": item.source_subtype,
                "authority": item.authority,
                "publication": item.publication,
                "legal_state_date": item.legal_state_date,
                "source_pages": item.source_pages,
                "legal_provisions": item.legal_provisions,
                "chunk_chars": len(item.chunk_text),
                "preview": item.chunk_text[:120],
                "selected_for_context": True,
            }
            for rank, item in enumerate(chunks, start=1)
        ],
        chunks=chunks,
        raw_candidate_pool=[],
    )


class HybridAuthorityIntentTests(unittest.TestCase):
    def test_intent_classification_uses_weighted_profiles(self) -> None:
        research = classify_legal_research_intent("Znajdź interpretacje i wyroki dotyczące look-through approach")
        mixed = classify_legal_research_intent("Jak należy rozliczyć sprzedaż udziałów w ASI?")
        rule = classify_legal_research_intent("Jaka jest aktualna reguła z ustawy o VAT?")

        self.assertEqual(research.answer_mode, "authority_research")
        self.assertGreater(research.authority_weight, research.primary_law_weight)
        self.assertEqual(mixed.answer_mode, "mixed_analysis")
        self.assertEqual(rule.answer_mode, "rule_first")
        self.assertTrue(rule.needs_interpretations)

    def test_clarifier_gating_uses_fixture_answers_without_inventing_facts(self) -> None:
        intent = classify_legal_research_intent("Jak rozliczyć WHT?")
        disabled = build_retrieval_clarification("Jak rozliczyć WHT?", intent, enabled=False)
        fixture = build_retrieval_clarification(
            "Jak rozliczyć WHT?",
            intent,
            enabled=True,
            fixture_answers={"payment_type": "odsetki"},
        )
        questions = build_retrieval_clarification("Jak rozliczyć WHT?", intent, enabled=True)

        self.assertFalse(disabled.should_ask)
        self.assertIn("payment_type: odsetki", fixture.augmented_query)
        self.assertEqual(fixture.mode, "fixture")
        self.assertTrue(questions.should_ask)
        self.assertLessEqual(len(questions.questions), 3)

    def test_fact_graph_preserves_roles(self) -> None:
        graph = build_fact_graph("Fundator wypłaca świadczenie beneficjentowi przez fundację rodzinną.")

        self.assertIn("founder", graph.roles)
        self.assertIn("beneficiary", graph.roles)
        self.assertIn("benefit", graph.payments)

    def test_issue_graph_decomposes_existing_axes_or_fallback(self) -> None:
        intent = classify_legal_research_intent("Jak rozliczyć WHT od odsetek i usług doradczych?")
        graph = build_fact_graph("Jak rozliczyć WHT od odsetek i usług doradczych?")
        issues = build_issue_graph("Jak rozliczyć WHT od odsetek i usług doradczych?", intent, graph)

        self.assertGreaterEqual(len(issues), 1)
        self.assertTrue(all(issue.issue_id for issue in issues))


class HybridAuthorityRetrievalUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = HybridAuthorityConfig(authority_card_cache_enabled=False)
        self.intent = classify_legal_research_intent("Jak rozliczyć WHT od odsetek?")
        self.fact_graph = build_fact_graph("Jak rozliczyć WHT od odsetek?")
        self.issue = build_issue_graph("Jak rozliczyć WHT od odsetek?", self.intent, self.fact_graph)[0]
        self.primary_chunk = chunk(
            chunk_id="s1",
            document_id="statute-cit-21",
            source_type="statute",
            source_subtype="consolidated_text",
            signature="ustawa CIT art. 21",
            subject="Ustawa o CIT art. 21",
            provisions=["[CIT] Ustawa o CIT-art. 21-ust. 1"],
        )
        self.primary = PrimaryLaneResult(
            issue_id=self.issue.issue_id,
            target_date="2026-01-01",
            queries=(),
            controlling_provisions=(self.primary_chunk,),
        )

    def test_provision_anchored_query_generation(self) -> None:
        queries = build_authority_queries(self.issue, "Jak rozliczyć WHT od odsetek?", self.fact_graph, self.primary)
        families = {item["family"] for item in queries}

        self.assertIn("natural_language", families)
        self.assertIn("issue_signature", families)
        self.assertIn("provision_anchored", families)
        self.assertTrue(any("art. 21" in item["query"] for item in queries))

    def test_primary_query_generation_keeps_issue_signature(self) -> None:
        queries = build_primary_queries(self.issue, "Jak rozliczyć WHT od odsetek?", self.fact_graph)

        self.assertTrue(any(item["family"] == "issue_signature" for item in queries))

    def test_authority_card_extraction_separates_taxpayer_and_authority(self) -> None:
        card = extract_authority_card(chunk(), target_date="2026-01-01", config=self.config)

        self.assertIn("Państwa stanowisko", card.taxpayer_position or "")
        self.assertIn("Ocena stanowiska", card.authority_holding or "")
        self.assertNotEqual(card.taxpayer_position, card.authority_holding)
        self.assertIn("taxpayer_position", card.source_spans)
        self.assertIn("authority_holding", card.source_spans)

    def test_contrary_authority_is_not_misread_as_favorable(self) -> None:
        negative = chunk(
            text="Państwa stanowisko w sprawie jest korzystne. Ocena stanowiska: Państwa stanowisko jest nieprawidłowe. Uzasadnienie interpretacji."
        )
        card = extract_authority_card(negative, target_date="2026-01-01", config=self.config)

        self.assertEqual(card.result_for_taxpayer, "unfavorable")

    def test_temporal_filter_penalizes_historical_documents(self) -> None:
        historical = chunk(published_date="2012-01-01")
        card = extract_authority_card(historical, target_date="2026-01-01", config=self.config)
        status, reasons = prefilter_authority_card(card, self.issue, self.intent, self.fact_graph, self.primary)

        self.assertEqual(card.temporal_status, "historical")
        self.assertEqual(status, "penalized")
        self.assertIn("historical_document", reasons)

    def test_wrong_neighbor_penalty_for_wht_interest_vs_services(self) -> None:
        services = chunk(
            subject="WHT od usług doradczych",
            text="Ocena stanowiska: usługi doradcze i zarządzania mieszczą się w analizowanym świadczeniu. Państwa stanowisko jest prawidłowe.",
            provisions=["[CIT] Ustawa o CIT-art. 21-ust. 1"],
        )
        issue = replace(self.issue, contrast="interest_vs_advisory_or_management_services")
        card = extract_authority_card(services, target_date="2026-01-01", config=self.config)
        score = score_authority_card(
            card,
            issue=issue,
            fact_graph=self.fact_graph,
            primary_result=self.primary,
            family_score=0.05,
            filter_status="penalized",
        )

        self.assertIn("wrong neighbor", score.negative_reasons)
        self.assertGreater(score.dimensions["wrong_neighbor_penalty"], 0)

    def test_evidence_bundle_limits_and_roles(self) -> None:
        good_card = extract_authority_card(chunk(document_id="good"), target_date="2026-01-01", config=self.config)
        bad_card = extract_authority_card(
            chunk(
                document_id="bad",
                text="Ocena stanowiska: Państwa stanowisko jest nieprawidłowe. Uzasadnienie interpretacji.",
            ),
            target_date="2026-01-01",
            config=self.config,
        )
        scored = []
        for card, doc_id, score_value in [(good_card, "good", 0.9), (bad_card, "bad", 0.85)]:
            scored.append(
                _scored_authority(
                    self.issue.issue_id,
                    card,
                    chunk(document_id=doc_id),
                    score_value,
                )
            )

        bundles = build_evidence_bundles((self.issue,), (self.primary,), scored, config=self.config)

        self.assertEqual(len(bundles[0].controlling_provisions), 1)
        self.assertEqual(len(bundles[0].supporting_authorities), 1)
        self.assertEqual(len(bundles[0].contrary_authorities), 1)
        self.assertLessEqual(len(bundles[0].supporting_authorities), self.config.authority_selected_limit_per_issue)

    def test_claim_binding_to_supporting_authorities(self) -> None:
        claim = LegalClaim(
            claim_id="claim_wht",
            axis_id=self.issue.issue_id,
            claim_type="legal_rule",
            text="WHT claim",
            source_provisions=("cit_art_21",),
        )
        bundle = EvidenceBundle(
            issue_id=self.issue.issue_id,
            supporting_authorities=({"signature": "0111", "source_spans": {"chunk_text": {"start": 0, "end": 10}}},),
            retrieval_confidence=0.82,
        )

        bound = bind_claims_to_evidence_bundles([claim], [bundle])[0]

        self.assertEqual(bound.supporting_authorities[0]["signature"], "0111")
        self.assertEqual(bound.authority_confidence, 0.82)

    def test_renderer_context_contains_signatures_and_missing_authority_message(self) -> None:
        with patch("app.hybrid_authority_rag.inspect_search", return_value=inspection([self.primary_chunk])):
            result = run_hybrid_authority_retrieval(
                "Jaka jest podstawa prawna WHT od odsetek?",
                include_interpretations=False,
                include_judgments=False,
                config=self.config,
            )

        context = build_authority_evidence_context(result)

        self.assertIn("W przeszukanym zbiorze", context)
        self.assertIn("issue=", context)

    def test_housing_relief_pre_2022_authority_is_historical(self) -> None:
        issue = replace(
            self.issue,
            issue_id="pit_housing_relief",
            query="Spłata kredytu zaciągniętego na zbywaną nieruchomość.",
            tax="PIT",
            mechanism="housing_relief",
            contrast="credit_on_sold_property_vs_credit_on_new_property",
        )
        old_authority = chunk(
            document_id="old",
            tax="PIT",
            subject="Ulga mieszkaniowa - kredyt na zbywaną nieruchomość",
            text=(
                "Stan faktyczny: spłata kredytu zaciągniętego na zbywaną nieruchomość. "
                "Ocena stanowiska: Państwa stanowisko jest nieprawidłowe. Uzasadnienie interpretacji."
            ),
            provisions=["[PIT] Ustawa o PIT-art. 21-ust. 30"],
            published_date="2021-12-31",
        )
        card = extract_authority_card(old_authority, target_date="2026-01-01", config=self.config)

        status, reasons = prefilter_authority_card(card, issue, self.intent, self.fact_graph, self.primary)

        self.assertEqual(status, "penalized")
        self.assertIn("pre_material_amendment", reasons)
        bundles = build_evidence_bundles(
            (issue,),
            (
                replace(
                    self.primary,
                    issue_id=issue.issue_id,
                    controlling_provisions=(self._housing_full_primary_chunk(),),
                ),
            ),
            [
                _scored_authority(
                    issue.issue_id,
                    card,
                    old_authority,
                    0.9,
                    filter_status=status,
                    filter_reasons=tuple(reasons),
                )
            ],
            config=self.config,
        )
        self.assertEqual(bundles[0].historical_authorities[0]["temporal_status"], "historical")
        self.assertFalse(bundles[0].supporting_authorities)

    def test_incomplete_housing_primary_bundle_suppresses_authority_merge(self) -> None:
        issue = replace(
            self.issue,
            issue_id="pit_housing_relief",
            query="Spłata kredytu zaciągniętego na zbywaną nieruchomość.",
            tax="PIT",
            mechanism="housing_relief",
            contrast="credit_on_sold_property_vs_credit_on_new_property",
        )
        incomplete_primary = chunk(
            chunk_id="pit30",
            document_id="pit_act",
            source_type="statute",
            source_subtype="consolidated_text",
            signature="ustawa PIT art. 21 ust. 30",
            subject="Ustawa PIT art. 21 ust. 30",
            provisions=["[PIT] Ustawa o PIT-art. 21-ust. 30"],
            text="art. 21 ust. 30 ustawy PIT",
            tax="PIT",
        )
        authority = chunk(
            document_id="current-auth",
            tax="PIT",
            subject="Ulga mieszkaniowa - kredyt na zbywaną nieruchomość",
            text=(
                "Stan faktyczny: spłata kredytu zaciągniętego na zbywaną nieruchomość. "
                "Ocena stanowiska: Państwa stanowisko jest prawidłowe. Uzasadnienie interpretacji art. 21 ust. 30a."
            ),
            provisions=["[PIT] Ustawa o PIT-art. 21-ust. 30a"],
            published_date="2026-01-15",
        )
        card = extract_authority_card(authority, target_date="2026-01-01", config=self.config)
        bundles = build_evidence_bundles(
            (issue,),
            (
                PrimaryLaneResult(
                    issue_id=issue.issue_id,
                    target_date="2026-01-01",
                    queries=(),
                    controlling_provisions=(incomplete_primary,),
                ),
            ),
            [_scored_authority(issue.issue_id, card, authority, 0.92)],
            config=self.config,
        )

        self.assertFalse(bundles[0].supporting_authorities)
        self.assertTrue(
            any(item.startswith("primary_bundle_incomplete") for item in bundles[0].missing_source_requirements)
        )

    def test_authority_citing_current_special_rule_is_boosted(self) -> None:
        issue = replace(
            self.issue,
            issue_id="pit_housing_relief",
            query="Spłata kredytu zaciągniętego na zbywaną nieruchomość.",
            tax="PIT",
            mechanism="housing_relief",
            contrast="credit_on_sold_property_vs_credit_on_new_property",
        )
        primary = replace(
            self.primary,
            issue_id=issue.issue_id,
            controlling_provisions=(self._housing_full_primary_chunk(),),
        )
        with_rule = extract_authority_card(
            chunk(
                document_id="with30a",
                tax="PIT",
                subject="Ulga mieszkaniowa - kredyt na zbywaną nieruchomość",
                text=(
                    "Stan faktyczny: spłata kredytu na zbywaną nieruchomość. "
                    "Ocena stanowiska: Państwa stanowisko jest prawidłowe na podstawie art. 21 ust. 30a."
                ),
                provisions=["[PIT] Ustawa o PIT-art. 21-ust. 30a"],
            ),
            target_date="2026-01-01",
            config=self.config,
        )
        without_rule = extract_authority_card(
            chunk(
                document_id="without30a",
                tax="PIT",
                subject="Ulga mieszkaniowa - kredyt na zbywaną nieruchomość",
                text=(
                    "Stan faktyczny: spłata kredytu na zbywaną nieruchomość. "
                    "Ocena stanowiska: Państwa stanowisko jest prawidłowe."
                ),
                provisions=["[PIT] Ustawa o PIT-art. 21-ust. 30"],
            ),
            target_date="2026-01-01",
            config=self.config,
        )

        boosted = score_authority_card(
            with_rule,
            issue=issue,
            fact_graph=self.fact_graph,
            primary_result=primary,
            family_score=0.0,
            filter_status="kept",
        )
        baseline = score_authority_card(
            without_rule,
            issue=issue,
            fact_graph=self.fact_graph,
            primary_result=primary,
            family_score=0.0,
            filter_status="kept",
        )

        self.assertGreater(boosted.score, baseline.score)
        self.assertIn("cites current special rule", boosted.positive_reasons)

    def _housing_full_primary_chunk(self) -> RagChunk:
        return chunk(
            chunk_id="pit-full",
            document_id="pit_act",
            source_type="statute",
            source_subtype="consolidated_text",
            signature="ustawa PIT art. 21",
            subject="Ustawa PIT art. 21 ust. 25 pkt 2, ust. 30 i ust. 30a",
            text="art. 21 ust. 25 pkt 2 ustawy PIT; art. 21 ust. 30 ustawy PIT; art. 21 ust. 30a ustawy PIT",
            provisions=[
                "[PIT] Ustawa o PIT-art. 21-ust. 25-pkt 2",
                "[PIT] Ustawa o PIT-art. 21-ust. 30",
                "[PIT] Ustawa o PIT-art. 21-ust. 30a",
            ],
            tax="PIT",
        )


class HybridAuthorityIntegrationTests(unittest.TestCase):
    def test_retrieval_mode_defaults_to_baseline_and_flag_enables_hybrid(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LEGAL_RETRIEVAL_MODE", None)
            self.assertEqual(get_legal_retrieval_mode(), "baseline")
        with patch.dict(os.environ, {"LEGAL_RETRIEVAL_MODE": "hybrid_authority"}):
            self.assertEqual(get_legal_retrieval_mode(), "hybrid_authority")

    def test_same_request_can_run_hybrid_with_fixture_clarification(self) -> None:
        statute = chunk(
            chunk_id="s1",
            document_id="statute",
            source_type="statute",
            source_subtype="consolidated_text",
            signature="ustawa CIT",
            subject="Ustawa CIT art. 21",
        )
        authority = chunk(document_id="auth", signature="0111-KDIB1.4010.1.2026.1.AA")

        def fake_inspect(_query: str, **kwargs: object) -> RetrievalInspection:
            source_types = kwargs.get("source_types")
            if source_types == {"statute"}:
                return inspection([statute])
            return inspection([authority])

        with patch("app.hybrid_authority_rag.inspect_search", side_effect=fake_inspect):
            result = run_hybrid_authority_retrieval(
                "Jak rozliczyć WHT?",
                clarifier_enabled=True,
                clarification_fixture={"payment_type": "odsetki"},
                config=HybridAuthorityConfig(authority_card_cache_enabled=False),
            )

        self.assertIn("payment_type: odsetki", result.retrieval_query)
        self.assertTrue(result.evidence_bundles)
        self.assertTrue(result.evidence_bundles[0].supporting_authorities)

    def test_no_authority_does_not_remove_primary_law_bundle(self) -> None:
        statute = chunk(source_type="statute", source_subtype="consolidated_text", signature="ustawa CIT")

        with patch("app.hybrid_authority_rag.inspect_search", return_value=inspection([statute])):
            result = run_hybrid_authority_retrieval(
                "Jaki przepis CIT reguluje WHT od odsetek?",
                include_interpretations=False,
                include_judgments=False,
                config=HybridAuthorityConfig(authority_card_cache_enabled=False),
            )

        bundle = result.evidence_bundles[0]
        self.assertTrue(bundle.controlling_provisions)
        self.assertIn("supporting_authority", bundle.missing_source_requirements)

    def test_missing_primary_law_is_explicit_in_bundle(self) -> None:
        authority = chunk(document_id="auth")

        def fake_inspect(_query: str, **kwargs: object) -> RetrievalInspection:
            if kwargs.get("source_types") == {"statute"}:
                return inspection([])
            return inspection([authority])

        with patch("app.hybrid_authority_rag.inspect_search", side_effect=fake_inspect):
            result = run_hybrid_authority_retrieval(
                "Jak rozliczyć WHT od odsetek?",
                config=HybridAuthorityConfig(authority_card_cache_enabled=False),
            )

        self.assertIn("controlling_primary_law", result.evidence_bundles[0].missing_source_requirements)
        self.assertFalse(result.evidence_bundles[0].controlling_provisions)

    def test_historical_document_not_presented_as_current_authority(self) -> None:
        statute = chunk(source_type="statute", source_subtype="consolidated_text", signature="ustawa CIT")
        historical = chunk(document_id="old", published_date="2011-01-01")

        def fake_inspect(_query: str, **kwargs: object) -> RetrievalInspection:
            if kwargs.get("source_types") == {"statute"}:
                return inspection([statute])
            return inspection([historical])

        with patch("app.hybrid_authority_rag.inspect_search", side_effect=fake_inspect):
            result = run_hybrid_authority_retrieval(
                "Jak rozliczyć WHT od odsetek?",
                config=HybridAuthorityConfig(authority_card_cache_enabled=False),
            )

        self.assertEqual(result.evidence_bundles[0].historical_authorities[0]["temporal_status"], "historical")


def _scored_authority(
    issue_id: str,
    card: AuthorityCard,
    source_chunk: RagChunk,
    score: float,
    *,
    filter_status: str = "kept",
    filter_reasons: tuple[str, ...] = (),
):
    from app.hybrid_authority_rag import RerankScore, ScoredAuthority

    return ScoredAuthority(
        issue_id=issue_id,
        card=card,
        chunk=source_chunk,
        candidate_rank=1,
        family_scores={"natural_language": 1.0},
        filter_status=filter_status,
        filter_reasons=filter_reasons,
        rerank=RerankScore(score=score, dimensions={}, positive_reasons=("same provision",), negative_reasons=()),
    )


if __name__ == "__main__":
    unittest.main()
