from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from app.legal_rag_v2.authority import (
    AuthorityDocument,
    HeuristicAuthorityExtractor,
    ModelAuthorityExtractor,
    SourceSpanValidationError,
)
from app.legal_rag_v2.embeddings import (
    EmbeddingInput,
    OfflineHashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    VersionedEmbeddingIndex,
)
from app.legal_rag_v2.provision_graph import (
    ProvisionEdge,
    ProvisionGraph,
    ProvisionParser,
)
from app.legal_rag_v2.retrieval import (
    AUTHORITY_SOURCE_TYPES,
    PRIMARY_SOURCE_TYPES,
    LegalRetriever,
    LegacySearchBackendAdapter,
    RetrievalCandidate,
    RetrievalConfig,
    reciprocal_rank_fusion,
)
from app.legal_rag_v2.schemas import (
    AuthorityCard,
    AuthoritySourceSpans,
    Clarification,
    DocumentSourceSpan,
    LegalIssue,
    LegalResearchPlan,
    QueryFamily,
    ResearchIntent,
)


class CountingEmbeddingProvider:
    model = "fake-semantic-v1"
    dimensions = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "norma" in lowered or "ustawa" in lowered else 0.0,
                    1.0 if "organ" in lowered or "interpretacja" in lowered else 0.0,
                    1.0,
                ]
            )
        return vectors


class FakeOpenAIEmbeddings:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ]
        )


class EmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_provider_uses_async_embeddings_and_preserves_input_order(self) -> None:
        resource = FakeOpenAIEmbeddings()
        client = SimpleNamespace(embeddings=resource)
        provider = OpenAIEmbeddingProvider(
            client=client,
            model="text-embedding-3-large",
            dimensions=2,
        )

        vectors = await provider.embed(["pierwszy", "drugi"])

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(resource.kwargs["model"], "text-embedding-3-large")
        self.assertEqual(resource.kwargs["dimensions"], 2)
        self.assertEqual(resource.kwargs["encoding_format"], "float")

    async def test_versioned_index_is_idempotent_resumable_and_queries_cosine(self) -> None:
        provider = CountingEmbeddingProvider()
        with tempfile.TemporaryDirectory() as directory:
            index = VersionedEmbeddingIndex(
                Path(directory) / "vectors.sqlite3",
                provider,
                schema_version="schema-7",
                chunker_version="chunker-4",
            )
            inputs = [
                EmbeddingInput(
                    "p1",
                    "norma z ustawy",
                    {"source_type": "statute", "tax_domains": ["VAT"]},
                ),
                EmbeddingInput(
                    "a1",
                    "interpretacja organu",
                    {"source_type": "interpretation", "tax_domains": ["VAT"]},
                ),
            ]

            first = await index.index(inputs, batch_size=1)
            second = await index.index(inputs, batch_size=1)

            self.assertEqual((first.indexed, first.batches_committed), (2, 2))
            self.assertEqual((second.indexed, second.skipped), (0, 2))
            self.assertEqual(len(provider.calls), 2)
            self.assertEqual(index.count(), 2)

            hits = await index.query(
                "norma ustawowa",
                limit=2,
                metadata_filters={"source_type": "statute"},
            )
            self.assertEqual([item.item_id for item in hits], ["p1"])
            self.assertEqual(hits[0].schema_version, "schema-7")
            self.assertEqual(hits[0].chunker_version, "chunker-4")

            changed = await index.index(
                [EmbeddingInput("p1", "zmieniona norma ustawy", inputs[0].metadata)]
            )
            self.assertEqual(changed.indexed, 1)
            self.assertEqual(index.count(), 2)
            self.assertEqual(index.count(current_only=False), 3)
            index.close()

    async def test_offline_hash_provider_is_only_an_explicit_provider(self) -> None:
        provider = OfflineHashEmbeddingProvider(dimensions=16)
        left, right = await provider.embed(["ten sam tekst", "ten sam tekst"])

        self.assertEqual(left, right)
        self.assertEqual(provider.trace_marker, "explicit_offline_hash_embedding")


class ProvisionGraphTests(unittest.TestCase):
    def test_graph_marks_extension_of_referenced_rule_as_special_rule(self) -> None:
        graph = ProvisionParser().build_graph(
            "Art. 21.\n25. Reguła wydatków.\n30a. Wydatki, o których mowa w ust. 25, obejmują także spłatę kredytu.",
            document_id="pit",
            version_id="current",
        )

        edge = next(edge for edge in graph.edges if edge.relationship == "special_rule_for")
        self.assertEqual(graph.get(edge.source_id).citation, "art. 21 ust. 30a")
        self.assertEqual(graph.get(edge.target_id).citation, "art. 21 ust. 25")

    def test_parser_preserves_editorial_granularity_and_infers_explicit_exception(self) -> None:
        text = (
            "Art. 21. Zwolnienia.\n"
            "1. Reguła podstawowa.\n"
            "1) Pierwszy warunek.\n"
            "a) Szczegółowy warunek.\n"
            "2. Z wyjątkiem art. 21 ust. 1 stosuje się inną regułę.\n"
            "§ 3. Przepis paragrafowy.\n"
        )
        graph = ProvisionParser().build_graph(
            text,
            document_id="act",
            version_id="v1",
            effective_from="2025-01-01",
        )

        citations = {item.citation for item in graph.provisions}
        self.assertIn("art. 21", citations)
        self.assertIn("art. 21 ust. 1 pkt 1 lit. a", citations)
        self.assertIn("art. 21 § 3", citations)
        exception = next(edge for edge in graph.edges if edge.relationship == "exception_to")
        target = graph.get(exception.target_id, "2026-01-01")
        self.assertIsNotNone(target)
        self.assertEqual(target.citation, "art. 21 ust. 1")

    def test_graph_filters_nodes_and_edges_for_target_date(self) -> None:
        parser = ProvisionParser()
        old = parser.parse(
            "Art. 1. Stare brzmienie.",
            document_id="act",
            version_id="old",
            effective_from="2020-01-01",
            effective_to="2024-12-31",
        )[0]
        new = parser.parse(
            "Art. 1. Nowe brzmienie.",
            document_id="act",
            version_id="new",
            effective_from="2025-01-01",
        )[0]
        graph = ProvisionGraph(
            (old, new),
            (ProvisionEdge(old.provision_id, new.provision_id, "temporal_successor"),),
        )

        historical = graph.filter_for_date("2024-06-01")
        current = graph.filter_for_date("2026-06-01")

        self.assertEqual([item.version_id for item in historical.provisions], ["old"])
        self.assertEqual([item.version_id for item in current.provisions], ["new"])
        self.assertEqual(historical.edges, ())
        self.assertEqual(current.edges, ())


def research_plan() -> LegalResearchPlan:
    return LegalResearchPlan(
        user_query="Pytanie użytkownika o rozliczenie",
        intent=ResearchIntent(mode="mixed_analysis"),
        target_date="2026-06-01",
        issues=[
            LegalIssue(
                issue_id="issue-1",
                label="Rozliczenie transakcji",
                tax_domains=["VAT"],
                legal_mechanism="ustalenie skutków podatkowych",
                taxpayer_roles=["sprzedawca"],
                transactions=["sprzedaż"],
                positive_fact_constraints=["zapłata"],
                negative_fact_constraints=["darowizna"],
                query_families=[
                    QueryFamily(
                        family="legal_concept",
                        query="wyłącznie planowana norma",
                        lane="primary_law",
                    ),
                    QueryFamily(
                        family="issue_signature",
                        query="wyłącznie planowany stan faktyczny",
                        lane="authority",
                    ),
                ],
            )
        ],
        clarification=Clarification(),
        confidence=0.9,
    )


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, frozenset[str], Mapping[str, Any]]] = []

    async def search(
        self,
        query: str,
        *,
        limit: int,
        source_types: frozenset[str],
        metadata_filters: Mapping[str, Any],
    ) -> list[RetrievalCandidate]:
        self.calls.append((query, source_types, metadata_filters))
        if source_types == PRIMARY_SOURCE_TYPES:
            return [
                RetrievalCandidate(
                    candidate_id="primary-candidate",
                    document_id="primary-document",
                    chunk_id="primary-chunk",
                    text="Sprzedawca dokonuje sprzedaży i otrzymuje zapłatę.",
                    source_type="statute",
                    metadata={
                        "tax_domains": ["VAT"],
                        "effective_from": "2025-01-01",
                    },
                )
            ]
        return [
            RetrievalCandidate(
                candidate_id="authority-candidate",
                document_id="authority-document",
                chunk_id="authority-chunk",
                text="Organ analizował sprzedaż przez sprzedawcę i zapłatę.",
                source_type="interpretation",
                metadata={"tax_domains": ["VAT"], "legal_state_date": "2026-01-01"},
            )
        ]


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_authority_citations_retry_primary_without_suppressing_authority_lane(self) -> None:
        class BackreferenceBackend:
            def __init__(self) -> None:
                self.calls: list[tuple[str, frozenset[str]]] = []

            async def search(self, query, *, limit, source_types, metadata_filters):
                self.calls.append((query, source_types))
                if source_types == AUTHORITY_SOURCE_TYPES:
                    return [
                        RetrievalCandidate(
                            "authority", "Interpretacja stosuje art. 21 ust. 30a.",
                            "interpretation", document_id="authority", chunk_id="authority",
                        )
                    ]
                if query.casefold().startswith("art. 21 ust. 30a"):
                    return [
                        RetrievalCandidate(
                            "exact-unit", "Art. 21 ust. 30a. Przepis szczególny.",
                            "statute", document_id="pit", chunk_id="pit-30a",
                            metadata={"provision_id": "pit-21-30a", "legal_provisions": ["art. 21 ust. 30a"]},
                        )
                    ]
                return []

        backend = BackreferenceBackend()
        result = await LegalRetriever(backend, config=RetrievalConfig(selected_limit_per_issue=5)).retrieve(research_plan())

        self.assertTrue(result.authorities[0].candidates)
        self.assertEqual(result.primary_law[0].candidates[0].candidate_id, "exact-unit")
        self.assertTrue(any(item.get("event") == "authority_backreference_retry" and item.get("executed") for item in result.trace))

    async def test_dual_lanes_run_per_issue_using_only_plan_query_families(self) -> None:
        backend = FakeBackend()
        retriever = LegalRetriever(
            backend,
            config=RetrievalConfig(selected_limit_per_issue=5),
        )

        result = await retriever.retrieve(research_plan())

        self.assertEqual(len(result.primary_law), 1)
        self.assertEqual(len(result.authorities), 1)
        self.assertEqual(
            {call[0] for call in backend.calls},
            {"wyłącznie planowana norma", "wyłącznie planowany stan faktyczny"},
        )
        self.assertEqual(backend.calls[0][1], PRIMARY_SOURCE_TYPES)
        self.assertEqual(backend.calls[1][1], AUTHORITY_SOURCE_TYPES)
        primary = result.primary_law[0].candidates[0]
        self.assertIn("fusion", primary.component_scores)
        self.assertTrue(primary.positive_reasons)
        self.assertEqual(result.primary_law[0].candidate_count_before_rerank, 1)

    async def test_legacy_adapter_isolated_and_trace_marked(self) -> None:
        calls: list[str] = []

        def search(query: str, *, limit: int, source_types: set[str]) -> list[dict[str, Any]]:
            calls.append(query)
            return [
                {
                    "chunk_id": "chunk",
                    "document_id": "document",
                    "chunk_text": "Treść przepisu",
                    "source_type": "statute",
                    "score": 1.0,
                }
            ]

        adapter = LegacySearchBackendAdapter(search)
        candidates = await adapter.search(
            "neutralne zapytanie",
            limit=3,
            source_types=PRIMARY_SOURCE_TYPES,
            metadata_filters={},
        )

        self.assertEqual(calls, ["neutralne zapytanie"])
        self.assertEqual(candidates[0].backend, "legacy_backend_adapter")

    def test_rrf_fuses_independent_lexical_and_vector_ranks(self) -> None:
        first = RetrievalCandidate("one", "one", "statute")
        second = RetrievalCandidate("two", "two", "statute")

        fused = reciprocal_rank_fusion(
            [
                ("lexical", "natural_language", [first, second]),
                ("vector", "natural_language", [second, first]),
            ]
        )

        self.assertEqual({item.candidate_id for item in fused}, {"one", "two"})
        self.assertEqual(len(fused[0].channel_ranks), 2)


class FakeGateway:
    def __init__(self, response: Any = None, error: Optional[Exception] = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: dict[str, Any] = {}

    async def generate_structured(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class AuthorityExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_extractor_uses_structured_schema_and_validates_spans(self) -> None:
        text = "Organ uznał stanowisko podatnika za prawidłowe."
        span = DocumentSourceSpan(
            start=0,
            end=len(text),
            quote=text,
            source_id="authority_document",
            document_id="source-document",
            chunk_id="source-chunk",
        )
        card = AuthorityCard(
            document_id="source-document",
            document_type="interpretation",
            authority_holding="Stanowisko uznano za prawidłowe.",
            source_spans=AuthoritySourceSpans(authority_holding=[span]),
            extraction_confidence=0.9,
        )
        gateway = FakeGateway(card)
        extractor = ModelAuthorityExtractor(gateway, model="configured-model")

        result = await extractor.extract(
            AuthorityDocument(
                "source-document",
                text,
                "interpretation",
                chunk_id="source-chunk",
            )
        )

        self.assertIs(result.card, card)
        self.assertIs(gateway.kwargs["response_model"], AuthorityCard)
        self.assertEqual(gateway.kwargs["model"], "configured-model")
        self.assertEqual(gateway.kwargs["reasoning_effort"], "low")
        self.assertFalse(result.fallback_used)

    async def test_invalid_model_span_uses_only_explicit_heuristic_fallback(self) -> None:
        text = (
            "Stan faktyczny:\nPodatnik sprzedał towar.\n"
            "Ocena stanowiska:\nStanowisko jest prawidłowe."
        )
        bad_span = DocumentSourceSpan(
            start=0,
            end=4,
            quote="inne",
            source_id="authority_document",
            document_id="source-document",
        )
        bad_card = AuthorityCard(
            document_id="source-document",
            document_type="interpretation",
            authority_holding="Stanowisko jest prawidłowe.",
            source_spans=AuthoritySourceSpans(authority_holding=[bad_span]),
            extraction_confidence=0.9,
        )
        extractor = ModelAuthorityExtractor(
            FakeGateway(bad_card),
            heuristic_fallback=HeuristicAuthorityExtractor(),
        )

        result = await extractor.extract(
            AuthorityDocument("source-document", text, "interpretation")
        )

        self.assertTrue(result.fallback_used)
        self.assertEqual(result.trace["fallback_reason"], "SourceSpanValidationError")
        self.assertIn("heuristic_authority_extractor", result.trace["extractor"])

    async def test_invalid_span_without_opt_in_fallback_is_rejected(self) -> None:
        text = "Krótki tekst."
        span = DocumentSourceSpan(
            start=0,
            end=3,
            quote="XYZ",
            source_id="authority_document",
            document_id="document",
        )
        card = AuthorityCard(
            document_id="document",
            document_type="judgment",
            outcome="Oddala skargę.",
            source_spans=AuthoritySourceSpans(outcome=[span]),
            extraction_confidence=0.8,
        )

        with self.assertRaises(SourceSpanValidationError):
            await ModelAuthorityExtractor(FakeGateway(card)).extract(
                AuthorityDocument("document", text, "judgment")
            )


if __name__ == "__main__":
    unittest.main()
