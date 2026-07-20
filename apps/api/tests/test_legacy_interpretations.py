from types import SimpleNamespace

from app import legacy_interpretations
from app.legacy_july7.rag import RagChunk


def test_july7_interpretation_search_forces_snapshot_sqlite_without_tax_domain(monkeypatch) -> None:
    calls: list[dict] = []
    interpretation = SimpleNamespace(
        source_type="interpretation",
        document_id="interpretation-1",
        subject="Implanty zębowe",
        chunk_text="Wydatek na implanty zębowe może być kosztem uzyskania przychodów.",
    )
    other_source = SimpleNamespace(source_type="judgment")

    monkeypatch.setattr(legacy_interpretations.july7_mysql_rag, "is_mysql_rag_configured", lambda: False)

    def fake_sqlite_search(query: str, **kwargs):
        calls.append({"query": query, **kwargs})
        return [interpretation, other_source]

    monkeypatch.setattr(legacy_interpretations, "_search_historical_sqlite", fake_sqlite_search)
    monkeypatch.setattr(legacy_interpretations, "hydrate_tax_interpretation_documents", lambda chunks: chunks)

    result = legacy_interpretations.search_tax_interpretations("implanty zębowe", limit=3)

    assert result == [interpretation]
    assert calls == [{"query": "implanty zębowe", "limit": 20}]


def test_relevance_gate_rejects_topic_only_result_without_tax_cost() -> None:
    chunk = SimpleNamespace(
        subject="Usługi dentystyczne",
        chunk_text="Implanty są usługą medyczną objętą zwolnieniem z VAT.",
    )

    assert not legacy_interpretations._chunk_matches_query_facts(
        chunk,
        "Czy implanty zębowe mogą być kosztem uzyskania przychodu?",
    )


def test_relevance_gate_accepts_related_dental_rehabilitation_relief() -> None:
    chunk = SimpleNamespace(
        subject="Ulga rehabilitacyjna na protezy zębowe",
        chunk_text="Odliczenie wydatku na protezę w ramach ulgi rehabilitacyjnej.",
    )

    assert legacy_interpretations._chunk_matches_query_facts(
        chunk,
        "Czy implanty zębowe mogą być kosztem uzyskania przychodu?",
    )


def test_july7_interpretation_search_uses_vendored_mysql_without_tax_domain(monkeypatch) -> None:
    calls: list[dict] = []
    interpretation = SimpleNamespace(
        source_type="interpretation",
        document_id="interpretation-1",
        subject="Implanty zębowe",
        chunk_text="Implanty zębowe są kosztem uzyskania przychodów.",
    )
    monkeypatch.setattr(legacy_interpretations.july7_mysql_rag, "is_mysql_rag_configured", lambda: True)

    def fake_mysql_search(query: str, **kwargs):
        calls.append({"query": query, **kwargs})
        return [interpretation]

    monkeypatch.setattr(legacy_interpretations.july7_mysql_rag, "search_chunks_mysql", fake_mysql_search)
    monkeypatch.setattr(legacy_interpretations, "hydrate_tax_interpretation_documents", lambda chunks: chunks)

    assert legacy_interpretations.search_tax_interpretations("implanty zębowe", limit=3) == [interpretation]
    assert calls == [{
        "query": "implanty zębowe",
        "limit": 20,
        "source_types": {"interpretation"},
        "enforce_query_domain": False,
        "tax_domains": None,
    }]


def test_hydrate_tax_interpretation_documents_returns_full_ordered_document(monkeypatch) -> None:
    selected_chunk = RagChunk(
        chunk_id="interpretation-1:1",
        document_id="interpretation-1",
        chunk_index=1,
        score=99.0,
        chunk_text="urwany środek dokumentu",
        subject="Implanty zębowe",
        signature="0112-KDIL2-2.4011.8.2026.3.MM",
        published_date="2026-03-04",
        source_url="https://example.test/interpretation-1",
        category="Interpretacja indywidualna",
    )
    rows = [
        {
            "chunk_id": "interpretation-1:0",
            "document_id": "interpretation-1",
            "chunk_index": 0,
            "chunk_text": "Początek interpretacji.",
            "subject": "Implanty zębowe",
            "signature": "0112-KDIL2-2.4011.8.2026.3.MM",
            "published_date": "2026-03-04",
            "source_url": "https://example.test/interpretation-1",
            "category": "Interpretacja indywidualna",
            "source": "eureka",
            "source_type": "interpretation",
            "source_subtype": "individual",
            "authority": "KIS",
            "publication": "eureka",
            "legal_state_date": None,
            "source_pages_json": "[]",
            "legal_provisions_json": "[]",
        },
        {
            "chunk_id": "interpretation-1:1",
            "document_id": "interpretation-1",
            "chunk_index": 1,
            "chunk_text": "Koniec interpretacji.",
            "subject": "Implanty zębowe",
            "signature": "0112-KDIL2-2.4011.8.2026.3.MM",
            "published_date": "2026-03-04",
            "source_url": "https://example.test/interpretation-1",
            "category": "Interpretacja indywidualna",
            "source": "eureka",
            "source_type": "interpretation",
            "source_subtype": "individual",
            "authority": "KIS",
            "publication": "eureka",
            "legal_state_date": None,
            "source_pages_json": "[]",
            "legal_provisions_json": "[]",
        },
    ]
    monkeypatch.setattr(legacy_interpretations.july7_mysql_rag, "is_mysql_rag_configured", lambda: True)
    monkeypatch.setattr(
        legacy_interpretations.july7_mysql_rag,
        "fetch_rows_by_document_ids_mysql",
        lambda document_ids, **kwargs: rows,
    )

    hydrated = legacy_interpretations.hydrate_tax_interpretation_documents([selected_chunk])

    assert len(hydrated) == 1
    assert hydrated[0].chunk_index == 0
    assert hydrated[0].chunk_text == "Początek interpretacji.\n\nKoniec interpretacji."


def test_dental_cost_query_expands_to_implants_and_prostheses(monkeypatch) -> None:
    monkeypatch.setattr(legacy_interpretations, "_active_user_query", legacy_interpretations.ContextVar(
        "test_july7_user_query",
        default=None,
    ))

    queries = legacy_interpretations._build_bounded_historical_mysql_queries(
        "Czy implanty zębowe mogą być kosztem uzyskania przychodu?"
    )

    assert queries == [
        "+implant* +zęb* +koszt* +uzyskan* +przychod*",
        "+protez* +zęb* +koszt* +uzyskan* +przychod*",
    ]


def test_search_filters_against_full_document_not_the_winning_chunk(monkeypatch) -> None:
    seed = RagChunk(
        chunk_id="interpretation-1:9",
        document_id="interpretation-1",
        chunk_index=9,
        score=99.0,
        chunk_text="Uzasadnienie prawne bez opisu wydatku.",
        subject="Zaliczenie wydatku do kosztów",
        signature="TEST-1",
        published_date=None,
        source_url=None,
        category="Interpretacja indywidualna",
    )
    hydrated = RagChunk(
        chunk_id=seed.chunk_id,
        document_id=seed.document_id,
        chunk_index=0,
        score=seed.score,
        chunk_text="Implanty zębowe nie mogą stanowić kosztów uzyskania przychodów.",
        subject=seed.subject,
        signature=seed.signature,
        published_date=None,
        source_url=None,
        category=seed.category,
    )
    monkeypatch.setattr(legacy_interpretations.july7_mysql_rag, "is_mysql_rag_configured", lambda: True)
    monkeypatch.setattr(legacy_interpretations.july7_mysql_rag, "search_chunks_mysql", lambda *args, **kwargs: [seed])
    monkeypatch.setattr(legacy_interpretations, "hydrate_tax_interpretation_documents", lambda chunks: [hydrated])

    result = legacy_interpretations.search_tax_interpretations(
        "Czy implanty zębowe mogą być kosztem uzyskania przychodu?",
        limit=5,
    )

    assert result == [hydrated]
