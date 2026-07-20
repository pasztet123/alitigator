from app import legacy_interpretations
from app.legacy_july7.rag import RagChunk


def make_chunk(*, document_id: str = "interpretation-1", score: float = 99.0, subject: str, text: str) -> RagChunk:
    return RagChunk(
        chunk_id=f"{document_id}:0",
        document_id=document_id,
        chunk_index=0,
        score=score,
        chunk_text=text,
        subject=subject,
        signature=document_id,
        published_date=None,
        source_url=None,
        category="Interpretacja indywidualna",
    )


def test_july7_interpretation_search_forces_snapshot_sqlite_without_tax_domain(monkeypatch) -> None:
    calls: list[dict] = []
    interpretation = make_chunk(
        subject="Implanty zębowe",
        text="Wydatek na implanty zębowe może być kosztem uzyskania przychodów.",
    )
    other_source = make_chunk(document_id="judgment-1", subject="Wyrok", text="Nie dotyczy.")
    other_source = RagChunk(**{**other_source.__dict__, "source_type": "judgment"})

    monkeypatch.setattr(legacy_interpretations.july7_mysql_rag, "is_mysql_rag_configured", lambda: False)

    def fake_sqlite_search(query: str, **kwargs):
        calls.append({"query": query, **kwargs})
        return [interpretation, other_source]

    monkeypatch.setattr(legacy_interpretations, "_search_historical_sqlite", fake_sqlite_search)
    monkeypatch.setattr(legacy_interpretations, "hydrate_tax_interpretation_documents", lambda chunks: chunks)

    result = legacy_interpretations.search_tax_interpretations("implanty zębowe", limit=3)

    assert result == [interpretation]
    assert calls == [{"query": "implanty zębowe", "limit": 12}]


def test_july7_interpretation_search_uses_vendored_mysql_without_tax_domain(monkeypatch) -> None:
    calls: list[dict] = []
    interpretation = make_chunk(
        subject="Implanty zębowe",
        text="Implanty zębowe są kosztem uzyskania przychodów.",
    )
    monkeypatch.setattr(legacy_interpretations.july7_mysql_rag, "is_mysql_rag_configured", lambda: True)

    def fake_mysql_search(query: str, **kwargs):
        calls.append({"query": query, **kwargs})
        return [interpretation]

    monkeypatch.setattr(legacy_interpretations, "_search_historical_mysql", fake_mysql_search)
    monkeypatch.setattr(legacy_interpretations, "hydrate_tax_interpretation_documents", lambda chunks: chunks)

    assert legacy_interpretations.search_tax_interpretations("implanty zębowe", limit=3) == [interpretation]
    assert calls == [{
        "query": "implanty zębowe",
        "limit": 12,
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


def test_generic_query_planner_keeps_late_distinctive_concept(monkeypatch) -> None:
    monkeypatch.setattr(legacy_interpretations, "_active_user_query", legacy_interpretations.ContextVar(
        "test_july7_user_query",
        default=None,
    ))

    queries = legacy_interpretations._build_bounded_historical_mysql_queries(
        "Czy wydatek udokumentowany fakturą wystawioną poza KSeF może być kosztem uzyskania przychodu?"
    )

    assert len(queries) == 1
    assert "ksef*" in queries[0]
    assert "faktur*" in queries[0]
    assert all("implant" not in query for query in queries)


def test_search_filters_against_full_document_not_the_winning_chunk(monkeypatch) -> None:
    seed = make_chunk(
        subject="Zaliczenie wydatku do kosztów",
        text="Uzasadnienie prawne bez opisu wydatku.",
    )
    hydrated = make_chunk(
        subject=seed.subject,
        text="Implanty zębowe nie mogą stanowić kosztów uzyskania przychodów.",
    )
    monkeypatch.setattr(legacy_interpretations.july7_mysql_rag, "is_mysql_rag_configured", lambda: True)
    monkeypatch.setattr(legacy_interpretations, "_search_historical_mysql", lambda *args, **kwargs: [seed])
    monkeypatch.setattr(legacy_interpretations, "hydrate_tax_interpretation_documents", lambda chunks: [hydrated])

    result = legacy_interpretations.search_tax_interpretations(
        "Czy implanty zębowe mogą być kosztem uzyskania przychodu?",
        limit=5,
    )

    assert result == [hydrated]


def test_coverage_ranking_prefers_document_covering_more_query_concepts() -> None:
    query = "Czy faktura wystawiona poza KSeF może być kosztem?"
    generic_invoice = make_chunk(
        document_id="generic",
        score=100.0,
        subject="Faktura dokumentująca zakup okularów",
        text="Faktura została wystawiona na przedsiębiorcę.",
    )
    ksef_invoice = make_chunk(
        document_id="ksef",
        score=1.0,
        subject="Koszty z faktur wystawionych poza KSeF",
        text="Faktura powinna zostać wystawiona w KSeF, ale została wystawiona poza KSeF.",
    )

    ranked = legacy_interpretations._dedupe_and_filter_relevant_chunks(
        [generic_invoice, ksef_invoice],
        query=query,
        limit=6,
    )

    assert ranked == [ksef_invoice, generic_invoice]
