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

    assert [item.document_id for item in result] == [interpretation.document_id]
    assert 0 <= result[0].score <= 100
    assert calls == [{"query": "implanty zębowe", "limit": 120}]


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

    result = legacy_interpretations.search_tax_interpretations("implanty zębowe", limit=3)
    assert [item.document_id for item in result] == [interpretation.document_id]
    assert calls == [{
        "query": "implanty zębowe",
        "limit": 120,
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

    assert any("ksef*" in query for query in queries)
    assert any("faktur*" in query for query in queries)
    assert any(query.startswith("+") for query in queries)
    assert all("implant" not in query for query in queries)


def test_generic_query_planner_keeps_listed_object_with_the_leading_action() -> None:
    query = (
        "Czy przedsiębiorca może zaliczyć do kosztów uzyskania przychodu wydatki na "
        "zakup, szkolenie, karmę i leczenie psa wykorzystywanego do ochrony siedziby firmy?"
    )

    pairs = legacy_interpretations._build_generic_probe_pairs(query)
    queries = legacy_interpretations._build_bounded_historical_mysql_queries(query)

    assert ("zakup", "psa") in pairs
    assert ("lecze", "psa") in pairs
    assert "+zakup* +psa*" in queries


def test_metadata_boost_rewards_matching_keywords_and_explicit_provision() -> None:
    row = {
        "keywords_json": '["wynajem mieszkania", "koszty eksploatacyjne"]',
        "legal_provisions_json": '["[PIT] art. 22-ust. 1"]',
    }
    query = "Czy wynajem mieszkania może być kosztem na podstawie art. 22 ust. 1 ustawy PIT?"

    keyword_pairs, keyword_coverage, provision_matches = legacy_interpretations._metadata_match_score(
        row,
        query=query,
    )

    assert keyword_pairs >= 1
    assert keyword_coverage >= 2
    assert provision_matches == 2


def test_metadata_provisions_do_not_boost_question_without_explicit_article() -> None:
    row = {
        "keywords_json": "[]",
        "legal_provisions_json": '["[PIT] art. 22-ust. 1"]',
    }

    assert legacy_interpretations._metadata_match_score(
        row,
        query="Czy wydatek może być kosztem uzyskania przychodu?",
    )[2] == 0


def test_subject_cooccurrence_outweighs_unrelated_full_text_overlap() -> None:
    query = "Czy wynajem mieszkania może być kosztem uzyskania przychodu?"
    exact_subject = make_chunk(
        document_id="exact-subject",
        score=1.0,
        subject="Koszty uzyskania przychodów - wynajem mieszkania",
        text="Treść interpretacji.",
    )
    broad_full_text = make_chunk(
        document_id="broad-full-text",
        score=99.0,
        subject="Ulga na powrót",
        text="W uzasadnieniu mimochodem opisano wynajem oraz mieszkanie i przychód.",
    )

    ranked = legacy_interpretations._dedupe_and_filter_relevant_chunks(
        [broad_full_text, exact_subject],
        query=query,
        limit=6,
    )

    assert [item.document_id for item in ranked] == ["exact-subject"]


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

    assert [item.document_id for item in result] == [hydrated.document_id]
    assert result[0].score > 0


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

    assert [item.document_id for item in ranked] == ["ksef"]


def test_final_document_order_keeps_stronger_candidate_rank_after_hydration() -> None:
    query = "Czy zakup psa może być kosztem uzyskania przychodu?"
    strong_candidate = make_chunk(
        document_id="strong",
        score=100_000.0,
        subject="Zakup psa jako koszt uzyskania przychodu",
        text="Pełna interpretacja zawiera dodatkowe, nieistotne wątki.",
    )
    broad_neighbour = make_chunk(
        document_id="broad",
        score=1.0,
        subject="Koszty szkoleń pracowników",
        text="Zakup, koszt oraz przychód występują wielokrotnie w treści.",
    )

    ranked = legacy_interpretations._dedupe_and_filter_relevant_chunks(
        [broad_neighbour, strong_candidate],
        query=query,
        limit=6,
    )

    assert [item.document_id for item in ranked] == ["strong"]
