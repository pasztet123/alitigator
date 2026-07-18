from types import SimpleNamespace

from app import legacy_interpretations


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

    assert legacy_interpretations.search_tax_interpretations("implanty zębowe", limit=3) == [interpretation]
    assert calls == [{
        "query": "implanty zębowe",
        "limit": 20,
        "source_types": {"interpretation"},
        "enforce_query_domain": False,
        "tax_domains": None,
    }]
