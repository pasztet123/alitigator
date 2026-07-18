from types import SimpleNamespace

from app import legacy_interpretations


def test_july7_interpretation_search_is_type_scoped(monkeypatch) -> None:
    calls: list[dict] = []
    interpretation = SimpleNamespace(source_type="interpretation")
    other_source = SimpleNamespace(source_type="judgment")

    monkeypatch.setattr(
        legacy_interpretations.july7_rag,
        "resolve_statute_tax_domains",
        lambda query: {"PIT"},
    )

    def fake_search_chunks(query: str, **kwargs):
        calls.append({"query": query, **kwargs})
        return [interpretation, other_source]

    monkeypatch.setattr(legacy_interpretations.july7_rag, "search_chunks", fake_search_chunks)

    result = legacy_interpretations.search_tax_interpretations("leasing samochodu", limit=3)

    assert result == [interpretation]
    assert calls == [{
        "query": "leasing samochodu",
        "limit": 3,
        "source_types": {"interpretation"},
        "enforce_query_domain": True,
        "tax_domains": {"PIT"},
    }]
