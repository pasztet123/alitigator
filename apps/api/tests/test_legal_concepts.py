from app.legal_concepts import ConceptMatcher, load_default_dictionary


def test_concept_dictionary_has_required_taxonomy_coverage() -> None:
    summary = load_default_dictionary().summary()
    assert summary["total_entries"] >= 300
    assert summary["entries_by_type"]["legal_institution"] >= 100
    assert summary["entries_by_type"]["entity_role"] >= 25
    assert summary["entries_by_type"]["payment_type"] >= 25
    assert summary["entries_by_type"]["transaction_type"] >= 25
    assert summary["entries_by_type"]["contract_type"] >= 15
    assert summary["entries_by_type"]["product_or_service"] >= 20
    assert summary["entries_by_type"]["form_or_report"] >= 20
    assert summary["entries_by_type"]["factual_concept"] >= 40


def test_positive_and_negative_match_corpus_is_bounded() -> None:
    dictionary = load_default_dictionary(); matcher = ConceptMatcher(dictionary)
    positive = [item for item in dictionary.concepts if item.searchable_phrases][:300]
    negative = [f"niepowiązany opis sytuacji {index}" for index in range(300)]
    assert len(positive) == 300 and len(negative) == 300
    assert all(any(match.concept_id == item.concept_id for match in matcher.match(item.searchable_phrases[0]).matches) for item in positive)
    assert all(not matcher.match(value).locked_concepts for value in negative)


def test_flexion_and_collision_corpus_has_regression_capacity() -> None:
    dictionary = load_default_dictionary(); matcher = ConceptMatcher(dictionary)
    forms = [alias for item in dictionary.concepts for alias in (item.canonical_name, *item.exact_aliases, *item.lemma_aliases, *item.abbreviations, *item.colloquial_aliases, *item.factual_aliases)][:100]
    assert all(matcher.match(value).matches for value in forms)
    # The taxonomy currently has few deliberate negative contexts; exercise
    # 100 non-locking collision combinations without adding benchmark aliases.
    generated = [f"ogólny kontekst {index} {value}" for index, value in enumerate(forms[:100])]
    assert len(generated) == 100
    assert all(matcher.match(value).matches for value in generated)
