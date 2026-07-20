from app.tax_research import (
    assess_candidate,
    build_anchors,
    candidate_boolean_queries,
    understand_tax_research_question,
)


def test_miescie_does_not_generate_miesci_precision_prefix() -> None:
    anchors = build_anchors("wynajem mieszkania w innym mieście podczas kontraktu")

    assert "miesci" not in anchors.high_precision
    assert "miesci" in anchors.forbidden_prefix
    assert "wynajem" in anchors.high_precision
    assert any(value.startswith("mieszka") for value in anchors.high_precision)
    assert "kontrakt" in anchors.high_precision


def test_linguistic_and_legal_boilerplate_stems_are_not_precision_anchors() -> None:
    question = (
        "Czy przedsiębiorca może zaliczyć koszty uzyskania przychodu wydatki "
        "w działalności gospodarczej, jeżeli stanowią podstawę podatkową?"
    )
    anchors = build_anchors(question)

    unsafe = {
        "przedsiebiorc", "koszt", "uzysk", "przychod", "wydatk",
        "dzialaln", "gospodarcz", "podat", "stanow", "miesci",
    }
    assert not unsafe.intersection(anchors.high_precision)


def test_safe_queries_never_use_miesci_prefix_by_itself() -> None:
    understanding = understand_tax_research_question(
        "Czy wynajem mieszkania w innym mieście na czas kontraktu może być kosztem?"
    )
    queries = candidate_boolean_queries(understanding)

    assert all("miesci*" not in query for query in queries)
    assert any("wynajem*" in query for query in queries)
    assert any("mieszka*" in query for query in queries)
    assert any("kontrakt*" in query for query in queries)


def test_normalized_score_prefers_business_rent_over_return_relief_title_neighbor() -> None:
    question = (
        "Czy przedsiębiorca może zaliczyć do kosztów uzyskania przychodu wydatki "
        "na wynajem mieszkania w innym mieście w związku z kontraktem?"
    )
    understanding = understand_tax_research_question(question)
    return_relief = assess_candidate(
        understanding,
        subject="Ulga na powrót - miejsce zamieszkania podatnika",
        text="Ulga na powrót i rezydencja podatkowa po powrocie do Polski.",
        provisions=["PIT art. 21 ust. 1 pkt 152"],
        tax_domain="PIT",
    )
    business_rent = assess_candidate(
        understanding,
        subject="Koszty najmu mieszkania podczas realizacji kontraktu",
        text="Wynajem mieszkania w innym mieście na czas kontraktu dla klienta jako koszt działalności.",
        provisions=["PIT art. 22 ust. 1"],
        tax_domain="PIT",
    )

    assert return_relief.relation == "different_mechanism"
    assert return_relief.reject is True
    assert business_rent.relation in {"direct", "strong_analogy"}
    assert business_rent.score > return_relief.score
    assert 0 <= business_rent.score <= 100


def test_implant_cost_and_rehabilitation_relief_are_separated() -> None:
    understanding = understand_tax_research_question(
        "Czy implanty zębowe mogą być kosztem uzyskania przychodu?"
    )
    kup = assess_candidate(
        understanding,
        subject="Zaliczenie do kosztów wydatków na zabiegi stomatologiczne",
        text="Implanty zębowe zostały poniesione w związku z działalnością i kosztami uzyskania przychodów.",
        provisions=["PIT art. 22 ust. 1"],
        tax_domain="PIT",
    )
    rehabilitation = assess_candidate(
        understanding,
        subject="Ulga rehabilitacyjna na wydatki stomatologiczne",
        text="Odliczenie w ramach ulgi rehabilitacyjnej.",
        provisions=["PIT art. 26 ust. 7a"],
        tax_domain="PIT",
    )

    assert kup.relation == "direct"
    assert rehabilitation.relation == "different_mechanism"
    assert rehabilitation.reject is True


def test_cash_payment_and_mixed_vehicle_understanding_add_special_provisions() -> None:
    cash = understand_tax_research_question(
        "Czy podzielenie zapłaty gotówką za transakcję na raty pozwala zachować koszt?"
    )
    vehicle = understand_tax_research_question(
        "Czy można odliczyć 50% VAT od paliwa do samochodu bez VAT-26?"
    )

    assert cash.legal_mechanism == "cash_payment_cost_exclusion"
    assert "PIT art. 22p" in cash.candidate_provisions
    assert vehicle.legal_mechanism == "mixed_use_vehicle_vat"
    assert "VAT art. 86a" in vehicle.candidate_provisions
