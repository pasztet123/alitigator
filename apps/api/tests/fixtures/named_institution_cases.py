"""Regression fixtures for the named-institution dictionary.

The broad cases are generated from the versioned data itself so adding a new
canonical institution automatically expands regression coverage.  Curated
cases below protect the Polish ambiguity rules that must never be solved by
prefix matching.
"""

from __future__ import annotations

from app.legal_institutions.dictionary import load_default_dictionary


def positive_cases() -> list[dict[str, object]]:
    dictionary = load_default_dictionary()
    cases: list[dict[str, object]] = []
    for definition in dictionary.institutions:
        phrase = definition.exact_aliases[0] if definition.exact_aliases else definition.canonical_name
        cases.append(
            {
                "name": definition.institution_id,
                "question": f"Czy w tej sprawie ma zastosowanie {phrase}?",
                "institution_id": definition.institution_id,
                "expects_lock": definition.status == "active",
            }
        )
    return cases


def negative_cases() -> list[dict[str, object]]:
    # These phrases deliberately resemble common tax vocabulary but omit the
    # discriminating named institution.  The suffix makes every fixture a
    # separate regression input without introducing another tax-law signal.
    ambiguous = (
        "Czy ryczałt będzie korzystny dla przedsiębiorcy",
        "Czy 50% kosztów można rozliczyć w zeznaniu rocznym",
        "Czy mieszkanie kupione od dewelopera ma znaczenie podatkowe",
        "Czy zakład produkcyjny może zwiększyć sprzedaż",
        "Czy globalny minimalny podatek dotyczy grupy",
        "Czy złe długi są problemem wierzyciela bez wskazania podatku",
        "Czy faktura zawiera poprawny numer",
        "Czy pojazd jest używany w firmie",
    )
    return [
        {
            "name": f"negative_{index}",
            "question": f"{ambiguous[index % len(ambiguous)]}; przypadek {index}.",
            "expects_lock": False,
        }
        for index in range(1, 161)
    ]


E2E_CASES = [
    {
        "name": "expansion_developer",
        "question": "Czy deweloper może skorzystać z ulgi na ekspansję przy wejściu na nowy rynek?",
        "institution_id": "expansion_relief",
        "expected_provisions": {"PIT art. 26gb", "CIT art. 18eb"},
    },
    {
        "name": "vat_bad_debts",
        "question": "Czy wierzyciel może zastosować ulgę na złe długi w VAT?",
        "institution_id": "bad_debt_relief_vat",
        "expected_provisions": {"VAT art. 89a", "VAT art. 89b"},
    },
    {
        "name": "ip_box",
        "question": "Czy programista spełniający warunki IP Box może stosować preferencję?",
        "institution_id": "ip_box",
        "expected_provisions": {"PIT art. 30ca", "CIT art. 24d"},
    },
    {
        "name": "split_payment",
        "question": "Czy transakcja podlega mechanizmowi podzielonej płatności MPP?",
        "institution_id": "split_payment",
        "expected_provisions": {"VAT art. 108a"},
    },
    {
        "name": "estonian_hidden_profits",
        "question": "Czy ukryte zyski w estońskim CIT obejmują świadczenie dla wspólnika?",
        "institution_id": "hidden_profits",
        "expected_provisions": {"CIT art. 28m"},
    },
    {
        "name": "wht_saas",
        "question": "Czy opłata SaaS dla nierezydenta podlega podatkowi u źródła WHT?",
        "institution_id": "withholding_tax",
        "expected_provisions": {"CIT art. 21", "CIT art. 22", "CIT art. 26"},
    },
    {
        "name": "ksef_input_vat",
        "question": "Czy VAT z faktury wystawionej poza KSeF nadal można odliczyć?",
        "institution_id": "input_vat_deduction",
        "expected_provisions": {"VAT art. 86"},
    },
]


COLLISION_CASES = [
    ("ryczalt_without_context", "Czy ryczałt jest prosty?", set()),
    ("creator_costs_not_vehicle_vat", "Czy 50% kosztów autorskich jest możliwe?", set()),
    ("housing_not_home_office", "Czy mieszkanie od dewelopera jest opodatkowane?", set()),
    ("bad_debts_without_tax", "Czy ulga na złe długi przysługuje wierzycielowi?", set()),
    ("global_minimum_not_cit", "Czy globalny minimalny podatek dotyczy grupy?", set()),
    ("ordinary_establishment_not_pe", "Czy zakład produkcyjny zatrudnia pracowników?", set()),
]
