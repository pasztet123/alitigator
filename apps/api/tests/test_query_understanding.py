from app.query_generation import build_query_families
from app.query_understanding.deterministic_extractor import build_question_card
from app.query_understanding.merger import merge_query_plan
from app.query_understanding.models import ModelQueryExpansion


WHT_SAAS_EULA = (
    "Czy polska spółka musi pobrać podatek u źródła od opłaty za dostęp "
    "do zagranicznego SaaS, jeżeli umowa to EULA?"
)


def test_wht_saas_eula_query_plan_is_data_driven() -> None:
    card, matches = build_question_card(WHT_SAAS_EULA)
    plan = merge_query_plan(card, matches, ModelQueryExpansion())
    families = build_query_families(plan)

    assert "withholding_tax" in card.locked_institutions
    assert "saas" in card.products_or_services
    assert "eula" in card.contract_types
    assert card.payment_direction == "poland_to_foreign_recipient"
    assert plan.has_any_provision(["art. 21", "art. 26"])
    assert {"locked_institution", "verified_provision", "product_or_service", "contract_type"}.issubset({item.type for item in families})
    assert len(families) >= 3


def test_model_cannot_replace_locked_primary_issue() -> None:
    card, matches = build_question_card(WHT_SAAS_EULA)
    plan = merge_query_plan(card, matches, ModelQueryExpansion(primary_issue="business_expense"))
    assert plan.primary_issue == "withholding_tax"
    assert plan.conflicts[0]["resolution"] == "keep_deterministic"


def test_unverified_model_provision_is_not_hard_filter() -> None:
    card, matches = build_question_card(WHT_SAAS_EULA)
    plan = merge_query_plan(card, matches, ModelQueryExpansion())
    for family in build_query_families(plan):
        assert all("999" not in term for term in family.hard_requirements)
