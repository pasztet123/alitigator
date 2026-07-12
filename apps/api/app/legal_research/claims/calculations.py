from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from app.legal_research.models import CalculationRecord


def calculate_deadline(
    *, calculation_id: str, event_date: date, statutory_years: int,
    provision_ids: list[str], fact_ids: list[str],
) -> CalculationRecord:
    """Calculate a period counted from the end of the event year."""
    if statutory_years < 0:
        raise ValueError("statutory_years cannot be negative")
    year_end = date(event_date.year, 12, 31)
    deadline = date(event_date.year + statutory_years, 12, 31)
    return CalculationRecord(
        calculation_id=calculation_id,
        calculation_type="statutory_deadline_from_year_end",
        input_values={
            "event_date": event_date.isoformat(),
            "sale_year_end": year_end.isoformat(),
            "statutory_period_years": statutory_years,
            "deadline": deadline.isoformat(),
        },
        formula="deadline = end_of_year(event_date) + statutory_period_years",
        result=deadline.isoformat(), units="date", rounding_rule=None,
        provision_ids=provision_ids, fact_ids=fact_ids, validation_status="valid",
    )


def calculate_proportion(
    *, calculation_id: str, income: Decimal, qualifying_expense: Decimal,
    revenue: Decimal, provision_ids: list[str], fact_ids: list[str],
) -> CalculationRecord:
    if revenue <= 0:
        raise ValueError("revenue must be positive")
    result = (income * qualifying_expense / revenue).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return CalculationRecord(
        calculation_id=calculation_id, calculation_type="proportional_exemption",
        input_values={"income": float(income), "qualifying_expense": float(qualifying_expense), "revenue": float(revenue)},
        formula="income × qualifying_expense / revenue", result=float(result), units="PLN",
        rounding_rule="ROUND_HALF_UP to 0.01 PLN", provision_ids=provision_ids,
        fact_ids=fact_ids, validation_status="valid",
    )


def condition_applied_to_deadline(*, event_date: date, deadline: date) -> str:
    return "before" if event_date < deadline else "on_deadline" if event_date == deadline else "after"


__all__ = ["calculate_deadline", "calculate_proportion", "condition_applied_to_deadline"]
