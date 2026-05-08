from __future__ import annotations

import re
from typing import Any, Dict

from cfo_finops_athena_rag_final import answer_analytical_question
from resolve_finops_analytics import load_selected_period_data, load_selected_period_daily_spend


def extract_amount(answer: str) -> str:
    """
    Extract the first currency amount like $4,130.80 from the answer.
    """
    match = re.search(r"\$[0-9,]+\.[0-9]{2}", answer)
    if match:
        return match.group(0)
    return "DATA_NOT_AVAILABLE"


def fmt_money(value: float) -> str:
    return f"${value:,.2f}"


# =========================================================
# EXPECTED VALUES (RAW CALCULATION)
# =========================================================
def compute_total_cost(selection: Dict[str, Any]) -> str:
    division_filtered, product_filtered = load_selected_period_data(selection)
    _ = division_filtered  # intentionally unused

    # IMPORTANT:
    # Use ONE source of truth to avoid double counting.
    total = float(product_filtered["actual_cost_usd"].sum())

    return fmt_money(total)


# =========================================================
# DIRECT ANALYTICAL ANSWER (NO LLM ROUTER)
# =========================================================
def pipeline_total_cost(selection: Dict[str, Any]) -> str:
    division_filtered, product_filtered = load_selected_period_data(selection)
    division_daily_spend, product_daily_spend = load_selected_period_daily_spend(selection)

    answer = answer_analytical_question(
        question="what is the total cost during this period",
        product_filtered=product_filtered,
        division_filtered=division_filtered,
        product_daily_spend=product_daily_spend,
        division_daily_spend=division_daily_spend,
    )

    print("\nPIPELINE RAW ANSWER:")
    print(answer)

    return extract_amount(answer)


# =========================================================
# VALIDATION
# =========================================================
def validate_case(label: str, selection: Dict[str, Any]) -> None:
    expected = compute_total_cost(selection)
    actual = pipeline_total_cost(selection)

    print("\n==================================================")
    print(f"CASE: {label}")
    print(f"EXPECTED TOTAL: {expected}")
    print(f"PIPELINE TOTAL: {actual}")
    print("==================================================")

    assert actual == expected, (
        f"\nValidation failed for {label}\n"
        f"Expected: {expected}\n"
        f"Actual:   {actual}\n"
    )


def main() -> None:
    days_selection = {
        "mode": "days",
        "block": "A",
        "week": 3,
        "days": 4,
        "period": "Monday to Thursday",
    }

    weeks_selection = {
        "mode": "weeks",
        "block": "B",
        "weeks": [5, 6],
        "number_of_weeks": 2,
    }

    months_selection = {
        "mode": "months",
        "monthly_block": "BC",
    }

    validate_case("DAYS", days_selection)
    validate_case("WEEKS", weeks_selection)
    validate_case("MONTHS", months_selection)

    print("\nALL TOTAL COST TESTS PASSED.")


if __name__ == "__main__":
    main()