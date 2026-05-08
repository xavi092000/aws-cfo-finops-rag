from __future__ import annotations

import re
from typing import Any, Dict

from cfo_finops_athena_rag_final import answer_analytical_question
from resolve_finops_analytics import load_selected_period_data, load_selected_period_daily_spend


def extract_amount(answer: str) -> str:
    match = re.search(r"\$[0-9,]+\.[0-9]{2}", answer)
    if match:
        return match.group(0)
    return "DATA_NOT_AVAILABLE"


def fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def compute_product_total(selection: Dict[str, Any]) -> str:
    division_filtered, product_filtered = load_selected_period_data(selection)
    _ = division_filtered
    total = float(product_filtered["actual_cost_usd"].sum())
    return fmt_money(total)


def compute_division_total(selection: Dict[str, Any]) -> str:
    division_filtered, product_filtered = load_selected_period_data(selection)
    _ = product_filtered
    total = float(division_filtered["actual_cost_usd"].sum())
    return fmt_money(total)


def get_pipeline_answer(selection: Dict[str, Any]) -> str:
    division_filtered, product_filtered = load_selected_period_data(selection)
    division_daily_spend, product_daily_spend = load_selected_period_daily_spend(selection)

    answer = answer_analytical_question(
        question="what is the total cost during this period",
        product_filtered=product_filtered,
        division_filtered=division_filtered,
        product_daily_spend=product_daily_spend,
        division_daily_spend=division_daily_spend,
    )
    return answer


def observe_case(label: str, selection: Dict[str, Any]) -> None:
    product_total = compute_product_total(selection)
    division_total = compute_division_total(selection)
    pipeline_answer = get_pipeline_answer(selection)
    pipeline_amount = extract_amount(pipeline_answer)

    print("\n==================================================")
    print(f"CASE: {label}")
    print(f"PRODUCT TOTAL  (raw): {product_total}")
    print(f"DIVISION TOTAL (raw): {division_total}")
    print(f"PIPELINE AMOUNT:      {pipeline_amount}")
    print("PIPELINE ANSWER:")
    print(pipeline_answer)
    print("==================================================")


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

    observe_case("DAYS", days_selection)
    observe_case("WEEKS", weeks_selection)
    observe_case("MONTHS", months_selection)

    print("\nTOTAL COST OBSERVATION COMPLETE.")


if __name__ == "__main__":
    main()