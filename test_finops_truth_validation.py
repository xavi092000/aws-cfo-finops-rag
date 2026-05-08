from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from cfo_finops_athena_rag_final import run_finops_cfo_pipeline
from resolve_finops_analytics import (
    load_selected_period_data,
    load_selected_period_daily_spend,
)


def extract_top_line(answer: str) -> str:
    """
    Extracts the first bullet line from the pipeline answer.
    Example:
    '- Bedrock: $181.94'
    """
    for line in answer.splitlines():
        line = line.strip()
        if line.startswith("- "):
            return line
    return "DATA_NOT_AVAILABLE"


def fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def compute_expected_service(selection: Dict[str, Any]) -> str:
    division_daily_spend, product_daily_spend = load_selected_period_daily_spend(selection)
    _ = product_daily_spend  # intentionally unused

    df = division_daily_spend.copy()

    grouped = (
        df.groupby("service", as_index=False)["actual_cost_usd"]
        .sum()
        .sort_values("actual_cost_usd", ascending=False)
        .reset_index(drop=True)
    )

    top = grouped.iloc[0]
    return f"- {top['service']}: {fmt_money(float(top['actual_cost_usd']))}"


def compute_expected_product(selection: Dict[str, Any]) -> str:
    division_filtered, product_filtered = load_selected_period_data(selection)
    _ = division_filtered  # intentionally unused

    df = product_filtered.copy()

    grouped = (
        df.groupby("product", as_index=False)["actual_cost_usd"]
        .sum()
        .sort_values("actual_cost_usd", ascending=False)
        .reset_index(drop=True)
    )

    top = grouped.iloc[0]
    return f"- {top['product']}: {fmt_money(float(top['actual_cost_usd']))}"


def compute_expected_division(selection: Dict[str, Any]) -> str:
    division_filtered, product_filtered = load_selected_period_data(selection)
    _ = product_filtered  # intentionally unused

    df = division_filtered.copy()

    grouped = (
        df.groupby("division", as_index=False)["actual_cost_usd"]
        .sum()
        .sort_values("actual_cost_usd", ascending=False)
        .reset_index(drop=True)
    )

    top = grouped.iloc[0]
    return f"- {top['division']}: {fmt_money(float(top['actual_cost_usd']))}"


def pipeline_top_line(selection: Dict[str, Any], question: str) -> str:
    result = run_finops_cfo_pipeline(
        selection=selection,
        question=question,
        enable_audio=False,
        auto_open_audio=False,
    )
    return extract_top_line(result["answer"])


def validate_case(
    label: str,
    selection: Dict[str, Any],
    question: str,
    expected_fn,
) -> None:
    expected = expected_fn(selection)
    actual = pipeline_top_line(selection, question)

    print("\n==================================================")
    print(f"CASE: {label}")
    print(f"QUESTION: {question}")
    print(f"EXPECTED: {expected}")
    print(f"ACTUAL:   {actual}")
    print("==================================================")

    assert actual == expected, (
        f"\nValidation failed for {label}\n"
        f"Question: {question}\n"
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

    cases = [
        (
            "DAYS - SERVICE",
            days_selection,
            "what was the service with the most cost during this period",
            compute_expected_service,
        ),
        (
            "DAYS - PRODUCT",
            days_selection,
            "what was the product with the most cost during this period",
            compute_expected_product,
        ),
        (
            "DAYS - DIVISION",
            days_selection,
            "what was the division with the most cost during this period",
            compute_expected_division,
        ),
        (
            "WEEKS - SERVICE",
            weeks_selection,
            "what was the service with the most cost during this period",
            compute_expected_service,
        ),
        (
            "WEEKS - PRODUCT",
            weeks_selection,
            "what was the product with the most cost during this period",
            compute_expected_product,
        ),
        (
            "WEEKS - DIVISION",
            weeks_selection,
            "what was the division with the most cost during this period",
            compute_expected_division,
        ),
        (
            "MONTHS - SERVICE",
            months_selection,
            "what was the service with the most cost during this period",
            compute_expected_service,
        ),
        (
            "MONTHS - PRODUCT",
            months_selection,
            "what was the product with the most cost during this period",
            compute_expected_product,
        ),
        (
            "MONTHS - DIVISION",
            months_selection,
            "what was the division with the most cost during this period",
            compute_expected_division,
        ),
    ]

    for label, selection, question, expected_fn in cases:
        validate_case(label, selection, question, expected_fn)

    print("\nALL TRUTH VALIDATION TESTS PASSED.")


if __name__ == "__main__":
    main()