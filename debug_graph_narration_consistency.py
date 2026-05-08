from __future__ import annotations

import json
from typing import Any

import pandas as pd

from load_athena_views import load_current_powerbi_views
from graph_narration import generate_graph_narration


def safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def format_money(value: float) -> str:
    return f"{safe_float(value):,.2f}"


def build_product_summary(product_df: pd.DataFrame) -> pd.DataFrame:
    required = {"product", "actual_cost_usd", "allocated_budget_usd", "variance_usd"}
    if product_df.empty or not required.issubset(product_df.columns):
        return pd.DataFrame(columns=["product", "actual_cost_usd", "allocated_budget_usd", "variance_usd"])

    grouped = (
        product_df.groupby("product", dropna=False)
        .agg(
            actual_cost_usd=("actual_cost_usd", "sum"),
            allocated_budget_usd=("allocated_budget_usd", "sum"),
            variance_usd=("variance_usd", "sum"),
        )
        .reset_index()
        .sort_values("variance_usd", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return grouped


def build_service_summary_from_division(division_df: pd.DataFrame) -> pd.DataFrame:
    required = {"service", "actual_cost_usd", "allocated_budget_usd", "variance_usd"}
    if division_df.empty or not required.issubset(division_df.columns):
        return pd.DataFrame(columns=["service", "actual_cost_usd", "allocated_budget_usd", "variance_usd"])

    grouped = (
        division_df.groupby("service", dropna=False)
        .agg(
            actual_cost_usd=("actual_cost_usd", "sum"),
            allocated_budget_usd=("allocated_budget_usd", "sum"),
            variance_usd=("variance_usd", "sum"),
        )
        .reset_index()
        .sort_values("actual_cost_usd", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return grouped


def compute_totals(product_df: pd.DataFrame) -> dict[str, float]:
    actual = safe_float(product_df["actual_cost_usd"].sum()) if "actual_cost_usd" in product_df.columns else 0.0
    budget = safe_float(product_df["allocated_budget_usd"].sum()) if "allocated_budget_usd" in product_df.columns else 0.0
    variance = safe_float(product_df["variance_usd"].sum()) if "variance_usd" in product_df.columns else 0.0
    variance_pct = (variance / budget * 100.0) if budget > 0 else 0.0

    return {
        "total_actual_cost_usd": actual,
        "total_budget_usd": budget,
        "total_variance_usd": variance,
        "total_variance_percent": variance_pct,
    }


def get_top_row(summary_df: pd.DataFrame, label_col: str, value_col: str, rank: int) -> dict[str, Any]:
    if summary_df.empty or label_col not in summary_df.columns or value_col not in summary_df.columns:
        return {"label": "DATA_NOT_AVAILABLE", "value": 0.0}

    idx = min(rank, len(summary_df) - 1)
    row = summary_df.iloc[idx]

    label = row[label_col]
    if pd.isna(label):
        label = "DATA_NOT_AVAILABLE"

    return {
        "label": str(label),
        "value": safe_float(row[value_col]),
    }


def extract_narration_numbers(
    product_summary: pd.DataFrame,
    service_summary: pd.DataFrame,
    totals: dict[str, float],
) -> dict[str, Any]:
    top_product = get_top_row(product_summary, "product", "variance_usd", 0)
    second_product = get_top_row(product_summary, "product", "variance_usd", 1)

    top_service = get_top_row(service_summary, "service", "actual_cost_usd", 0)
    second_service = get_top_row(service_summary, "service", "actual_cost_usd", 1)

    return {
        "top_product_by_variance": top_product,
        "second_product_by_variance": second_product,
        "top_service_by_actual_cost": top_service,
        "second_service_by_actual_cost": second_service,
        "totals": totals,
    }


def print_dataframe_preview(name: str, df: pd.DataFrame, max_rows: int = 10) -> None:
    print(f"\n===== {name} =====")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    if df.empty:
        print("DataFrame is empty.")
        return
    print(df.head(max_rows).to_string(index=False))


def compare_with_expected(
    actuals: dict[str, Any],
    expected: dict[str, Any],
    tolerance: float = 0.01,
) -> list[str]:
    messages: list[str] = []

    comparisons = [
        ("top_product_by_variance", "value"),
        ("second_product_by_variance", "value"),
        ("top_service_by_actual_cost", "value"),
        ("second_service_by_actual_cost", "value"),
    ]

    total_comparisons = [
        "total_actual_cost_usd",
        "total_budget_usd",
        "total_variance_usd",
        "total_variance_percent",
    ]

    for section, field in comparisons:
        actual_value = safe_float(actuals[section][field])
        expected_value = safe_float(expected[section][field])
        delta = abs(actual_value - expected_value)

        if delta <= tolerance:
            messages.append(f"PASS - {section}.{field}: actual={actual_value:.2f}, expected={expected_value:.2f}")
        else:
            messages.append(f"FAIL - {section}.{field}: actual={actual_value:.2f}, expected={expected_value:.2f}, delta={delta:.2f}")

    for total_key in total_comparisons:
        actual_value = safe_float(actuals["totals"][total_key])
        expected_value = safe_float(expected["totals"][total_key])
        delta = abs(actual_value - expected_value)

        if delta <= tolerance:
            messages.append(f"PASS - totals.{total_key}: actual={actual_value:.2f}, expected={expected_value:.2f}")
        else:
            messages.append(f"FAIL - totals.{total_key}: actual={actual_value:.2f}, expected={expected_value:.2f}, delta={delta:.2f}")

    if actuals["top_product_by_variance"]["label"] == expected["top_product_by_variance"]["label"]:
        messages.append(
            f"PASS - top_product_by_variance.label: {actuals['top_product_by_variance']['label']}"
        )
    else:
        messages.append(
            f"FAIL - top_product_by_variance.label: actual={actuals['top_product_by_variance']['label']}, expected={expected['top_product_by_variance']['label']}"
        )

    if actuals["second_product_by_variance"]["label"] == expected["second_product_by_variance"]["label"]:
        messages.append(
            f"PASS - second_product_by_variance.label: {actuals['second_product_by_variance']['label']}"
        )
    else:
        messages.append(
            f"FAIL - second_product_by_variance.label: actual={actuals['second_product_by_variance']['label']}, expected={expected['second_product_by_variance']['label']}"
        )

    if actuals["top_service_by_actual_cost"]["label"] == expected["top_service_by_actual_cost"]["label"]:
        messages.append(
            f"PASS - top_service_by_actual_cost.label: {actuals['top_service_by_actual_cost']['label']}"
        )
    else:
        messages.append(
            f"FAIL - top_service_by_actual_cost.label: actual={actuals['top_service_by_actual_cost']['label']}, expected={expected['top_service_by_actual_cost']['label']}"
        )

    if actuals["second_service_by_actual_cost"]["label"] == expected["second_service_by_actual_cost"]["label"]:
        messages.append(
            f"PASS - second_service_by_actual_cost.label: {actuals['second_service_by_actual_cost']['label']}"
        )
    else:
        messages.append(
            f"FAIL - second_service_by_actual_cost.label: actual={actuals['second_service_by_actual_cost']['label']}, expected={expected['second_service_by_actual_cost']['label']}"
        )

    return messages


def main() -> None:
    selection = {"mode": "months", "monthly_block": "BC"}

    expected = {
        "top_product_by_variance": {
            "label": "Innovation Lab Projects",
            "value": 1441.99,
        },
        "second_product_by_variance": {
            "label": "AI Assistant",
            "value": 1140.86,
        },
        "top_service_by_actual_cost": {
            "label": "Bedrock",
            "value": 2324.80,
        },
        "second_service_by_actual_cost": {
            "label": "CloudWatch",
            "value": 837.58,
        },
        "totals": {
            "total_actual_cost_usd": 64701.69,
            "total_budget_usd": 59845.37,
            "total_variance_usd": 4856.32,
            "total_variance_percent": 8.11,
        },
    }

    division_df, product_df, service_df = load_current_powerbi_views()

    print("\n===== RAW DATA CHECK =====")
    print(f"Selection: {selection}")
    print(f"division_df rows: {len(division_df)}")
    print(f"product_df rows: {len(product_df)}")
    print(f"service_df rows: {len(service_df)}")

    print_dataframe_preview("DIVISION DF", division_df)
    print_dataframe_preview("PRODUCT DF", product_df)
    print_dataframe_preview("SERVICE DF", service_df)

    product_summary = build_product_summary(product_df)
    service_summary = build_service_summary_from_division(division_df)
    totals = compute_totals(product_df)

    print_dataframe_preview("PRODUCT SUMMARY", product_summary, max_rows=20)
    print_dataframe_preview("SERVICE SUMMARY FROM DIVISION", service_summary, max_rows=20)

    actuals = extract_narration_numbers(product_summary, service_summary, totals)

    print("\n===== NARRATION INPUT VALUES =====")
    print(json.dumps(actuals, indent=2))

    print("\n===== EXPECTED VALUES =====")
    print(json.dumps(expected, indent=2))

    print("\n===== CONSISTENCY CHECK =====")
    messages = compare_with_expected(actuals, expected, tolerance=0.05)
    for message in messages:
        print(message)

    narration = generate_graph_narration(product_df, division_df, service_df, selection)
    print("\n===== GENERATED NARRATION =====")
    print(narration)


if __name__ == "__main__":
    main()