from pathlib import Path
import pandas as pd

from resolve_finops_analysis_scope import get_analysis_data


# =========================
# HELPERS
# =========================
def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
    return df


def find_first_existing_column(df: pd.DataFrame, possible_names: list[str]):
    normalized = {col.lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


def clean_currency_series(series: pd.Series) -> pd.Series:
    """
    Convert strings like '$1,234.56' or '$1,23' safely to numeric.
    Keeps digits, comma, dot and minus, then normalizes.
    """
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9,.\-]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0)


def detect_amount_column(df: pd.DataFrame):
    possible_amount_columns = [
        "actual cost ($)",
        "allocated budget ($)",
        "variance ($)",
        "actual cost",
        "allocated budget",
        "variance",
        "cost",
        "spend",
        "actual_cost",
        "actual_spend",
        "daily_cost",
        "weekly_cost",
        "monthly_cost",
        "budget",
        "budget_amount",
        "daily_budget",
        "weekly_budget",
        "monthly_budget",
        "allocated_budget",
        "planned_budget",
        "amount",
        "value"
    ]

    normalized = {col.lower(): col for col in df.columns}

    for name in possible_amount_columns:
        if name.lower() in normalized:
            return normalized[name.lower()]

    return None


def detect_dimension_columns(df: pd.DataFrame):
    candidates = {
        "product": ["product", "product_name", "service", "service_name"],
        "division": ["division", "business_unit", "department", "team"],
        "project": ["project", "project_name"],
        "environment": ["environment", "env"],
        "region": ["region"],
        "category": ["category", "cost_category"]
    }

    found = {}
    for label, names in candidates.items():
        col = find_first_existing_column(df, names)
        if col:
            found[label] = col

    return found


def summarize_filtered_table(df: pd.DataFrame, table_name: str) -> None:
    print("\n" + "=" * 70)
    print(f"{table_name}")
    print("=" * 70)

    print(f"Rows matched: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if df.empty:
        print("No rows matched this scope.")
        return

    amount_col = detect_amount_column(df)
    dimension_cols = detect_dimension_columns(df)

    if amount_col:
        numeric_series = clean_currency_series(df[amount_col])
        print(f"Detected amount column: {amount_col}")
        print(f"Total amount: {numeric_series.sum():,.2f}")
        print(f"Average amount: {numeric_series.mean():,.2f}")
    else:
        print("⚠️ No amount column detected automatically.")

    for label, col in dimension_cols.items():
        unique_values = df[col].dropna().astype(str).unique().tolist()
        preview = unique_values[:10]
        print(f"{label.title()} column detected: {col}")
        print(f"Sample values ({min(len(preview), 10)} shown): {preview}")

    print("\nFirst 10 rows:")
    print(df.head(10))


def aggregate_by_dimensions(
    df: pd.DataFrame,
    table_name: str,
    preferred_group_order: list[str] = None
) -> None:
    if df.empty:
        print("\nNo aggregation possible because the filtered table is empty.")
        return

    amount_col = detect_amount_column(df)
    if not amount_col:
        print("\n⚠️ Aggregation skipped because no amount column was detected.")
        return

    dimension_cols = detect_dimension_columns(df)
    if not dimension_cols:
        print("\n⚠️ Aggregation skipped because no business dimension column was detected.")
        return

    if preferred_group_order is None:
        preferred_group_order = ["division", "product", "project", "environment", "region", "category"]

    chosen_group_cols = []
    for dim in preferred_group_order:
        if dim in dimension_cols:
            chosen_group_cols.append(dimension_cols[dim])

    if not chosen_group_cols:
        print("\n⚠️ Aggregation skipped because no usable grouping column was found.")
        return

    temp = df.copy()
    temp[amount_col] = clean_currency_series(temp[amount_col])

    grouped = (
        temp.groupby(chosen_group_cols, dropna=False)[amount_col]
        .sum()
        .reset_index()
        .sort_values(amount_col, ascending=False)
    )

    print("\n" + "-" * 70)
    print(f"AGGREGATED VIEW - {table_name}")
    print("-" * 70)
    print(grouped.head(20))


# =========================
# MAIN
# =========================
def main():
    # Example selection
    # Replace this later with the output from your agent script
    selection = {
        "mode": "days",
        "block": "A",
        "week": 1,
        "days": 2,
        "period": "Monday to Tuesday"
    }

    result = get_analysis_data(selection)

    scope = result["scope"]
    products_usage_filtered = normalize_string_columns(result["products_usage_filtered"])
    internal_usage_filtered = normalize_string_columns(result["internal_usage_filtered"])
    products_budget_filtered = normalize_string_columns(result["products_budget_filtered"])
    internal_budget_filtered = normalize_string_columns(result["internal_budget_filtered"])

    print("\n" + "=" * 70)
    print("RESOLVED ANALYSIS SCOPE")
    print("=" * 70)
    print(scope)

    summarize_filtered_table(products_usage_filtered, "PRODUCTS USAGE FILTERED")
    aggregate_by_dimensions(products_usage_filtered, "PRODUCTS USAGE FILTERED")

    summarize_filtered_table(internal_usage_filtered, "INTERNAL USAGE FILTERED")
    aggregate_by_dimensions(internal_usage_filtered, "INTERNAL USAGE FILTERED")

    summarize_filtered_table(products_budget_filtered, "PRODUCTS BUDGET FILTERED")
    aggregate_by_dimensions(products_budget_filtered, "PRODUCTS BUDGET FILTERED")

    summarize_filtered_table(internal_budget_filtered, "INTERNAL BUDGET FILTERED")
    aggregate_by_dimensions(internal_budget_filtered, "INTERNAL BUDGET FILTERED")


if __name__ == "__main__":
    main()