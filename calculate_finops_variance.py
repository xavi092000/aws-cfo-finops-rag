import pandas as pd
from pathlib import Path


# =========================
# PATHS
# =========================
DATA_DIR = Path("data")

PRODUCTS_USAGE = DATA_DIR / "products_usage_84_days.csv"
INTERNAL_USAGE = DATA_DIR / "internal_usage_84_days.csv"

AWS_PRICING = DATA_DIR / "aws_pricing.csv"

PRODUCTS_BUDGET = DATA_DIR / "Products_Daily_Budget.csv"
INTERNAL_BUDGET = DATA_DIR / "Internal_Daily_Budget.csv"

OUTPUT_PRODUCTS = "products_finops_analysis.csv"
OUTPUT_INTERNAL = "internal_finops_analysis.csv"


# =========================
# HELPERS
# =========================
def load_csv(path):
    try:
        df = pd.read_csv(path, sep=";")
        if len(df.columns) == 1:
            df = pd.read_csv(path)
        return df
    except Exception:
        return pd.read_csv(path)


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def normalize_string_columns(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
    return df


def parse_mixed_number(value):
    """
    Gère correctement :
    - $1,82      -> 1.82
    - 11,5%      -> 11.5
    - 10 000     -> 10000
    - 1,234.56   -> 1234.56
    - 1234.56    -> 1234.56
    - 1234       -> 1234.0
    """
    if pd.isna(value):
        return 0.0

    s = str(value).strip()

    if s == "":
        return 0.0

    s = s.replace("\u00A0", "")
    s = s.replace(" ", "")
    s = s.replace("$", "")
    s = s.replace("%", "")

    # Cas US: 1,234.56 -> enlever les virgules milliers
    if "," in s and "." in s:
        s = s.replace(",", "")
        try:
            return float(s)
        except ValueError:
            return 0.0

    # Cas FR: 1,82 -> 1.82
    if "," in s and "." not in s:
        s = s.replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return 0.0

    # Cas simple
    try:
        return float(s)
    except ValueError:
        return 0.0


def clean_numeric(series):
    return series.apply(parse_mixed_number)


def load_and_prepare(path):
    df = load_csv(path)
    df = normalize_columns(df)
    df = normalize_string_columns(df)
    return df


# =========================
# LOAD DATA
# =========================
def load_data():
    print("Loading datasets...")

    products_usage = load_and_prepare(PRODUCTS_USAGE)
    internal_usage = load_and_prepare(INTERNAL_USAGE)
    pricing = load_and_prepare(AWS_PRICING)
    products_budget = load_and_prepare(PRODUCTS_BUDGET)
    internal_budget = load_and_prepare(INTERNAL_BUDGET)

    return (
        products_usage,
        internal_usage,
        pricing,
        products_budget,
        internal_budget,
    )


# =========================
# COST COMPUTATION
# =========================
def compute_products_cost(products_usage, pricing):
    products_usage = products_usage.copy()
    pricing = pricing.copy()

    products_usage["usage_value"] = clean_numeric(products_usage["usage_value"])
    pricing["unit_price_usd"] = clean_numeric(pricing["unit_price_usd"])

    merged = products_usage.merge(
        pricing,
        on=["service", "usage_unit"],
        how="left"
    )

    merged["Actual Cost"] = merged["usage_value"] * merged["unit_price_usd"]

    return merged


def compute_internal_cost(internal_usage, pricing):
    internal_usage = internal_usage.copy()
    pricing = pricing.copy()

    internal_usage["usage_value"] = clean_numeric(internal_usage["usage_value"])
    pricing["unit_price_usd"] = clean_numeric(pricing["unit_price_usd"])

    merged = internal_usage.merge(
        pricing,
        on=["service", "usage_unit"],
        how="left"
    )

    merged["Actual Cost"] = merged["usage_value"] * merged["unit_price_usd"]

    return merged


# =========================
# AGGREGATION HELPERS
# =========================
def aggregate_products_actual_cost(cost_df):
    grouped = (
        cost_df.groupby(
            ["date", "day", "week", "month", "product", "division"],
            dropna=False
        )["Actual Cost"]
        .sum()
        .reset_index()
    )
    return grouped


def aggregate_internal_actual_cost(cost_df):
    grouped = (
        cost_df.groupby(
            ["date", "day", "week", "month", "division"],
            dropna=False
        )["Actual Cost"]
        .sum()
        .reset_index()
    )
    return grouped


def aggregate_products_budget(budget_df):
    budget_df = budget_df.copy()

    budget_df["Allocated Budget ($)"] = clean_numeric(budget_df["Allocated Budget ($)"])
    budget_df["Actual Cost ($)"] = clean_numeric(budget_df["Actual Cost ($)"])
    budget_df["Variance ($)"] = clean_numeric(budget_df["Variance ($)"])
    budget_df["Variance %"] = clean_numeric(budget_df["Variance %"])
    budget_df["Target Overage vs Budget %"] = clean_numeric(budget_df["Target Overage vs Budget %"])

    grouped = (
        budget_df.groupby(
            ["Date", "Day", "Week", "Month", "Product", "Division"],
            dropna=False
        )[[
            "Allocated Budget ($)",
            "Actual Cost ($)",
            "Variance ($)",
            "Variance %",
            "Target Overage vs Budget %"
        ]]
        .sum()
        .reset_index()
    )

    return grouped


def aggregate_internal_budget(budget_df):
    budget_df = budget_df.copy()

    budget_df["Allocated Budget ($)"] = clean_numeric(budget_df["Allocated Budget ($)"])
    budget_df["Actual Cost ($)"] = clean_numeric(budget_df["Actual Cost ($)"])
    budget_df["Variance ($)"] = clean_numeric(budget_df["Variance ($)"])
    budget_df["Variance %"] = clean_numeric(budget_df["Variance %"])
    budget_df["Target Overage vs Budget %"] = clean_numeric(budget_df["Target Overage vs Budget %"])

    grouped = (
        budget_df.groupby(
            ["Date", "Day", "Week", "Month", "Division"],
            dropna=False
        )[[
            "Allocated Budget ($)",
            "Actual Cost ($)",
            "Variance ($)",
            "Variance %",
            "Target Overage vs Budget %"
        ]]
        .sum()
        .reset_index()
    )

    return grouped


# =========================
# FINAL ANALYSIS
# =========================
def compute_products_analysis(products_cost, products_budget):
    actuals = aggregate_products_actual_cost(products_cost)
    budgets = aggregate_products_budget(products_budget)

    merged = actuals.merge(
        budgets,
        left_on=["date", "day", "week", "month", "product", "division"],
        right_on=["Date", "Day", "Week", "Month", "Product", "Division"],
        how="left"
    )

    merged["Variance Recomputed"] = merged["Actual Cost"] - merged["Allocated Budget ($)"]
    merged["Variance % Recomputed"] = (
        merged["Variance Recomputed"] / merged["Allocated Budget ($)"]
    ).replace([float("inf"), -float("inf")], 0).fillna(0) * 100

    return merged


def compute_internal_analysis(internal_cost, internal_budget):
    actuals = aggregate_internal_actual_cost(internal_cost)
    budgets = aggregate_internal_budget(internal_budget)

    merged = actuals.merge(
        budgets,
        left_on=["date", "day", "week", "month", "division"],
        right_on=["Date", "Day", "Week", "Month", "Division"],
        how="left"
    )

    merged["Variance Recomputed"] = merged["Actual Cost"] - merged["Allocated Budget ($)"]
    merged["Variance % Recomputed"] = (
        merged["Variance Recomputed"] / merged["Allocated Budget ($)"]
    ).replace([float("inf"), -float("inf")], 0).fillna(0) 

    return merged


# =========================
# MAIN
# =========================
def main():
    (
        products_usage,
        internal_usage,
        pricing,
        products_budget,
        internal_budget,
    ) = load_data()

    print("Computing products actual cost...")
    products_cost = compute_products_cost(products_usage, pricing)

    print("Computing internal actual cost...")
    internal_cost = compute_internal_cost(internal_usage, pricing)

    print("Computing products variance...")
    products_analysis = compute_products_analysis(products_cost, products_budget)

    print("Computing internal variance...")
    internal_analysis = compute_internal_analysis(internal_cost, internal_budget)

    products_analysis.to_csv(OUTPUT_PRODUCTS, index=False)
    internal_analysis.to_csv(OUTPUT_INTERNAL, index=False)

    print("Done")
    print("Files created:")
    print(OUTPUT_PRODUCTS)
    print(OUTPUT_INTERNAL)


if __name__ == "__main__":
    main()