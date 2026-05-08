import pandas as pd
from pathlib import Path


# =========================
# PATHS
# =========================
PROJECT_DIR = Path(".")
DATA_DIR = Path("data")

PRODUCTS_ANALYSIS_FILE = PROJECT_DIR / "products_finops_analysis.csv"
INTERNAL_ANALYSIS_FILE = PROJECT_DIR / "internal_finops_analysis.csv"

PRODUCTS_USAGE_FILE = DATA_DIR / "products_usage_84_days.csv"
INTERNAL_USAGE_FILE = DATA_DIR / "internal_usage_84_days.csv"
AWS_PRICING_FILE = DATA_DIR / "aws_pricing.csv"


# =========================
# HELPERS
# =========================
def load_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";")
        if len(df.columns) == 1:
            df = pd.read_csv(path)
        return df
    except Exception:
        return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
    return df


def clean_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("\u00A0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )


def find_first_existing_column(df: pd.DataFrame, possible_names: list[str]):
    normalized = {col.lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_string_columns(normalize_columns(df))


# =========================
# LOAD
# =========================
def load_all():
    products_analysis = prepare_df(load_csv(PRODUCTS_ANALYSIS_FILE))
    internal_analysis = prepare_df(load_csv(INTERNAL_ANALYSIS_FILE))
    products_usage = prepare_df(load_csv(PRODUCTS_USAGE_FILE))
    internal_usage = prepare_df(load_csv(INTERNAL_USAGE_FILE))
    aws_pricing = prepare_df(load_csv(AWS_PRICING_FILE))

    return products_analysis, internal_analysis, products_usage, internal_usage, aws_pricing


# =========================
# AUDITS
# =========================
def audit_pricing_table(aws_pricing: pd.DataFrame):
    print("\n" + "=" * 80)
    print("AUDIT 1 - AWS PRICING TABLE")
    print("=" * 80)

    price_col = find_first_existing_column(aws_pricing, ["unit_price_usd", "unit_price", "price_usd", "price"])
    service_col = find_first_existing_column(aws_pricing, ["service"])
    unit_col = find_first_existing_column(aws_pricing, ["usage_unit", "unit"])

    if price_col is None or service_col is None or unit_col is None:
        print("⚠️ Missing required columns in aws_pricing.csv")
        print("Columns found:", list(aws_pricing.columns))
        return

    aws_pricing[price_col] = clean_numeric(aws_pricing[price_col])

    print("Pricing rows:", len(aws_pricing))
    print("Services:", sorted(aws_pricing[service_col].dropna().unique().tolist()))
    print("\nPricing preview:")
    print(aws_pricing[[service_col, unit_col, price_col]].head(20))

    zero_or_negative = aws_pricing[aws_pricing[price_col] <= 0]
    if len(zero_or_negative) > 0:
        print("\n⚠️ Services with zero or negative pricing:")
        print(zero_or_negative[[service_col, unit_col, price_col]])
    else:
        print("\n✅ No zero or negative prices detected.")


def audit_usage_coverage(products_usage: pd.DataFrame, internal_usage: pd.DataFrame, aws_pricing: pd.DataFrame):
    print("\n" + "=" * 80)
    print("AUDIT 2 - USAGE TO PRICING COVERAGE")
    print("=" * 80)

    price_service_col = find_first_existing_column(aws_pricing, ["service"])
    price_unit_col = find_first_existing_column(aws_pricing, ["usage_unit", "unit"])

    def check_coverage(label: str, usage_df: pd.DataFrame):
        usage_service_col = find_first_existing_column(usage_df, ["service"])
        usage_unit_col = find_first_existing_column(usage_df, ["usage_unit", "unit"])

        usage_pairs = (
            usage_df[[usage_service_col, usage_unit_col]]
            .drop_duplicates()
            .rename(columns={usage_service_col: "service", usage_unit_col: "usage_unit"})
        )

        pricing_pairs = (
            aws_pricing[[price_service_col, price_unit_col]]
            .drop_duplicates()
            .rename(columns={price_service_col: "service", price_unit_col: "usage_unit"})
        )

        merged = usage_pairs.merge(
            pricing_pairs,
            on=["service", "usage_unit"],
            how="left",
            indicator=True
        )

        missing = merged[merged["_merge"] == "left_only"]

        print(f"\n{label}")
        print(f"Unique usage pairs: {len(usage_pairs)}")
        print(f"Missing pricing pairs: {len(missing)}")

        if len(missing) > 0:
            print("⚠️ Missing service/unit mappings:")
            print(missing[["service", "usage_unit"]])
        else:
            print("✅ All service/unit pairs are covered by aws_pricing.csv")

    check_coverage("PRODUCTS USAGE COVERAGE", products_usage)
    check_coverage("INTERNAL USAGE COVERAGE", internal_usage)


def audit_actual_costs(products_analysis: pd.DataFrame, internal_analysis: pd.DataFrame):
    print("\n" + "=" * 80)
    print("AUDIT 3 - ACTUAL COST PLAUSIBILITY")
    print("=" * 80)

    def summarize(label: str, df: pd.DataFrame):
        actual_col = find_first_existing_column(df, ["Actual Cost", "Actual Cost Computed"])
        budget_col = find_first_existing_column(df, ["Allocated Budget ($)", "Allocated Budget Computed"])
        variance_col = find_first_existing_column(df, ["Variance"])
        variance_pct_col = find_first_existing_column(df, ["Variance %"])

        print(f"\n{label}")

        if actual_col is None:
            print("⚠️ Actual cost column not found.")
            return

        df = df.copy()
        df[actual_col] = clean_numeric(df[actual_col])

        if budget_col is not None:
            df[budget_col] = clean_numeric(df[budget_col])

        if variance_col is not None:
            df[variance_col] = clean_numeric(df[variance_col])

        if variance_pct_col is not None:
            df[variance_pct_col] = clean_numeric(df[variance_pct_col])

        print(f"Rows: {len(df)}")
        print(f"Total actual cost: {df[actual_col].sum():,.2f}")
        print(f"Average actual cost: {df[actual_col].mean():,.2f}")
        print(f"Median actual cost: {df[actual_col].median():,.2f}")
        print(f"Max actual cost: {df[actual_col].max():,.2f}")

        if budget_col is not None:
            print(f"Total allocated budget: {df[budget_col].sum():,.2f}")

        if variance_col is not None:
            print(f"Total variance: {df[variance_col].sum():,.2f}")

        if variance_pct_col is not None:
            suspicious = df[df[variance_pct_col].abs() > 500]
            print(f"Rows with |variance %| > 500: {len(suspicious)}")

            if len(suspicious) > 0:
                cols_to_show = [c for c in ["date", "day", "week", "month", "product", "division", actual_col, budget_col, variance_col, variance_pct_col] if c in df.columns]
                print("⚠️ Suspicious rows preview:")
                print(suspicious[cols_to_show].head(20))

    summarize("PRODUCTS FINOPS ANALYSIS", products_analysis)
    summarize("INTERNAL FINOPS ANALYSIS", internal_analysis)


def audit_cost_by_service(products_usage: pd.DataFrame, internal_usage: pd.DataFrame, aws_pricing: pd.DataFrame):
    print("\n" + "=" * 80)
    print("AUDIT 4 - SERVICE COST ORDER OF MAGNITUDE")
    print("=" * 80)

    price_col = find_first_existing_column(aws_pricing, ["unit_price_usd", "unit_price", "price_usd", "price"])
    price_service_col = find_first_existing_column(aws_pricing, ["service"])
    price_unit_col = find_first_existing_column(aws_pricing, ["usage_unit", "unit"])

    aws_pricing = aws_pricing.copy()
    aws_pricing[price_col] = clean_numeric(aws_pricing[price_col])

    def compute(label: str, usage_df: pd.DataFrame):
        usage_service_col = find_first_existing_column(usage_df, ["service"])
        usage_unit_col = find_first_existing_column(usage_df, ["usage_unit", "unit"])
        usage_value_col = find_first_existing_column(usage_df, ["usage_value", "usage"])

        temp = usage_df.copy()
        temp[usage_value_col] = clean_numeric(temp[usage_value_col])

        merged = temp.merge(
            aws_pricing,
            left_on=[usage_service_col, usage_unit_col],
            right_on=[price_service_col, price_unit_col],
            how="left"
        )

        merged["computed_cost"] = merged[usage_value_col] * merged[price_col]

        grouped = (
            merged.groupby(usage_service_col, dropna=False)["computed_cost"]
            .sum()
            .reset_index()
            .sort_values("computed_cost", ascending=False)
        )

        print(f"\n{label}")
        print(grouped)

    compute("PRODUCTS COST BY SERVICE", products_usage)
    compute("INTERNAL COST BY SERVICE", internal_usage)


# =========================
# MAIN
# =========================
def main():
    products_analysis, internal_analysis, products_usage, internal_usage, aws_pricing = load_all()

    audit_pricing_table(aws_pricing)
    audit_usage_coverage(products_usage, internal_usage, aws_pricing)
    audit_actual_costs(products_analysis, internal_analysis)
    audit_cost_by_service(products_usage, internal_usage, aws_pricing)


if __name__ == "__main__":
    main()