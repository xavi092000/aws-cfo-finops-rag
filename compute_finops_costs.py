import pandas as pd
from pathlib import Path

USAGE_PRODUCTS_FILE = "products_usage_84_days.csv"
USAGE_INTERNAL_FILE = "internal_usage_84_days.csv"
PRICING_FILE = "product_pricing.csv"

OUTPUT_PRODUCTS = "products_usage_with_cost.csv"
OUTPUT_INTERNAL = "internal_usage_with_cost.csv"


def normalize_string_columns(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
    return df


def load_data():
    products_usage = pd.read_csv(USAGE_PRODUCTS_FILE)
    internal_usage = pd.read_csv(USAGE_INTERNAL_FILE)
    pricing = pd.read_csv(PRICING_FILE)

    products_usage = normalize_string_columns(products_usage)
    internal_usage = normalize_string_columns(internal_usage)
    pricing = normalize_string_columns(pricing)

    return products_usage, internal_usage, pricing


def compute_cost(df, pricing):
    merged = df.merge(
        pricing,
        on="service",
        how="left"
    )

    merged["Actual Cost"] = merged["usage_value"] * merged["price"]

    return merged


def main():

    products_usage, internal_usage, pricing = load_data()

    print("Computing product costs...")
    products_cost = compute_cost(products_usage, pricing)

    print("Computing internal costs...")
    internal_cost = compute_cost(internal_usage, pricing)

    products_cost.to_csv(OUTPUT_PRODUCTS, index=False)
    internal_cost.to_csv(OUTPUT_INTERNAL, index=False)

    print("Cost computation completed")
    print(f"Saved: {OUTPUT_PRODUCTS}")
    print(f"Saved: {OUTPUT_INTERNAL}")


if __name__ == "__main__":
    main()