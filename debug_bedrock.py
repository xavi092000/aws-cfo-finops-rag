import pandas as pd

usage = pd.read_csv("data/products_usage_84_days.csv")
pricing = pd.read_csv("data/aws_pricing.csv")

usage.columns = usage.columns.str.strip()
pricing.columns = pricing.columns.str.strip()

print("\n==============================")
print("USAGE COLUMNS")
print("==============================")
print(list(usage.columns))

print("\n==============================")
print("PRICING COLUMNS")
print("==============================")
print(list(pricing.columns))

bedrock_usage = usage[usage["service"].astype(str).str.strip() == "Bedrock"].copy()
bedrock_price = pricing[pricing["service"].astype(str).str.strip() == "Bedrock"].copy()

print("\n==============================")
print("BEDROCK USAGE SAMPLE")
print("==============================")
print(bedrock_usage.head(10).to_string(index=False))

print("\nTOTAL BEDROCK ROWS:", len(bedrock_usage))

print("\n==============================")
print("BEDROCK PRICING")
print("==============================")
print(bedrock_price.to_string(index=False))

print("\n==============================")
print("BEDROCK USAGE NUMERIC SUMMARY")
print("==============================")
print(bedrock_usage.describe(include="all"))

possible_cost_cols = [
    "actual_cost_usd",
    "actual_cost",
    "cost",
    "cost_usd",
    "total_cost",
]

found_cost_col = None
for col in possible_cost_cols:
    if col in bedrock_usage.columns:
        found_cost_col = col
        break

print("\n==============================")
print("DETECTED COST COLUMN")
print("==============================")
print(found_cost_col if found_cost_col else "NONE FOUND")

if found_cost_col:
    total_cost = pd.to_numeric(bedrock_usage[found_cost_col], errors="coerce").fillna(0).sum()
    print("\n==============================")
    print("TOTAL BEDROCK COST")
    print("==============================")
    print(f"${total_cost:,.2f}")
else:
    print("\nNo direct cost column found.")
    print("We need to inspect token/usage columns and recompute cost manually.")