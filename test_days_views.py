from load_athena_views import load_current_powerbi_views
from agent_entree_finopsrag import update_powerbi_views_from_selection

def main() -> None:
    selection = {
        "mode": "days",
        "block": "B",
        "week": 7,
        "days": 4,
        "period": "Monday to Thursday",
    }

    print("=== STEP 1: update_powerbi_views_from_selection ===")
    ok = update_powerbi_views_from_selection(selection)
    print(f"Views updated: {ok}")

    print("\n=== STEP 2: load_current_powerbi_views ===")
    division_df, product_df, service_df = load_current_powerbi_views()

    print(f"division rows: {len(division_df)}")
    print(f"product rows: {len(product_df)}")
    print(f"service rows: {len(service_df)}")

    print("\n=== DISTINCT period/day/week/month ===")
    if "period_value" in product_df.columns:
        print("\nproduct period_value:")
        print(product_df["period_value"].drop_duplicates().sort_values().to_string(index=False))

    if "day" in product_df.columns:
        print("\nproduct day:")
        print(product_df["day"].drop_duplicates().sort_values().to_string(index=False))

    if "week" in product_df.columns:
        print("\nproduct week:")
        print(product_df["week"].drop_duplicates().sort_values().to_string(index=False))

    if "month" in product_df.columns:
        print("\nproduct month:")
        print(product_df["month"].drop_duplicates().sort_values().to_string(index=False))

    print("\n=== KPI CHECK ===")
    total_actual = product_df["actual_cost_usd"].sum()
    total_budget = product_df["allocated_budget_usd"].sum()
    total_variance = product_df["variance_usd"].sum()
    variance_pct = (total_variance / total_budget * 100) if total_budget else 0.0

    print(f"Total actual   : {total_actual:,.2f}")
    print(f"Total budget   : {total_budget:,.2f}")
    print(f"Total variance : {total_variance:,.2f}")
    print(f"Variance %     : {variance_pct:,.2f}")

    print("\n=== TOP SERVICE FROM division_df (Power BI visual logic) ===")
    top_service = (
        division_df.groupby("service", as_index=False)["actual_cost_usd"]
        .sum()
        .sort_values("actual_cost_usd", ascending=False)
        .reset_index(drop=True)
    )
    print(top_service.head(10).to_string(index=False))

    print("\n=== TOP PRODUCT VARIANCE ===")
    top_product = (
        product_df.groupby("product", as_index=False)["variance_usd"]
        .sum()
        .sort_values("variance_usd", ascending=False)
        .reset_index(drop=True)
    )
    print(top_product.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

    