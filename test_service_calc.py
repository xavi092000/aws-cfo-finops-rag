from cfo_finops_athena_rag_final import (
    build_service_spend_scope,
    load_selected_period_daily_spend,
)

selection = {
    "mode": "days",
    "block": "C",
    "week": 9,
    "days": 3,
    "period": "Monday to Wednesday",
}

question = "what was the service with the most cost during that period of time"

division_daily_spend, product_daily_spend = load_selected_period_daily_spend(selection)

service_scoped = build_service_spend_scope(
    product_daily_spend=product_daily_spend,
    division_daily_spend=division_daily_spend,
    question=question,
)

print("\n==============================")
print("SERVICE SCOPED PREVIEW")
print("==============================")
print(service_scoped[["Dataset", "Service", "Actual Cost"]].head(20))

if service_scoped.empty:
    print("\nNo service-level rows found for this selection.")
else:
    grouped = (
        service_scoped.groupby("Service", dropna=False)["Actual Cost"]
        .sum()
        .reset_index()
        .sort_values("Actual Cost", ascending=False)
        .reset_index(drop=True)
    )

    print("\n==============================")
    print("TOP 10 SERVICES")
    print("==============================")
    print(grouped.head(10).to_string(index=False))

    print("\n==============================")
    print("TOP SERVICE")
    print("==============================")
    top_row = grouped.iloc[0]
    print(f"Service: {top_row['Service']}")
    print(f"Actual Cost: ${top_row['Actual Cost']:,.2f}")

    total_services = grouped["Actual Cost"].sum()
    print("\n==============================")
    print("SERVICE TOTAL")
    print("==============================")
    print(f"Total service spend across grouped services: ${total_services:,.2f}")