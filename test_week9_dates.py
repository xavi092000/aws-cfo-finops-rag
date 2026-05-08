from resolve_finops_analytics import load_selected_period_data

selection = {
    "mode": "days",
    "block": "C",
    "week": 9,
    "days": 3,
    "period": "Monday to Wednesday",
}

division_filtered, product_filtered = load_selected_period_data(selection)

print("\n==============================")
print("PRODUCT DATES")
print("==============================")
product_dates = (
    product_filtered[["date", "day", "week", "month"]]
    .drop_duplicates()
    .sort_values("date")
)
print(product_dates.to_string(index=False))

print("\n==============================")
print("INTERNAL DATES")
print("==============================")
internal_dates = (
    division_filtered[["date", "day", "week", "month"]]
    .drop_duplicates()
    .sort_values("date")
)
print(internal_dates.to_string(index=False))