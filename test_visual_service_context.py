from load_athena_views import load_current_powerbi_views
import pandas as pd

def main() -> None:
    division_df, product_df, service_df = load_current_powerbi_views()

    print("\n=== RAW current_service_view ===")
    print(service_df.head(20).to_string(index=False))
    print(f"\nRows: {len(service_df)}")
    print(f"Columns: {service_df.columns.tolist()}")

    # Ajuste ici selon le contexte exact du visuel
    # Exemple actuel observé: le visuel semble montrer seulement Week 11
    visual_df = service_df.copy()

    # ⚠️ change cette valeur selon le tooltip / contexte du visuel
    visual_df = visual_df[visual_df["week"] == "Week 11"].copy()

    print("\n=== FILTERED TO VISUAL CONTEXT ===")
    print(visual_df.head(20).to_string(index=False))
    print(f"\nFiltered rows: {len(visual_df)}")

    grouped = (
        visual_df.groupby("service", as_index=False)["actual_cost_usd"]
        .sum()
        .sort_values("actual_cost_usd", ascending=False)
        .reset_index(drop=True)
    )

    print("\n=== VISUAL-CONTEXT SERVICE TOTALS ===")
    print(grouped.to_string(index=False))

    total = grouped["actual_cost_usd"].sum()
    print(f"\nVisual-context total actual_cost_usd: {total:,.2f}")

if __name__ == "__main__":
    main()