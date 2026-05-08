from pathlib import Path

TARGET = Path("cfo_finops_athena_rag_final.py")
BACKUP = Path("cfo_finops_athena_rag_final_backup_before_repair.py")

NEW_FUNCTION = r'''def answer_analytical_question(
    question: str,
    product_filtered: pd.DataFrame,
    division_filtered: pd.DataFrame,
    product_daily_spend: Optional[pd.DataFrame] = None,
    division_daily_spend: Optional[pd.DataFrame] = None,
) -> str:
    q = question.lower()
    metric = detect_metric(question)
    dimension = detect_dimension(question).lower()
    base_df = get_base_dataset(dimension, product_filtered, division_filtered)

    if base_df is None or base_df.empty:
        return "No data available for the selected period."

    base_df = standardize_columns(base_df)
    scope_label = build_scope_label(question)

    total_scope = standardize_columns(product_filtered.copy())

    if "Variance" not in total_scope.columns and "Actual Cost" in total_scope.columns and "Allocated Budget" in total_scope.columns:
        total_scope["Variance"] = total_scope["Actual Cost"] - total_scope["Allocated Budget"]

    if "most important numerical insight" in q:
        actual = float(total_scope["Actual Cost"].sum())
        budget = float(total_scope["Allocated Budget"].sum())
        variance = actual - budget
        return (
            f"The most important insight is a variance of "
            f"{format_currency(variance, force_sign=True)} "
            f"({'over' if variance > 0 else 'under'} budget)."
        )

    if "budget" in q and "total" in q:
        total_budget = float(total_scope["Allocated Budget"].sum())
        return f"The total allocated budget is {format_currency(total_budget)}."

    if "variance" in q and ("dollar" in q or "$" in q):
        total_variance = float(total_scope["Variance"].sum())
        return f"The total variance is {format_currency(total_variance, force_sign=True)}."

    if "variance" in q and ("percent" in q or "percentage" in q or "%" in q):
        actual = float(total_scope["Actual Cost"].sum())
        budget = float(total_scope["Allocated Budget"].sum())

        if budget == 0:
            return "Cannot compute variance percentage."

        pct = ((actual - budget) / budget) * 100
        return f"The variance percentage is {format_percent(pct, force_sign=True)}."

    if "overage" in q or "over budget" in q or "overspend" in q:
        overage = float(total_scope["Variance"].sum())
        return f"The budget overage is {format_currency(overage, force_sign=True)}."

    if "average" in q or "avg" in q:
        avg_value = float(base_df["Actual Cost"].mean())
        return f"The average actual cost is {format_currency(avg_value)}."

    if "versus" in q or "vs" in q:
        return answer_compare_question(base_df, "Actual Cost")

    if "summary" in q or "insight" in q or "important" in q or "review" in q or "prioritize" in q:
        actual = float(total_scope["Actual Cost"].sum())
        budget = float(total_scope["Allocated Budget"].sum())
        variance = actual - budget

        return (
            f"The most critical insight is that total spend is "
            f"{'over' if variance > 0 else 'under'} budget by "
            f"{format_currency(variance, force_sign=True)}. "
            f"The CFO should prioritize reviewing the highest variance drivers."
        )

    if dimension in ["service", "product", "division"]:
        working_scope = base_df.copy()

        if working_scope.empty:
            return "No rows matched the selected period for the requested analytical scope."

        dim_col_map = {
            "service": "Service",
            "product": "Product",
            "division": "Division",
        }

        dim_col = dim_col_map[dimension]

        if dim_col not in working_scope.columns:
            return (
                f"I found the requested analytical scope, but the dimension '{dim_col}' "
                f"is not available in the current dataset."
            )

        working_scope = working_scope[
            working_scope[dim_col].astype(str) != "DATA_NOT_AVAILABLE"
        ].copy()

        if working_scope.empty:
            return (
                f"I found the requested analytical scope, but no valid '{dim_col}' "
                f"data is available for the selected period."
            )

        working_scope = standardize_columns(working_scope)

        if "Variance" not in working_scope.columns and "Actual Cost" in working_scope.columns and "Allocated Budget" in working_scope.columns:
            working_scope["Variance"] = working_scope["Actual Cost"] - working_scope["Allocated Budget"]

        available_cols = [
            col for col in ["Actual Cost", "Allocated Budget", "Variance"]
            if col in working_scope.columns
        ]

        working_scope = working_scope.groupby(dim_col, as_index=False)[available_cols].sum()

        return answer_top_bottom_question(
            working_scope,
            question,
            metric,
            dim_col,
            scope_label,
        )

    if dimension == "dataset":
        actual = float(total_scope["Actual Cost"].sum())
        budget = float(total_scope["Allocated Budget"].sum())
        variance = actual - budget
        variance_pct = safe_variance_pct(actual, budget)

        return (
            f"Selected-period analytical summary:\n"
            f"- Total actual cost: {format_currency(actual)}\n"
            f"- Allocated budget: {format_currency(budget)}\n"
            f"- Variance: {format_currency(variance, force_sign=True)} "
            f"({format_percent(variance_pct, force_sign=True)})"
        )

    return "No supported analytical route matched this question."
'''

def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    BACKUP.write_text(text, encoding="utf-8")

    start_marker = "def answer_analytical_question("
    end_marker = "# =========================================================\n# GENERAL RAG\n# ========================================================="

    start = text.find(start_marker)
    if start == -1:
        raise RuntimeError("Could not find answer_analytical_question().")

    end = text.find(end_marker, start)
    if end == -1:
        raise RuntimeError("Could not find GENERAL RAG marker after answer_analytical_question().")

    new_text = text[:start] + NEW_FUNCTION + "\n\n" + text[end:]
    TARGET.write_text(new_text, encoding="utf-8")

    print("Repair completed.")
    print(f"Backup saved to: {BACKUP}")
    print(f"Updated file: {TARGET}")

if __name__ == "__main__":
    main()