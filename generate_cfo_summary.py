import pandas as pd
from pathlib import Path


# =========================
# FILES
# =========================
PRODUCTS_FILE = Path("products_finops_analysis.csv")
INTERNAL_FILE = Path("internal_finops_analysis.csv")


# =========================
# USER SELECTION
# =========================
selection = {
    "mode": "weeks",
    "weeks": [5, 6]
}


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
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )


def find_first_existing_column(df: pd.DataFrame, possible_names: list[str]):
    normalized = {col.lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


def safe_variance_pct(actual: float, budget: float) -> float:
    if budget == 0:
        return 0.0
    return ((actual - budget) / budget) * 100.0


def format_currency(value: float, force_sign: bool = False) -> str:
    if force_sign:
        sign = "+" if value > 0 else ""
        return f"{sign}${value:,.2f}"
    return f"${value:,.2f}"


def format_percent(value: float, force_sign: bool = False) -> str:
    if force_sign:
        sign = "+" if value > 0 else ""
        return f"{sign}{value:,.2f}%"
    return f"{value:,.2f}%"


def build_status_text(variance_value: float) -> str:
    if variance_value > 0:
        return "Over Budget"
    if variance_value < 0:
        return "Under Budget"
    return "On Budget"


# =========================
# FILTER BY SCOPE
# =========================
def filter_by_selection(df: pd.DataFrame, selection: dict) -> pd.DataFrame:
    df = df.copy()

    week_col = find_first_existing_column(df, ["week", "Week"])
    day_col = find_first_existing_column(df, ["day", "Day"])
    month_col = find_first_existing_column(df, ["month", "Month"])

    if selection["mode"] == "weeks":
        week_labels = [f"Week {w}" for w in selection["weeks"]]
        return df[df[week_col].isin(week_labels)].copy()

    if selection["mode"] == "days":
        week_label = f"Week {selection['week']}"
        return df[
            (df[week_col] == week_label) &
            (df[day_col].isin(selection["day_labels"]))
        ].copy()

    if selection["mode"] == "months":
        month_labels = selection["month_labels"]
        return df[df[month_col].isin(month_labels)].copy()

    raise ValueError(f"Unsupported mode: {selection['mode']}")


# =========================
# DAILY DETAIL TABLE
# =========================
def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    actual_col = find_first_existing_column(
        df,
        ["Actual Cost", "Actual Cost Computed", "Actual Cost ($)"]
    )
    budget_col = find_first_existing_column(
        df,
        ["Allocated Budget ($)", "Allocated Budget Computed", "Allocated Budget"]
    )
    variance_col = find_first_existing_column(
        df,
        ["Variance Recomputed", "Variance ($)", "Variance"]
    )
    week_col = find_first_existing_column(df, ["week", "Week"])
    day_col = find_first_existing_column(df, ["day", "Day"])

    if actual_col is None:
        raise ValueError(f"Could not find actual cost column. Found columns: {list(df.columns)}")
    if budget_col is None:
        raise ValueError(f"Could not find budget column. Found columns: {list(df.columns)}")
    if variance_col is None:
        raise ValueError(f"Could not find variance column. Found columns: {list(df.columns)}")

    df[actual_col] = clean_numeric(df[actual_col])
    df[budget_col] = clean_numeric(df[budget_col])
    df[variance_col] = clean_numeric(df[variance_col])

    grouped = (
        df.groupby([week_col, day_col], dropna=False)[[actual_col, budget_col, variance_col]]
        .sum()
        .reset_index()
    )

    grouped["Variance %"] = grouped.apply(
        lambda row: safe_variance_pct(row[actual_col], row[budget_col]),
        axis=1
    )

    grouped = grouped.rename(
        columns={
            week_col: "Week",
            day_col: "Day",
            actual_col: "Actual Cost",
            budget_col: "Allocated Budget",
            variance_col: "Variance"
        }
    )

    return grouped


# =========================
# HORIZONTAL CARDS
# =========================
def make_day_card(row: pd.Series, width: int = 34) -> list[str]:
    lines = [
        f"{row['Week']} {row['Day']}",
        f"Actual Cost: {format_currency(row['Actual Cost'])}",
        f"Budget: {format_currency(row['Allocated Budget'])}",
        f"Variance: {format_currency(row['Variance'], force_sign=True)} ({format_percent(row['Variance %'], force_sign=True)})"
    ]
    return [line.ljust(width)[:width] for line in lines]


def render_cards_side_by_side(df: pd.DataFrame, cards_per_row: int = 3, width: int = 34, gap: int = 4) -> str:
    if df.empty:
        return "No matching days."

    card_blocks = [make_day_card(row, width=width) for _, row in df.iterrows()]
    rendered_lines = []

    for i in range(0, len(card_blocks), cards_per_row):
        row_cards = card_blocks[i:i + cards_per_row]
        max_lines = max(len(card) for card in row_cards)

        for line_idx in range(max_lines):
            row_line = (" " * gap).join(card[line_idx] for card in row_cards)
            rendered_lines.append(row_line.rstrip())

        if i + cards_per_row < len(card_blocks):
            rendered_lines.append("")

    return "\n".join(rendered_lines)


# =========================
# BUILD SUMMARY
# =========================
def generate_summary(df: pd.DataFrame, label: str) -> str:
    df = df.copy()

    if df.empty:
        return f"""
======================================================================
CFO SUMMARY - {label.upper()}
======================================================================

No rows matched the selected scope.
"""

    daily_summary = build_daily_summary(df)

    total_budget = daily_summary["Allocated Budget"].sum()
    total_actual = daily_summary["Actual Cost"].sum()
    total_variance = daily_summary["Variance"].sum()
    total_variance_pct = safe_variance_pct(total_actual, total_budget)
    status = build_status_text(total_variance)

    over_budget_days = daily_summary[daily_summary["Variance"] > 0].copy()
    under_budget_days = daily_summary[daily_summary["Variance"] < 0].copy()

    top_over = over_budget_days.sort_values("Variance", ascending=False).head(3)
    top_under = under_budget_days.sort_values("Variance", ascending=True).head(3)

    lines = []
    lines.append("")
    lines.append("======================================================================")
    lines.append(f"CFO SUMMARY - {label.upper()}")
    lines.append("======================================================================")
    lines.append("")
    lines.append(f"Total Budget: {format_currency(total_budget)}")
    lines.append(f"Total Actual Cost: {format_currency(total_actual)}")
    lines.append(f"Variance: {format_currency(total_variance, force_sign=True)} ({format_percent(total_variance_pct, force_sign=True)})")
    lines.append(f"Status: {status}")
    lines.append("")
    lines.append("Top Over Budget Days:")
    lines.append("")

    if top_over.empty:
        lines.append("No over budget days in selected scope.")
    else:
        lines.append(render_cards_side_by_side(top_over, cards_per_row=3, width=34, gap=4))

    lines.append("")
    lines.append("Top Under Budget Days:")
    lines.append("")

    if top_under.empty:
        lines.append("No under budget days in selected scope.")
    else:
        for _, row in top_under.iterrows():
            lines.append(f"{row['Week']} {row['Day']}")
            lines.append(f"Actual Cost: {format_currency(row['Actual Cost'])}")
            lines.append(f"Budget: {format_currency(row['Allocated Budget'])}")
            lines.append(f"Variance: {format_currency(row['Variance'], force_sign=True)} ({format_percent(row['Variance %'], force_sign=True)})")
            lines.append("")

    return "\n".join(lines).rstrip()


# =========================
# MAIN
# =========================
def main():
    products = normalize_string_columns(normalize_columns(load_csv(PRODUCTS_FILE)))
    internal = normalize_string_columns(normalize_columns(load_csv(INTERNAL_FILE)))

    products_filtered = filter_by_selection(products, selection)
    internal_filtered = filter_by_selection(internal, selection)

    print("\n======================================================================")
    print("SELECTION USED")
    print("======================================================================")
    print(selection)

    print("\n======================================================================")
    print("FILTERED ROW COUNTS")
    print("======================================================================")
    print(f"Products rows matched: {len(products_filtered)}")
    print(f"Internal rows matched: {len(internal_filtered)}")

    print(generate_summary(products_filtered, "Products"))
    print(generate_summary(internal_filtered, "Internal"))


if __name__ == "__main__":
    main()