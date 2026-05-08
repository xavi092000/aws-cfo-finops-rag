from pathlib import Path
import pandas as pd


# =========================
# CONFIG
# =========================
DATA_DIR = Path("data")

PRODUCTS_USAGE_FILE = DATA_DIR / "products_usage_84_days.csv"
INTERNAL_USAGE_FILE = DATA_DIR / "internal_usage_84_days.csv"

PRODUCTS_DAILY_BUDGET_FILE = DATA_DIR / "Products_Daily_Budget.csv"
PRODUCTS_WEEKLY_BUDGET_FILE = DATA_DIR / "Products_Weekly_Budgets.csv"
PRODUCTS_MONTHLY_BUDGET_FILE = DATA_DIR / "Products_Monthly_Budget.csv"

INTERNAL_DAILY_BUDGET_FILE = DATA_DIR / "Internal_Daily_Budget.csv"
INTERNAL_WEEKLY_BUDGET_FILE = DATA_DIR / "Internal_Weekly_Budget.csv"
INTERNAL_MONTHLY_BUDGET_FILE = DATA_DIR / "Internal_Monthly_Budget.csv"


# =========================
# LABEL MAPS
# =========================
DAY_LABELS_MAP = {
    1: ["Monday"],
    2: ["Monday", "Tuesday"],
    3: ["Monday", "Tuesday", "Wednesday"],
    4: ["Monday", "Tuesday", "Wednesday", "Thursday"],
    5: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    6: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
}

MONTH_BLOCK_MAP = {
    "A": ["Month 1"],
    "B": ["Month 2"],
    "C": ["Month 3"],
    "AB": ["Month 1", "Month 2"],
    "BC": ["Month 2", "Month 3"],
    "ABC": ["Month 1", "Month 2", "Month 3"],
}


# =========================
# LOAD CSV
# =========================
def load_csv_auto(file_path: Path) -> pd.DataFrame:
    """
    Try semicolon first, then comma.
    """
    try:
        df = pd.read_csv(file_path, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(file_path, sep=",")
        return df
    except Exception:
        return pd.read_csv(file_path)


# =========================
# CLEANUP HELPERS
# =========================
def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
    return df


def find_column(df: pd.DataFrame, possible_names: list[str]) -> str:
    """
    Find a matching column name, case-insensitive.
    """
    normalized_targets = [name.lower() for name in possible_names]

    for col in df.columns:
        if col.lower() in normalized_targets:
            return col

    raise ValueError(
        f"Column not found. Expected one of {possible_names}. Found columns: {list(df.columns)}"
    )


# =========================
# RESOLVE USER SELECTION
# =========================
def resolve_analysis_scope(selection: dict) -> dict:
    """
    Transform user selection into a normalized analytical scope.
    """

    mode = selection["mode"]

    if mode == "days":
        week_number = selection["week"]
        week_label = f"Week {week_number}"
        day_labels = DAY_LABELS_MAP[selection["days"]]

        if 1 <= week_number <= 4:
            month_labels = ["Month 1"]
        elif 5 <= week_number <= 8:
            month_labels = ["Month 2"]
        elif 9 <= week_number <= 12:
            month_labels = ["Month 3"]
        else:
            month_labels = []

        return {
            "mode": "days",
            "budget_level": "daily",
            "week_labels": [week_label],
            "day_labels": day_labels,
            "month_labels": month_labels,
        }

    if mode == "weeks":
        week_numbers = selection["weeks"]
        week_labels = [f"Week {w}" for w in week_numbers]

        month_labels = []
        if any(1 <= w <= 4 for w in week_numbers):
            month_labels.append("Month 1")
        if any(5 <= w <= 8 for w in week_numbers):
            month_labels.append("Month 2")
        if any(9 <= w <= 12 for w in week_numbers):
            month_labels.append("Month 3")

        return {
            "mode": "weeks",
            "budget_level": "weekly",
            "week_labels": week_labels,
            "day_labels": [],
            "month_labels": month_labels,
        }

    if mode == "months":
        monthly_block = selection["monthly_block"]
        month_labels = MONTH_BLOCK_MAP[monthly_block]

        if monthly_block == "A":
            week_labels = [f"Week {i}" for i in range(1, 5)]
        elif monthly_block == "B":
            week_labels = [f"Week {i}" for i in range(5, 9)]
        elif monthly_block == "C":
            week_labels = [f"Week {i}" for i in range(9, 13)]
        elif monthly_block == "AB":
            week_labels = [f"Week {i}" for i in range(1, 9)]
        elif monthly_block == "BC":
            week_labels = [f"Week {i}" for i in range(5, 13)]
        elif monthly_block == "ABC":
            week_labels = [f"Week {i}" for i in range(1, 13)]
        else:
            raise ValueError(f"Unsupported monthly block: {monthly_block}")

        return {
            "mode": "months",
            "budget_level": "monthly",
            "week_labels": week_labels,
            "day_labels": [],
            "month_labels": month_labels,
        }

    raise ValueError(f"Unsupported mode: {mode}")


# =========================
# FILTER USAGE TABLES
# =========================
def filter_usage_tables(
    products_usage_df: pd.DataFrame,
    internal_usage_df: pd.DataFrame,
    scope: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter expense / usage tables based on the resolved scope.
    """
    products_usage_df = normalize_string_columns(products_usage_df)
    internal_usage_df = normalize_string_columns(internal_usage_df)

    products_week_col = find_column(products_usage_df, ["week"])
    products_day_col = find_column(products_usage_df, ["day"])
    products_month_col = find_column(products_usage_df, ["month"])

    internal_week_col = find_column(internal_usage_df, ["week"])
    internal_day_col = find_column(internal_usage_df, ["day"])
    internal_month_col = find_column(internal_usage_df, ["month"])

    mode = scope["mode"]
    week_labels = scope["week_labels"]
    day_labels = scope["day_labels"]
    month_labels = scope["month_labels"]

    if mode == "days":
        products_filtered = products_usage_df[
            products_usage_df[products_week_col].isin(week_labels) &
            products_usage_df[products_day_col].isin(day_labels)
        ].copy()

        internal_filtered = internal_usage_df[
            internal_usage_df[internal_week_col].isin(week_labels) &
            internal_usage_df[internal_day_col].isin(day_labels)
        ].copy()

        return products_filtered, internal_filtered

    if mode == "weeks":
        products_filtered = products_usage_df[
            products_usage_df[products_week_col].isin(week_labels)
        ].copy()

        internal_filtered = internal_usage_df[
            internal_usage_df[internal_week_col].isin(week_labels)
        ].copy()

        return products_filtered, internal_filtered

    if mode == "months":
        products_filtered = products_usage_df[
            products_usage_df[products_month_col].isin(month_labels)
        ].copy()

        internal_filtered = internal_usage_df[
            internal_usage_df[internal_month_col].isin(month_labels)
        ].copy()

        return products_filtered, internal_filtered

    raise ValueError(f"Unsupported mode: {mode}")


# =========================
# FILTER BUDGET TABLES
# =========================
def filter_budget_table(
    daily_budget_df: pd.DataFrame,
    weekly_budget_df: pd.DataFrame,
    monthly_budget_df: pd.DataFrame,
    scope: dict
) -> pd.DataFrame:
    """
    Select the correct budget table and apply the correct filter.
    """
    daily_budget_df = normalize_string_columns(daily_budget_df)
    weekly_budget_df = normalize_string_columns(weekly_budget_df)
    monthly_budget_df = normalize_string_columns(monthly_budget_df)

    budget_level = scope["budget_level"]
    week_labels = scope["week_labels"]
    day_labels = scope["day_labels"]
    month_labels = scope["month_labels"]

    if budget_level == "daily":
        week_col = find_column(daily_budget_df, ["week"])
        day_col = find_column(daily_budget_df, ["day"])

        budget_filtered = daily_budget_df[
            daily_budget_df[week_col].isin(week_labels) &
            daily_budget_df[day_col].isin(day_labels)
        ].copy()

        return budget_filtered

    if budget_level == "weekly":
        week_col = find_column(weekly_budget_df, ["week"])

        budget_filtered = weekly_budget_df[
            weekly_budget_df[week_col].isin(week_labels)
        ].copy()

        return budget_filtered

    if budget_level == "monthly":
        month_col = find_column(monthly_budget_df, ["month"])

        budget_filtered = monthly_budget_df[
            monthly_budget_df[month_col].isin(month_labels)
        ].copy()

        return budget_filtered

    raise ValueError(f"Unsupported budget_level: {budget_level}")


# =========================
# MAIN ORCHESTRATION
# =========================
def get_analysis_data(selection: dict) -> dict:
    """
    Main function:
    1. resolve scope
    2. load CSVs
    3. filter usage
    4. filter budgets
    """

    scope = resolve_analysis_scope(selection)

    products_usage_df = load_csv_auto(PRODUCTS_USAGE_FILE)
    internal_usage_df = load_csv_auto(INTERNAL_USAGE_FILE)

    products_daily_budget_df = load_csv_auto(PRODUCTS_DAILY_BUDGET_FILE)
    products_weekly_budget_df = load_csv_auto(PRODUCTS_WEEKLY_BUDGET_FILE)
    products_monthly_budget_df = load_csv_auto(PRODUCTS_MONTHLY_BUDGET_FILE)

    internal_daily_budget_df = load_csv_auto(INTERNAL_DAILY_BUDGET_FILE)
    internal_weekly_budget_df = load_csv_auto(INTERNAL_WEEKLY_BUDGET_FILE)
    internal_monthly_budget_df = load_csv_auto(INTERNAL_MONTHLY_BUDGET_FILE)

    products_usage_filtered, internal_usage_filtered = filter_usage_tables(
        products_usage_df,
        internal_usage_df,
        scope
    )

    products_budget_filtered = filter_budget_table(
        products_daily_budget_df,
        products_weekly_budget_df,
        products_monthly_budget_df,
        scope
    )

    internal_budget_filtered = filter_budget_table(
        internal_daily_budget_df,
        internal_weekly_budget_df,
        internal_monthly_budget_df,
        scope
    )

    return {
        "scope": scope,
        "products_usage_filtered": products_usage_filtered,
        "internal_usage_filtered": internal_usage_filtered,
        "products_budget_filtered": products_budget_filtered,
        "internal_budget_filtered": internal_budget_filtered,
    }


# =========================
# EXAMPLE
# =========================
if __name__ == "__main__":
    selection_example = {
        "mode": "days",
        "block": "A",
        "week": 2,
        "days": 3,
        "period": "Monday to Wednesday"
    }

    result = get_analysis_data(selection_example)

    print("=== RESOLVED SCOPE ===")
    print(result["scope"])

    print("\n=== PRODUCTS USAGE FILTERED ===")
    print(result["products_usage_filtered"].head())

    print("\n=== INTERNAL USAGE FILTERED ===")
    print(result["internal_usage_filtered"].head())

    print("\n=== PRODUCTS BUDGET FILTERED ===")
    print(result["products_budget_filtered"].head())

    print("\n=== INTERNAL BUDGET FILTERED ===")
    print(result["internal_budget_filtered"].head())