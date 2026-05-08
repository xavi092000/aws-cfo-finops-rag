from pathlib import Path
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
DATA_DIR = Path("data")

FILES_TO_FIX = [
    DATA_DIR / "products_usage_84_days.csv",
    DATA_DIR / "internal_usage_84_days.csv",
    DATA_DIR / "Products_Daily_Budget.csv",
    DATA_DIR / "Internal_Daily_Budget.csv",
]

EXCLUDED_DATES = {
    pd.Timestamp("2025-03-24"),
    pd.Timestamp("2025-03-25"),
}

WEEK1_START = pd.Timestamp("2025-01-01")
WEEK1_END = pd.Timestamp("2025-01-05")
WEEK2_START = pd.Timestamp("2025-01-06")
FINAL_INCLUDED_DATE = pd.Timestamp("2025-03-23")


# =========================================================
# CSV HELPERS
# =========================================================
def load_csv_auto(file_path: Path) -> tuple[pd.DataFrame, str]:
    """
    Try semicolon first, then comma.
    Return dataframe + detected separator for safe overwrite.
    """
    try:
        df = pd.read_csv(file_path, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(file_path, sep=",")
            return df, ","
        return df, ";"
    except Exception:
        df = pd.read_csv(file_path)
        return df, ","


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
    return df


def find_column(df: pd.DataFrame, possible_names: list[str]) -> str:
    normalized_targets = [name.lower() for name in possible_names]
    for col in df.columns:
        if col.lower() in normalized_targets:
            return col
    raise ValueError(
        f"Column not found. Expected one of {possible_names}. Found columns: {list(df.columns)}"
    )


# =========================================================
# BUSINESS TIME LOGIC
# =========================================================
def compute_business_week(ts: pd.Timestamp):
    """
    Business rule:
    - Week 1 = 2025-01-01 to 2025-01-05
    - Week 2 to Week 12 = Monday to Sunday from 2025-01-06
    - 2025-03-24 and 2025-03-25 are excluded from analysis
    """
    if pd.isna(ts):
        return pd.NA

    if ts in EXCLUDED_DATES:
        return pd.NA

    if ts < WEEK1_START or ts > FINAL_INCLUDED_DATE:
        return pd.NA

    if WEEK1_START <= ts <= WEEK1_END:
        return "Week 1"

    delta_days = (ts - WEEK2_START).days
    week_num = 2 + (delta_days // 7)

    if 2 <= week_num <= 12:
        return f"Week {week_num}"

    return pd.NA


def compute_business_month(week_label):
    if pd.isna(week_label):
        return pd.NA

    week_label_str = str(week_label).strip()
    if not week_label_str or week_label_str.lower() == "nan":
        return pd.NA

    week_num = int(week_label_str.replace("Week", "").strip())

    if 1 <= week_num <= 4:
        return "Month 1"
    if 5 <= week_num <= 8:
        return "Month 2"
    if 9 <= week_num <= 12:
        return "Month 3"

    return pd.NA


def compute_day_name(ts: pd.Timestamp):
    if pd.isna(ts):
        return pd.NA
    return ts.day_name()


# =========================================================
# CORE FIXER
# =========================================================
def fix_file(file_path: Path) -> None:
    print("\n==================================================")
    print(f"PROCESSING: {file_path.name}")
    print("==================================================")

    df, sep = load_csv_auto(file_path)
    df = normalize_string_columns(df)

    original_rows = len(df)

    date_col = find_column(df, ["date"])
    day_col = find_column(df, ["day"])
    week_col = find_column(df, ["week"])
    month_col = find_column(df, ["month"])

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if df[date_col].isna().any():
        bad_count = int(df[date_col].isna().sum())
        raise ValueError(
            f"{file_path.name}: {bad_count} rows have invalid dates and cannot be fixed safely."
        )

    # Recompute business calendar
    df["_new_week"] = df[date_col].apply(compute_business_week)
    df["_new_month"] = df["_new_week"].apply(compute_business_month)
    df["_new_day"] = df[date_col].apply(compute_day_name)

    # Exclude rows outside the analysis scope
    excluded_rows = int(df["_new_week"].isna().sum())
    df = df[df["_new_week"].notna()].copy()

    # Apply corrected columns
    df[day_col] = df["_new_day"]
    df[week_col] = df["_new_week"]
    df[month_col] = df["_new_month"]

    # Restore YYYY-MM-DD string format for compatibility
    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

    # Cleanup temp columns
    df = df.drop(columns=["_new_week", "_new_month", "_new_day"])

    # Save backup first
    backup_path = file_path.with_name(file_path.stem + "_backup_before_time_fix" + file_path.suffix)
    if not backup_path.exists():
        original_df, _ = load_csv_auto(file_path)
        original_df.to_csv(backup_path, index=False, sep=sep)
        print(f"Backup created: {backup_path.name}")
    else:
        print(f"Backup already exists: {backup_path.name}")

    # Overwrite original file
    df.to_csv(file_path, index=False, sep=sep)

    print(f"Original rows: {original_rows}")
    print(f"Excluded rows: {excluded_rows}")
    print(f"Remaining rows: {len(df)}")

    print("\nSample of corrected dates:")
    preview_cols = [date_col, day_col, week_col, month_col]
    print(df[preview_cols].head(10).to_string(index=False))

    print("\nLast corrected rows:")
    print(df[preview_cols].tail(10).to_string(index=False))


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    print("Starting correction of business time logic on 4 CSV files...")

    for file_path in FILES_TO_FIX:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")
        fix_file(file_path)

    print("\n==================================================")
    print("DONE")
    print("==================================================")
    print("The 4 CSV files were corrected with this logic:")
    print("- Week 1 = 2025-01-01 to 2025-01-05")
    print("- Week 2 to Week 12 = Monday to Sunday")
    print("- 2025-03-24 and 2025-03-25 excluded from analysis")


if __name__ == "__main__":
    main()