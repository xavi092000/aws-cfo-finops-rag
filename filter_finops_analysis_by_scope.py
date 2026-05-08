from pathlib import Path
import pandas as pd

from resolve_finops_analysis_scope import resolve_analysis_scope


# =========================
# PATHS
# =========================
PROJECT_DIR = Path(".")
DATA_DIR = Path("data")

PRODUCTS_ANALYSIS_CANDIDATES = [
    PROJECT_DIR / "products_finops_analysis.csv",
    DATA_DIR / "products_finops_analysis.csv",
]

INTERNAL_ANALYSIS_CANDIDATES = [
    PROJECT_DIR / "internal_finops_analysis.csv",
    DATA_DIR / "internal_finops_analysis.csv",
]


# =========================
# HELPERS
# =========================
def resolve_existing_file(candidates: list[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find {label}. Checked: {[str(p) for p in candidates]}"
    )


def load_csv_auto(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(file_path)
        return df
    except Exception:
        return pd.read_csv(file_path)


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


# =========================
# LOAD DATA
# =========================
def load_analysis_tables():
    products_file = resolve_existing_file(
        PRODUCTS_ANALYSIS_CANDIDATES,
        "products_finops_analysis.csv"
    )
    internal_file = resolve_existing_file(
        INTERNAL_ANALYSIS_CANDIDATES,
        "internal_finops_analysis.csv"
    )

    print(f"Using products analysis file: {products_file}")
    print(f"Using internal analysis file: {internal_file}")

    products_df = load_csv_auto(products_file)
    internal_df = load_csv_auto(internal_file)

    products_df = normalize_string_columns(normalize_columns(products_df))
    internal_df = normalize_string_columns(normalize_columns(internal_df))

    return products_df, internal_df


# =========================
# FILTER
# =========================
def filter_by_scope(df: pd.DataFrame, scope: dict) -> pd.DataFrame:
    mode = scope["mode"]
    week_labels = scope["week_labels"]
    day_labels = scope["day_labels"]
    month_labels = scope["month_labels"]

    week_col = "week" if "week" in df.columns else "Week"
    day_col = "day" if "day" in df.columns else "Day"
    month_col = "month" if "month" in df.columns else "Month"

    if mode == "days":
        filtered = df[
            df[week_col].isin(week_labels) &
            df[day_col].isin(day_labels)
        ].copy()
        return filtered

    if mode == "weeks":
        filtered = df[
            df[week_col].isin(week_labels)
        ].copy()
        return filtered

    if mode == "months":
        filtered = df[
            df[month_col].isin(month_labels)
        ].copy()
        return filtered

    raise ValueError(f"Unsupported mode: {mode}")


# =========================
# MAIN
# =========================
def main():
    products_df, internal_df = load_analysis_tables()

    # Example selection from the agent
    selection = {
        "mode": "days",
        "block": "A",
        "week": 1,
        "days": 2,
        "period": "Monday to Tuesday"
    }

    scope = resolve_analysis_scope(selection)

    print("\n=== RESOLVED SCOPE ===")
    print(scope)

    products_filtered = filter_by_scope(products_df, scope)
    internal_filtered = filter_by_scope(internal_df, scope)

    print("\n=== PRODUCTS FINOPS FILTERED ===")
    print(products_filtered.head(10))
    print(f"Rows matched: {len(products_filtered)}")

    print("\n=== INTERNAL FINOPS FILTERED ===")
    print(internal_filtered.head(10))
    print(f"Rows matched: {len(internal_filtered)}")


if __name__ == "__main__":
    main()