from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List
import os

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE", "terraform-runner")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

DATA_DIR = Path("data")

PRODUCTS_USAGE_FILE = DATA_DIR / "products_usage_84_days.csv"
INTERNAL_USAGE_FILE = DATA_DIR / "internal_usage_84_days.csv"

PRODUCTS_DAILY_BUDGET_FILE = DATA_DIR / "Products_Daily_Budget.csv"
PRODUCTS_WEEKLY_BUDGET_FILE = DATA_DIR / "Products_Weekly_Budgets.csv"
PRODUCTS_MONTHLY_BUDGET_FILE = DATA_DIR / "Products_Monthly_Budget.csv"

INTERNAL_DAILY_BUDGET_FILE = DATA_DIR / "Internal_Daily_Budget.csv"
INTERNAL_WEEKLY_BUDGET_FILE = DATA_DIR / "Internal_Weekly_Budget.csv"
INTERNAL_MONTHLY_BUDGET_FILE = DATA_DIR / "Internal_Monthly_Budget.csv"

PRICING_FILE = DATA_DIR / "aws_pricing.csv"

DAY_LABELS_MAP = {
    1: ["Monday"],
    2: ["Monday", "Tuesday"],
    3: ["Monday", "Tuesday", "Wednesday"],
    4: ["Monday", "Tuesday", "Wednesday", "Thursday"],
    5: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    6: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
}

WEEK_1_DAY_LABELS_MAP = {
    1: ["Wednesday"],
    2: ["Wednesday", "Thursday"],
    3: ["Wednesday", "Thursday", "Friday"],
    4: ["Wednesday", "Thursday", "Friday", "Saturday"],
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
# AWS PROFILE
# =========================
def force_aws_profile() -> None:
    for key in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_PROFILE",
    ]:
        if key in os.environ:
            del os.environ[key]

    os.environ["AWS_PROFILE"] = AWS_PROFILE
    os.environ["AWS_DEFAULT_REGION"] = AWS_REGION

    boto3.setup_default_session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


# =========================
# CSV LOADING
# =========================
def load_csv_auto(file_path: Path) -> pd.DataFrame:
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
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
    return df


def clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("\u00A0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
    )

    us_mask = cleaned.str.contains(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$", regex=True, na=False)
    cleaned.loc[us_mask] = cleaned.loc[us_mask].str.replace(",", "", regex=False)

    eu_decimal_mask = cleaned.str.contains(r"^-?\d+,\d+$", regex=True, na=False)
    cleaned.loc[eu_decimal_mask] = cleaned.loc[eu_decimal_mask].str.replace(",", ".", regex=False)

    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "none": np.nan, "None": np.nan})

    return pd.to_numeric(cleaned, errors="coerce")


def find_column(df: pd.DataFrame, possible_names: List[str]) -> str:
    candidates = [name.lower() for name in possible_names]
    for col in df.columns:
        if col.lower() in candidates:
            return col
    raise ValueError(
        f"Column not found. Expected one of {possible_names}. Found columns: {list(df.columns)}"
    )


def find_optional_column(df: pd.DataFrame, possible_names: List[str]) -> str | None:
    candidates = [name.lower() for name in possible_names]
    for col in df.columns:
        if col.lower() in candidates:
            return col
    return None


def safe_variance_pct(actual: float, budget: float) -> float:
    if pd.isna(actual) or pd.isna(budget) or budget == 0:
        return 0.0
    return ((actual - budget) / budget) * 100.0


def normalize_service_name(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
    )


def normalize_usage_unit(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("hour", "hours", regex=False)
        .str.replace("request", "requests", regex=False)
        .str.replace("token", "tokens", regex=False)
        .str.replace("dpu_hour", "dpu_hours", regex=False)
    )


# =========================
# PRICING
# =========================
def load_pricing_table() -> pd.DataFrame:
    pricing_df = load_csv_auto(PRICING_FILE)
    pricing_df = normalize_string_columns(normalize_columns(pricing_df)).copy()

    service_col = find_column(pricing_df, ["service"])
    usage_unit_col = find_column(pricing_df, ["usage_unit"])
    unit_price_col = find_column(pricing_df, ["unit_price_usd"])

    pricing_df["service_norm"] = normalize_service_name(pricing_df[service_col])
    pricing_df["usage_unit_norm"] = normalize_usage_unit(pricing_df[usage_unit_col])
    pricing_df["unit_price_usd"] = clean_numeric(pricing_df[unit_price_col]).fillna(0.0)

    pricing_df = pricing_df[["service_norm", "usage_unit_norm", "unit_price_usd"]].drop_duplicates()

    return pricing_df


# =========================
# SCOPE RESOLUTION
# =========================
def resolve_analysis_scope(selection: dict) -> dict:
    mode = selection["mode"]

    if mode == "days":
        week_number = selection["week"]
        week_label = f"Week {week_number}"

        if "day_labels" in selection and selection["day_labels"]:
            day_labels = selection["day_labels"]
        else:
            if week_number == 1:
                day_labels = WEEK_1_DAY_LABELS_MAP[selection["days"]]
            else:
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
# FILTER HELPERS
# =========================
def filter_usage_tables(
    products_usage_df: pd.DataFrame,
    internal_usage_df: pd.DataFrame,
    scope: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    products_usage_df = normalize_string_columns(normalize_columns(products_usage_df))
    internal_usage_df = normalize_string_columns(normalize_columns(internal_usage_df))

    products_week_col = find_optional_column(products_usage_df, ["week"])
    products_day_col = find_optional_column(products_usage_df, ["day"])
    products_month_col = find_optional_column(products_usage_df, ["month"])

    internal_week_col = find_optional_column(internal_usage_df, ["week"])
    internal_day_col = find_optional_column(internal_usage_df, ["day"])
    internal_month_col = find_optional_column(internal_usage_df, ["month"])

    mode = scope["mode"]
    week_labels = scope["week_labels"]
    day_labels = scope["day_labels"]
    month_labels = scope["month_labels"]

    if mode == "days":
        products_mask = pd.Series(True, index=products_usage_df.index)
        internal_mask = pd.Series(True, index=internal_usage_df.index)

        if products_week_col:
            products_mask &= products_usage_df[products_week_col].isin(week_labels)
        if internal_week_col:
            internal_mask &= internal_usage_df[internal_week_col].isin(week_labels)

        if products_day_col:
            products_mask &= products_usage_df[products_day_col].isin(day_labels)
        if internal_day_col:
            internal_mask &= internal_usage_df[internal_day_col].isin(day_labels)

        return products_usage_df[products_mask].copy(), internal_usage_df[internal_mask].copy()

    if mode == "weeks":
        products_mask = pd.Series(True, index=products_usage_df.index)
        internal_mask = pd.Series(True, index=internal_usage_df.index)

        if products_week_col:
            products_mask &= products_usage_df[products_week_col].isin(week_labels)
        if internal_week_col:
            internal_mask &= internal_usage_df[internal_week_col].isin(week_labels)

        return products_usage_df[products_mask].copy(), internal_usage_df[internal_mask].copy()

    if mode == "months":
        products_mask = pd.Series(True, index=products_usage_df.index)
        internal_mask = pd.Series(True, index=internal_usage_df.index)

        if products_month_col:
            products_mask &= products_usage_df[products_month_col].isin(month_labels)
        if internal_month_col:
            internal_mask &= internal_usage_df[internal_month_col].isin(month_labels)

        return products_usage_df[products_mask].copy(), internal_usage_df[internal_mask].copy()

    raise ValueError(f"Unsupported mode: {mode}")


def filter_budget_table(
    daily_budget_df: pd.DataFrame,
    weekly_budget_df: pd.DataFrame,
    monthly_budget_df: pd.DataFrame,
    scope: dict
) -> pd.DataFrame:
    daily_budget_df = normalize_string_columns(normalize_columns(daily_budget_df))
    weekly_budget_df = normalize_string_columns(normalize_columns(weekly_budget_df))
    monthly_budget_df = normalize_string_columns(normalize_columns(monthly_budget_df))

    budget_level = scope["budget_level"]
    week_labels = scope["week_labels"]
    day_labels = scope["day_labels"]
    month_labels = scope["month_labels"]

    if budget_level == "daily":
        week_col = find_optional_column(daily_budget_df, ["week"])
        day_col = find_optional_column(daily_budget_df, ["day"])

        mask = pd.Series(True, index=daily_budget_df.index)
        if week_col:
            mask &= daily_budget_df[week_col].isin(week_labels)
        if day_col:
            mask &= daily_budget_df[day_col].isin(day_labels)

        return daily_budget_df[mask].copy()

    if budget_level == "weekly":
        week_col = find_optional_column(weekly_budget_df, ["week"])

        mask = pd.Series(True, index=weekly_budget_df.index)
        if week_col:
            mask &= weekly_budget_df[week_col].isin(week_labels)

        return weekly_budget_df[mask].copy()

    if budget_level == "monthly":
        month_col = find_optional_column(monthly_budget_df, ["month"])

        mask = pd.Series(True, index=monthly_budget_df.index)
        if month_col:
            mask &= monthly_budget_df[month_col].isin(month_labels)

        return monthly_budget_df[mask].copy()

    raise ValueError(f"Unsupported budget_level: {budget_level}")


# =========================
# COST / GRAIN HELPERS
# =========================
def ensure_actual_cost_column(
    df: pd.DataFrame,
    dataset_name: str,
    pricing_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Supported cases:
    1) actual_cost_usd / actual_cost / cost / cost_usd already exists
    2) usage_value * unit_price
    """
    df = normalize_string_columns(normalize_columns(df)).copy()

    actual_col = find_optional_column(
        df,
        ["actual_cost_usd", "actual_cost", "cost", "cost_usd"]
    )

    if actual_col is not None:
        df["actual_cost_usd"] = clean_numeric(df[actual_col]).fillna(0.0)
        return df

    usage_col = find_optional_column(df, ["usage_value", "usage", "quantity", "amount"])
    service_col = find_optional_column(df, ["service", "service_name", "aws_service", "cloud_service"])
    usage_unit_col = find_optional_column(df, ["usage_unit"])

    if usage_col is None or service_col is None or usage_unit_col is None:
        raise ValueError(
            f"Could not derive actual cost for {dataset_name}. "
            f"Expected usage_value + service + usage_unit, or an existing actual cost column. "
            f"Found columns: {list(df.columns)}"
        )

    df["usage_value_num"] = clean_numeric(df[usage_col]).fillna(0.0)
    df["service_norm"] = normalize_service_name(df[service_col])
    df["usage_unit_norm"] = normalize_usage_unit(df[usage_unit_col])

    merged = df.merge(
        pricing_df,
        on=["service_norm", "usage_unit_norm"],
        how="left"
    )

    missing_price = merged["unit_price_usd"].isna()
    if missing_price.any():
        sample_missing = (
            merged.loc[missing_price, ["service_norm", "usage_unit_norm"]]
            .drop_duplicates()
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"Missing pricing rows for {dataset_name}. "
            f"Examples: {sample_missing}"
        )

    merged["actual_cost_usd"] = merged["usage_value_num"] * merged["unit_price_usd"]
    return merged


def normalize_budget_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_string_columns(normalize_columns(df)).copy()

    actual_col = find_optional_column(df, ["actual_cost_usd", "actual_cost", "actual cost ($)", "actual cost"])
    budget_col = find_optional_column(df, ["allocated_budget_usd", "allocated_budget", "allocated budget ($)", "allocated budget"])
    variance_col = find_optional_column(df, ["variance_usd", "variance ($)", "variance"])
    variance_pct_col = find_optional_column(df, ["variance_pct", "variance percent", "variance %", "variance_percent"])

    if actual_col is not None:
        df["actual_cost_usd"] = clean_numeric(df[actual_col]).fillna(0.0)
    else:
        df["actual_cost_usd"] = 0.0

    if budget_col is not None:
        df["allocated_budget_usd"] = clean_numeric(df[budget_col]).fillna(0.0)
    else:
        df["allocated_budget_usd"] = 0.0

    if variance_col is not None:
        df["variance_usd"] = clean_numeric(df[variance_col]).fillna(0.0)
    else:
        df["variance_usd"] = df["actual_cost_usd"] - df["allocated_budget_usd"]

    if variance_pct_col is not None:
        df["variance_pct"] = clean_numeric(df[variance_pct_col]).fillna(0.0)
    else:
        df["variance_pct"] = df.apply(
            lambda row: safe_variance_pct(row["actual_cost_usd"], row["allocated_budget_usd"]),
            axis=1,
        )

    return df


def get_common_period_columns(scope: dict, df: pd.DataFrame) -> List[str]:
    mode = scope["mode"]
    cols = [str(c).lower() for c in df.columns]

    if mode == "days":
        desired = ["date", "day", "week", "month"]
    elif mode == "weeks":
        desired = ["week", "month"]
    else:
        desired = ["month"]

    return [c for c in desired if c in cols]


def build_products_granular_frame(
    products_usage_filtered: pd.DataFrame,
    products_budget_filtered: pd.DataFrame,
    scope: dict,
    pricing_df: pd.DataFrame
) -> pd.DataFrame:
    usage = ensure_actual_cost_column(products_usage_filtered, "products usage", pricing_df)
    budget = normalize_budget_df(products_budget_filtered)

    usage = normalize_string_columns(normalize_columns(usage))
    budget = normalize_string_columns(normalize_columns(budget))

    product_col = find_column(usage, ["product", "product_name"])
    division_col = find_column(usage, ["division", "division_name"])
    service_col = find_optional_column(usage, ["service", "service_name", "aws_service", "cloud_service"])

    period_cols_usage = get_common_period_columns(scope, usage)
    period_cols_budget = get_common_period_columns(scope, budget)

    group_cols_usage = period_cols_usage + [product_col, division_col]
    if service_col is not None:
        group_cols_usage.append(service_col)

    usage_grouped = (
        usage.groupby(group_cols_usage, dropna=False)[["actual_cost_usd"]]
        .sum()
        .reset_index()
    )

    budget_product_col = find_optional_column(budget, ["product", "product_name"])
    budget_division_col = find_optional_column(budget, ["division", "division_name"])

    if budget_product_col is None or budget_division_col is None:
        raise ValueError(
            "Products budget table must contain product and division columns. "
            f"Found columns: {list(budget.columns)}"
        )

    budget_group_cols = period_cols_budget + [budget_product_col, budget_division_col]
    budget_grouped = (
        budget.groupby(budget_group_cols, dropna=False)[["allocated_budget_usd", "actual_cost_usd", "variance_usd", "variance_pct"]]
        .sum()
        .reset_index()
    )

    rename_map_usage = {
        product_col: "product",
        division_col: "division",
    }
    if service_col is not None:
        rename_map_usage[service_col] = "service"

    usage_grouped = usage_grouped.rename(columns=rename_map_usage)
    budget_grouped = budget_grouped.rename(
        columns={
            budget_product_col: "product",
            budget_division_col: "division",
        }
    )

    merge_keys = [
        c for c in ["date", "day", "week", "month", "product", "division"]
        if c in usage_grouped.columns and c in budget_grouped.columns
    ]

    # print("MERGE KEYS PRODUCTS:", merge_keys)
    # print("USAGE GROUPED SAMPLE:")
    # print(usage_grouped[merge_keys].head().to_string(index=False))
    # print("BUDGET GROUPED SAMPLE:")
    # print(budget_grouped[merge_keys + ["allocated_budget_usd"]].head().to_string(index=False))

    merged = usage_grouped.merge(
        budget_grouped,
        on=merge_keys,
        how="left",
        suffixes=("", "_budget")
    )

    if "service" not in merged.columns:
        merged["service"] = "DATA_NOT_AVAILABLE"

    allocation_group_cols = [c for c in ["date", "day", "week", "month", "product", "division"] if c in merged.columns]

    if allocation_group_cols:
        merged["group_actual_total"] = merged.groupby(allocation_group_cols)["actual_cost_usd"].transform("sum")
        merged["group_service_count"] = merged.groupby(allocation_group_cols)["service"].transform("count")
    else:
        merged["group_actual_total"] = merged["actual_cost_usd"].sum()
        merged["group_service_count"] = len(merged)

    merged["allocated_budget_usd"] = merged["allocated_budget_usd"].fillna(0.0)

    merged["allocated_budget_usd"] = np.where(
        merged["group_actual_total"] > 0,
        merged["allocated_budget_usd"] * (merged["actual_cost_usd"] / merged["group_actual_total"]),
        np.where(
            merged["group_service_count"] > 0,
            merged["allocated_budget_usd"] / merged["group_service_count"],
            0.0,
        ),
    )

    merged["variance_usd"] = merged["actual_cost_usd"] - merged["allocated_budget_usd"]
    merged["variance_pct"] = merged.apply(
        lambda row: safe_variance_pct(row["actual_cost_usd"], row["allocated_budget_usd"]),
        axis=1,
    )

    base_cols = [c for c in ["date", "day", "week", "month"] if c in merged.columns]
    final_cols = base_cols + ["product", "division", "service", "actual_cost_usd", "allocated_budget_usd", "variance_usd", "variance_pct"]

    return merged[final_cols].copy()


def build_internal_granular_frame(
    internal_usage_filtered: pd.DataFrame,
    internal_budget_filtered: pd.DataFrame,
    scope: dict,
    pricing_df: pd.DataFrame
) -> pd.DataFrame:
    usage = ensure_actual_cost_column(internal_usage_filtered, "internal usage", pricing_df)
    budget = normalize_budget_df(internal_budget_filtered)

    usage = normalize_string_columns(normalize_columns(usage))
    budget = normalize_string_columns(normalize_columns(budget))

    division_col = find_column(usage, ["division", "division_name", "internal_division"])
    service_col = find_optional_column(usage, ["service", "service_name", "aws_service", "cloud_service"])

    period_cols_usage = get_common_period_columns(scope, usage)
    period_cols_budget = get_common_period_columns(scope, budget)

    group_cols_usage = period_cols_usage + [division_col]
    if service_col is not None:
        group_cols_usage.append(service_col)

    usage_grouped = (
        usage.groupby(group_cols_usage, dropna=False)[["actual_cost_usd"]]
        .sum()
        .reset_index()
    )

    budget_division_col = find_optional_column(budget, ["division", "division_name", "internal_division"])
    if budget_division_col is None:
        raise ValueError(
            "Internal budget table must contain a division column. "
            f"Found columns: {list(budget.columns)}"
        )

    budget_group_cols = period_cols_budget + [budget_division_col]
    budget_grouped = (
        budget.groupby(budget_group_cols, dropna=False)[["allocated_budget_usd", "actual_cost_usd", "variance_usd", "variance_pct"]]
        .sum()
        .reset_index()
    )

    rename_map_usage = {division_col: "division"}
    if service_col is not None:
        rename_map_usage[service_col] = "service"

    usage_grouped = usage_grouped.rename(columns=rename_map_usage)
    budget_grouped = budget_grouped.rename(columns={budget_division_col: "division"})

    merge_keys = [
        c for c in ["date", "day", "week", "month", "division"]
        if c in usage_grouped.columns and c in budget_grouped.columns
    ]

    merged = usage_grouped.merge(
        budget_grouped,
        on=merge_keys,
        how="left",
        suffixes=("", "_budget")
    )

    if "service" not in merged.columns:
        merged["service"] = "DATA_NOT_AVAILABLE"

    allocation_group_cols = [c for c in ["date", "day", "week", "month", "division"] if c in merged.columns]

    if allocation_group_cols:
        merged["group_actual_total"] = merged.groupby(allocation_group_cols)["actual_cost_usd"].transform("sum")
        merged["group_service_count"] = merged.groupby(allocation_group_cols)["service"].transform("count")
    else:
        merged["group_actual_total"] = merged["actual_cost_usd"].sum()
        merged["group_service_count"] = len(merged)

    merged["allocated_budget_usd"] = merged["allocated_budget_usd"].fillna(0.0)

    merged["allocated_budget_usd"] = np.where(
        merged["group_actual_total"] > 0,
        merged["allocated_budget_usd"] * (merged["actual_cost_usd"] / merged["group_actual_total"]),
        np.where(
            merged["group_service_count"] > 0,
            merged["allocated_budget_usd"] / merged["group_service_count"],
            0.0,
        ),
    )

    merged["variance_usd"] = merged["actual_cost_usd"] - merged["allocated_budget_usd"]
    merged["variance_pct"] = merged.apply(
        lambda row: safe_variance_pct(row["actual_cost_usd"], row["allocated_budget_usd"]),
        axis=1,
    )

    base_cols = [c for c in ["date", "day", "week", "month"] if c in merged.columns]
    final_cols = base_cols + ["division", "service", "actual_cost_usd", "allocated_budget_usd", "variance_usd", "variance_pct"]

    return merged[final_cols].copy()


def build_products_daily_spend_frame(
    products_usage_filtered: pd.DataFrame,
    pricing_df: pd.DataFrame
) -> pd.DataFrame:
    usage = ensure_actual_cost_column(products_usage_filtered, "products usage", pricing_df)
    usage = normalize_string_columns(normalize_columns(usage))

    product_col = find_optional_column(usage, ["product", "product_name"])
    division_col = find_optional_column(usage, ["division", "division_name"])
    service_col = find_optional_column(usage, ["service", "service_name", "aws_service", "cloud_service"])

    period_cols = [c for c in ["date", "day", "week", "month"] if c in usage.columns]
    group_cols = period_cols.copy()

    if product_col:
        group_cols.append(product_col)
    if division_col:
        group_cols.append(division_col)
    if service_col:
        group_cols.append(service_col)

    daily = (
        usage.groupby(group_cols, dropna=False)[["actual_cost_usd"]]
        .sum()
        .reset_index()
    )

    rename_map = {}
    if product_col:
        rename_map[product_col] = "product"
    if division_col:
        rename_map[division_col] = "division"
    if service_col:
        rename_map[service_col] = "service"

    daily = daily.rename(columns=rename_map)

    if "product" not in daily.columns:
        daily["product"] = "DATA_NOT_AVAILABLE"
    if "division" not in daily.columns:
        daily["division"] = "DATA_NOT_AVAILABLE"
    if "service" not in daily.columns:
        daily["service"] = "DATA_NOT_AVAILABLE"

    return daily


def build_internal_daily_spend_frame(
    internal_usage_filtered: pd.DataFrame,
    pricing_df: pd.DataFrame
) -> pd.DataFrame:
    usage = ensure_actual_cost_column(internal_usage_filtered, "internal usage", pricing_df)
    usage = normalize_string_columns(normalize_columns(usage))

    division_col = find_optional_column(usage, ["division", "division_name", "internal_division"])
    service_col = find_optional_column(usage, ["service", "service_name", "aws_service", "cloud_service"])

    period_cols = [c for c in ["date", "day", "week", "month"] if c in usage.columns]
    group_cols = period_cols.copy()

    if division_col:
        group_cols.append(division_col)
    if service_col:
        group_cols.append(service_col)

    daily = (
        usage.groupby(group_cols, dropna=False)[["actual_cost_usd"]]
        .sum()
        .reset_index()
    )

    rename_map = {}
    if division_col:
        rename_map[division_col] = "division"
    if service_col:
        rename_map[service_col] = "service"

    daily = daily.rename(columns=rename_map)

    daily["product"] = "DATA_NOT_AVAILABLE"
    if "division" not in daily.columns:
        daily["division"] = "DATA_NOT_AVAILABLE"
    if "service" not in daily.columns:
        daily["service"] = "DATA_NOT_AVAILABLE"

    return daily


# =========================
# MAIN ORCHESTRATION
# =========================
def load_selected_period_data(selection: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scope = resolve_analysis_scope(selection)

    pricing_df = load_pricing_table()

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

    division_df = build_internal_granular_frame(
        internal_usage_filtered=internal_usage_filtered,
        internal_budget_filtered=internal_budget_filtered,
        scope=scope,
        pricing_df=pricing_df,
    )

    product_df = build_products_granular_frame(
        products_usage_filtered=products_usage_filtered,
        products_budget_filtered=products_budget_filtered,
        scope=scope,
        pricing_df=pricing_df,
    )

    return division_df, product_df


def load_selected_period_daily_spend(selection: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scope = resolve_analysis_scope(selection)

    pricing_df = load_pricing_table()

    products_usage_df = load_csv_auto(PRODUCTS_USAGE_FILE)
    internal_usage_df = load_csv_auto(INTERNAL_USAGE_FILE)

    products_usage_filtered, internal_usage_filtered = filter_usage_tables(
        products_usage_df,
        internal_usage_df,
        scope
    )

    division_daily_df = build_internal_daily_spend_frame(
        internal_usage_filtered=internal_usage_filtered,
        pricing_df=pricing_df,
    )

    product_daily_df = build_products_daily_spend_frame(
        products_usage_filtered=products_usage_filtered,
        pricing_df=pricing_df,
    )

    return division_daily_df, product_daily_df


if __name__ == "__main__":
    force_aws_profile()

    selection = {
        "mode": "weeks",
        "block": "B",
        "weeks": [5, 6],
        "number_of_weeks": 2,
    }

    division_df, product_df = load_selected_period_data(selection)

    print("Division rows:", len(division_df))
    print("Product rows:", len(product_df))
    print("\nDivision preview:")
    print(division_df.head())
    print("\nProduct preview:")
    print(product_df.head())

    print("\n--- DAILY SPEND PREVIEW ---")
    division_daily_df, product_daily_df = load_selected_period_daily_spend(selection)
    print("Division daily rows:", len(division_daily_df))
    print("Product daily rows:", len(product_daily_df))
    print(division_daily_df.head())
    print(product_daily_df.head())