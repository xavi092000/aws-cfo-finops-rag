from __future__ import annotations

from io import StringIO
from typing import Dict, Tuple
import time

import boto3
import pandas as pd

from resolve_finops_analytics import (
    force_aws_profile,
    build_products_granular_frame,
    build_internal_granular_frame,
    resolve_analysis_scope,
    load_csv_auto,
    filter_usage_tables,
    filter_budget_table,
    PRODUCTS_USAGE_FILE,
    INTERNAL_USAGE_FILE,
    PRODUCTS_DAILY_BUDGET_FILE,
    PRODUCTS_WEEKLY_BUDGET_FILE,
    PRODUCTS_MONTHLY_BUDGET_FILE,
    INTERNAL_DAILY_BUDGET_FILE,
    INTERNAL_WEEKLY_BUDGET_FILE,
    INTERNAL_MONTHLY_BUDGET_FILE,
    load_pricing_table,
)

# =========================
# AWS / ATHENA CONFIG
# =========================
AWS_PROFILE = "terraform-runner"
AWS_REGION = "us-east-1"

ATHENA_DATABASE = "cfo_finops_db"
ATHENA_WORKGROUP = "primary"
ATHENA_RESULTS_S3 = "s3://kestra-zoomcamp-ny-taxi-mtl9999999999988888/athena-results/"

# Base S3 path where the daily mart CSV files will live
DAILY_MARTS_S3_BASE = "s3://kestra-zoomcamp-ny-taxi-mtl9999999999988888/powerbi_daily_marts/"

PRODUCT_DAILY_PREFIX = "product/"
DIVISION_DAILY_PREFIX = "division/"

PRODUCT_TABLE_NAME = "mart_product_cfo_daily"
DIVISION_TABLE_NAME = "mart_division_cfo_daily"


# =========================
# HELPERS
# =========================
def get_boto3_session() -> boto3.Session:
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    no_scheme = s3_uri.replace("s3://", "", 1)
    bucket, _, key = no_scheme.partition("/")
    return bucket, key


def execute_athena_query(sql: str) -> str:
    session = get_boto3_session()
    athena = session.client("athena", region_name=AWS_REGION)

    response = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DATABASE},
        ResultConfiguration={"OutputLocation": ATHENA_RESULTS_S3},
        WorkGroup=ATHENA_WORKGROUP,
    )

    query_execution_id = response["QueryExecutionId"]

    while True:
        status_response = athena.get_query_execution(QueryExecutionId=query_execution_id)
        state = status_response["QueryExecution"]["Status"]["State"]

        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            break

        time.sleep(1)

    if state != "SUCCEEDED":
        reason = status_response["QueryExecution"]["Status"].get(
            "StateChangeReason",
            "Unknown Athena error"
        )
        raise RuntimeError(f"Athena query failed with state {state}: {reason}")

    return query_execution_id


def delete_s3_prefix(s3_uri: str) -> None:
    session = get_boto3_session()
    s3 = session.client("s3", region_name=AWS_REGION)

    bucket, prefix = parse_s3_uri(s3_uri)

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    keys_to_delete = []
    for page in pages:
        for obj in page.get("Contents", []):
            keys_to_delete.append({"Key": obj["Key"]})

    if not keys_to_delete:
        return

    for i in range(0, len(keys_to_delete), 1000):
        batch = keys_to_delete[i:i + 1000]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})


def upload_df_as_csv_to_s3(df: pd.DataFrame, s3_uri: str, filename: str) -> str:
    session = get_boto3_session()
    s3 = session.client("s3", region_name=AWS_REGION)

    bucket, prefix = parse_s3_uri(s3_uri)
    key = f"{prefix.rstrip('/')}/{filename}"

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )

    return f"s3://{bucket}/{key}"


def normalize_product_daily_df(product_daily_df: pd.DataFrame) -> pd.DataFrame:
    df = product_daily_df.copy()

    required_cols = [
        "date",
        "day",
        "week",
        "month",
        "product",
        "division",
        "service",
        "actual_cost_usd",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required product daily column: {col}")

    for col in ["allocated_budget_usd", "variance_usd", "variance_pct"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    ordered_cols = [
        "date",
        "day",
        "week",
        "month",
        "product",
        "division",
        "service",
        "actual_cost_usd",
        "allocated_budget_usd",
        "variance_usd",
        "variance_pct",
    ]
    return df[ordered_cols].copy()


def normalize_division_daily_df(division_daily_df: pd.DataFrame) -> pd.DataFrame:
    df = division_daily_df.copy()

    required_cols = [
        "date",
        "day",
        "week",
        "month",
        "division",
        "service",
        "actual_cost_usd",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required division daily column: {col}")

    for col in ["allocated_budget_usd", "variance_usd", "variance_pct"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    ordered_cols = [
        "date",
        "day",
        "week",
        "month",
        "division",
        "service",
        "actual_cost_usd",
        "allocated_budget_usd",
        "variance_usd",
        "variance_pct",
    ]
    return df[ordered_cols].copy()


def ensure_daily_tables_exist() -> None:
    product_location = f"{DAILY_MARTS_S3_BASE.rstrip('/')}/{PRODUCT_DAILY_PREFIX}"
    division_location = f"{DAILY_MARTS_S3_BASE.rstrip('/')}/{DIVISION_DAILY_PREFIX}"

    drop_product_sql = f"DROP TABLE IF EXISTS {ATHENA_DATABASE}.{PRODUCT_TABLE_NAME}"
    drop_division_sql = f"DROP TABLE IF EXISTS {ATHENA_DATABASE}.{DIVISION_TABLE_NAME}"

    product_sql = f"""
    CREATE EXTERNAL TABLE {ATHENA_DATABASE}.{PRODUCT_TABLE_NAME} (
        date string,
        day string,
        week string,
        month string,
        product string,
        division string,
        service string,
        actual_cost_usd double,
        allocated_budget_usd double,
        variance_usd double,
        variance_pct double
    )
    ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
    WITH SERDEPROPERTIES (
        'separatorChar' = ',',
        'quoteChar' = '"',
        'escapeChar' = '\\\\'
    )
    STORED AS TEXTFILE
    LOCATION '{product_location}'
    TBLPROPERTIES ('skip.header.line.count'='1')
    """

    division_sql = f"""
    CREATE EXTERNAL TABLE {ATHENA_DATABASE}.{DIVISION_TABLE_NAME} (
        date string,
        day string,
        week string,
        month string,
        division string,
        service string,
        actual_cost_usd double,
        allocated_budget_usd double,
        variance_usd double,
        variance_pct double
    )
    ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
    WITH SERDEPROPERTIES (
        'separatorChar' = ',',
        'quoteChar' = '"',
        'escapeChar' = '\\\\'
    )
    STORED AS TEXTFILE
    LOCATION '{division_location}'
    TBLPROPERTIES ('skip.header.line.count'='1')
    """

    execute_athena_query(drop_product_sql)
    execute_athena_query(drop_division_sql)
    execute_athena_query(product_sql)
    execute_athena_query(division_sql)


def repair_daily_tables() -> None:
    execute_athena_query(f"MSCK REPAIR TABLE {ATHENA_DATABASE}.{PRODUCT_TABLE_NAME}")
    execute_athena_query(f"MSCK REPAIR TABLE {ATHENA_DATABASE}.{DIVISION_TABLE_NAME}")


def build_and_publish_daily_marts(selection: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        scope,
    )

    products_budget_filtered = filter_budget_table(
        products_daily_budget_df,
        products_weekly_budget_df,
        products_monthly_budget_df,
        scope,
    )

    internal_budget_filtered = filter_budget_table(
        internal_daily_budget_df,
        internal_weekly_budget_df,
        internal_monthly_budget_df,
        scope,
    )

    product_daily_df = build_products_granular_frame(
        products_usage_filtered,
        products_budget_filtered,
        scope,
        pricing_df,
    )

    division_daily_df = build_internal_granular_frame(
        internal_usage_filtered,
        internal_budget_filtered,
        scope,
        pricing_df,
    )

    product_out = normalize_product_daily_df(product_daily_df)
    division_out = normalize_division_daily_df(division_daily_df)

    product_s3_uri = f"{DAILY_MARTS_S3_BASE.rstrip('/')}/{PRODUCT_DAILY_PREFIX}"
    division_s3_uri = f"{DAILY_MARTS_S3_BASE.rstrip('/')}/{DIVISION_DAILY_PREFIX}"

    delete_s3_prefix(product_s3_uri)
    delete_s3_prefix(division_s3_uri)

    upload_df_as_csv_to_s3(product_out, product_s3_uri, "product_daily_current.csv")
    upload_df_as_csv_to_s3(division_out, division_s3_uri, "division_daily_current.csv")

    ensure_daily_tables_exist()
    repair_daily_tables()

    return division_out, product_out


if __name__ == "__main__":
    force_aws_profile()

    test_selection = {
        "mode": "days",
        "block": "C",
        "week": 9,
        "days": 3,
        "period": "Monday to Wednesday",
    }

    division_df, product_df = build_and_publish_daily_marts(test_selection)
    print("\n==============================")
    print("RAW DIVISION DAILY DF")
    print("==============================")
    print(division_df[["date", "day", "week"]].drop_duplicates().sort_values("date").to_string(index=False))

    print("\n==============================")
    print("RAW PRODUCT DAILY DF")
    print("==============================")
    print(product_df[["date", "day", "week"]].drop_duplicates().sort_values("date").to_string(index=False))