from __future__ import annotations

import time
import boto3
import pandas as pd

AWS_PROFILE = "terraform-runner"
AWS_REGION = "us-east-1"

ATHENA_DATABASE = "cfo_finops_db"
ATHENA_WORKGROUP = "primary"
ATHENA_RESULTS_S3 = "s3://kestra-zoomcamp-ny-taxi-mtl9999999999988888/athena-results/"


def get_boto3_session() -> boto3.Session:
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


def run_athena_query(sql: str) -> str:
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
            "Unknown Athena error",
        )
        raise RuntimeError(f"Athena query failed with state {state}: {reason}")

    return query_execution_id


def athena_results_to_dataframe(query_execution_id: str) -> pd.DataFrame:
    session = get_boto3_session()
    athena = session.client("athena", region_name=AWS_REGION)

    paginator = athena.get_paginator("get_query_results")
    pages = paginator.paginate(QueryExecutionId=query_execution_id)

    rows = []
    headers = None

    for page in pages:
        result_rows = page["ResultSet"]["Rows"]

        if not result_rows:
            continue

        if headers is None:
            headers = [col.get("VarCharValue", "") for col in result_rows[0]["Data"]]
            data_rows = result_rows[1:]
        else:
            data_rows = result_rows

        for row in data_rows:
            values = [col.get("VarCharValue", "") for col in row["Data"]]
            if headers and len(values) < len(headers):
                values += [""] * (len(headers) - len(values))
            rows.append(values)

    if headers is None:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=headers)


def _clean_numeric_column(df: pd.DataFrame, column_name: str) -> None:
    if column_name in df.columns:
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce").fillna(0.0)


def load_current_powerbi_views() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    product_sql = """
    SELECT
        period_value,
        date,
        day,
        week,
        month,
        product,
        division,
        service,
        actual_cost_usd,
        allocated_budget_usd,
        variance_usd,
        variance_percent
    FROM cfo_finops_db.current_product_view
    """

    division_sql = """
    SELECT
        period_value,
        date,
        day,
        week,
        month,
        division,
        service,
        actual_cost_usd,
        allocated_budget_usd,
        variance_usd,
        variance_percent
    FROM cfo_finops_db.current_division_view
    """

    # 🔥 FIX ICI
    service_sql = """
    SELECT
        period_value,
        date,
        day,
        week,
        month,
        service,
        actual_cost_usd,
        allocated_budget_usd,
        variance_usd,
        variance_percent
    FROM cfo_finops_db.current_service_view
    """

    product_df = athena_results_to_dataframe(run_athena_query(product_sql))
    division_df = athena_results_to_dataframe(run_athena_query(division_sql))
    service_df = athena_results_to_dataframe(run_athena_query(service_sql))

    for df in [product_df, division_df, service_df]:
        for col in ["actual_cost_usd", "allocated_budget_usd", "variance_usd", "variance_percent"]:
            _clean_numeric_column(df, col)

    return division_df, product_df, service_df