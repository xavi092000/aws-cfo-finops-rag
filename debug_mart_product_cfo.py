from __future__ import annotations

import os
from typing import Optional

import boto3
import pandas as pd
from dotenv import load_dotenv
from pyathena import connect

load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE", "terraform-runner")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
ATHENA_SCHEMA = os.getenv("ATHENA_SCHEMA", "cfo_finops_db")
ATHENA_S3_STAGING_DIR = os.getenv(
    "ATHENA_S3_STAGING_DIR",
    "s3://kestra-zoomcamp-ny-taxi-mtl9999999999988888/athena-results/"
)
ATHENA_WORKGROUP = os.getenv("ATHENA_WORKGROUP", "primary")


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


def get_connection():
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return connect(
        region_name=AWS_REGION,
        schema_name=ATHENA_SCHEMA,
        s3_staging_dir=ATHENA_S3_STAGING_DIR,
        work_group=ATHENA_WORKGROUP,
        boto3_session=session,
        profile_name=AWS_PROFILE,
    )


def run_query(title: str, sql: str) -> Optional[pd.DataFrame]:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(sql.strip())
    print("-" * 80)

    conn = get_connection()
    try:
        df = pd.read_sql(sql, conn)
        print(df.to_string(index=False))
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        return None
    finally:
        conn.close()


def main() -> None:
    force_aws_profile()

    # 1) Colonnes du mart
    run_query(
        "1) COLUMNS - mart_product_cfo",
        f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = '{ATHENA_SCHEMA}'
          AND table_name = 'mart_product_cfo'
        ORDER BY ordinal_position
        """
    )

    # 2) Colonnes du staging
    run_query(
        "2) COLUMNS - stg_products_daily_budget",
        f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = '{ATHENA_SCHEMA}'
          AND table_name = 'stg_products_daily_budget'
        ORDER BY ordinal_position
        """
    )

    # 3) Lignes brutes du mart
    run_query(
        "3) RAW ROWS - mart_product_cfo for Week 5 and Week 6",
        f"""
        SELECT *
        FROM {ATHENA_SCHEMA}.mart_product_cfo
        WHERE week IN ('Week 5', 'Week 6')
        ORDER BY week, product
        """
    )

    # 4) Totaux du mart par semaine
    run_query(
        "4) TOTALS - mart_product_cfo by week",
        f"""
        SELECT
            week,
            COUNT(*) AS row_count,
            SUM(actual_cost_usd) AS mart_actual_sum,
            SUM(allocated_budget_usd) AS mart_budget_sum,
            SUM(actual_cost_usd - allocated_budget_usd) AS mart_variance_recomputed
        FROM {ATHENA_SCHEMA}.mart_product_cfo
        WHERE week IN ('Week 5', 'Week 6')
        GROUP BY week
        ORDER BY week
        """
    )

    # 5) Totaux du staging agrégés au même niveau
    run_query(
        "5) TOTALS - stg_products_daily_budget aggregated by week",
        f"""
        SELECT
            week,
            COUNT(*) AS row_count,
            SUM(actual_cost_usd) AS stg_actual_sum,
            SUM(allocated_budget_usd) AS stg_budget_sum,
            SUM(actual_cost_usd - allocated_budget_usd) AS stg_variance_recomputed
        FROM {ATHENA_SCHEMA}.stg_products_daily_budget
        WHERE week IN ('Week 5', 'Week 6')
        GROUP BY week
        ORDER BY week
        """
    )

    # 6) Mart par produit
    run_query(
        "6) MART - by week + product",
        f"""
        SELECT
            week,
            product,
            COUNT(*) AS row_count,
            SUM(actual_cost_usd) AS mart_actual_sum,
            SUM(allocated_budget_usd) AS mart_budget_sum,
            SUM(actual_cost_usd - allocated_budget_usd) AS mart_variance_recomputed
        FROM {ATHENA_SCHEMA}.mart_product_cfo
        WHERE week IN ('Week 5', 'Week 6')
        GROUP BY week, product
        ORDER BY week, product
        """
    )

    # 7) Staging par produit
    run_query(
        "7) STAGING - by week + product",
        f"""
        SELECT
            week,
            product,
            COUNT(*) AS row_count,
            SUM(actual_cost_usd) AS stg_actual_sum,
            SUM(allocated_budget_usd) AS stg_budget_sum,
            SUM(actual_cost_usd - allocated_budget_usd) AS stg_variance_recomputed
        FROM {ATHENA_SCHEMA}.stg_products_daily_budget
        WHERE week IN ('Week 5', 'Week 6')
        GROUP BY week, product
        ORDER BY week, product
        """
    )

    # 8) Comparaison mart vs staging au niveau week + product
    run_query(
        "8) COMPARE - mart vs staging by week + product",
        f"""
        WITH mart AS (
            SELECT
                week,
                product,
                SUM(actual_cost_usd) AS mart_actual_sum,
                SUM(allocated_budget_usd) AS mart_budget_sum,
                SUM(actual_cost_usd - allocated_budget_usd) AS mart_variance_recomputed
            FROM {ATHENA_SCHEMA}.mart_product_cfo
            WHERE week IN ('Week 5', 'Week 6')
            GROUP BY week, product
        ),
        stg AS (
            SELECT
                week,
                product,
                SUM(actual_cost_usd) AS stg_actual_sum,
                SUM(allocated_budget_usd) AS stg_budget_sum,
                SUM(actual_cost_usd - allocated_budget_usd) AS stg_variance_recomputed
            FROM {ATHENA_SCHEMA}.stg_products_daily_budget
            WHERE week IN ('Week 5', 'Week 6')
            GROUP BY week, product
        )
        SELECT
            COALESCE(mart.week, stg.week) AS week,
            COALESCE(mart.product, stg.product) AS product,
            mart.mart_actual_sum,
            stg.stg_actual_sum,
            mart.mart_budget_sum,
            stg.stg_budget_sum,
            mart.mart_variance_recomputed,
            stg.stg_variance_recomputed,
            (COALESCE(mart.mart_actual_sum, 0) - COALESCE(stg.stg_actual_sum, 0)) AS actual_diff,
            (COALESCE(mart.mart_budget_sum, 0) - COALESCE(stg.stg_budget_sum, 0)) AS budget_diff,
            (COALESCE(mart.mart_variance_recomputed, 0) - COALESCE(stg.stg_variance_recomputed, 0)) AS variance_diff
        FROM mart
        FULL OUTER JOIN stg
          ON mart.week = stg.week
         AND mart.product = stg.product
        ORDER BY week, product
        """
    )

    # 9) Détection simple de doublons dans le mart
    run_query(
        "9) DUPLICATE CHECK - mart_product_cfo rows per week + product",
        f"""
        SELECT
            week,
            product,
            COUNT(*) AS row_count
        FROM {ATHENA_SCHEMA}.mart_product_cfo
        WHERE week IN ('Week 5', 'Week 6')
        GROUP BY week, product
        HAVING COUNT(*) > 1
        ORDER BY row_count DESC, week, product
        """
    )


if __name__ == "__main__":
    main()