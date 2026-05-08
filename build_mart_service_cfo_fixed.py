import time
from agent_entree_finopsrag import (
    get_boto3_session,
    ATHENA_DATABASE,
    ATHENA_S3_STAGING_DIR,
    ATHENA_WORKGROUP,
)

ATHENA_SCHEMA = "cfo_finops_db"

sql = f"""
CREATE OR REPLACE VIEW {ATHENA_SCHEMA}.mart_service_cfo_fixed AS
SELECT
    week,
    month,
    service,
    SUM(actual_cost_usd) AS actual_cost_usd,
    SUM(allocated_budget_usd) AS allocated_budget_usd,
    SUM(variance_usd) AS variance_usd,
    CASE
        WHEN SUM(allocated_budget_usd) > 0
        THEN (SUM(variance_usd) / SUM(allocated_budget_usd)) * 100
        ELSE 0
    END AS variance_percent
FROM {ATHENA_SCHEMA}.mart_product_cfo_daily
GROUP BY week, month, service
"""

session = get_boto3_session()
athena = session.client("athena")

response = athena.start_query_execution(
    QueryString=sql,
    QueryExecutionContext={"Database": ATHENA_DATABASE},
    ResultConfiguration={"OutputLocation": ATHENA_S3_STAGING_DIR},
    WorkGroup=ATHENA_WORKGROUP,
)

query_execution_id = response["QueryExecutionId"]
print("QueryExecutionId:", query_execution_id)

while True:
    status_response = athena.get_query_execution(QueryExecutionId=query_execution_id)
    state = status_response["QueryExecution"]["Status"]["State"]

    if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
        print("State:", state)
        if state != "SUCCEEDED":
            print("Reason:", status_response["QueryExecution"]["Status"].get("StateChangeReason", "Unknown error"))
        break

    time.sleep(1)