from agent_entree_finopsrag import (
    get_boto3_session,
    ATHENA_DATABASE,
    ATHENA_S3_STAGING_DIR,
    ATHENA_WORKGROUP,
)
import time


def run_query_rows(sql: str, max_rows: int = 20):
    athena = get_boto3_session().client("athena")

    resp = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DATABASE},
        ResultConfiguration={"OutputLocation": ATHENA_S3_STAGING_DIR},
        WorkGroup=ATHENA_WORKGROUP,
    )
    qid = resp["QueryExecutionId"]

    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]["State"]
        if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            print(f"status = {status}")
            break
        time.sleep(1)

    if status != "SUCCEEDED":
        print("Query failed")
        return []

    res = athena.get_query_results(QueryExecutionId=qid)
    rows = res["ResultSet"]["Rows"]

    parsed = []
    for row in rows[1:max_rows+1]:
        parsed.append([col.get("VarCharValue", "") for col in row["Data"]])

    return parsed


print("DISTINCT week values in division daily:")
rows = run_query_rows(
    "SELECT DISTINCT week FROM cfo_finops_db.mart_division_cfo_daily ORDER BY week",
    max_rows=50,
)

for r in rows:
    print(r[0])