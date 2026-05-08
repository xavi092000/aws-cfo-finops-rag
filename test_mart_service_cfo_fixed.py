import time
from agent_entree_finopsrag import (
    get_boto3_session,
    ATHENA_DATABASE,
    ATHENA_S3_STAGING_DIR,
    ATHENA_WORKGROUP,
)

session = get_boto3_session()
athena = session.client("athena")

sql = "SELECT week, month, service, actual_cost_usd, variance_usd FROM cfo_finops_db.mart_service_cfo_fixed LIMIT 20"

response = athena.start_query_execution(
    QueryString=sql,
    QueryExecutionContext={"Database": ATHENA_DATABASE},
    ResultConfiguration={"OutputLocation": ATHENA_S3_STAGING_DIR},
    WorkGroup=ATHENA_WORKGROUP,
)

qid = response["QueryExecutionId"]
print("QueryExecutionId:", qid)

while True:
    status_response = athena.get_query_execution(QueryExecutionId=qid)
    state = status_response["QueryExecution"]["Status"]["State"]

    if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
        print("State:", state)
        if state != "SUCCEEDED":
            print("Reason:", status_response["QueryExecution"]["Status"].get("StateChangeReason", "Unknown error"))
        break

    time.sleep(1)

if state == "SUCCEEDED":
    results = athena.get_query_results(QueryExecutionId=qid)
    rows = results["ResultSet"]["Rows"]

    print("\nPreview:")
    for row in rows:
        print([col.get("VarCharValue", "") for col in row["Data"]])