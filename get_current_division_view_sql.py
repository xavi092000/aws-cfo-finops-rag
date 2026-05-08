import time
import boto3

AWS_PROFILE = "terraform-runner"
AWS_REGION = "us-east-1"
SCHEMA_NAME = "cfo_finops_db"
S3_STAGING_DIR = "s3://kestra-zoomcamp-ny-taxi-mtl9999999999988888/athena-results/"
WORK_GROUP = "primary"

QUERY = f"SHOW CREATE VIEW {SCHEMA_NAME}.current_division_view"

session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
athena = session.client("athena")

response = athena.start_query_execution(
    QueryString=QUERY,
    QueryExecutionContext={"Database": SCHEMA_NAME},
    ResultConfiguration={"OutputLocation": S3_STAGING_DIR},
    WorkGroup=WORK_GROUP,
)

query_execution_id = response["QueryExecutionId"]
print("QueryExecutionId:", query_execution_id)

while True:
    result = athena.get_query_execution(QueryExecutionId=query_execution_id)
    state = result["QueryExecution"]["Status"]["State"]
    print("State:", state)

    if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
        break

    time.sleep(2)

if state != "SUCCEEDED":
    reason = result["QueryExecution"]["Status"].get("StateChangeReason", "No reason returned")
    raise RuntimeError(f"Athena query failed with state={state}. Reason={reason}")

results = athena.get_query_results(QueryExecutionId=query_execution_id)

print("\n==============================")
print("SHOW CREATE VIEW RESULT")
print("==============================")

for row in results["ResultSet"]["Rows"]:
    values = [col.get("VarCharValue", "") for col in row["Data"]]
    print(" | ".join(values))