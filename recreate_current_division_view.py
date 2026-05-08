from pathlib import Path
import time
import boto3

AWS_PROFILE = "terraform-runner"
AWS_REGION = "us-east-1"
SCHEMA_NAME = "cfo_finops_db"
S3_STAGING_DIR = "s3://kestra-zoomcamp-ny-taxi-mtl9999999999988888/athena-results/"
WORK_GROUP = "primary"

SQL_FILE = Path("recreate_current_division_view.sql")

query = SQL_FILE.read_text(encoding="utf-8")

session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
athena = session.client("athena")

response = athena.start_query_execution(
    QueryString=query,
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

print("\nView recreated successfully.")