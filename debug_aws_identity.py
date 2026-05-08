import os
import boto3

AWS_PROFILE = "terraform-runner"
AWS_REGION = "us-east-1"

print("=== ENV VARIABLES ===")
print("AWS_PROFILE env:", os.getenv("AWS_PROFILE"))
print("AWS_ACCESS_KEY_ID exists:", bool(os.getenv("AWS_ACCESS_KEY_ID")))
print("AWS_SECRET_ACCESS_KEY exists:", bool(os.getenv("AWS_SECRET_ACCESS_KEY")))
print("AWS_SESSION_TOKEN exists:", bool(os.getenv("AWS_SESSION_TOKEN")))
print()

print("=== DEFAULT BOTO3 SESSION ===")
default_sts = boto3.client("sts")
print(default_sts.get_caller_identity())
print()

print("=== EXPLICIT PROFILE SESSION ===")
session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
profile_sts = session.client("sts")
print(profile_sts.get_caller_identity())
print()

print("=== AVAILABLE PROFILE NAME IN SESSION ===")
print("Session profile:", session.profile_name)