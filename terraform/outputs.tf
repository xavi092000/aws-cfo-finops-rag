output "bucket_name" {
  value = data.aws_s3_bucket.existing.bucket
}

output "glue_database_name" {
  value = aws_glue_catalog_database.this.name
}

output "athena_workgroup_name" {
  value = aws_athena_workgroup.this.name
}

output "athena_results_path" {
  value = "s3://${var.existing_bucket_name}/aws-cfo-finops-rag/audit/athena-results/"
}