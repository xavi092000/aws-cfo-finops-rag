data "aws_s3_bucket" "existing" {
  bucket = var.existing_bucket_name
}

resource "aws_glue_catalog_database" "this" {
  name = var.glue_database_name
}

resource "aws_athena_workgroup" "this" {
  name = var.athena_workgroup_name

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true

    result_configuration {
      output_location = "s3://${var.existing_bucket_name}/aws-cfo-finops-rag/audit/athena-results/"
    }
  }

  state = "ENABLED"
}

resource "aws_s3_object" "prefixes" {
  for_each = toset([
    "aws-cfo-finops-rag/raw/products/",
    "aws-cfo-finops-rag/raw/internal/",
    "aws-cfo-finops-rag/raw/pricing/",
    "aws-cfo-finops-rag/prepared/",
    "aws-cfo-finops-rag/curated/",
    "aws-cfo-finops-rag/rag_ready/",
    "aws-cfo-finops-rag/audit/athena-results/"
  ])

  bucket  = var.existing_bucket_name
  key     = each.value
  content = ""
}