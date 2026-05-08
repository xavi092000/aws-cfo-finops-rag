variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name prefix"
  type        = string
  default     = "aws-cfo-finops-rag"
}

variable "existing_bucket_name" {
  description = "Existing S3 bucket name"
  type        = string
}

variable "glue_database_name" {
  description = "Glue Catalog database name"
  type        = string
  default     = "cfo_finops_db"
}

variable "athena_workgroup_name" {
  description = "Athena workgroup name"
  type        = string
  default     = "cfo-finops-wg"
}