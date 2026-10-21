# CFO FinOps - AWS Analytics and RAG Project

A personal engineering project exploring cloud-cost analysis, Athena-backed queries, retrieval-assisted financial explanations, and Power BI integration.

![Dashboard example](screenshots/cfo_dashboard.png)

## Scope and status

This repository is a portfolio implementation with cloud and local-environment dependencies. It is not a self-contained offline application, a verified currently hosted service, or proof of cost savings for a real organization. The dashboard image is an example, not a live availability indicator.

## Implementation to inspect

| Area | Entry point |
|---|---|
| Application entry | [agent_entree_finopsrag.py](agent_entree_finopsrag.py) |
| Analytics and routing | [cfo_final_router.py](cfo_final_router.py), [resolve_finops_analytics.py](resolve_finops_analytics.py) |
| Athena / RAG | [finops_agent_athena.py](finops_agent_athena.py), [src/semantic_rag_general.py](src/semantic_rag_general.py) |
| Financial explanations | [generate_cfo_summary.py](generate_cfo_summary.py) |
| Power BI integration | [powerbi_refresh.py](powerbi_refresh.py) |
| Evaluation code | [analytics_test_harness.py](analytics_test_harness.py), [rag_test_harness.py](rag_test_harness.py), [cfo_evaluation_suite.py](cfo_evaluation_suite.py) |
| Infrastructure | [Terraform files](terraform/) |

## Runtime preparation

Use Python 3.11 in an isolated environment:

```bash
python -m venv .venv
# Activate the environment for your shell, then:
python -m pip install -r requirements.txt
```

The dependency file is derived from imports and uses compatibility ranges. Installation alone does not validate the connected application. A full environment lock and end-to-end clean-machine run remain to be established.

The application requires account-specific Athena/S3 resources and tables, AWS credentials supplied locally, model configuration, and local data paths. Power BI refresh additionally requires tenant, application, workspace, and dataset configuration and the appropriate permissions. Inspect the entry points for expected environment names and account-specific paths before running. Do not commit credentials. Model downloads, external API calls, cloud queries, and dashboard refreshes can have network or billing effects.

After configuring the required services and data, the entry point is:

```bash
python agent_entree_finopsrag.py
```

The Dockerfile installs the same requirements; a successful build would not by itself prove that external services and datasets are configured.

## AWS and Athena configuration

AWS and Athena settings are configured through environment variables.
See [.env.example](.env.example) for non-sensitive examples.

| Variable | Purpose |
|---|---|
| `AWS_PROFILE` | Local AWS profile |
| `AWS_REGION` | AWS region |
| `ATHENA_DATABASE` | Athena database |
| `ATHENA_WORKGROUP` | Athena workgroup |
| `ATHENA_RESULTS_S3` | S3 destination for Athena query results |
| `DAILY_MARTS_S3_BASE` | S3 destination for Power BI daily marts |

Set `ATHENA_RESULTS_S3` and `DAILY_MARTS_S3_BASE` to valid,
authorized S3 locations before running the related AWS operations.

The application validates these S3 paths and raises an explicit
configuration error when a required path is missing or malformed.

Local Python syntax and configuration-validation tests have passed.
End-to-end AWS, Athena, and Power BI operations have not been
revalidated as part of this configuration change.

## Validation evidence
The repository contains evaluation and test scripts, but no public GitHub Actions run was available for this project during the 2026-10-21 review. Earlier README tables labelled every validation area `PASSED` without a directly linked, versioned execution record. Those labels are not treated as current verified results here.

To establish a new result, run the relevant suite in a configured environment and record the commit, date, inputs, dependencies, outcomes, and limitations. This documentation update did not execute cloud-backed tests or a deployment.

## Limitations

- Local paths and account-specific integration settings require adaptation.
- The RAG corpus, financial inputs, and analytical rules must be checked for the intended use case.
- A generated explanation is not independent verification of a financial result.
- Terraform configuration demonstrates intended infrastructure, not an active deployment.

Maintained by Felix Brillant.
