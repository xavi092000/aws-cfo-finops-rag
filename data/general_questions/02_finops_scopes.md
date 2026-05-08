# FinOps Scopes

## Purpose
This document defines how spending can be segmented into practical scopes for analysis and decision-making.

## Definition
A FinOps scope is a defined segment of technology-related spending aligned to business constructs such as products, cost centers, environments, or teams.

## Why scopes matter
Scopes make FinOps usable. Without scopes, cost analysis stays too broad and actionability stays weak.

## Recommended scopes for this AWS RAG project
### Scope: Product
Use when a platform supports multiple products or customer-facing offerings.

### Scope: Cost Center
Use when finance accountability is organized around departments or budgets.

### Scope: Environment
Use when separating production, staging, development, and sandbox cost is important.

### Scope: Team
Use when ownership is tied to engineering squads, analytics teams, or platform groups.

### Scope: Account
Use when AWS accounts map cleanly to business or platform boundaries.

### Scope: Workload
Use when a single workload, pipeline, model, dashboard, or application needs dedicated tracking.

## Example scope model
- Product: FinOps Assistant
- Team: Data Platform
- Environment: Production
- Account: aws-prod-finops
- Workload: daily-cost-ingestion-pipeline

## Typical questions this document helps answer
- What unit should we use to analyze spend?
- Should we group by product, environment, or team?
- How do we align cost analysis with business ownership?

## Metadata
- source: finops_framework_adapted
- section: scopes
- document_type: framework_reference
- authority: official_adapted
