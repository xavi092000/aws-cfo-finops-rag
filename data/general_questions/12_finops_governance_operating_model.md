# FinOps Governance and Operating Model

## Purpose
Define how FinOps is run as a cross-functional practice.

## Governance goals
- Clear ownership for spend, usage, value, and remediation.
- Consistent decisions across finance, engineering, and product.
- Fast escalation for material anomalies or budget risks.
- Repeatable policy with auditable decisions.

## Governance layers
### Executive steering
Participants: CFO, CTO, Head of Engineering, FinOps lead, key business owners.
Responsibilities:
- approve policy and targets
- resolve trade-offs between growth, resilience, and cost
- review quarterly commitments and major variances

### Monthly FinOps review
Participants: FinOps, finance partner, engineering managers, product owners.
Responsibilities:
- review actuals vs budget and forecast
- explain material deltas
- assign optimization actions
- review unit economics and KPI trends

### Weekly operational review
Participants: FinOps analysts, platform team, service owners.
Responsibilities:
- review anomalies
- check unresolved budget risks
- validate tagging and allocation quality
- confirm action progress

## RACI summary
### Finance
- accountable for budget governance and forecasting alignment
- consulted on holdbacks, accruals, and reporting standards

### Engineering
- responsible for remediation, rightsizing, scheduling, architecture, and service selection

### Product / Business
- accountable for value definition, demand assumptions, and prioritization trade-offs

### FinOps Practice
- responsible for analytics, policy coordination, recommendations, and operating cadence

## Minimum policies
- tagging policy for owner, environment, application, cost center, and data classification where needed
- budget and forecast refresh policy
- anomaly management severity thresholds
- commitment management review policy
- exception handling and approval policy
- retention policy for cost and usage data

## Standard decisions that should be documented
- who can approve budget increases
- who can launch material new workloads
- who can purchase commitments
- when idle resources must be stopped or removed
- what variance triggers executive review

## Measures of success
- high allocation coverage
- high tag compliance on spend
- forecast accuracy improving over time
- budget variance within defined thresholds
- remediation cycle time for anomalies decreasing
- rising share of spend linked to unit metrics and business outcomes

## Metadata
- source: internal_project_docs
- document_type: governance_operating_model
- audience: CFO, CTO, FinOps lead
