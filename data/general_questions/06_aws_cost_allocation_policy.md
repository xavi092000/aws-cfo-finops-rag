# AWS Cost Allocation Policy

## Purpose

This document defines how AWS cost should be allocated across products, teams, environments, and workloads in the project.

## Allocation hierarchy

1. Direct assignment using mandatory tags
2. Account-based assignment if tagging is missing
3. Shared-cost allocation using approved allocation rules
4. Unallocated bucket only as a temporary exception

## Mandatory tags

* owner\_team
* product
* environment
* workload
* cost\_center

## Allocation rules

### Direct costs

Assign directly when resource-level ownership is clear.

### Shared platform costs

Allocate by a documented driver such as:

* percentage of compute hours
* percentage of storage consumed
* number of active workloads
* number of users or consumers
* fixed platform share

### Temporary unallocated costs

Allow only when:

* the resource is newly deployed
* tagging enforcement failed
* upstream metadata is missing

All temporary unallocated cost must be remediated within the next reporting cycle.

## Examples

* Shared S3 bucket for analytics landing zone: allocate by data volume written per workload
* Shared Redshift cluster: allocate by compute usage or scheduled workload share
* Shared orchestration environment: allocate by pipeline execution count

## Governance

* FinOps Practitioner reviews allocation exceptions weekly
* Engineering remediates missing ownership metadata
* Finance validates reporting consistency monthly

## Typical questions this document helps answer

* How should shared cost be allocated?
* What happens when tagging is incomplete?
* Which tag fields are mandatory?

## Metadata

* source: internal\_project\_docs
* domain: Understand Usage \& Cost
* capability: Allocation
* document\_type: policy
* authority: project\_internal

\## Cloud Cost Allocation



Cloud cost allocation is the practice of assigning cloud costs to specific teams, products, or business units.



It ensures:

\- visibility of spending

\- accountability

\- accurate financial reporting



This is typically done using tagging, cost centers, and allocation rules.

