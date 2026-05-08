from cfo_finops_athena_rag_final import run_finops_cfo_pipeline

question = "What was the average cost for services during this period and what FinOps habits should we improve?"

selection = {
    "mode": "weeks",
    "block": "C",
    "weeks": [11, 12],
    "number_of_weeks": 2,
}


print("\n=== HYBRID QUESTION TEST ===\n")

result = run_finops_cfo_pipeline(
    question=question,
    selection=selection
)

print("QUESTION:", result["question"])
print("ROUTE:", result["route"])
print("FALLBACK:", result.get("fallback"))
print("SOURCES:", result.get("sources"))

print("\nANSWER:\n")
print(result["answer"])