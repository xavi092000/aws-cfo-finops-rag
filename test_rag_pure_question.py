from cfo_finops_athena_rag_final import run_finops_cfo_pipeline

question = "What are good FinOps habits?"

selection = {
    "mode": "weeks",
    "weeks": [5]
}

print("\n=== PURE RAG TEST ===\n")

result = run_finops_cfo_pipeline(
    question=question,
    selection=selection
)

print(result)