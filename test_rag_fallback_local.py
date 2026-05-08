from cfo_finops_athena_rag_final import handle_rag_failure

# Simule la mauvaise réponse actuelle du RAG
rag_answer = "I don't have enough information in the retrieved documents."

analytical_answer = "Within the selected period, the average actual cost for Services is $5.84."

final_answer = handle_rag_failure(
    rag_answer=rag_answer,
    analytical_answer=analytical_answer
)

print("\n==============================")
print("LOCAL RAG FALLBACK TEST")
print("==============================")
print(final_answer)