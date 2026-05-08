from cfo_finops_athena_rag_final import (
    embedding_model_global,
    reranker_global,
    handle_general_question,
    llm_rewrite_question_for_rag,
)

questions = [
    "what are some good fin ops habits",
    "What are good FinOps habits?",
    "What are the most important FinOps best practices a CFO should prioritize?",
    "What FinOps best practices should a CFO prioritize for cost control, governance, forecasting, and accountability?"
]

for q in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", q)

    rewritten = llm_rewrite_question_for_rag(q)
    print("REWRITTEN:", rewritten)

    result = handle_general_question(
        rewritten,
        embedding_model_global,
        reranker_global
    )

    print("FALLBACK:", result["fallback"])
    print("SOURCES:", result["sources"])
    print("ANSWER PREVIEW:")
    print(result["answer"][:700])