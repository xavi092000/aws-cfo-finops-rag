from cfo_finops_athena_rag_final import llm_rewrite_question_for_rag

question = "Our cloud costs seem messy, what should we fix?"

rewritten = llm_rewrite_question_for_rag(question)

print("ORIGINAL:", question)
print("REWRITTEN:", rewritten)