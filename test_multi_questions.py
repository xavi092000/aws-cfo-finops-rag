from cfo_finops_athena_rag_final import run_finops_cfo_pipeline

# Sélection fixe pour les tests
selection = {
    "mode": "weeks",
    "weeks": [5]
}

# 3 questions à tester
questions = [
    "What are good FinOps habits?",
    "What was the average cost for services and how can we improve FinOps habits?",
    "Our cloud costs seem messy, what should we fix?"
]

print("\n==============================")
print("MULTI-QUESTION RAG TEST")
print("==============================\n")

for i, question in enumerate(questions, 1):
    print(f"\n--- TEST {i} ---")
    print("QUESTION:", question)
    print("-" * 50)

    result = run_finops_cfo_pipeline(
        question=question,
        selection=selection,
        enable_audio=False  # important pour aller plus vite
    )

    print("ROUTE:", result["route"])
    print("FALLBACK:", result["fallback"])
    print("SOURCES:", result["sources"])

    print("\nANSWER:\n")
    print(result["answer"])

    print("\n" + "="*60)