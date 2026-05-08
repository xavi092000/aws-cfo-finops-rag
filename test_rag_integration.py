from cfo_finops_athena_rag_final import handle_rag_failure

def fake_rag_pipeline():
    # simulate RAG failure
    return "I don't have enough information in the retrieved documents."

def main():
    rag_answer = fake_rag_pipeline()

    final = handle_rag_failure(rag_answer)

    print("\n=== INTEGRATION TEST ===\n")
    print(final)

if __name__ == "__main__":
    main()