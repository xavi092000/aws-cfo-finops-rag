from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from sentence_transformers import CrossEncoder, SentenceTransformer

from cfo_finops_athena_rag_final import classify_question, handle_general_question

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_QUESTIONS = [
    # Niveau 1
    {"id": 1, "question": "What is FinOps", "expected_route": "general", "expected_behavior": "answer"},
    {"id": 2, "question": "What are the core principles of FinOps", "expected_route": "general", "expected_behavior": "answer"},
    {"id": 3, "question": "What is unit economics in FinOps", "expected_route": "general", "expected_behavior": "answer"},
    {"id": 4, "question": "What is cloud cost allocation", "expected_route": "general", "expected_behavior": "answer"},
    {"id": 5, "question": "What is showback vs chargeback in FinOps", "expected_route": "general", "expected_behavior": "answer"},

    # Niveau 2
    {"id": 6, "question": "Why is FinOps important for CFO decision making", "expected_route": "general", "expected_behavior": "answer"},
    {"id": 7, "question": "How does FinOps help control cloud spending", "expected_route": "general", "expected_behavior": "answer"},
    {"id": 8, "question": "What are the main drivers of cloud cost growth", "expected_route": "general", "expected_behavior": "answer"},
    {"id": 9, "question": "What are common FinOps KPIs", "expected_route": "general", "expected_behavior": "answer"},
    {"id": 10, "question": "What is cost optimization in FinOps", "expected_route": "general", "expected_behavior": "answer"},

    # Niveau 3
    {"id": 11, "question": "What is the FinOps lifecycle", "expected_route": "general", "expected_behavior": "answer"},
    {"id": 12, "question": "How do teams collaborate in FinOps", "expected_route": "general", "expected_behavior": "answer"},
    {"id": 13, "question": "What is the difference between forecasting and budgeting in FinOps", "expected_route": "general", "expected_behavior": "answer"},
    {"id": 14, "question": "What is cloud financial management", "expected_route": "general", "expected_behavior": "answer"},
    {"id": 15, "question": "What are common FinOps maturity stages", "expected_route": "general", "expected_behavior": "answer"},

    # Anti-hallucination
    {"id": 16, "question": "What is FinOps in Kubernetes autoscaling", "expected_route": "general", "expected_behavior": "fallback"},
    {"id": 17, "question": "What is FinOps GPU optimization for LLM training", "expected_route": "general", "expected_behavior": "fallback"},
    {"id": 18, "question": "What is FinOps for quantum computing", "expected_route": "general", "expected_behavior": "fallback"},
]

FALLBACK_TEXT = "I don't have enough information in the retrieved documents."


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def classify_status(expected_route: str, actual_route: str, expected_behavior: str, result: Dict) -> str:
    if actual_route != expected_route:
        return "FAIL"

    answer = result.get("answer", "").strip()
    fallback = bool(result.get("fallback", False))

    if expected_behavior == "fallback":
        if fallback or answer == FALLBACK_TEXT:
            return "PASS"
        return "FAIL"

    if expected_behavior == "answer":
        if answer and answer != FALLBACK_TEXT and not fallback:
            return "PASS"
        return "PARTIAL"

    return "FAIL"


def main() -> None:
    print("============================================================")
    print("AUTOMATED RAG TEST HARNESS")
    print("============================================================")
    print(f"Questions: {len(TEST_QUESTIONS)}")
    print("Hybrid tests: DISABLED")
    print("============================================================\n")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    results: List[Dict] = []

    for test in TEST_QUESTIONS:
        question = test["question"]
        expected_route = test["expected_route"]
        expected_behavior = test["expected_behavior"]

        actual_route = classify_question(question)

        if actual_route == "general":
            result = handle_general_question(question, embedding_model, reranker)
        else:
            result = {
                "answer": "",
                "fallback": False,
                "confidence": {"label": "N/A", "reason": "Not run because classification was not general"},
                "sources": "N/A",
            }

        status = classify_status(expected_route, actual_route, expected_behavior, result)

        row = {
            "id": test["id"],
            "question": question,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "expected_behavior": expected_behavior,
            "status": status,
            "answer": result.get("answer", ""),
            "fallback": result.get("fallback", False),
            "confidence_label": result.get("confidence", {}).get("label", "N/A"),
            "confidence_reason": result.get("confidence", {}).get("reason", "N/A"),
            "sources": result.get("sources", "N/A"),
        }
        results.append(row)

        print("------------------------------------------------------------")
        print(f"[Q{test['id']}] {question}")
        print(f"Expected route   : {expected_route}")
        print(f"Actual route     : {actual_route}")
        print(f"Expected behavior: {expected_behavior}")
        print(f"Status           : {status}")
        print(f"Answer preview   : {row['answer'][:300].replace(chr(10), ' ')}")
        if status != "PASS":
            print("Top results:")
            for r in result.get("top_results", [])[:3]:
                print(
                f"file={r.get('filename')} | "
                f"rerank={r.get('rerank_score')} | "
                f"semantic={r.get('semantic_score')} | "
                f"bm25={r.get('bm25_score')}"
        )

    pass_count = sum(1 for r in results if r["status"] == "PASS")
    partial_count = sum(1 for r in results if r["status"] == "PARTIAL")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"rag_test_results_{timestamp}.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n============================================================")
    print("FINAL RAG TEST SUMMARY")
    print("============================================================")
    print(f"Total tests : {len(results)}")
    print(f"PASS        : {pass_count}")
    print(f"PARTIAL     : {partial_count}")
    print(f"FAIL        : {fail_count}")
    print("============================================================")
    print(f"JSON report : {json_path}")
    print("============================================================")


if __name__ == "__main__":
    main()