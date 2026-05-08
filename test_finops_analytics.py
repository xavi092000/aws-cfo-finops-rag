from __future__ import annotations

from cfo_finops_athena_rag_final import run_finops_cfo_pipeline


def run_test(question: str, expected_substring: str, selection: dict) -> None:
    result = run_finops_cfo_pipeline(
        selection=selection,
        question=question,
        enable_audio=False,
        auto_open_audio=False,
    )

    answer = result["answer"]

    print("\n==================================================")
    print(f"QUESTION: {question}")
    print("ANSWER:")
    print(answer)
    print("==================================================")

    assert expected_substring in answer, (
        f"Test failed.\n"
        f"Expected to find: {expected_substring}\n"
        f"Actual answer:\n{answer}"
    )


def main() -> None:
    selection = {
        "mode": "days",
        "block": "A",
        "week": 3,
        "days": 4,
        "period": "Monday to Thursday",
    }

    tests = [
        {
            "question": "what was the service with the most cost during this period",
            "expected": "Bedrock: $181.94",
        },
        {
            "question": "what was the product with the most cost during this period",
            "expected": "Innovation Lab Projects: $1,299.50",
        },
        {
            "question": "what was the division with the most cost during this period",
            "expected": "Data Engineering: $115.08",
        },
        {
            "question": "which day had the highest cost during this period",
            "expected": "The highest-cost day during the selected period was:",
        },
    ]

    for test in tests:
        run_test(
            question=test["question"],
            expected_substring=test["expected"],
            selection=selection,
        )

    print("\nALL TESTS PASSED.")


if __name__ == "__main__":
    main()