from __future__ import annotations

from cfo_finops_athena_rag_final import run_finops_cfo_pipeline


def run_test(question: str, expected_prefix: str, selection: dict, label: str) -> None:
    result = run_finops_cfo_pipeline(
        selection=selection,
        question=question,
        enable_audio=False,
        auto_open_audio=False,
    )

    answer = result["answer"]

    print("\n==================================================")
    print(f"TEST LABEL: {label}")
    print(f"QUESTION: {question}")
    print("ANSWER:")
    print(answer)
    print("==================================================")

    assert expected_prefix in answer, (
        f"Test failed.\n"
        f"Expected to find: {expected_prefix}\n"
        f"Actual answer:\n{answer}"
    )


def main() -> None:
    weeks_selection = {
        "mode": "weeks",
        "block": "B",
        "weeks": [5, 6],
        "number_of_weeks": 2,
    }

    months_selection = {
        "mode": "months",
        "monthly_block": "BC",
    }

    tests = [
        {
            "label": "WEEKS - SERVICE",
            "selection": weeks_selection,
            "question": "what was the service with the most cost during this period",
            "expected": "The highest-cost service during the selected period was:",
        },
        {
            "label": "WEEKS - PRODUCT",
            "selection": weeks_selection,
            "question": "what was the product with the most cost during this period",
            "expected": "The highest-cost product during the selected period was:",
        },
        {
            "label": "WEEKS - DIVISION",
            "selection": weeks_selection,
            "question": "what was the division with the most cost during this period",
            "expected": "The highest-cost division during the selected period was:",
        },
        {
            "label": "MONTHS - SERVICE",
            "selection": months_selection,
            "question": "what was the service with the most cost during this period",
            "expected": "The highest-cost service during the selected period was:",
        },
        {
            "label": "MONTHS - PRODUCT",
            "selection": months_selection,
            "question": "what was the product with the most cost during this period",
            "expected": "The highest-cost product during the selected period was:",
        },
        {
            "label": "MONTHS - DIVISION",
            "selection": months_selection,
            "question": "what was the division with the most cost during this period",
            "expected": "The highest-cost division during the selected period was:",
        },
    ]

    for test in tests:
        run_test(
            question=test["question"],
            expected_prefix=test["expected"],
            selection=test["selection"],
            label=test["label"],
        )

    print("\nALL PERIOD TESTS PASSED.")


if __name__ == "__main__":
    main()