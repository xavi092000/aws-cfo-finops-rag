from __future__ import annotations

import csv
import json
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cfo_finops_athena_rag_final as pipeline_module
from cfo_finops_athena_rag_final import run_finops_cfo_pipeline


# =========================================================
# OUTPUT CONFIG
# =========================================================
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

NUMBER_OF_RUNS = 3
QUESTIONS_PER_RUN = 6
RANDOM_SEED = None  # None = different random behavior each launch


# =========================================================
# DISABLE AUDIO / AUTO-OPEN FOR TEST MODE
# =========================================================
def disable_audio_side_effects() -> None:
    """
    Prevent audio auto-open behavior during automated tests.
    """
    try:
        if hasattr(pipeline_module, "os") and hasattr(pipeline_module.os, "startfile"):
            pipeline_module.os.startfile = lambda *args, **kwargs: None
    except Exception:
        pass


# =========================================================
# VALID RANDOM SELECTIONS
# =========================================================
VALID_SELECTIONS: List[Dict[str, Any]] = [
    {
        "label": "Weeks 5 to 6",
        "selection": {
            "mode": "weeks",
            "block": "B",
            "weeks": [5, 6],
            "number_of_weeks": 2,
        },
    },
    {
        "label": "Weeks 7 to 8",
        "selection": {
            "mode": "weeks",
            "block": "B",
            "weeks": [7, 8],
            "number_of_weeks": 2,
        },
    },
    {
        "label": "Weeks 9 to 10",
        "selection": {
            "mode": "weeks",
            "block": "C",
            "weeks": [9, 10],
            "number_of_weeks": 2,
        },
    },
    {
        "label": "Month Block B",
        "selection": {
            "mode": "months",
            "monthly_block": "B",
        },
    },
    {
        "label": "Month Block C",
        "selection": {
            "mode": "months",
            "monthly_block": "C",
        },
    },
    {
        "label": "Month Block ABC",
        "selection": {
            "mode": "months",
            "monthly_block": "ABC",
        },
    },
    {
        "label": "Week 1 Wednesday to Friday",
        "selection": {
            "mode": "days",
            "block": "A",
            "week": 1,
            "days": 3,
        },
    },
    {
        "label": "Week 2 Monday to Thursday",
        "selection": {
            "mode": "days",
            "block": "A",
            "week": 2,
            "days": 4,
        },
    },
    {
        "label": "Week 6 Monday to Thursday",
        "selection": {
            "mode": "days",
            "block": "B",
            "week": 6,
            "days": 4,
        },
    },
    {
        "label": "Week 10 Monday to Friday",
        "selection": {
            "mode": "days",
            "block": "C",
            "week": 10,
            "days": 5,
        },
    },
]


# =========================================================
# HYBRID TEST QUESTIONS
# =========================================================
TEST_QUESTIONS: List[str] = [
    "Which product had the highest variance, and what does that suggest from a FinOps perspective?",
    "Which product was the most over budget during the selected period, and what should a CFO prioritize first?",
    "Which division had the highest total cost during the selected period, and what FinOps action fits best?",
    "What were the top 3 over-budget products during the selected period, and what does that imply for cost control?",
    "Which day had the highest total spend during the selected period, and what should leadership review first?",
    "What was the total variance for the selected period, and what does that mean from a FinOps perspective?",
]


# =========================================================
# HELPERS
# =========================================================
def shorten_text(text: str, max_len: int = 500) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def choose_random_selection(rng: random.Random) -> Dict[str, Any]:
    return rng.choice(VALID_SELECTIONS)


def classify_hybrid_status(
    route: str,
    answer: str,
    error_text: str | None,
) -> str:
    if error_text:
        return "FAIL"

    if not answer or not answer.strip():
        return "FAIL"

    if route != "hybrid":
        return "FAIL"

    lowered = answer.lower()

    required_markers = [
        "analytical answer",
        "finops interpretation",
        "cfo priority",
    ]

    missing_markers = [marker for marker in required_markers if marker not in lowered]

    if missing_markers:
        return "PARTIAL"

    if "i don't have enough information in the retrieved documents." in lowered:
        # Hybrid can still be partial if the RAG side falls back,
        # but the route is correct and the analytical portion exists.
        return "PARTIAL"

    return "PASS"


# =========================================================
# SINGLE TEST
# =========================================================
def run_one_test(
    run_id: int,
    selection_wrapper: Dict[str, Any],
    question_number: int,
    question: str,
) -> Dict[str, Any]:
    selection_label = selection_wrapper["label"]
    selection = selection_wrapper["selection"]

    error_text = None
    answer = ""
    route = ""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    traceback_text = ""

    try:
        result = run_finops_cfo_pipeline(
            selection=selection,
            question=question,
            enable_audio=False,
            auto_open_audio=False,
        )

        if isinstance(result, dict):
            answer = str(result.get("answer", "")).strip()
            route = str(result.get("route", "")).strip()
        else:
            answer = str(result).strip()

    except Exception as e:
        error_text = f"{type(e).__name__}: {e}"
        traceback_text = traceback.format_exc()

    status = classify_hybrid_status(route=route, answer=answer, error_text=error_text)

    return {
        "timestamp": timestamp,
        "run_id": run_id,
        "selection_label": selection_label,
        "selection_json": json.dumps(selection, ensure_ascii=False),
        "question_number": question_number,
        "question": question,
        "route": route,
        "status": status,
        "answer_preview": shorten_text(answer),
        "full_answer": answer,
        "error": error_text or "",
        "traceback": traceback_text,
    }


# =========================================================
# RUN SUITE
# =========================================================
def run_full_suite() -> List[Dict[str, Any]]:
    disable_audio_side_effects()

    rng = random.Random(RANDOM_SEED)
    results: List[Dict[str, Any]] = []

    print("\n============================================================")
    print("AUTOMATED HYBRID TEST HARNESS")
    print("============================================================")
    print(f"Number of runs      : {NUMBER_OF_RUNS}")
    print(f"Questions per run   : {QUESTIONS_PER_RUN}")
    print("Audio auto-open     : DISABLED")
    print("============================================================\n")

    for run_id in range(1, NUMBER_OF_RUNS + 1):
        selection_wrapper = choose_random_selection(rng)

        print("------------------------------------------------------------")
        print(f"RUN {run_id}")
        print(f"Random selection: {selection_wrapper['label']}")
        print(json.dumps(selection_wrapper["selection"], indent=2))
        print("------------------------------------------------------------")

        for question_number, question in enumerate(TEST_QUESTIONS[:QUESTIONS_PER_RUN], start=1):
            print(f"[Run {run_id} | Q{question_number}] {question}")

            row = run_one_test(
                run_id=run_id,
                selection_wrapper=selection_wrapper,
                question_number=question_number,
                question=question,
            )
            results.append(row)

            print(f"Status: {row['status']}")
            if row["route"]:
                print(f"Route: {row['route']}")
            if row["error"]:
                print(f"Error: {row['error']}")
            else:
                print(f"Answer preview: {row['answer_preview']}")
            print("-" * 60)

    return results


# =========================================================
# SAVE RESULTS
# =========================================================
def save_results(results: List[Dict[str, Any]]) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = OUTPUT_DIR / f"hybrid_test_results_{timestamp}.csv"
    json_path = OUTPUT_DIR / f"hybrid_test_results_{timestamp}.json"

    fieldnames = [
        "timestamp",
        "run_id",
        "selection_label",
        "selection_json",
        "question_number",
        "question",
        "route",
        "status",
        "answer_preview",
        "full_answer",
        "error",
        "traceback",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return csv_path, json_path


# =========================================================
# SUMMARY
# =========================================================
def print_summary(results: List[Dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    partial = sum(1 for r in results if r["status"] == "PARTIAL")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    print("\n============================================================")
    print("FINAL HYBRID TEST SUMMARY")
    print("============================================================")
    print(f"Total tests : {total}")
    print(f"PASS        : {passed}")
    print(f"PARTIAL     : {partial}")
    print(f"FAIL        : {failed}")
    print("============================================================")

    if failed > 0:
        print("\nFAILED TESTS:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"- Run {r['run_id']} | Q{r['question_number']}")
                print(f"  Selection: {r['selection_label']}")
                print(f"  Question : {r['question']}")
                print(f"  Route    : {r['route']}")
                print(f"  Error    : {r['error'] or 'Hybrid structure not respected'}")
                print("")

    if partial > 0:
        print("\nPARTIAL TESTS:")
        for r in results:
            if r["status"] == "PARTIAL":
                print(f"- Run {r['run_id']} | Q{r['question_number']}")
                print(f"  Selection: {r['selection_label']}")
                print(f"  Question : {r['question']}")
                print(f"  Route    : {r['route']}")
                print(f"  Preview  : {r['answer_preview']}")
                print("")

    print(f"CSV report  : {csv_path}")
    print(f"JSON report : {json_path}")
    print("============================================================\n")


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    results = run_full_suite()
    csv_path, json_path = save_results(results)
    print_summary(results, csv_path, json_path)


if __name__ == "__main__":
    main()