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
QUESTIONS_PER_RUN = 10
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
# 10 TEST QUESTIONS
# =========================================================
TEST_QUESTIONS: List[str] = [
    "What was the total actual cost for the selected period?",
    "What was the total budget for the selected period?",
    "What was the total variance for the selected period?",
    "Which product had the highest total cost during the selected period?",
    "Which division had the highest total cost during the selected period?",
    "Which product was the most over budget during the selected period?",
    "Which division was the most under budget during the selected period?",
    "What were the top 3 over-budget products during the selected period?",
    "Which day had the highest total spend during the selected period?",
    "Which product had the highest variance, and what does that suggest from a FinOps perspective?",
]


# =========================================================
# HELPERS
# =========================================================
def shorten_text(text: str, max_len: int = 400) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def is_documentation_gap(answer: str) -> bool:
    """
    Detect whether the answer indicates that the knowledge base
    does not contain enough information to respond.
    """
    lowered = (answer or "").lower()

    documentation_gap_markers = [
        "i don't have enough information",
        "not enough information",
        "could not find this information",
        "documentation gap",
        "knowledge base gap",
    ]

    return any(marker in lowered for marker in documentation_gap_markers)


def is_probably_analytical_question(question: str) -> bool:
    """
    Heuristic to detect questions that should normally route
    to analytical processing rather than general RAG.
    """
    lowered = (question or "").lower()

    analytical_markers = [
        "total",
        "actual cost",
        "budget",
        "variance",
        "highest",
        "lowest",
        "top",
        "over budget",
        "under budget",
        "selected period",
        "during the selected period",
        "which product",
        "which division",
        "which day",
        "total spend",
    ]

    return any(marker in lowered for marker in analytical_markers)


def classify_test_status(answer: str, error_text: str | None) -> str:
    if error_text:
        return "FAIL"

    if not answer or not answer.strip():
        return "FAIL"

    lowered = answer.lower()

    partial_markers = [
        "i don't have enough information",
        "no rows matched",
        "does not match a specific granular pattern yet",
        "i classified the question as analytical, but it does not match",
        "could not",
        "data_not_available",
        "documentation gap",
    ]

    if any(marker in lowered for marker in partial_markers):
        return "PARTIAL"

    return "PASS"


def choose_random_selection(rng: random.Random) -> Dict[str, Any]:
    return rng.choice(VALID_SELECTIONS)


# =========================================================
# SINGLE TEST
# =========================================================
def run_one_test(run_id: int, selection_wrapper: Dict[str, Any], question_number: int, question: str) -> Dict[str, Any]:
    selection_label = selection_wrapper["label"]
    selection = selection_wrapper["selection"]

    error_text = None
    answer = ""
    route = ""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    traceback_text = ""

    try:
        result = run_finops_cfo_pipeline(selection, question)

        if isinstance(result, dict):
            answer = str(result.get("answer", "")).strip()
            route = str(result.get("route", "")).strip()
        else:
            answer = str(result).strip()

    except Exception as e:
        error_text = f"{type(e).__name__}: {e}"
        traceback_text = traceback.format_exc()

    status = classify_test_status(answer, error_text)

    documentation_gap = (
        is_documentation_gap(answer)
        and route == "general"
        and not is_probably_analytical_question(question)
    )

    routing_error = (
        route == "general"
        and is_probably_analytical_question(question)
        and is_documentation_gap(answer)
    )

    return {
        "timestamp": timestamp,
        "run_id": run_id,
        "selection_label": selection_label,
        "selection_json": json.dumps(selection, ensure_ascii=False),
        "question_number": question_number,
        "question": question,
        "route": route,
        "status": status,
        "documentation_gap": documentation_gap,
        "routing_error": routing_error,
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
    print("AUTOMATED ANALYTICS TEST HARNESS")
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
            if row["documentation_gap"]:
                print("Documentation gap detected: YES")
            if row["routing_error"]:
                print("Routing error detected    : YES")
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

    csv_path = OUTPUT_DIR / f"analytics_test_results_{timestamp}.csv"
    json_path = OUTPUT_DIR / f"analytics_test_results_{timestamp}.json"

    fieldnames = [
        "timestamp",
        "run_id",
        "selection_label",
        "selection_json",
        "question_number",
        "question",
        "route",
        "status",
        "documentation_gap",
        "routing_error",
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
    doc_gaps = sum(1 for r in results if r.get("documentation_gap"))
    routing_errors = sum(1 for r in results if r.get("routing_error"))

    print("\n============================================================")
    print("FINAL TEST SUMMARY")
    print("============================================================")
    print(f"Total tests         : {total}")
    print(f"PASS                : {passed}")
    print(f"PARTIAL             : {partial}")
    print(f"FAIL                : {failed}")
    print(f"DOCUMENTATION GAPS  : {doc_gaps}")
    print(f"ROUTING ERRORS      : {routing_errors}")
    print("============================================================")

    if failed > 0:
        print("\nFAILED TESTS:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"- Run {r['run_id']} | Q{r['question_number']}")
                print(f"  Selection: {r['selection_label']}")
                print(f"  Question : {r['question']}")
                print(f"  Error    : {r['error']}")
                print("")

    if partial > 0:
        print("\nPARTIAL TESTS:")
        for r in results:
            if r["status"] == "PARTIAL":
                print(f"- Run {r['run_id']} | Q{r['question_number']}")
                print(f"  Selection          : {r['selection_label']}")
                print(f"  Question           : {r['question']}")
                print(f"  Documentation gap  : {r['documentation_gap']}")
                print(f"  Routing error      : {r['routing_error']}")
                print(f"  Preview            : {r['answer_preview']}")
                print("")

    if doc_gaps > 0:
        print("\nDOCUMENTATION GAP LOG:")
        for r in results:
            if r.get("documentation_gap"):
                print(f"- Run {r['run_id']} | Q{r['question_number']}")
                print(f"  Selection: {r['selection_label']}")
                print(f"  Question : {r['question']}")
                print("")

    if routing_errors > 0:
        print("\nROUTING ERROR LOG:")
        for r in results:
            if r.get("routing_error"):
                print(f"- Run {r['run_id']} | Q{r['question_number']}")
                print(f"  Selection: {r['selection_label']}")
                print(f"  Question : {r['question']}")
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