
from __future__ import annotations

from cfo_finops_athena_rag_final import run_cfo_evaluation_question

import csv
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FALLBACK_TEXT = "I don't have enough information in the retrieved documents."


@dataclass
class EvalCase:
    case_id: str
    category: str  # analytical | rag | hybrid
    question: str
    expected_behavior: str  # answer | fallback
    expected_keywords: List[str]
    period_type: str  # days | weeks | months
    period_value: str
    period_label: str


@dataclass
class EvalResult:
    case_id: str
    category: str
    period_type: str
    period_value: str
    period_label: str
    question: str
    expected_behavior: str
    passed: bool
    score: float
    latency_seconds: float
    notes: str
    answer_preview: str


TEST_CASES: List[EvalCase] = [
    # ANALYTICAL — DAYS
    EvalCase("A01", "analytical", "What is the total actual cost for the selected period?", "answer", ["actual", "cost"], "days", "week_2_monday", "1 day — Week 2 Monday"),
    EvalCase("A02", "analytical", "Which product has the highest actual cost?", "answer", ["product", "highest"], "days", "week_2_monday_to_wednesday", "3 days — Week 2 Monday to Wednesday"),
    EvalCase("A03", "analytical", "Which division has the highest actual cost?", "answer", ["division", "highest"], "days", "week_3_monday_to_saturday", "6 days — Week 3 Monday to Saturday"),
    EvalCase("A04", "analytical", "Which product is most over budget?", "answer", ["product", "budget"], "days", "week_4_tuesday_to_thursday", "3 days — Week 4 Tuesday to Thursday"),
    EvalCase("A05", "analytical", "Which division is most over budget?", "answer", ["division", "budget"], "days", "week_5_monday", "1 day — Week 5 Monday"),

    # ANALYTICAL — WEEKS
    EvalCase("A06", "analytical", "What is the total allocated budget?", "answer", ["allocated", "budget"], "weeks", "week_1", "1 week — Week 1"),
    EvalCase("A07", "analytical", "What is the total variance in dollars?", "answer", ["variance"], "weeks", "weeks_2_to_3", "2 weeks — Weeks 2 to 3"),
    EvalCase("A08", "analytical", "What is the variance percentage?", "answer", ["variance", "percent"], "weeks", "weeks_5_to_7", "3 weeks — Weeks 5 to 7"),
    EvalCase("A09", "analytical", "Rank products by actual cost.", "answer", ["product", "rank"], "weeks", "week_8", "1 week — Week 8"),
    EvalCase("A10", "analytical", "Rank divisions by actual cost.", "answer", ["division", "rank"], "weeks", "weeks_9_to_10", "2 weeks — Weeks 9 to 10"),
    EvalCase("A11", "analytical", "Which service drives the highest cost?", "answer", ["service", "cost"], "weeks", "weeks_10_to_12", "3 weeks — Weeks 10 to 12"),

    # ANALYTICAL — MONTHS
    EvalCase("A12", "analytical", "Show actual cost versus budget by product.", "answer", ["actual", "budget", "product"], "months", "A", "Month A — Weeks 1 to 4"),
    EvalCase("A13", "analytical", "Show actual cost versus budget by division.", "answer", ["actual", "budget", "division"], "months", "B", "Month B — Weeks 5 to 8"),
    EvalCase("A14", "analytical", "Which product has the largest negative variance?", "answer", ["product", "variance"], "months", "C", "Month C — Weeks 9 to 12"),
    EvalCase("A15", "analytical", "Which division has the largest negative variance?", "answer", ["division", "variance"], "months", "AB", "Months A+B — Weeks 1 to 8"),
    EvalCase("A16", "analytical", "What is the average daily actual cost?", "answer", ["average", "daily"], "months", "BC", "Months B+C — Weeks 5 to 12"),
    EvalCase("A17", "analytical", "What is the average weekly actual cost?", "answer", ["average", "weekly"], "months", "ABC", "Months A+B+C — Full 12 weeks"),
    EvalCase("A18", "analytical", "Which product is under budget?", "answer", ["product", "budget"], "months", "A", "Month A — Weeks 1 to 4"),
    EvalCase("A19", "analytical", "Which division is under budget?", "answer", ["division", "budget"], "months", "B", "Month B — Weeks 5 to 8"),
    EvalCase("A20", "analytical", "What is the cost distribution by service?", "answer", ["service", "distribution"], "months", "C", "Month C — Weeks 9 to 12"),
    EvalCase("A21", "analytical", "Identify the top 3 cost drivers.", "answer", ["top", "cost", "drivers"], "months", "ABC", "Months A+B+C — Full 12 weeks"),
    EvalCase("A22", "analytical", "What is the selected period being analyzed?", "answer", ["period"], "weeks", "weeks_2_to_3", "2 weeks — Weeks 2 to 3"),
    EvalCase("A23", "analytical", "Compare product spend across the selected period.", "answer", ["product", "spend"], "months", "AB", "Months A+B — Weeks 1 to 8"),
    EvalCase("A24", "analytical", "Compare division spend across the selected period.", "answer", ["division", "spend"], "months", "BC", "Months B+C — Weeks 5 to 12"),
    EvalCase("A25", "analytical", "Which entity should the CFO review first based on variance?", "answer", ["CFO", "variance"], "months", "ABC", "Months A+B+C — Full 12 weeks"),
    EvalCase("A26", "analytical", "What is the budget overage amount?", "answer", ["budget", "overage"], "weeks", "weeks_5_to_7", "3 weeks — Weeks 5 to 7"),
    EvalCase("A27", "analytical", "What percentage of spend comes from the top product?", "answer", ["percentage", "product"], "days", "week_3_monday_to_saturday", "6 days — Week 3 Monday to Saturday"),
    EvalCase("A28", "analytical", "What percentage of spend comes from the top division?", "answer", ["percentage", "division"], "weeks", "weeks_10_to_12", "3 weeks — Weeks 10 to 12"),
    EvalCase("A29", "analytical", "Summarize the financial performance of the period.", "answer", ["financial", "performance"], "months", "ABC", "Months A+B+C — Full 12 weeks"),
    EvalCase("A30", "analytical", "Give the CFO the most important numerical insight.", "answer", ["CFO", "insight"], "months", "ABC", "Months A+B+C — Full 12 weeks"),

    # RAG — 10
    EvalCase("R01", "rag", "What are FinOps best practices for managing budget variance?", "answer", ["FinOps", "budget"], "months", "A", "Month A — Weeks 1 to 4"),
    EvalCase("R02", "rag", "How should a CFO interpret persistent cloud overspending?", "answer", ["CFO", "overspending"], "months", "B", "Month B — Weeks 5 to 8"),
    EvalCase("R03", "rag", "What should we do when cloud tagging is incomplete?", "answer", ["tagging"], "weeks", "week_8", "1 week — Week 8"),
    EvalCase("R04", "rag", "How can cloud cost accountability be improved?", "answer", ["accountability"], "weeks", "weeks_9_to_10", "2 weeks — Weeks 9 to 10"),
    EvalCase("R05", "rag", "What are best practices for cloud cost allocation?", "answer", ["cost", "allocation"], "months", "C", "Month C — Weeks 9 to 12"),
    EvalCase("R06", "rag", "What should a FinOps team do when spend anomalies appear?", "answer", ["FinOps", "anomalies"], "days", "week_4_tuesday_to_thursday", "3 days — Week 4 Tuesday to Thursday"),
    EvalCase("R07", "rag", "How should teams respond to unused cloud resources?", "answer", ["unused", "resources"], "months", "AB", "Months A+B — Weeks 1 to 8"),
    EvalCase("R08", "rag", "What governance practices help reduce cloud waste?", "answer", ["governance", "waste"], "months", "BC", "Months B+C — Weeks 5 to 12"),
    EvalCase("R09", "rag", "What will AWS prices be next year?", "fallback", [], "months", "ABC", "Months A+B+C — Full 12 weeks"),
    EvalCase("R10", "rag", "What is our competitor's exact monthly AWS bill?", "fallback", [], "weeks", "weeks_2_to_3", "2 weeks — Weeks 2 to 3"),

    # HYBRID — 10
    EvalCase("H01", "hybrid", "Why is the highest-cost product important from a FinOps perspective?", "answer", ["product", "FinOps"], "days", "week_2_monday_to_wednesday", "3 days — Week 2 Monday to Wednesday"),
    EvalCase("H02", "hybrid", "What should the CFO prioritize based on the current variance?", "answer", ["CFO", "variance"], "weeks", "weeks_5_to_7", "3 weeks — Weeks 5 to 7"),
    EvalCase("H03", "hybrid", "Explain the biggest cost driver and recommend actions.", "answer", ["cost", "recommend"], "months", "A", "Month A — Weeks 1 to 4"),
    EvalCase("H04", "hybrid", "Which over-budget area should be investigated first and why?", "answer", ["budget", "investigate"], "months", "B", "Month B — Weeks 5 to 8"),
    EvalCase("H05", "hybrid", "Give a CFO-level explanation of the main cloud cost issue.", "answer", ["CFO", "cloud", "cost"], "months", "C", "Month C — Weeks 9 to 12"),
    EvalCase("H06", "hybrid", "Based on the numbers, what FinOps control should be strengthened?", "answer", ["FinOps", "control"], "months", "AB", "Months A+B — Weeks 1 to 8"),
    EvalCase("H07", "hybrid", "Which team should be accountable for the largest variance?", "answer", ["accountable", "variance"], "months", "BC", "Months B+C — Weeks 5 to 12"),
    EvalCase("H08", "hybrid", "Summarize the financial risk and the operational recommendation.", "answer", ["risk", "recommendation"], "months", "ABC", "Months A+B+C — Full 12 weeks"),
    EvalCase("H09", "hybrid", "Explain whether the selected period suggests budget governance problems.", "answer", ["budget", "governance"], "weeks", "weeks_10_to_12", "3 weeks — Weeks 10 to 12"),
    EvalCase("H10", "hybrid", "Give the CFO a final decision recommendation based on spend and FinOps context.", "answer", ["CFO", "recommendation"], "months", "ABC", "Months A+B+C — Full 12 weeks"),
]


def build_period_context(case: EvalCase) -> Dict[str, str]:
    return {
        "period_type": case.period_type,
        "period_value": case.period_value,
        "period_label": case.period_label,
    }


def safe_import_project_functions() -> Dict[str, Any]:
    functions: Dict[str, Any] = {}

    try:
        from cfo_finops_athena_rag_final import classify_question  # type: ignore
        functions["classify_question"] = classify_question
    except Exception as exc:
        functions["classify_question_error"] = str(exc)

    try:
        from cfo_finops_athena_rag_final import answer_question  # type: ignore
        functions["answer_question"] = answer_question
    except Exception as exc:
        functions["answer_question_error"] = str(exc)

    try:
        from cfo_finops_athena_rag_final import run_cfo_question  # type: ignore
        functions["run_cfo_question"] = run_cfo_question
    except Exception as exc:
        functions["run_cfo_question_error"] = str(exc)

    try:
        from cfo_finops_athena_rag_final import answer_cfo_question  # type: ignore
        functions["answer_cfo_question"] = answer_cfo_question
    except Exception as exc:
        functions["answer_cfo_question_error"] = str(exc)

    return functions


def fallback_local_answer(case: EvalCase) -> str:
    if case.expected_behavior == "fallback":
        return FALLBACK_TEXT

    return (
        "LOCAL_EVAL_PLACEHOLDER: Direct CFO engine function not mapped yet. "
        f"Category={case.category}. "
        f"Period={case.period_label}. "
        f"Question={case.question}"
    )


def try_call_with_period(func: Any, case: EvalCase) -> Optional[str]:
    period_context = build_period_context(case)

    call_patterns = [
        lambda: func(case.question, period_context),
        lambda: func(question=case.question, period_context=period_context),
        lambda: func(question=case.question, period_type=case.period_type, period_value=case.period_value),
        lambda: func(case.question, case.period_type, case.period_value),
        lambda: func(case.question),
    ]

    for pattern in call_patterns:
        try:
            return str(pattern())
        except TypeError:
            continue
        except Exception as exc:
            return f"ERROR_FROM_{getattr(func, '__name__', 'function')}: {exc}"

    return None
   


def score_answer(case: EvalCase, answer: str) -> tuple[bool, float, str]:
    answer_lower = answer.lower()

    if answer.startswith("ERROR_FROM_"):
        return False, 0.0, answer[:250]

    if case.expected_behavior == "fallback":
        fallback_detected = (
            FALLBACK_TEXT.lower() in answer_lower
            or "don't have enough information" in answer_lower
            or "not enough information" in answer_lower
            or "insufficient context" in answer_lower
            or "retrieved documents" in answer_lower
        )

        if fallback_detected:
            return True, 1.0, "Correct fallback triggered."

        return False, 0.0, "Expected fallback, but answer was generated."

    if FALLBACK_TEXT.lower() in answer_lower:
        return False, 0.0, "Unexpected fallback for supported question."

    if answer.startswith("LOCAL_EVAL_PLACEHOLDER"):
        return False, 0.0, "Project answer function not mapped yet."

    if not case.expected_keywords:
        return True, 1.0, "No keyword requirement."

    matched = sum(1 for kw in case.expected_keywords if kw.lower() in answer_lower)
    score = matched / len(case.expected_keywords)

    if score >= 0.5:
        return True, score, f"Matched {matched}/{len(case.expected_keywords)} expected keywords."

    return False, score, f"Only matched {matched}/{len(case.expected_keywords)} expected keywords."

def call_project_answer(case: EvalCase, functions: Dict[str, Any]) -> str:
    try:
        print("\nQUESTION:", case.question)
        print("PERIOD:", case.period_type, case.period_value)

        answer = run_cfo_evaluation_question(
            case.question,
            case.period_type,
            case.period_value
        )

        print("ANSWER:", str(answer)[:300])
        return str(answer)

    except Exception as exc:
        return f"ERROR_FROM_run_cfo_evaluation_question: {exc}"


def run_evaluation() -> List[EvalResult]:
    functions = safe_import_project_functions()

    print("\nCFO FINOPS RAG ASSISTANT — OFFLINE EVALUATION")
    print("=" * 80)
    print("Scope: analytical + RAG + hybrid questions across multiple periods")
    print("Pipeline excluded: Power BI, Amazon Polly, full user interface")
    print("=" * 80)

    if "answer_cfo_question" not in functions and "answer_question" not in functions and "run_cfo_question" not in functions:
        print("\nWARNING: No direct CFO answer function found.")
        print("The suite will run, but cases will fail until we map the correct project function.")
        print("\nImport details:")
        for key, value in functions.items():
            if key.endswith("_error"):
                print(f"- {key}: {value}")

    results: List[EvalResult] = []

    for case in TEST_CASES:
        start = time.perf_counter()
        answer = call_project_answer(case, functions)
        latency = round(time.perf_counter() - start, 4)

        passed, score, notes = score_answer(case, answer)

        result = EvalResult(
            case_id=case.case_id,
            category=case.category,
            period_type=case.period_type,
            period_value=case.period_value,
            period_label=case.period_label,
            question=case.question,
            expected_behavior=case.expected_behavior,
            passed=passed,
            score=round(score, 4),
            latency_seconds=latency,
            notes=notes,
            answer_preview=answer.replace("\n", " ")[:300],
        )
        results.append(result)

        status = "PASS" if passed else "FAIL"
        print(
            f"{status} | {case.case_id} | {case.category} | "
            f"{case.period_label} | score={score:.2f} | {latency:.2f}s"
        )

    return results


def summarize_results(results: List[EvalResult]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    by_category: Dict[str, Dict[str, Any]] = {}
    for category in ["analytical", "rag", "hybrid"]:
        subset = [r for r in results if r.category == category]
        if subset:
            by_category[category] = {
                "total": len(subset),
                "passed": sum(1 for r in subset if r.passed),
                "pass_rate": round(sum(1 for r in subset if r.passed) / len(subset), 4),
                "average_score": round(sum(r.score for r in subset) / len(subset), 4),
            }

    by_period_type: Dict[str, Dict[str, Any]] = {}
    for period_type in ["days", "weeks", "months"]:
        subset = [r for r in results if r.period_type == period_type]
        if subset:
            by_period_type[period_type] = {
                "total": len(subset),
                "passed": sum(1 for r in subset if r.passed),
                "pass_rate": round(sum(1 for r in subset if r.passed) / len(subset), 4),
                "average_score": round(sum(r.score for r in subset) / len(subset), 4),
            }

    fallback_cases = [r for r in results if r.expected_behavior == "fallback"]
    fallback_accuracy = (
        round(sum(1 for r in fallback_cases if r.passed) / len(fallback_cases), 4)
        if fallback_cases
        else None
    )

    hallucination_proxy_cases = [
        r for r in fallback_cases
        if not r.passed
    ]

    return {
        "evaluation_timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_questions": total,
        "passed_questions": passed,
        "overall_pass_rate": round(passed / total, 4) if total else 0,
        "by_category": by_category,
        "by_period_type": by_period_type,
        "fallback_accuracy": fallback_accuracy,
        "hallucination_proxy_rate": round(len(hallucination_proxy_cases) / total, 4) if total else 0,
        "average_latency_seconds": round(sum(r.latency_seconds for r in results) / total, 4) if total else 0,
        "failed_cases": [r.case_id for r in results if not r.passed],
    }


def save_outputs(results: List[EvalResult], summary: Dict[str, Any]) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = OUTPUT_DIR / f"cfo_eval_results_{timestamp}.json"
    csv_path = OUTPUT_DIR / f"cfo_eval_results_{timestamp}.csv"
    summary_path = OUTPUT_DIR / f"cfo_eval_summary_{timestamp}.json"
    latest_summary_path = OUTPUT_DIR / "cfo_eval_summary_latest.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with latest_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    print("\nSaved outputs:")
    print(f"- {json_path}")
    print(f"- {csv_path}")
    print(f"- {summary_path}")
    print(f"- {latest_summary_path}")


def main() -> None:
    results = run_evaluation()
    summary = summarize_results(results)

    print("\nFINAL SUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    save_outputs(results, summary)


if __name__ == "__main__":
    main()
