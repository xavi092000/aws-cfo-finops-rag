from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

# =========================================================
# FILES / CONFIG
# =========================================================
PRODUCTS_FILE = Path("products_finops_analysis.csv")
INTERNAL_FILE = Path("internal_finops_analysis.csv")

INPUT_FILE = Path("data/chunks_general_questions.json")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

AWS_PROFILE = "terraform-runner"
AWS_REGION = "us-east-1"
VOICE_ID = "Joanna"
ENGINE = "neural"

SYSTEM_PROMPT = """
You are a FinOps AI assistant answering CFO-level questions using Retrieval-Augmented Generation (RAG).

STRICT RULES:
1. Only use the provided retrieved context
2. Do not use outside knowledge
3. Do not invent facts, numbers, explanations, or examples
4. Do not guess
5. Do not infer beyond what is explicitly supported by the retrieved context
6. Do not mention specific cloud services, platforms, tools, products, numbers, percentages, or examples unless they appear explicitly in the retrieved context
7. If the answer is not clearly supported by the retrieved context, respond exactly with:
   "I don't have enough information in the retrieved documents."
8. If rule 7 applies, STOP immediately and do not add any extra explanation, bullets, recommendations, or priorities
9. If only part of the question is supported, answer only the supported part and clearly state what is missing
10. Stay grounded in the retrieved documents only

ANSWER STYLE:
- Be concise
- Be structured
- Be executive-friendly
- Focus on factual FinOps guidance
- Do not add speculation

OUTPUT FORMAT:
1. Short executive answer
2. 3 to 5 key best practices
3. What a CFO should prioritize first
"""

selection = {
    "mode": "weeks",
    "weeks": [5, 6]
}

# =========================================================
# CSV HELPERS
# =========================================================
def load_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";")
        if len(df.columns) == 1:
            df = pd.read_csv(path)
        return df
    except Exception:
        return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
    return df


def clean_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("\u00A0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )


def find_first_existing_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    normalized = {col.lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


def safe_variance_pct(actual: float, budget: float) -> float:
    if budget == 0:
        return 0.0
    return ((actual - budget) / budget) * 100.0


def format_currency(value: float, force_sign: bool = False) -> str:
    if force_sign:
        sign = "+" if value > 0 else ""
        return f"{sign}${value:,.2f}"
    return f"${value:,.2f}"


def format_percent(value: float, force_sign: bool = False) -> str:
    if force_sign:
        sign = "+" if value > 0 else ""
        return f"{sign}{value:,.2f}%"
    return f"{value:,.2f}%"


def build_status_text(variance_value: float) -> str:
    if variance_value > 0:
        return "Over Budget"
    if variance_value < 0:
        return "Under Budget"
    return "On Budget"


# =========================================================
# FILTER BY SELECTED PERIOD
# =========================================================
def filter_by_selection(df: pd.DataFrame, selection: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()

    week_col = find_first_existing_column(df, ["week", "Week"])
    day_col = find_first_existing_column(df, ["day", "Day"])
    month_col = find_first_existing_column(df, ["month", "Month"])

    if selection["mode"] == "weeks":
        week_labels = [f"Week {w}" for w in selection["weeks"]]
        return df[df[week_col].isin(week_labels)].copy()

    if selection["mode"] == "days":
        week_label = f"Week {selection['week']}"
        return df[
            (df[week_col] == week_label) &
            (df[day_col].isin(selection["day_labels"]))
        ].copy()

    if selection["mode"] == "months":
        month_labels = selection["month_labels"]
        return df[df[month_col].isin(month_labels)].copy()

    raise ValueError(f"Unsupported mode: {selection['mode']}")


# =========================================================
# STANDARDIZE DATA FOR GRANULAR ANALYTICS
# =========================================================
def prepare_standardized_df(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    df = df.copy()

    actual_col = find_first_existing_column(
        df,
        ["Actual Cost", "Actual Cost Computed", "Actual Cost ($)"]
    )
    budget_col = find_first_existing_column(
        df,
        ["Allocated Budget ($)", "Allocated Budget Computed", "Allocated Budget"]
    )
    variance_col = find_first_existing_column(
        df,
        ["Variance Recomputed", "Variance ($)", "Variance"]
    )

    week_col = find_first_existing_column(df, ["week", "Week"])
    day_col = find_first_existing_column(df, ["day", "Day"])
    month_col = find_first_existing_column(df, ["month", "Month"])

    if actual_col is None:
        raise ValueError(f"Could not find actual cost column. Found columns: {list(df.columns)}")
    if budget_col is None:
        raise ValueError(f"Could not find budget column. Found columns: {list(df.columns)}")
    if variance_col is None:
        raise ValueError(f"Could not find variance column. Found columns: {list(df.columns)}")

    df[actual_col] = clean_numeric(df[actual_col])
    df[budget_col] = clean_numeric(df[budget_col])
    df[variance_col] = clean_numeric(df[variance_col])

    standardized = pd.DataFrame(index=df.index)
    standardized["Dataset"] = [dataset_label] * len(df)

    if month_col is not None:
        standardized["Month"] = df[month_col].values
    else:
        standardized["Month"] = ["DATA_NOT_AVAILABLE"] * len(df)

    if week_col is not None:
        standardized["Week"] = df[week_col].values
    else:
        standardized["Week"] = ["DATA_NOT_AVAILABLE"] * len(df)

    if day_col is not None:
        standardized["Day"] = df[day_col].values
    else:
        standardized["Day"] = ["DATA_NOT_AVAILABLE"] * len(df)

    standardized["Actual Cost"] = df[actual_col].values
    standardized["Allocated Budget"] = df[budget_col].values
    standardized["Variance"] = df[variance_col].values
    standardized["Variance %"] = standardized.apply(
        lambda row: safe_variance_pct(row["Actual Cost"], row["Allocated Budget"]),
        axis=1
    )

    return standardized.reset_index(drop=True)


# =========================================================
# GENERAL CFO SUMMARY ON SELECTED PERIOD
# =========================================================
def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    actual_col = find_first_existing_column(
        df,
        ["Actual Cost", "Actual Cost Computed", "Actual Cost ($)"]
    )
    budget_col = find_first_existing_column(
        df,
        ["Allocated Budget ($)", "Allocated Budget Computed", "Allocated Budget"]
    )
    variance_col = find_first_existing_column(
        df,
        ["Variance Recomputed", "Variance ($)", "Variance"]
    )
    week_col = find_first_existing_column(df, ["week", "Week"])
    day_col = find_first_existing_column(df, ["day", "Day"])

    if actual_col is None:
        raise ValueError(f"Could not find actual cost column. Found columns: {list(df.columns)}")
    if budget_col is None:
        raise ValueError(f"Could not find budget column. Found columns: {list(df.columns)}")
    if variance_col is None:
        raise ValueError(f"Could not find variance column. Found columns: {list(df.columns)}")

    df[actual_col] = clean_numeric(df[actual_col])
    df[budget_col] = clean_numeric(df[budget_col])
    df[variance_col] = clean_numeric(df[variance_col])

    grouped = (
        df.groupby([week_col, day_col], dropna=False)[[actual_col, budget_col, variance_col]]
        .sum()
        .reset_index()
    )

    grouped["Variance %"] = grouped.apply(
        lambda row: safe_variance_pct(row[actual_col], row[budget_col]),
        axis=1
    )

    grouped = grouped.rename(
        columns={
            week_col: "Week",
            day_col: "Day",
            actual_col: "Actual Cost",
            budget_col: "Allocated Budget",
            variance_col: "Variance"
        }
    )

    return grouped


def generate_summary(df: pd.DataFrame, label: str) -> str:
    df = df.copy()

    if df.empty:
        return f"""
======================================================================
CFO SUMMARY - {label.upper()}
======================================================================

No rows matched the selected scope.
""".strip()

    daily_summary = build_daily_summary(df)

    total_budget = daily_summary["Allocated Budget"].sum()
    total_actual = daily_summary["Actual Cost"].sum()
    total_variance = daily_summary["Variance"].sum()
    total_variance_pct = safe_variance_pct(total_actual, total_budget)
    status = build_status_text(total_variance)

    lines: List[str] = []
    lines.append("======================================================================")
    lines.append(f"CFO SUMMARY - {label.upper()}")
    lines.append("======================================================================")
    lines.append("")
    lines.append(f"Total Budget: {format_currency(total_budget)}")
    lines.append(f"Total Actual Cost: {format_currency(total_actual)}")
    lines.append(f"Variance: {format_currency(total_variance, force_sign=True)} ({format_percent(total_variance_pct, force_sign=True)})")
    lines.append(f"Status: {status}")

    return "\n".join(lines).rstrip()


# =========================================================
# ANALYTICAL QUESTION ROUTER
# =========================================================
ANALYTICAL_KEYWORDS = [
    "total", "sum", "average", "avg", "highest", "lowest", "most", "least",
    "top", "bottom", "variance", "budget", "actual", "cost", "spend",
    "over budget", "under budget", "compare", "comparison", "versus", "vs",
    "percent", "percentage", "day", "week", "month", "product", "internal",
    "breakdown", "which", "show me the numbers", "calculate", "why", "explain"
]

GENERAL_KEYWORDS = [
    "what is finops", "explain", "best practice", "best practices",
    "recommend", "recommendation", "recommendations", "optimize", "optimization",
    "governance", "waste", "rightsizing", "showback", "chargeback",
    "allocation", "tagging", "what should we do", "how can we improve",
    "why does this matter", "what does this mean"
]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def classify_question(question: str) -> str:
    text = normalize_text(question)

    analytical_hits = sum(1 for keyword in ANALYTICAL_KEYWORDS if keyword in text)
    general_hits = sum(1 for keyword in GENERAL_KEYWORDS if keyword in text)

    if analytical_hits > 0 and general_hits > 0:
        return "hybrid"
    if analytical_hits > 0:
        return "analytical"
    return "general"


def combine_filtered_data(products_filtered: pd.DataFrame, internal_filtered: pd.DataFrame) -> pd.DataFrame:
    products_std = prepare_standardized_df(products_filtered, "Products")
    internal_std = prepare_standardized_df(internal_filtered, "Internal")
    return pd.concat([products_std, internal_std], ignore_index=True)


def extract_top_n(question: str, default: int = 3) -> int:
    match = re.search(r"\btop\s+(\d+)\b", question.lower())
    if match:
        return int(match.group(1))
    match = re.search(r"\bbottom\s+(\d+)\b", question.lower())
    if match:
        return int(match.group(1))
    return default


def detect_metric(question: str) -> str:
    q = question.lower()

    if "variance %" in q or "variance percent" in q or "percentage" in q or "percent" in q:
        return "Variance %"
    if "budget" in q and "actual" not in q and "variance" not in q:
        return "Allocated Budget"
    if "actual" in q or "actual cost" in q or "spend" in q or "cost" in q:
        return "Actual Cost"
    if "variance" in q or "over budget" in q or "under budget" in q:
        return "Variance"

    return "Actual Cost"


def detect_dimension(question: str) -> str:
    q = question.lower()

    if "day" in q or "daily" in q:
        return "Day"
    if "week" in q or "weekly" in q:
        return "Week"
    if "month" in q or "monthly" in q:
        return "Month"
    return "Dataset"


def filter_question_scope(df: pd.DataFrame, question: str) -> pd.DataFrame:
    q = question.lower().strip()
    scoped = df.copy()

    scoped["Dataset"] = scoped["Dataset"].astype(str).str.strip().str.lower()

    if "product" in q and "internal" not in q:
        scoped = scoped[scoped["Dataset"] == "products"].copy()
    elif "internal" in q and "product" not in q:
        scoped = scoped[scoped["Dataset"] == "internal"].copy()

    return scoped


def aggregate_by_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    grouped = (
        df.groupby(dimension, dropna=False)[["Actual Cost", "Allocated Budget", "Variance"]]
        .sum()
        .reset_index()
    )

    grouped["Variance %"] = grouped.apply(
        lambda row: safe_variance_pct(row["Actual Cost"], row["Allocated Budget"]),
        axis=1
    )

    return grouped


def answer_total_question(df: pd.DataFrame, metric: str, scope_label: str) -> str:
    total_value = df[metric].sum()

    if metric == "Variance %":
        total_value = safe_variance_pct(df["Actual Cost"].sum(), df["Allocated Budget"].sum())
        return f"Within the selected period, the total variance percentage for {scope_label} is {format_percent(total_value, force_sign=True)}."

    return f"Within the selected period, the total {metric.lower()} for {scope_label} is {format_currency(total_value, force_sign=(metric == 'Variance'))}."


def answer_average_question(df: pd.DataFrame, metric: str, scope_label: str) -> str:
    if metric == "Variance %":
        value = df["Variance %"].mean()
        return f"Within the selected period, the average variance percentage for {scope_label} is {format_percent(value, force_sign=True)}."

    value = df[metric].mean()
    return f"Within the selected period, the average {metric.lower()} for {scope_label} is {format_currency(value, force_sign=(metric == 'Variance'))}."


def answer_why_budget_question(df: pd.DataFrame, scope_label: str) -> str:
    total_actual = df["Actual Cost"].sum()
    total_budget = df["Allocated Budget"].sum()
    total_variance = df["Variance"].sum()
    total_variance_pct = safe_variance_pct(total_actual, total_budget)
    status = build_status_text(total_variance)

    lines = [
        f"Within the selected period, {scope_label} is {status.lower()}.",
        f"- Total Actual Cost: {format_currency(total_actual)}",
        f"- Total Allocated Budget: {format_currency(total_budget)}",
        f"- Total Variance: {format_currency(total_variance, force_sign=True)}",
        f"- Variance %: {format_percent(total_variance_pct, force_sign=True)}",
    ]

    if total_variance > 0:
        lines.append(
            "This indicates that actual spending exceeded the allocated budget during the selected period."
        )
    elif total_variance < 0:
        lines.append(
            "This indicates that actual spending remained below the allocated budget during the selected period."
        )
    else:
        lines.append(
            "This indicates that actual spending matched the allocated budget during the selected period."
        )

    return "\n".join(lines)


def answer_top_bottom_question(df: pd.DataFrame, question: str, metric: str, dimension: str, scope_label: str) -> str:
    grouped = aggregate_by_dimension(df, dimension)
    n = extract_top_n(question, default=3)
    q = question.lower()

    if "under budget" in q:
        ranked = grouped.sort_values("Variance", ascending=True).head(n)
        label = f"top {n} most under budget"
    elif "bottom" in q or "lowest" in q or "least" in q:
        ranked = grouped.sort_values(metric, ascending=True).head(n)
        label = f"bottom {n}"
    else:
        ranked = grouped.sort_values(metric, ascending=False).head(n)
        label = f"top {n}"

    lines = [f"Within the selected period, here is the {label} breakdown for {scope_label} by {dimension.lower()}:"]

    for _, row in ranked.iterrows():
        if metric == "Variance %":
            metric_text = format_percent(row[metric], force_sign=True)
        else:
            metric_text = format_currency(row[metric], force_sign=(metric == "Variance"))

        lines.append(
            f"- {row[dimension]}: {metric_text} "
            f"(Actual: {format_currency(row['Actual Cost'])}, "
            f"Budget: {format_currency(row['Allocated Budget'])}, "
            f"Variance: {format_currency(row['Variance'], force_sign=True)}, "
            f"Variance %: {format_percent(row['Variance %'], force_sign=True)})"
        )

    return "\n".join(lines)


def answer_compare_question(df: pd.DataFrame, metric: str) -> str:
    grouped = aggregate_by_dimension(df, "Dataset")

    if len(grouped) < 2:
        return "I cannot compare Products and Internal because only one dataset is available within the selected period."

    grouped["Dataset"] = grouped["Dataset"].astype(str).str.strip().str.lower()

    products_row = grouped[grouped["Dataset"] == "products"]
    internal_row = grouped[grouped["Dataset"] == "internal"]

    if products_row.empty or internal_row.empty:
        return "I cannot compare Products and Internal because one side has no data within the selected period."

    p = products_row.iloc[0]
    i = internal_row.iloc[0]

    if metric == "Variance %":
        p_metric_text = format_percent(p[metric], force_sign=True)
        i_metric_text = format_percent(i[metric], force_sign=True)
    else:
        p_metric_text = format_currency(p[metric], force_sign=(metric == "Variance"))
        i_metric_text = format_currency(i[metric], force_sign=(metric == "Variance"))

    lines = [
        "Within the selected period, here is the comparison between Products and Internal:",
        f"- Products {metric}: {p_metric_text}",
        f"- Internal {metric}: {i_metric_text}",
    ]

    additional_metrics = ["Actual Cost", "Allocated Budget", "Variance", "Variance %"]
    additional_metrics = [m for m in additional_metrics if m != metric]

    for extra_metric in additional_metrics:
        if extra_metric == "Variance %":
            p_text = format_percent(p[extra_metric], force_sign=True)
            i_text = format_percent(i[extra_metric], force_sign=True)
        else:
            p_text = format_currency(p[extra_metric], force_sign=(extra_metric == "Variance"))
            i_text = format_currency(i[extra_metric], force_sign=(extra_metric == "Variance"))

        lines.append(f"- Products {extra_metric}: {p_text}")
        lines.append(f"- Internal {extra_metric}: {i_text}")

    return "\n".join(lines)


def build_scope_label(question: str) -> str:
    q = question.lower()
    if "product" in q and "internal" not in q:
        return "Products"
    if "internal" in q and "product" not in q:
        return "Internal"
    return "Products and Internal combined"


def is_why_budget_question(question: str) -> bool:
    q = question.lower()

    why_signals = [
        "why",
        "what caused",
        "what drove",
        "what explains",
        "explain",
        "reason"
    ]

    budget_signals = [
        "over budget",
        "under budget",
        "variance",
        "budget"
    ]

    return any(signal in q for signal in why_signals) and any(signal in q for signal in budget_signals)


def build_hybrid_finops_query(question: str, analytical_answer: str) -> str:
    q = question.lower()

    if is_why_budget_question(question):
        if "product" in q and "internal" not in q:
            return (
                "What FinOps best practices should a CFO prioritize when Products are over budget "
                "during a selected period, including budget variance review, cost allocation validation, "
                "anomaly investigation, and accountability?"
            )
        if "internal" in q and "product" not in q:
            return (
                "What FinOps best practices should a CFO prioritize when Internal spend is over budget "
                "during a selected period, including budget variance review, cost allocation validation, "
                "anomaly investigation, and accountability?"
            )
        return (
            "What FinOps best practices should a CFO prioritize when actual spend exceeds budget "
            "during a selected period, including variance review, cost allocation validation, "
            "anomaly investigation, and accountability?"
        )

    if "compare" in q or "vs" in q or "versus" in q:
        return (
            "What FinOps best practices should a CFO use when comparing spend, budget, and variance "
            "between two cost scopes during a selected period?"
        )

    if "average" in q or "avg" in q:
        return (
            "How should a CFO interpret average cost or average variance over a selected period in FinOps?"
        )

    if "total" in q or "sum" in q:
        return (
            "How should a CFO interpret total spend, budget, and variance over a selected period in FinOps?"
        )

    return (
        "What FinOps best practices are most relevant to interpret a selected-period budget and spend result?"
    )


def build_contextual_finops_fallback(question: str, analytical_answer: str) -> str:
    q = question.lower()

    if is_why_budget_question(question):
        return (
            "The retrieved documents do not provide enough specific FinOps guidance for this exact case. "
            "Based on the analytical result, the immediate CFO priority is to review the budget variance, "
            "validate cost allocation accuracy, investigate whether the overspend is expected or anomalous, "
            "and assign accountability for corrective action."
        )

    if "compare" in q or "vs" in q or "versus" in q:
        return (
            "The retrieved documents do not provide enough specific FinOps guidance for this exact comparison. "
            "Based on the analytical result, the immediate CFO priority is to compare cost drivers, budget assumptions, "
            "and ownership across the two scopes before deciding on corrective action."
        )

    if "total" in q or "sum" in q or "average" in q or "avg" in q:
        return (
            "The retrieved documents do not provide enough specific FinOps guidance for this exact metric. "
            "Based on the analytical result, the immediate CFO priority is to interpret the metric against budget expectations, "
            "ownership, and recent spending behavior."
        )

    return (
        "The retrieved documents do not provide enough specific FinOps guidance for this exact case. "
        "The immediate CFO priority is to use the selected-period analytical result as the basis for review, ownership, and action."
    )


def answer_analytical_question(
    question: str,
    products_filtered: pd.DataFrame,
    internal_filtered: pd.DataFrame
) -> str:
    combined = combine_filtered_data(products_filtered, internal_filtered)

    if combined.empty:
        return "No rows matched the selected period, so no analytical answer can be calculated."

    scoped = filter_question_scope(combined, question)

    if scoped.empty:
        return "No rows matched the selected period for the requested analytical scope."

    q = question.lower()
    metric = detect_metric(question)
    dimension = detect_dimension(question)
    scope_label = build_scope_label(question)

    scoped = scoped.copy()
    scoped["Variance %"] = scoped.apply(
        lambda row: safe_variance_pct(row["Actual Cost"], row["Allocated Budget"]),
        axis=1
    )

    if is_why_budget_question(question):
        return answer_why_budget_question(scoped, scope_label)

    if "compare" in q or "vs" in q or "versus" in q:
        return answer_compare_question(scoped, metric)

    if "top" in q or "bottom" in q or "highest" in q or "lowest" in q or "most" in q or "least" in q or "over budget" in q or "under budget" in q:
        return answer_top_bottom_question(scoped, question, metric, dimension, scope_label)

    if "average" in q or "avg" in q:
        return answer_average_question(scoped, metric, scope_label)

    if "total" in q or "sum" in q or "how much" in q:
        return answer_total_question(scoped, metric, scope_label)

    return (
        "I classified the question as analytical, but it does not match a specific granular pattern yet. "
        "Here is the selected-period summary instead:\n\n"
        f"{generate_summary(products_filtered, 'Products')}\n\n"
        f"{generate_summary(internal_filtered, 'Internal')}"
    )


# =========================================================
# GENERAL RAG
# =========================================================
def load_chunks(input_file: Path):
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_bm25(chunks):
    chunk_texts = [chunk["text"] for chunk in chunks]
    tokenized_chunks = [text.lower().split() for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25, chunk_texts


def semantic_search(question, chunks, embedding_model, top_k=5):
    question_embedding = embedding_model.encode(question)
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = embedding_model.encode(chunk_texts)

    semantic_similarities = np.dot(chunk_embeddings, question_embedding)
    top_indices = np.argsort(semantic_similarities)[-top_k:][::-1]

    return semantic_similarities, top_indices


def lexical_search(question, bm25, top_k=5):
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[::-1][:top_k]

    return bm25_scores, top_indices


def rerank_results(question, candidate_indices, chunks, semantic_similarities, bm25_scores, reranker):
    pairs = [(question, chunks[i]["text"]) for i in candidate_indices]
    rerank_scores = reranker.predict(pairs)

    reranked_results = []
    for idx, rerank_score in zip(candidate_indices, rerank_scores):
        reranked_results.append({
            "index": idx,
            "filename": chunks[idx]["filename"],
            "chunk_id": chunks[idx]["chunk_id"],
            "text": chunks[idx]["text"],
            "semantic_score": float(semantic_similarities[idx]),
            "bm25_score": float(bm25_scores[idx]),
            "rerank_score": float(rerank_score),
        })

    reranked_results = sorted(
        reranked_results,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_results


def build_context(top_results):
    selected_chunks = []

    for item in top_results:
        selected_chunks.append(
            f"Source file: {item['filename']}\n"
            f"Chunk ID: {item['chunk_id']}\n"
            f"Semantic score: {item['semantic_score']:.4f}\n"
            f"BM25 score: {item['bm25_score']:.4f}\n"
            f"Rerank score: {item['rerank_score']:.4f}\n"
            f"Content:\n{item['text']}"
        )

    return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(selected_chunks)


def get_confidence_info(top_results):
    if not top_results:
        return {
            "label": "Low",
            "reason": "No retrieved results",
            "best_rerank_score": None,
            "best_semantic_score": None,
            "best_bm25_score": None,
        }

    best_result = top_results[0]
    best_rerank_score = best_result["rerank_score"]
    best_semantic_score = best_result["semantic_score"]
    best_bm25_score = best_result["bm25_score"]

    if best_rerank_score >= 2.0 and best_semantic_score >= 0.60:
        label = "High"
        reason = "Strong rerank match and strong semantic relevance"
    elif best_rerank_score >= -4.0 and best_semantic_score >= 0.45:
        label = "Medium"
        reason = "Moderate retrieval relevance"
    else:
        label = "Low"
        reason = "Weak retrieval relevance"

    return {
        "label": label,
        "reason": reason,
        "best_rerank_score": best_rerank_score,
        "best_semantic_score": best_semantic_score,
        "best_bm25_score": best_bm25_score,
    }


def should_fallback(top_results, rerank_threshold=-5.0):
    if not top_results:
        return True

    best_rerank_score = top_results[0]["rerank_score"]
    return best_rerank_score < rerank_threshold


def build_sources(top_results):
    lines = []
    for item in top_results:
        lines.append(f"- {item['filename']} (chunk {item['chunk_id']})")
    return "\n".join(lines)


def ask_llm(question, context):
    client = OpenAI()

    user_prompt = f"""
Question:
{question}

Retrieved context:
{context}

Answer the question using only the retrieved context.
"""

    response = client.responses.create(
        model="gpt-5.4",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.output_text


def rag_answer(question, chunks, embedding_model, reranker):
    bm25, _ = build_bm25(chunks)

    semantic_similarities, top_semantic_indices = semantic_search(
        question, chunks, embedding_model, top_k=5
    )

    bm25_scores, top_bm25_indices = lexical_search(
        question, bm25, top_k=5
    )

    combined_indices = list(set(list(top_semantic_indices) + list(top_bm25_indices)))

    reranked_results = rerank_results(
        question,
        combined_indices,
        chunks,
        semantic_similarities,
        bm25_scores,
        reranker
    )

    top_final_results = reranked_results[:5]
    confidence = get_confidence_info(top_final_results)
    fallback = should_fallback(top_final_results)
    context = build_context(top_final_results)
    sources = build_sources(top_final_results)

    if fallback:
        final_answer = "I don't have enough information in the retrieved documents."
    else:
        final_answer = ask_llm(question, context)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "question": question,
        "answer": final_answer,
        "top_results": top_final_results,
        "confidence": confidence,
        "sources": sources,
        "timestamp": timestamp,
        "fallback": fallback,
    }


def handle_general_question(
    question: str,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder
) -> Dict[str, Any]:
    chunks = load_chunks(INPUT_FILE)
    return rag_answer(question, chunks, embedding_model, reranker)


# =========================================================
# HYBRID MODE - CONTEXTUAL
# =========================================================
def handle_hybrid_question(
    question: str,
    products_filtered: pd.DataFrame,
    internal_filtered: pd.DataFrame,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder
) -> Dict[str, Any]:
    analytical_answer = answer_analytical_question(
        question=question,
        products_filtered=products_filtered,
        internal_filtered=internal_filtered
    )

    contextual_finops_query = build_hybrid_finops_query(question, analytical_answer)
    general_result = handle_general_question(contextual_finops_query, embedding_model, reranker)

    if general_result["fallback"]:
        general_answer = build_contextual_finops_fallback(question, analytical_answer)
    else:
        general_answer = general_result["answer"]

    combined_answer = (
        "Hybrid CFO Answer\n\n"
        "1. Analytical Answer restricted to the selected period:\n"
        f"{analytical_answer}\n\n"
        "2. FinOps Interpretation linked to the analytical result:\n"
        f"{general_answer}\n\n"
        "3. CFO Priority:\n"
        "Use the selected-period analytical result first, then apply the contextual FinOps interpretation to guide action."
    )

    return {
        "question": question,
        "answer": combined_answer,
        "top_results": general_result["top_results"],
        "confidence": general_result["confidence"],
        "sources": general_result["sources"],
        "timestamp": general_result["timestamp"],
        "fallback": general_result["fallback"],
        "contextual_finops_query": contextual_finops_query,
    }


# =========================================================
# DISPLAY / SPOKEN TEXT
# =========================================================
def build_display_text(result: Dict[str, Any]) -> str:
    return result["answer"]


def build_spoken_text(answer: str) -> str:
    fallback_text = "I don't have enough information in the retrieved documents."

    if answer.strip() == fallback_text:
        return (
            "I do not have enough information in the retrieved documents, "
            "to answer confidently."
        )

    spoken = answer.strip()

    # Remove markdown formatting
    spoken = re.sub(r"\*\*(.*?)\*\*", r"\1", spoken)
    spoken = re.sub(r"__(.*?)__", r"\1", spoken)
    spoken = re.sub(r"`(.*?)`", r"\1", spoken)

    # Hybrid headings -> speech-friendly structure
    spoken = spoken.replace("Hybrid CFO Answer", "Hybrid C F O answer,")
    spoken = spoken.replace(
        "1. Analytical Answer restricted to the selected period:",
        "First, the analytical answer for the selected period,"
    )
    spoken = spoken.replace(
        "2. FinOps Interpretation linked to the analytical result:",
        "Next, the Fin Ops interpretation linked to the analytical result,"
    )
    spoken = spoken.replace(
        "2. General FinOps Interpretation:",
        "Next, the general Fin Ops interpretation,"
    )
    spoken = spoken.replace(
        "3. CFO Priority:",
        "Finally, the C F O priority,"
    )

    # Cleaner labels
    spoken = spoken.replace("Total Actual Cost:", "Total actual cost,")
    spoken = spoken.replace("Total Allocated Budget:", "Total allocated budget,")
    spoken = spoken.replace("Total Variance:", "Total variance,")
    spoken = spoken.replace("Variance %:", "Variance percentage,")
    spoken = spoken.replace("Executive answer", "Executive answer,")
    spoken = spoken.replace("Key best practices", "Key best practices,")
    spoken = spoken.replace(
        "What a C F O should prioritize first",
        "What the C F O should prioritize first,"
    )

    # Bullets -> spoken markers
    spoken = re.sub(r"(?m)^\s*-\s+", "Next point, ", spoken)
    spoken = re.sub(r"(?m)^\s*\d+\.\s+", "", spoken)

    # Acronyms
    acronym_map = {
        r"\bCFO\b": "C F O",
        r"\bAI\b": "A I",
        r"\bAWS\b": "A W S",
        r"\bSQL\b": "S Q L",
        r"\bRAG\b": "R A G",
        r"\bFinOps\b": "Fin Ops",
        r"\bKPI\b": "K P I",
        r"\bROI\b": "R O I",
        r"\bETL\b": "E T L",
    }

    for pattern, replacement in acronym_map.items():
        spoken = re.sub(pattern, replacement, spoken)

    # Currency -> more natural speech
    def currency_to_speech(match):
        raw = match.group(1).replace(",", "")
        value = float(raw)
        dollars = int(value)
        cents = int(round((value - dollars) * 100))

        if cents == 0:
            return f"{dollars} dollars"
        return f"{dollars} dollars and {cents} cents"

    spoken = re.sub(r"\$([0-9,]+(?:\.[0-9]{1,2})?)", currency_to_speech, spoken)

    # Signed values and symbols
    spoken = spoken.replace("%", " percent")
    spoken = spoken.replace("+", "plus ")
    spoken = spoken.replace("&", " and ")
    spoken = spoken.replace("/", " slash ")
    spoken = spoken.replace("•", "")

    # Decimals
    spoken = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", spoken)

    # Add commas for natural pacing
    spoken = spoken.replace("This indicates that", "This indicates that,")
    spoken = spoken.replace("Based on the analytical result", "Based on the analytical result,")
    spoken = spoken.replace(
        "When Products are over budget during a selected period",
        "When Products are over budget during a selected period,"
    )
    spoken = spoken.replace(
        "When Internal spend is over budget during a selected period",
        "When Internal spend is over budget during a selected period,"
    )
    spoken = spoken.replace("a C F O should", "a C F O should,")
    spoken = spoken.replace("The goal is to", "The goal is to,")
    spoken = spoken.replace(
        "Use the selected-period analytical result first",
        "Use the selected-period analytical result first,"
    )
    spoken = spoken.replace(
        "then apply the contextual Fin Ops interpretation to guide action",
        "then apply the contextual Fin Ops interpretation, to guide action"
    )

    # Normalize whitespace
    spoken = re.sub(r"\s+", " ", spoken).strip()

    # Convert some punctuation into more voice-friendly rhythm
    spoken = spoken.replace(": ", ", ")
    spoken = spoken.replace("; ", ", ")

    # Cleanup repeated commas
    spoken = re.sub(r",\s*,+", ", ", spoken)
    spoken = re.sub(r"\s+", " ", spoken).strip()

    return f"Here is your executive briefing. {spoken}"


def escape_ssml_text(text: str) -> str:
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def apply_ssml_formatting(text: str) -> str:
    safe_text = escape_ssml_text(text)

    # Strong pauses for major sections
    safe_text = safe_text.replace(
        "First, the analytical answer for the selected period,",
        "First, <break time='300ms'/> the analytical answer for the selected period, <break time='900ms'/>"
    )
    safe_text = safe_text.replace(
        "Next, the Fin Ops interpretation linked to the analytical result,",
        "Next, <break time='300ms'/> the Fin Ops interpretation linked to the analytical result, <break time='900ms'/>"
    )
    safe_text = safe_text.replace(
        "Next, the general Fin Ops interpretation,",
        "Next, <break time='300ms'/> the general Fin Ops interpretation, <break time='900ms'/>"
    )
    safe_text = safe_text.replace(
        "Finally, the C F O priority,",
        "Finally, <break time='300ms'/> the C F O priority, <break time='1000ms'/>"
    )

    # Sentence pauses
    safe_text = safe_text.replace(". ", ". <break time='700ms'/> ")

    # Comma pauses
    safe_text = safe_text.replace(", ", ", <break time='350ms'/> ")

    # Colon / semicolon pauses
    safe_text = safe_text.replace(": ", ": <break time='450ms'/> ")
    safe_text = safe_text.replace("; ", "; <break time='450ms'/> ")

    # Pause after financial labels
    safe_text = safe_text.replace("Total actual cost,", "Total actual cost, <break time='250ms'/>")
    safe_text = safe_text.replace("Total allocated budget,", "Total allocated budget, <break time='250ms'/>")
    safe_text = safe_text.replace("Total variance,", "Total variance, <break time='250ms'/>")
    safe_text = safe_text.replace("Variance percentage,", "Variance percentage, <break time='250ms'/>")

    # Pause after money amounts
    safe_text = re.sub(
        r"(dollars and \d+ cents)",
        r"\1 <break time='500ms'/>",
        safe_text
    )

    safe_text = re.sub(
        r"(\d+ dollars)(?! and)",
        r"\1 <break time='500ms'/>",
        safe_text
    )

    # Pause after percentages
    safe_text = re.sub(
        r"(percent)",
        r"\1 <break time='500ms'/>",
        safe_text
    )

    # Cleanup duplicated breaks
    safe_text = re.sub(
        r"(<break time='[0-9]+ms'/>\s*){2,}",
        "<break time='700ms'/> ",
        safe_text
    )

    return safe_text


def text_to_speech(text, output_file=None):
    if output_file is None:
        timestamp_for_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"rag_answer_{timestamp_for_file}.mp3"
    else:
        output_file = Path(output_file)

    session = boto3.Session(profile_name=AWS_PROFILE)
    polly = session.client("polly", region_name=AWS_REGION)

    formatted_text = apply_ssml_formatting(text)

    ssml_text = f"""
    <speak>
        <prosody rate="90%">
            {formatted_text}
        </prosody>
    </speak>
    """

    response = polly.synthesize_speech(
        Text=ssml_text,
        TextType="ssml",
        OutputFormat="mp3",
        VoiceId=VOICE_ID,
        Engine=ENGINE
    )

    with open(output_file, "wb") as f:
        f.write(response["AudioStream"].read())

    return str(output_file)


def speak_response(answer_text: str):
    spoken_text = build_spoken_text(answer_text)
    audio_file = text_to_speech(spoken_text)
    return spoken_text, audio_file


# =========================================================
# FINAL ROUTER
# =========================================================
def build_result_wrapper(
    question: str,
    route: str,
    answer: str,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "question": question,
        "route": route,
        "answer": answer,
        "top_results": [],
        "confidence": {
            "label": "Rule-Based",
            "reason": "Analytical result based on selected-period calculations"
        },
        "sources": "Selected-period CSV analytics only",
        "timestamp": timestamp,
        "fallback": False,
    }


def main():
    print("Loading selected-period datasets...")

    products = normalize_string_columns(normalize_columns(load_csv(PRODUCTS_FILE)))
    internal = normalize_string_columns(normalize_columns(load_csv(INTERNAL_FILE)))

    products_filtered = filter_by_selection(products, selection)
    internal_filtered = filter_by_selection(internal, selection)

    print("Loading models...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    question = input("Enter the user's final question: ").strip()

    if not question:
        print("No question entered.")
        return

    route = classify_question(question)

    if route == "general":
        result = handle_general_question(question, embedding_model, reranker)
        result["route"] = "general"

    elif route == "analytical":
        analytical_answer = answer_analytical_question(
            question=question,
            products_filtered=products_filtered,
            internal_filtered=internal_filtered
        )
        result = build_result_wrapper(
            question=question,
            route="analytical",
            answer=analytical_answer
        )

    else:
        result = handle_hybrid_question(
            question=question,
            products_filtered=products_filtered,
            internal_filtered=internal_filtered,
            embedding_model=embedding_model,
            reranker=reranker
        )
        result["route"] = "hybrid"

    display_text = build_display_text(result)
    spoken_text, audio_file = speak_response(display_text)

    print("\n======================================================================")
    print("SELECTION USED")
    print("======================================================================")
    print(selection)

    print("\n======================================================================")
    print("FILTERED ROW COUNTS")
    print("======================================================================")
    print(f"Products rows matched: {len(products_filtered)}")
    print(f"Internal rows matched: {len(internal_filtered)}")

    print(f"\nTIMESTAMP: {result['timestamp']}")

    print("\nROUTE:")
    print(result["route"])

    print("\nQUESTION:")
    print(result["question"])

    if result.get("contextual_finops_query"):
        print("\nCONTEXTUAL FINOPS QUERY:")
        print(result["contextual_finops_query"])

    if result.get("top_results"):
        print("\nTOP CHUNKS RETRIEVED (HYBRID + RERANK):")
        for item in result["top_results"]:
            print(
                f"- {item['filename']} | "
                f"chunk {item['chunk_id']} | "
                f"semantic={item['semantic_score']:.4f} | "
                f"bm25={item['bm25_score']:.4f} | "
                f"rerank={item['rerank_score']:.4f}"
            )

    print("\nDISPLAY TEXT:\n")
    print(display_text)

    print("\nSPOKEN TEXT:\n")
    print(spoken_text)

    print("\nSOURCES:")
    print(result["sources"])

    print("\nCONFIDENCE:")
    confidence = result.get("confidence", {})
    print(f"Label: {confidence.get('label')}")
    print(f"Reason: {confidence.get('reason')}")

    if confidence.get("best_rerank_score") is not None:
        print(f"Best rerank score: {confidence['best_rerank_score']:.4f}")
    if confidence.get("best_semantic_score") is not None:
        print(f"Best semantic score: {confidence['best_semantic_score']:.4f}")
    if confidence.get("best_bm25_score") is not None:
        print(f"Best BM25 score: {confidence['best_bm25_score']:.4f}")

    print("\nAUDIO FILE:")
    print(audio_file)

    try:
        os.startfile(audio_file)
        time.sleep(0.5)
    except Exception as e:
        print(f"Audio auto-play skipped: {e}")


if __name__ == "__main__":
    main()
    