import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pyathena import connect
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

# =========================================================
# FILES / CONFIG
# =========================================================
INPUT_FILE = Path("data/chunks_general_questions.json")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

AWS_PROFILE = "terraform-runner"
AWS_REGION = "us-east-1"
VOICE_ID = "Joanna"
ENGINE = "neural"

ATHENA_SCHEMA = "cfo_finops_db"
ATHENA_S3_STAGING_DIR = "s3://kestra-zoomcamp-ny-taxi-mtl9999999999988888/athena-results/"
ATHENA_WORKGROUP = "primary"

SYSTEM_PROMPT = """
You are a FinOps AI assistant answering CFO-level questions using Retrieval-Augmented Generation (RAG).

STRICT RULES:
1. Only use the provided retrieved context.
2. Do not use outside knowledge.
3. Do not invent facts, numbers, explanations, examples, tools, or practices that are not supported by the retrieved context.
4. Do not guess.
5. Stay grounded in the retrieved documents only.
6. Do not mention specific cloud services, platforms, tools, products, numbers, percentages, or examples unless they appear explicitly in the retrieved context.
7. If the retrieved context contains no relevant information for the question, respond exactly with:
   "I don't have enough information in the retrieved documents."
8. If rule 7 applies, STOP immediately and do not add any extra explanation, bullets, recommendations, or priorities.
9. If the retrieved context contains partial but relevant information, you are allowed to answer using only that available information.
10. When the context is partial, clearly state the limitation and do not fill gaps with outside knowledge.
11. Prefer a partial factual answer over a total refusal when the retrieved context is meaningfully relevant.

ANSWER STYLE:
- Be concise
- Be structured
- Be executive-friendly
- Focus on factual FinOps guidance
- Do not add speculation

OUTPUT FORMAT:
1. Short executive answer
2. 3 to 5 key best practices or supporting points from the retrieved context
3. What a CFO should prioritize first
""".strip()
# =========================================================
# EXAMPLE SELECTION
# Replace with your real selected period object later
# =========================================================
selection = {
    "mode": "weeks",
    "block": "B",
    "weeks": [5, 6],
    "number_of_weeks": 2
}

# =========================================================
# SELECTION LOGIC MAPPING
# =========================================================
MONTH_BLOCK_MAPPING = {
    "A": ["Month 1"],
    "B": ["Month 2"],
    "C": ["Month 3"],
    "AB": ["Month 1", "Month 2"],
    "BC": ["Month 2", "Month 3"],
    "ABC": ["Month 1", "Month 2", "Month 3"],
}

# =========================================================
# ATHENA TABLE ROUTING
# IMPORTANT:
# We always load BOTH scopes:
# - Product
# - Division
# because the agent must systematically have both.
# =========================================================
def get_target_tables(selection: Dict[str, Any]) -> Dict[str, str]:
    mode = selection["mode"]

    mapping = {
        "days": {
            "division": "stg_internal_daily_budget",
            "product": "stg_products_daily_budget",
        },
        "weeks": {
            "division": "mart_division_cfo",
            "product": "mart_product_cfo",
        },
        "months": {
            "division": "stg_internal_monthly_budget",
            "product": "stg_products_monthly_budget",
        },
    }

    if mode not in mapping:
        raise ValueError(f"Unsupported mode: {mode}")

    return mapping[mode]


def build_where_clause(selection: Dict[str, Any]) -> str:
    """
    Build SQL filter from the final user-selected period.

    Supported:
    - days   -> week + consecutive days
    - weeks  -> selected weeks
    - months -> monthly block A/B/C/AB/BC/ABC
    """
    mode = selection["mode"]

    if mode == "weeks":
        weeks = selection["weeks"]
        week_sql = ", ".join([f"'Week {w}'" for w in weeks])
        return f"week in ({week_sql})"

    if mode == "months":
        block = selection["monthly_block"]
        if block not in MONTH_BLOCK_MAPPING:
            raise ValueError(f"Unsupported monthly block: {block}")

        months = MONTH_BLOCK_MAPPING[block]
        month_sql = ", ".join([f"'{m}'" for m in months])
        return f"month in ({month_sql})"

    if mode == "days":
        week = selection["week"]
        day_count = selection["days"]

        if week == 1:
            available_days = ["Wednesday", "Thursday", "Friday", "Saturday"]
        else:
            available_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        if day_count < 1 or day_count > len(available_days):
            raise ValueError(
                f"Invalid day count {day_count} for week {week}. "
                f"Allowed max is {len(available_days)}."
            )

        selected_days = available_days[:day_count]
        day_sql = ", ".join([f"'{d}'" for d in selected_days])
        return f"week = 'Week {week}' and day in ({day_sql})"

    raise ValueError(f"Unsupported mode: {mode}")


def build_select_sql(table_name: str, where_clause: str) -> str:
    return f"""
    SELECT *
    FROM {ATHENA_SCHEMA}.{table_name}
    WHERE {where_clause}
    """


def query_athena(sql: str) -> pd.DataFrame:
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)

    conn = connect(
        region_name=AWS_REGION,
        schema_name=ATHENA_SCHEMA,
        s3_staging_dir=ATHENA_S3_STAGING_DIR,
        work_group=ATHENA_WORKGROUP,
        boto3_session=session,
    )

    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


def load_selected_period_data(selection: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    - product_df
    - division_df

    Both are already restricted to the selected period.
    """
    tables = get_target_tables(selection)
    where_clause = build_where_clause(selection)

    product_sql = build_select_sql(tables["product"], where_clause)
    division_sql = build_select_sql(tables["division"], where_clause)

    product_df = query_athena(product_sql)
    division_df = query_athena(division_sql)

    return product_df, division_df


# =========================================================
# DATA HELPERS
# =========================================================
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
# STANDARDIZE ATHENA DATA FOR ANALYTICS
# IMPORTANT:
# Replace old Products/Internal logic with Product/Division.
# =========================================================
def prepare_standardized_df(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    df = normalize_string_columns(normalize_columns(df))

    actual_col = find_first_existing_column(df, ["actual_cost_usd", "Actual Cost", "Actual Cost ($)"])
    budget_col = find_first_existing_column(df, ["allocated_budget_usd", "Allocated Budget", "Allocated Budget ($)"])
    variance_col = find_first_existing_column(df, ["variance_usd", "Variance", "Variance ($)"])
    variance_pct_col = find_first_existing_column(df, ["variance_pct", "variance_percent", "Variance %"])

    week_col = find_first_existing_column(df, ["week", "Week"])
    day_col = find_first_existing_column(df, ["day", "Day"])
    month_col = find_first_existing_column(df, ["month", "Month"])
    date_col = find_first_existing_column(df, ["date", "Date"])

    standardized = pd.DataFrame(index=df.index)
    standardized["Dataset"] = [dataset_label] * len(df)

    standardized["Month"] = df[month_col].values if month_col else ["DATA_NOT_AVAILABLE"] * len(df)
    standardized["Week"] = df[week_col].values if week_col else ["DATA_NOT_AVAILABLE"] * len(df)
    standardized["Day"] = df[day_col].values if day_col else ["DATA_NOT_AVAILABLE"] * len(df)
    standardized["Date"] = df[date_col].values if date_col else ["DATA_NOT_AVAILABLE"] * len(df)

    if actual_col is None:
        raise ValueError(f"Could not find actual cost column. Found columns: {list(df.columns)}")

    standardized["Actual Cost"] = clean_numeric(df[actual_col])

    if budget_col is not None:
        standardized["Allocated Budget"] = clean_numeric(df[budget_col])
    else:
        standardized["Allocated Budget"] = 0.0

    if variance_col is not None:
        standardized["Variance"] = clean_numeric(df[variance_col])
    else:
        standardized["Variance"] = standardized["Actual Cost"] - standardized["Allocated Budget"]

    if variance_pct_col is not None:
        standardized["Variance %"] = clean_numeric(df[variance_pct_col])
    else:
        standardized["Variance %"] = standardized.apply(
            lambda row: safe_variance_pct(row["Actual Cost"], row["Allocated Budget"]),
            axis=1
        )

    return standardized.reset_index(drop=True)


def combine_filtered_data(product_filtered: pd.DataFrame, division_filtered: pd.DataFrame) -> pd.DataFrame:
    product_std = prepare_standardized_df(product_filtered, "Product")
    division_std = prepare_standardized_df(division_filtered, "Division")
    return pd.concat([product_std, division_std], ignore_index=True)


# =========================================================
# GENERAL CFO SUMMARY ON SELECTED PERIOD
# =========================================================
def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    grouped = (
        df.groupby(["Week", "Day"], dropna=False)[["Actual Cost", "Allocated Budget", "Variance"]]
        .sum()
        .reset_index()
    )

    grouped["Variance %"] = grouped.apply(
        lambda row: safe_variance_pct(row["Actual Cost"], row["Allocated Budget"]),
        axis=1
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

    if "Week" not in df.columns or "Day" not in df.columns:
        total_budget = df["Allocated Budget"].sum()
        total_actual = df["Actual Cost"].sum()
        total_variance = df["Variance"].sum()
        total_variance_pct = safe_variance_pct(total_actual, total_budget)
        status = build_status_text(total_variance)

        lines = [
            "======================================================================",
            f"CFO SUMMARY - {label.upper()}",
            "======================================================================",
            "",
            f"Total Budget: {format_currency(total_budget)}",
            f"Total Actual Cost: {format_currency(total_actual)}",
            f"Variance: {format_currency(total_variance, force_sign=True)} ({format_percent(total_variance_pct, force_sign=True)})",
            f"Status: {status}",
        ]
        return "\n".join(lines).rstrip()

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
    "percent", "percentage", "day", "week", "month", "product", "division",
    "breakdown", "which", "show me the numbers", "calculate", "why", "explain"
]

GENERAL_KEYWORDS = [
    "what is finops", "best practice", "best practices",
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
    return "hybrid"


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

    if "product" in q and "division" not in q:
        scoped = scoped[scoped["Dataset"] == "product"].copy()
    elif "division" in q and "product" not in q:
        scoped = scoped[scoped["Dataset"] == "division"].copy()

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
        return "I cannot compare Product and Division because only one dataset is available within the selected period."

    grouped["Dataset"] = grouped["Dataset"].astype(str).str.strip().str.lower()

    product_row = grouped[grouped["Dataset"] == "product"]
    division_row = grouped[grouped["Dataset"] == "division"]

    if product_row.empty or division_row.empty:
        return "I cannot compare Product and Division because one side has no data within the selected period."

    p = product_row.iloc[0]
    d = division_row.iloc[0]

    if metric == "Variance %":
        p_metric_text = format_percent(p[metric], force_sign=True)
        d_metric_text = format_percent(d[metric], force_sign=True)
    else:
        p_metric_text = format_currency(p[metric], force_sign=(metric == "Variance"))
        d_metric_text = format_currency(d[metric], force_sign=(metric == "Variance"))

    lines = [
        "Within the selected period, here is the comparison between Product and Division:",
        f"- Product {metric}: {p_metric_text}",
        f"- Division {metric}: {d_metric_text}",
    ]

    additional_metrics = ["Actual Cost", "Allocated Budget", "Variance", "Variance %"]
    additional_metrics = [m for m in additional_metrics if m != metric]

    for extra_metric in additional_metrics:
        if extra_metric == "Variance %":
            p_text = format_percent(p[extra_metric], force_sign=True)
            d_text = format_percent(d[extra_metric], force_sign=True)
        else:
            p_text = format_currency(p[extra_metric], force_sign=(extra_metric == "Variance"))
            d_text = format_currency(d[extra_metric], force_sign=(extra_metric == "Variance"))

        lines.append(f"- Product {extra_metric}: {p_text}")
        lines.append(f"- Division {extra_metric}: {d_text}")

    return "\n".join(lines)


def build_scope_label(question: str) -> str:
    q = question.lower()
    if "product" in q and "division" not in q:
        return "Product"
    if "division" in q and "product" not in q:
        return "Division"
    return "Product and Division combined"


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
        if "product" in q and "division" not in q:
            return (
                "What FinOps best practices should a CFO prioritize when Product spend is over budget "
                "during a selected period, including budget variance review, cost allocation validation, "
                "anomaly investigation, and accountability?"
            )
        if "division" in q and "product" not in q:
            return (
                "What FinOps best practices should a CFO prioritize when Division spend is over budget "
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
    product_filtered: pd.DataFrame,
    division_filtered: pd.DataFrame
) -> str:
    combined = combine_filtered_data(product_filtered, division_filtered)

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
        f"{generate_summary(product_filtered, 'Product')}\n\n"
        f"{generate_summary(division_filtered, 'Division')}"
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
        question, chunks, embedding_model, top_k=10
    )

    bm25_scores, top_bm25_indices = lexical_search(
        question, bm25, top_k=10
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
# HYBRID MODE
# =========================================================
def handle_hybrid_question(
    question: str,
    product_filtered: pd.DataFrame,
    division_filtered: pd.DataFrame,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder
) -> Dict[str, Any]:
    analytical_answer = answer_analytical_question(
        question=question,
        product_filtered=product_filtered,
        division_filtered=division_filtered
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

    spoken = re.sub(r"\*\*(.*?)\*\*", r"\1", spoken)
    spoken = re.sub(r"__(.*?)__", r"\1", spoken)
    spoken = re.sub(r"`(.*?)`", r"\1", spoken)

    spoken = spoken.replace("Hybrid CFO Answer", "Hybrid C F O answer,")
    spoken = spoken.replace(
        "1. Analytical Answer restricted to the selected period:",
        "First, the analytical answer for the selected period,"
    )
    spoken = spoken.replace(
        "2. FinOps Interpretation linked to the analytical result:",
        "Second, the Fin Ops interpretation linked to the analytical result,"
    )
    spoken = spoken.replace(
        "3. CFO Priority:",
        "Third, the C F O priority,"
    )

    spoken = spoken.replace("%", " percent")
    spoken = spoken.replace("&", " and ")
    spoken = spoken.replace("/", " slash ")
    spoken = spoken.replace("•", "")
    spoken = spoken.replace(" - ", ". ")

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

    spoken = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", spoken)
    spoken = re.sub(r"(?m)^\s*[-*]\s+", "Next point. ", spoken)
    spoken = re.sub(r"(?m)^\s*\d+\.\s+", "Step. ", spoken)

    spoken = re.sub(r"\s+", " ", spoken).strip()

    return spoken


def escape_ssml_text(text: str) -> str:
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def apply_ssml_formatting(text: str) -> str:
    safe_text = escape_ssml_text(text)
    safe_text = safe_text.replace(". ", ". <break time='650ms'/> ")
    safe_text = safe_text.replace(": ", ": <break time='400ms'/> ")
    safe_text = safe_text.replace("; ", "; <break time='350ms'/> ")
    safe_text = safe_text.replace("? ", "? <break time='700ms'/> ")
    return safe_text


def text_to_speech(text: str, output_file: Optional[Path] = None) -> str:
    if output_file is None:
        timestamp_for_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_file = OUTPUT_DIR / f"agent_answer_{timestamp_for_file}.mp3"

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


# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading selected-period data from Athena.")
    product_filtered, division_filtered = load_selected_period_data(selection)

    print("Loading models.")
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
            product_filtered=product_filtered,
            division_filtered=division_filtered
        )

        result = {
            "question": question,
            "answer": analytical_answer,
            "route": "analytical",
            "top_results": [],
            "confidence": {"label": "Data-driven", "reason": "Analytical answer from Athena-filtered data"},
            "sources": "Athena filtered product and division datasets",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fallback": False,
        }

    else:
        result = handle_hybrid_question(
            question=question,
            product_filtered=product_filtered,
            division_filtered=division_filtered,
            embedding_model=embedding_model,
            reranker=reranker
        )
        result["route"] = "hybrid"

    display_text = build_display_text(result)
    spoken_text = build_spoken_text(display_text)

    print("\n==============================")
    print("FINAL ANSWER")
    print("==============================")
    print(display_text)

    audio_file = text_to_speech(spoken_text)
    print(f"\nAudio saved to: {audio_file}")


# =========================================================
# ATHENA LOADER
# =========================================================
MONTH_BLOCK_MAPPING = {
    "A": ["Month 1"],
    "B": ["Month 2"],
    "C": ["Month 3"],
    "AB": ["Month 1", "Month 2"],
    "BC": ["Month 2", "Month 3"],
    "ABC": ["Month 1", "Month 2", "Month 3"],
}


def get_target_tables(selection: Dict[str, Any]) -> Dict[str, str]:
    mode = selection["mode"]

    mapping = {
        "days": {
            "division": "stg_internal_daily_budget",
            "product": "stg_products_daily_budget",
        },
        "weeks": {
            "division": "mart_division_cfo",
            "product": "mart_product_cfo",
        },
        "months": {
            "division": "stg_internal_monthly_budget",
            "product": "stg_products_monthly_budget",
        },
    }

    if mode not in mapping:
        raise ValueError(f"Unsupported mode: {mode}")

    return mapping[mode]


def build_where_clause(selection: Dict[str, Any]) -> str:
    mode = selection["mode"]

    if mode == "weeks":
        weeks = selection["weeks"]
        week_sql = ", ".join([f"'Week {w}'" for w in weeks])
        return f"week in ({week_sql})"

    if mode == "months":
        block = selection["monthly_block"]
        if block not in MONTH_BLOCK_MAPPING:
            raise ValueError(f"Unsupported monthly block: {block}")

        months = MONTH_BLOCK_MAPPING[block]
        month_sql = ", ".join([f"'{m}'" for m in months])
        return f"month in ({month_sql})"

    if mode == "days":
        week = selection["week"]
        day_count = selection["days"]

        if week == 1:
            available_days = ["Wednesday", "Thursday", "Friday", "Saturday"]
        else:
            available_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        if day_count < 1 or day_count > len(available_days):
            raise ValueError(
                f"Invalid day count {day_count} for week {week}. "
                f"Allowed max is {len(available_days)}."
            )

        selected_days = available_days[:day_count]
        day_sql = ", ".join([f"'{d}'" for d in selected_days])
        return f"week = 'Week {week}' and day in ({day_sql})"

    raise ValueError(f"Unsupported mode: {mode}")


def build_select_sql(table_name: str, where_clause: str) -> str:
    return f"""
    SELECT *
    FROM {ATHENA_SCHEMA}.{table_name}
    WHERE {where_clause}
    """


def load_selected_period_data(selection: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tables = get_target_tables(selection)
    where_clause = build_where_clause(selection)

    product_sql = build_select_sql(tables["product"], where_clause)
    division_sql = build_select_sql(tables["division"], where_clause)

    product_df = query_athena(product_sql)
    division_df = query_athena(division_sql)

    return product_df, division_df


if __name__ == "__main__":
    main()