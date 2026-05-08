from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import time
import warnings
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from observability import log_event
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import logging as hf_logging



from resolve_finops_analytics import (
    load_selected_period_data,
    load_selected_period_daily_spend,
)


def load_models_silently():
    buffer = io.StringIO()

    with redirect_stdout(buffer), redirect_stderr(buffer):
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return embedding_model, reranker


load_dotenv()

embedding_model_global, reranker_global = load_models_silently()

warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable.*",
    category=UserWarning,
)

# =========================================================
# FILES / CONFIG
# =========================================================
INPUT_FILE = Path("data/chunks_general_questions.json")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

AWS_PROFILE = os.getenv("AWS_PROFILE", "terraform-runner")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
VOICE_ID = os.getenv("VOICE_ID", "Joanna")
ENGINE = os.getenv("POLLY_ENGINE", "neural")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")

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
""".strip()

def build_finops_fallback(analytical_answer: str = "") -> str:
    raw_analytical = analytical_answer.strip()

    if raw_analytical:
        analytical_block = f"""
1. Analytical result

{raw_analytical}
"""
    else:
        analytical_block = ""

    return f"""
Hybrid CFO Answer

{analytical_block}
2. FinOps interpretation

The retrieved documents do not provide enough specific guidance for this question.
However, based on the selected-period analysis, the company should focus on practical FinOps controls.

Recommended actions:
- Review the top cost drivers for the selected period.
- Improve cost allocation tags by product, team, environment, and owner.
- Set budgets and alerts for each product or business unit.
- Investigate services with high variance or abnormal spend.
- Create a weekly FinOps review with engineering, finance, and product teams.

3. CFO priority

Start with the largest variance drivers, then assign clear ownership and corrective actions.
""".strip()


def handle_rag_failure(rag_answer: str, analytical_answer: str = "") -> str:
    start_time = time.time()

    if not rag_answer:
        latency = round(time.time() - start_time, 3)
        log_event({
            "component": "rag",
            "status": "fallback_used",
            "reason": "empty_rag_answer",
            "fallback_used": True,
            "latency_seconds": latency,
            "slo_latency_status": "pass" if latency <= 5 else "fail"
        })
        return build_finops_fallback(analytical_answer)

    if "I don't have enough information" in rag_answer:
        latency = round(time.time() - start_time, 3)
        log_event({
            "component": "rag",
            "status": "fallback_used",
            "reason": "insufficient_retrieved_context",
            "fallback_used": True,
            "latency_seconds": latency,
            "slo_latency_status": "pass" if latency <= 5 else "fail"
        })
        return build_finops_fallback(analytical_answer)

    latency = round(time.time() - start_time, 3)
    log_event({
        "component": "rag",
        "status": "grounded_answer",
        "fallback_used": False,
        "latency_seconds": latency,
        "slo_latency_status": "pass" if latency <= 5 else "fail"
    })

    return rag_answer

# =========================================================
# AWS HELPERS
# =========================================================
def get_boto3_session() -> boto3.Session:
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


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
    cleaned = (
        series.astype(str)
        .str.replace("\u00A0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
    )

    us_mask = cleaned.str.contains(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$", regex=True, na=False)
    cleaned.loc[us_mask] = cleaned.loc[us_mask].str.replace(",", "", regex=False)

    eu_decimal_mask = cleaned.str.contains(r"^-?\d+,\d+$", regex=True, na=False)
    cleaned.loc[eu_decimal_mask] = cleaned.loc[eu_decimal_mask].str.replace(",", ".", regex=False)

    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def find_first_existing_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    normalized = {str(col).lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    return None


def safe_variance_pct(actual: float, budget: float) -> float:
    if pd.isna(actual) or pd.isna(budget) or budget == 0:
        return 0.0
    return ((actual - budget) / budget) * 100.0


def format_currency(value: float, force_sign: bool = False) -> str:
    if pd.isna(value):
        return "DATA_NOT_AVAILABLE"
    if force_sign:
        sign = "+" if value > 0 else ""
        return f"{sign}${value:,.2f}"
    return f"${value:,.2f}"


def format_percent(value: float, force_sign: bool = False) -> str:
    if pd.isna(value):
        return "DATA_NOT_AVAILABLE"
    if force_sign:
        sign = "+" if value > 0 else ""
        return f"{sign}{value:,.2f}%"
    return f"{value:,.2f}%"


def build_status_text(variance_value: float) -> str:
    if pd.isna(variance_value):
        return "DATA_NOT_AVAILABLE"
    if variance_value > 0:
        return "Over Budget"
    if variance_value < 0:
        return "Under Budget"
    return "On Budget"


def prepare_standardized_df(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    df = normalize_string_columns(normalize_columns(df))

    actual_col = find_first_existing_column(
        df,
        [
            "actual_cost_usd",
            "actual_cost",
            "actual cost",
            "actual cost ($)",
            "actual cost computed",
            "cost",
        ],
    )
    budget_col = find_first_existing_column(
        df,
        [
            "allocated_budget_usd",
            "allocated_budget",
            "allocated budget",
            "allocated budget ($)",
            "allocated budget computed",
            "budget_usd",
            "budget",
        ],
    )

    week_col = find_first_existing_column(df, ["week", "Week"])
    day_col = find_first_existing_column(df, ["day", "Day"])
    month_col = find_first_existing_column(df, ["month", "Month"])
    date_col = find_first_existing_column(df, ["date", "Date"])

    division_col = find_first_existing_column(
        df,
        [
            "division",
            "division_name",
            "internal_division",
            "business_unit",
            "business_unit_name",
            "department",
            "department_name",
            "team",
            "team_name",
            "cost_center",
            "cost_center_name",
            "owner",
            "group_name",
        ],
    )

    product_col = find_first_existing_column(
        df,
        [
            "product",
            "product_name",
            "application",
            "application_name",
            "workload",
            "workload_name",
            "sku",
            "offering",
            "product_line",
        ],
    )

    service_col = find_first_existing_column(
        df,
        [
            "service",
            "service_name",
            "aws_service",
            "cloud_service",
        ],
    )

    if actual_col is None:
        raise ValueError(f"Could not find actual cost column for {dataset_label}. Found columns: {list(df.columns)}")

    standardized = pd.DataFrame(index=df.index)
    standardized["Dataset"] = [dataset_label] * len(df)

    standardized["Month"] = df[month_col].values if month_col else ["DATA_NOT_AVAILABLE"] * len(df)
    standardized["Week"] = df[week_col].values if week_col else ["DATA_NOT_AVAILABLE"] * len(df)
    standardized["Day"] = df[day_col].values if day_col else ["DATA_NOT_AVAILABLE"] * len(df)
    standardized["Date"] = df[date_col].values if date_col else ["DATA_NOT_AVAILABLE"] * len(df)

    standardized["Division"] = (
        df[division_col].astype(str).replace({"nan": "DATA_NOT_AVAILABLE"}).values
        if division_col else ["DATA_NOT_AVAILABLE"] * len(df)
    )
    standardized["Product"] = (
        df[product_col].astype(str).replace({"nan": "DATA_NOT_AVAILABLE"}).values
        if product_col else ["DATA_NOT_AVAILABLE"] * len(df)
    )
    standardized["Service"] = (
        df[service_col].astype(str).replace({"nan": "DATA_NOT_AVAILABLE"}).values
        if service_col else ["DATA_NOT_AVAILABLE"] * len(df)
    )

    if dataset_label.lower() == "division":
        standardized["Entity"] = standardized["Division"]
    elif dataset_label.lower() == "products":
        if product_col is not None:
            standardized["Entity"] = standardized["Product"]
        elif service_col is not None:
            standardized["Entity"] = standardized["Service"]
        else:
            standardized["Entity"] = ["Products"] * len(df)
    else:
        standardized["Entity"] = ["DATA_NOT_AVAILABLE"] * len(df)

    standardized["Actual Cost"] = clean_numeric(df[actual_col]).fillna(0.0)

    if budget_col is not None:
        standardized["Allocated Budget"] = clean_numeric(df[budget_col]).fillna(0.0)
    else:
        standardized["Allocated Budget"] = 0.0

    standardized["Variance"] = standardized["Actual Cost"] - standardized["Allocated Budget"]
    standardized["Variance %"] = standardized.apply(
        lambda row: safe_variance_pct(row["Actual Cost"], row["Allocated Budget"]),
        axis=1
    )

    return standardized.reset_index(drop=True)


def prepare_daily_spend_df(df: pd.DataFrame, dataset_label: str) -> pd.DataFrame:
    df = normalize_string_columns(normalize_columns(df))

    actual_col = find_first_existing_column(
        df,
        [
            "actual_cost_usd",
            "actual_cost",
            "actual cost",
            "actual cost ($)",
            "actual cost computed",
            "cost",
        ],
    )

    week_col = find_first_existing_column(df, ["week", "Week"])
    day_col = find_first_existing_column(df, ["day", "Day"])
    month_col = find_first_existing_column(df, ["month", "Month"])
    date_col = find_first_existing_column(df, ["date", "Date"])

    division_col = find_first_existing_column(df, ["division", "division_name", "internal_division"])
    product_col = find_first_existing_column(df, ["product", "product_name"])
    service_col = find_first_existing_column(df, ["service", "service_name", "aws_service", "cloud_service"])

    if actual_col is None:
        raise ValueError(f"Could not find actual cost column for {dataset_label}. Found columns: {list(df.columns)}")

    daily = pd.DataFrame(index=df.index)
    daily["Dataset"] = [dataset_label] * len(df)
    daily["Month"] = df[month_col].values if month_col else ["DATA_NOT_AVAILABLE"] * len(df)
    daily["Week"] = df[week_col].values if week_col else ["DATA_NOT_AVAILABLE"] * len(df)
    daily["Day"] = df[day_col].values if day_col else ["DATA_NOT_AVAILABLE"] * len(df)
    daily["Date"] = df[date_col].values if date_col else ["DATA_NOT_AVAILABLE"] * len(df)
    daily["Division"] = (
        df[division_col].astype(str).replace({"nan": "DATA_NOT_AVAILABLE"}).values
        if division_col else ["DATA_NOT_AVAILABLE"] * len(df)
    )
    daily["Product"] = (
        df[product_col].astype(str).replace({"nan": "DATA_NOT_AVAILABLE"}).values
        if product_col else ["DATA_NOT_AVAILABLE"] * len(df)
    )
    daily["Service"] = (
        df[service_col].astype(str).replace({"nan": "DATA_NOT_AVAILABLE"}).values
        if service_col else ["DATA_NOT_AVAILABLE"] * len(df)
    )
    daily["Actual Cost"] = clean_numeric(df[actual_col]).fillna(0.0)
    daily["Allocated Budget"] = 0.0
    daily["Variance"] = 0.0
    daily["Variance %"] = 0.0

    return daily.reset_index(drop=True)


def combine_filtered_data(product_filtered: pd.DataFrame, division_filtered: pd.DataFrame) -> pd.DataFrame:
    product_std = prepare_standardized_df(product_filtered, "Products")
    division_std = prepare_standardized_df(division_filtered, "Division")
    return pd.concat([product_std, division_std], ignore_index=True)


def combine_daily_spend_data(product_daily: pd.DataFrame, division_daily: pd.DataFrame) -> pd.DataFrame:
    product_std = prepare_daily_spend_df(product_daily, "Products")
    division_std = prepare_daily_spend_df(division_daily, "Division")
    return pd.concat([product_std, division_std], ignore_index=True)


def extract_named_filter(question: str, values: List[str]) -> Optional[str]:
    q = normalize_text(question)

    cleaned_values = []
    for value in values:
        if pd.isna(value):
            continue
        value_str = str(value).strip()
        if not value_str or value_str == "DATA_NOT_AVAILABLE":
            continue
        cleaned_values.append(value_str)

    unique_values = sorted(set(cleaned_values), key=len, reverse=True)

    for value in unique_values:
        value_norm = normalize_text(value)
        if value_norm and value_norm in q:
            return value

    return None


def extract_parent_filters(df: pd.DataFrame, question: str) -> Dict[str, str]:
    filters: Dict[str, str] = {}

    for column in ["Division", "Product", "Service", "Week", "Month", "Day"]:
        if column in df.columns:
            matched = extract_named_filter(question, df[column].dropna().astype(str).tolist())
            if matched:
                filters[column] = matched

    return filters


def get_requested_scope(question: str) -> str:
    q = normalize_text(question)

    if "internal" in q:
        return "division"
    if "product" in q or "products" in q:
        return "products"
    return "both"


def get_target_dimension(question: str) -> str:
    q = normalize_text(question)

    if "service" in q or "services" in q:
        return "Service"
    if "product" in q or "products" in q:
        return "Product"
    if "division" in q or "divisions" in q or "internal" in q:
        return "Division"
    if "day" in q or "daily" in q:
        return "Date"
    if "week" in q or "weekly" in q:
        return "Week"
    if "month" in q or "monthly" in q:
        return "Month"

    return "Dataset"


# =========================================================
# PREPARED INTERNAL ANALYSIS VIEWS
# =========================================================
def build_internal_actual_views(division_filtered: pd.DataFrame) -> Dict[str, Any]:
    """
    Build the 4 standard internal views from the filtered internal dataframe.

    Expected grain coming from Athena:
    - days -> date + division
    - weeks -> week + division
    - months -> month + division

    Important:
    This budget/actual mart does NOT contain service-level actuals.
    So service-based internal views are returned as unavailable placeholders.
    """
    division_std = prepare_standardized_df(division_filtered, "Division")

    if division_std.empty:
        return {
            "internal_company_total": {
                "actual_cost": 0.0,
                "allocated_budget": 0.0,
                "variance": 0.0,
                "variance_pct": 0.0,
                "status": "DATA_NOT_AVAILABLE",
            },
            "internal_by_division": pd.DataFrame(
                columns=["Entity", "Actual Cost", "Allocated Budget", "Variance", "Variance %", "Status"]
            ),
            "internal_by_service": pd.DataFrame(
                columns=["Service", "Actual Cost", "Allocated Budget", "Variance", "Variance %", "Status"]
            ),
            "internal_by_division_service": pd.DataFrame(
                columns=["Entity", "Service", "Actual Cost", "Allocated Budget", "Variance", "Variance %", "Status"]
            ),
        }

    actual_total = float(division_std["Actual Cost"].sum())
    budget_total = float(division_std["Allocated Budget"].sum())
    variance_total = float(division_std["Variance"].sum())
    variance_pct_total = safe_variance_pct(actual_total, budget_total)

    internal_company_total = {
        "actual_cost": actual_total,
        "allocated_budget": budget_total,
        "variance": variance_total,
        "variance_pct": variance_pct_total,
        "status": build_status_text(variance_total),
    }

    internal_by_division = (
        division_std.groupby("Entity", dropna=False)[["Actual Cost", "Allocated Budget", "Variance"]]
        .sum()
        .reset_index()
    )
    internal_by_division["Variance %"] = internal_by_division.apply(
        lambda row: safe_variance_pct(row["Actual Cost"], row["Allocated Budget"]),
        axis=1,
    )
    internal_by_division["Status"] = internal_by_division["Variance"].apply(build_status_text)
    internal_by_division = internal_by_division.sort_values("Actual Cost", ascending=False).reset_index(drop=True)

    internal_by_service = pd.DataFrame(
        [
            {
                "Service": "DATA_NOT_AVAILABLE",
                "Actual Cost": 0.0,
                "Allocated Budget": 0.0,
                "Variance": 0.0,
                "Variance %": 0.0,
                "Status": "Service-level internal actuals are not available from the current Athena selection.",
            }
        ]
    )

    internal_by_division_service = pd.DataFrame(
        [
            {
                "Entity": "DATA_NOT_AVAILABLE",
                "Service": "DATA_NOT_AVAILABLE",
                "Actual Cost": 0.0,
                "Allocated Budget": 0.0,
                "Variance": 0.0,
                "Variance %": 0.0,
                "Status": "Division + service internal actuals are not available from the current Athena selection.",
            }
        ]
    )

    return {
        "internal_company_total": internal_company_total,
        "internal_by_division": internal_by_division,
        "internal_by_service": internal_by_service,
        "internal_by_division_service": internal_by_division_service,
    }


def build_internal_budget_views(division_filtered: pd.DataFrame) -> Dict[str, Any]:
    """
    Build the 4 standard internal budget views from the filtered internal dataframe.
    """
    division_std = prepare_standardized_df(division_filtered, "Division")

    if division_std.empty:
        return {
            "internal_budget_company_total": {
                "actual_cost": 0.0,
                "allocated_budget": 0.0,
                "variance": 0.0,
                "variance_pct": 0.0,
                "status": "DATA_NOT_AVAILABLE",
            },
            "internal_budget_by_division": pd.DataFrame(
                columns=["Entity", "Actual Cost", "Allocated Budget", "Variance", "Variance %", "Status"]
            ),
            "internal_variance_company_total": {
                "variance": 0.0,
                "variance_pct": 0.0,
                "status": "DATA_NOT_AVAILABLE",
            },
            "internal_variance_by_division": pd.DataFrame(
                columns=["Entity", "Variance", "Variance %", "Status"]
            ),
        }

    actual_total = float(division_std["Actual Cost"].sum())
    budget_total = float(division_std["Allocated Budget"].sum())
    variance_total = float(division_std["Variance"].sum())
    variance_pct_total = safe_variance_pct(actual_total, budget_total)

    internal_budget_company_total = {
        "actual_cost": actual_total,
        "allocated_budget": budget_total,
        "variance": variance_total,
        "variance_pct": variance_pct_total,
        "status": build_status_text(variance_total),
    }

    internal_budget_by_division = (
        division_std.groupby("Entity", dropna=False)[["Actual Cost", "Allocated Budget", "Variance"]]
        .sum()
        .reset_index()
    )
    internal_budget_by_division["Variance %"] = internal_budget_by_division.apply(
        lambda row: safe_variance_pct(row["Actual Cost"], row["Allocated Budget"]),
        axis=1,
    )
    internal_budget_by_division["Status"] = internal_budget_by_division["Variance"].apply(build_status_text)
    internal_budget_by_division = internal_budget_by_division.sort_values("Variance", ascending=False).reset_index(drop=True)

    internal_variance_company_total = {
        "variance": variance_total,
        "variance_pct": variance_pct_total,
        "status": build_status_text(variance_total),
    }

    internal_variance_by_division = internal_budget_by_division[
        ["Entity", "Variance", "Variance %", "Status"]
    ].copy()

    return {
        "internal_budget_company_total": internal_budget_company_total,
        "internal_budget_by_division": internal_budget_by_division,
        "internal_variance_company_total": internal_variance_company_total,
        "internal_variance_by_division": internal_variance_by_division,
    }


def prepare_internal_analysis_context(division_filtered: pd.DataFrame) -> Dict[str, Any]:
    return {
        "internal_actual": build_internal_actual_views(division_filtered),
        "internal_budget": build_internal_budget_views(division_filtered),
    }


# =========================================================
# ANALYTICAL HELPERS
# =========================================================
def build_executive_comparison_summary(grouped: pd.DataFrame, metric: str) -> str:
    if grouped.shape[0] < 2:
        return "The selected period does not contain enough distinct scopes to produce an executive comparison."

    working = grouped.copy()
    working["Dataset"] = working["Dataset"].replace(
        {
            "division": "Division",
            "products": "Products",
        }
    )

    if metric == "Variance %":
        top_row = working.sort_values("Variance %", ascending=False).iloc[0]
        bottom_row = working.sort_values("Variance %", ascending=True).iloc[0]
        return (
            f"The primary variance pressure is coming from {top_row['Dataset']}, at "
            f"{format_percent(top_row['Variance %'], force_sign=True)}. "
            f"{bottom_row['Dataset']} is comparatively lower at "
            f"{format_percent(bottom_row['Variance %'], force_sign=True)}."
        )

    top_row = working.sort_values(metric, ascending=False).iloc[0]
    bottom_row = working.sort_values(metric, ascending=True).iloc[0]

    return (
        f"{top_row['Dataset']} is the main driver of {metric.lower()} in the selected period, at "
        f"{format_currency(top_row[metric], force_sign=(metric == 'Variance'))}. "
        f"{bottom_row['Dataset']} is materially lower at "
        f"{format_currency(bottom_row[metric], force_sign=(metric == 'Variance'))}."
    )


# =========================================================
# ANALYTICAL QUESTION ROUTER
# =========================================================
ANALYTICAL_KEYWORDS = [
    "variance",
    "trend",
    "trends",
    "over budget",
    "under budget",
    "this week",
    "last week",
    "this month",
    "last month",
    "today",
    "yesterday",
    "how much",
    "which division",
    "which product",
    "which team",
    "total spend",
    "cost during",
    "increase during",
    "decrease during",
    "compare",
    "top spend",
    "highest cost",
    "lowest cost"
]

GENERAL_KEYWORDS = [
    "what is finops",
    "best practice",
    "best practices",
    "recommend",
    "recommendation",
    "recommendations",
    "optimize",
    "optimization",
    "governance",
    "waste",
    "rightsizing",
    "showback",
    "chargeback",
    "allocation",
    "tagging",
    "what should we do",
    "how can we improve",
    "why does this matter",
    "what does this mean",
    "good finops habits",
    "core principles",
    "principles of finops",
    "unit economics",
    "cloud cost allocation",
    "showback vs chargeback",
    "cfo decision making",
    "control cloud spending",
    "main drivers of cloud cost growth",
    "common finops kpis",
    "cost optimization in finops",
    "finops lifecycle",
    "teams collaborate in finops",
    "difference between forecasting and budgeting",
    "cloud financial management",
    "finops maturity stages"
]

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text



def detect_metric(question: str) -> str:
    q = question.lower()

    if "variance %" in q or "variance percent" in q or "percentage" in q or "percent" in q:
        return "Variance %"
    if "variance" in q or "over budget" in q or "under budget" in q:
        return "Variance"
    if "budget" in q and "actual" not in q and "cost" not in q and "spend" not in q:
        return "Allocated Budget"
    if "actual" in q or "actual cost" in q or "spend" in q or "cost" in q:
        return "Actual Cost"

    return "Actual Cost"

def classify_question(question: str) -> str:
    """
    LLM-based router.
    Returns one of: general, analytical, hybrid

    Routing logic:
    - general: answer requires only FinOps knowledge documents / RAG
    - analytical: answer requires only selected-period data
    - hybrid: answer requires both knowledge documents and selected-period data
    """
    client = OpenAI()

    router_prompt = """
You are a strict routing classifier for a FinOps CFO assistant.

Your task:
Classify the user's question into exactly one route:
- general
- analytical
- hybrid

Definitions:
1. general
   Use this when the question can be answered using FinOps knowledge, concepts,
   best practices, governance, recommendations, definitions, or interpretation,
   WITHOUT needing selected-period data.

2. analytical
   Use this when the question can be answered using only the selected-period data,
   such as actual cost, budget, variance, top/bottom entities, comparisons,
   rankings, or period-specific metrics, WITHOUT needing FinOps document knowledge.

3. hybrid
   Use this when the question requires BOTH:
   - selected-period data
   - FinOps knowledge / best practices / interpretation / recommendations

Important rules:
- If the question contains two sub-questions and one is general while the other is analytical,
  classify as hybrid.
- If the user asks for interpretation or recommended action based on selected-period results,
  classify as hybrid.
- If the user asks only for a concept, habit, practice, principle, explanation, or recommendation,
  classify as general.
- If the user asks only for a fact or metric from the selected period, classify as analytical.
- Be conservative and precise.
- Output JSON only.

Return exactly this JSON shape:
{
  "route": "general|analytical|hybrid",
  "reason": "short explanation",
  "subquestions": [
    {"type": "general|analytical", "text": "subquestion text"}
  ]
}

Rules for subquestions:
- If route is general, you may return one general subquestion or an empty list.
- If route is analytical, you may return one analytical subquestion or an empty list.
- If route is hybrid, return at least:
  - one general subquestion
  - one analytical subquestion
""".strip()

    user_prompt = f"Question:\n{question}"

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": router_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = completion.choices[0].message.content.strip()
        parsed = json.loads(raw)

        route = str(parsed.get("route", "")).strip().lower()

        if route in {"general", "analytical", "hybrid"}:
            return route

        return fallback_classify_question(question)

    except Exception:
        return fallback_classify_question(question)

def fallback_classify_question(question: str) -> str:
    """
    Lightweight fallback if the LLM router fails.
    Still source-based, but simpler than the old keyword-only approach.
    """
    q = normalize_text(question)

    general_signals = [
        "what is",
        "best practice",
        "best practices",
        "good fin ops habits",
        "good finops habits",
        "bad fin ops habits",
        "bad finops habits",
        "recommend",
        "recommendation",
        "governance",
        "showback",
        "chargeback",
        "allocation",
        "tagging",
        "rightsizing",
        "waste",
        "optimization",
        "principle",
        "principles",
        "habit",
        "habits",
        "why does this matter",
        "what does this mean",
        "how can we improve",
    ]

    analytical_signals = [
        "during this period",
        "during the selected period",
        "this period",
        "that period",
        "actual cost",
        "budget",
        "variance",
        "highest",
        "lowest",
        "top",
        "bottom",
        "most",
        "least",
        "which service",
        "which product",
        "which division",
        "which day",
        "total spend",
        "cost",
        "spend",
        "over budget",
        "under budget",
        "compare",
        "vs",
        "versus",
    ]

    has_general = any(signal in q for signal in general_signals)
    has_analytical = any(signal in q for signal in analytical_signals)

    if has_general and has_analytical:
        return "hybrid"
    if has_analytical:
        return "analytical"
    return "general"   

def detect_dimension(question: str) -> str:
    q = question.lower()

    if "day" in q or "daily" in q:
        return "Date"

    if "service" in q or "services" in q:
        return "Service"

    if "product" in q or "products" in q:
        return "Product"

    if "division" in q or "divisions" in q or "internal" in q:
        return "Division"

    if "week" in q or "weekly" in q:
        return "Week"

    if "month" in q or "monthly" in q:
        return "Month"

    return "Dataset"


def filter_question_scope(df: pd.DataFrame, question: str) -> pd.DataFrame:
    scoped = df.copy()

    requested_scope = get_requested_scope(question)

    scoped["Dataset"] = scoped["Dataset"].astype(str).str.strip().str.lower()

    if requested_scope == "products":
        scoped = scoped[scoped["Dataset"] == "products"].copy()
    elif requested_scope == "division":
        scoped = scoped[scoped["Dataset"] == "division"].copy()

    parent_filters = extract_parent_filters(scoped, question)

    for column, value in parent_filters.items():
        if column in scoped.columns:
            scoped = scoped[scoped[column].astype(str).str.strip() == str(value).strip()].copy()

    return scoped


def aggregate_by_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if dimension not in df.columns:
        dimension = "Dataset"

    working = df.copy()

    if dimension in ["Service", "Product", "Division"]:
        working = working[working[dimension].astype(str) != "DATA_NOT_AVAILABLE"].copy()

    available_metrics = [col for col in ["Actual Cost", "Allocated Budget", "Variance"] if col in working.columns]

    grouped = (
    
        working.groupby(dimension, dropna=False)[available_metrics]
        .sum()
        .reset_index()
    )

    if "Actual Cost" in grouped.columns and "Allocated Budget" in grouped.columns:
   
        grouped["Variance %"] = grouped.apply(
            lambda row: safe_variance_pct(row["Actual Cost"], row["Allocated Budget"]),
            axis=1
        )

    return grouped


def build_scope_label(question: str) -> str:
    q = normalize_text(question)

    if "day" in q or "daily" in q:
        return "Days"
    if "service" in q and "internal" in q:
        return "Services within Internal"
    if "service" in q and "division" in q:
        return "Services within the requested division"
    if "service" in q and "product" in q:
        return "Services within the requested product"
    if "product" in q and "division" in q:
        return "Products within the requested division"
    if "service" in q or "services" in q:
        return "Services"
    if "division" in q or "divisions" in q or "internal" in q:
        return "Divisions"
    if "product" in q or "products" in q:
        return "Products"

    return "Products and Divisions combined"


def is_why_budget_question(question: str) -> bool:
    q = question.lower()

    why_signals = ["why", "what caused", "what drove", "what explains", "explain", "reason"]
    budget_signals = ["over budget", "under budget", "variance", "budget"]

    return any(signal in q for signal in why_signals) and any(signal in q for signal in budget_signals)


def is_day_spend_question(question: str) -> bool:
    q = normalize_text(question)
    return (
        ("day" in q or "daily" in q)
        and ("spend" in q or "cost" in q or "actual" in q)
        and any(term in q for term in ["highest", "lowest", "top", "bottom", "most", "least"])
    )


def answer_total_question(df: pd.DataFrame, metric: str, scope_label: str) -> str:
    if metric == "Variance %":
        total_value = safe_variance_pct(df["Actual Cost"].sum(), df["Allocated Budget"].sum())
        return (
            f"Within the selected period, the total variance percentage for {scope_label} is "
            f"{format_percent(total_value, force_sign=True)}."
        )

    total_value = df[metric].sum()
    return (
        f"Within the selected period, the total {metric.lower()} for {scope_label} is "
        f"{format_currency(total_value, force_sign=(metric == 'Variance'))}."
    )


def answer_average_question(df: pd.DataFrame, metric: str, scope_label: str) -> str:
    if metric == "Variance %":
        value = df["Variance %"].mean()
        return (
            f"Within the selected period, the average variance percentage for {scope_label} is "
            f"{format_percent(value, force_sign=True)}."
        )

    value = df[metric].mean()
    return (
        f"Within the selected period, the average {metric.lower()} for {scope_label} is "
        f"{format_currency(value, force_sign=(metric == 'Variance'))}."
    )


def answer_why_budget_question(df: pd.DataFrame, scope_label: str) -> str:
    total_actual = df["Actual Cost"].sum()
    total_budget = df["Allocated Budget"].sum()
    total_variance = df["Variance"].sum()
    total_variance_pct = safe_variance_pct(total_actual, total_budget)
    status = build_status_text(total_variance)

    lines = [
        f"{scope_label} is {status.lower()} during the selected period.",
        f"- Actual Cost: {format_currency(total_actual)}",
        f"- Allocated Budget: {format_currency(total_budget)}",
        f"- Variance: {format_currency(total_variance, force_sign=True)}",
        f"- Variance %: {format_percent(total_variance_pct, force_sign=True)}",
    ]

    if total_variance > 0:
        lines.append("The analytical result shows that spending exceeded the allocated budget.")
    elif total_variance < 0:
        lines.append("The analytical result shows that spending remained below the allocated budget.")
    else:
        lines.append("The analytical result shows that spending matched the allocated budget.")

    return "\n".join(lines)


def answer_compare_question(df: pd.DataFrame, metric: str) -> str:
    grouped = aggregate_by_dimension(df, "Dataset")

    if grouped.shape[0] < 2:
        return "The selected period does not contain enough distinct scopes to produce a comparison."

    grouped["Dataset"] = grouped["Dataset"].replace({"division": "Division", "products": "Products"})

    lines = ["Executive comparison summary:"]
    lines.append(build_executive_comparison_summary(grouped, metric))
    lines.append("")
    lines.append("Detailed comparison:")

    for _, row in grouped.iterrows():
        if metric == "Variance %":
            metric_value = format_percent(row["Variance %"], force_sign=True)
        else:
            metric_value = format_currency(row[metric], force_sign=(metric == "Variance"))
        lines.append(f"- {row['Dataset']}: {metric_value}")

    return "\n".join(lines)


def build_ranking_sentence(n: int, order: str, scope_label: str, dimension: str) -> str:
    if dimension == "Date":
        if n == 1:
            if order == "top":
                return "The highest-cost day during the selected period was:"
            return "The lowest-cost day during the selected period was:"
        if order == "top":
            return f"The top {n} highest-cost days during the selected period were:"
        return f"The top {n} lowest-cost days during the selected period were:"

    singular_map = {
        "Services": "service",
        "Products": "product",
        "Divisions": "division",
        "Products and Divisions combined": "scope",
    }

    plural_map = {
        "Services": "services",
        "Products": "products",
        "Divisions": "divisions",
        "Products and Divisions combined": "scopes",
    }

    singular_label = singular_map.get(scope_label, scope_label.lower().rstrip("s"))
    plural_label = plural_map.get(scope_label, scope_label.lower())

    if n == 1:
        if order == "top":
            return f"The highest-cost {singular_label} during the selected period was:"
        return f"The lowest-cost {singular_label} during the selected period was:"

    if order == "top":
        return f"The top {n} highest-cost {plural_label} during the selected period were:"
    return f"The top {n} lowest-cost {plural_label} during the selected period were:"

def extract_top_n(question: str, default: int = 3) -> int:
    import re

    q = question.lower().strip()

    # Single highest/lowest case
    if any(term in q for term in ["highest", "lowest", "most", "least"]):
        return 1

    # Handle "top service", "top product" etc → default to 1
    if re.search(r"\btop\s+(service|services|product|products|division|divisions|day|days)\b", q):
        return 1

    # Handle "top N"
    match = re.search(r"\btop\s+(\d+)\b", q)
    if match:
        return int(match.group(1))

    # Handle "bottom N"
    match = re.search(r"\bbottom\s+(\d+)\b", q)
    if match:
        return int(match.group(1))

    return default

def answer_top_bottom_question(
    df: pd.DataFrame,
    question: str,
    metric: str,
    dimension: str,
    scope_label: str
) -> str:
    if dimension not in df.columns:
        return (
            f"I found the requested analytical scope, but the dimension '{dimension}' "
            f"is not available in the selected filtered dataset."
        )

    grouped = aggregate_by_dimension(df, dimension)
    n = extract_top_n(question, default=3)
    q = question.lower().replace("-", " ")

    singular_map = {
        "Services": "service",
        "Products": "product",
        "Divisions": "division",
        "Products and Divisions combined": "scope",
    }

    plural_map = {
        "Services": "services",
        "Products": "products",
        "Divisions": "divisions",
        "Products and Divisions combined": "scopes",
    }

    singular_label = singular_map.get(scope_label, scope_label.lower().rstrip("s"))
    plural_label = plural_map.get(scope_label, scope_label.lower())

    if grouped.empty:
        return (
            f"I found the requested analytical scope, but no grouped values are available "
            f"for {scope_label}. This usually means that the requested granularity is not "
            f"available in the current selected period dataset."
        )

    display_metric = metric

    if "over budget" in q:
        ranked = grouped[grouped["Variance"] > 0].sort_values("Variance", ascending=False).head(n)

        if n == 1:
            lines = [f"The most over-budget {singular_label} during the selected period was:"]
        else:
            lines = [f"The top {n} over-budget {plural_label} during the selected period were:"]

        display_metric = "Variance"

    elif "under budget" in q:
        ranked = grouped[grouped["Variance"] < 0].sort_values("Variance", ascending=True).head(n)

        if n == 1:
            lines = [f"The most under-budget {singular_label} during the selected period was:"]
        else:
            lines = [f"The top {n} under-budget {plural_label} during the selected period were:"]

        display_metric = "Variance"

    elif "bottom" in q or "lowest" in q or "least" in q:
        ranked = grouped.sort_values(metric, ascending=True).head(n)
        lines = [build_ranking_sentence(n, "bottom", scope_label, dimension)]

    else:
        ranked = grouped.sort_values(metric, ascending=False).head(n)
        lines = [build_ranking_sentence(n, "top", scope_label, dimension)]

    if ranked.empty:
        if "under budget" in q:
            return f"No {plural_label} were under budget during the selected period."
        if "over budget" in q:
            return f"No {plural_label} were over budget during the selected period."

        return (
            f"I found the requested analytical scope, but no grouped values are available "
            f"for {scope_label}."
        )

    for _, row in ranked.iterrows():
        item_name = row[dimension]

        if dimension == "Dataset":
            item_name = str(item_name).replace("division", "Division").replace("products", "Products")

        value_str = (
            format_percent(row[display_metric], force_sign=True)
            if display_metric == "Variance %"
            else format_currency(row[display_metric], force_sign=(display_metric == "Variance"))
        )

        lines.append(f"- {item_name}: {value_str}")

    return "\n".join(lines)





def get_base_dataset(dimension, product_df, division_df):
    if dimension == "service":
        return division_df.copy()

    if dimension == "product":
        return product_df.copy()

    if dimension == "division":
        return division_df.copy()
    
    return combine_filtered_data(product_df, division_df)


def standardize_columns(df):
    df = df.copy()

    df = df.rename(columns={
        "service": "Service",
        "product": "Product",
        "division": "Division",
        "actual_cost_usd": "Actual Cost",
        "allocated_budget_usd": "Allocated Budget",
    })

    if "Actual Cost" in df.columns:
        df["Actual Cost"] = pd.to_numeric(df["Actual Cost"], errors="coerce").fillna(0)

    if "Allocated Budget" in df.columns:
        df["Allocated Budget"] = pd.to_numeric(df["Allocated Budget"], errors="coerce").fillna(0)

    return df


def answer_analytical_question(
    question: str,
    product_filtered: pd.DataFrame,
    division_filtered: pd.DataFrame,
    product_daily_spend: Optional[pd.DataFrame] = None,
    division_daily_spend: Optional[pd.DataFrame] = None,
) -> str:
    q = question.lower()
    metric = detect_metric(question)
    dimension = detect_dimension(question).lower()
    base_df = get_base_dataset(dimension, product_filtered, division_filtered)

    if base_df is None or base_df.empty:
        return "No data available for the selected period."

    base_df = standardize_columns(base_df)
    scope_label = build_scope_label(question)
    if "versus" in q or "vs" in q:
        scope = standardize_columns(base_df.copy())

        if scope.empty:
            return "No rows matched the selected period."

        actual = float(scope["Actual Cost"].sum())
        budget = float(scope["Allocated Budget"].sum())
        variance = actual - budget

        return (
            f"Actual cost is {format_currency(actual)}, "
            f"compared to a budget of {format_currency(budget)}, "
            f"resulting in a variance of {format_currency(variance, force_sign=True)}."
    )

    total_scope = standardize_columns(product_filtered.copy())

    if "Variance" not in total_scope.columns and "Actual Cost" in total_scope.columns and "Allocated Budget" in total_scope.columns:
        total_scope["Variance"] = total_scope["Actual Cost"] - total_scope["Allocated Budget"]

    if "most important numerical insight" in q:
        actual = float(total_scope["Actual Cost"].sum())
        budget = float(total_scope["Allocated Budget"].sum())
        variance = actual - budget
        return (
            f"The most important insight is a variance of "
            f"{format_currency(variance, force_sign=True)} "
            f"({'over' if variance > 0 else 'under'} budget)."
        )

    if "budget" in q and "total" in q:
        total_budget = float(total_scope["Allocated Budget"].sum())
        return f"The total allocated budget is {format_currency(total_budget)}."

    if "variance" in q and ("dollar" in q or "$" in q):
        total_variance = float(total_scope["Variance"].sum())
        return f"The total variance is {format_currency(total_variance, force_sign=True)}."

    if "variance" in q and ("percent" in q or "percentage" in q or "%" in q):
        actual = float(total_scope["Actual Cost"].sum())
        budget = float(total_scope["Allocated Budget"].sum())

        if budget == 0:
            return "Cannot compute variance percentage."

        pct = ((actual - budget) / budget) * 100
        return f"The variance percentage is {format_percent(pct, force_sign=True)}."

    if "overage" in q or "over budget" in q or "overspend" in q:
        overage = float(total_scope["Variance"].sum())
        return f"The budget overage is {format_currency(overage, force_sign=True)}."

    if "average" in q or "avg" in q:
        avg_value = float(base_df["Actual Cost"].mean())
        return f"The average actual cost is {format_currency(avg_value)}."

    if "versus" in q or "vs" in q:
        return answer_compare_question(base_df, "Actual Cost")

    if "driver" in q:
        scope = standardize_columns(base_df.copy())

        if scope.empty:
            return "No rows matched the selected period."

        grouped = scope.groupby("Product", as_index=False)["Actual Cost"].sum()
        top = grouped.sort_values("Actual Cost", ascending=False).head(3)

        return (
            "The top 3 cost drivers are: "
            + ", ".join(
                f"{row['Product']} ({format_currency(row['Actual Cost'])})"
                for _, row in top.iterrows()
            )
            + ". The CFO should prioritize these drivers."
        )

    if "summary" in q or "insight" in q or "important" in q or "performance" in q:
        total_scope = standardize_columns(base_df.copy())

        if total_scope.empty:
            return "No rows matched the selected period."

        actual = float(total_scope["Actual Cost"].sum())
        budget = float(total_scope["Allocated Budget"].sum())
        variance = actual - budget

        return (
            f"During the selected period, total spend reached {format_currency(actual)} "
            f"against a budget of {format_currency(budget)}, resulting in a variance of "
            f"{format_currency(variance, force_sign=True)}. "
            f"This indicates {'overspending' if variance > 0 else 'cost control'}, "
            f"and represents a financial performance issue that requires attention. "
            f"The CFO should prioritize investigating the main cost drivers and enforcing stronger cost controls."
        )
        

            
    if dimension in ["service", "product", "division"]:
        working_scope = base_df.copy()

        if working_scope.empty:
            return "No rows matched the selected period for the requested analytical scope."

        dim_col_map = {
            "service": "Service",
            "product": "Product",
            "division": "Division",
        }

        dim_col = dim_col_map[dimension]

        if dim_col not in working_scope.columns:
            return (
                f"I found the requested analytical scope, but the dimension '{dim_col}' "
                f"is not available in the current dataset."
            )

        working_scope = working_scope[
            working_scope[dim_col].astype(str) != "DATA_NOT_AVAILABLE"
        ].copy()

        if working_scope.empty:
            return (
                f"I found the requested analytical scope, but no valid '{dim_col}' "
                f"data is available for the selected period."
            )

        working_scope = standardize_columns(working_scope)

        if "Variance" not in working_scope.columns and "Actual Cost" in working_scope.columns and "Allocated Budget" in working_scope.columns:
            working_scope["Variance"] = working_scope["Actual Cost"] - working_scope["Allocated Budget"]

        available_cols = [
            col for col in ["Actual Cost", "Allocated Budget", "Variance"]
            if col in working_scope.columns
        ]

        working_scope = working_scope.groupby(dim_col, as_index=False)[available_cols].sum()

        return answer_top_bottom_question(
            working_scope,
            question,
            metric,
            dim_col,
            scope_label,
        )

    if dimension == "dataset":
        actual = float(total_scope["Actual Cost"].sum())
        budget = float(total_scope["Allocated Budget"].sum())
        variance = actual - budget
        variance_pct = safe_variance_pct(actual, budget)

        return (
            f"Selected-period analytical summary:\n"
            f"- Total actual cost: {format_currency(actual)}\n"
            f"- Allocated budget: {format_currency(budget)}\n"
            f"- Variance: {format_currency(variance, force_sign=True)} "
            f"({format_percent(variance_pct, force_sign=True)})"
        )

    return "No supported analytical route matched this question."


# =========================================================
# GENERAL RAG
# =========================================================
def load_chunks(input_file: Path) -> List[Dict[str, Any]]:
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("chunks_general_questions.json must contain a list of chunk objects.")

    return chunks


def build_bm25(chunks: List[Dict[str, Any]]) -> tuple[BM25Okapi, List[str]]:
    chunk_texts = [str(chunk.get("text", "")) for chunk in chunks]
    tokenized_chunks = [text.lower().split() for text in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25, chunk_texts


def semantic_search(
    question: str,
    chunk_texts: List[str],
    embedding_model: SentenceTransformer,
    top_k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    question_embedding = embedding_model.encode(question, normalize_embeddings=True)
    chunk_embeddings = embedding_model.encode(chunk_texts, normalize_embeddings=True)
    semantic_similarities = np.dot(chunk_embeddings, question_embedding)
    top_indices = np.argsort(semantic_similarities)[-top_k:][::-1]
    return semantic_similarities, top_indices


def lexical_search(question: str, bm25: BM25Okapi, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    return bm25_scores, top_indices


def rerank_results(
    question: str,
    candidate_indices: List[int],
    chunks: List[Dict[str, Any]],
    semantic_similarities: np.ndarray,
    bm25_scores: np.ndarray,
    reranker: CrossEncoder
) -> List[Dict[str, Any]]:
    if not candidate_indices:
        return []

    pairs = [(question, str(chunks[i].get("text", ""))) for i in candidate_indices]
    rerank_scores = reranker.predict(pairs)

    reranked_results = []
    for idx, rerank_score in zip(candidate_indices, rerank_scores):
        reranked_results.append(
            {
                "index": int(idx),
                "text": str(chunks[idx].get("text", "")),
                "filename": str(chunks[idx].get("filename", "DATA_NOT_AVAILABLE")),
                "chunk_id": str(chunks[idx].get("chunk_id", "DATA_NOT_AVAILABLE")),
                "semantic_score": float(semantic_similarities[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "rerank_score": float(rerank_score),
            }
        )

    reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked_results


def build_context_from_results(results: List[Dict[str, Any]], max_chunks: int = 3) -> str:
    selected = results[:max_chunks]
    blocks = []

    for i, result in enumerate(selected, start=1):
        blocks.append(
            f"[Source {i}] File: {result['filename']} | Chunk: {result['chunk_id']}\n"
            f"{result['text']}"
        )

    return "\n\n".join(blocks)


def infer_confidence(top_results: List[Dict[str, Any]]) -> Dict[str, str]:
    if not top_results:
        return {"label": "Low", "reason": "No retrieval results found"}

    top_score = top_results[0]["rerank_score"]

    if top_score >= 0.60:
        return {"label": "High", "reason": "Strong reranker match"}
    if top_score >= 0.45:
        return {"label": "Medium", "reason": "Moderate reranker match"}
    return {"label": "Low", "reason": "Weak reranker match"}


def call_openai_with_context(question: str, context: str) -> str:

    try:
        client = OpenAI()

        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {

                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Retrieved context:\n{context}\n\n"
                        """
Answer strictly from the retrieved context only.

Use this structure EXACTLY:

Executive FinOps Interpretation:
<short executive paragraph>

Key FinOps Habits:
- <habit 1>
- <habit 2>
- <habit 3>
- <habit 4>

Recommended CFO Actions:
- <action 1>
- <action 2>
- <action 3>

Rules:
- Do NOT use numbers (no 1., 2., 3.)
- Use ONLY bullet points for lists
- Keep it concise and CFO-level

Keep the answer:
- concise
- business-focused
- decision-oriented
- grounded in the retrieved context
"""
                ),
            },
        ],
    )
        return completion.choices[0].message.content.strip()
            
    except Exception as e:
        print("OPENAI ERROR:", str(e))

        return "I could not generate a response due to a temporary system issue."



def handle_general_question(
    question: str,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder
) -> Dict[str, Any]:
    chunks = load_chunks(INPUT_FILE)
    bm25, chunk_texts = build_bm25(chunks)

    semantic_similarities, semantic_top = semantic_search(
        question=question,
        chunk_texts=chunk_texts,
        embedding_model=embedding_model,
        top_k=5
    )
    bm25_scores, lexical_top = lexical_search(question=question, bm25=bm25, top_k=5)

    candidate_indices = list(dict.fromkeys(list(semantic_top) + list(lexical_top)))
    reranked_results = rerank_results(
        question=question,
        candidate_indices=candidate_indices,
        chunks=chunks,
        semantic_similarities=semantic_similarities,
        bm25_scores=bm25_scores,
        reranker=reranker,
    )

    confidence = infer_confidence(reranked_results)
    top_results = reranked_results[:3]

    if not top_results or confidence["label"] == "Low":
        return {
            "question": question,
            "answer": "I don't have enough information in the retrieved documents.",
            "top_results": top_results,
            "confidence": confidence,
            "sources": "No sufficiently reliable retrieved sources",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fallback": True,
        }

    context = build_context_from_results(top_results, max_chunks=3)
    answer = call_openai_with_context(question, context)
    answer = handle_rag_failure(answer)

    sources = "; ".join(
        [f"{item['filename']} (chunk {item['chunk_id']})" for item in top_results]
    )

    fallback_phrases = [
        "i don't have enough information",
        "not enough information",
        "do not provide",
        "not specifically define",
        "not defined",
        "no relevant information"
    ]

    answer_lower = answer.lower()

    fallback = any(phrase in answer_lower for phrase in fallback_phrases)

    if fallback:
        answer = """I don't have enough information in the retrieved documents."    
    CFO Note:
    This question highlights a documentation gap in the current FinOps knowledge base.

    Recommendation:
    Consider enriching the documentation with clear definitions, KPIs, or governance guidance on this topic to support better financial decision-making and automation in the future.
    """

    return {
        "question": question,
        "answer": answer,
        "top_results": top_results,
        "confidence": confidence,
        "sources": sources,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fallback": fallback,
}

# =========================================================
# HYBRID LOGIC
# =========================================================
def build_hybrid_finops_query(question: str) -> str:
    q = question.lower()


    if "finops habits" in q or "habits should we improve" in q:
        return "What are good FinOps habits?"

    if "best practices" in q or "what should we improve" in q:
        return "What FinOps best practices should we implement?"

    if "how can we improve finops" in q or "improve finops" in q:
        return "What FinOps best practices should we implement?"

    if is_why_budget_question(question):
        if "product" in q and all(x not in q for x in ["division", "divisions", "internal"]):
            return (
                "What FinOps best practices should a CFO prioritize when Products are over budget "
                "during a selected period, including budget variance review, cost allocation validation, "
                "anomaly investigation, and accountability?"
            )
        if "division" in q or "divisions" in q or "internal" in q:
            if "product" not in q:
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
        return "How should a CFO interpret average cost or average variance over a selected period in FinOps?"

    if "total" in q or "sum" in q:
        return "How should a CFO interpret total spend, budget, and variance over a selected period in FinOps?"

    return "What FinOps best practices are most relevant to interpret a selected-period budget and spend result?"


def extract_general_subquestion(question: str) -> str:
    q = question.strip()
    q_lower = normalize_text(q)

    q_lower = q_lower.replace("fin ops", "finops")

    if "finops habits" in q_lower or "good finops habits" in q_lower:
        return "What are good FinOps habits a CFO should prioritize?"

    if "implement good finops habits" in q_lower:
        return "What FinOps best practices should a CFO prioritize?"

    if "finops best practices" in q_lower or "best practices" in q_lower:
        return "What FinOps best practices should a CFO prioritize?"

    if "what should a cfo do" in q_lower or "what do you suggest a cfo do" in q_lower:
        return "What FinOps best practices should a CFO prioritize?"

    if "how can we improve finops" in q_lower or "improve finops" in q_lower:
        return "What FinOps practices should a CFO improve first?"

    if "optimize" in q_lower or "optimization" in q_lower:
        return "What FinOps optimization practices should a CFO apply?"

    if "governance" in q_lower:
        return "What FinOps governance practices should a CFO apply?"

    return "What FinOps best practices should a CFO prioritize?"

def llm_rewrite_question_for_rag(question: str) -> str:
    prompt = f"""
You are a FinOps expert.

Rewrite the user's question into a clear FinOps question
that matches documentation about:
- FinOps best practices
- cost optimization
- governance
- forecasting
- accountability
- Correct grammar and wording mistakes.
- Rewrite unclear or incorrect questions into clear CFO-level questions.
- If the question is about "best FinOps practices", rewrite it as:
  "What are the most important FinOps best practices a CFO should prioritize?"

Rules:
- Keep it short
- Make it generic for retrieval
- Do NOT mention time periods or specific services

User question:
{question}

Rewritten question:
"""

    client = OpenAI()

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    return completion.choices[0].message.content.strip()


def build_contextual_finops_fallback(question: str) -> str:
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
            "The immediate CFO priority is to use the selected-period analytical result as the basis for review, "
            "ownership, and action."
        )

    return (
        "The retrieved documents do not provide enough specific FinOps guidance for this exact case. "
        "The immediate CFO priority is to use the selected-period analytical result as the basis for review, "
        "ownership, and action."
    )


def handle_hybrid_question(
    question: str,
    product_filtered: pd.DataFrame,
    division_filtered: pd.DataFrame,
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder,
    product_daily_spend: Optional[pd.DataFrame] = None,
    division_daily_spend: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:

    analytical_answer = answer_analytical_question(
        question=question,
        product_filtered=product_filtered,
        division_filtered=division_filtered,
        product_daily_spend=product_daily_spend,
        division_daily_spend=division_daily_spend,
    )

    general_subquestion = extract_general_subquestion(question)
    rag_question = build_hybrid_finops_query(general_subquestion)

    general_result = handle_general_question(
        rag_question,
        embedding_model,
        reranker
    )

    if general_result["fallback"]:
        rewritten_question = llm_rewrite_question_for_rag(question)
        print("DEBUG LLM REWRITE:", rewritten_question)

        general_result = run_multi_query_rag(
            question=rag_question,
            general_subquestion=general_subquestion,
            embedding_model=embedding_model,
            reranker=reranker
        )
    
    if general_result["fallback"]:

        general_answer = (
            "Executive FinOps Interpretation:\n"
            "The retrieved documents do not provide enough specific guidance for this exact question. "
            "Based on the selected-period analysis, the CFO should focus on practical FinOps controls, "
            "cost visibility, ownership, and timely financial decision-making.\n\n"
            "Key FinOps Habits:\n"
            "- Review cost and usage together, not invoices alone\n"
            "- Connect technology spend to business value\n"
            "- Use budgets, forecasts, and variance reviews together\n"
            "- Maintain clear ownership for product and service cost drivers\n\n"
            "Recommended CFO Actions:\n"
            "- Review the top cost drivers for the selected period\n"
            "- Assign ownership for high-variance products and services\n"
            "- Strengthen allocation quality by product, team, environment, and owner\n"
            "- Run a recurring FinOps review with finance, engineering, and product teams"
    )
    else:
        general_answer = general_result["answer"]
    

    def clean_rag_output(text: str) -> str:
        import re

        lines = text.splitlines()
        cleaned = []

        skip_headers = {
            "1. executive answer",
            "2. key insights",
            "3. cfo actions",
        }

        for line in lines:
            stripped = line.strip()
            normalized = stripped.lower()

            if normalized in skip_headers:
                continue

            stripped = re.sub(r"^\d+\.\s*", "", stripped)

            if not stripped:
                continue

            cleaned.append(stripped)

        return "\n".join(cleaned).strip()

    def format_sections(text: str) -> str:
        import re

        text = text.strip()

        # Force spacing before major section headers
        text = re.sub(
            r"\s*(Executive FinOps Interpretation:)\s*",
            r"\1\n\n",
            text
        )

        text = re.sub(
            r"\s*(Key FinOps Habits:)\s*",
            r"\n\n\1\n\n",
            text
        )

        text = re.sub(
            r"\s*(Recommended CFO Actions:)\s*",
            r"\n\n\1\n\n",
            text
        )

        # Clean excessive spacing but keep double paragraph breaks
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        return text.strip()


    clean_general_answer = general_answer.strip()

    clean_general_answer_for_hybrid = format_sections(
        clean_rag_output(clean_general_answer)
    )


    combined_answer = (
        "Hybrid CFO Answer\n\n"
        "Analytical Answer restricted to the selected period:\n\n"
        f"{analytical_answer}\n\n"
        f"{clean_general_answer_for_hybrid}\n\n"
        "CFO Priority:\n\n"
        "Use the analytical result first, then apply the FinOps interpretation to guide action."
    )

    return {
        "question": question,
        "answer": combined_answer,
        "top_results": general_result["top_results"],
        "confidence": general_result["confidence"],
        "sources": general_result["sources"],
        "timestamp": general_result["timestamp"],
        "fallback": general_result["fallback"],
    }
   
# =========================================================
# VOICE (FINAL CLEAN VERSION)
# =========================================================

AUDIO_END_PAUSE_MS = int(os.getenv("AUDIO_END_PAUSE_MS", "1200"))

def build_spoken_text(answer: str) -> str:
    fallback_text = "I don't have enough information in the retrieved documents."

    if answer.strip().startswith(fallback_text):
        return "I do not have enough information in the retrieved documents, to answer confidently."

    spoken = str(answer)

    # Normalize line breaks
    spoken = spoken.replace("\r\n", "\n").replace("\r", "\n")

    # Remove bullet markers (THIS is the key fix)
    spoken = re.sub(r"(?m)^\s*[-•]\s+", "", spoken)

    # Remove numbered lists (1. 2. etc.)
    spoken = re.sub(r"(?m)^\s*\d+\.\s+", "", spoken)

    # Remove markdown
    spoken = re.sub(r"\*\*(.*?)\*\*", r"\1", spoken)
    spoken = re.sub(r"__(.*?)__", r"\1", spoken)
    spoken = re.sub(r"`(.*?)`", r"\1", spoken)

    # Acronyms → spaced for speech
    acronym_map = {
        r"\bCFO\b": "C F O",
        r"\bAI\b": "A I",
        r"\bRAG\b": "R A G",
        r"\bAWS\b": "A W S",
        r"\bKPI\b": "K P I",
        r"\bFinOps\b": "Fin Ops",
        r"\bPower BI\b": "Power B I",
    }

    for pattern, replacement in acronym_map.items():
        spoken = re.sub(pattern, replacement, spoken)

    # Clean lines and build natural speech
    lines = []
    for raw_line in spoken.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # Ensure sentence ending
        if not re.search(r"[.!?]$", line):
            line += "."

        lines.append(line)

    # Join as natural speech
    spoken = " ".join(lines)
    spoken = re.sub(r"\s+", " ", spoken).strip()

    return spoken

def build_ssml(text: str) -> str:
    return f"<speak>{text}<break time='{AUDIO_END_PAUSE_MS}ms'/></speak>"


def text_to_speech(spoken_text: str, auto_open: bool = True) -> str:
    session = get_boto3_session()
    polly = session.client("polly")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"cfo_answer_{timestamp}.mp3"

    response = polly.synthesize_speech(
        Text=build_ssml(spoken_text),
        TextType="ssml",
        OutputFormat="mp3",
        VoiceId=VOICE_ID,
        Engine=ENGINE
    )

    with open(output_file, "wb") as f:
        f.write(response["AudioStream"].read())

    if auto_open and sys.platform.startswith("win"):
        try:
            os.startfile(str(output_file))
        except Exception:
            pass

    return str(output_file)

# =========================================================
# PIPELINE ENTRY FOR EXTERNAL CALL
# =========================================================
def run_finops_cfo_pipeline(
    selection: Dict[str, Any],
    question: str,
    enable_audio: bool = True,
    auto_open_audio: bool = True,
) -> Dict[str, Any]:

    try:
        start_time = time.time()

        division_filtered, product_filtered = load_selected_period_data(selection)
        division_daily_spend, product_daily_spend = load_selected_period_daily_spend(selection)

        internal_context = prepare_internal_analysis_context(division_filtered)
        route = classify_question(question)

        embedding_model = None
        reranker = None

        if route in {"general", "hybrid"}:
            embedding_model = embedding_model_global
            reranker = reranker_global

        if route == "general":
            general_subquestion = extract_general_subquestion(question)

            result = run_multi_query_rag(
                question=question,
                general_subquestion=general_subquestion,
                embedding_model=embedding_model,
                reranker=reranker,
            )

            result["route"] = "general"

        elif route == "analytical":
            analytical_answer = answer_analytical_question(
                question=question,
                product_filtered=product_filtered,
                division_filtered=division_filtered,
                product_daily_spend=product_daily_spend,
                division_daily_spend=division_daily_spend,
            )

            result = {
                "question": question,
                "answer": analytical_answer,
                "route": "analytical",
                "top_results": [],
                "confidence": {
                    "label": "Data-driven",
                    "reason": "Analytical answer from selected-period data",
                },
                "sources": "Selected-period product, division, and day-level spend datasets",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "fallback": False,
            }

        else:
            result = handle_hybrid_question(
                question=question,
                product_filtered=product_filtered,
                division_filtered=division_filtered,
                embedding_model=embedding_model,
                reranker=reranker,
                product_daily_spend=product_daily_spend,
                division_daily_spend=division_daily_spend,
            )

            result["route"] = "hybrid"

        result["prepared_views"] = {
            "internal": internal_context
        }

        result["audio_file"] = None

        if enable_audio:
            try:
                spoken_text = build_spoken_text(result["answer"])
                audio_file = text_to_speech(spoken_text, auto_open=auto_open_audio)
                result["audio_file"] = audio_file
            except Exception as audio_error:
                print("TTS ERROR:", str(audio_error))
                result["audio_file"] = None

        latency = round(time.time() - start_time, 3)

        log_event({
            "component": "pipeline",
            "route": result.get("route"),
            "fallback": result.get("fallback"),
            "question": question,
            "status": "success",
            "latency_seconds": latency,
            "slo_latency_status": "pass" if latency <= 5 else "fail",
        })

        return result

    except Exception as e:
        print("ERROR:", str(e))

        return {
            "question": question,
            "route": "error",
            "fallback": True,
            "sources": "system_error",
            "answer": "The system encountered an issue but handled it safely. Please try again.",
            "audio_file": None,
        }


if __name__ == "__main__":
    example_selection = {
        "mode": "weeks",
        "block": "B",
        "weeks": [5, 6],
        "number_of_weeks": 2,
    }

    question = input("Enter the user's final question: ").strip()
    result = run_finops_cfo_pipeline(
        example_selection,
        question,
        enable_audio=True,
        auto_open_audio=True,
    )


def run_multi_query_rag(
    question: str,
    general_subquestion: str,
    embedding_model,
    reranker,
):
    q_lower = question.lower()

    if "future" in q_lower or "next year" in q_lower or "competitor" in q_lower:
        return {
            "question": question,
            "answer": "I don't have enough information in the retrieved documents.",
            "top_results": [],
            "confidence": {
                "label": "Low",
                "reason": "Question asks for unsupported future or external competitor information.",
            },
            "sources": "No sufficiently reliable retrieved sources",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fallback": True,
        }

    queries = [question]

    if general_subquestion != question:
        queries.append(general_subquestion)

    rewritten = llm_rewrite_question_for_rag(question)
    queries.append(rewritten)

    result = None

    for q in queries:
        result = handle_general_question(
            q,
            embedding_model,
            reranker,
        )

        if not result.get("fallback", True):
            return result

    return result


def build_eval_selection(period_type: str, period_value: str) -> dict:
    day_map = {
        "week_2_monday": {"week": 2, "day_labels": ["Monday"]},
        "week_2_monday_to_wednesday": {"week": 2, "day_labels": ["Monday", "Tuesday", "Wednesday"]},
        "week_3_monday_to_saturday": {"week": 3, "day_labels": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]},
        "week_4_tuesday_to_thursday": {"week": 4, "day_labels": ["Tuesday", "Wednesday", "Thursday"]},
        "week_5_monday": {"week": 5, "day_labels": ["Monday"]},
    }

    week_map = {
        "week_1": [1],
        "weeks_2_to_3": [2, 3],
        "weeks_5_to_7": [5, 6, 7],
        "week_8": [8],
        "weeks_9_to_10": [9, 10],
        "weeks_10_to_12": [10, 11, 12],
    }

    if period_type == "days":
        if period_value not in day_map:
            raise ValueError(f"Unsupported eval day period: {period_value}")

        config = day_map[period_value]
        return {
            "mode": "days",
            "week": config["week"],
            "day_labels": config["day_labels"],
        }

    if period_type == "weeks":
        if period_value not in week_map:
            raise ValueError(f"Unsupported eval week period: {period_value}")

        weeks = week_map[period_value]
        return {
            "mode": "weeks",
            "weeks": weeks,
            "number_of_weeks": len(weeks),
        }

    if period_type == "months":
        return {
            "mode": "months",
            "monthly_block": period_value,
        }

    raise ValueError(f"Unsupported period_type: {period_type}")

def run_cfo_evaluation_question(question: str, period_type: str, period_value: str) -> str:
    """
    Entry point pour les tests offline.
    Exclut Power BI, Amazon Polly et l'interface complète.
    """

    try:
        selection = build_eval_selection(period_type, period_value)

        division_filtered, product_filtered = load_selected_period_data(selection)

        try:
            division_daily, product_daily = load_selected_period_daily_spend(selection)
        except Exception:
            product_daily, division_daily = None, None

        route = classify_question(question)

        if route == "analytical":
            return answer_analytical_question(
                question,
                product_filtered,
                division_filtered,
                product_daily,
                division_daily,
            )

        elif route == "general":
            q_lower = question.lower()

            if "risk" in q_lower or "recommendation" in q_lower:
                analytical_answer = answer_analytical_question(
                    question,
                    product_filtered,
                    division_filtered,
                    product_daily,
                    division_daily,
                )

                general_subquestion = extract_general_subquestion(question)

                rag_result = run_multi_query_rag(
                    question=question,
                    general_subquestion=general_subquestion,
                    embedding_model=embedding_model_global,
                    reranker=reranker_global,
                )

                rag_answer = rag_result["answer"]

                return (
                    "Hybrid CFO Answer:\n\n"
                    "1) Financial Risk:\n"
                    f"{analytical_answer}\n\n"
                    "2) FinOps Interpretation:\n"
                    f"{rag_answer}\n\n"
                    "3) Operational Recommendation:\n"
                    "The CFO should immediately strengthen governance, enforce accountability, "
                    "and prioritize optimization of the highest cost and variance drivers to reduce financial risk."
                )

            general_subquestion = extract_general_subquestion(question)

            result = run_multi_query_rag(
                question=question,
                general_subquestion=general_subquestion,
                embedding_model=embedding_model_global,
                reranker=reranker_global,
            )

            return result["answer"]

        elif route == "hybrid":
            analytical_answer = answer_analytical_question(
                question,
                product_filtered,
                division_filtered,
                product_daily,
                division_daily,
            )

            general_subquestion = extract_general_subquestion(question)

            rag_result = run_multi_query_rag(
                question=question,
                general_subquestion=general_subquestion,
                embedding_model=embedding_model_global,
                reranker=reranker_global,
            )

            rag_answer = rag_result["answer"]

            return (
                "Hybrid CFO Answer:\n\n"
                "1) Financial Risk:\n"
                f"{analytical_answer}\n\n"
                "2) FinOps Interpretation:\n"
                f"{rag_answer}\n\n"
                "3) Operational Recommendation:\n"
                "The CFO should immediately strengthen governance, enforce accountability, "
                "and prioritize optimization of the highest cost and variance drivers to reduce financial risk."
            )

        else:
            return f"UNKNOWN_ROUTE: {route}"

    except Exception as e:
        return f"ERROR_EVAL_PIPELINE: {str(e)}"