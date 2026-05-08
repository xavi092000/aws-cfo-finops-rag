from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import os
import re
import time
import warnings
import webbrowser
import  wave
from typing import Any

import boto3
import pandas as pd
import requests
from dotenv import load_dotenv

from build_athena_daily_marts import build_and_publish_daily_marts
from resolve_finops_analytics import load_selected_period_data, force_aws_profile
from cfo_finops_athena_rag_final import run_finops_cfo_pipeline
from load_athena_views import load_current_powerbi_views
from graph_narration import generate_graph_narration

load_dotenv()

TENANT_ID = os.getenv("TENANT_ID", "")
CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
WORKSPACE_ID = os.getenv("WORKSPACE_ID", "")
DATASET_ID = os.getenv("DATASET_ID", "")
POWERBI_REPORT_URL = os.getenv("POWERBI_REPORT_URL", "")

POWERBI_SCOPE = "https://analysis.windows.net/powerbi/api/.default"

warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable.*",
    category=UserWarning,
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

AWS_PROFILE = "terraform-runner"
AWS_REGION = "us-east-1"
ATHENA_SCHEMA = "cfo_finops_db"
ATHENA_DATABASE = ATHENA_SCHEMA
ATHENA_WORKGROUP = "primary"
ATHENA_S3_STAGING_DIR = "s3://kestra-zoomcamp-ny-taxi-mtl9999999999988888/athena-results/"

VOICE_ID = "Joanna"
ENGINE = "neural"
AUDIO_END_PAUSE_SECONDS = 2.5

REFRESH_STATE_FILE = OUTPUT_DIR / "powerbi_refresh_state.json"
AUTO_REFRESH_COOLDOWN_MINUTES = 20
RECENT_429_BLOCK_MINUTES = 30
RECENT_REFRESH_WINDOW_MINUTES = 60
RECENT_REFRESH_THRESHOLD = 3

BLOCKS = {
    "A": [1, 2, 3, 4],
    "B": [5, 6, 7, 8],
    "C": [9, 10, 11, 12]
}

VALID_MONTH_SELECTIONS = ["A", "B", "C", "AB", "BC", "ABC"]

STANDARD_END_DAY_MAP = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday"
}

WEEK_1_END_DAY_MAP = {
    1: "Wednesday",
    2: "Thursday",
    3: "Friday",
    4: "Saturday"
}

WEEK_1_AVAILABLE_DAYS = ["Wednesday", "Thursday", "Friday", "Saturday"]


# =========================
# AWS / ATHENA HELPERS
# =========================
def get_boto3_session() -> boto3.Session:
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


def execute_athena_query(sql: str) -> str:
    session = get_boto3_session()
    athena = session.client("athena", region_name=AWS_REGION)

    response = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DATABASE},
        ResultConfiguration={"OutputLocation": ATHENA_S3_STAGING_DIR},
        WorkGroup=ATHENA_WORKGROUP,
    )

    query_execution_id = response["QueryExecutionId"]

    while True:
        status_response = athena.get_query_execution(QueryExecutionId=query_execution_id)
        state = status_response["QueryExecution"]["Status"]["State"]

        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            break

        time.sleep(1)

    if state != "SUCCEEDED":
        reason = status_response["QueryExecution"]["Status"].get(
            "StateChangeReason",
            "Unknown Athena error"
        )
        raise RuntimeError(f"Athena query failed with state {state}: {reason}")

    return query_execution_id


def build_week_in_clause(weeks: list[int]) -> str:
    if not weeks:
        raise ValueError("Weeks selection is empty.")
    return ", ".join([f"'Week {week}'" for week in weeks])


def build_day_in_clause(day_names: list[str]) -> str:
    if not day_names:
        raise ValueError("Day selection is empty.")
    return ", ".join([f"'{day}'" for day in day_names])


def build_month_block_week_list(monthly_block: str) -> list[int]:
    block_map = {
        "A": [1, 2, 3, 4],
        "B": [5, 6, 7, 8],
        "C": [9, 10, 11, 12],
        "AB": [1, 2, 3, 4, 5, 6, 7, 8],
        "BC": [5, 6, 7, 8, 9, 10, 11, 12],
        "ABC": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    }

    if monthly_block not in block_map:
        raise ValueError(f"Unsupported monthly block: {monthly_block}")

    return block_map[monthly_block]


def build_month_case_sql(week_column: str = "week") -> str:
    return f"""
        CASE
            WHEN {week_column} IN ('Week 1', 'Week 2', 'Week 3', 'Week 4') THEN 'Month 1'
            WHEN {week_column} IN ('Week 5', 'Week 6', 'Week 7', 'Week 8') THEN 'Month 2'
            WHEN {week_column} IN ('Week 9', 'Week 10', 'Week 11', 'Week 12') THEN 'Month 3'
            ELSE 'DATA_NOT_AVAILABLE'
        END
    """.strip()


def update_powerbi_views_for_weeks(selection: dict) -> bool:
    weeks = selection.get("weeks", [])
    if not weeks:
        raise ValueError("Weeks mode selected, but no weeks were provided in selection.")

    week_sql = build_week_in_clause(weeks)

    product_view_sql = f"""
    CREATE OR REPLACE VIEW current_product_view AS
    SELECT
        week AS period_value,
        CAST(NULL AS varchar) AS date,
        CAST(NULL AS varchar) AS day,
        week,
        month,
        product,
        division,
        service,
        SUM(actual_cost_usd) AS actual_cost_usd,
        SUM(allocated_budget_usd) AS allocated_budget_usd,
        SUM(variance_usd) AS variance_usd,
        CASE
            WHEN SUM(allocated_budget_usd) > 0
            THEN (SUM(variance_usd) / SUM(allocated_budget_usd)) * 100
            ELSE 0
        END AS variance_percent
    FROM {ATHENA_SCHEMA}.mart_product_cfo_daily_full
    WHERE week IN ({week_sql})
    GROUP BY week, month, product, division, service
    """

    division_view_sql = f"""
    CREATE OR REPLACE VIEW current_division_view AS
    SELECT
        week AS period_value,
        CAST(NULL AS varchar) AS date,
        CAST(NULL AS varchar) AS day,
        week,
        month,
        division,
        service,
        SUM(actual_cost_usd) AS actual_cost_usd,
        SUM(allocated_budget_usd) AS allocated_budget_usd,
        SUM(variance_usd) AS variance_usd,
        CASE
            WHEN SUM(allocated_budget_usd) > 0
            THEN (SUM(variance_usd) / SUM(allocated_budget_usd)) * 100
            ELSE 0
        END AS variance_percent
    FROM {ATHENA_SCHEMA}.mart_division_cfo_daily_full
    WHERE week IN ({week_sql})
    GROUP BY week, month, division, service
    """

    service_view_sql = f"""
    CREATE OR REPLACE VIEW current_service_view AS
    SELECT
        week AS period_value,
        CAST(NULL AS varchar) AS date,
        CAST(NULL AS varchar) AS day,
        week,
        month,
        service,
        SUM(actual_cost_usd) AS actual_cost_usd,
        SUM(allocated_budget_usd) AS allocated_budget_usd,
        SUM(variance_usd) AS variance_usd,
        CASE
            WHEN SUM(allocated_budget_usd) > 0
            THEN (SUM(variance_usd) / SUM(allocated_budget_usd)) * 100
            ELSE 0
        END AS variance_percent
    FROM {ATHENA_SCHEMA}.mart_division_cfo_daily_full
    WHERE week IN ({week_sql})
    GROUP BY week, month, service
    """

    print("\nUpdating Power BI views in Athena for weeks mode...")
    execute_athena_query(product_view_sql)
    execute_athena_query(division_view_sql)
    execute_athena_query(service_view_sql)
    print("Power BI views updated successfully for weeks mode.")
    return True


def update_powerbi_views_for_months(selection: dict) -> bool:
    monthly_block = selection.get("monthly_block")
    if not monthly_block:
        raise ValueError("Months mode selected, but no monthly_block was provided.")

    selected_weeks = build_month_block_week_list(monthly_block)
    week_sql = build_week_in_clause(selected_weeks)
    month_case_sql = build_month_case_sql("week")

    product_view_sql = f"""
    CREATE OR REPLACE VIEW current_product_view AS
    SELECT
        {month_case_sql} AS period_value,
        CAST(NULL AS varchar) AS date,
        CAST(NULL AS varchar) AS day,
        week,
        {month_case_sql} AS month,
        product,
        division,
        service,
        SUM(actual_cost_usd) AS actual_cost_usd,
        SUM(allocated_budget_usd) AS allocated_budget_usd,
        SUM(variance_usd) AS variance_usd,
        CASE
            WHEN SUM(allocated_budget_usd) > 0
            THEN (SUM(variance_usd) / SUM(allocated_budget_usd)) * 100
            ELSE 0
        END AS variance_percent
    FROM {ATHENA_SCHEMA}.mart_product_cfo_daily_full
    WHERE week IN ({week_sql})
    GROUP BY week, {month_case_sql}, product, division, service
    """

    division_view_sql = f"""
    CREATE OR REPLACE VIEW current_division_view AS
    SELECT
        {month_case_sql} AS period_value,
        CAST(NULL AS varchar) AS date,
        CAST(NULL AS varchar) AS day,
        week,
        {month_case_sql} AS month,
        division,
        service,
        SUM(actual_cost_usd) AS actual_cost_usd,
        SUM(allocated_budget_usd) AS allocated_budget_usd,
        SUM(variance_usd) AS variance_usd,
        CASE
            WHEN SUM(allocated_budget_usd) > 0
            THEN (SUM(variance_usd) / SUM(allocated_budget_usd)) * 100
            ELSE 0
        END AS variance_percent
    FROM {ATHENA_SCHEMA}.mart_division_cfo_daily_full
    WHERE week IN ({week_sql})
    GROUP BY week, {month_case_sql}, division, service
    """

    service_view_sql = f"""
    CREATE OR REPLACE VIEW current_service_view AS
    SELECT
        {month_case_sql} AS period_value,
        CAST(NULL AS varchar) AS date,
        CAST(NULL AS varchar) AS day,
        week,
        {month_case_sql} AS month,
        service,
        SUM(actual_cost_usd) AS actual_cost_usd,
        SUM(allocated_budget_usd) AS allocated_budget_usd,
        SUM(variance_usd) AS variance_usd,
        CASE
            WHEN SUM(allocated_budget_usd) > 0
            THEN (SUM(variance_usd) / SUM(allocated_budget_usd)) * 100
            ELSE 0
        END AS variance_percent
    FROM {ATHENA_SCHEMA}.mart_product_cfo_daily_full
    WHERE week IN ({week_sql})
    GROUP BY week, {month_case_sql}, service
    """

    print("\nUpdating Power BI views in Athena for months mode...")
    execute_athena_query(product_view_sql)
    execute_athena_query(division_view_sql)
    execute_athena_query(service_view_sql)
    print("Power BI views updated successfully for months mode.")
    return True


def update_powerbi_views_from_selection(selection: dict) -> bool:
    mode = selection.get("mode")

    if mode == "days":
        week = selection.get("week")
        day_count = selection.get("days")

        if not week or not day_count:
            raise ValueError("Days mode selected, but week or days are missing.")

        if week == 1:
            day_map = {
                1: ["Wednesday"],
                2: ["Wednesday", "Thursday"],
                3: ["Wednesday", "Thursday", "Friday"],
                4: ["Wednesday", "Thursday", "Friday", "Saturday"],
            }
        else:
            day_map = {
                1: ["Monday"],
                2: ["Monday", "Tuesday"],
                3: ["Monday", "Tuesday", "Wednesday"],
                4: ["Monday", "Tuesday", "Wednesday", "Thursday"],
                5: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                6: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
            }

        selected_days = day_map.get(day_count)
        if not selected_days:
            raise ValueError(f"Unsupported day_count for days mode: {day_count}")

        print("\nBuilding and publishing daily marts to Athena source...")
        build_and_publish_daily_marts(selection)

        day_sql = build_day_in_clause(selected_days)
        week_sql = f"'Week {week}'"

        product_view_sql = f"""
        CREATE OR REPLACE VIEW current_product_view AS
        SELECT
            date AS period_value,
            date,
            day,
            week,
            month,
            product,
            division,
            service,
            actual_cost_usd,
            allocated_budget_usd,
            variance_usd,
            variance_pct AS variance_percent
        FROM {ATHENA_SCHEMA}.mart_product_cfo_daily
        WHERE week = {week_sql}
          AND day IN ({day_sql})
        """

        division_view_sql = f"""
        CREATE OR REPLACE VIEW current_division_view AS
        SELECT
            date AS period_value,
            date,
            day,
            week,
            month,
            division,
            service,
            actual_cost_usd,
            allocated_budget_usd,
            variance_usd,
            variance_pct AS variance_percent
        FROM {ATHENA_SCHEMA}.mart_division_cfo_daily
        WHERE week = {week_sql}
          AND day IN ({day_sql})
        """

        service_view_sql = f"""
        CREATE OR REPLACE VIEW current_service_view AS
        SELECT
            date AS period_value,
            date,
            day,
            week,
            month,
            service,
            actual_cost_usd,
            allocated_budget_usd,
            variance_usd,
            variance_pct AS variance_percent
        FROM {ATHENA_SCHEMA}.mart_division_cfo_daily
        WHERE week = {week_sql}
          AND day IN ({day_sql})
        """

        print("\nUpdating Power BI views in Athena for days mode...")
        execute_athena_query(product_view_sql)
        execute_athena_query(division_view_sql)
        execute_athena_query(service_view_sql)
        print("Power BI views updated successfully for days mode.")
        return True

    if mode == "weeks":
        return update_powerbi_views_for_weeks(selection)

    if mode == "months":
        return update_powerbi_views_for_months(selection)

    print(f"\nPower BI views update skipped for unsupported mode: {mode}")
    return False


# =========================
# VOICE HELPERS
# =========================
def build_spoken_text(text: str) -> str:
    spoken = str(text).strip()

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
        r"\bPower BI\b": "Power B I",
    }

    for pattern, replacement in acronym_map.items():
        spoken = re.sub(pattern, replacement, spoken)

    spoken = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", spoken)
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


def text_to_speech(text: str, output_file=None, auto_open: bool = False) -> str:
    if output_file is None:
        timestamp_for_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_file = OUTPUT_DIR / f"agent_prompt_{timestamp_for_file}.wav"
    else:
        output_file = Path(output_file)

    session = boto3.Session(profile_name=AWS_PROFILE)
    polly = session.client("polly", region_name=AWS_REGION)

    spoken_text = build_spoken_text(text)
    formatted_text = apply_ssml_formatting(spoken_text)

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
        OutputFormat="pcm",
        VoiceId=VOICE_ID,
        Engine=ENGINE
    )

    import wave

    audio_bytes = response["AudioStream"].read()

    with wave.open(str(output_file), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_bytes)

    if auto_open:
        try:
            os.startfile(str(output_file))
        except Exception:
            pass

    return str(output_file)


def split_for_polly(text: str, max_len: int = 1200) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current) + len(sentence) + 1 <= max_len:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def _estimate_playback_wait_seconds(text: str) -> float:
    word_count = max(1, len(str(text).split()))
    estimated = word_count / 2.3
    return max(2.5, min(12.0, estimated + AUDIO_END_PAUSE_SECONDS))


def speak_long_message(message: str) -> None:
    chunks = split_for_polly(message, max_len=1200)
  
    audio_files = []


    for chunk in chunks:
        print(f"\nFinOps Agent: {chunk}")

        output_file = text_to_speech(chunk, auto_open=False)
        audio_files.append(output_file)


    final_output = OUTPUT_DIR / f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    with wave.open(str(final_output), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)

        for file in audio_files:
            with wave.open(file, "rb") as rf:
                wf.writeframes(rf.readframes(rf.getnframes()))

    # ▶️ ouvrir UNE seule fois
    try:
        os.startfile(str(final_output))
    except Exception:
        pass

def speak(message: str) -> None:
    speak_long_message(message)


# =========================
# INPUT HELPERS
# =========================
def ask_input(prompt: str) -> str:
    return input(prompt).strip()


def get_non_empty_input(prompt: str, retry_message: str) -> str:
    while True:
        value = ask_input(prompt)
        if value:
            return value
        speak(retry_message)


def normalize_mode(mode: str) -> str:
    return mode.strip().lower()


def normalize_block(block: str) -> str:
    return block.strip().upper().replace(" ", "")


def parse_weeks_input(text: str) -> list[int]:
    cleaned = text.replace(",", " ")
    parts = [p.strip() for p in cleaned.split() if p.strip()]
    return [int(p) for p in parts]


# =========================
# INTRO
# =========================
def intro_message() -> None:
    message = """
Welcome to the FinOps AI assistant.

This pipeline will help you configure your financial analysis within the period of your choosing.

First, you will provide your name.

Then, you will select your analysis mode.

You can choose between days, weeks, or months.

Important rule.
Week 1 includes a holiday shutdown period.
Monday and Tuesday of Week 1 were days when the company operations were shut down, and this is why these 2 days are not available for analysis.
Operational selection for Week 1 starts on Wednesday.

I will, as your assistant, guide you step by step in this process.

Let's begin, so please fill out your name and thank you for your patience.
"""
    speak(message)


# =========================
# STEP 1 - USER NAME
# =========================
def ask_user_name() -> str:
    name = get_non_empty_input(
        "Enter your name: ",
        "Your name cannot be empty. Please enter your name."
    )
    speak(f"Hello {name}. I will now guide you through the steps to implement your financial analysis.")
    return name



# =========================
# STEP 2 - MODE
# =========================
def ask_analysis_mode() -> str:
    while True:
        speak("Please choose your analysis period type. Option 1 is days. Option 2 is weeks. Option 3 is months.")

        choice = get_non_empty_input(
            "\nSelect analysis period type:\n"
            "1. Days\n"
            "2. Weeks\n"
            "3. Months\n"
            "> ",
            "The selection cannot be empty. Please enter 1, 2, or 3."
        ).strip()

        if choice == "1":
            return "days"
        if choice == "2":
            return "weeks"
        if choice == "3":
            return "months"

        speak("Invalid selection. Please enter 1, 2, or 3.")
        print("Invalid selection. Please enter 1, 2, or 3.")


# =========================
# DAYS MODE
# =========================
def validate_days(block: str, week: int, day_count: int) -> tuple[bool, str, str]:
    block = normalize_block(block)

    if block not in BLOCKS:
        return False, "Invalid block. Please choose A, B, or C.", ""

    if week not in BLOCKS[block]:
        return False, f"Week {week} does not belong to block {block}.", ""

    if week == 1:
        if day_count < 1 or day_count > 4:
            return (
                False,
                "For Week 1, the number of days must be between 1 and 4 because Monday and Tuesday are unavailable.",
                ""
            )

        end_day = WEEK_1_END_DAY_MAP[day_count]
        period = f"Wednesday to {end_day}"

        return True, (
            f"Valid selection. Block {block}, week {week}, period from Wednesday to {end_day}. "
            f"Monday and Tuesday of Week 1 are excluded due to the holiday shutdown."
        ), period

    if day_count < 1 or day_count > 6:
        return False, "The number of days must be between 1 and 6.", ""

    end_day = STANDARD_END_DAY_MAP[day_count]
    period = f"Monday to {end_day}"

    return True, f"Valid selection. Block {block}, week {week}, period from Monday to {end_day}.", period


def ask_days_period() -> dict:
    while True:
        speak(
            "You selected days mode. "
            "Please choose your analysis block. "
            "Block A corresponds to weeks 1 to 4. "
            "Block B corresponds to weeks 5 to 8. "
            "Block C corresponds to weeks 9 to 12. "
            "This grouping helps structure the financial analysis period."
        )

        block = normalize_block(
            get_non_empty_input(
                "Choose a block (A / B / C): ",
                "The block cannot be empty. Please choose A, B, or C."
            )
        )

        if block not in BLOCKS:
            speak("Invalid block. Please choose A, B, or C.")
            continue

        speak(
            f"You selected block {block}. "
            f"This block includes the following weeks: {BLOCKS[block]}."
        )

        try:
            week = int(
                get_non_empty_input(
                    "Choose one week within this block: ",
                    "The week cannot be empty. Please enter one week number."
                )
            )
        except ValueError:
            speak("Invalid input. In days mode, you must choose one single week number only.")
            continue

        if week == 1:
            speak(
                "Important reminder. For Week 1, Monday and Tuesday are not available because of the holiday shutdown. "
                "You may only select consecutive operational days from Wednesday to Saturday."
            )
            day_prompt = "Choose the number of consecutive operational days for Week 1 (1 to 4): "
            day_retry = "The number of days cannot be empty. Please enter a value from 1 to 4."
        else:
            day_prompt = "Choose the number of consecutive days, from 1 to 6: "
            day_retry = "The number of days cannot be empty. Please enter a value from 1 to 6."

        try:
            day_count = int(get_non_empty_input(day_prompt, day_retry))
        except ValueError:
            speak("Invalid input. The number of days must be numeric.")
            continue

        is_valid, message, period = validate_days(block, week, day_count)
        speak(message)

        if is_valid:
            return {
                "mode": "days",
                "block": block,
                "week": week,
                "days": day_count,
                "period": period
            }


# =========================
# WEEKS MODE
# =========================
def validate_weeks(block: str, weeks: list[int]) -> tuple[bool, str]:
    block = normalize_block(block)

    if block not in BLOCKS:
        return False, "Invalid block. Please choose A, B, or C."

    if not weeks:
        return False, "No weeks were selected."

    if len(weeks) > 3:
        return False, "You cannot select more than three weeks."

    if len(weeks) != len(set(weeks)):
        return False, "Duplicate weeks are not allowed."

    for week in weeks:
        if week not in BLOCKS[block]:
            return False, f"Week {week} does not belong to block {block}."

    weeks_sorted = sorted(weeks)

    for i in range(1, len(weeks_sorted)):
        if weeks_sorted[i] != weeks_sorted[i - 1] + 1:
            return False, "Weeks must be consecutive."

    if 1 in weeks_sorted:
        return True, (
            f"Valid selection. Block {block}, weeks {weeks_sorted}. "
            "Reminder. Week 1 includes a holiday shutdown, so Monday and Tuesday are non-operational days."
        )

    return True, f"Valid selection. Block {block}, weeks {weeks_sorted}."

def ask_weeks_period() -> dict:
    speak(
        "You selected weeks mode. "
        "Please choose your analysis block. "
        "Block A corresponds to weeks 1 to 4. "
        "Block B corresponds to weeks 5 to 8. "
        "Block C corresponds to weeks 9 to 12. "
        "Weeks must be consecutive and cannot exceed three weeks."
    )

    while True:
        block = normalize_block(
            get_non_empty_input(
                "Choose a block:\n"
                "A = Weeks 1-4\n"
                "B = Weeks 5-8\n"
                "C = Weeks 9-12\n"
                "> ",
                "The block cannot be empty. Please choose A, B, or C."
            )
        )

        if block not in BLOCKS:
            speak("Invalid block. Please choose A, B, or C.")
            print("Invalid block. Please choose A, B, or C.")
            continue

        speak(
            f"You selected block {block}. "
            f"This block includes the following weeks: {BLOCKS[block]}. "
            "You can now choose one to three consecutive weeks within this range."
        )

        while True:
            weeks_text = get_non_empty_input(
                "Choose 1 to 3 consecutive weeks for this block "
                "(example: 5 or 5,6 or 6,7,8): ",
                "The weeks field cannot be empty. Please enter one, two, or three consecutive weeks."
            )

            try:
                weeks = parse_weeks_input(weeks_text)
            except ValueError:
                speak("Invalid format. Please enter only week numbers.")
                print("Invalid format. Please enter only week numbers.")
                continue

            is_valid, message = validate_weeks(block, weeks)
            speak(message)
            print(message)

            if is_valid:
                return {
                    "mode": "weeks",
                    "block": block,
                    "weeks": sorted(weeks),
                    "number_of_weeks": len(weeks)
                }


# =========================
# MONTHS MODE
# =========================
def validate_months(selection: str) -> tuple[bool, str]:
    selection = normalize_block(selection)

    if selection not in VALID_MONTH_SELECTIONS:
        return False, "Invalid selection. Valid options are A, B, C, AB, BC, or ABC."

    if selection in ["A", "AB", "ABC"]:
        return True, (
            f"Valid selection. Monthly block {selection}. "
            f"Reminder. Week 1 within this monthly scope includes a holiday shutdown, so Monday and Tuesday are non-operational days."
        )

    return True, f"Valid selection. Monthly block {selection}."


def ask_months_period() -> dict:
    while True:
        speak(
            "You selected months mode. "
            "Available options are A for month 1, B for month 2, and C for month 3. "
            "Your available combined choices are AB, BC, or ABC. "
            "If your monthly scope includes block A, please note that Week 1 Monday and Tuesday are treated as non-operational holiday shutdown days."
        )

        selection = normalize_block(
            get_non_empty_input(
                "Choose your monthly block: ",
                "The monthly block cannot be empty. Please enter A, B, C, AB, BC, or ABC."
            )
        )

        is_valid, message = validate_months(selection)
        speak(message)

        if is_valid:
            return {
                "mode": "months",
                "monthly_block": selection
            }


# =========================
# PREVIEW HELPERS
# =========================
def clean_numeric_preview(series: pd.Series) -> pd.Series:
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

    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def format_money_for_preview(value: float, force_sign: bool = False) -> str:
    if value < 0:
        return f"-{abs(value):,.2f}"
    if force_sign and value > 0:
        return f"+{value:,.2f}"
    return f"{value:,.2f}"


def finalize_preview_table(
    df: pd.DataFrame,
    name_col: str,
    total_label: str,
    sort_by: str = "Actual"
) -> pd.DataFrame:
    working = df.copy()

    data_rows = working[working[name_col] != total_label].copy()
    total_rows = working[working[name_col] == total_label].copy()

    data_rows = data_rows.sort_values(sort_by, ascending=False).reset_index(drop=True)
    final_df = pd.concat([data_rows, total_rows], ignore_index=True)

    final_df["Budget"] = final_df["Budget"].apply(lambda x: format_money_for_preview(x, force_sign=False))
    final_df["Actual"] = final_df["Actual"].apply(lambda x: format_money_for_preview(x, force_sign=False))
    final_df["Variance"] = final_df["Variance"].apply(lambda x: format_money_for_preview(x, force_sign=True))

    return final_df


def build_products_quick_summary(product_df: pd.DataFrame) -> pd.DataFrame:
    working = product_df.copy()

    working["actual_cost_num"] = clean_numeric_preview(working["actual_cost_usd"])
    working["allocated_budget_num"] = clean_numeric_preview(working["allocated_budget_usd"])

    grouped = (
        working.groupby("product", dropna=False)[["allocated_budget_num", "actual_cost_num"]]
        .sum()
        .reset_index()
        .rename(
            columns={
                "product": "Product",
                "allocated_budget_num": "Budget",
                "actual_cost_num": "Actual",
            }
        )
    )

    grouped["Variance"] = grouped["Actual"] - grouped["Budget"]

    total_row = pd.DataFrame(
        [
            {
                "Product": "TOTAL",
                "Budget": grouped["Budget"].sum(),
                "Actual": grouped["Actual"].sum(),
                "Variance": grouped["Variance"].sum(),
            }
        ]
    )

    grouped = pd.concat([grouped, total_row], ignore_index=True)
    return finalize_preview_table(grouped, name_col="Product", total_label="TOTAL", sort_by="Actual")


def build_internal_quick_summary(division_df: pd.DataFrame) -> pd.DataFrame:
    working = division_df.copy()

    working["actual_cost_num"] = clean_numeric_preview(working["actual_cost_usd"])
    working["allocated_budget_num"] = clean_numeric_preview(working["allocated_budget_usd"])

    grouped = (
        working.groupby("division", dropna=False)[["allocated_budget_num", "actual_cost_num"]]
        .sum()
        .reset_index()
        .rename(
            columns={
                "division": "Division",
                "allocated_budget_num": "Budget",
                "actual_cost_num": "Actual",
            }
        )
    )

    grouped["Variance"] = grouped["Actual"] - grouped["Budget"]

    total_row = pd.DataFrame(
        [
            {
                "Division": "TOTAL",
                "Budget": grouped["Budget"].sum(),
                "Actual": grouped["Actual"].sum(),
                "Variance": grouped["Variance"].sum(),
            }
        ]
    )

    grouped = pd.concat([grouped, total_row], ignore_index=True)
    return finalize_preview_table(grouped, name_col="Division", total_label="TOTAL", sort_by="Actual")


def build_selected_period_label(selection: dict) -> str:
    mode = selection.get("mode")

    if mode == "days":
        return f"Selected Period: Week {selection['week']} — {selection['period']}"
    if mode == "weeks":
        weeks = ", ".join([f"Week {w}" for w in selection["weeks"]])
        return f"Selected Period: {weeks}"
    if mode == "months":
        return f"Selected Period: Monthly Block {selection['monthly_block']}"
    return "Selected Period: DATA_NOT_AVAILABLE"


def render_table_with_spacing(df: pd.DataFrame, name_col: str) -> str:
    name_width = max(len(name_col), df[name_col].astype(str).map(len).max()) + 4
    budget_width = max(len("Budget"), df["Budget"].astype(str).map(len).max()) + 4
    actual_width = max(len("Actual"), df["Actual"].astype(str).map(len).max()) + 4
    variance_width = max(len("Variance"), df["Variance"].astype(str).map(len).max()) + 4

    header = (
        f"{name_col:<{name_width}}"
        f"{'Budget':>{budget_width}}"
        f"{'Actual':>{actual_width}}"
        f"{'Variance':>{variance_width}}"
    )

    lines = [header]

    for _, row in df.iterrows():
        lines.append(
            f"{str(row[name_col]):<{name_width}}"
            f"{str(row['Budget']):>{budget_width}}"
            f"{str(row['Actual']):>{actual_width}}"
            f"{str(row['Variance']):>{variance_width}}"
        )

    return "\n".join(lines)


def preview_loaded_data(selection: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    division_df, product_df = load_selected_period_data(selection)

    print_quick_summary_tables(selection, division_df, product_df)

    return division_df, product_df

def print_quick_summary_tables(selection: dict, division_df: pd.DataFrame, product_df: pd.DataFrame) -> None:
    budget_col = "allocated_budget_usd"
    actual_col = "actual_cost_usd"

    if budget_col not in product_df.columns or actual_col not in product_df.columns:
        print("\n==============================")
        print("STEP 1 — DATA VALIDATION SUMMARY")
        print("==============================")
        print(build_selected_period_label(selection))
        print("\nData loaded successfully.")
        print(f"Product rows loaded: {len(product_df)}")
        print(f"Internal division rows loaded: {len(division_df)}")
        print("\nFinancial summary skipped because expected cost columns were not found.")
        return

    product_budget = product_df[budget_col].sum()
    product_actual = product_df[actual_col].sum()
    product_variance = product_actual - product_budget
    product_variance_pct = (product_variance / product_budget * 100) if product_budget else 0

    print("\n==============================")
    print("STEP 1 — DATA VALIDATION SUMMARY")
    print("==============================")
    print(build_selected_period_label(selection))

    print("\nData loaded successfully.")
    print(f"Product rows loaded: {len(product_df)}")
    print(f"Internal division rows loaded: {len(division_df)}")



# =========================
# POWER BI STATE HELPERS
# =========================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def datetime_to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    value = str(value).strip()
    if not value:
        return None

    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except Exception:
        return None


def read_refresh_state() -> dict[str, Any]:
    if not REFRESH_STATE_FILE.exists():
        return {}

    try:
        return json.loads(REFRESH_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_refresh_state(state: dict[str, Any]) -> None:
    REFRESH_STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def mark_recent_429() -> None:
    state = read_refresh_state()
    state["last_429_time"] = datetime_to_iso(utc_now())
    write_refresh_state(state)


def clear_recent_429() -> None:
    state = read_refresh_state()
    if "last_429_time" in state:
        del state["last_429_time"]
    write_refresh_state(state)


def mark_auto_refresh_attempt() -> None:
    state = read_refresh_state()
    state["last_auto_refresh_attempt_time"] = datetime_to_iso(utc_now())
    write_refresh_state(state)


def mark_auto_refresh_success() -> None:
    state = read_refresh_state()
    state["last_auto_refresh_success_time"] = datetime_to_iso(utc_now())
    write_refresh_state(state)


# =========================
# POWER BI REFRESH HELPERS
# =========================
def get_powerbi_access_token() -> str:
    token_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"

    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": POWERBI_SCOPE,
    }

    response = requests.post(token_url, data=data, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to get Power BI token. "
            f"Status={response.status_code}. Response={response.text}"
        )

    payload: dict[str, Any] = response.json()
    access_token = payload.get("access_token")

    if not access_token:
        raise RuntimeError("Power BI token response did not contain access_token.")

    return access_token


def get_powerbi_refresh_history(top: int = 10) -> list[dict[str, Any]]:
    access_token = get_powerbi_access_token()

    refresh_history_url = (
        f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}"
        f"/datasets/{DATASET_ID}/refreshes?$top={top}"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
    }

    response = requests.get(refresh_history_url, headers=headers, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to get Power BI refresh history. "
            f"Status={response.status_code}. Response={response.text}"
        )

    payload: dict[str, Any] = response.json()
    return payload.get("value", [])


def get_latest_refresh_entry() -> dict[str, Any] | None:
    history = get_powerbi_refresh_history(top=1)
    if not history:
        return None
    return history[0]


def extract_refresh_timestamp(refresh_entry: dict[str, Any] | None) -> datetime | None:
    if not refresh_entry:
        return None

    candidate_keys = [
        "endTime",
        "startTime",
        "requestTime",
        "createdTime",
    ]

    for key in candidate_keys:
        parsed = parse_iso_datetime(refresh_entry.get(key))
        if parsed:
            return parsed

    return None


def is_powerbi_refresh_running() -> bool:
    latest = get_latest_refresh_entry()

    if not latest:
        return False

    status = str(latest.get("status", "")).strip().lower()
    print(f"Latest Power BI refresh status before trigger: {status}")

    return status in {"unknown", "queued", "inprogress", "in_progress"}


def count_recent_refreshes(window_minutes: int = RECENT_REFRESH_WINDOW_MINUTES) -> int:
    history = get_powerbi_refresh_history(top=20)
    now = utc_now()
    count = 0

    for entry in history:
        refresh_time = extract_refresh_timestamp(entry)
        if not refresh_time:
            continue

        elapsed_seconds = (now - refresh_time.astimezone(timezone.utc)).total_seconds()
        if elapsed_seconds <= window_minutes * 60:
            count += 1

    return count


def get_recent_429_age_minutes() -> float | None:
    state = read_refresh_state()
    last_429_dt = parse_iso_datetime(state.get("last_429_time"))
    if not last_429_dt:
        return None

    age_minutes = (utc_now() - last_429_dt.astimezone(timezone.utc)).total_seconds() / 60
    return age_minutes


def can_use_automatic_refresh() -> tuple[bool, str]:
    if is_powerbi_refresh_running():
        return False, "refresh_running"

    recent_429_age = get_recent_429_age_minutes()
    if recent_429_age is not None and recent_429_age < RECENT_429_BLOCK_MINUTES:
        return False, "recent_429"

    latest = get_latest_refresh_entry()
    latest_refresh_time = extract_refresh_timestamp(latest)

    if latest_refresh_time is not None:
        elapsed_minutes = (
            utc_now() - latest_refresh_time.astimezone(timezone.utc)
        ).total_seconds() / 60

        if elapsed_minutes < AUTO_REFRESH_COOLDOWN_MINUTES:
            return False, "refresh_too_recent"

    recent_refresh_count = count_recent_refreshes(window_minutes=RECENT_REFRESH_WINDOW_MINUTES)
    if recent_refresh_count >= RECENT_REFRESH_THRESHOLD:
        return False, "too_many_recent_refreshes"

    return True, "automatic_refresh_allowed"


def trigger_powerbi_refresh() -> bool:
    mark_auto_refresh_attempt()

    access_token = get_powerbi_access_token()

    refresh_url = (
        f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}"
        f"/datasets/{DATASET_ID}/refreshes"
    )

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(refresh_url, headers=headers, timeout=30)

    if response.status_code == 202:
        print("Power BI refresh triggered.")
        clear_recent_429()
        return True

    if response.status_code == 429:
        mark_recent_429()

    raise RuntimeError(
        f"Power BI refresh trigger failed. "
        f"Status={response.status_code}. Response={response.text}"
    )


def wait_for_powerbi_refresh_completion(
    timeout_seconds: int = 300,
    poll_interval_seconds: int = 10,
) -> None:
    start_time = time.time()

    while True:
        latest = get_latest_refresh_entry()

        if latest:
            status = str(latest.get("status", "")).strip().lower()
            print(f"Current Power BI refresh status: {status}")

            if status == "completed":
                print("Power BI refresh completed successfully.")
                mark_auto_refresh_success()
                return

            if status == "failed":
                raise RuntimeError("Power BI refresh failed.")

            if status in {"unknown", "queued", "inprogress", "in_progress"}:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    raise RuntimeError(
                        f"Timed out while waiting for Power BI refresh to complete. "
                        f"Last known status: {status}"
                    )
                time.sleep(poll_interval_seconds)
                continue

        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise RuntimeError("Timed out while waiting for Power BI refresh to complete.")

        time.sleep(poll_interval_seconds)


def open_powerbi_report() -> None:
    if not POWERBI_REPORT_URL.strip():
        print("POWERBI_REPORT_URL is empty. Skipping automatic browser open.")
        return

    print("\nOpening Power BI report...")
    webbrowser.open(POWERBI_REPORT_URL, new=2)
    time.sleep(5)


def ask_refresh_mode() -> str:
    while True:
        choice = input("\nRefresh mode (automatic / manual): ").strip().lower()
        if choice in {"automatic", "manual"}:
            return choice
        print("Invalid choice. Please enter automatic or manual.")


def explain_auto_refresh_block_reason(reason: str) -> None:
    reason_map = {
        "refresh_running": (
            "Automatic refresh is temporarily unavailable because a Power B I refresh is already running. "
            "We will continue safely with a guided manual refresh."
        ),
        "recent_429": (
            "Automatic refresh is temporarily unavailable because the Power B I API was rate limited recently. "
            "We will continue safely with a guided manual refresh."
        ),
        "refresh_too_recent": (
            "Automatic refresh is temporarily unavailable because the semantic model was refreshed recently. "
            "We will continue safely with a guided manual refresh."
        ),
        "too_many_recent_refreshes": (
            "Automatic refresh is temporarily unavailable because the semantic model has already been refreshed several times recently. "
            "We will continue safely with a guided manual refresh."
        ),
    }

    message = reason_map.get(
        reason,
        "Automatic refresh is temporarily unavailable. We will continue safely with a guided manual refresh."
    )

    print(f"\nAutomatic refresh blocked. Reason: {reason}")
    speak(message)


def guide_manual_refresh_conversation() -> None:
    message = """
The automatic refresh is temporarily unavailable right now.

That is not a problem.

To ensure that the dashboard graphs are fully updated before I begin the analysis, we will continue with a guided manual refresh.

I am opening Power BI for you now.

Please go to your workspace and locate the semantic model, also called the dataset, for this dashboard.

Please make sure that you refresh the semantic model, not only the report itself.

Open the semantic model and click Refresh now.

Then wait until the refresh status shows Completed.

Once the refresh is completed, return here and press Enter, and I will continue with the updated dashboard flow.
"""
    speak(message)
    open_powerbi_report()
    print("\nPower BI report opened. Please complete the manual semantic model refresh in Power BI.")
    print("After the refresh is completed, return here. The narration step will ask for final confirmation once the dashboard is fully visible.")
    


def guide_manual_refresh_conversation_direct() -> None:
    message = """
We will use a guided manual refresh.

I am opening Power BI for you now.

Please go to your workspace and locate the semantic model, also called the dataset, for this dashboard.

Please make sure that you refresh the semantic model, not only the report itself.

Open the semantic model and click Refresh now.

Then wait until the refresh status shows Completed.

Once the refresh is completed, return here and press Enter, and I will continue with the updated dashboard flow.
"""
    speak(message)
    open_powerbi_report()
    print("\nPower BI report opened. Please complete the manual semantic model refresh in Power BI.")
    print("After the refresh is completed, return here. The narration step will ask for final confirmation once the dashboard is fully visible.")


def prepare_dashboard_before_narration(refresh_mode: str) -> None:
    if refresh_mode == "manual":
        message = """
We will proceed with a guided manual refresh.

I am opening Power BI now.

Please refresh the semantic model and wait until the status shows Completed.

Once done, return here and press Enter to continue.
"""
        speak_long_message(message)
        open_powerbi_report()
        input("\nPress ENTER once the Power BI refresh is completed...")
        return

    if refresh_mode == "automatic":
        speak_long_message(
            "The Power B I views were updated successfully. I am now triggering the Power B I refresh automatically."
        )

        try:
            trigger_powerbi_refresh()
            wait_for_powerbi_refresh_completion()
            open_powerbi_report()
            return
        except Exception:
            speak_long_message(
                "Automatic refresh failed. Please complete the refresh manually."
            )
            open_powerbi_report()
            input("\nPress ENTER once the Power BI refresh is completed...")
            return



# =========================
# FAST MODE HELPERS
# =========================

def ask_start_mode() -> str:
    while True:
        choice = input(
            "\nHow would you like to start?\n"
            "1. Full walkthrough\n"
            "2. Quick setup\n"
            "> "
        ).strip()

        if choice == "1":
            return "full_walkthrough"

        if choice == "2":
            return "quick_setup"

        print("Invalid choice. Please enter 1 or 2.")


def ask_user_name_fast() -> str:
    while True:
        name = input("Enter your name: ").strip()
        if name:
            return name
        print("Your name cannot be empty.")


def ask_analysis_mode_fast() -> str:
    while True:
        choice = input(
            "\nSelect analysis period type:\n"
            "1. Days\n"
            "2. Weeks\n"
            "3. Months\n"
            "> "
        ).strip()

        if choice == "1":
            return "days"
        if choice == "2":
            return "weeks"
        if choice == "3":
            return "months"

        print("Invalid selection. Please enter 1, 2, or 3.")



def ask_days_period_fast() -> dict:
    while True:
        block = input("Choose a block (A=Weeks 1-4 / B=Weeks 5-8 / C=Weeks 9-12): ").strip().upper().replace(" ", "")

        try:
            week = int(input("Choose one week within this block: ").strip())
        except ValueError:
            print("Invalid input. In days mode, you must choose one single week number only.")
            continue

        if week == 1:
            prompt = "Choose the number of consecutive operational days for Week 1 (1 to 4): "
        else:
            prompt = "Choose the number of consecutive days, from 1 to 6: "

        try:
            day_count = int(input(prompt).strip())
        except ValueError:
            print("Invalid input. The number of days must be numeric.")
            continue

        is_valid, message, period = validate_days(block, week, day_count)
        print(message)

        if is_valid:
            return {
                "mode": "days",
                "block": block,
                "week": week,
                "days": day_count,
                "period": period,
            }


def ask_weeks_period_fast() -> dict:
    while True:
        block = input(
            "Choose a block:\n"
            "A = Weeks 1-4\n"
            "B = Weeks 5-8\n"
            "C = Weeks 9-12\n"
            "> "
        ).strip().upper().replace(" ", "")

        if block not in ["A", "B", "C"]:
            print("Invalid block. Please choose A, B, or C.")
            continue

        while True:
            weeks_text = input(
                "Choose 1 to 3 consecutive weeks for this block "
                "(example: 5 or 5,6 or 6,7,8): "
            ).strip()

            try:
                weeks = parse_weeks_input(weeks_text)
            except ValueError:
                print("Invalid format. Please enter only week numbers.")
                continue

            is_valid, message = validate_weeks(block, weeks)
            print(message)

            if is_valid:
                return {
                    "mode": "weeks",
                    "block": block,
                    "weeks": sorted(weeks),
                    "number_of_weeks": len(weeks),
                }


def ask_months_period_fast() -> dict:
    while True:
        selection = input(
            "Choose your monthly block "
            "(A=Month 1 / B=Month 2 / C=Month 3 / AB=Months 1-2 / BC=Months 2-3 / ABC=Months 1-3): "
        ).strip().upper().replace(" ", "")

        is_valid, message = validate_months(selection)
        print(message)

        if is_valid:
            return {
                "mode": "months",
                "monthly_block": selection,
            }





# =========================
# MAIN
# =========================

def main():
    force_aws_profile()

    mode_choice = ask_start_mode()

    if mode_choice == "quick_setup":
        print("\nExpert mode selected. Initial voice guidance skipped.\n")

        user_name = ask_user_name_fast()
        mode = ask_analysis_mode_fast()

        if mode == "days":
            result = ask_days_period_fast()
        elif mode == "weeks":
            result = ask_weeks_period_fast()
        else:
            result = ask_months_period_fast()

    else:
        print("\nFull walkthrough selected.\n")
        intro_message()
        user_name = ask_user_name()
        mode = ask_analysis_mode()

        if mode == "days":
            result = ask_days_period()
        elif mode == "weeks":
            result = ask_weeks_period()
        else:
            result = ask_months_period()

    speak(f"Thank you {user_name}. Your analysis period has been recorded successfully.")

    print("\n==============================")
    print("FINAL SELECTION SUMMARY")
    print("==============================")
    print(f"user_name: {user_name}")
    for key, value in result.items():
        print(f"{key}: {value}")

    try:
        preview_division_df, preview_product_df = preview_loaded_data(result)

        message = (
            "The selected-period data has been validated successfully. "
            "I am now preparing the Power B I dashboard as the official business view."
        )
        speak_long_message(message)

        speak_long_message("Would you like to refresh the dashboard automatically, or manually?")
        print("\nChoose refresh mode:")
        print("1. Automatic refresh")
        print("2. Manual refresh")

        choice = input("> ").strip()

        if choice == "1":
            refresh_mode = "automatic"
        else:
            refresh_mode = "manual"

        print(f"\nRefresh mode selected: {refresh_mode.upper()}")

        print("\nUpdating Power BI data source...")
        powerbi_views_updated = update_powerbi_views_from_selection(result)

        if not powerbi_views_updated:
            raise RuntimeError("Power BI views were not updated successfully.")

        prepare_dashboard_before_narration(refresh_mode)

        current_division_df, current_product_df, current_service_df = load_current_powerbi_views()

        narration = generate_graph_narration(
            current_product_df,
            current_division_df,
            current_service_df,
            result
        )

        print("\n==============================")
        print("GRAPH NARRATION")
        print("==============================")
        print(narration)

        speak("Please confirm when the dashboard is fully loaded and ready.")
        input("\nPress ENTER when the CFO is ready to start the dashboard narration...")

        speak("Starting dashboard analysis now.")

        audio_file = text_to_speech(narration)
        print(f"\nGraph narration audio saved to: {audio_file}")

        audio_path = os.path.abspath(audio_file)
        print(f"Absolute graph narration audio path: {audio_path}")
        print(f"File exists: {os.path.exists(audio_path)}")

        try:
            os.startfile(audio_path)
            print("Graph narration audio launched successfully.")
        except Exception as e:
            print(f"ERROR WHILE OPENING GRAPH NARRATION AUDIO: {e}")

        print("\nPress ENTER to continue after listening to the narration...")
        input()

        speak("You can now ask your final C F O question.")

        question = get_non_empty_input(
            "\nEnter the user's final question: ",
            "The final question cannot be empty. Please enter your C F O question."
        )

        final_result = run_finops_cfo_pipeline(result, question)

        print("\n==============================")
        print("FINAL ANSWER")
        print("==============================")
        print(final_result["answer"])
        print(f"\nAudio saved to: {final_result['audio_file']}")

    except Exception as e:
        print("\nERROR WHILE LOADING ATHENA DATA, TRIGGERING POWER BI, OR RUNNING FINAL ANALYSIS")
        print(str(e))
        speak(
            "Your selection was recorded, but I could not complete the Athena preview, "
            "Power B I refresh, or final analysis. Please review the connection, "
            "table configuration, refresh settings, or final pipeline settings before continuing."
        )


if __name__ == "__main__":
    main()