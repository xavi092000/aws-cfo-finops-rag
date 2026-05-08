from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os
import re
import time
import warnings
from typing import Any

import boto3
import pandas as pd

from resolve_finops_analytics import load_selected_period_data, force_aws_profile
from cfo_finops_athena_rag_final import run_finops_cfo_pipeline

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


def update_powerbi_views_from_selection(selection: dict[str, Any]) -> None:
    mode = selection.get("mode")

    if mode != "weeks":
        print(f"\nPower BI views update skipped for mode: {mode}")
        return

    weeks = selection.get("weeks", [])
    if not weeks:
        raise ValueError("Weeks mode selected, but no weeks were provided in selection.")

    week_sql = build_week_in_clause(weeks)

    product_view_sql = f"""
    CREATE OR REPLACE VIEW current_product_view AS
    SELECT
        week AS period_value,
        product,
        actual_cost_usd,
        allocated_budget_usd,
        variance_usd,
        variance_percent
    FROM {ATHENA_SCHEMA}.mart_product_cfo_fixed
    WHERE week IN ({week_sql})
    """

    division_view_sql = f"""
    CREATE OR REPLACE VIEW current_division_view AS
    SELECT
        week AS period_value,
        division,
        actual_cost_usd,
        allocated_budget_usd,
        variance_usd,
        variance_percent
    FROM {ATHENA_SCHEMA}.mart_division_cfo
    WHERE week IN ({week_sql})
    """

    print("\nUpdating Power BI views in Athena...")
    execute_athena_query(product_view_sql)
    execute_athena_query(division_view_sql)
    print("Power BI views updated successfully.")


# =========================
# VOICE HELPERS
# =========================
def build_spoken_text(text: str) -> str:
    spoken = text.strip()

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


def text_to_speech(text: str, output_file=None) -> str:
    if output_file is None:
        timestamp_for_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_file = OUTPUT_DIR / f"agent_prompt_{timestamp_for_file}.mp3"
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


def speak(message: str) -> None:
    print(f"\nFinOps Agent: {message}")
    spoken_text = build_spoken_text(message)
    audio_file = text_to_speech(spoken_text)

    try:
        os.startfile(audio_file)
        time.sleep(AUDIO_END_PAUSE_SECONDS)
    except Exception as e:
        print(f"Audio auto-play skipped: {e}")


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

This pipeline will help you configure your financial analysis period.

First, you will provide your name.

Then, you will select your analysis mode.

You can choose between days, weeks, or months.

Important rule.
Week 1 includes a holiday shutdown period.
Monday and Tuesday of Week 1 are not available for analysis.
Operational selection for Week 1 starts on Wednesday.

The assistant will now guide you step by step.

Let's begin.
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
    speak(f"Hello {name}. I will now configure your analysis period.")
    return name


# =========================
# STEP 2 - MODE
# =========================
def ask_analysis_mode() -> str:
    while True:
        speak("Please choose your analysis mode. Available options are days, weeks, or months.")
        mode = normalize_mode(
            get_non_empty_input(
                "Mode, days, weeks, or months: ",
                "The mode cannot be empty. Please enter days, weeks, or months."
            )
        )

        if mode in ["days", "weeks", "months"]:
            return mode

        speak("Invalid mode. Please choose days, weeks, or months.")


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
        speak("You selected days mode.")
        speak("Available blocks are A for weeks 1 to 4, B for weeks 5 to 8, and C for weeks 9 to 12.")

        block = normalize_block(
            get_non_empty_input(
                "Choose a block, A, B, or C: ",
                "The block cannot be empty. Please choose A, B, or C."
            )
        )

        if block not in BLOCKS:
            speak("Invalid block. Please choose A, B, or C.")
            continue

        speak(f"You selected block {block}. Available weeks are {BLOCKS[block]}.")

        try:
            week = int(get_non_empty_input(
                "Choose a week within this block: ",
                "The week cannot be empty. Please enter a week number."
            ))
        except ValueError:
            speak("Invalid input. The week must be a number.")
            continue

        if week == 1:
            speak(
                "Important reminder. For Week 1, Monday and Tuesday are not available because of the holiday shutdown. "
                "You may only select consecutive operational days from Wednesday to Saturday."
            )
            day_prompt = "Choose the number of consecutive operational days for Week 1, from 1 to 4: "
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
            f"Reminder. Week 1 includes a holiday shutdown, so Monday and Tuesday are non-operational days."
        )

    return True, f"Valid selection. Block {block}, weeks {weeks_sorted}."


def ask_weeks_period() -> dict:
    while True:
        speak("You selected weeks mode.")
        speak("Please choose block A, B, or C.")
        speak("Reminder. Weeks must be consecutive and cannot exceed three weeks.")
        speak("Additional rule. If Week 1 is included, Monday and Tuesday are treated as non-operational holiday shutdown days.")

        block = normalize_block(
            get_non_empty_input(
                "Choose a block, A, B, or C: ",
                "The block cannot be empty. Please choose A, B, or C."
            )
        )

        if block not in BLOCKS:
            speak("Invalid block. Please choose A, B, or C.")
            continue

        speak(f"You selected block {block}. Available weeks are {BLOCKS[block]}.")

        weeks_text = get_non_empty_input(
            "Choose 1 to 3 consecutive weeks, example 1 or 1,2 or 6,7,8: ",
            "The weeks field cannot be empty. Please enter one, two, or three consecutive weeks."
        )

        try:
            weeks = parse_weeks_input(weeks_text)
        except ValueError:
            speak("Invalid format. Please enter only week numbers.")
            continue

        is_valid, message = validate_weeks(block, weeks)
        speak(message)

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
        speak("You selected months mode.")
        speak("Available options are A, B, C, AB, BC, or ABC.")
        speak("If your monthly scope includes Week 1, Monday and Tuesday of Week 1 are treated as non-operational holiday shutdown days.")

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
def print_preview(df: pd.DataFrame, title: str, max_rows: int = 5) -> None:
    print("\n" + "=" * 30)
    print(title)
    print("=" * 30)

    if df.empty:
        print("No rows returned.")
        return

    print(df.head(max_rows).to_string(index=False))


def preview_loaded_data(selection: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    division_df, product_df = load_selected_period_data(selection)

    print_preview(division_df, "DIVISION PREVIEW")
    print_preview(product_df, "PRODUCT PREVIEW")

    return division_df, product_df


# =========================
# MAIN
# =========================
def main():
    force_aws_profile()

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
        division_df, product_df = preview_loaded_data(result)

        speak(
            "Your selection has been applied successfully. "
            "Division and product preview data are now displayed on screen."
        )

        print("\nLoaded successfully.")
        print(f"Division rows: {len(division_df)}")
        print(f"Product rows: {len(product_df)}")

        update_powerbi_views_from_selection(result)

        if result.get("mode") == "weeks":
            speak(
                "Your Power B I charts have been updated successfully. "
                "Please refresh your Power B I report to see the new selected period. "
                "You can now ask your final C F O question."
            )
        else:
            speak(
                "The Athena preview completed successfully. "
                "Automatic Power B I chart updates are currently enabled for weeks mode only. "
                "You can now ask your final C F O question."
            )

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
        print("\nERROR WHILE LOADING ATHENA DATA, UPDATING POWER BI VIEWS, OR RUNNING FINAL ANALYSIS")
        print(str(e))
        speak(
            "Your selection was recorded, but I could not complete the Athena preview, "
            "the Power B I chart update, or the final analysis. "
            "Please review the Athena connection, view configuration, or pipeline settings before continuing."
        )


if __name__ == "__main__":
    main()