from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import boto3


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

AWS_PROFILE = os.getenv("AWS_PROFILE", "terraform-runner")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
VOICE_ID = os.getenv("VOICE_ID", "Joanna")
ENGINE = os.getenv("POLLY_ENGINE", "neural")

# Real end pause inside SSML, not just Python sleep
AUDIO_END_PAUSE_MS = int(os.getenv("AUDIO_END_PAUSE_MS", "2200"))


def get_boto3_session() -> boto3.Session:
    return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def split_long_sentences(text: str) -> str:
    """
    Make spoken output easier for Polly by simplifying dense business text
    without breaking the logic too aggressively.
    """
    text = normalize_spaces(text)

    replacements = [
        ("Executive Summary", "Executive summary."),
        ("Executive summary", "Executive summary."),
        ("Analytical Answer", "Analytical answer."),
        ("FinOps Interpretation", "Fin Ops interpretation."),
        ("CFO Priority", "C F O priority."),
        ("Actual Cost", "Actual cost"),
        ("Allocated Budget", "Allocated budget"),
        ("Variance %", "Variance percent"),
        ("Variance", "Variance"),
        ("Status", "Status"),
        ("Products Company Total", "Products company total."),
        ("Internal Company Total", "Internal company total."),
        ("Over Budget", "Over budget"),
        ("Under Budget", "Under budget"),
        ("On Budget", "On budget"),
        ("Hybrid CFO Answer", "Hybrid C F O answer."),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    # Lighter structural cleanup
    text = text.replace("•", ". ")
    text = text.replace("|", ". ")
    text = text.replace(" - ", ". ")
    text = text.replace(" / ", " slash ")

    # Only soften labels, not every colon blindly
    text = re.sub(r"\b(Status|Actual cost|Allocated budget|Variance percent|Variance|Executive summary)\s*:\s*", r"\1. ", text, flags=re.IGNORECASE)

    return normalize_spaces(text)


def expand_acronyms(text: str) -> str:
    """
    Make acronyms easier to pronounce.
    """
    acronym_map = {
        r"\bCFO\b": "C F O",
        r"\bAI\b": "A I",
        r"\bAWS\b": "A W S",
        r"\bSQL\b": "S Q L",
        r"\bRAG\b": "R A G",
        r"\bKPI\b": "K P I",
        r"\bKPIs\b": "K P I s",
        r"\bROI\b": "R O I",
        r"\bETL\b": "E T L",
        r"\bEC2\b": "E C 2",
        r"\bS3\b": "S 3",
        r"\bSSML\b": "S S M L",
        r"\bFinOps\b": "Fin Ops",
    }

    spoken = text
    for pattern, replacement in acronym_map.items():
        spoken = re.sub(pattern, replacement, spoken)

    return spoken


def currency_to_spoken(value: float) -> str:
    sign = "minus " if value < 0 else ""
    value = abs(value)

    dollars = int(value)
    cents = int(round((value - dollars) * 100))

    if dollars >= 1_000_000:
        millions = dollars / 1_000_000
        dollars_part = f"{millions:.2f}".rstrip("0").rstrip(".")
        spoken = f"{dollars_part} million dollars"
    elif dollars >= 1_000:
        thousands = dollars // 1_000
        remainder = dollars % 1_000
        if remainder == 0:
            spoken = f"{thousands} thousand dollars"
        else:
            spoken = f"{thousands} thousand and {remainder} dollars"
    else:
        spoken = f"{dollars} dollars"

    if cents > 0:
        spoken += f" and {cents} cents"

    return f"{sign}{spoken}"


def percent_to_spoken(value: float) -> str:
    sign = "minus " if value < 0 else ""
    value = abs(value)

    text = f"{value:.2f}".rstrip("0").rstrip(".")
    text = text.replace(".", " point ")
    return f"{sign}{text} percent"


def speak_currency_in_text(text: str) -> str:
    """
    Replace $ amounts with friendlier spoken forms.
    """

    def repl(match: re.Match) -> str:
        raw = match.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            return match.group(0)

        return currency_to_spoken(value)

    return re.sub(r"\$([+-]?[0-9][0-9,]*(?:\.[0-9]{1,2})?)", repl, text)


def speak_percent_in_text(text: str) -> str:
    """
    Replace percentages with a more spoken form.
    """

    def repl(match: re.Match) -> str:
        raw = match.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            return match.group(0)

        return percent_to_spoken(value)

    return re.sub(r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*%", repl, text)


def build_spoken_text(text: str) -> str:
    """
    Main preprocessing before SSML.
    """
    spoken = str(text).strip()

    # Simplify structure first
    spoken = split_long_sentences(spoken)

    # Finance notation second
    spoken = speak_currency_in_text(spoken)
    spoken = speak_percent_in_text(spoken)

    # Symbols
    spoken = spoken.replace("&", " and ")
    spoken = spoken.replace("/", " slash ")

    # Decimal fallback
    spoken = re.sub(r"(\d+)\.(\d+)", r"\1 point \2", spoken)

    # Section cleanup for hybrid answers
    spoken = spoken.replace("1.", "First.")
    spoken = spoken.replace("2.", "Second.")
    spoken = spoken.replace("3.", "Third.")

    # Acronyms last
    spoken = expand_acronyms(spoken)

    return normalize_spaces(spoken)


def build_spoken_cfo_summary(
    actual_cost: float,
    allocated_budget: float,
    variance: float,
    variance_pct: float,
    status: str,
) -> str:
    """
    Shorter and more natural executive spoken summary.
    """
    actual_cost = safe_float(actual_cost)
    allocated_budget = safe_float(allocated_budget)
    variance = safe_float(variance)
    variance_pct = safe_float(variance_pct)

    summary = (
        f"Executive summary. "
        f"Actual cost was {currency_to_spoken(actual_cost)}. "
        f"Allocated budget was {currency_to_spoken(allocated_budget)}. "
        f"Variance was {currency_to_spoken(variance)}. "
        f"That represents {percent_to_spoken(variance_pct)}. "
        f"Status. {status}."
    )

    return normalize_spaces(summary)


def build_spoken_top_driver(label: str, name: str, value: float) -> str:
    """
    Example:
    Top driver. AI Engineering. 38 thousand dollars.
    """
    return normalize_spaces(
        f"{label}. {name}. {currency_to_spoken(safe_float(value))}."
    )


def escape_ssml_text(text: str) -> str:
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def apply_ssml_formatting(text: str) -> str:
    """
    Add clearer pauses for CFO-style delivery.
    """
    safe_text = escape_ssml_text(text)

    # Major section pauses
    safe_text = safe_text.replace("Executive summary.", "Executive summary. <break time='900ms'/> ")
    safe_text = safe_text.replace("Analytical answer.", "Analytical answer. <break time='700ms'/> ")
    safe_text = safe_text.replace("Fin Ops interpretation.", "Fin Ops interpretation. <break time='700ms'/> ")
    safe_text = safe_text.replace("C F O priority.", "C F O priority. <break time='750ms'/> ")
    safe_text = safe_text.replace("Top driver.", "Top driver. <break time='700ms'/> ")
    safe_text = safe_text.replace("Status.", "Status. <break time='500ms'/> ")

    # Sentence pauses
    safe_text = safe_text.replace(". ", ". <break time='600ms'/> ")
    safe_text = safe_text.replace("? ", "? <break time='750ms'/> ")
    safe_text = safe_text.replace("! ", "! <break time='700ms'/> ")

    # Softer pauses
    safe_text = safe_text.replace(", ", ", <break time='220ms'/> ")
    safe_text = safe_text.replace(": ", ": <break time='320ms'/> ")
    safe_text = safe_text.replace("; ", "; <break time='260ms'/> ")

    # Real ending silence inside the MP3
    safe_text = f"{safe_text} <break time='{AUDIO_END_PAUSE_MS}ms'/>"

    return safe_text.strip()


def text_to_speech(text: str, output_file: Optional[str] = None, auto_open: bool = False) -> str:
    if output_file is None:
        timestamp_for_file = time.strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"cfo_voice_{timestamp_for_file}.mp3"
    else:
        output_path = Path(output_file)

    session = get_boto3_session()
    polly = session.client("polly", region_name=AWS_REGION)

    spoken_text = build_spoken_text(text)
    formatted_text = apply_ssml_formatting(spoken_text)

    ssml_text = f"""
    <speak>
        <prosody rate="88%">
            {formatted_text}
        </prosody>
    </speak>
    """

    response = polly.synthesize_speech(
        Text=ssml_text,
        TextType="ssml",
        OutputFormat="mp3",
        VoiceId=VOICE_ID,
        Engine=ENGINE,
    )

    with open(output_path, "wb") as f:
        f.write(response["AudioStream"].read())

    if auto_open:
        try:
            os.startfile(str(output_path))
        except Exception as e:
            print(f"Audio auto-play skipped: {e}")

    return str(output_path)


def speak(message: str) -> None:
    print(f"\nFinOps Agent: {message}")
    text_to_speech(message, auto_open=True)


if __name__ == "__main__":
    sample = build_spoken_cfo_summary(
        actual_cost=135000,
        allocated_budget=120000,
        variance=15000,
        variance_pct=12.5,
        status="Over budget",
    )
    print(sample)
    text_to_speech(sample, auto_open=True)