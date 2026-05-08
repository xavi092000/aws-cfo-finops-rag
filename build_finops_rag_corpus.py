from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "general_questions"
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "chunks_general_questions.json"

INPUT_FILES = [
    "01_finops_principles.md",
    "02_finops_scopes.md",
    "03_finops_personas.md",
    "04_finops_domains_capabilities.md",
    "05_finops_kpis_glossary.md",
    "06_aws_cost_allocation_policy.md",
    "10_budget_forecast_unit_economics.md",
    "11_cfo_finops_best_practices.md",
    "12_finops_governance_operating_model.md",
    "13_finops_cfo_faq.md",
    "14_finops_kpi_decision_tree.md",
    "15_showback_chargeback_policy.md",
    "16_commitment_management_strategy.md",
    "17_finops_cfo_response_patterns.md",
    "18_finops_maturity_roadmap.md",
    "24_finops_agent_response_policy.md",
    "25_finops_monthly_business_review.md",
    "26_finops_general_best_practices_master.md",
]

MAX_CHARS_PER_CHUNK = 1200
MIN_CHARS_PER_CHUNK = 200


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_by_headers(text: str) -> List[str]:
    lines = text.split("\n")
    sections: List[str] = []
    current: List[str] = []

    for line in lines:
        if line.startswith("#") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    return [section for section in sections if section.strip()]


def split_large_section(section: str, max_chars: int) -> List[str]:
    if len(section) <= max_chars:
        return [section]

    paragraphs = section.split("\n\n")
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        candidate = paragraph if not current else f"{current}\n\n{paragraph}"

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def build_chunks_for_file(file_path: Path) -> List[Dict[str, str]]:
    text = clean_text(file_path.read_text(encoding="utf-8"))
    sections = split_by_headers(text)

    raw_chunks: List[str] = []
    for section in sections:
        raw_chunks.extend(split_large_section(section, MAX_CHARS_PER_CHUNK))

    final_chunks: List[Dict[str, str]] = []
    chunk_counter = 1

    for chunk_text in raw_chunks:
        chunk_text = chunk_text.strip()
        if len(chunk_text) < MIN_CHARS_PER_CHUNK:
            continue

        final_chunks.append(
            {
                "filename": file_path.name,
                "chunk_id": str(chunk_counter),
                "text": chunk_text,
            }
        )
        chunk_counter += 1

    return final_chunks


def main() -> None:
    all_chunks: List[Dict[str, str]] = []

    print(f"Docs dir    : {DOCS_DIR}")
    print(f"Output file : {OUTPUT_FILE}\n")

    for filename in INPUT_FILES:
        file_path = DOCS_DIR / filename

        if not file_path.exists():
            print(f"[WARNING] Missing file: {filename}")
            continue

        file_chunks = build_chunks_for_file(file_path)
        all_chunks.extend(file_chunks)
        print(f"[OK] {filename}: {len(file_chunks)} chunks")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("\n============================================================")
    print("FINOPS RAG CORPUS BUILT")
    print("============================================================")
    print(f"Output file : {OUTPUT_FILE}")
    print(f"Total chunks: {len(all_chunks)}")


if __name__ == "__main__":
    main()