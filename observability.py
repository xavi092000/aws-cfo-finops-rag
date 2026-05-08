import json
import time
from pathlib import Path
from typing import Any, Dict


LOG_DIR = Path("outputs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "pipeline_events.jsonl"


def log_event(event: Dict[str, Any]) -> None:
    event["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def check_latency_slo(latency_seconds: float, target_seconds: float = 5.0) -> str:
    if latency_seconds <= target_seconds:
        return "pass"
    return "fail"