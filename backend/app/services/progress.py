from __future__ import annotations

import json
from pathlib import Path


_ENUMS = Path(__file__).resolve().parents[3] / "contracts" / "enums.json"


def _weights() -> dict[str, list[tuple[str, float]]]:
    data = json.loads(_ENUMS.read_text(encoding="utf-8"))
    return {name: [(item["id"], float(item["weight"])) for item in stages]
            for name, stages in data["stages"].items()}


def compute_progress(task_type: str, stage: str, stage_progress: float) -> float:
    """Return weighted task progress, always bounded to the public 0..100 contract."""
    value = max(0.0, min(100.0, float(stage_progress)))
    stages = _weights().get(task_type)
    if not stages:
        return value
    for index, (stage_id, weight) in enumerate(stages):
        if stage_id == stage:
            completed = sum(item[1] for item in stages[:index])
            return max(0.0, min(100.0, completed + weight * value / 100.0))
    raise ValueError(f"unknown stage '{stage}' for task type '{task_type}'")
