from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from room3d_worker.common.errors import QualityGateError


@lru_cache(maxsize=1)
def _stage_weights() -> dict[str, list[tuple[str, float]]]:
    contract = Path(__file__).resolve().parents[3] / "contracts" / "enums.json"
    try:
        payload = json.loads(contract.read_text(encoding="utf-8"))
        return {
            task: [(stage["id"], float(stage["weight"])) for stage in stages]
            for task, stages in payload["stages"].items()
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise QualityGateError(f"Could not read stage weights from {contract}: {exc}") from exc


def compute_overall_progress(task_type: str, stage_id: str, stage_fraction: float) -> float:
    if not 0.0 <= stage_fraction <= 1.0:
        raise QualityGateError("stage_fraction must be between 0 and 1")
    weights = _stage_weights()
    stages = weights.get(task_type)
    if stages is None:
        raise QualityGateError(f"No configured stage weight for {task_type}/{stage_id}")
    stage_index = next((index for index, (key, _) in enumerate(stages) if key == stage_id), None)
    if stage_index is None:
        raise QualityGateError(f"No configured stage weight for {task_type}/{stage_id}")
    completed = sum(weight for _, weight in stages[:stage_index])
    current_weight = stages[stage_index][1]
    total_weight = sum(weight for _, weight in stages)
    return 100.0 * (completed + current_weight * stage_fraction) / total_weight
