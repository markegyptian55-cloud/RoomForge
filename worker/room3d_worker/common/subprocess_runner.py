from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass

from room3d_worker.common.errors import StageExecutionError


@dataclass(frozen=True)
class StageResult:
    command: tuple[str, ...]
    returncode: int
    elapsed_sec: float
    stdout: str
    stderr: str


def _normalize_cuda_device(cuda_device: str) -> str:
    normalized = cuda_device.strip()
    if not normalized:
        raise StageExecutionError("cuda_device must not be empty or whitespace")
    if normalized.lower().startswith("cuda:"):
        normalized = normalized[5:]
    if not normalized:
        raise StageExecutionError("cuda_device must include a device index or UUID")
    return normalized


def run_isolated_stage(cmd: list[str], cuda_device: str | None, env_extra: dict[str, str] | None, timeout_sec: int) -> StageResult:
    if not cmd or timeout_sec <= 0:
        raise StageExecutionError("Stage command must be non-empty and timeout_sec must be positive")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = _normalize_cuda_device(cuda_device)
    started = time.perf_counter()
    try:
        result = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True, timeout=timeout_sec, shell=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StageExecutionError(f"Stage command failed to execute {cmd!r}: {exc}") from exc
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise StageExecutionError(
            f"Stage command exited {result.returncode}: {cmd!r}; stderr={result.stderr.strip()}"
        )
    return StageResult(tuple(cmd), result.returncode, elapsed, result.stdout, result.stderr)
