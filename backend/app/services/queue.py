from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from ..models import Task, TaskAttempt
from .progress import compute_progress


class ConflictError(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _supports(task: Task, capabilities: set[str]) -> bool:
    payload = json.loads(task.payload_json or "{}")
    required = set(payload.get("required_capabilities", []))
    return required.issubset(capabilities)


def claim_next_task(session: Session, worker_id: str, capabilities: list[str], interactive_only: bool) -> dict[str, Any] | None:
    candidates = select(Task).where(Task.status == "queued").order_by(Task.created_at, Task.id)
    if session.bind and session.bind.dialect.name == "postgresql":
        candidates = candidates.with_for_update(skip_locked=True)
    try:
        for task in session.scalars(candidates).all():
            payload = json.loads(task.payload_json or "{}")
            if interactive_only and not payload.get("interactive", False):
                continue
            if not _supports(task, set(capabilities)):
                continue
            attempt_id = str(uuid.uuid4())
            if session.bind and session.bind.dialect.name == "sqlite":
                changed = session.execute(
                    update(Task).where(Task.id == task.id, Task.status == "queued").values(
                        status="claimed", active_attempt_id=attempt_id, updated_at=_now()
                    )
                ).rowcount
                if changed != 1:
                    session.rollback()
                    continue
                session.refresh(task)
            else:
                task.status = "claimed"
                task.active_attempt_id = attempt_id
            attempt = TaskAttempt(id=attempt_id, task_id=task.id, worker_id=worker_id, status="claimed")
            session.add(attempt)
            session.commit()
            return {"task_id": task.id, "attempt_id": attempt_id, "project_id": task.project_id,
                    "task_type": task.task_type, "payload": payload}
    except OperationalError:
        session.rollback()
        raise
    session.commit()
    return None


def _locked_task(session: Session, task_id: str) -> Task:
    task = session.get(Task, task_id)
    if not task:
        raise LookupError(f"task '{task_id}' not found")
    return task


def _check_attempt(task: Task, attempt_id: str) -> None:
    if task.active_attempt_id != attempt_id:
        raise ConflictError("stale task attempt")


def heartbeat_task(session: Session, task_id: str, attempt_id: str, worker_id: str) -> None:
    task = _locked_task(session, task_id)
    _check_attempt(task, attempt_id)
    attempt = session.get(TaskAttempt, attempt_id)
    if not attempt or attempt.worker_id != worker_id:
        raise ConflictError("invalid task attempt")
    attempt.heartbeat_at = _now()
    if task.status == "claimed":
        task.status = "running"
    session.commit()


def update_progress(session: Session, task_id: str, attempt_id: str, stage: str, progress_pct: float,
                    message: str | None = None, telemetry: dict[str, Any] | None = None) -> float:
    task = _locked_task(session, task_id)
    _check_attempt(task, attempt_id)
    progress = compute_progress(task.task_type, stage, progress_pct)
    task.current_stage, task.progress_pct, task.message, task.status = stage, progress, message, "running"
    if telemetry:
        task.payload_json = json.dumps({**json.loads(task.payload_json or "{}"), "telemetry": telemetry})
    session.commit()
    return progress


def complete_task(session: Session, task_id: str, attempt_id: str, result_payload: dict[str, Any],
                  result_manifest_key: str | None = None) -> None:
    task = _locked_task(session, task_id)
    _check_attempt(task, attempt_id)
    result = dict(result_payload)
    if result_manifest_key:
        result["manifest_key"] = result_manifest_key
    task.status, task.progress_pct, task.result_json = "completed", 100.0, json.dumps(result)
    attempt = session.get(TaskAttempt, attempt_id)
    if attempt:
        attempt.status, attempt.finished_at = "completed", _now()
    session.commit()


def fail_task(session: Session, task_id: str, attempt_id: str, error_code: str, error_message: str,
              retryable: bool) -> None:
    task = _locked_task(session, task_id)
    _check_attempt(task, attempt_id)
    task.status = "queued" if retryable else "failed"
    task.retry_count += 1 if retryable else 0
    attempt = session.get(TaskAttempt, attempt_id)
    if attempt:
        attempt.status, attempt.finished_at = "failed", _now()
        attempt.error_code, attempt.error_message = error_code, error_message
    task.active_attempt_id = None
    session.commit()
