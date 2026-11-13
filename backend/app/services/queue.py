from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from ..models import EditSession, Project, SceneVersion, Task, TaskAttempt
from ..storage.base import validate_key
from .progress import compute_progress

MAX_TASK_RETRIES = 3


class ConflictError(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_payload(session: Session, task: Task) -> dict[str, Any] | None:
    try:
        parsed = json.loads(task.payload_json or "{}")
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    task.status = "failed"
    task.message = "Malformed task payload_json"
    _mark_edit_failed(session, task)
    session.commit()
    return None


def _supports(task: Task, payload: dict[str, Any], capabilities: set[str]) -> bool:
    req_raw = payload.get("required_capabilities")
    if req_raw is None:
        required: set[str] = set()
    elif isinstance(req_raw, list):
        required = {str(item) for item in req_raw}
    else:
        return False
    if not required.issubset(capabilities):
        return False
    return not capabilities or not task.task_type or task.task_type in capabilities


def _mark_edit_failed(session: Session, task: Task) -> None:
    if not task.edit_session_id:
        return
    edit = session.get(EditSession, task.edit_session_id)
    if edit is not None:
        edit.status = "failed"


def _dependency_ready(session: Session, task: Task, payload: dict[str, Any]) -> bool:
    dep_id = payload.get("depends_on_task_id")
    if not isinstance(dep_id, str):
        return True
    dep_task = session.get(Task, dep_id)
    if dep_task is None or dep_task.status in {"failed", "cancelled"}:
        reason = dep_task.status if dep_task is not None else "missing"
        task.status = "failed"
        task.message = f"Prerequisite task '{dep_id}' is {reason}"
        _mark_edit_failed(session, task)
        session.commit()
        return False
    return dep_task.status == "completed"


_INTERACTIVE_TASKS = frozenset({"segment_init", "segment_refine"})


def _is_interactive(task: Task, payload: dict[str, Any]) -> bool:
    return task.task_type in _INTERACTIVE_TASKS or bool(payload.get("interactive", False))


def _preflight_claim(
    session: Session, task_id: str, capabilities: list[str], interactive_only: bool
) -> tuple[Task, dict[str, Any]] | None:
    task = session.get(Task, task_id)
    if task is None or task.status != "queued":
        return None
    payload = _parse_payload(session, task)
    if payload is None:
        return None
    if (interactive_only and not _is_interactive(task, payload)) or not _supports(task, payload, set(capabilities)):
        return None
    if not _dependency_ready(session, task, payload):
        return None
    return task, payload


def _try_claim_sqlite(
    session: Session, task_id: str, worker_id: str, capabilities: list[str], interactive_only: bool
) -> dict[str, Any] | None:
    ready = _preflight_claim(session, task_id, capabilities, interactive_only)
    if ready is None:
        return None
    task, payload = ready
    attempt_id = str(uuid.uuid4())
    changed = session.execute(
        update(Task).where(Task.id == task.id, Task.status == "queued").values(
            status="claimed", active_attempt_id=attempt_id, updated_at=_now()
        )
    ).rowcount
    if changed != 1:
        session.rollback()
        return None
    session.add(TaskAttempt(id=attempt_id, task_id=task.id, worker_id=worker_id, status="claimed"))
    session.commit()
    return {"task_id": task.id, "attempt_id": attempt_id, "project_id": task.project_id,
            "task_type": task.task_type, "payload": payload}


def _try_claim_postgres(
    session: Session, task_id: str, worker_id: str, capabilities: list[str], interactive_only: bool
) -> dict[str, Any] | None:
    ready = _preflight_claim(session, task_id, capabilities, interactive_only)
    if ready is None:
        return None
    _, payload = ready
    stmt = (
        select(Task)
        .where(Task.id == task_id, Task.status == "queued")
        .with_for_update(skip_locked=True)
        .execution_options(populate_existing=True)
    )
    task = session.scalars(stmt).first()
    if task is None:
        return None
    attempt_id = str(uuid.uuid4())
    task.status = "claimed"
    task.active_attempt_id = attempt_id
    task.updated_at = _now()
    session.add(TaskAttempt(id=attempt_id, task_id=task.id, worker_id=worker_id, status="claimed"))
    session.commit()
    return {"task_id": task.id, "attempt_id": attempt_id, "project_id": task.project_id,
            "task_type": task.task_type, "payload": payload}


def claim_next_task(session: Session, worker_id: str, capabilities: list[str], interactive_only: bool) -> dict[str, Any] | None:
    stmt = select(Task.id).where(Task.status == "queued")
    if capabilities:
        stmt = stmt.where(Task.task_type.in_(capabilities))
    if interactive_only:
        stmt = stmt.where(Task.task_type.in_(_INTERACTIVE_TASKS) | Task.payload_json.contains("interactive"))
    candidate_ids = list(session.scalars(stmt.order_by(Task.created_at, Task.id).limit(50)).all())
    is_pg = bool(session.bind and session.bind.dialect.name == "postgresql")
    try:
        for task_id in candidate_ids:
            claimed = (_try_claim_postgres if is_pg else _try_claim_sqlite)(
                session, task_id, worker_id, capabilities, interactive_only
            )
            if claimed is not None:
                return claimed
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


def _check_active_attempt(task: Task, attempt_id: str) -> None:
    if task.status not in {"claimed", "running"}:
        raise ConflictError("task is no longer active")
    if task.active_attempt_id != attempt_id:
        raise ConflictError("stale task attempt")


def heartbeat_task(session: Session, task_id: str, attempt_id: str, worker_id: str) -> None:
    task = _locked_task(session, task_id)
    _check_active_attempt(task, attempt_id)
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
    _check_active_attempt(task, attempt_id)
    progress = compute_progress(task.task_type, stage, progress_pct)
    task.current_stage, task.progress_pct, task.message, task.status = stage, progress, message, "running"
    if telemetry:
        task.payload_json = json.dumps({**json.loads(task.payload_json or "{}"), "telemetry": telemetry})
    session.commit()
    return progress


def _register_completed_version(
    session: Session,
    task: Task,
    result: dict[str, Any],
    result_manifest_key: str | None,
) -> None:
    if task.task_type not in {"reconstruct_week1", "rebuild_week2"}:
        return
    version_id = str(result.get("version_id") or "")
    if not version_id and result_manifest_key:
        parts = result_manifest_key.split("/")
        if len(parts) >= 3:
            version_id = parts[1]
    if not version_id:
        return
    existing = session.get(SceneVersion, version_id)
    if existing is None:
        session.add(
            SceneVersion(
                id=version_id,
                project_id=task.project_id,
                parent_version_id=result.get("parent_version_id"),
                label=str(result.get("label") or version_id),
                manifest_relpath=result_manifest_key,
                spz_relpath=f"{task.project_id}/{version_id}/scene.spz",
                ply_relpath=f"{task.project_id}/{version_id}/scene.ply",
            )
        )
    else:
        existing.manifest_relpath = result_manifest_key or existing.manifest_relpath
    project = session.get(Project, task.project_id)
    if project is not None:
        project.active_version_id = version_id
        project.status = "completed"
    if task.edit_session_id:
        edit = session.get(EditSession, task.edit_session_id)
        if edit is not None:
            edit.target_version_id = version_id
            edit.status = "completed"


def complete_task(session: Session, task_id: str, attempt_id: str, result_payload: dict[str, Any],
                  result_manifest_key: str | None = None) -> None:
    task = _locked_task(session, task_id)
    _check_active_attempt(task, attempt_id)
    result = dict(result_payload)
    if result_manifest_key:
        result_manifest_key = validate_key(result_manifest_key)
        result["manifest_key"] = result_manifest_key
    task.status, task.progress_pct, task.result_json = "completed", 100.0, json.dumps(result)
    task.active_attempt_id = None
    attempt = session.get(TaskAttempt, attempt_id)
    if attempt:
        attempt.status, attempt.finished_at = "completed", _now()
    _register_completed_version(session, task, result, result_manifest_key)
    session.commit()


def fail_task(session: Session, task_id: str, attempt_id: str, error_code: str, error_message: str,
              retryable: bool) -> None:
    task = _locked_task(session, task_id)
    _check_active_attempt(task, attempt_id)
    should_retry = retryable and task.retry_count < MAX_TASK_RETRIES
    task.status = "queued" if should_retry else "failed"
    if should_retry:
        task.retry_count += 1
    else:
        _mark_edit_failed(session, task)
    attempt = session.get(TaskAttempt, attempt_id)
    if attempt:
        attempt.status, attempt.finished_at = "failed", _now()
        attempt.error_code, attempt.error_message = error_code, error_message
    task.active_attempt_id = None
    session.commit()
