from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_db
from ..security import require_worker_token
from ..services.queue import ConflictError, claim_next_task, complete_task, fail_task, heartbeat_task, update_progress

router = APIRouter(prefix="/api/tasks", tags=["tasks"], dependencies=[Depends(require_worker_token)])


class ClaimRequest(BaseModel):
    worker_id: str
    capabilities: list[str] = Field(default_factory=list)
    interactive_only: bool = False


class AttemptRequest(BaseModel):
    attempt_id: str


class HeartbeatRequest(AttemptRequest):
    worker_id: str


class ProgressRequest(AttemptRequest):
    stage: str
    progress_pct: float = Field(ge=0, le=100)
    message: str | None = None
    telemetry: dict[str, Any] | None = None


class CompleteRequest(BaseModel):
    attempt_id: str
    result_manifest_key: str | None = None
    result_payload: dict[str, Any] = Field(default_factory=dict)


class FailRequest(AttemptRequest):
    error_code: str
    error_message: str
    retryable: bool = False


def _conflict(exc: ConflictError) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


@router.post("/claim", response_model=None)
def claim(body: ClaimRequest, session: Session = Depends(get_db)) -> dict[str, Any] | Response:
    result = claim_next_task(session, body.worker_id, body.capabilities, body.interactive_only)
    return result or Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{task_id}/heartbeat")
def heartbeat(task_id: str, body: HeartbeatRequest, session: Session = Depends(get_db)) -> dict[str, bool]:
    try:
        heartbeat_task(session, task_id, body.attempt_id, body.worker_id or "")
    except ConflictError as exc:
        raise _conflict(exc) from exc
    return {"ok": True}


@router.post("/{task_id}/progress")
def progress(task_id: str, body: ProgressRequest, session: Session = Depends(get_db)) -> dict[str, object]:
    try:
        value = update_progress(session, task_id, body.attempt_id, body.stage, body.progress_pct, body.message, body.telemetry)
    except ConflictError as exc:
        raise _conflict(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True, "progress_pct": value}


@router.post("/{task_id}/complete")
def complete(task_id: str, body: CompleteRequest, session: Session = Depends(get_db)) -> dict[str, bool]:
    try:
        complete_task(session, task_id, body.attempt_id, body.result_payload, body.result_manifest_key)
    except ConflictError as exc:
        raise _conflict(exc) from exc
    return {"ok": True}


@router.post("/{task_id}/fail")
def fail(task_id: str, body: FailRequest, session: Session = Depends(get_db)) -> dict[str, bool]:
    try:
        fail_task(session, task_id, body.attempt_id, body.error_code, body.error_message, body.retryable)
    except ConflictError as exc:
        raise _conflict(exc) from exc
    return {"ok": True}
