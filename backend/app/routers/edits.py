from __future__ import annotations

import json
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..models import EditSession, Project, SceneVersion, Task
from ..security import get_current_user
from ..services.celery_bridge import dispatch_task_if_configured
from ..storage import get_storage
from .projects import _project

_WORKER_SRC = Path(__file__).resolve().parents[3] / "worker"
if str(_WORKER_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKER_SRC))

from room3d_worker.common.click_resolve import resolve_viewer_click  # noqa: E402

router = APIRouter(tags=["edits"])
_MAX_MASK_BYTES = 16 * 1024 * 1024


class ResolveClickRequest(BaseModel):
    screenX: float = 0.0
    screenY: float = 0.0
    worldHitColmap: tuple[float, float, float] | None = None
    rayOriginColmap: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rayDirColmap: tuple[float, float, float] = (0.0, 0.0, -1.0)
    nearestFrameIdx: int | None = None


class ClickResolveResponse(BaseModel):
    edit_id: str
    ref_frame_idx: int
    u: float
    v: float
    score: float
    method: Literal["projection", "nearest_snap"]
    task_id: str


class PromptPoint(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    label: Literal[0, 1]


class RefineRequest(BaseModel):
    points: list[PromptPoint] = Field(min_length=1)


class PreviewSeedsRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    seeds: list[int] = Field(min_length=1, max_length=8)
    model_variant: Literal["flux2_klein_4b", "sdxl_inpaint_t4"] = "flux2_klein_4b"


class ApplyEditRequest(BaseModel):
    chosen_seed_id: str = Field(min_length=1)
    rebuild_mode: Literal["masked_finetune", "from_scratch"] = "masked_finetune"
    prompt: str | None = Field(default=None, max_length=1000)


class TaskViewResponse(BaseModel):
    id: str
    type: str
    status: str
    progress_pct: float
    stage: str | None
    message: str | None


class ApplyEditResponse(TaskViewResponse):
    edit_id: str
    target_version_id: str
    diffusion_task_id: str


class EditFrameItem(BaseModel):
    frame_idx: int
    image_relpath: str | None
    mask_key: str | None
    mask_status: str
    skip_frame: bool


class EditFramesResponse(BaseModel):
    edit_id: str
    page: int
    page_size: int
    total: int
    frames: list[EditFrameItem]


class MaskUpdateResponse(BaseModel):
    edit_id: str
    frame_idx: int
    status: str
    skip_frame: bool
    mask_key: str | None = None


class EditSummaryResponse(BaseModel):
    id: str
    project_id: str
    base_version_id: str | None
    target_version_id: str | None
    ref_frame_idx: int | None
    prompt: str | None
    chosen_seed_id: str | None
    status: str
    created_at: datetime | None
    updated_at: datetime | None


def _load_active_manifest(project: Project, session: Session) -> dict[str, Any] | None:
    if not project.active_version_id:
        return None
    version = session.scalar(
        select(SceneVersion).where(
            SceneVersion.id == project.active_version_id,
            SceneVersion.project_id == project.id,
        )
    )
    if not version or not version.manifest_relpath:
        return None
    try:
        with get_storage(get_settings()).open_read(version.manifest_relpath) as stream:
            data = json.load(stream)
            return data if isinstance(data, dict) else None
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return None


def _resolve_target(manifest: dict[str, Any] | None, body: ResolveClickRequest) -> tuple[int, float, float, float, Literal["projection", "nearest_snap"]]:
    frames = [f for f in (manifest or {}).get("frames", []) if not f.get("excluded")]
    if not frames or manifest is None:
        idx = body.nearestFrameIdx if body.nearestFrameIdx is not None else 0
        intr = (manifest or {}).get("intrinsics") or {}
        return idx, float(intr.get("cx", 320.0)), float(intr.get("cy", 240.0)), 0.0, "nearest_snap"
    res = resolve_viewer_click(
        manifest=manifest,
        world_hit_colmap=body.worldHitColmap,
        ray_origin_colmap=body.rayOriginColmap,
        ray_dir_colmap=body.rayDirColmap,
        nearest_frame_idx=body.nearestFrameIdx,
    )
    return res.ref_frame_idx, res.u, res.v, res.score, res.method


def _edit_for_user(session: Session, edit_id: str, user_id: str | None) -> tuple[EditSession, Project]:
    edit = session.get(EditSession, edit_id)
    if not edit:
        raise HTTPException(status_code=404, detail="edit session not found")
    return edit, _project(session, edit.project_id, user_id)


def _task_view(task: Task) -> TaskViewResponse:
    return TaskViewResponse(
        id=task.id,
        type=task.task_type,
        status=task.status,
        progress_pct=task.progress_pct,
        stage=task.current_stage,
        message=task.message,
    )


def _queue_edit_task(session: Session, edit: EditSession, task_type: str, payload: dict[str, Any]) -> Task:
    task = Task(
        project_id=edit.project_id,
        edit_session_id=edit.id,
        task_type=task_type,
        status="queued",
        payload_json=json.dumps(payload),
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    dispatch_task_if_configured(task.id, task.task_type, get_settings(), payload)
    return task


Image.MAX_IMAGE_PIXELS = 50_000_000

def _validate_binary_mask_png(raw: bytes, expected_size: tuple[int, int] | None) -> bytes:
    if not raw:
        raise HTTPException(status_code=422, detail="Mask PNG upload is empty")
    try:
        with Image.open(BytesIO(raw)) as img:
            if img.format != "PNG" or img.mode != "L":
                raise HTTPException(status_code=422, detail=f"Mask must be 1-channel uint8 PNG ('L'), got {img.format}/{img.mode}")
            if img.width < 64 or img.height < 64 or img.width % 16 != 0 or img.height % 16 != 0:
                raise HTTPException(status_code=422, detail=f"Mask dimensions {img.size} must be >= 64 and divisible by 16")
            if expected_size is not None and img.size != expected_size:
                raise HTTPException(status_code=422, detail=f"Mask dimensions {img.size} do not match scene {expected_size}")
            img.load()
            if img.tobytes().translate(None, b"\x00\xff"):
                raise HTTPException(status_code=422, detail="Mask pixel values must be strictly 0 or 255")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="Invalid PNG image upload") from exc
    return raw


async def _read_bounded_upload(upload: Any, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(65536):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="Mask upload exceeds size limit")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/api/projects/{project_id}/edits/resolve-click", status_code=201, response_model=ClickResolveResponse)
def resolve_click(
    project_id: str,
    body: ResolveClickRequest,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> ClickResolveResponse:
    project = _project(session, project_id, user_id)
    manifest = _load_active_manifest(project, session)
    ref_frame_idx, u, v, score, method = _resolve_target(manifest, body)
    initial_points = [{"x": u, "y": v, "label": 1}]
    edit = EditSession(
        project_id=project.id,
        base_version_id=project.active_version_id,
        ref_frame_idx=ref_frame_idx,
        points_json=json.dumps(initial_points),
        masks_summary_json=json.dumps({}),
        status="segmenting",
    )
    session.add(edit)
    session.flush()
    task = _queue_edit_task(
        session, edit, "segment_init", {"edit_id": edit.id, "ref_frame_idx": ref_frame_idx, "points": initial_points}
    )
    return ClickResolveResponse(
        edit_id=edit.id, ref_frame_idx=ref_frame_idx, u=u, v=v, score=score, method=method, task_id=task.id
    )


@router.post("/api/edits/{edit_id}/refine", response_model=TaskViewResponse)
def refine_edit(
    edit_id: str,
    body: RefineRequest,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> TaskViewResponse:
    edit, _ = _edit_for_user(session, edit_id, user_id)
    all_points = [*json.loads(edit.points_json or "[]"), *(p.model_dump() for p in body.points)]
    edit.points_json = json.dumps(all_points)
    edit.status = "segmenting"
    task = _queue_edit_task(
        session, edit, "segment_refine", {"edit_id": edit.id, "ref_frame_idx": edit.ref_frame_idx, "points": all_points}
    )
    return _task_view(task)


@router.post("/api/edits/{edit_id}/track", response_model=TaskViewResponse)
def track_edit(
    edit_id: str,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> TaskViewResponse:
    edit, _ = _edit_for_user(session, edit_id, user_id)
    edit.status = "tracking"
    task = _queue_edit_task(
        session, edit, "track_propagate", {"edit_id": edit.id, "base_version_id": edit.base_version_id, "ref_frame_idx": edit.ref_frame_idx}
    )
    return _task_view(task)


@router.get("/api/edits/{edit_id}/frames", response_model=EditFramesResponse)
def list_edit_frames(
    edit_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> EditFramesResponse:
    edit, project = _edit_for_user(session, edit_id, user_id)
    manifest = _load_active_manifest(project, session)
    summary: dict[str, Any] = json.loads(edit.masks_summary_json or "{}")
    frames_meta = (manifest or {}).get("frames") or [
        {"frame_idx": int(i), "image_relpath": f"images/frame_{int(i):06d}.jpg"} for i in sorted(summary, key=int)
    ]
    start = (page - 1) * page_size
    items: list[EditFrameItem] = []
    for item in frames_meta[start : start + page_size]:
        idx = int(item["frame_idx"])
        entry = summary.get(str(idx)) or {}
        status = str(entry.get("status") or ("skipped" if entry.get("skip_frame") else "empty"))
        items.append(
            EditFrameItem(
                frame_idx=idx,
                image_relpath=item.get("image_relpath"),
                mask_key=entry.get("mask_key"),
                mask_status=status,
                skip_frame=bool(entry.get("skip_frame", False)),
            )
        )
    return EditFramesResponse(edit_id=edit.id, page=page, page_size=page_size, total=len(frames_meta), frames=items)


def _apply_skip_flag(session: Session, edit: EditSession, frame_idx: int, payload: Any) -> MaskUpdateResponse:
    if not isinstance(payload, dict) or not isinstance(payload.get("skip_frame"), bool):
        raise HTTPException(status_code=422, detail="Expected JSON body {skip_frame: boolean}")
    summary: dict[str, Any] = json.loads(edit.masks_summary_json or "{}")
    entry = dict(summary.get(str(frame_idx)) or {})
    skip_flag = payload["skip_frame"]
    entry["skip_frame"] = skip_flag
    entry["status"] = "skipped" if skip_flag else ("manual" if entry.get("mask_key") else "empty")
    summary[str(frame_idx)] = entry
    edit.masks_summary_json = json.dumps(summary)
    session.commit()
    return MaskUpdateResponse(edit_id=edit.id, frame_idx=frame_idx, status=entry["status"], skip_frame=skip_flag, mask_key=entry.get("mask_key"))


async def _apply_mask_upload(
    session: Session, edit: EditSession, project: Project, frame_idx: int, request: Request
) -> MaskUpdateResponse:
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=422, detail="Missing 'file' multipart field")
    settings = get_settings()
    raw = await _read_bounded_upload(upload, min(settings.max_upload_bytes, _MAX_MASK_BYTES))
    manifest = _load_active_manifest(project, session)
    expected_size = (int(manifest["width"]), int(manifest["height"])) if manifest else None
    validated = _validate_binary_mask_png(raw, expected_size)
    mask_key = f"{project.id}/edits/{edit.id}/masks/{frame_idx:06d}.png"
    get_storage(settings).put_stream(mask_key, BytesIO(validated))
    summary: dict[str, Any] = json.loads(edit.masks_summary_json or "{}")
    entry = {"status": "manual", "mask_key": mask_key, "skip_frame": False}
    summary[str(frame_idx)] = entry
    edit.masks_summary_json = json.dumps(summary)
    session.commit()
    return MaskUpdateResponse(edit_id=edit.id, frame_idx=frame_idx, status="manual", skip_frame=False, mask_key=mask_key)


@router.put("/api/edits/{edit_id}/masks/{frame_idx}", response_model=MaskUpdateResponse)
async def update_edit_mask(
    edit_id: str,
    frame_idx: int,
    request: Request,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> MaskUpdateResponse:
    if frame_idx < 0:
        raise HTTPException(status_code=422, detail="frame_idx must be non-negative")
    edit, project = _edit_for_user(session, edit_id, user_id)
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        return _apply_skip_flag(session, edit, frame_idx, await request.json())
    if "multipart/form-data" in content_type:
        return await _apply_mask_upload(session, edit, project, frame_idx, request)
    raise HTTPException(status_code=415, detail="Use application/json or multipart/form-data")


@router.post("/api/edits/{edit_id}/preview-seeds", response_model=TaskViewResponse)
def preview_seeds(
    edit_id: str,
    body: PreviewSeedsRequest,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> TaskViewResponse:
    edit, _ = _edit_for_user(session, edit_id, user_id)
    edit.prompt = body.prompt
    edit.status = "previewing"
    task = _queue_edit_task(
        session,
        edit,
        "diffusion_preview",
        {"edit_id": edit.id, "ref_frame_idx": edit.ref_frame_idx, "prompt": body.prompt, "seeds": body.seeds, "model_variant": body.model_variant},
    )
    return _task_view(task)


@router.post("/api/edits/{edit_id}/apply", response_model=ApplyEditResponse)
def apply_edit(
    edit_id: str,
    body: ApplyEditRequest,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> ApplyEditResponse:
    edit, _ = _edit_for_user(session, edit_id, user_id)
    active_task = session.scalar(
        select(Task).where(
            Task.edit_session_id == edit.id,
            Task.task_type == "rebuild_week2",
            Task.status.in_(("queued", "claimed", "running")),
        )
    )
    if active_task is not None:
        raise HTTPException(status_code=409, detail="An apply/rebuild operation is already in progress for this edit")
    prompt = (body.prompt or edit.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Edit prompt must be set before applying diffusion edits")
    target_version_id = edit.target_version_id or f"edit_{edit.id[:8]}"
    edit.prompt = prompt
    edit.chosen_seed_id = body.chosen_seed_id
    edit.target_version_id = target_version_id
    edit.status = "rebuilding"
    diffusion_payload = {"edit_id": edit.id, "chosen_seed_id": body.chosen_seed_id, "prompt": prompt, "target_version_id": target_version_id}
    diffusion_task = Task(
        project_id=edit.project_id,
        edit_session_id=edit.id,
        task_type="diffusion_apply",
        status="queued",
        payload_json=json.dumps(diffusion_payload),
    )
    session.add(diffusion_task)
    session.flush()
    rebuild_task = _queue_edit_task(
        session,
        edit,
        "rebuild_week2",
        {
            "edit_id": edit.id,
            "base_version_id": edit.base_version_id,
            "target_version_id": target_version_id,
            "rebuild_mode": body.rebuild_mode,
            "depends_on_task_id": diffusion_task.id,
        },
    )
    dispatch_task_if_configured(diffusion_task.id, diffusion_task.task_type, get_settings(), diffusion_payload)
    return ApplyEditResponse(**_task_view(rebuild_task).model_dump(), edit_id=edit.id, target_version_id=target_version_id, diffusion_task_id=diffusion_task.id)


@router.get("/api/projects/{project_id}/edits", response_model=list[EditSummaryResponse])
def list_project_edits(
    project_id: str,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> list[EditSummaryResponse]:
    _project(session, project_id, user_id)
    edits = session.scalars(select(EditSession).where(EditSession.project_id == project_id).order_by(EditSession.created_at.desc())).all()
    return [
        EditSummaryResponse(
            id=e.id,
            project_id=e.project_id,
            base_version_id=e.base_version_id,
            target_version_id=e.target_version_id,
            ref_frame_idx=e.ref_frame_idx,
            prompt=e.prompt,
            chosen_seed_id=e.chosen_seed_id,
            status=e.status,
            created_at=e.created_at,
            updated_at=e.updated_at,
        )
        for e in edits
    ]
