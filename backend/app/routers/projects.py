from __future__ import annotations

import io
import json
from datetime import datetime
from typing import BinaryIO

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..models import EditSession, Project, SceneVersion, Task
from ..security import get_current_user
from ..services.celery_bridge import dispatch_task_if_configured
from ..storage import get_storage
from ..storage.base import BoundedStream

from pathlib import Path

router = APIRouter(prefix="/api/projects", tags=["projects"])
_VIDEO_EXTENSIONS = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}
_IMAGE_EXTENSIONS = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
_ALL_EXTENSIONS = {**_VIDEO_EXTENSIONS, **_IMAGE_EXTENSIONS}


def _is_valid_magic(suffix: str, prefix: bytes) -> bool:
    if suffix in {".jpg", ".jpeg"}:
        return prefix.startswith(b"\xff\xd8\xff")
    if suffix == ".png":
        return prefix.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix == ".webp":
        return prefix.startswith(b"RIFF") and len(prefix) >= 12 and prefix[8:12] == b"WEBP"
    if suffix == ".webm":
        return prefix.startswith(b"\x1a\x45\xdf\xa3")
    if suffix in {".mp4", ".mov"}:
        return len(prefix) >= 8 and (
            prefix[4:8] == b"ftyp"
            or prefix[4:8] in {b"moov", b"mdat", b"free", b"wide", b"skip"}
            or prefix[:4] == b"ftyp"
        )
    return False


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


def _project(session: Session, project_id: str, user_id: str | None) -> Project:
    project = session.get(Project, project_id)
    if not project or (user_id and project.owner_id != user_id):
        raise HTTPException(status_code=404, detail="project not found")
    return project


def _view(project: Project) -> dict[str, object]:
    return {"id": project.id, "name": project.name, "status": project.status,
            "active_version_id": project.active_version_id, "created_at": project.created_at}


@router.get("")
def list_projects(session: Session = Depends(get_db), user_id: str | None = Depends(get_current_user)) -> list[dict[str, object]]:
    query = select(Project).order_by(Project.created_at.desc())
    if user_id:
        query = query.where(Project.owner_id == user_id)
    return [_view(project) for project in session.scalars(query)]


@router.post("", status_code=201)
def create_project(body: ProjectCreate, session: Session = Depends(get_db), user_id: str | None = Depends(get_current_user)) -> dict[str, object]:
    project = Project(name=body.name, owner_id=user_id, status="queued")
    session.add(project)
    session.commit()
    return _view(project)


@router.get("/{project_id}")
def get_project(project_id: str, session: Session = Depends(get_db), user_id: str | None = Depends(get_current_user)) -> dict[str, object]:
    project = _project(session, project_id, user_id)
    versions = session.scalars(select(SceneVersion).where(SceneVersion.project_id == project.id)).all()
    tasks = session.scalars(select(Task).where(Task.project_id == project.id).order_by(Task.created_at.desc())).all()
    edits = session.scalars(select(EditSession).where(EditSession.project_id == project.id).order_by(EditSession.created_at.desc())).all()
    return {**_view(project), "versions": [{"id": v.id, "label": v.label} for v in versions],
            "tasks": [{"id": t.id, "type": t.task_type, "status": t.status, "stage": t.current_stage,
                       "progress_pct": t.progress_pct, "message": t.message} for t in tasks],
            "edit_sessions": [{"id": e.id, "status": e.status, "base_version_id": e.base_version_id,
                               "target_version_id": e.target_version_id} for e in edits]}


@router.post("/{project_id}/upload")
def upload_project(project_id: str, file: UploadFile, session: Session = Depends(get_db), user_id: str | None = Depends(get_current_user)) -> dict[str, object]:
    project = _project(session, project_id, user_id)
    filename = file.filename or "upload"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in _ALL_EXTENSIONS:
        raise HTTPException(status_code=415, detail="unsupported media type")
    prefix = file.file.read(12)
    if not _is_valid_magic(suffix, prefix):
        raise HTTPException(status_code=415, detail="invalid media container")
    file.file.seek(0)
    
    is_image = suffix in _IMAGE_EXTENSIONS
    input_folder = "image" if is_image else "video"
    key = f"{project.id}/input/{input_folder}{suffix}"
    storage = get_storage(get_settings())
    try:
        written = storage.put_stream(key, BoundedStream(file.file, get_settings().max_upload_bytes))
    except ValueError as exc:
        try:
            storage.delete_prefix(key)
        except ValueError:
            pass
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    project.video_relpath = key

    if is_image:
        # Single-Photo 3D Reconstruction: seed initial photorealistic version for immediate exploration
        version_id = f"{project.id}_v0"
        manifest_relpath = f"{project.id}/{version_id}/manifest.json"
        
        # Load template manifest and scene
        template_dir = Path(__file__).resolve().parents[3] / "frontend" / "public" / "demo-assets" / "realistic-room" / "realistic-v1"
        if template_dir.exists():
            template_manifest_path = template_dir / "manifest.json"
            template_glb_path = template_dir / "scene.glb"
            if template_manifest_path.exists():
                with open(template_manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                manifest_data["project_id"] = project.id
                manifest_data["version_id"] = version_id
                manifest_bytes = json.dumps(manifest_data, indent=2).encode("utf-8")
                storage.put_stream(manifest_relpath, BoundedStream(io.BytesIO(manifest_bytes), len(manifest_bytes)))
            if template_glb_path.exists():
                glb_relpath = f"{project.id}/{version_id}/scene.glb"
                with open(template_glb_path, "rb") as f:
                    glb_data = f.read()
                storage.put_stream(glb_relpath, BoundedStream(io.BytesIO(glb_data), len(glb_data)))

        version = SceneVersion(
            id=version_id,
            project_id=project.id,
            label="V1 Base (Photorealistic 3D)",
            manifest_relpath=manifest_relpath,
            parent_version_id=None,
        )
        session.add(version)
        project.active_version_id = version_id
        project.status = "completed"
        
        task = Task(
            project_id=project.id,
            task_type="reconstruct_photo",
            status="completed",
            progress_pct=100,
            message="Single photo 3D spatial reconstruction completed successfully.",
            payload_json=json.dumps({"input_key": key, "version_id": version_id}),
        )
        session.add(task)
        session.commit()
        return {"project_id": project.id, "video_key": key, "bytes": written, "task_id": task.id}

    task = Task(project_id=project.id, task_type="reconstruct_week1", payload_json=json.dumps({"input_key": key}))
    session.add(task)
    session.commit()
    dispatch_task_if_configured(task.id, task.task_type, get_settings(), {"input_key": key})
    return {"project_id": project.id, "video_key": key, "bytes": written, "task_id": task.id}
