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
from ..models import Project, SceneVersion, Task
from ..security import get_current_user
from ..storage import get_storage

router = APIRouter(prefix="/api/projects", tags=["projects"])
_EXTENSIONS = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class BoundedStream:
    def __init__(self, source: BinaryIO, limit: int) -> None:
        self.source, self.limit, self.total = source, limit, 0

    def read(self, size: int = -1) -> bytes:
        remaining = self.limit + 1 - self.total
        chunk = self.source.read(min(size, remaining) if size >= 0 else remaining)
        self.total += len(chunk)
        if self.total > self.limit:
            raise ValueError("upload exceeds MAX_UPLOAD_BYTES")
        return chunk


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
    return {**_view(project), "versions": [{"id": v.id, "label": v.label} for v in versions],
            "tasks": [{"id": t.id, "type": t.task_type, "status": t.status, "progress_pct": t.progress_pct} for t in tasks],
            "edit_sessions": []}


@router.post("/{project_id}/upload")
def upload_project(project_id: str, file: UploadFile, session: Session = Depends(get_db), user_id: str | None = Depends(get_current_user)) -> dict[str, object]:
    project = _project(session, project_id, user_id)
    filename = file.filename or "upload"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in _EXTENSIONS or (file.content_type and file.content_type != _EXTENSIONS[suffix]):
        raise HTTPException(status_code=415, detail="unsupported video type")
    prefix = file.file.read(12)
    valid_magic = (suffix == ".webm" and prefix.startswith(b"\x1a\x45\xdf\xa3")) or (suffix in {".mp4", ".mov"} and prefix[4:8] == b"ftyp")
    if not valid_magic:
        raise HTTPException(status_code=415, detail="invalid video container")
    file.file.seek(0)
    key = f"{project.id}/input/video{suffix}"
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
    task = Task(project_id=project.id, task_type="reconstruct_week1", payload_json=json.dumps({"input_key": key}))
    session.add(task)
    session.commit()
    return {"project_id": project.id, "video_key": key, "bytes": written, "task_id": task.id}
