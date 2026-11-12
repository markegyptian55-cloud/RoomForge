from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.db import Base, get_db
from app.main import app
from app.models import Project, SceneVersion, Task
from app.storage import LocalStorage


@pytest.fixture()
def api(tmp_path, monkeypatch: pytest.MonkeyPatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'api.db'}", connect_args={"check_same_thread": False}, poolclass=NullPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'api.db'}", local_storage_dir=str(tmp_path / "storage"), max_upload_bytes=1024)
    monkeypatch.setattr("app.routers.projects.get_settings", lambda: settings)
    monkeypatch.setattr("app.routers.scenes.get_settings", lambda: settings)

    def override_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as client:
        yield client, session_factory, settings
    app.dependency_overrides.clear()


def _video(size: int = 32) -> bytes:
    return b"\x00\x00\x00\x18ftypmp42" + b"x" * size


def test_register_login_upload_and_task_creation(api) -> None:
    client, session_factory, _ = api
    registered = client.post("/api/auth/register", json={"email": "user@example.test", "password": "correct horse"})
    assert registered.status_code == 201
    assert client.post("/api/auth/login", json={"email": "user@example.test", "password": "correct horse"}).status_code == 200
    project = client.post("/api/projects", json={"name": "Demo"}).json()
    response = client.post(f"/api/projects/{project['id']}/upload", files={"file": ("room.mp4", _video(), "video/mp4")})
    assert response.status_code == 200
    with session_factory() as session:
        assert session.query(Task).filter_by(project_id=project["id"], task_type="reconstruct_week1").count() == 1


def test_only_one_competing_worker_claims(api) -> None:
    client, session_factory, _ = api
    project = client.post("/api/projects", json={"name": "Queue"}).json()
    client.post(f"/api/projects/{project['id']}/upload", files={"file": ("room.mp4", _video(), "video/mp4")})

    def claim(worker: str):
        return client.post("/api/tasks/claim", headers={"X-Worker-Token": "dev-worker-token"}, json={"worker_id": worker, "capabilities": []})

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ["w1", "w2"]))
    assert sum(result.status_code == 200 for result in results) == 1


def test_stale_completion_and_progress_persist(api) -> None:
    client, session_factory, _ = api
    project = client.post("/api/projects", json={"name": "Progress"}).json()
    client.post(f"/api/projects/{project['id']}/upload", files={"file": ("room.mp4", _video(), "video/mp4")})
    claimed = client.post("/api/tasks/claim", headers={"X-Worker-Token": "dev-worker-token"}, json={"worker_id": "w1"}).json()
    headers = {"X-Worker-Token": "dev-worker-token"}
    assert client.post(f"/api/tasks/{claimed['task_id']}/progress", headers=headers, json={"attempt_id": claimed["attempt_id"], "stage": "01_extract", "progress_pct": 50}).status_code == 200
    with session_factory() as session:
        task = session.get(Task, claimed["task_id"])
        assert task and task.progress_pct == 7.5
    stale = client.post(f"/api/tasks/{claimed['task_id']}/complete", headers=headers, json={"attempt_id": "stale", "result_payload": {}})
    assert stale.status_code == 409


def test_storage_traversal_is_rejected(api) -> None:
    client, _, _ = api
    project = client.post("/api/projects", json={"name": "Storage"}).json()
    response = client.get(f"/api/storage/{project['id']}/version/../secret")
    assert response.status_code in {400, 403, 404}
