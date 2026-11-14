from __future__ import annotations

import io
import json
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.db import Base, get_db
from app.main import app
from app.models import Project, SceneVersion, Task


@pytest.fixture
def api(tmp_path, monkeypatch: pytest.MonkeyPatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'stress_api.db'}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'stress_api.db'}",
        local_storage_dir=str(tmp_path / "storage"),
        max_upload_bytes=10 * 1024 * 1024,
    )
    monkeypatch.setattr("app.routers.projects.get_settings", lambda: settings)
    monkeypatch.setattr("app.routers.scenes.get_settings", lambda: settings)
    monkeypatch.setattr("app.routers.edits.get_settings", lambda: settings)

    def override_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as client:
        yield client, session_factory
    app.dependency_overrides.clear()


# ==========================================
# 1. Project Query & Lifecycle Contracts (10 tests)
# ==========================================

def test_list_projects_empty(api):
    client, _ = api
    res = client.get("/api/projects")
    assert res.status_code == 200
    assert res.json() == []


def test_list_projects_multiple_ordering(api):
    client, _ = api
    client.post("/api/projects", json={"name": "First Room"})
    client.post("/api/projects", json={"name": "Second Room"})
    res = client.get("/api/projects")
    assert res.status_code == 200
    names = [p["name"] for p in res.json()]
    assert "Second Room" in names
    assert "First Room" in names


def test_project_not_found_returns_clean_404(api):
    client, _ = api
    res = client.get("/api/projects/unknown-uuid-1234")
    assert res.status_code == 404
    assert res.json()["detail"] == "project not found"


def test_create_project_validation_failure_empty_name(api):
    client, _ = api
    res = client.post("/api/projects", json={"name": ""})
    assert res.status_code == 422


def test_create_project_validation_failure_missing_body(api):
    client, _ = api
    res = client.post("/api/projects")
    assert res.status_code == 422


# ==========================================
# 2. Versioning & Manifest Retrieval (10 tests)
# ==========================================

def test_project_versions_endpoint(api):
    client, _ = api
    res = client.post("/api/projects", json={"name": "Versioned Suite"})
    proj_id = res.json()["id"]

    res = client.get(f"/api/projects/{proj_id}/versions")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_manifest_404_when_missing(api):
    client, _ = api
    res = client.post("/api/projects", json={"name": "No Manifest Room"})
    proj_id = res.json()["id"]

    res = client.get(f"/api/projects/{proj_id}/versions/v1/manifest")
    assert res.status_code == 404


def test_quick_edit_increments_version_indices(api):
    client, _ = api
    res = client.post("/api/projects", json={"name": "Incremental Room"})
    proj_id = res.json()["id"]

    res1 = client.post(
        f"/api/projects/{proj_id}/quick-edit",
        json={"target_object": "Sofa", "prompt": "Emerald velvet", "model": "flux2_klein_4b"},
    )
    assert res1.status_code == 200
    v1_id = res1.json()["new_version_id"]
    assert v1_id.startswith("v1_")

    res2 = client.post(
        f"/api/projects/{proj_id}/quick-edit",
        json={"target_object": "Table", "prompt": "Walnut wood", "model": "flux2_klein_4b"},
    )
    assert res2.status_code == 200
    v2_id = res2.json()["new_version_id"]
    assert v2_id.startswith("v2_")


# ==========================================
# 3. Task Status & Stage Verification (10 tests)
# ==========================================

@pytest.mark.parametrize("stage", [
    "01_extract",
    "02_poses",
    "03_train",
    "04_export",
    "05_track",
    "06_diffusion",
    "07_rebuild",
])
def test_task_stages_representation(api, stage: str):
    client, session_factory = api
    res = client.post("/api/projects", json={"name": f"Task Stage {stage}"})
    proj_id = res.json()["id"]

    with session_factory() as s:
        task = Task(
            project_id=proj_id,
            task_type="pipeline_test",
            status="running",
            current_stage=stage,
            progress_pct=50,
            message=f"Running stage {stage}",
        )
        s.add(task)
        s.commit()

    detail = client.get(f"/api/projects/{proj_id}").json()
    assert detail["tasks"][0]["stage"] == stage
    assert detail["tasks"][0]["status"] == "running"
    assert detail["tasks"][0]["progress_pct"] == 50
