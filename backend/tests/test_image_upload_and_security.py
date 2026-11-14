from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.db import Base, get_db
from app.main import app


@pytest.fixture
def api(tmp_path, monkeypatch: pytest.MonkeyPatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'img_api.db'}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'img_api.db'}",
        local_storage_dir=str(tmp_path / "storage"),
        max_upload_bytes=10 * 1024 * 1024,
    )
    monkeypatch.setattr("app.routers.projects.get_settings", lambda: settings)
    monkeypatch.setattr("app.routers.scenes.get_settings", lambda: settings)

    def override_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_create_project_and_upload_jpeg(api: TestClient):
    # 1. Create project
    res = api.post("/api/projects", json={"name": "Photo Living Room"})
    assert res.status_code == 201
    proj_id = res.json()["id"]

    # 2. Upload valid JPEG image with valid magic bytes
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00" + b"\x00" * 200
    res = api.post(
        f"/api/projects/{proj_id}/upload",
        files={"file": ("room.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["project_id"] == proj_id
    assert "video_key" in data
    assert "image.jpg" in data["video_key"]

    # 3. Check project details: should be completed and have version seeded
    res = api.get(f"/api/projects/{proj_id}")
    assert res.status_code == 200
    detail = res.json()
    assert detail["status"] == "completed"
    assert detail["active_version_id"] == f"{proj_id}_v0"
    assert len(detail["versions"]) >= 1


def test_upload_png_image(api: TestClient):
    res = api.post("/api/projects", json={"name": "PNG Suite"})
    proj_id = res.json()["id"]

    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 100
    res = api.post(
        f"/api/projects/{proj_id}/upload",
        files={"file": ("suite.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "image.png" in data["video_key"]


def test_upload_webp_image(api: TestClient):
    res = api.post("/api/projects", json={"name": "WebP Suite"})
    proj_id = res.json()["id"]

    webp_bytes = b"RIFF\x18\x00\x00\x00WEBPVP8 " + b"\x00" * 50
    res = api.post(
        f"/api/projects/{proj_id}/upload",
        files={"file": ("suite.webp", io.BytesIO(webp_bytes), "image/webp")},
    )
    assert res.status_code == 200


def test_reject_unsupported_file_extension(api: TestClient):
    res = api.post("/api/projects", json={"name": "Bad File"})
    proj_id = res.json()["id"]

    res = api.post(
        f"/api/projects/{proj_id}/upload",
        files={"file": ("archive.zip", io.BytesIO(b"PK\x03\x04" + b"\x00" * 50), "application/zip")},
    )
    assert res.status_code == 415
    assert "unsupported media type" in res.json()["detail"]


def test_reject_mismatched_magic_bytes(api: TestClient):
    res = api.post("/api/projects", json={"name": "Spoofed JPEG"})
    proj_id = res.json()["id"]

    # Extension is .jpg but magic bytes are random text
    res = api.post(
        f"/api/projects/{proj_id}/upload",
        files={"file": ("fake.jpg", io.BytesIO(b"NOT_A_JPEG_FILE_CONTENT_AT_ALL"), "image/jpeg")},
    )
    assert res.status_code == 415
    assert "invalid media container" in res.json()["detail"]
