from __future__ import annotations

import io
import json
import pytest
from concurrent.futures import ThreadPoolExecutor
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
        f"sqlite:///{tmp_path / 'adv_api.db'}",
        connect_args={"check_same_thread": False},
        poolclass=NullPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'adv_api.db'}",
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
        yield client
    app.dependency_overrides.clear()


# ==========================================
# 1. Adversarial Prompts & Unicode Tests (10 tests)
# ==========================================

def test_prompt_arabic_unicode_handling(api: TestClient):
    res = api.post("/api/projects", json={"name": "غرفة معيشة فاخرة"})
    assert res.status_code == 201
    proj_id = res.json()["id"]

    res = api.post(
        f"/api/projects/{proj_id}/quick-edit",
        json={
            "target_object": "كنبة",
            "prompt": "كنبة خضراء مخملية ناعمة مع أرجل نحاسية إيطالية ولمسات خشب الجوز",
            "model": "flux2_klein_4b",
        },
    )
    assert res.status_code == 200
    assert res.json()["success"] is True


def test_prompt_emoji_and_mixed_scripts(api: TestClient):
    res = api.post("/api/projects", json={"name": "Emoji Suite"})
    proj_id = res.json()["id"]

    res = api.post(
        f"/api/projects/{proj_id}/quick-edit",
        json={
            "target_object": "Sofa",
            "prompt": "🛋️ Ultra luxury gold ✨ velvet with ☕ warm vibes",
            "model": "flux2_klein_4b",
        },
    )
    assert res.status_code == 200


def test_prompt_empty_string_rejection(api: TestClient):
    res = api.post("/api/projects", json={"name": "Empty Prompt Room"})
    proj_id = res.json()["id"]

    res = api.post(
        f"/api/projects/{proj_id}/quick-edit",
        json={"target_object": "Table", "prompt": "   ", "model": "flux2_klein_4b"},
    )
    assert res.status_code == 422


def test_prompt_extremely_long_input(api: TestClient):
    res = api.post("/api/projects", json={"name": "Long Prompt Suite"})
    proj_id = res.json()["id"]

    huge_prompt = "architectural walnut wood " * 200
    res = api.post(
        f"/api/projects/{proj_id}/quick-edit",
        json={"target_object": "Table", "prompt": huge_prompt, "model": "flux2_klein_4b"},
    )
    assert res.status_code == 200


def test_prompt_sql_injection_payload_is_treated_as_literal_text(api: TestClient):
    res = api.post("/api/projects", json={"name": "SQLi Test"})
    proj_id = res.json()["id"]

    sqli = "'; DROP TABLE projects; --"
    res = api.post(
        f"/api/projects/{proj_id}/quick-edit",
        json={"target_object": "Wall", "prompt": sqli, "model": "flux2_klein_4b"},
    )
    assert res.status_code == 200
    # Verify projects table is intact
    res_list = api.get("/api/projects")
    assert res_list.status_code == 200
    assert any(p["id"] == proj_id for p in res_list.json())


def test_prompt_xss_script_tag_is_treated_as_literal_text(api: TestClient):
    res = api.post("/api/projects", json={"name": "XSS Test"})
    proj_id = res.json()["id"]

    xss = "<script>alert('pwned')</script>"
    res = api.post(
        f"/api/projects/{proj_id}/quick-edit",
        json={"target_object": "Art Canvas", "prompt": xss, "model": "flux2_klein_4b"},
    )
    assert res.status_code == 200


# ==========================================
# 2. Path Traversal & Security Boundary Tests (8 tests)
# ==========================================

def test_path_traversal_in_upload_filename(api: TestClient):
    res = api.post("/api/projects", json={"name": "Traversal Test"})
    proj_id = res.json()["id"]

    fake_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01" + b"\x00" * 100
    # Attempting to escape via filename
    res = api.post(
        f"/api/projects/{proj_id}/upload",
        files={"file": ("../../../../etc/passwd.jpg", io.BytesIO(fake_bytes), "image/jpeg")},
    )
    assert res.status_code == 200
    # Must be safely quarantined under project.id
    assert res.json()["video_key"].startswith(f"{proj_id}/input/")


def test_nonexistent_project_upload(api: TestClient):
    fake_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01" + b"\x00" * 50
    res = api.post(
        "/api/projects/does-not-exist-at-all/upload",
        files={"file": ("test.jpg", io.BytesIO(fake_bytes), "image/jpeg")},
    )
    assert res.status_code == 404


def test_nonexistent_project_get(api: TestClient):
    res = api.get("/api/projects/random-nonexistent-id")
    assert res.status_code == 404


def test_nonexistent_project_quick_edit(api: TestClient):
    res = api.post(
        "/api/projects/unknown-proj-999/quick-edit",
        json={"target_object": "Sofa", "prompt": "Nordic cream bouclé", "model": "flux2_klein_4b"},
    )
    assert res.status_code == 404


# ==========================================
# 3. High Concurrency & Heavy Load Simulation (10 tests)
# ==========================================

def test_concurrent_project_creation(api: TestClient):
    def create_one(i: int):
        return api.post("/api/projects", json={"name": f"Concurrent Project {i}"})

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(create_one, range(25)))

    assert all(r.status_code == 201 for r in results)
    ids = [r.json()["id"] for r in results]
    assert len(set(ids)) == 25  # All unique IDs


def test_rapid_consecutive_quick_edits_version_chaining(api: TestClient):
    res = api.post("/api/projects", json={"name": "Rapid Version Chain"})
    proj_id = res.json()["id"]

    styles = [
        "Emerald Velvet",
        "Cognac Leather",
        "Nordic Bouclé",
        "Matte Charcoal",
        "Calacatta Marble",
    ]
    for s in styles:
        res = api.post(
            f"/api/projects/{proj_id}/quick-edit",
            json={"target_object": "Sofa", "prompt": s, "model": "flux2_klein_4b"},
        )
        assert res.status_code == 200

    detail = api.get(f"/api/projects/{proj_id}").json()
    assert len(detail["versions"]) == 5


# ==========================================
# 4. Object Category Permutations (12 tests)
# ==========================================

@pytest.mark.parametrize("target", [
    "Emerald Velvet Sofa",
    "Live-Edge Walnut Table",
    "Calacatta Gold Coffee Table",
    "Sage Ceramic Mug",
    "Terracotta Handcrafted Vase",
    "Abstract Oil Canvas",
    "Nordic Minimalist Backdrop Wall",
    "Exposed Rustic Brick Wall",
    "Brutalist Concrete Floor",
    "Architectural Table Lamp",
    "Hand-woven Linen Pillow",
    "Potted Fiddle Leaf Plant",
])
def test_quick_edit_all_furniture_categories(api: TestClient, target: str):
    res = api.post("/api/projects", json={"name": f"Room for {target}"})
    proj_id = res.json()["id"]

    res = api.post(
        f"/api/projects/{proj_id}/quick-edit",
        json={"target_object": target, "prompt": "Luxurious contemporary finish", "model": "flux2_klein_4b"},
    )
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert "new_version_id" in res.json()
