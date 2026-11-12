from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app  # noqa: E402
from app.security import hash_password  # noqa: E402


@pytest.fixture()
def isolated_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    storage_dir = tmp_path / "storage"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("JWT_SECRET", "unit-test-secret-with-enough-length")
    monkeypatch.setenv("WORKER_TOKEN", "test-worker-secret")

    import app.config as cfg
    import app.db as db_mod
    import app.models as models_mod

    cfg.get_settings.cache_clear()
    db_mod.settings = cfg.get_settings()
    engine = db_mod.create_engine(
        db_mod.settings.database_url,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
    db_mod.engine = engine
    db_mod.SessionLocal.configure(bind=engine)
    models_mod.Base.metadata.drop_all(engine)
    models_mod.Base.metadata.create_all(engine)

    with db_mod.SessionLocal() as session:
        user_a = models_mod.User(email="owner@test.dev", password_hash=hash_password("pw-owner-123"))
        user_b = models_mod.User(email="other@test.dev", password_hash=hash_password("pw-other-123"))
        session.add_all([user_a, user_b])
        session.flush()
        proj = models_mod.Project(id="proj-s21", owner_id=user_a.id, name="Living Room", status="completed", active_version_id="v1")
        manifest_rel = "proj-s21/v1/manifest.json"
        manifest_file = storage_dir / manifest_rel
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "1.0.0",
            "project_id": "proj-s21",
            "version_id": "v1",
            "coord_convention": "opencv_colmap",
            "T_viewer_from_colmap": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            "camera_model": "PINHOLE",
            "width": 160,
            "height": 128,
            "intrinsics": {"fx": 160.0, "fy": 160.0, "cx": 80.0, "cy": 64.0},
            "test_every": 0,
            "frames": [
                {"frame_idx": 0, "orig_video_frame": 0, "timestamp_sec": 0.0, "image_relpath": "images/0.png",
                 "colmap_image_id": 1, "cam_from_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
                {"frame_idx": 1, "orig_video_frame": 1, "timestamp_sec": 0.5, "image_relpath": "images/1.png",
                 "colmap_image_id": 2, "cam_from_world": [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 10], [0, 0, 0, 1]]},
            ],
            "artifacts": {"sparse_dir": "sparse/0", "spz_relpath": "scene.spz", "sparse_points_relpath": "sparse_points.json"},
        }
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        session.add(proj)
        session.add(models_mod.SceneVersion(id="v1", project_id="proj-s21", label="v1", manifest_relpath=manifest_rel))
        session.commit()
    return TestClient(app)


def _auth(client: TestClient, email: str, password: str) -> dict[str, str]:
    resp = client.post("/api/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _png_bytes(width: int, height: int, mode: str = "L", color: int | tuple[int, int, int] = 0) -> bytes:
    buf = BytesIO()
    if mode == "L":
        arr = np.full((height, width), int(color), dtype=np.uint8)
        arr[: height // 2, : width // 2] = 255
        img = Image.fromarray(arr)
    else:
        img = Image.new(mode, (width, height), color)
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_edit_ownership_isolation_and_strict_mask_validation(isolated_client: TestClient) -> None:
    owner_h = _auth(isolated_client, "owner@test.dev", "pw-owner-123")
    other_h = _auth(isolated_client, "other@test.dev", "pw-other-123")

    res = isolated_client.post(
        "/api/projects/proj-s21/edits/resolve-click",
        json={"screenX": 80, "screenY": 64, "worldHitColmap": [0.0, 0.0, 4.0], "nearestFrameIdx": 1},
        headers=owner_h,
    )
    assert res.status_code == 201
    data = res.json()
    assert data["ref_frame_idx"] == 0
    assert data["method"] == "projection"
    edit_id = data["edit_id"]

    # Other user cannot touch or query owner's edit session.
    assert isolated_client.post(f"/api/edits/{edit_id}/refine", json={"points": [{"x": 10, "y": 10, "label": 1}]}, headers=other_h).status_code == 404
    assert isolated_client.get(f"/api/edits/{edit_id}/frames", headers=other_h).status_code == 404

    # Non-L (RGB) mask, gray pixel value (128), and wrong dimensions must fail with 422.
    bad_rgb = _png_bytes(160, 128, mode="RGB", color=(255, 255, 255))
    assert isolated_client.put(f"/api/edits/{edit_id}/masks/0", files={"file": ("mask.png", bad_rgb, "image/png")}, headers=owner_h).status_code == 422
    bad_gray = _png_bytes(160, 128, mode="L", color=128)
    assert isolated_client.put(f"/api/edits/{edit_id}/masks/0", files={"file": ("mask.png", bad_gray, "image/png")}, headers=owner_h).status_code == 422
    bad_dim = _png_bytes(64, 64, mode="L", color=0)
    assert isolated_client.put(f"/api/edits/{edit_id}/masks/0", files={"file": ("mask.png", bad_dim, "image/png")}, headers=owner_h).status_code == 422

    # Valid mod-16 1-channel uint8 {0, 255} PNG succeeds.
    good_png = _png_bytes(160, 128, mode="L", color=0)
    ok = isolated_client.put(f"/api/edits/{edit_id}/masks/0", files={"file": ("mask.png", good_png, "image/png")}, headers=owner_h)
    assert ok.status_code == 200
    assert ok.json()["status"] == "manual"


def test_apply_barrier_completion_failure_cascade_and_idempotency(isolated_client: TestClient) -> None:
    owner_h = _auth(isolated_client, "owner@test.dev", "pw-owner-123")
    worker_h = {"X-Worker-Token": "test-worker-secret"}

    res = isolated_client.post("/api/projects/proj-s21/edits/resolve-click", json={"worldHitColmap": [0.0, 0.0, 4.0]}, headers=owner_h)
    edit_id = res.json()["edit_id"]

    # Apply before prompt is set must fail cleanly with 422.
    assert isolated_client.post(f"/api/edits/{edit_id}/apply", json={"chosen_seed_id": "seed_1"}, headers=owner_h).status_code == 422

    # Drain initial segment_init task cleanly.
    c0 = isolated_client.post("/api/tasks/claim", json={"worker_id": "w1", "capabilities": [], "interactive_only": False}, headers=worker_h).json()
    isolated_client.post(f"/api/tasks/{c0['task_id']}/complete", json={"attempt_id": c0["attempt_id"], "result_payload": {"ok": True}}, headers=worker_h)
    # Late progress after completion must be rejected (409 Conflict).
    late = isolated_client.post(f"/api/tasks/{c0['task_id']}/progress", json={"attempt_id": c0["attempt_id"], "stage": "05_track", "progress_pct": 50.0}, headers=worker_h)
    assert late.status_code == 409

    isolated_client.post(f"/api/edits/{edit_id}/preview-seeds", json={"prompt": "blue velvet sofa", "seeds": [11, 22]}, headers=owner_h)
    c1 = isolated_client.post("/api/tasks/claim", json={"worker_id": "w1", "capabilities": [], "interactive_only": False}, headers=worker_h).json()
    isolated_client.post(f"/api/tasks/{c1['task_id']}/complete", json={"attempt_id": c1["attempt_id"], "result_payload": {"seeds": []}}, headers=worker_h)

    applied = isolated_client.post(f"/api/edits/{edit_id}/apply", json={"chosen_seed_id": "seed_11"}, headers=owner_h)
    assert applied.status_code == 200
    rebuild_id = applied.json()["id"]
    diff_id = applied.json()["diffusion_task_id"]

    # Duplicate apply while rebuild is active returns 409 Conflict.
    assert isolated_client.post(f"/api/edits/{edit_id}/apply", json={"chosen_seed_id": "seed_22"}, headers=owner_h).status_code == 409

    # Dependency barrier prevents claiming rebuild_week2 before diffusion_apply finishes.
    c2 = isolated_client.post("/api/tasks/claim", json={"worker_id": "w1", "capabilities": [], "interactive_only": False}, headers=worker_h).json()
    assert c2["task_id"] == diff_id

    # Failing diffusion_apply cleanly fails dependent rebuild_week2 and EditSession instead of creating zombies.
    isolated_client.post(
        f"/api/tasks/{diff_id}/fail",
        json={"attempt_id": c2["attempt_id"], "error_code": "OOM", "error_message": "gpu error", "retryable": False},
        headers=worker_h,
    )
    claim_after_fail = isolated_client.post("/api/tasks/claim", json={"worker_id": "w1", "capabilities": [], "interactive_only": False}, headers=worker_h)
    assert claim_after_fail.status_code == 204

    proj_state = isolated_client.get("/api/projects/proj-s21", headers=owner_h).json()
    rebuild_task = next(t for t in proj_state["tasks"] if t["id"] == rebuild_id)
    assert rebuild_task["status"] == "failed"
    edits = isolated_client.get("/api/projects/proj-s21/edits", headers=owner_h).json()
    assert edits[0]["status"] == "failed"


def test_malformed_payload_is_isolated_and_does_not_poison_queue(isolated_client: TestClient) -> None:
    import app.db as db_mod
    from app.models import Project, Task

    with db_mod.SessionLocal() as session:
        session.add(Project(id="proj-bad", name="Malformed Payload Test", status="queued"))
        session.add(Task(id="t-bad-json", project_id="proj-bad", task_type="segment_init", payload_json="{not-json"))
        session.add(
            Task(
                id="t-good",
                project_id="proj-bad",
                task_type="segment_init",
                payload_json=json.dumps({"required_capabilities": None}),
            )
        )
        session.commit()

    worker_h = {"X-Worker-Token": "test-worker-secret"}
    res = isolated_client.post(
        "/api/tasks/claim",
        json={"worker_id": "sam-1", "capabilities": ["segment_init"], "interactive_only": True},
        headers=worker_h,
    )
    assert res.status_code == 200
    assert res.json()["task_id"] == "t-good"

    with db_mod.SessionLocal() as session:
        bad = session.get(Task, "t-bad-json")
        assert bad is not None and bad.status == "failed"

