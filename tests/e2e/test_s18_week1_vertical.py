from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

REPO_ROOT = Path(__file__).resolve().parents[2]
for subdir in ("backend", "worker"):
    path_str = str(REPO_ROOT / subdir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from app.config import Settings  # noqa: E402
from app.db import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402
from room3d_worker.client import WorkerClient  # noqa: E402
from room3d_worker.common.manifest import load_manifest  # noqa: E402
from room3d_worker.pipeline.s01_extract import extract_keyframes  # noqa: E402
from room3d_worker.pipeline.s02_poses import BackendOutput, PoseRequest, estimate_poses  # noqa: E402
from room3d_worker.pipeline.s03_train import build_train_config, initialize_scales_knn, radial_outlier_mask  # noqa: E402
from room3d_worker.pipeline.s04_export import ExportRequest, export_scene  # noqa: E402


def _make_synthetic_video(path: Path, frames: int = 15) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 320
        stream.height = 240
        stream.pix_fmt = "yuv420p"
        for idx in range(frames):
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            shift = idx * 14
            cv2.rectangle(img, (20 + shift, 30), (110 + shift, 160), (240, 140, 30), -1)
            cv2.circle(img, (60 + shift, 190), 22, (40, 230, 180), -1)
            cv2.line(img, (0, 230 - idx * 10), (319, idx * 10), (200, 200, 255), 3)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@dataclass
class _SyntheticPoseBackend:
    def estimate(self, request: PoseRequest) -> BackendOutput:
        sparse_dir = request.output_dir / "_raw_sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("cameras.bin", "images.bin", "points3D.bin"):
            (sparse_dir / filename).write_bytes(f"synthetic-{filename}".encode("utf-8"))
        n = len(request.frame_records)
        rng = np.random.default_rng(42)
        points = rng.uniform(-1.5, 1.5, size=(120, 3))
        cameras = np.column_stack((np.linspace(-1.0, 1.0, n), np.zeros(n), np.full(n, 2.5)))
        return BackendOutput(n, 0.35, n, points, cameras, list(request.frame_records), sparse_dir)


def _run_worker_week1(worker: WorkerClient, project_id: str, work_dir: Path, expected_task_id: str) -> str:
    claimed = worker.claim("e2e-worker", ["reconstruct_week1"], interactive_only=False)
    assert claimed is not None
    assert claimed.task_id == expected_task_id

    version_id = f"{project_id}_v0"
    _, storage_version, storage_path = claimed.payload["input_key"].split("/", 2)
    downloaded_video = work_dir / "input.mp4"
    worker.download(project_id, storage_version, storage_path, downloaded_video)

    worker.heartbeat(claimed)
    worker.progress(claimed, "01_extract", 50.0, "Extracting keyframes")
    extracted = extract_keyframes(downloaded_video, work_dir / "frames", max_long_side=640, min_parallax_frac=0.02)
    assert len(extracted.index["frames"]) >= 5

    worker.progress(claimed, "02_poses", 50.0, "Solving poses")
    frame_records = [
        {
            "frame_idx": f["frame_idx"],
            "orig_video_frame": f["orig_video_frame"],
            "timestamp_sec": f["timestamp_sec"],
            "image_relpath": f["image_relpath"],
            "colmap_image_id": f["frame_idx"] + 1,
            "cam_from_world": np.eye(4).tolist(),
        }
        for f in extracted.index["frames"]
    ]
    pose_req = PoseRequest(
        image_dir=work_dir / "frames",
        output_dir=work_dir / "scene",
        project_id=project_id,
        version_id=version_id,
        total_frames=len(frame_records),
        width=320,
        height=240,
        intrinsics={"fx": 280.0, "fy": 280.0, "cx": 160.0, "cy": 120.0},
        backend="colmap_global",
        frame_records=frame_records,  # type: ignore[arg-type]
    )
    pose_manifest, _ = estimate_poses(pose_req, {"colmap_global": _SyntheticPoseBackend()})

    worker.progress(claimed, "03_train", 60.0, "Optimizing Gaussians")
    train_cfg = build_train_config(work_dir / "scene")
    assert train_cfg.test_every == 0
    rng = np.random.default_rng(7)
    pts = rng.uniform(-1.0, 1.0, size=(200, 3)).astype(np.float32)
    pts = pts[radial_outlier_mask(pts)]
    scales = initialize_scales_knn(pts).astype(np.float32)
    scale_vec = np.repeat(scales[:, None], 3, axis=1)
    opacity = np.full(len(pts), 0.9, dtype=np.float32)

    worker.progress(claimed, "04_export", 80.0, "Exporting bundles")
    export_scene(
        ExportRequest(
            output_dir=work_dir / "scene",
            project_id=project_id,
            version_id=version_id,
            frames=[f.model_dump() for f in pose_manifest.frames],
            width=pose_manifest.width,
            height=pose_manifest.height,
            intrinsics=pose_manifest.intrinsics.model_dump(),
            T_viewer_from_colmap=pose_manifest.T_viewer_from_colmap,
            pose_backend=pose_manifest.pose_backend,
        ),
        pts,
        opacity,
        scale_vec,
    )
    assert load_manifest(work_dir / "scene" / "manifest.json").project_id == project_id

    for rel in ("manifest.json", "scene.spz", "scene.ply", "sparse_points.json"):
        worker.upload(project_id, version_id, rel, work_dir / "scene" / rel)
    for frame in extracted.index.get("frames", []):
        img_rel = frame.get("image_relpath")
        if img_rel:
            frame_file = work_dir / "frames" / img_rel
            if frame_file.exists():
                worker.upload(project_id, version_id, img_rel, frame_file)

    worker.complete(
        claimed,
        result_manifest_key=f"{project_id}/{version_id}/manifest.json",
        result_payload={"version_id": version_id, "label": "Base v0"},
    )
    return version_id


def test_week1_vertical_pipeline_backend_to_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "e2e.db"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, poolclass=NullPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    settings = Settings(
        database_url=f"sqlite:///{db_path}",
        local_storage_dir=str(tmp_path / "storage"),
        max_upload_bytes=16 * 1024 * 1024,
        auth_required=False,
    )
    for module in ("app.routers.projects", "app.routers.scenes", "app.security"):
        monkeypatch.setattr(f"{module}.get_settings", lambda: settings)

    def override_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    try:
        with TestClient(app) as client:
            project = client.post("/api/projects", json={"name": "E2E Living Room"}).json()
            project_id = project["id"]
            video_path = tmp_path / "room_15f.mp4"
            _make_synthetic_video(video_path, frames=15)
            upload_res = client.post(
                f"/api/projects/{project_id}/upload",
                files={"file": ("room_15f.mp4", video_path.read_bytes(), "video/mp4")},
            )
            assert upload_res.status_code == 200
            task_id = upload_res.json()["task_id"]

            def _asgi_bridge(request: httpx.Request) -> httpx.Response:
                resp = client.request(
                    method=request.method,
                    url=str(request.url.raw_path.decode("ascii")),
                    headers=dict(request.headers),
                    content=request.content,
                )
                return httpx.Response(resp.status_code, headers=dict(resp.headers), content=resp.content)

            worker = WorkerClient("http://testserver", settings.worker_token, transport=httpx.MockTransport(_asgi_bridge))
            version_id = _run_worker_week1(worker, project_id, tmp_path / "work", task_id)

            detail = client.get(f"/api/projects/{project_id}").json()
            assert detail["active_version_id"] == version_id
            assert detail["tasks"][0]["status"] == "completed"
            assert detail["tasks"][0]["progress_pct"] == 100.0
            assert detail["tasks"][0]["stage"] == "04_export"

            fetched_manifest = client.get(f"/api/projects/{project_id}/versions/{version_id}/manifest")
            assert fetched_manifest.status_code == 200
            assert fetched_manifest.json()["artifacts"]["spz_relpath"] == "scene.spz"
            assert len(json.loads(client.get(f"/api/storage/{project_id}/{version_id}/sparse_points.json").content)) > 0
            assert len(client.get(f"/api/storage/{project_id}/{version_id}/scene.spz").content) > 0
    finally:
        app.dependency_overrides.clear()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_week1_live_browser_renders_reconstructed_room(tmp_path: Path) -> None:
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if not npx:
        pytest.skip("npx not available")

    backend_port = _free_port()
    frontend_port = _free_port()
    db_path = tmp_path / "live.db"
    backend_env = {
        **os.environ,
        "PYTHONPATH": f"{REPO_ROOT / 'backend'}{os.pathsep}{REPO_ROOT / 'worker'}",
        "ROOM3D_DATABASE_URL": f"sqlite:///{db_path}",
        "ROOM3D_LOCAL_STORAGE_DIR": str(tmp_path / "live_storage"),
        "ROOM3D_AUTH_REQUIRED": "false",
    }
    log_path = tmp_path / "backend.log"
    log_file = log_path.open("w", encoding="utf-8")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(backend_port)],
        cwd=str(REPO_ROOT / "backend"),
        env=backend_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        base_url = f"http://127.0.0.1:{backend_port}"
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{base_url}/api/projects", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.25)
        else:
            raise AssertionError("Live FastAPI backend failed to start")

        project = httpx.post(f"{base_url}/api/projects", json={"name": "E2E Live Room"}, timeout=5.0).json()
        project_id = project["id"]
        video_path = tmp_path / "live_15f.mp4"
        _make_synthetic_video(video_path, frames=15)
        with video_path.open("rb") as stream:
            upload_res = httpx.post(
                f"{base_url}/api/projects/{project_id}/upload",
                files={"file": ("live_15f.mp4", stream, "video/mp4")},
                timeout=10.0,
            )
        upload_res.raise_for_status()
        task_id = upload_res.json()["task_id"]

        worker = WorkerClient(base_url, "dev-worker-token")
        _run_worker_week1(worker, project_id, tmp_path / "live_work", task_id)
        worker.close()

        pw_env = {
            **os.environ,
            "ROOM3D_BACKEND_URL": base_url,
            "ROOM3D_FRONTEND_PORT": str(frontend_port),
            "ROOM3D_LIVE_PROJECT_PATH": f"/projects/{project_id}",
        }
        result = subprocess.run(
            [npx, "playwright", "test", "tests/e2e_live.spec.ts"],
            cwd=str(REPO_ROOT / "frontend"),
            env=pw_env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Playwright live E2E failed:\n{result.stdout}\n{result.stderr}"
    finally:
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_proc.kill()
        log_file.close()
