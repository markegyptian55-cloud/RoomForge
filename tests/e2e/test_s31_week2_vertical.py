from __future__ import annotations

import io
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
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
from app.models import Project, SceneVersion, Task  # noqa: E402
from room3d_worker.client import WorkerClient  # noqa: E402
from room3d_worker.common.manifest import SceneManifest, load_manifest  # noqa: E402
from room3d_worker.pipeline.s04_export import ExportRequest, export_scene  # noqa: E402
from room3d_worker.pipeline.s05_track import (  # noqa: E402
    ManualOverride,
    PromptPoint,
    WarmSamSession,
    run_tracking_pipeline,
)
from room3d_worker.pipeline.s06_diffusion import (  # noqa: E402
    DiffusionConfig,
    generate_first_frame_seeds,
    propagate_masked_edits,
)
from room3d_worker.pipeline.s07_rebuild import (  # noqa: E402
    GaussianState,
    RebuildRequest,
    rebuild_scene,
    save_gaussian_state,
)


class _StubInteractiveSam:
    def encode_image(self, image_path: Path) -> tuple[int, int]:
        with Image.open(image_path) as img:
            return img.size[1], img.size[0]

    def predict(self, embedding: tuple[int, int], points: list[PromptPoint]) -> np.ndarray:
        del points
        h, w = embedding
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[16:48, 16:48] = 255
        return mask


class _StubVideoSam:
    def propagate(
        self,
        image_paths: dict[int, Path],
        ref_frame_idx: int,
        seed_mask: np.ndarray,
        inference_state_device: str,
    ) -> dict[int, np.ndarray]:
        del ref_frame_idx, inference_state_device
        return {idx: seed_mask.copy() for idx in sorted(image_paths)}


class _StubInpaint:
    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        seed: int,
        image_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        del prompt, seed, image_reference
        out = image.copy()
        out[mask > 0] = [210, 40, 70]
        return out


def _build_seed_frames(images_dir: Path) -> list[dict[str, Any]]:
    images_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    for idx in range(4):
        img = np.full((64, 64, 3), 60 + idx * 10, dtype=np.uint8)
        img[16:48, 16:48] = [40, 140, 200]
        Image.fromarray(img).save(images_dir / f"frame_{idx:06d}.jpg")
        frames.append(
            {
                "frame_idx": idx,
                "orig_video_frame": idx,
                "timestamp_sec": float(idx),
                "image_relpath": f"images/frame_{idx:06d}.jpg",
                "colmap_image_id": idx + 1,
                "cam_from_world": np.eye(4).tolist(),
            }
        )
    return frames


def _seed_v1_scene(storage_dir: Path, project_id: str, version_id: str) -> Path:
    scene_dir = storage_dir / project_id / version_id
    (scene_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (scene_dir / "sparse" / "0" / name).write_bytes(f"v1-{name}".encode("utf-8"))
    frames = _build_seed_frames(scene_dir / "images")
    pts = np.array([[0.0, 0.0, 2.0], [0.1, 0.1, 2.0], [5.0, 5.0, 10.0]], dtype=np.float32)
    opacity = np.full(3, 0.9, dtype=np.float32)
    scales = np.full((3, 3), 0.05, dtype=np.float32)
    export_scene(
        ExportRequest(
            output_dir=scene_dir,
            project_id=project_id,
            version_id=version_id,
            frames=frames,
            width=64,
            height=64,
            intrinsics={"fx": 64.0, "fy": 64.0, "cx": 32.0, "cy": 32.0},
            T_viewer_from_colmap=np.eye(4).tolist(),
            pose_backend="colmap_global",
        ),
        pts,
        opacity,
        scales,
    )
    save_gaussian_state(scene_dir, GaussianState(pts, np.full((3, 3), 120, dtype=np.uint8), opacity, scales))
    return scene_dir


def _setup_test_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[sessionmaker[Any], Settings, Path]:
    db_path = tmp_path / "e2e_w2.db"
    storage_dir = tmp_path / "storage"
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, poolclass=NullPool)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    settings = Settings(
        database_url=f"sqlite:///{db_path}",
        local_storage_dir=str(storage_dir),
        max_upload_bytes=16 * 1024 * 1024,
        auth_required=False,
    )
    for mod in ("app.routers.projects", "app.routers.scenes", "app.routers.edits"):
        monkeypatch.setattr(f"{mod}.get_settings", lambda: settings)
    with session_factory() as session:
        session.add(Project(id="proj-w2", name="Week 2 Studio", status="completed", active_version_id="v1"))
        session.add(
            SceneVersion(
                id="v1",
                project_id="proj-w2",
                label="V1 Base",
                manifest_relpath="proj-w2/v1/manifest.json",
                spz_relpath="proj-w2/v1/scene.spz",
            )
        )
        session.add_all(
            [Task(id=f"backlog-{i}", project_id="proj-w2", task_type="reconstruct_week1") for i in range(60)]
        )
        session.commit()
    v1_dir = _seed_v1_scene(storage_dir, "proj-w2", "v1")
    return session_factory, settings, v1_dir


def _make_worker(client: TestClient, worker_token: str) -> WorkerClient:
    def _bridge(request: httpx.Request) -> httpx.Response:
        resp = client.request(
            method=request.method,
            url=str(request.url.raw_path.decode("ascii")),
            headers=dict(request.headers),
            content=request.content,
        )
        return httpx.Response(resp.status_code, headers=dict(resp.headers), content=resp.content)

    return WorkerClient("http://testserver", worker_token, transport=httpx.MockTransport(_bridge))


def _run_interactive_seg(client: TestClient, worker: WorkerClient) -> str:
    click_res = client.post(
        "/api/projects/proj-w2/edits/resolve-click",
        json={
            "screenX": 32.0, "screenY": 32.0,
            "worldHitColmap": [0.0, 0.0, 2.0],
            "rayOriginColmap": [0.0, 0.0, 0.0],
            "rayDirColmap": [0.0, 0.0, 1.0],
        },
    )
    assert click_res.status_code == 201 and click_res.json()["method"] == "projection"
    edit_id = str(click_res.json()["edit_id"])
    # Negative + non-starvation check: batch worker must not claim interactive queue, while SAM bypasses backlog.
    assert worker.claim("batch-gpu1", ["reconstruct_week1"], interactive_only=True) is None
    init_claim = worker.claim("sam-gpu1", ["segment_init"], interactive_only=True)
    assert init_claim is not None
    worker.complete(init_claim, {"status": "ok"})
    assert client.post(f"/api/edits/{edit_id}/refine", json={"points": [{"x": 32, "y": 32, "label": 1}]}).status_code == 200
    refine_claim = worker.claim("sam-gpu1", ["segment_refine"], interactive_only=True)
    assert refine_claim is not None
    worker.complete(refine_claim, {"status": "refined"})
    return edit_id


def _run_track_and_overrides(
    client: TestClient,
    worker: WorkerClient,
    edit_id: str,
    manifest: SceneManifest,
    image_paths: dict[int, Path],
    tmp_path: Path,
) -> tuple[dict[int, Path], Path]:
    assert client.post(f"/api/edits/{edit_id}/track").status_code == 200
    assert worker.claim("sam-gpu1", ["segment_init", "segment_refine"], interactive_only=True) is None
    track_claim = worker.claim("batch-gpu1", ["track_propagate"])
    assert track_claim is not None
    tracked_dir = tmp_path / "tracked_masks"
    raw_masks, report = run_tracking_pipeline(
        manifest, image_paths, ref_frame_idx=0, points=[PromptPoint(32, 32, 1)],
        sam_session=WarmSamSession(_StubInteractiveSam()), video_backend=_StubVideoSam(),
        output_dir=tracked_dir, overrides=[ManualOverride(frame_idx=3, skip_frame=True)],
    )
    assert len(raw_masks) == 4 and 3 in report.get("skipped_frames", [])
    for idx, mpath in raw_masks.items():
        worker.upload("proj-w2", "v1", f"edits/{edit_id}/masks/mask_{idx:06d}.png", mpath)
    worker.complete(track_claim, {"edit_id": edit_id, "tracked_frames": 4})
    assert client.put(f"/api/edits/{edit_id}/masks/3", json={"skip_frame": True}).json()["skip_frame"] is True
    return _apply_and_fetch_manual_mask(client, edit_id, raw_masks, tmp_path)


def _apply_and_fetch_manual_mask(
    client: TestClient, edit_id: str, raw_masks: dict[int, Path], tmp_path: Path
) -> tuple[dict[int, Path], Path]:
    custom = np.zeros((64, 64), dtype=np.uint8)
    custom[20:44, 20:44] = 255
    buf = io.BytesIO()
    Image.fromarray(custom).save(buf, format="PNG")
    up_res = client.put(
        f"/api/edits/{edit_id}/masks/2",
        files={"file": ("mask_000002.png", buf.getvalue(), "image/png")},
    )
    assert up_res.status_code == 200 and up_res.json()["status"] == "manual"
    frames_res = client.get(f"/api/edits/{edit_id}/frames")
    assert frames_res.status_code == 200 and frames_res.json()["frames"][3]["skip_frame"] is True
    mask2_key = frames_res.json()["frames"][2]["mask_key"]
    assert isinstance(mask2_key, str) and mask2_key.endswith("000002.png")
    dl = client.get(f"/api/storage/{mask2_key}")
    assert dl.status_code == 200
    fetched = np.asarray(Image.open(io.BytesIO(dl.content)))
    assert np.array_equal(fetched, custom)
    override_dir = tmp_path / "override_masks"
    override_dir.mkdir(parents=True, exist_ok=True)
    effective: dict[int, Path] = {}
    for idx, src in raw_masks.items():
        dest = override_dir / f"{idx:06d}.png"
        dest.write_bytes(dl.content if idx == 2 else src.read_bytes())
        effective[idx] = dest
    return effective, override_dir


def _run_diffusion_and_rebuild(
    client: TestClient,
    worker: WorkerClient,
    edit_id: str,
    v1_dir: Path,
    image_paths: dict[int, Path],
    mask_paths: dict[int, Path],
    masks_dir: Path,
    tmp_path: Path,
) -> str:
    assert client.post(
        f"/api/edits/{edit_id}/preview-seeds",
        json={"prompt": "red velvet chair", "seeds": [11, 22], "model_variant": "flux2_klein_4b"},
    ).status_code == 200
    prev_claim = worker.claim("diff-gpu0", ["diffusion_preview"])
    assert prev_claim is not None
    seeds = generate_first_frame_seeds(
        image_paths[0], mask_paths[0], "red velvet chair", [11, 22], tmp_path / "seeds", _StubInpaint(),
        config=DiffusionConfig(grid_2x2=False),
    )
    assert len(seeds) == 2 and all(Path(s["path"]).exists() for s in seeds)
    worker.complete(prev_claim, {"seeds": [{"seed_id": s["seed_id"]} for s in seeds]})
    apply_res = client.post(
        f"/api/edits/{edit_id}/apply",
        json={"chosen_seed_id": "seed_11", "rebuild_mode": "masked_finetune"},
    )
    assert apply_res.status_code == 200
    target_version = str(apply_res.json()["target_version_id"])
    assert worker.claim("reb-gpu1", ["rebuild_week2"]) is None
    diff_out = _complete_diffusion_apply(worker, image_paths, mask_paths, Path(seeds[0]["path"]), tmp_path)
    _complete_rebuild_v2(worker, v1_dir, diff_out / "edited_frames", masks_dir, target_version, tmp_path)
    return target_version


def _complete_diffusion_apply(
    worker: WorkerClient,
    image_paths: dict[int, Path],
    mask_paths: dict[int, Path],
    ref_seed_path: Path,
    tmp_path: Path,
) -> Path:
    diff_claim = worker.claim("diff-gpu0", ["diffusion_apply"])
    assert diff_claim is not None
    diff_out = tmp_path / "diffusion_out"
    diff_summary = propagate_masked_edits(
        image_paths, mask_paths, 0, ref_seed_path, "red velvet chair", 11, diff_out, _StubInpaint(),
        skip_frames={3}, config=DiffusionConfig(grid_2x2=False),
    )
    assert diff_summary["max_unmasked_diff"] == 0
    worker.complete(diff_claim, {"max_unmasked_diff": 0})
    return diff_out


def _complete_rebuild_v2(
    worker: WorkerClient,
    v1_dir: Path,
    edited_dir: Path,
    masks_dir: Path,
    target_version: str,
    tmp_path: Path,
) -> None:
    reb_claim = worker.claim("reb-gpu1", ["rebuild_week2"])
    assert reb_claim is not None
    v2_dir = tmp_path / "v2_build"

    def _cpu_rebuild(
        req: RebuildRequest, base_manifest: Any, base_state: GaussianState, roi_mask: np.ndarray
    ) -> GaussianState:
        del req, base_manifest
        colors = base_state.colors.copy()
        colors[roi_mask] = [210, 40, 70]
        return GaussianState(base_state.points_xyz.copy(), colors, base_state.opacity.copy(), base_state.scales.copy())

    report = rebuild_scene(
        RebuildRequest(
            base_dir=v1_dir, edited_frames_dir=edited_dir, masks_dir=masks_dir, output_dir=v2_dir,
            project_id="proj-w2", version_id=target_version, parent_version_id="v1",
            mode="masked_finetune", exclude_frames=(3,),
        ),
        backend=_cpu_rebuild,
    )
    assert report["max_unmasked_gaussian_drift"] == 0.0 and report["max_pose_drift"] == 0.0
    for rel in ("manifest.json", "scene.spz", "scene.ply", "sparse_points.json", "gs_state.npz"):
        worker.upload("proj-w2", target_version, rel, v2_dir / rel)
    for sparse_file in ("cameras.bin", "images.bin", "points3D.bin"):
        worker.upload("proj-w2", target_version, f"sparse/0/{sparse_file}", v2_dir / "sparse" / "0" / sparse_file)
    worker.complete(
        reb_claim,
        result_manifest_key=f"proj-w2/{target_version}/manifest.json",
        result_payload={"version_id": target_version, "parent_version_id": "v1", "label": "V2 Red Chair"},
    )


def test_week2_vertical_e2e_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_factory, settings, v1_dir = _setup_test_env(tmp_path, monkeypatch)
    v1_manifest = load_manifest(v1_dir / "manifest.json")
    image_paths = {idx: v1_dir / f"images/frame_{idx:06d}.jpg" for idx in range(4)}

    def override_db():
        with session_factory() as session:
            yield session

    with ExitStack() as stack:
        app.dependency_overrides[get_db] = override_db
        stack.callback(app.dependency_overrides.clear)
        client = stack.enter_context(TestClient(app))
        worker = _make_worker(client, settings.worker_token)
        edit_id = _run_interactive_seg(client, worker)
        mask_paths, masks_dir = _run_track_and_overrides(client, worker, edit_id, v1_manifest, image_paths, tmp_path)
        target_version = _run_diffusion_and_rebuild(
            client, worker, edit_id, v1_dir, image_paths, mask_paths, masks_dir, tmp_path
        )
        detail = client.get("/api/projects/proj-w2").json()
        assert detail["active_version_id"] == target_version
        assert {v["id"] for v in detail["versions"]} == {"v1", target_version}
        v2_manifest = client.get(f"/api/projects/proj-w2/versions/{target_version}/manifest").json()
        assert v2_manifest["parent_version_id"] == "v1" and v2_manifest["frames"][3]["excluded"] is True
        assert len(client.get(f"/api/storage/proj-w2/{target_version}/scene.spz").content) > 0
        assert client.get("/api/projects/proj-w2/edits").json()[0]["status"] == "completed"
