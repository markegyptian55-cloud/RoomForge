from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for subdir in ("backend", "worker"):
    path_str = str(REPO_ROOT / subdir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from app.config import Settings, get_settings  # noqa: E402
from app.db import Base, create_engine, sessionmaker  # noqa: E402
from app.models import EditSession, Project, SceneVersion, Task  # noqa: E402
from room3d_worker.pipeline.s04_export import ExportRequest, export_scene  # noqa: E402
from room3d_worker.pipeline.s07_rebuild import GaussianState, save_gaussian_state  # noqa: E402


def _make_plane(
    axis: str, const: float, u_range: tuple[float, float], v_range: tuple[float, float],
    step: float, color: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    us = np.arange(u_range[0], u_range[1], step, dtype=np.float32)
    vs = np.arange(v_range[0], v_range[1], step, dtype=np.float32)
    ug, vg = np.meshgrid(us, vs)
    n = ug.size
    pts = np.zeros((n, 3), dtype=np.float32)
    if axis == "y":
        pts[:, 0], pts[:, 1], pts[:, 2] = ug.ravel(), const, vg.ravel()
    elif axis == "z":
        pts[:, 0], pts[:, 1], pts[:, 2] = ug.ravel(), vg.ravel(), const
    elif axis == "x":
        pts[:, 0], pts[:, 1], pts[:, 2] = const, ug.ravel(), vg.ravel()
    noise = np.random.normal(0, step * 0.05, pts.shape).astype(np.float32)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (n, 1))
    return pts + noise, colors


def _make_box(
    x_range: tuple[float, float], y_range: tuple[float, float], z_range: tuple[float, float],
    step: float, color: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(x_range[0], x_range[1], step, dtype=np.float32)
    ys = np.arange(y_range[0], y_range[1], step, dtype=np.float32)
    zs = np.arange(z_range[0], z_range[1], step, dtype=np.float32)
    xg, yg, zg = np.meshgrid(xs, ys, zs)
    pts = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
    colors = np.tile(np.asarray(color, dtype=np.uint8), (len(pts), 1))
    return pts, colors


def _build_room_geometry(sofa_color: list[int]) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    parts: list[tuple[np.ndarray, np.ndarray]] = [
        _make_plane("y", 0.0, (-3.0, 3.0), (-3.0, 3.0), 0.08, [180, 140, 100]),
        _make_plane("y", 2.6, (-3.0, 3.0), (-3.0, 3.0), 0.12, [230, 230, 235]),
        _make_plane("z", 3.0, (-3.0, 3.0), (0.0, 2.6), 0.08, [160, 180, 200]),
        _make_plane("z", -3.0, (-3.0, 3.0), (0.0, 2.6), 0.08, [220, 215, 210]),
        _make_plane("x", -3.0, (0.0, 2.6), (-3.0, 3.0), 0.08, [45, 105, 115]),
        _make_plane("x", 3.0, (0.0, 2.6), (-3.0, 3.0), 0.08, [170, 185, 205]),
        _make_plane("y", 0.02, (-1.8, 0.8), (0.0, 2.2), 0.06, [210, 200, 180]),
        _make_box((-1.5, 0.5), (0.0, 0.45), (1.4, 2.4), 0.06, sofa_color),
        _make_box((-1.5, 0.5), (0.45, 0.85), (2.1, 2.4), 0.06, sofa_color),
        _make_box((-1.0, 0.0), (0.0, 0.4), (0.3, 0.9), 0.06, [140, 95, 55]),
    ]
    pts = np.vstack([p[0] for p in parts])
    colors = np.vstack([p[1] for p in parts])
    return pts, colors


def _make_camera_poses(num_views: int = 8) -> list[dict[str, Any]]:
    frames = []
    center = np.array([0.0, 1.0, 0.0])
    radius = 2.4
    for idx in range(num_views):
        angle = 2.0 * math.pi * idx / num_views
        eye = np.array([radius * math.sin(angle), 1.3, radius * math.cos(angle)])
        forward = (center - eye) / np.linalg.norm(center - eye)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = right
        c2w[:3, 1] = -true_up
        c2w[:3, 2] = forward
        c2w[:3, 3] = eye
        w2c = np.linalg.inv(c2w)
        frames.append({
            "frame_idx": idx, "orig_video_frame": idx * 10, "timestamp_sec": float(idx * 0.5),
            "image_relpath": f"images/frame_{idx:06d}.jpg", "colmap_image_id": idx + 1,
            "cam_from_world": w2c.tolist(),
        })
    return frames


def _render_synthetic_frames(images_dir: Path, frames: list[dict[str, Any]], sofa_rgb: list[int]) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    w, h = 320, 240
    for frame in frames:
        idx = frame["frame_idx"]
        img = np.full((h, w, 3), 180, dtype=np.uint8)
        img[: h // 2, :] = [215, 220, 230]
        img[h // 2 :, :] = [170, 130, 90]
        sw, sh = 100, 50
        sx, sy = (w - sw) // 2 + int(math.sin(idx) * 20), (h - sh) // 2 + 10
        img[sy : sy + sh, sx : sx + sw] = sofa_rgb
        Image.fromarray(img).save(images_dir / f"frame_{idx:06d}.jpg", quality=92)


def _seed_version(
    storage_dir: Path, project_id: str, version_id: str, parent_id: str | None,
    sofa_color: list[int], frames: list[dict[str, Any]]
) -> Path:
    version_dir = storage_dir / project_id / version_id
    (version_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (version_dir / "sparse" / "0" / name).write_bytes(f"demo-{name}".encode("utf-8"))
    _render_synthetic_frames(version_dir / "images", frames, sofa_color)
    pts, colors = _build_room_geometry(sofa_color)
    opacity = np.full(len(pts), 0.9, dtype=np.float32)
    scales = np.full((len(pts), 3), 0.04, dtype=np.float32)
    export_scene(
        ExportRequest(
            output_dir=version_dir, project_id=project_id, version_id=version_id,
            parent_version_id=parent_id, frames=frames, width=320, height=240,
            intrinsics={"fx": 280.0, "fy": 280.0, "cx": 160.0, "cy": 120.0},
            T_viewer_from_colmap=np.eye(4).tolist(), pose_backend="colmap_global",
        ),
        pts, opacity, scales, colors=colors,
    )
    save_gaussian_state(version_dir, GaussianState(pts, colors, opacity, scales))
    return version_dir


def seed_demo_room(project_id: str = "demo-room") -> str:
    settings = get_settings()
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    storage_dir = Path(settings.local_storage_dir).resolve()
    frames = _make_camera_poses(8)
    _seed_version(storage_dir, project_id, "v1", None, [30, 50, 110], frames)
    _seed_version(storage_dir, project_id, "v2", "v1", [190, 35, 55], frames)
    with session_factory() as session:
        proj = session.get(Project, project_id) or Project(id=project_id)
        proj.name = "Modern Living Room (Demo)"
        proj.status = "completed"
        proj.active_version_id = "v2"
        v1 = session.get(SceneVersion, "v1") or SceneVersion(id="v1", project_id=project_id)
        v1.label, v1.manifest_relpath, v1.spz_relpath = "V1 Base (Navy Sofa)", f"{project_id}/v1/manifest.json", f"{project_id}/v1/scene.spz"
        v2 = session.get(SceneVersion, "v2") or SceneVersion(id="v2", project_id=project_id)
        v2.label, v2.parent_version_id = "V2 Red Velvet Sofa", "v1"
        v2.manifest_relpath, v2.spz_relpath = f"{project_id}/v2/manifest.json", f"{project_id}/v2/scene.spz"
        edit = session.get(EditSession, "edit-demo") or EditSession(id="edit-demo", project_id=project_id)
        edit.base_version_id, edit.target_version_id, edit.ref_frame_idx = "v1", "v2", 0
        edit.prompt, edit.status = "red velvet sofa with soft tufted cushions", "completed"
        task = session.get(Task, "task-demo") or Task(id="task-demo", project_id=project_id)
        task.task_type, task.status, task.progress_pct, task.current_stage = "rebuild_week2", "completed", 100.0, "07_rebuild"
        session.add_all([proj, v1, v2, edit, task])
        session.commit()
    print(f"Successfully seeded demo room '{project_id}' into {settings.database_url}")
    return project_id


if __name__ == "__main__":
    seed_demo_room()
