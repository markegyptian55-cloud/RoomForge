from __future__ import annotations

import gzip
import importlib
import json
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from room3d_worker.common.manifest import SceneManifest, save_manifest


@dataclass(frozen=True)
class ExportRequest:
    output_dir: Path
    project_id: str = "project"
    version_id: str = "version"
    parent_version_id: str | None = None
    width: int = 640
    height: int = 480
    intrinsics: dict[str, float] | None = None
    frames: list[dict[str, Any]] | None = None
    test_every: int = 0
    pose_backend: str | None = None
    T_viewer_from_colmap: list[list[float]] | None = None
    max_spz_bytes: int = 30 * 1024 * 1024


def filter_floaters(points_xyz: np.ndarray, opacity: np.ndarray, scales: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    alpha = np.asarray(opacity, dtype=np.float32).reshape(-1)
    size = np.asarray(scales, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(alpha) or len(points) != len(size):
        raise ValueError("points, opacity, and scales must have the same Gaussian count")
    finite = np.isfinite(points).all(1) & np.isfinite(alpha) & np.isfinite(size).all(1)
    radius = np.linalg.norm(size, axis=1)
    return finite & (alpha >= 0.01) & (alpha <= 1.0) & (radius >= 1e-5) & (radius <= 10.0)


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertex_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertices = np.empty(len(points), dtype=vertex_dtype)
    vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertices["red"], vertices["green"], vertices["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode()
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(vertices.tobytes())


def _write_fallback_spz(path: Path, points: np.ndarray, colors: np.ndarray, opacity: np.ndarray) -> None:
    payload = b"NGSP" + struct.pack("<II", 2, len(points)) + points.astype("<f4").tobytes()
    payload += colors.astype("u1").tobytes() + opacity.astype("<f4").tobytes()
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as handle:
            handle.write(payload)


def _write_spz(path: Path, points: np.ndarray, colors: np.ndarray, opacity: np.ndarray) -> None:
    try:
        spz = importlib.import_module("spz")
        cloud = spz.GaussianCloud(positions=points, colors=colors, opacities=opacity)
        spz.save_spz(str(path), cloud)
        return
    except (ImportError, AttributeError, TypeError, ValueError, OSError) as exc:
        warnings.warn(f"Native spz writer unavailable ({exc}); writing gzipped fallback .spz", RuntimeWarning, stacklevel=2)
        _write_fallback_spz(path, points, colors, opacity)


def _default_frames() -> list[dict[str, Any]]:
    return [{"frame_idx": 0, "orig_video_frame": 0, "timestamp_sec": 0.0,
             "image_relpath": "images/0.png", "colmap_image_id": 1,
             "cam_from_world": np.eye(4).tolist()}]


def export_scene(request: ExportRequest, points_xyz: np.ndarray, opacity: np.ndarray,
                 scales: np.ndarray, colors: np.ndarray | None = None) -> SceneManifest:
    out = request.output_dir
    out.mkdir(parents=True, exist_ok=True)
    keep = filter_floaters(points_xyz, opacity, scales)
    points, alpha = np.asarray(points_xyz, np.float32)[keep], np.asarray(opacity, np.float32).reshape(-1)[keep]
    rgb = np.asarray(colors if colors is not None else np.full((len(points_xyz), 3), 200), np.uint8)[keep]
    _write_ply(out / "scene.ply", points, rgb)
    _write_spz(out / "scene.spz", points, rgb, alpha)
    if (out / "scene.spz").stat().st_size >= request.max_spz_bytes:
        warnings.warn("scene.spz exceeds the 30 MB viewer budget", RuntimeWarning, stacklevel=2)
    step = max(1, len(points) // 100_000)
    sparse = [
        {
            "position": [float(v) for v in pt],
            "x": float(pt[0]),
            "y": float(pt[1]),
            "z": float(pt[2]),
            "color": [int(v) for v in clr],
        }
        for pt, clr in zip(points[::step], rgb[::step])
    ]
    (out / "sparse_points.json").write_text(json.dumps(sparse, separators=(",", ":")) + "\n", encoding="utf-8")
    intrinsic = request.intrinsics or {"fx": 500.0, "fy": 500.0, "cx": request.width / 2, "cy": request.height / 2}
    manifest = SceneManifest.model_validate({"schema_version": "1.0.0", "project_id": request.project_id,
        "version_id": request.version_id, "parent_version_id": request.parent_version_id,
        "coord_convention": "opencv_colmap",
        "T_viewer_from_colmap": request.T_viewer_from_colmap or np.eye(4).tolist(), "camera_model": "PINHOLE",
        "width": request.width, "height": request.height, "intrinsics": intrinsic, "test_every": request.test_every,
        "pose_backend": request.pose_backend, "frames": request.frames or _default_frames(),
        "artifacts": {"sparse_dir": "sparse/0", "ply_relpath": "scene.ply", "spz_relpath": "scene.spz",
                       "sparse_points_relpath": "sparse_points.json", "ckpt_relpath": "gs_ckpt.pt"},
        "metrics": {"gaussian_count": int(len(points)), "spz_bytes": (out / "scene.spz").stat().st_size}})
    save_manifest(out / "manifest.json", manifest)
    return manifest
