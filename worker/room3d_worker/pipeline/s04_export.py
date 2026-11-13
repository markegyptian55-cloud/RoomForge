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
    if size.ndim == 1:
        size = size[:, None]
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(alpha) or len(points) != len(size):
        raise ValueError("points, opacity, and scales must have the same Gaussian count")
    finite = np.isfinite(points).all(1) & np.isfinite(alpha) & np.isfinite(size).all(1)
    radius = np.linalg.norm(size, axis=1)
    return finite & (alpha >= 0.01) & (alpha <= 1.0) & (radius >= 1e-5) & (radius <= 10.0)


def _write_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    opacity: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    rotations: np.ndarray | None = None,
) -> None:
    n = len(points)
    if opacity is not None and scales is not None and n > 0:
        alpha = np.asarray(opacity, dtype=np.float32).reshape(-1)
        sc = np.asarray(scales, dtype=np.float32)
        if sc.ndim == 1:
            sc = np.repeat(sc[:, None], 3, axis=1)
        elif sc.shape[1] == 1:
            sc = np.repeat(sc, 3, axis=1)

        if rotations is not None and len(rotations) == n:
            rot = np.asarray(rotations, dtype=np.float32)
        else:
            rot = np.zeros((n, 4), dtype=np.float32)
            rot[:, 0] = 1.0

        sh_dc = (colors.astype(np.float32) / 255.0 - 0.5) / 0.28209479177387814
        log_scales = np.log(np.maximum(sc, 1e-7))
        clipped_alpha = np.clip(alpha, 1e-4, 1.0 - 1e-4)
        logit_opacity = np.log(clipped_alpha / (1.0 - clipped_alpha))

        vertex_dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
            ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        vertices = np.empty(n, dtype=vertex_dtype)
        vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
        vertices["nx"], vertices["ny"], vertices["nz"] = 0.0, 0.0, 0.0
        vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = sh_dc[:, 0], sh_dc[:, 1], sh_dc[:, 2]
        vertices["opacity"] = logit_opacity
        vertices["scale_0"], vertices["scale_1"], vertices["scale_2"] = log_scales[:, 0], log_scales[:, 1], log_scales[:, 2]
        vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"] = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
        vertices["red"], vertices["green"], vertices["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

        header = (
            f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
            "property float opacity\n"
            "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
            "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        ).encode()
        with path.open("wb") as handle:
            handle.write(header)
            handle.write(vertices.tobytes())
    else:
        vertex_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        vertices = np.empty(n, dtype=vertex_dtype)
        vertices["x"], vertices["y"], vertices["z"] = points[:, 0], points[:, 1], points[:, 2]
        vertices["red"], vertices["green"], vertices["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
        header = (
            f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        ).encode()
        with path.open("wb") as handle:
            handle.write(header)
            handle.write(vertices.tobytes())


def _write_fallback_spz(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray | None = None,
    rotations: np.ndarray | None = None,
) -> None:
    n = len(points)
    payload = b"NGSP" + struct.pack("<II", 3, n) + points.astype("<f4").tobytes()
    payload += colors.astype("u1").tobytes() + opacity.astype("<f4").tobytes()
    if scales is not None:
        sc = np.asarray(scales, dtype=np.float32)
        if sc.ndim == 1:
            sc = np.repeat(sc[:, None], 3, axis=1)
        payload += sc.astype("<f4").tobytes()
    if rotations is not None and len(rotations) == n:
        payload += np.asarray(rotations, dtype=np.float32).tobytes()
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as handle:
            handle.write(payload)


def _write_spz(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray | None = None,
    rotations: np.ndarray | None = None,
) -> None:
    try:
        spz = importlib.import_module("spz")
        kwargs: dict[str, Any] = {"positions": points, "colors": colors, "opacities": opacity}
        if scales is not None:
            kwargs["scales"] = scales
        if rotations is not None:
            kwargs["rotations"] = rotations
        cloud = spz.GaussianCloud(**kwargs)
        spz.save_spz(str(path), cloud)
        return
    except (ImportError, AttributeError, TypeError, ValueError, OSError) as exc:
        warnings.warn(f"Native spz writer unavailable ({exc}); writing gzipped fallback .spz", RuntimeWarning, stacklevel=2)
        _write_fallback_spz(path, points, colors, opacity, scales, rotations)


def _default_frames() -> list[dict[str, Any]]:
    return [{"frame_idx": 0, "orig_video_frame": 0, "timestamp_sec": 0.0,
             "image_relpath": "images/0.png", "colmap_image_id": 1,
             "cam_from_world": np.eye(4).tolist()}]


def export_scene(
    request: ExportRequest,
    points_xyz: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray,
    colors: np.ndarray | None = None,
    rotations: np.ndarray | None = None,
) -> SceneManifest:
    out = request.output_dir
    out.mkdir(parents=True, exist_ok=True)
    keep = filter_floaters(points_xyz, opacity, scales)
    points, alpha = np.asarray(points_xyz, np.float32)[keep], np.asarray(opacity, np.float32).reshape(-1)[keep]
    sc = np.asarray(scales, np.float32)[keep]
    rgb = np.asarray(colors if colors is not None else np.full((len(points_xyz), 3), 200), np.uint8)[keep]
    rot = np.asarray(rotations, np.float32)[keep] if rotations is not None else None
    _write_ply(out / "scene.ply", points, rgb, alpha, sc, rot)
    _write_spz(out / "scene.spz", points, rgb, alpha, sc, rot)
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
