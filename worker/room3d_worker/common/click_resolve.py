from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from room3d_worker.common.coords import project_world_to_image
from room3d_worker.common.errors import GeometryError
from room3d_worker.common.manifest import SceneManifest


@dataclass(frozen=True)
class ClickResolution:
    ref_frame_idx: int
    u: float
    v: float
    score: float
    method: Literal["projection", "nearest_snap"]


def _manifest_parts(manifest: SceneManifest | dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float], int, int]:
    data = manifest.model_dump() if isinstance(manifest, SceneManifest) else manifest
    width, height = int(data.get("width", 640)), int(data.get("height", 480))
    intr = data.get("intrinsics") or {}
    norm_intr = {
        "fx": float(intr.get("fx", 500.0)),
        "fy": float(intr.get("fy", 500.0)),
        "cx": float(intr.get("cx", width * 0.5)),
        "cy": float(intr.get("cy", height * 0.5)),
    }
    return list(data.get("frames") or []), norm_intr, width, height


def _score_hit_frame(
    frame: dict[str, Any],
    point_world: np.ndarray,
    intrinsics: dict[str, float],
    width: int,
    height: int,
    half_diag: float,
) -> ClickResolution | None:
    pose = frame.get("cam_from_world")
    if pose is None:
        return None
    pose_arr = np.asarray(pose, dtype=np.float64)
    if float((pose_arr[2, :3] @ point_world) + pose_arr[2, 3]) <= 1e-4:
        return None
    u, v, depth = project_world_to_image(point_world, pose_arr, intrinsics)
    if depth <= 1e-4 or not (0.0 <= u < width and 0.0 <= v < height):
        return None
    forward_world = pose_arr[2, :3]
    cam_pos = -(pose_arr[:3, :3].T @ pose_arr[:3, 3])
    view_vec = point_world - cam_pos
    dist = float(np.linalg.norm(view_vec))
    alignment = float(np.dot(forward_world, view_vec / max(dist, 1e-6)))
    if alignment <= 0.0:
        return None
    center_penalty = math.hypot(u - intrinsics["cx"], v - intrinsics["cy"]) / half_diag
    score = alignment * 3.0 - center_penalty * 2.0 - 0.1 * depth
    return ClickResolution(int(frame["frame_idx"]), float(u), float(v), float(score), "projection")


def _score_ray_frame(
    frame: dict[str, Any],
    origin: np.ndarray,
    direction: np.ndarray,
    intrinsics: dict[str, float],
    nearest_frame_idx: int | None,
) -> ClickResolution | None:
    pose = frame.get("cam_from_world")
    if pose is None:
        return None
    pose_arr = np.asarray(pose, dtype=np.float64)
    frame_idx = int(frame["frame_idx"])
    forward = pose_arr[2, :3]
    cam_pos = -(pose_arr[:3, :3].T @ pose_arr[:3, 3])
    vec = cam_pos - origin
    pos_dist = float(np.linalg.norm(vec))
    nearest_bonus = 2.0 if nearest_frame_idx is not None and frame_idx == nearest_frame_idx else 0.0
    dir_align = float(np.dot(forward, direction))
    origin_align = float(np.dot(vec / max(pos_dist, 1e-6), direction)) if pos_dist > 1e-6 else 1.0
    score = nearest_bonus + dir_align * 1.5 + origin_align - 0.1 * pos_dist
    return ClickResolution(frame_idx, intrinsics["cx"], intrinsics["cy"], float(score), "nearest_snap")


def _resolve_by_ray_snap(
    frames: Sequence[dict[str, Any]],
    ray_origin_colmap: Sequence[float],
    ray_dir_colmap: Sequence[float],
    intrinsics: dict[str, float],
    nearest_frame_idx: int | None,
) -> ClickResolution:
    origin = np.asarray(ray_origin_colmap, dtype=np.float64)
    direction = np.asarray(ray_dir_colmap, dtype=np.float64)
    if origin.shape != (3,) or direction.shape != (3,):
        raise GeometryError("ray_origin_colmap and ray_dir_colmap must be 3-vectors")
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

    best: ClickResolution | None = None
    for frame in frames:
        if frame.get("excluded"):
            continue
        candidate = _score_ray_frame(frame, origin, direction, intrinsics, nearest_frame_idx)
        if candidate is not None and (best is None or candidate.score > best.score):
            best = candidate
    if best is None:
        fallback_idx = nearest_frame_idx if nearest_frame_idx is not None else int(frames[0]["frame_idx"])
        return ClickResolution(fallback_idx, intrinsics["cx"], intrinsics["cy"], 0.0, "nearest_snap")
    return best


def resolve_viewer_click(
    manifest: SceneManifest | dict[str, Any],
    world_hit_colmap: Sequence[float] | None = None,
    ray_origin_colmap: Sequence[float] = (0.0, 0.0, 0.0),
    ray_dir_colmap: Sequence[float] = (0.0, 0.0, -1.0),
    nearest_frame_idx: int | None = None,
) -> ClickResolution:
    """Resolve a 3D viewer click into the best reference frame (frame_idx, u, v)."""
    frames, norm_intr, width, height = _manifest_parts(manifest)
    active_frames = [f for f in frames if not f.get("excluded")]
    if not active_frames:
        raise GeometryError("no active non-excluded frames available for click resolution")

    if world_hit_colmap is not None:
        point_world = np.asarray(world_hit_colmap, dtype=np.float64)
        if point_world.shape != (3,) or not np.isfinite(point_world).all():
            raise GeometryError("world_hit_colmap must be a finite 3-vector")
        half_diag = max(math.hypot(width * 0.5, height * 0.5), 1.0)
        best: ClickResolution | None = None
        for frame in active_frames:
            candidate = _score_hit_frame(frame, point_world, norm_intr, width, height, half_diag)
            if candidate is not None and (best is None or candidate.score > best.score):
                best = candidate
        if best is not None:
            return best

    return _resolve_by_ray_snap(active_frames, ray_origin_colmap, ray_dir_colmap, norm_intr, nearest_frame_idx)
