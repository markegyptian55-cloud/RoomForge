from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from room3d_worker.common.errors import GeometryError

FloatArray = NDArray[np.float64]


def _array(value: NDArray[Any], name: str, shape_tail: tuple[int, ...]) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != len(shape_tail) + 1 or result.shape[1:] != shape_tail:
        raise GeometryError(f"{name} must have shape (N, {', '.join(map(str, shape_tail))})")
    if not np.isfinite(result).all():
        raise GeometryError(f"{name} contains non-finite values")
    return result


def _rotation_between(source: FloatArray, target: FloatArray) -> FloatArray:
    source, target = source / np.linalg.norm(source), target / np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = float(np.dot(source, target))
    if np.linalg.norm(cross) < 1e-10:
        if cosine > 0:
            return np.eye(3)
        axis = np.cross(source, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-10:
            axis = np.cross(source, np.array([0.0, 0.0, 1.0]))
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / np.dot(cross, cross))


def _floor_normal(points: FloatArray) -> FloatArray:
    if len(points) < 3:
        raise GeometryError("At least three points are required to estimate room up")
    rng = np.random.default_rng(0)
    best: tuple[int, FloatArray, float] | None = None
    tolerance = max(float(np.linalg.norm(np.ptp(points, axis=0))) * 0.01, 1e-6)
    for indices in rng.integers(0, len(points), size=(min(256, len(points) * 8), 3)):
        a, b, c = points[indices]
        normal = np.cross(b - a, c - a)
        magnitude = float(np.linalg.norm(normal))
        if magnitude < 1e-10:
            continue
        normal /= magnitude
        offset = float(np.dot(normal, a))
        inliers = int(np.count_nonzero(np.abs(points @ normal - offset) <= tolerance))
        if best is None or inliers > best[0]:
            best = inliers, normal, offset
    if best is not None and best[0] >= max(3, int(len(points) * 0.15)):
        return best[1]
    centered = points - points.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return vectors[-1]


def align_gravity_up(points_xyz: np.ndarray, cam_positions_xyz: np.ndarray) -> np.ndarray:
    points = _array(points_xyz, "points_xyz", (3,))
    cameras = _array(cam_positions_xyz, "cam_positions_xyz", (3,))
    normal = _floor_normal(points)
    if len(cameras) and float(np.dot(normal, cameras.mean(axis=0) - points.mean(axis=0))) < 0:
        normal = -normal
    rotation = _rotation_between(normal, np.array([0.0, 1.0, 0.0]))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ points.mean(axis=0)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8) or np.linalg.det(rotation) < 0.999999:
        raise GeometryError("Gravity alignment did not produce a proper rigid rotation")
    return transform


def _intrinsic(intrinsics: dict[str, float]) -> tuple[float, float, float, float]:
    try:
        fx, fy, cx, cy = (float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy"))
    except (KeyError, TypeError, ValueError) as exc:
        raise GeometryError(f"Invalid PINHOLE intrinsics: {exc}") from exc
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise GeometryError("Focal lengths must be positive and all intrinsics finite")
    return fx, fy, cx, cy


def _pose(matrix: np.ndarray) -> FloatArray:
    pose = np.asarray(matrix, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise GeometryError("cam_from_world must be a finite 4x4 matrix")
    return pose


def project_world_to_image(point_world: np.ndarray, cam_from_world: np.ndarray, intrinsics: dict[str, float]) -> tuple[float, float, float]:
    point = np.asarray(point_world, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise GeometryError("point_world must be a finite 3-vector")
    camera = _pose(cam_from_world) @ np.append(point, 1.0)
    depth = float(camera[2])
    if depth <= 1e-6:
        raise GeometryError(f"Point is behind camera or too near (depth={depth})")
    fx, fy, cx, cy = _intrinsic(intrinsics)
    return fx * camera[0] / depth + cx, fy * camera[1] / depth + cy, depth


def unproject_pixel_to_ray(u: float, v: float, cam_from_world: np.ndarray, intrinsics: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    pose = _pose(cam_from_world)
    fx, fy, cx, cy = _intrinsic(intrinsics)
    direction_camera = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    direction_world = pose[:3, :3].T @ direction_camera
    direction_world /= np.linalg.norm(direction_world)
    origin_world = -pose[:3, :3].T @ pose[:3, 3]
    return origin_world, direction_world


def mod16_crop_or_pad_params(width: int, height: int, cx: float, cy: float, fx: float, fy: float, multiple: int = 16) -> tuple[int, int, float, float, float, float, tuple[int, int, int, int]]:
    if min(width, height, multiple) <= 0 or fx <= 0 or fy <= 0:
        raise GeometryError("Dimensions, multiple, and focal lengths must be positive")
    new_w, new_h = width - width % multiple, height - height % multiple
    if new_w == 0 or new_h == 0:
        raise GeometryError("Image dimensions are smaller than the requested multiple")
    x0, y0 = (width - new_w) // 2, (height - new_h) // 2
    return new_w, new_h, fx, fy, cx - x0, cy - y0, (x0, y0, x0 + new_w, y0 + new_h)
