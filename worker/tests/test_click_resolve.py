from __future__ import annotations

import numpy as np
import pytest

from room3d_worker.common.click_resolve import resolve_viewer_click
from room3d_worker.common.errors import GeometryError
from room3d_worker.common.manifest import SceneManifest


def _manifest(*, excluded_first: bool = False) -> SceneManifest:
    pose_front = np.eye(4).tolist()
    pose_right = np.eye(4)
    pose_right[:3, 3] = [-1.5, 0.0, 0.0]
    pose_behind = np.eye(4)
    pose_behind[:3, 3] = [0.0, 0.0, -10.0]
    return SceneManifest.model_validate(
        {
            "schema_version": "1.0.0",
            "project_id": "proj-1",
            "version_id": "v0_base",
            "coord_convention": "opencv_colmap",
            "T_viewer_from_colmap": np.eye(4).tolist(),
            "camera_model": "PINHOLE",
            "width": 640,
            "height": 480,
            "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
            "test_every": 0,
            "frames": [
                {
                    "frame_idx": 0,
                    "orig_video_frame": 0,
                    "timestamp_sec": 0.0,
                    "image_relpath": "images/00000.jpg",
                    "colmap_image_id": 1,
                    "cam_from_world": pose_front,
                    "excluded": excluded_first,
                },
                {
                    "frame_idx": 1,
                    "orig_video_frame": 2,
                    "timestamp_sec": 0.2,
                    "image_relpath": "images/00001.jpg",
                    "colmap_image_id": 2,
                    "cam_from_world": pose_right.tolist(),
                    "excluded": False,
                },
                {
                    "frame_idx": 2,
                    "orig_video_frame": 4,
                    "timestamp_sec": 0.4,
                    "image_relpath": "images/00002.jpg",
                    "colmap_image_id": 3,
                    "cam_from_world": pose_behind.tolist(),
                    "excluded": False,
                },
            ],
            "artifacts": {
                "sparse_dir": "sparse/0",
                "spz_relpath": "scene.spz",
                "sparse_points_relpath": "sparse_points.json",
            },
        }
    )


def test_projection_selects_centered_camera_with_subpixel_accuracy() -> None:
    manifest = _manifest()
    # In frame 0 identity pose, world point (0.2, -0.1, 2.0) projects to (320 + 50, 240 - 25).
    resolution = resolve_viewer_click(
        manifest=manifest,
        world_hit_colmap=(0.2, -0.1, 2.0),
        ray_origin_colmap=(0.0, 0.0, 0.0),
        ray_dir_colmap=(0.099, -0.049, 0.993),
        nearest_frame_idx=None,
    )
    assert resolution.method == "projection"
    assert resolution.ref_frame_idx == 0
    assert abs(resolution.u - 370.0) < 1.0
    assert abs(resolution.v - 215.0) < 1.0


def test_excluded_frame_is_skipped_for_fallback_projection() -> None:
    manifest = _manifest(excluded_first=True)
    # Frame 1 sits at cam_from_world tx=-1.5, so world x=1.5 lands right at principal cx=320.
    resolution = resolve_viewer_click(
        manifest=manifest,
        world_hit_colmap=(1.5, 0.0, 3.0),
        ray_origin_colmap=(1.5, 0.0, 0.0),
        ray_dir_colmap=(0.0, 0.0, 1.0),
        nearest_frame_idx=0,
    )
    assert resolution.method == "projection"
    assert resolution.ref_frame_idx == 1
    assert abs(resolution.u - 320.0) < 1e-6


def test_ray_snap_when_no_3d_hit_provided() -> None:
    manifest = _manifest()
    resolution = resolve_viewer_click(
        manifest=manifest,
        world_hit_colmap=None,
        ray_origin_colmap=(0.0, 0.0, 0.0),
        ray_dir_colmap=(0.0, 0.0, 1.0),
        nearest_frame_idx=1,
    )
    assert resolution.method == "nearest_snap"
    assert resolution.ref_frame_idx == 1
    assert resolution.u == 320.0 and resolution.v == 240.0


def test_all_excluded_raises_geometry_error() -> None:
    manifest = _manifest(excluded_first=True)
    for frame in manifest.frames:
        frame.excluded = True
    with pytest.raises(GeometryError):
        resolve_viewer_click(manifest, (0.0, 0.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), None)
