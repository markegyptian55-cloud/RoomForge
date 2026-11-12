from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from room3d_worker.common.coords import (
    align_gravity_up,
    mod16_crop_or_pad_params,
    project_world_to_image,
    unproject_pixel_to_ray,
)
from room3d_worker.common.errors import GeometryError, ManifestValidationError, StageExecutionError
from room3d_worker.common.manifest import SceneManifest, load_manifest, save_manifest
from room3d_worker.common.quality import compute_overall_progress
from room3d_worker.common.subprocess_runner import run_isolated_stage


def _manifest(width: int = 640, height: int = 480) -> dict[str, object]:
    return {
        "schema_version": "1.0.0", "project_id": "p1", "version_id": "v1",
        "coord_convention": "opencv_colmap", "T_viewer_from_colmap": np.eye(4).tolist(),
        "camera_model": "PINHOLE", "width": width, "height": height,
        "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}, "test_every": 0,
        "frames": [{"frame_idx": 0, "orig_video_frame": 0, "timestamp_sec": 0.0,
                    "image_relpath": "images/0.png", "colmap_image_id": 1,
                    "cam_from_world": np.eye(4).tolist()}],
        "artifacts": {"sparse_dir": "sparse/0", "spz_relpath": "scene.spz",
                      "sparse_points_relpath": "sparse/0/points3D.bin"},
    }


def test_manifest_schema_validation_save_and_load(tmp_path: Path) -> None:
    manifest = SceneManifest.model_validate(_manifest())
    target = tmp_path / "nested" / "manifest.json"
    save_manifest(target, manifest)
    assert load_manifest(target) == manifest
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(_manifest(width=641))
    invalid = _manifest()
    del invalid["schema_version"]
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(invalid)
    target.write_text(json.dumps({"width": 32}), encoding="utf-8")
    with pytest.raises(ManifestValidationError):
        load_manifest(target)


def test_projection_and_ray_unprojection_roundtrip() -> None:
    intrinsics = {"fx": 800.0, "fy": 780.0, "cx": 320.0, "cy": 240.0}
    pose = np.eye(4)
    pose[:3, 3] = [0.3, -0.2, 0.1]
    point = np.array([0.8, 0.4, 4.0])
    u, v, depth = project_world_to_image(point, pose, intrinsics)
    origin, direction = unproject_pixel_to_ray(u, v, pose, intrinsics)
    recovered = origin + direction * np.linalg.norm(point - origin)
    projected = project_world_to_image(recovered, pose, intrinsics)
    assert depth > 0
    assert np.linalg.norm(np.array(projected[:2]) - [u, v]) < 1e-4
    with pytest.raises(GeometryError):
        project_world_to_image(np.array([0.0, 0.0, -1.0]), np.eye(4), intrinsics)


def test_gravity_alignment_is_a_proper_rotation() -> None:
    x, z = np.meshgrid(np.linspace(-2, 2, 8), np.linspace(-2, 2, 8))
    points = np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()])
    cameras = np.array([[0.0, 2.0, 0.0], [1.0, 2.0, 1.0]])
    transform = align_gravity_up(points, cameras)
    rotation = transform[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert np.allclose((transform @ np.array([0.0, 0.0, 0.0, 1.0]))[:3], 0.0, atol=1e-8)


def test_mod16_center_crop_keeps_intrinsics_pixel_exact() -> None:
    result = mod16_crop_or_pad_params(641, 481, 320.25, 240.75, 500.0, 501.0)
    width, height, fx, fy, cx, cy, bounds = result
    assert (width, height) == (640, 480)
    assert bounds == (0, 0, 640, 480)
    assert (fx, fy, cx, cy) == (500.0, 501.0, 320.25, 240.75)
    cropped = mod16_crop_or_pad_params(650, 490, 325.0, 245.0, 500.0, 500.0)
    assert cropped[-1] == (5, 5, 645, 485)
    assert (cropped[4], cropped[5]) == (320.0, 240.0)


def test_progress_uses_contract_stage_weights() -> None:
    assert compute_overall_progress("reconstruct_week1", "02_poses", 0.5) == pytest.approx(30.0)
    assert compute_overall_progress("rebuild_week2", "06_diffusion", 1.0) == pytest.approx(60.0)


def test_subprocess_normalizes_cuda_device_for_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs["env"])
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    run_isolated_stage([sys.executable, "-c", ""], "cuda:1", None, 5)
    assert captured["CUDA_VISIBLE_DEVICES"] == "1"
    with pytest.raises(StageExecutionError, match="must not be empty"):
        run_isolated_stage([sys.executable, "-c", ""], "   ", None, 5)


def test_subprocess_failure_surfaces_domain_error() -> None:
    with pytest.raises(StageExecutionError, match="exited 7"):
        run_isolated_stage([sys.executable, "-c", "raise SystemExit(7)"], None, None, 5)
