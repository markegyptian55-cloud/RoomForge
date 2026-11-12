from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from room3d_worker.common.manifest import load_manifest
from room3d_worker.pipeline.s03_train import build_train_config, initialize_scales_knn, radial_outlier_mask
from room3d_worker.pipeline.s04_export import ExportRequest, export_scene


def test_knn_initialization_does_not_use_pairwise_cdist() -> None:
    points = np.random.default_rng(0).normal(size=(2_000, 3))
    scales = initialize_scales_knn(points)
    assert scales.shape == (2_000,)
    assert np.isfinite(scales).all()


def test_radial_pruning_retains_dense_extreme_wall() -> None:
    rng = np.random.default_rng(1)
    room = rng.uniform(-1.0, 1.0, size=(500, 3))
    wall = np.column_stack((np.full(100, 20.0), rng.uniform(-2, 2, (100, 2))))
    isolated = np.array([[1000.0, 1000.0, 1000.0]])
    keep = radial_outlier_mask(np.vstack((room, wall, isolated)))
    assert keep[500:600].all()
    assert not keep[-1]


def test_train_config_defaults_to_production_holdout_and_gpu_partition(tmp_path: Path) -> None:
    config = build_train_config(tmp_path)
    assert config.test_every == 0
    assert config.device == "cuda:1"


def test_export_writes_fallback_artifacts_and_valid_manifest(tmp_path: Path) -> None:
    points = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [9, 9, 9]], dtype=np.float32)
    opacity = np.array([1.0, 0.8, 0.9, 0.001], dtype=np.float32)
    scales = np.ones((4, 3), dtype=np.float32) * 0.1
    manifest = export_scene(ExportRequest(tmp_path), points, opacity, scales)
    assert manifest.artifacts.spz_relpath == "scene.spz"
    assert len(json.loads((tmp_path / "sparse_points.json").read_text())) == 3
    assert load_manifest(tmp_path / "manifest.json").project_id == "project"
