from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest

from room3d_worker.common.errors import StageExecutionError
from room3d_worker.pipeline.s04_export import ExportRequest, export_scene
from room3d_worker.pipeline.s07_rebuild import (
    GaussianState,
    RebuildRequest,
    compute_masked_l1_loss,
    rebuild_scene,
)


def _setup_base_scene(tmp_path: Path, with_sparse: bool = True) -> tuple[Path, Path, Path]:
    base_dir = tmp_path / "v1"
    edited_dir = tmp_path / "edited"
    masks_dir = tmp_path / "masks"
    edited_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    pts = np.array([[0.0, 0.0, 4.0], [3.5, 0.0, 4.0], [-3.5, 0.0, 4.0], [0.2, 0.2, 4.0]], dtype=np.float32)
    op = np.full(4, 0.85, dtype=np.float32)
    scales = np.full((4, 3), 0.02, dtype=np.float32)
    colors = np.full((4, 3), 120, dtype=np.uint8)
    frames = [
        {"frame_idx": 0, "orig_video_frame": 0, "timestamp_sec": 0.0, "image_relpath": "images/frame_000000.jpg",
         "colmap_image_id": 1, "cam_from_world": np.eye(4).tolist()},
        {"frame_idx": 1, "orig_video_frame": 1, "timestamp_sec": 0.5, "image_relpath": "images/frame_000001.jpg",
         "colmap_image_id": 2, "cam_from_world": [[1, 0, 0, -0.1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
    ]
    export_scene(
        ExportRequest(output_dir=base_dir, project_id="proj-1", version_id="v1", width=160, height=128,
                      intrinsics={"fx": 160.0, "fy": 160.0, "cx": 80.0, "cy": 64.0}, frames=frames),
        points_xyz=pts, opacity=op, scales=scales, colors=colors,
    )
    if with_sparse:
        (base_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
        (base_dir / "sparse" / "0" / "cameras.bin").write_bytes(b"cam")
    mask = np.zeros((128, 160), dtype=np.uint8)
    mask[56:80, 72:104] = 255
    for idx in (0, 1):
        Image.fromarray(mask).save(masks_dir / f"{idx:05d}.png", format="PNG")
        Image.fromarray(np.full((128, 160, 3), 240, dtype=np.uint8)).save(edited_dir / f"{idx:05d}.png", format="PNG")
    return base_dir, edited_dir, masks_dir


def test_masked_finetune_pytorch_opt_freezes_background_and_updates_roi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    base_dir, edited_dir, masks_dir = _setup_base_scene(tmp_path)
    out_dir = tmp_path / "v2_masked"

    def _fake_rasterize(*, colors: Any, means: Any, width: int, height: int, **_kwargs: Any) -> tuple[Any, ...]:
        # Differentiable proxy projecting ROI/all Gaussian colors onto the image canvas for Adam autograd verification.
        _ = means
        fg_color = colors[0]
        canvas = fg_color.view(1, 1, 1, 3).expand(1, height, width, 3)
        return (canvas,)

    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(rasterization=_fake_rasterize))
    metrics = rebuild_scene(
        RebuildRequest(
            base_dir=base_dir, edited_frames_dir=edited_dir, masks_dir=masks_dir, output_dir=out_dir,
            project_id="proj-1", version_id="v2_masked", parent_version_id="v1",
            mode="masked_finetune", fine_tune_steps=25, exclude_frames=(1,), roi_pad_px=2, device="cpu",
        )
    )
    assert metrics["roi_gaussian_count"] == 2
    assert metrics["frozen_gaussian_count"] == 2
    assert metrics["max_pose_drift"] == 0.0
    assert metrics["max_unmasked_gaussian_drift"] == 0.0
    with np.load(out_dir / "gs_state.npz", allow_pickle=False) as archive:
        cols = archive["colors"]
        assert cols[0, 0] > 120
        assert tuple(cols[1]) == (120, 120, 120)
    assert (out_dir / "sparse" / "0" / "cameras.bin").read_bytes() == b"cam"


def test_leaking_backend_mutating_background_fails_loudly(tmp_path: Path) -> None:
    base_dir, edited_dir, masks_dir = _setup_base_scene(tmp_path)

    def _bad_backend(_r: Any, _m: Any, state: GaussianState, _roi: np.ndarray) -> GaussianState:
        mutated = state.colors.copy()
        mutated[:] = 250
        return GaussianState(points_xyz=state.points_xyz, colors=mutated, opacity=state.opacity, scales=state.scales)

    with pytest.raises(StageExecutionError, match="mutated frozen background Gaussians"):
        rebuild_scene(
            RebuildRequest(
                base_dir=base_dir, edited_frames_dir=edited_dir, masks_dir=masks_dir, output_dir=tmp_path / "fail",
                project_id="proj-1", version_id="v2_bad", parent_version_id="v1", mode="masked_finetune",
            ),
            backend=_bad_backend,
        )


def test_from_scratch_ablation_without_masks_and_missing_sparse_fail(tmp_path: Path) -> None:
    base_dir, edited_dir, _ = _setup_base_scene(tmp_path)
    out_scratch = tmp_path / "v2_scratch"
    metrics = rebuild_scene(
        RebuildRequest(
            base_dir=base_dir, edited_frames_dir=edited_dir, masks_dir=tmp_path / "empty_masks",
            output_dir=out_scratch, project_id="proj-1", version_id="v2_scratch", parent_version_id="v1",
            mode="from_scratch",
        ),
        backend=lambda _r, _m, state, _roi: state,
    )
    assert metrics["steps"] == 15000
    assert metrics["frozen_gaussian_count"] == 0

    no_sparse, ed2, m2 = _setup_base_scene(tmp_path / "ns", with_sparse=False)
    with pytest.raises(StageExecutionError, match="sparse/0"):
        rebuild_scene(
            RebuildRequest(
                base_dir=no_sparse, edited_frames_dir=ed2, masks_dir=m2,
                output_dir=tmp_path / "v2_ns", project_id="proj-1", version_id="v2_ns", parent_version_id="v1",
            ),
            backend=lambda _r, _m, state, _roi: state,
        )


def test_masked_l1_loss_ignores_background_pixels() -> None:
    rendered = np.zeros((16, 16, 3), dtype=np.float32)
    target = np.zeros((16, 16, 3), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 255
    rendered[4:8, 4:8] = 1.0
    target[:4, :4] = 999.0
    assert compute_masked_l1_loss(rendered, target, mask) == pytest.approx(1.0)
