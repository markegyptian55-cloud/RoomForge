from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import pytest

from room3d_worker.common.errors import StageExecutionError
from room3d_worker.pipeline.s06_diffusion import (
    DiffusionConfig,
    assert_unmasked_unchanged,
    composite_strict_mask,
    generate_first_frame_seeds,
    load_diffusion_backend,
    pad_to_multiple,
    propagate_masked_edits,
    resolve_t4_dtype,
)


class _RecordingDiffusionBackend:
    def __init__(self, fill_rgb: tuple[int, int, int] = (200, 40, 20)) -> None:
        self.fill_rgb = fill_rgb
        self.calls: list[tuple[tuple[int, int], int, bool, tuple[int, int] | None]] = []

    def inpaint(
        self,
        image: NDArray[np.uint8],
        mask: NDArray[np.uint8],
        prompt: str,
        seed: int,
        image_reference: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        _ = (mask, prompt)
        ref_hw = image_reference.shape[:2] if image_reference is not None else None
        self.calls.append((image.shape[:2], int(seed), image_reference is not None, ref_hw))
        return np.full_like(image, self.fill_rgb, dtype=np.uint8)


def test_t4_compute_capability_and_license_guard() -> None:
    assert resolve_t4_dtype((7, 5)) == "float16"
    assert resolve_t4_dtype(None) == "float32"
    assert resolve_t4_dtype((8, 6)) == "bfloat16"
    with pytest.raises(StageExecutionError, match="license guardrails"):
        DiffusionConfig(model_id="black-forest-labs/FLUX.2-klein-base-9B")
    with pytest.raises(StageExecutionError, match="license guardrails"):
        DiffusionConfig(model_id="Qwen/Qwen-Image-2.1")


def test_strict_mask_compositor_validates_types_and_zero_unmasked_drift() -> None:
    rng = np.random.default_rng(7)
    orig = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    edited = np.full_like(orig, 255)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:40, 20:44] = 255

    composited = composite_strict_mask(orig, edited, mask, feather_px=3)
    assert assert_unmasked_unchanged(orig, composited, mask) == 0
    assert np.any(composited[mask == 255] != orig[mask == 255])

    with pytest.raises(StageExecutionError, match="2D uint8"):
        composite_strict_mask(orig, edited, mask.astype(np.int64))


def test_grid_oom_guard_empty_masks_and_ref_padding(tmp_path: Path) -> None:
    frame_paths: dict[int, Path] = {}
    mask_paths: dict[int, Path] = {}
    for idx in range(5):
        img = np.full((50, 70, 3), 20 + idx * 10, dtype=np.uint8)
        mask = np.zeros((50, 70), dtype=np.uint8)
        if idx != 4:
            mask[10:30, 15:45] = 255
        fp = tmp_path / f"f_{idx}.png"
        mp = tmp_path / f"m_{idx}.png"
        Image.fromarray(img).save(fp, format="PNG")
        Image.fromarray(mask).save(mp, format="PNG")
        frame_paths[idx] = fp
        mask_paths[idx] = mp

    backend = _RecordingDiffusionBackend()
    seeds = generate_first_frame_seeds(frame_paths[0], mask_paths[0], "red velvet chair", [101, 202], tmp_path / "seeds", backend)
    assert len(seeds) == 2

    ref_path = Path(seeds[0]["path"])
    metrics = propagate_masked_edits(
        frame_paths=frame_paths,
        mask_paths=mask_paths,
        ref_frame_idx=0,
        ref_edited_path=ref_path,
        prompt="red velvet chair",
        seed=101,
        output_dir=tmp_path / "out_grid",
        backend=backend,
        skip_frames=set(),
        config=DiffusionConfig(grid_2x2=True),
    )
    # Frames 0..3 execute as one 2x2 grid batch; frame 4 has an empty mask and skips diffusion.
    assert metrics["frames_edited"] == 4
    assert metrics["max_unmasked_diff"] == 0
    grid_call = next(c for c in backend.calls if c[0] == (112, 144))
    assert grid_call[2] is True
    assert grid_call[3] == (112, 144)


def test_sequential_mode_and_oversized_grid_fallback(tmp_path: Path) -> None:
    frame_paths: dict[int, Path] = {}
    mask_paths: dict[int, Path] = {}
    for idx in range(4):
        img = np.full((600, 600, 3), 50, dtype=np.uint8)
        mask = np.zeros((600, 600), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        fp, mp = tmp_path / f"big_{idx}.png", tmp_path / f"big_m_{idx}.png"
        Image.fromarray(img).save(fp, format="PNG")
        Image.fromarray(mask).save(mp, format="PNG")
        frame_paths[idx], mask_paths[idx] = fp, mp

    backend = _RecordingDiffusionBackend()
    padded_img, padded_mask, orig_hw = pad_to_multiple(np.zeros((50, 70, 3), dtype=np.uint8), np.zeros((50, 70), dtype=np.uint8), 16)
    assert (padded_img.shape[:2], padded_mask.shape[:2], orig_hw) == ((64, 80), (64, 80), (50, 70))

    metrics = propagate_masked_edits(
        frame_paths=frame_paths,
        mask_paths=mask_paths,
        ref_frame_idx=0,
        ref_edited_path=frame_paths[0],
        prompt="blue armchair",
        seed=7,
        output_dir=tmp_path / "out_big",
        backend=backend,
        config=DiffusionConfig(grid_2x2=True),
    )
    assert metrics["frames_edited"] == 4
    assert all(call[0][0] < 1200 for call in backend.calls)


def test_load_diffusion_backend_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _FakePipe:
        def to(self, device: str) -> None:
            calls.append(f"to:{device}")

        def enable_vae_tiling(self) -> None:
            calls.append("tiling")

        def enable_vae_slicing(self) -> None:
            calls.append("slicing")

    class _FakeFlux:
        @classmethod
        def from_pretrained(cls, _repo: str, torch_dtype: Any, token: Any = None) -> _FakePipe:
            calls.append(f"dtype:{torch_dtype}")
            return _FakePipe()

    fake_torch = SimpleNamespace(
        float32="fp32",
        float16="fp16",
        bfloat16="bf16",
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    fake_diffusers = SimpleNamespace(Flux2KleinInpaintPipeline=_FakeFlux)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    backend = load_diffusion_backend(DiffusionConfig(device="cuda:0"))
    assert backend is not None
    assert "dtype:fp32" in calls
    assert "to:cpu" in calls
    assert "slicing" in calls
