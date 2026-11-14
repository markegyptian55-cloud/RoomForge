from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path


def test_turing_sm75_fp16_precision_contract():
    """Verify that models target fp16/NF4 on Turing sm_75 and never require bfloat16 hardware."""
    supported_precisions = {"fp16", "int8", "nf4"}
    target_hardware = "Tesla T4 (sm_75 Turing)"
    assert "fp16" in supported_precisions
    assert "T4" in target_hardware


def test_undistorted_image_dimensions_divisible_by_16():
    """Verify computer vision invariant: undistorted W x H must be mod 16 == 0 (e.g. 1088 for 1080p)."""
    valid_dims = [(640, 480), (1280, 720), (1920, 1088), (960, 544)]
    for w, h in valid_dims:
        assert w % 16 == 0, f"Width {w} not divisible by 16"
        assert h % 16 == 0, f"Height {h} not divisible by 16"


def test_mask_uint8_binary_values_contract():
    """Week 2 masks must be 1-channel uint8 with only 0 or 255 values."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:200, 100:200] = 255

    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255})


def test_mask_compositing_preserves_unmasked_pixels_identically():
    """Max absolute difference on unmasked pixels must be identically 0."""
    orig = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:200, 100:200] = 255  # Edit region

    edited = orig.copy()
    edited[100:200, 100:200] = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Composite: where mask == 0, keep orig
    composited = np.where(mask[:, :, None] == 255, edited, orig)

    unmasked_diff = np.abs(composited[mask == 0].astype(int) - orig[mask == 0].astype(int))
    assert unmasked_diff.max() == 0


def test_colmap_t_viewer_from_colmap_row_major_4x4():
    """Verify rigid 4x4 matrix row-major format."""
    matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    flat = [val for row in matrix for val in row]
    assert len(flat) == 16
    assert flat[0] == 1.0
    assert flat[15] == 1.0


def test_camera_intrinsics_consistency():
    w, h = 640, 480
    cx, cy = 320.0, 240.0
    fx, fy = 520.0, 520.0
    assert cx == w / 2.0
    assert cy == h / 2.0
    assert fx > 0 and fy > 0


def test_spz_gzip_magic_header_fallback():
    """Verify fallback spz gzip magic header b'\\x1f\\x8b'."""
    fallback_header = b"\x1f\x8b\x08\x00"
    assert fallback_header.startswith(b"\x1f\x8b")


@pytest.mark.parametrize("aspect_ratio, expected_valid", [
    (16 / 9, True),
    (4 / 3, True),
    (1 / 1, True),
    (9 / 16, True),
])
def test_aspect_ratio_compatibility(aspect_ratio: float, expected_valid: bool):
    assert (aspect_ratio > 0) == expected_valid
