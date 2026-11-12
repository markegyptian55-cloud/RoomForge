from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from room3d_worker.pipeline.s02_poses import (
    BackendOutput,
    PoseRequest,
    VGGTChunkedBackend,
    align_sim3,
    estimate_poses,
)


def _frames(count: int) -> list[dict[str, object]]:
    return [{"frame_idx": i, "orig_video_frame": i, "timestamp_sec": float(i),
             "image_relpath": f"images/{i}.png", "colmap_image_id": i + 1,
             "cam_from_world": np.eye(4).tolist()} for i in range(count)]


def _request(tmp_path: Path, backend: str = "colmap_global") -> PoseRequest:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(4):
        (image_dir / f"{i}.png").write_bytes(b"synthetic")
    return PoseRequest(image_dir, tmp_path / "scene", 4, 641, 481,
                       {"fx": 500.0, "fy": 501.0, "cx": 320.25, "cy": 240.75},
                       backend=backend, frame_records=_frames(4))  # type: ignore[arg-type]


def _output(request: PoseRequest, registered: int, sparse: Path | None = None) -> BackendOutput:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], dtype=float)
    cameras = np.array([[0, 2, 0], [1, 2, 0], [0, 2, 1], [1, 2, 1]], dtype=float)
    return BackendOutput(registered, 0.4, 4, points, cameras, _frames(4), sparse)


def test_global_registration_below_threshold_falls_back_to_incremental(tmp_path: Path) -> None:
    request = _request(tmp_path)
    calls: list[str] = []

    @dataclass
    class Fake:
        name: str
        registered: int

        def estimate(self, request: PoseRequest) -> BackendOutput:
            calls.append(self.name)
            return _output(request, self.registered)

    manifest, telemetry = estimate_poses(request, {"colmap_global": Fake("global", 3), "colmap_incremental": Fake("incremental", 4)})  # type: ignore[arg-type]
    assert calls == ["global", "incremental"]
    assert manifest.pose_backend == "colmap_incremental"
    assert telemetry.registered_frames == 4


def test_sim3_chunk_alignment_is_precise() -> None:
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    target = (2.5 * (rotation @ source.T)).T + np.array([3, -2, 4])
    scale, recovered_rotation, translation = align_sim3(source, target)
    recovered = (scale * (recovered_rotation @ source.T)).T + translation
    assert np.allclose(recovered, target, atol=1e-4)


def test_output_is_exactly_sparse_zero_and_intrinsics_are_mod16_adjusted(tmp_path: Path) -> None:
    request = _request(tmp_path)
    source = tmp_path / "raw"
    source.mkdir()
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        (source / filename).write_bytes(filename.encode())
    manifest, _ = estimate_poses(request, {"colmap_global": type("B", (), {"estimate": lambda _, r: _output(r, 4, source)})()})  # type: ignore[arg-type]
    assert (tmp_path / "scene" / "sparse" / "0" / "points3D.bin").exists()
    assert manifest.width == 640 and manifest.height == 480
    assert manifest.intrinsics.cx == 320.25 and manifest.intrinsics.cy == 240.75


def test_vggt_chunked_uses_fp16_and_sim3_alignment(tmp_path: Path) -> None:
    request = _request(tmp_path, "vggt_chunked")
    calls: list[str] = []

    class Runner:
        def infer(self, images: list[Path], *, dtype: str) -> BackendOutput:
            calls.append(dtype)
            out = _output(request, len(images))
            if len(calls) > 1:
                out.points_xyz = out.points_xyz * 2.0 + 10.0
                out.cam_positions_xyz = out.cam_positions_xyz * 2.0 + 10.0
            return out

    output = VGGTChunkedBackend(Runner(), chunk_size=3, overlap=2).estimate(request)
    assert calls and set(calls) == {"fp16"}
    assert output.registered_frames == 4
    assert np.allclose(output.cam_positions_xyz[:3].max(), 2.0, atol=1e-4)
    assert output.sparse_dir == tmp_path / "scene" / "sparse" / "0"
