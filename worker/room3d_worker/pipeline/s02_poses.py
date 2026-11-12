from __future__ import annotations

import importlib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from room3d_worker.common.coords import align_gravity_up, mod16_crop_or_pad_params
from room3d_worker.common.errors import GeometryError, StageExecutionError
from room3d_worker.common.manifest import SceneManifest

BackendName = Literal["colmap_global", "colmap_incremental", "vggt_chunked", "map_anything"]
FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class PoseRequest:
    image_dir: Path
    output_dir: Path
    total_frames: int
    width: int
    height: int
    intrinsics: dict[str, float]
    project_id: str = "project"
    version_id: str = "version"
    database_path: Path | None = None
    backend: BackendName = "colmap_global"
    frame_records: list[dict[str, Any]] = field(default_factory=list)
    manifest_path: Path | None = None
    depth_dir: Path | None = None
    vocab_tree_path: Path | None = None


@dataclass(frozen=True)
class PoseTelemetry:
    registered_frames: int
    total_frames: int
    mean_reproj_px: float
    num_points3d: int
    runtime_sec: float
    peak_vram_mb: float


@dataclass
class BackendOutput:
    registered_frames: int
    mean_reproj_px: float
    num_points3d: int
    points_xyz: NDArray[np.float64]
    cam_positions_xyz: NDArray[np.float64]
    frames: list[dict[str, Any]]
    sparse_dir: Path | None = None
    depth_dir: Path | None = None


class PoseBackend(Protocol):
    def estimate(self, request: PoseRequest) -> BackendOutput: ...


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise StageExecutionError(f"Pose backend dependency is unavailable: {name}") from exc


def _call_optional(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except TypeError:
        return fn(*args)


def _as_points(value: Any, name: str) -> NDArray[np.float64]:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise GeometryError(f"{name} must contain at least three XYZ points")
    if not np.isfinite(points).all():
        raise GeometryError(f"{name} contains non-finite values")
    return points


def align_sim3(source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Return scale, rotation, translation mapping source points onto target points."""
    source, target = _as_points(source_xyz, "source_xyz"), _as_points(target_xyz, "target_xyz")
    if source.shape != target.shape:
        raise GeometryError("Sim(3) alignment requires equal point counts")
    source_mean, target_mean = source.mean(0), target.mean(0)
    x, y = source - source_mean, target - target_mean
    covariance = y.T @ x / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(x * x, axis=1)))
    if variance <= 1e-12:
        raise GeometryError("Sim(3) source points have zero variance")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def _peak_vram_mb() -> float:
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except ImportError:
        return 0.0
    return 0.0


class PyColmapBackend:
    def __init__(self, mode: Literal["global", "incremental"]) -> None:
        self.mode = mode

    def estimate(self, request: PoseRequest) -> BackendOutput:
        pycolmap = _module("pycolmap")
        database = request.database_path or request.output_dir / "database.db"
        if database.resolve().parent == request.image_dir.resolve() or request.image_dir.resolve() in database.resolve().parents:
            raise StageExecutionError("database.db must be outside image_dir")
        database.parent.mkdir(parents=True, exist_ok=True)
        extraction = self._extract(pycolmap, database, request)
        if self.mode == "global":
            result = self._global(pycolmap, database, request, request.output_dir / "raw_sparse", extraction)
        else:
            result = self._incremental(pycolmap, database, request.image_dir, request.output_dir / "raw_sparse", extraction)
        return _coerce_backend_output(result, request)

    def _extract(self, pycolmap: Any, database: Path, request: PoseRequest) -> Any:
        enums = (getattr(pycolmap, "FeatureExtractorType", None), getattr(pycolmap, "FeatureMatcherType", None))
        onnx_ready = all(enum is not None and hasattr(enum, name) for enum, name in zip(enums, ("ALIKED_N16ROT", "ALIKED_LIGHTGLUE")))
        if onnx_ready:
            extractor, matcher = enums[0].ALIKED_N16ROT, enums[1].ALIKED_LIGHTGLUE
        else:
            extractor, matcher = "DSP-SIFT", "sequential"
        extraction_options = {"feature_type": extractor, "max_num_features": 4096}
        _call_optional(pycolmap.extract_features, str(database), str(request.image_dir), extraction_options=extraction_options)
        match_options = {"overlap": 12, "loop_detection": True, "matcher_type": matcher}
        return _call_optional(pycolmap.match_sequential, str(database), match_options=match_options)

    def _global(self, pycolmap: Any, database: Path, request: PoseRequest, sparse: Path, _: Any) -> Any:
        sparse.mkdir(parents=True, exist_ok=True)
        options = {"vocab_tree_path": str(request.vocab_tree_path) if request.vocab_tree_path else None, "use_faiss": True}
        result = _call_optional(pycolmap.global_mapping, str(database), str(request.image_dir), str(sparse), options=options)
        undistort = getattr(pycolmap, "undistort_images", None)
        if undistort is not None:
            input_sparse = sparse / "0" if (sparse / "0").is_dir() else sparse
            _call_optional(undistort, str(request.image_dir), str(input_sparse), str(sparse))
        return result

    def _incremental(self, pycolmap: Any, database: Path, image_dir: Path, sparse: Path, _: Any) -> Any:
        sparse.mkdir(parents=True, exist_ok=True)
        options = {"ba_use_gpu": True, "ba_global_frames_ratio": 1.25, "ba_global_max_num_iterations": 30}
        return _call_optional(pycolmap.incremental_mapping, str(database), str(image_dir), str(sparse), options=options)


class VGGTRunner(Protocol):
    def infer(self, images: Sequence[Path], *, dtype: str) -> BackendOutput: ...


class VGGTChunkedBackend:
    def __init__(self, runner: VGGTRunner, chunk_size: int = 32, overlap: int = 6) -> None:
        self.runner, self.chunk_size, self.overlap = runner, chunk_size, overlap

    def estimate(self, request: PoseRequest) -> BackendOutput:
        images = sorted(request.image_dir.glob("*"))
        chunks: list[BackendOutput] = []
        step = self.chunk_size - self.overlap
        for start in range(0, len(images), step):
            chunk = self.runner.infer(images[start : start + self.chunk_size], dtype="fp16")
            if chunks:
                self._align_chunk(chunks[0], chunk)
            chunks.append(chunk)
            if start + self.chunk_size >= len(images):
                break
        if not chunks:
            raise StageExecutionError("VGGT received no images")
        result = chunks[0]
        for chunk in chunks[1:]:
            trim = min(self.overlap, len(chunk.frames))
            result.points_xyz = np.vstack((result.points_xyz, chunk.points_xyz))
            result.cam_positions_xyz = np.vstack((result.cam_positions_xyz, chunk.cam_positions_xyz[trim:]))
            result.frames.extend(chunk.frames[trim:])
            result.registered_frames += max(0, chunk.registered_frames - trim)
            result.num_points3d += chunk.num_points3d
        result.sparse_dir = request.output_dir / "sparse" / "0"
        result.depth_dir = request.depth_dir
        return result

    def _align_chunk(self, reference: BackendOutput, chunk: BackendOutput) -> None:
        count = min(len(reference.cam_positions_xyz), len(chunk.cam_positions_xyz))
        if count < 3:
            raise GeometryError("VGGT chunk overlap needs at least three camera centers for Sim(3)")
        count = max(3, min(self.overlap, count))
        scale, rotation, translation = align_sim3(chunk.cam_positions_xyz[:count], reference.cam_positions_xyz[-count:])
        chunk.points_xyz = (scale * (rotation @ chunk.points_xyz.T)).T + translation
        chunk.cam_positions_xyz = (scale * (rotation @ chunk.cam_positions_xyz.T)).T + translation


class MapAnythingBackend:
    def __init__(self, runner: VGGTRunner) -> None:
        self.runner = runner

    def estimate(self, request: PoseRequest) -> BackendOutput:
        return self.runner.infer(sorted(request.image_dir.glob("*")), dtype="fp16")


def _from_reconstruction(rec: Any, request: PoseRequest, sparse_dir: Path) -> BackendOutput:
    points = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64)
    if len(points) < 3:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam_positions = np.array([img.projection_center() for img in rec.images.values()], dtype=np.float64)
    if len(cam_positions) == 0:
        cam_positions = np.zeros((1, 3), dtype=np.float64)
    reproj = float(rec.compute_mean_reprojection_error()) if hasattr(rec, "compute_mean_reprojection_error") and rec.points3D else 0.0
    return BackendOutput(
        registered_frames=len(rec.images),
        mean_reproj_px=reproj,
        num_points3d=len(rec.points3D),
        points_xyz=points,
        cam_positions_xyz=cam_positions,
        frames=request.frame_records,
        sparse_dir=sparse_dir,
    )


def _coerce_backend_output(value: Any, request: PoseRequest) -> BackendOutput:
    if isinstance(value, BackendOutput):
        return value
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return BackendOutput(**value)
    if isinstance(value, dict) and value:
        best = max(value.values(), key=lambda rec: len(getattr(rec, "images", {})))
        return _from_reconstruction(best, request, request.output_dir / "raw_sparse")
    raise StageExecutionError(f"Pose backend returned unsupported result: {type(value).__name__}")


def _copy_sparse(source: Path | None, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if source is None:
        return
    source = source / "0" if source is not None and (source / "0").is_dir() else source
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        candidate = source / name
        if candidate.exists():
            shutil.copy2(candidate, target / name)


def _default_backends() -> dict[BackendName, PoseBackend]:
    return {
        "colmap_global": PyColmapBackend("global"),
        "colmap_incremental": PyColmapBackend("incremental"),
    }


def estimate_poses(request: PoseRequest, backends: dict[BackendName, PoseBackend] | None = None) -> tuple[SceneManifest, PoseTelemetry]:
    started = time.perf_counter()
    selected = backends or _default_backends()
    backend_name = request.backend
    output = selected[backend_name].estimate(request)
    if backend_name == "colmap_global" and output.registered_frames / max(request.total_frames, 1) < 0.85:
        backend_name = "colmap_incremental"
        output = selected[backend_name].estimate(request)
    width, height, fx, fy, cx, cy, _ = mod16_crop_or_pad_params(
        request.width, request.height, request.intrinsics["cx"], request.intrinsics["cy"], request.intrinsics["fx"], request.intrinsics["fy"]
    )
    transform = align_gravity_up(output.points_xyz, output.cam_positions_xyz)
    sparse_dir = request.output_dir / "sparse" / "0"
    _copy_sparse(output.sparse_dir, sparse_dir)
    frames = output.frames or request.frame_records
    runtime_sec = time.perf_counter() - started
    peak_vram_mb = _peak_vram_mb()
    metrics = {"registered_frames": output.registered_frames, "total_frames": request.total_frames,
               "mean_reproj_px": output.mean_reproj_px, "num_points3d": output.num_points3d,
               "runtime_sec": runtime_sec, "peak_vram_mb": peak_vram_mb}
    manifest = SceneManifest.model_validate({
        "schema_version": "1.0.0", "project_id": request.project_id, "version_id": request.version_id,
        "coord_convention": "opencv_colmap", "T_viewer_from_colmap": transform.tolist(),
        "camera_model": "PINHOLE", "width": width, "height": height,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy}, "test_every": 0,
        "pose_backend": backend_name, "frames": frames,
        "artifacts": {"sparse_dir": "sparse/0", "spz_relpath": "scene.spz", "sparse_points_relpath": "sparse/0/points3D.bin"},
        "metrics": metrics,
    })
    telemetry = PoseTelemetry(output.registered_frames, request.total_frames, output.mean_reproj_px, output.num_points3d, runtime_sec, peak_vram_mb)
    if request.manifest_path:
        request.manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest, telemetry
