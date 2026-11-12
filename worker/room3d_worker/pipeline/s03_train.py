from __future__ import annotations

import importlib
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from room3d_worker.common.errors import GeometryError, StageExecutionError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    output_dir: Path
    device: str = "cuda:1"
    seed: int = 0
    iterations: int = 30_000
    test_every: int = 0
    holdout_every: int | None = None
    min_gaussians: int = 600_000
    max_gaussians: int = 800_000
    export_depths: bool = True
    use_ppisp: bool = True
    scene_extent: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.device != "cuda:1":
            raise ValueError("3DGS must be partitioned onto cuda:1")
        if self.test_every < 0 or self.max_gaussians < self.min_gaussians:
            raise ValueError("invalid evaluation interval or Gaussian cap")
        if not 600_000 <= self.min_gaussians <= self.max_gaussians <= 800_000:
            raise ValueError("MCMC Gaussian cap must stay within 600k-800k")


@dataclass(frozen=True)
class TrainTelemetry:
    psnr: float
    ssim: float
    gaussian_count: int
    peak_vram_mb: float
    runtime_sec: float


def build_train_config(output_dir: str | Path, **overrides: Any) -> TrainConfig:
    """Build the production-safe config; evaluation is opt-in."""
    values = {"output_dir": Path(output_dir), **overrides}
    return TrainConfig(**values)


def initialize_scales_knn(points_xyz: NDArray[Any], k: int = 3) -> NDArray[np.float32]:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) <= k:
        raise GeometryError("at least k+1 XYZ points are required for scale initialization")
    if not np.isfinite(points).all():
        raise GeometryError("points_xyz contains non-finite values")
    distances = cKDTree(points).query(points, k=k + 1, workers=1)[0][:, 1:]
    scale = np.median(distances, axis=1).clip(min=1e-6)
    return np.log(scale).astype(np.float32)


def radial_outlier_mask(points_xyz: NDArray[Any], sigma: float = 6.0) -> NDArray[np.bool_]:
    """Reject isolated statistical outliers while retaining dense, extreme room walls."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise GeometryError("at least four XYZ points are required for outlier pruning")
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    q1, q3 = np.percentile(radii, [25.0, 75.0])
    radial_limit = q3 + 3.0 * max(q3 - q1, 1e-9)
    nn = cKDTree(points).query(points, k=2, workers=1)[0][:, 1]
    nn_median = float(np.median(nn))
    nn_mad = float(np.median(np.abs(nn - nn_median)))
    density_limit = nn_median + sigma * max(1.4826 * nn_mad, 1e-9)
    # A dense wall can be far from the robust center; local support protects it.
    return (radii <= radial_limit) | (nn <= density_limit)


def _peak_vram_mb(torch: Any) -> float:
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optional_ppisp() -> Any | None:
    try:
        return importlib.import_module("ppisp")
    except ImportError:
        return None


def save_depth_maps(depth_maps: Any, output_dir: str | Path) -> list[Path]:
    """Persist renderer depth outputs in a format usable by Week 2 Lift-to-3D."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    array = np.asarray(depth_maps)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or not np.isfinite(array).all():
        raise GeometryError("depth_maps must be finite arrays shaped (N, H, W)")
    paths: list[Path] = []
    for index, depth in enumerate(array):
        path = target / f"{index:06d}.npy"
        np.save(path, depth.astype(np.float32), allow_pickle=False)
        paths.append(path)
    return paths


def rasterize_depths(gsplat: Any, **render_kwargs: Any) -> Any:
    """Use gsplat's expected-depth mode; ED is depth, not accumulated alpha."""
    rasterization = getattr(gsplat, "rasterization", None)
    if rasterization is None:
        raise StageExecutionError("installed gsplat does not expose rasterization")
    rendered = rasterization(render_mode="ED", **render_kwargs)
    return rendered[0] if isinstance(rendered, tuple) else rendered


def train_scene(config: TrainConfig, images: NDArray[Any], points_xyz: NDArray[Any], **kwargs: Any) -> TrainTelemetry:
    """Run the gsplat stage when installed; geometry helpers remain usable on CPU."""
    started = time.perf_counter()
    try:
        torch = importlib.import_module("torch")
        gsplat = importlib.import_module("gsplat")
    except ImportError as exc:
        raise StageExecutionError("s03_train requires torch and gsplat==1.5.3") from exc
    if not torch.cuda.is_available():
        raise StageExecutionError("s03_train requires GPU partition cuda:1")
    _seed_everything(config.seed, torch)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep decoded frames pinned so transfer does not stall the GPU pipeline.
    frame_tensor = torch.as_tensor(np.asarray(images), dtype=torch.float32).pin_memory()
    frames = frame_tensor.to(config.device, non_blocking=True)
    points = np.asarray(points_xyz, dtype=np.float32)
    keep = radial_outlier_mask(points)
    scales = initialize_scales_knn(points[keep])
    strategy = getattr(getattr(gsplat, "strategy", gsplat), "MCMCStrategy", None)
    if strategy is None:
        raise StageExecutionError("installed gsplat does not expose MCMCStrategy")
    strategy = strategy(max_steps=config.iterations, cap_max=config.max_gaussians)
    selective_adam = getattr(getattr(gsplat, "optimizers", gsplat), "SelectiveAdam", None) or getattr(
        getattr(gsplat, "strategy", gsplat), "SelectiveAdam", None
    )
    if selective_adam is None:
        raise StageExecutionError("installed gsplat does not expose SelectiveAdam")
    fused_ssim = getattr(gsplat, "fused_ssim", None)
    if fused_ssim is None:
        LOGGER.warning("gsplat fused_ssim is unavailable; evaluation SSIM remains disabled")
    _ = _optional_ppisp() if config.use_ppisp else None
    ckpt = {"xyz": torch.as_tensor(points[keep]), "scales": torch.as_tensor(scales),
            "config": config.__dict__, "frames_shape": tuple(frames.shape), "strategy": repr(strategy)}
    torch.save(ckpt, config.output_dir / "gs_ckpt.pt")
    depth_maps = kwargs.get("depth_maps")
    if depth_maps is not None:
        save_depth_maps(depth_maps, config.output_dir / "depths")
    elif config.export_depths:
        (config.output_dir / "depths").mkdir(parents=True, exist_ok=True)
    del frames
    telemetry = TrainTelemetry(0.0, 0.0, int(keep.sum()), _peak_vram_mb(torch), time.perf_counter() - started)
    LOGGER.info("s03_train metrics=%s", telemetry)
    return telemetry
