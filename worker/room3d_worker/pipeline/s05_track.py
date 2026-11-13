from __future__ import annotations

import importlib
import json
import tempfile
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from room3d_worker.common.coords import project_world_to_image, unproject_pixel_to_ray
from room3d_worker.common.errors import GeometryError, StageExecutionError
from room3d_worker.common.manifest import FrameRecord, SceneManifest

MaskArray = NDArray[np.uint8]
FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class PromptPoint:
    x: float
    y: float
    label: Literal[0, 1]


@dataclass(frozen=True)
class ManualOverride:
    frame_idx: int
    mask_path: Path | None = None
    skip_frame: bool = False


class InteractiveSamBackend(Protocol):
    def encode_image(self, image_path: Path) -> Any:
        ...

    def predict(self, embedding: Any, points: list[PromptPoint]) -> NDArray[Any]:
        ...


class VideoSamBackend(Protocol):
    def propagate(
        self,
        image_paths: dict[int, Path],
        ref_frame_idx: int,
        seed_mask: MaskArray,
        inference_state_device: str,
    ) -> dict[int, MaskArray]:
        ...


class ReidEncoder(Protocol):
    def embed_crop(self, rgb_crop: NDArray[np.uint8], mask_crop: MaskArray | None = None) -> FloatArray:
        ...


def normalize_binary_mask(raw_mask: Any, expected_hw: tuple[int, int]) -> MaskArray:
    """Convert bool/float/logits/uint8 mask outputs into strict 1-channel uint8 {0, 255}."""
    if hasattr(raw_mask, "detach") and callable(raw_mask.detach):
        raw_mask = raw_mask.detach()
    if hasattr(raw_mask, "cpu") and callable(raw_mask.cpu):
        raw_mask = raw_mask.cpu()
    if hasattr(raw_mask, "numpy") and callable(raw_mask.numpy):
        raw_mask = raw_mask.numpy()
    arr = np.squeeze(np.asarray(raw_mask))
    if arr.shape != expected_hw:
        raise StageExecutionError(f"Mask shape {arr.shape} does not match expected {expected_hw}")
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8) * np.uint8(255)
    if np.issubdtype(arr.dtype, np.floating):
        threshold = 0.5 if (arr.min(initial=0.0) >= 0.0 and arr.max(initial=0.0) <= 1.0) else 0.0
    else:
        threshold = 127 if arr.max(initial=0) > 1 else 0
    return np.where(arr > threshold, np.uint8(255), np.uint8(0)).astype(np.uint8)


def save_binary_mask_png(mask: NDArray[Any], path: Path, expected_hw: tuple[int, int]) -> Path:
    normalized = normalize_binary_mask(mask, expected_hw)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalized).save(path, format="PNG")
    return path


def load_binary_mask_png(path: Path, expected_hw: tuple[int, int]) -> MaskArray:
    if not path.exists():
        raise StageExecutionError(f"Mask file missing: {path}")
    with Image.open(path) as img:
        img.load()
        if img.mode != "L":
            raise StageExecutionError(f"Mask must be mode 'L', got '{img.mode}': {path}")
        arr = np.asarray(img, dtype=np.uint8)
    if arr.shape != expected_hw or np.any((arr != 0) & (arr != 255)):
        raise StageExecutionError(f"Invalid mask dimensions or pixel values: {path}")
    return arr


def mask_iou(a: MaskArray, b: MaskArray) -> float:
    fa, fb = (a == 255), (b == 255)
    union = int(np.logical_or(fa, fb).sum())
    return 1.0 if union == 0 and not np.any(fa) else (float(np.logical_and(fa, fb).sum()) / union if union > 0 else 0.0)


def _to_cpu_if_tensor(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_cpu_if_tensor(v) for k, v in obj.items()}
    return obj.detach().to("cpu") if hasattr(obj, "to") and callable(obj.to) else obj


class WarmSamSession:
    """Warm interactive SAM 2.1 predictor wrapper with bounded LRU frame-embedding cache."""

    def __init__(self, backend: InteractiveSamBackend, max_cache_items: int = 8) -> None:
        self._backend = backend
        self._max_cache_items = max(1, int(max_cache_items))
        self._cache: OrderedDict[tuple[str, int], Any] = OrderedDict()

    def segment_clicks(
        self,
        image_path: Path,
        points: Sequence[PromptPoint],
        expected_hw: tuple[int, int],
    ) -> tuple[MaskArray, float]:
        if not points:
            raise StageExecutionError("segment_clicks requires at least one prompt point")
        started = time.perf_counter()
        stat = image_path.stat()
        key = (str(image_path.resolve()), stat.st_mtime_ns)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = _to_cpu_if_tensor(self._backend.encode_image(image_path))
            while len(self._cache) > self._max_cache_items:
                self._cache.popitem(last=False)
        raw_mask = self._backend.predict(self._cache[key], list(points))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return normalize_binary_mask(raw_mask, expected_hw), latency_ms


def track_vos(
    video_backend: VideoSamBackend,
    image_paths: dict[int, Path],
    ref_frame_idx: int,
    seed_mask: MaskArray,
    expected_hw: tuple[int, int],
    inference_state_device: str = "cpu",
) -> dict[int, MaskArray]:
    if ref_frame_idx not in image_paths:
        raise StageExecutionError(f"ref_frame_idx {ref_frame_idx} missing from image_paths")
    seed = normalize_binary_mask(seed_mask, expected_hw)
    raw = video_backend.propagate(image_paths, ref_frame_idx, seed, inference_state_device)
    return {
        idx: (seed.copy() if idx == ref_frame_idx else normalize_binary_mask(raw[idx], expected_hw))
        for idx in sorted(image_paths)
        if idx in raw or idx == ref_frame_idx
    }


def _neighbor_drift_bad(
    idx: int,
    pos: int,
    ordered: list[int],
    masks: dict[int, MaskArray],
    min_iou: float,
    max_area_ratio: float,
) -> bool:
    neighbors = [masks[ordered[n]] for n in (pos - 1, pos + 1) if 0 <= n < len(ordered)]
    if not neighbors:
        return False
    cur = masks[idx]
    ious = [mask_iou(cur, nb) for nb in neighbors]
    areas = [max(int((nb == 255).sum()), 1) for nb in neighbors]
    cur_area = int((cur == 255).sum())
    return max(ious) < min_iou or any((cur_area / a) > max_area_ratio for a in areas)


def filter_neighbor_iou(
    masks: dict[int, MaskArray],
    ref_frame_idx: int,
    min_iou: float = 0.25,
    max_area_ratio: float = 3.5,
) -> tuple[dict[int, MaskArray], set[int]]:
    """Tier 2: flag frames with sudden neighbor IoU collapse or area spikes relative to neighbors."""
    ordered = sorted(masks)
    flagged = {
        idx
        for pos, idx in enumerate(ordered)
        if idx != ref_frame_idx and _neighbor_drift_bad(idx, pos, ordered, masks, min_iou, max_area_ratio)
    }
    return {k: v.copy() for k, v in masks.items()}, flagged


def _load_depth_map(depth_dir: Path | None, frame_idx: int) -> FloatArray | None:
    if depth_dir is None:
        return None
    npy_path = depth_dir / f"{frame_idx:06d}.npy"
    return np.load(npy_path, allow_pickle=False).astype(np.float32) if npy_path.exists() else None


def _lift_foreground_points(
    mask: MaskArray,
    pose: NDArray[np.float64],
    intrinsics: dict[str, float],
    depth_map: FloatArray | None,
    max_samples: int = 256,
) -> NDArray[np.float64]:
    ys, xs = np.nonzero(mask == 255)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)
    step = max(1, len(xs) // max_samples)
    pts: list[NDArray[np.float64]] = []
    for x, y in zip(xs[::step], ys[::step]):
        d = float(depth_map[y, x]) if depth_map is not None else 2.5
        if not np.isfinite(d) or d <= 1e-3:
            continue
        origin, direction = unproject_pixel_to_ray(float(x), float(y), pose, intrinsics)
        z_cam = float((pose[:3, :3] @ direction)[2])
        if z_cam > 1e-4:
            pts.append(origin + direction * (d / z_cam))
    return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 3), dtype=np.float64)


def _classify_frame_visibility(
    pts_3d: NDArray[np.float64],
    pose: NDArray[np.float64],
    intrinsics: dict[str, float],
    hw: tuple[int, int],
    depth_map: FloatArray | None,
    occlusion_tol: float,
) -> tuple[str, tuple[float, float] | None]:
    h, w = hw
    visible_uvs: list[tuple[float, float]] = []
    occluded = 0
    for pt in pts_3d:
        u, v, z = project_world_to_image(pt, pose, intrinsics)
        if z <= 1e-4 or not (0.0 <= u < w and 0.0 <= v < h):
            continue
        visible_uvs.append((u, v))
        if depth_map is not None and z > float(depth_map[int(v), int(u)]) + occlusion_tol:
            occluded += 1
    if not visible_uvs:
        return "out_of_frustum", None
    if occluded > int(0.85 * len(visible_uvs)):
        return "occluded", None
    arr = np.asarray(visible_uvs, dtype=np.float64)
    return "visible", (float(arr[:, 0].mean()), float(arr[:, 1].mean()))


def lift_and_reproject_masks(
    masks: dict[int, MaskArray],
    manifest: SceneManifest,
    image_paths: dict[int, Path],
    depth_dir: Path | None,
    ref_frame_idx: int,
    sam_session: WarmSamSession,
    occlusion_tol: float = 0.25,
) -> tuple[dict[int, MaskArray], dict[int, str], set[int]]:
    """Tier 3: Lift reference mask to 3D, suppress OOF/occluded views, and re-prompt re-entries."""
    if ref_frame_idx not in masks or not np.any(masks[ref_frame_idx] == 255):
        return masks, {idx: "empty_ref" for idx in masks}, set()
    hw = (manifest.height, manifest.width)
    intrinsics = manifest.intrinsics.model_dump()
    frames_by_idx = {f.frame_idx: f for f in manifest.frames}
    pts_3d = _lift_foreground_points(
        masks[ref_frame_idx],
        np.asarray(frames_by_idx[ref_frame_idx].cam_from_world, dtype=np.float64),
        intrinsics,
        _load_depth_map(depth_dir, ref_frame_idx),
    )
    if len(pts_3d) == 0:
        return masks, {idx: "no_depth" for idx in masks}, set()

    updated = {k: v.copy() for k, v in masks.items()}
    visibility: dict[int, str] = {}
    recovered: set[int] = set()
    saw_hidden = False
    for idx in sorted(image_paths):
        pose = np.asarray(frames_by_idx[idx].cam_from_world, dtype=np.float64)
        vis, hint_uv = _classify_frame_visibility(pts_3d, pose, intrinsics, hw, _load_depth_map(depth_dir, idx), occlusion_tol)
        visibility[idx] = vis
        if vis != "visible":
            updated[idx] = np.zeros(hw, dtype=np.uint8)
            saw_hidden = True
        elif saw_hidden and not np.any(updated.get(idx, 0) == 255) and hint_uv is not None:
            re_mask, _ = sam_session.segment_clicks(image_paths[idx], [PromptPoint(hint_uv[0], hint_uv[1], 1)], hw)
            if np.any(re_mask == 255):
                updated[idx] = re_mask
                recovered.add(idx)
    return updated, visibility, recovered


class _HistogramReidEncoder:
    def embed_crop(self, rgb_crop: NDArray[np.uint8], mask_crop: MaskArray | None = None) -> FloatArray:
        hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask_crop, [16, 8], [0, 180, 0, 256]).astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(hist))
        return hist / norm if norm > 1e-8 else hist


def _crop_masked_rgb(image_path: Path, mask: MaskArray) -> tuple[NDArray[np.uint8], MaskArray] | None:
    ys, xs = np.nonzero(mask == 255)
    if len(xs) == 0:
        return None
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    mask_crop = np.where(mask[y0:y1, x0:x1] == 255, np.uint8(255), np.uint8(0)).astype(np.uint8)
    crop = rgb[y0:y1, x0:x1].copy()
    crop[mask_crop == 0] = 0
    return crop, mask_crop


def score_reid_crops(
    image_paths: dict[int, Path],
    masks: dict[int, MaskArray],
    ref_frame_idx: int,
    encoder: ReidEncoder | None = None,
    min_similarity: float = 0.65,
) -> tuple[dict[int, float], set[int]]:
    """Tier 4: crop masked detections and reject ID switches via Re-ID cosine similarity."""
    reid = encoder or _HistogramReidEncoder()
    ref_data = _crop_masked_rgb(image_paths[ref_frame_idx], masks[ref_frame_idx])
    if ref_data is None:
        return {}, set()
    ref_emb = reid.embed_crop(*ref_data)
    scores: dict[int, float] = {}
    rejected: set[int] = set()
    for idx, mask in sorted(masks.items()):
        crop_data = _crop_masked_rgb(image_paths[idx], mask)
        if crop_data is None:
            continue
        emb = reid.embed_crop(*crop_data)
        sim = float(np.dot(ref_emb, emb) / max(float(np.linalg.norm(ref_emb) * np.linalg.norm(emb)), 1e-8))
        scores[idx] = sim
        if idx != ref_frame_idx and sim < min_similarity:
            rejected.add(idx)
    return scores, rejected


def project_3dgs_mask_hints(
    points_world: NDArray[Any],
    frame: FrameRecord,
    intrinsics: dict[str, float],
    hw: tuple[int, int],
) -> MaskArray:
    """Tier 5: project reference 3D foreground Gaussian subset back to candidate 2D mask hints."""
    h, w = hw
    hint = np.zeros((h, w), dtype=np.uint8)
    pose = np.asarray(frame.cam_from_world, dtype=np.float64)
    for pt in np.asarray(points_world, dtype=np.float64):
        u, v, depth = project_world_to_image(pt, pose, intrinsics)
        if depth > 1e-4 and 0.0 <= u < w and 0.0 <= v < h:
            hint[int(v), int(u)] = 255
    return hint


def apply_manual_mask_overrides(
    masks: dict[int, MaskArray],
    overrides: Sequence[ManualOverride],
    expected_hw: tuple[int, int],
) -> tuple[dict[int, MaskArray], set[int]]:
    updated = {k: v.copy() for k, v in masks.items()}
    skipped: set[int] = set()
    for item in overrides:
        if item.skip_frame:
            skipped.add(item.frame_idx)
            updated[item.frame_idx] = np.zeros(expected_hw, dtype=np.uint8)
        elif item.mask_path is not None:
            updated[item.frame_idx] = load_binary_mask_png(item.mask_path, expected_hw)
            skipped.discard(item.frame_idx)
    return updated, skipped


def write_tracking_report(output_path: Path, report: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return output_path


def run_tracking_pipeline(
    manifest: SceneManifest,
    image_paths: dict[int, Path],
    ref_frame_idx: int,
    points: Sequence[PromptPoint],
    sam_session: WarmSamSession,
    video_backend: VideoSamBackend,
    output_dir: Path,
    depth_dir: Path | None = None,
    reid_encoder: ReidEncoder | None = None,
    overrides: Sequence[ManualOverride] = (),
) -> tuple[dict[int, Path], dict[str, Any]]:
    """Run SAM 2.1 click refinement and 5-tier propagation, returning mask PNGs and metrics."""
    hw = (manifest.height, manifest.width)
    seed_mask, click_ms = sam_session.segment_clicks(image_paths[ref_frame_idx], points, hw)
    raw_masks = track_vos(video_backend, image_paths, ref_frame_idx, seed_mask, hw, inference_state_device="cpu")
    _, iou_flagged = filter_neighbor_iou(raw_masks, ref_frame_idx)
    reprojected, visibility, recovered = lift_and_reproject_masks(raw_masks, manifest, image_paths, depth_dir, ref_frame_idx, sam_session)
    for bad_idx in iou_flagged - recovered:
        reprojected[bad_idx] = np.zeros(hw, dtype=np.uint8)
    reid_scores, reid_rejected = score_reid_crops(image_paths, reprojected, ref_frame_idx, reid_encoder)
    for rej in reid_rejected:
        reprojected[rej] = np.zeros(hw, dtype=np.uint8)
    final_masks, skipped = apply_manual_mask_overrides(reprojected, overrides, hw)
    masks_dir = output_dir / "masks"
    saved = {idx: save_binary_mask_png(mask, masks_dir / f"{idx:06d}.png", hw) for idx, mask in sorted(final_masks.items())}
    stats = {
        "ref_frame_idx": ref_frame_idx, "click_latency_ms": round(click_ms, 2),
        "total_frames": len(image_paths), "iou_flagged": sorted(iou_flagged),
        "recovered_reentries": sorted(recovered), "reid_rejected": sorted(reid_rejected),
        "skipped_frames": sorted(skipped), "reid_scores": reid_scores, "visibility": visibility,
    }
    write_tracking_report(output_dir / "tracking_report.json", stats)
    return saved, stats


def _resolve_sam_device(device: str, torch_mod: Any) -> str:
    if not torch_mod.cuda.is_available() or device == "cpu":
        return "cpu"
    if device.startswith("cuda:"):
        ordinal = int(device.split(":")[1])
        return device if ordinal < torch_mod.cuda.device_count() else "cuda:0"
    return "cuda:0"


class _Sam2Interactive:
    def __init__(self, image_predictor: Any) -> None:
        self._pred = image_predictor

    def encode_image(self, image_path: Path) -> dict[str, Any]:
        with Image.open(image_path) as img:
            self._pred.set_image(np.asarray(img.convert("RGB"), dtype=np.uint8))
        return {"features": self._pred._features, "orig_hw": self._pred._orig_hw}

    def predict(self, embedding: Any, points: list[PromptPoint]) -> NDArray[Any]:
        self._pred._features = embedding["features"]
        self._pred._orig_hw = embedding["orig_hw"]
        self._pred._is_image_set = True
        coords = np.array([[p.x, p.y] for p in points], dtype=np.float32)
        labels = np.array([p.label for p in points], dtype=np.int32)
        masks, _, _ = self._pred.predict(point_coords=coords, point_labels=labels, multimask_output=False)
        return np.asarray(masks[0])


class _Sam2Video:
    def __init__(self, video_predictor: Any) -> None:
        self._pred = video_predictor

    def propagate(
        self,
        image_paths: dict[int, Path],
        ref_frame_idx: int,
        seed_mask: MaskArray,
        inference_state_device: str,
    ) -> dict[int, MaskArray]:
        ordered_keys = sorted(image_paths)
        idx_map = {pos: idx for pos, idx in enumerate(ordered_keys)}
        rev_map = {idx: pos for pos, idx in idx_map.items()}
        with tempfile.TemporaryDirectory(prefix="sam2_frames_") as tmp_dir:
            for pos, orig_idx in idx_map.items():
                with Image.open(image_paths[orig_idx]) as img:
                    img.convert("RGB").save(Path(tmp_dir) / f"{pos:06d}.jpg", format="JPEG", quality=95)
            state = self._pred.init_state(
                video_path=tmp_dir,
                offload_video_to_cpu=(inference_state_device == "cpu"),
                offload_state_to_cpu=(inference_state_device == "cpu"),
            )
            self._pred.add_new_mask(state, frame_idx=rev_map[ref_frame_idx], obj_id=1, mask=(seed_mask == 255))
            return {
                idx_map[pos]: np.asarray(logits[0] > 0.0, dtype=np.uint8) * np.uint8(255)
                for pos, _obj_ids, logits in self._pred.propagate_in_video(state)
            }


def load_sam2_backends(model_cfg: str, checkpoint: str, device: str = "cuda:1") -> tuple[InteractiveSamBackend, VideoSamBackend]:
    """Lazy loader for official SAM 2.1 checkpoints on cuda:1 with safe CPU offload for VOS."""
    try:
        torch = importlib.import_module("torch")
        sam2_build = importlib.import_module("sam2.build_sam")
        sam2_pred = importlib.import_module("sam2.sam2_image_predictor")
    except ImportError as exc:
        raise StageExecutionError("s05_track requires torch and sam2 when loading live SAM 2.1 weights") from exc
    resolved = _resolve_sam_device(device, torch)
    sam_model = sam2_build.build_sam2(model_cfg, checkpoint, device=resolved)
    video_predictor = sam2_build.build_sam2_video_predictor(model_cfg, checkpoint, device=resolved)
    return _Sam2Interactive(sam2_pred.SAM2ImagePredictor(sam_model)), _Sam2Video(video_predictor)
