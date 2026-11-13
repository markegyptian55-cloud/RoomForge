from __future__ import annotations

import importlib
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import binary_dilation

from room3d_worker.common.errors import GeometryError, StageExecutionError
from room3d_worker.common.manifest import FrameRecord, SceneManifest, load_manifest
from room3d_worker.pipeline.s04_export import ExportRequest, export_scene
from room3d_worker.pipeline.s05_track import load_binary_mask_png

ArrayF32 = NDArray[np.float32]
ArrayBool = NDArray[np.bool_]
ArrayU8 = NDArray[np.uint8]
RebuildMode = Literal["masked_finetune", "from_scratch"]


@dataclass(frozen=True)
class RebuildRequest:
    base_dir: Path
    edited_frames_dir: Path
    masks_dir: Path
    output_dir: Path
    project_id: str
    version_id: str
    parent_version_id: str
    mode: RebuildMode = "masked_finetune"
    fine_tune_steps: int = 5000
    from_scratch_steps: int = 15000
    roi_pad_px: int = 8
    min_mask_hits: int = 1
    exclude_frames: tuple[int, ...] = ()
    device: str = "cuda:1"
    seed: int = 0
    test_every: int = 0

    def __post_init__(self) -> None:
        if self.device not in {"cuda:0", "cuda:1", "cpu"}:
            raise ValueError("Week 2 3DGS rebuild device must be cuda:1, cuda:0, or cpu")
        if self.mode not in {"masked_finetune", "from_scratch"}:
            raise ValueError(f"Unsupported rebuild mode: {self.mode}")
        if self.fine_tune_steps <= 0 or self.from_scratch_steps <= 0 or self.roi_pad_px < 0 or self.min_mask_hits < 1:
            raise ValueError("Invalid optimization step or ROI padding configuration")


@dataclass(frozen=True)
class GaussianState:
    points_xyz: ArrayF32
    colors: ArrayU8
    opacity: ArrayF32
    scales: ArrayF32
    extra: dict[str, Any] = field(default_factory=dict)


def _resolve_frame_file(folder: Path, frame_idx: int) -> Path | None:
    for name in (f"{frame_idx:06d}.png", f"{frame_idx:05d}.png", f"frame_{frame_idx:06d}.png"):
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def _read_ply_vertices(path: Path) -> tuple[ArrayF32, ArrayU8]:
    raw = path.read_bytes()
    marker = b"end_header\n"
    idx = raw.find(marker)
    if idx < 0:
        raise StageExecutionError(f"Malformed PLY header: {path}")
    vertex_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    data = np.frombuffer(raw[idx + len(marker) :], dtype=vertex_dtype)
    points = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    colors = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.uint8)
    return points, colors


def load_base_scene(base_dir: Path) -> tuple[SceneManifest, GaussianState]:
    manifest = load_manifest(base_dir / "manifest.json")
    npz_path = base_dir / "gs_state.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as archive:
            return manifest, GaussianState(
                points_xyz=np.asarray(archive["points_xyz"], dtype=np.float32),
                colors=np.asarray(archive["colors"], dtype=np.uint8),
                opacity=np.asarray(archive["opacity"], dtype=np.float32),
                scales=np.asarray(archive["scales"], dtype=np.float32),
            )
    ply_path = base_dir / (manifest.artifacts.ply_relpath or "scene.ply")
    if not ply_path.exists():
        raise StageExecutionError(f"Base scene artifacts missing under {base_dir}")
    points, colors = _read_ply_vertices(ply_path)
    return manifest, GaussianState(
        points_xyz=points, colors=colors, opacity=np.full(len(points), 0.9, dtype=np.float32), scales=np.full((len(points), 3), 0.01, dtype=np.float32)
    )


def save_gaussian_state(output_dir: Path, state: GaussianState) -> None:
    np.savez_compressed(
        output_dir / "gs_state.npz",
        points_xyz=state.points_xyz.astype(np.float32),
        colors=state.colors.astype(np.uint8),
        opacity=state.opacity.astype(np.float32),
        scales=state.scales.astype(np.float32),
    )


def _dilate_mask(mask_uint8: ArrayU8, pad_px: int) -> ArrayBool:
    binary = mask_uint8 == 255
    if pad_px <= 0 or not binary.any():
        return binary
    size = 2 * pad_px + 1
    return np.asarray(binary_dilation(binary, structure=np.ones((size, size), dtype=bool)), dtype=bool)


def compute_roi_gaussian_mask(
    points_xyz: ArrayF32,
    manifest: SceneManifest,
    masks_dir: Path,
    skip_frames: set[int] | None = None,
    roi_pad_px: int = 8,
    min_hits: int = 1,
) -> ArrayBool:
    """Identify Gaussians projecting inside dilated foreground masks across active views."""
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise GeometryError("points_xyz must have shape (N, 3)")
    skipped = skip_frames or set()
    intrinsics = manifest.intrinsics.model_dump()
    hits = np.zeros(len(points), dtype=np.int32)
    used_masks = 0
    for frame in manifest.frames:
        mask_path = _resolve_frame_file(masks_dir, frame.frame_idx)
        if frame.excluded or frame.frame_idx in skipped or mask_path is None:
            continue
        dilated = _dilate_mask(load_binary_mask_png(mask_path, (manifest.height, manifest.width)), roi_pad_px)
        used_masks += 1
        pose = np.asarray(frame.cam_from_world, dtype=np.float64)
        cam = (pose[:3, :3] @ points.T) + pose[:3, 3:4]
        z = cam[2]
        valid = z > 1e-4
        u = np.rint(intrinsics["fx"] * (cam[0, valid] / z[valid]) + intrinsics["cx"]).astype(np.int64)
        v = np.rint(intrinsics["fy"] * (cam[1, valid] / z[valid]) + intrinsics["cy"]).astype(np.int64)
        in_bounds = (u >= 0) & (u < manifest.width) & (v >= 0) & (v < manifest.height)
        idx = np.flatnonzero(valid)[in_bounds]
        hits[idx[dilated[v[in_bounds], u[in_bounds]]]] += 1
    if used_masks == 0:
        raise StageExecutionError("No active binary masks found for Week 2 ROI computation")
    return hits >= min_hits


def compute_masked_l1_loss(
    rendered_rgb: ArrayF32,
    target_rgb: ArrayF32,
    mask_uint8: ArrayU8,
    unmasked_weight: float = 0.0,
) -> float:
    rendered = np.asarray(rendered_rgb, dtype=np.float32)
    target = np.asarray(target_rgb, dtype=np.float32)
    mask = np.asarray(mask_uint8, dtype=np.uint8)
    if rendered.shape != target.shape or rendered.ndim != 3 or mask.shape != rendered.shape[:2]:
        raise GeometryError("rendered, target, and mask shapes must agree")
    if not (np.isfinite(rendered).all() and np.isfinite(target).all()):
        raise GeometryError("rendered and target RGB tensors must be finite")
    fg = mask == 255
    fg_loss = float(np.abs(rendered[fg] - target[fg]).mean()) if fg.any() else 0.0
    if unmasked_weight <= 0.0 or (~fg).sum() == 0:
        return fg_loss
    return fg_loss + unmasked_weight * float(np.abs(rendered[~fg] - target[~fg]).mean())


def assert_frozen_gaussians_unchanged(base: GaussianState, updated: GaussianState, frozen_mask: ArrayBool) -> float:
    if not frozen_mask.any():
        return 0.0
    drifts = [
        float(np.max(np.abs(getattr(updated, key)[frozen_mask].astype(np.float64) - getattr(base, key)[frozen_mask].astype(np.float64))))
        for key in ("points_xyz", "colors", "opacity", "scales")
    ]
    max_drift = max(drifts, default=0.0)
    if max_drift != 0.0:
        raise StageExecutionError(f"Masked finetune mutated frozen background Gaussians: drift={max_drift}")
    return max_drift


def _load_training_views(
    request: RebuildRequest, manifest: SceneManifest, skip_frames: set[int]
) -> list[tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.bool_] | None]]:
    views: list[tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.bool_] | None]] = []
    for frame in manifest.frames:
        if frame.excluded or frame.frame_idx in skip_frames:
            continue
        rgb_path = _resolve_frame_file(request.edited_frames_dir, frame.frame_idx)
        if rgb_path is None:
            continue
        with Image.open(rgb_path) as img:
            rgb = (np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)
        mask_path = _resolve_frame_file(request.masks_dir, frame.frame_idx)
        fg = load_binary_mask_png(mask_path, (manifest.height, manifest.width)) == 255 if mask_path is not None else None
        views.append((np.asarray(frame.cam_from_world, dtype=np.float64), rgb, fg))
    if not views:
        raise StageExecutionError("No edited RGB views found under edited_frames_dir for 3DGS rebuild")
    return views


def _build_k_matrix(manifest: SceneManifest, torch_mod: Any, device: str) -> Any:
    intr = manifest.intrinsics
    k_mat = [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]]
    return torch_mod.tensor(k_mat, dtype=torch_mod.float32, device=device).unsqueeze(0)


def _compose_scene_tensors(
    base_state: GaussianState, roi_idx: Any, bg_idx: Any, active_vars: tuple[Any, ...], bg_vars: tuple[Any, ...], torch_mod: Any
) -> tuple[Any, Any, Any, Any]:
    n_pts = len(base_state.points_xyz)
    out: list[Any] = []
    for act, bg in zip(active_vars, bg_vars, strict=True):
        buf = torch_mod.empty((n_pts, *act.shape[1:]), dtype=act.dtype, device=act.device)
        buf[roi_idx] = act
        if len(bg_idx) > 0:
            buf[bg_idx] = bg
        out.append(buf)
    return out[0], out[1], out[2], out[3]


def _run_opt_loop(
    request: RebuildRequest,
    manifest: SceneManifest,
    base_state: GaussianState,
    views: list[tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.bool_] | None]],
    roi_mask: ArrayBool,
    torch_mod: Any,
    rasterize: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    dev = request.device
    roi_idx = torch_mod.as_tensor(np.flatnonzero(roi_mask), dtype=torch_mod.long, device=dev)
    bg_idx = torch_mod.as_tensor(np.flatnonzero(~roi_mask), dtype=torch_mod.long, device=dev)
    pts_t = torch_mod.as_tensor(base_state.points_xyz, dtype=torch_mod.float32, device=dev)
    cols_t = torch_mod.as_tensor(base_state.colors.astype(np.float32) / 255.0, dtype=torch_mod.float32, device=dev)
    op_t = torch_mod.as_tensor(base_state.opacity, dtype=torch_mod.float32, device=dev)
    sc_t = torch_mod.as_tensor(base_state.scales, dtype=torch_mod.float32, device=dev)
    active = tuple(t[roi_idx].detach().clone().requires_grad_(True) for t in (pts_t, cols_t, op_t, sc_t))
    frozen = tuple(t[bg_idx].detach() for t in (pts_t, cols_t, op_t, sc_t))
    quats = torch_mod.zeros((len(pts_t), 4), dtype=torch_mod.float32, device=dev)
    quats[:, 0] = 1.0
    quats = quats.detach()
    ks = _build_k_matrix(manifest, torch_mod, dev)
    opt = torch_mod.optim.Adam(list(active), lr=5e-3)
    steps = request.fine_tune_steps if request.mode == "masked_finetune" else request.from_scratch_steps
    for step in range(steps):
        pose_np, tgt_np, fg_np = views[step % len(views)]
        opt.zero_grad()
        m_all, c_all, o_all, s_all = _compose_scene_tensors(base_state, roi_idx, bg_idx, active, frozen, torch_mod)
        viewmat = torch_mod.as_tensor(pose_np, dtype=torch_mod.float32, device=dev).unsqueeze(0)
        rendered, *_ = rasterize(
            means=m_all, quats=quats, scales=s_all.clamp(min=1e-5), opacities=o_all.clamp(0.0, 1.0),
            colors=c_all.clamp(0.0, 1.0), viewmats=viewmat, Ks=ks, width=manifest.width, height=manifest.height,
        )
        pred, target = rendered[0], torch_mod.as_tensor(tgt_np, dtype=torch_mod.float32, device=dev)
        if request.mode == "masked_finetune" and fg_np is not None and fg_np.any():
            fg_mask = torch_mod.as_tensor(fg_np, dtype=torch_mod.bool, device=dev)
            loss = (pred[fg_mask] - target[fg_mask]).abs().mean()
        else:
            loss = (pred - target).abs().mean()
        loss.backward()
        opt.step()
        del rendered, pred, target, loss, viewmat
    return roi_idx, bg_idx, active, frozen, torch_mod, quats


def _default_gpu_rebuild(
    request: RebuildRequest,
    manifest: SceneManifest,
    base_state: GaussianState,
    roi_mask: ArrayBool,
) -> GaussianState:
    """Run differentiable masked gsplat optimization where only ROI Gaussians receive gradients."""
    try:
        torch = importlib.import_module("torch")
        gsplat = importlib.import_module("gsplat")
    except ImportError as exc:
        raise StageExecutionError("s07_rebuild requires torch and gsplat when no backend is injected") from exc
    if request.device != "cpu" and not torch.cuda.is_available():
        raise StageExecutionError("s07_rebuild requires CUDA GPU or explicit cpu device")
    rasterize = getattr(gsplat, "rasterization", None)
    if rasterize is None:
        raise StageExecutionError("installed gsplat does not expose rasterization")
    torch.manual_seed(request.seed)
    views = _load_training_views(request, manifest, set(request.exclude_frames))
    roi_idx, bg_idx, active, frozen, torch_mod, _ = _run_opt_loop(request, manifest, base_state, views, roi_mask, torch, rasterize)
    with torch_mod.no_grad():
        m_f, c_f, o_f, s_f = _compose_scene_tensors(base_state, roi_idx, bg_idx, active, frozen, torch_mod)
        pts_out = m_f.detach().cpu().numpy().astype(np.float32)
        cols_out = np.rint(c_f.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
        op_out = o_f.detach().cpu().numpy().clip(0.0, 1.0).astype(np.float32)
        sc_out = s_f.detach().cpu().numpy().clip(min=1e-5).astype(np.float32)
    return GaussianState(points_xyz=pts_out, colors=cols_out, opacity=op_out, scales=sc_out)


def _apply_frame_exclusions(frames: list[FrameRecord], skip_frames: set[int]) -> list[dict[str, Any]]:
    return [{**frame.model_dump(), "excluded": True if frame.frame_idx in skip_frames else frame.excluded} for frame in frames]


def _verify_pose_invariance(base: SceneManifest, rebuilt: SceneManifest) -> float:
    if rebuilt.intrinsics != base.intrinsics or rebuilt.width != base.width or rebuilt.height != base.height:
        raise StageExecutionError("Week 2 rebuild altered canonical camera intrinsics or dimensions")
    t_rebuilt = np.asarray(rebuilt.T_viewer_from_colmap or np.eye(4), dtype=np.float64)
    t_base = np.asarray(base.T_viewer_from_colmap or np.eye(4), dtype=np.float64)
    if not np.allclose(t_rebuilt, t_base, atol=1e-7):
        raise StageExecutionError("Week 2 rebuild altered canonical viewer transform")
    max_drift = max(
        float(np.max(np.abs(np.asarray(r.cam_from_world) - np.asarray(b.cam_from_world))))
        for b, r in zip(base.frames, rebuilt.frames, strict=True)
    )
    if max_drift > 1e-7:
        raise StageExecutionError(f"Week 2 rebuild drifted camera poses: max_drift={max_drift}")
    return max_drift


def _validate_rebuild_preconditions(request: RebuildRequest, base_manifest: SceneManifest) -> None:
    if request.parent_version_id != base_manifest.version_id:
        raise StageExecutionError(f"parent_version_id '{request.parent_version_id}' != base version '{base_manifest.version_id}'")
    if not (request.base_dir / "sparse" / "0").exists():
        raise StageExecutionError(f"Missing canonical sparse/0 directory under {request.base_dir}")


def _export_rebuilt_scene(
    request: RebuildRequest, base_manifest: SceneManifest, updated: GaussianState, skip_set: set[int]
) -> SceneManifest:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(request.base_dir / "sparse", request.output_dir / "sparse", dirs_exist_ok=True)
    save_gaussian_state(request.output_dir, updated)
    return export_scene(
        ExportRequest(
            output_dir=request.output_dir,
            project_id=request.project_id,
            version_id=request.version_id,
            parent_version_id=request.parent_version_id,
            width=base_manifest.width,
            height=base_manifest.height,
            intrinsics=base_manifest.intrinsics.model_dump(),
            frames=_apply_frame_exclusions(base_manifest.frames, skip_set),
            test_every=request.test_every,
            pose_backend=base_manifest.pose_backend,
            T_viewer_from_colmap=base_manifest.T_viewer_from_colmap,
        ),
        points_xyz=updated.points_xyz,
        opacity=updated.opacity,
        scales=updated.scales,
        colors=updated.colors,
    )


def rebuild_scene(
    request: RebuildRequest,
    backend: Callable[[RebuildRequest, SceneManifest, GaussianState, ArrayBool], GaussianState] | None = None,
) -> dict[str, Any]:
    """Execute Week 2 localized 3DGS fine-tune or from-scratch ablation without touching COLMAP poses."""
    started = time.perf_counter()
    base_manifest, base_state = load_base_scene(request.base_dir)
    _validate_rebuild_preconditions(request, base_manifest)
    skip_set = set(request.exclude_frames)
    if request.mode == "masked_finetune":
        roi_mask = compute_roi_gaussian_mask(
            base_state.points_xyz, base_manifest, request.masks_dir, skip_set, request.roi_pad_px, request.min_mask_hits
        )
    else:
        roi_mask = np.ones(len(base_state.points_xyz), dtype=bool)
    updated = (backend or _default_gpu_rebuild)(request, base_manifest, base_state, roi_mask)
    max_unmasked_drift = assert_frozen_gaussians_unchanged(base_state, updated, ~roi_mask) if request.mode == "masked_finetune" else 0.0
    rebuilt_manifest = _export_rebuilt_scene(request, base_manifest, updated, skip_set)
    max_pose_drift = _verify_pose_invariance(base_manifest, rebuilt_manifest)
    steps = request.fine_tune_steps if request.mode == "masked_finetune" else request.from_scratch_steps
    metrics = {
        "mode": request.mode, "steps": steps, "runtime_sec": round(time.perf_counter() - started, 4),
        "total_gaussian_count": int(len(updated.points_xyz)), "roi_gaussian_count": int(roi_mask.sum()),
        "frozen_gaussian_count": int((~roi_mask).sum()) if request.mode == "masked_finetune" else 0,
        "excluded_frame_count": int(sum(1 for f in rebuilt_manifest.frames if f.excluded)),
        "max_pose_drift": max_pose_drift, "max_unmasked_gaussian_drift": max_unmasked_drift,
        "parent_version_id": request.parent_version_id, "version_id": request.version_id,
    }
    (request.output_dir / "rebuild_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
