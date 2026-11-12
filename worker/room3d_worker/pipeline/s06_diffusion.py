from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from room3d_worker.common.errors import StageExecutionError

RGBImage = NDArray[np.uint8]
MaskArray = NDArray[np.uint8]
MAX_GRID_PIXELS = 1024 * 1024
FORBIDDEN_MODEL_PATTERNS = ("9b", "qwen")


@dataclass(frozen=True)
class DiffusionConfig:
    model_id: str = "black-forest-labs/FLUX.2-klein-base-4B"
    fallback_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    device: str = "cuda:0"
    num_inference_steps: int = 8
    guidance_scale: float = 3.5
    strength: float = 0.85
    feather_px: int = 3
    use_sequential_offload: bool = False
    grid_2x2: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.strength <= 1.0:
            raise ValueError("strength must be in (0.0, 1.0]")
        if self.num_inference_steps < 1 or self.guidance_scale < 0.0 or self.feather_px < 0:
            raise ValueError("invalid diffusion step, guidance, or feather setting")
        if not (self.device == "cpu" or self.device.startswith("cuda")):
            raise ValueError(f"unsupported device string: {self.device}")
        _assert_allowed_model_id(self.model_id)
        _assert_allowed_model_id(self.fallback_model_id)


class InpaintBackend(Protocol):
    def inpaint(
        self,
        image: RGBImage,
        mask: MaskArray,
        prompt: str,
        seed: int,
        image_reference: RGBImage | None = None,
    ) -> RGBImage:
        ...


class _DiffusersBackend:
    def __init__(self, pipe: Any, torch_mod: Any, config: DiffusionConfig, supports_ref: bool, exec_device: str) -> None:
        self._pipe = pipe
        self._torch = torch_mod
        self._config = config
        self._supports_ref = supports_ref
        self._exec_device = exec_device

    def inpaint(
        self,
        image: RGBImage,
        mask: MaskArray,
        prompt: str,
        seed: int,
        image_reference: RGBImage | None = None,
    ) -> RGBImage:
        generator = self._torch.Generator(device=self._exec_device).manual_seed(int(seed))
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": Image.fromarray(image),
            "mask_image": Image.fromarray(mask),
            "num_inference_steps": self._config.num_inference_steps,
            "guidance_scale": self._config.guidance_scale,
            "strength": self._config.strength,
            "generator": generator,
        }
        if self._supports_ref and image_reference is not None:
            kwargs["image_reference"] = Image.fromarray(image_reference)
        result = self._pipe(**kwargs).images[0]
        return np.asarray(result.convert("RGB"), dtype=np.uint8)


def _assert_allowed_model_id(model_id: str) -> None:
    lower = model_id.lower()
    if any(token in lower for token in FORBIDDEN_MODEL_PATTERNS):
        raise StageExecutionError(f"Model '{model_id}' violates commercial license guardrails")


def resolve_t4_dtype(compute_capability: tuple[int, int] | None) -> str:
    """Enforce float32 on CPU, float16 on pre-Ampere (Turing T4 sm_75), float16/bfloat16 on >= sm_80."""
    if compute_capability is None:
        return "float32"
    if compute_capability[0] < 8:
        return "float16"
    return "bfloat16"


def _validate_rgb_and_mask(image: RGBImage, binary_mask: MaskArray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise StageExecutionError("image must be HxWx3 uint8 RGB")
    if binary_mask.ndim != 2 or binary_mask.dtype != np.uint8:
        raise StageExecutionError("binary_mask must be a 2D uint8 array")
    if image.shape[:2] != binary_mask.shape[:2]:
        raise StageExecutionError(f"mask shape {binary_mask.shape} != image shape {image.shape[:2]}")
    if np.any((binary_mask != 0) & (binary_mask != 255)):
        raise StageExecutionError("binary_mask must contain strictly 0 or 255")


def pad_to_multiple(image: RGBImage, binary_mask: MaskArray, multiple: int = 16) -> tuple[RGBImage, MaskArray, tuple[int, int]]:
    _validate_rgb_and_mask(image, binary_mask)
    h, w = image.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, binary_mask, (h, w)
    padded_img = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    padded_mask = np.pad(binary_mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return np.asarray(padded_img, dtype=np.uint8), np.asarray(padded_mask, dtype=np.uint8), (h, w)


def _pad_rgb_only(image: RGBImage, target_hw: tuple[int, int]) -> RGBImage:
    h, w = image.shape[:2]
    th, tw = target_hw
    if (h, w) == (th, tw):
        return image
    return np.asarray(np.pad(image, ((0, th - h), (0, tw - w), (0, 0)), mode="edge"), dtype=np.uint8)


def composite_strict_mask(orig_rgb: RGBImage, edited_rgb: RGBImage, binary_mask: MaskArray, feather_px: int = 3) -> RGBImage:
    """Composite edited pixels strictly inside the mask; outside pixels keep orig_rgb bitwise identical."""
    _validate_rgb_and_mask(orig_rgb, binary_mask)
    if edited_rgb.shape != orig_rgb.shape or edited_rgb.dtype != np.uint8:
        raise StageExecutionError("edited_rgb must match orig_rgb HxWx3 uint8")
    out = orig_rgb.copy()
    fg = binary_mask == 255
    if not np.any(fg):
        return out
    if feather_px <= 0:
        out[fg] = edited_rgb[fg]
        return out
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather_px + 1, 2 * feather_px + 1))
    inner = cv2.erode(binary_mask, kernel, iterations=1) == 255
    alpha = np.where(fg, 0.5, 0.0).astype(np.float32)
    alpha[inner] = 1.0
    blended = np.rint((1.0 - alpha[..., None]) * orig_rgb + alpha[..., None] * edited_rgb).clip(0, 255).astype(np.uint8)
    out[fg] = blended[fg]
    return out


def assert_unmasked_unchanged(orig_rgb: RGBImage, composited_rgb: RGBImage, binary_mask: MaskArray) -> int:
    bg = binary_mask == 0
    if not np.any(bg):
        return 0
    max_diff = int(np.max(np.abs(composited_rgb[bg].astype(np.int32) - orig_rgb[bg].astype(np.int32))))
    if max_diff != 0:
        raise StageExecutionError(f"Unmasked background drift detected: max_abs_diff={max_diff}")
    return max_diff


def _load_rgb(path: Path) -> RGBImage:
    if not path.exists():
        raise StageExecutionError(f"Missing frame image: {path}")
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path, hw: tuple[int, int]) -> MaskArray:
    if not path.exists():
        return np.zeros(hw, dtype=np.uint8)
    with Image.open(path) as img:
        img.load()
        if img.mode != "L":
            raise StageExecutionError(f"Mask must be mode 'L', got '{img.mode}': {path}")
        arr = np.asarray(img, dtype=np.uint8)
    if arr.shape != hw or np.any((arr != 0) & (arr != 255)):
        raise StageExecutionError(f"Invalid mask shape or pixel values: {path}")
    return arr


def _run_padded_inpaint(
    orig: RGBImage,
    mask: MaskArray,
    prompt: str,
    seed: int,
    backend: InpaintBackend,
    feather_px: int,
    image_reference: RGBImage | None = None,
) -> RGBImage:
    padded_img, padded_mask, (h, w) = pad_to_multiple(orig, mask, multiple=16)
    padded_ref = _pad_rgb_only(image_reference, padded_img.shape[:2]) if image_reference is not None else None
    raw_out = backend.inpaint(padded_img, padded_mask, prompt, seed=seed, image_reference=padded_ref)[:h, :w]
    composited = composite_strict_mask(orig, raw_out, mask, feather_px=feather_px)
    assert_unmasked_unchanged(orig, composited, mask)
    return composited


def generate_first_frame_seeds(
    ref_image_path: Path,
    ref_mask_path: Path,
    prompt: str,
    seeds: list[int],
    output_dir: Path,
    backend: InpaintBackend,
    config: DiffusionConfig | None = None,
) -> list[dict[str, Any]]:
    cfg = config or DiffusionConfig()
    orig = _load_rgb(ref_image_path)
    mask = _load_mask(ref_mask_path, orig.shape[:2])
    output_dir.mkdir(parents=True, exist_ok=True)
    previews: list[dict[str, Any]] = []
    for seed in seeds:
        comp = _run_padded_inpaint(orig, mask, prompt, int(seed), backend, cfg.feather_px)
        out_path = output_dir / f"seed_{int(seed)}.png"
        Image.fromarray(comp).save(out_path, format="PNG")
        previews.append({"seed_id": f"seed_{int(seed)}", "seed": int(seed), "path": str(out_path)})
    return previews


def _tile_2x2(frames: list[RGBImage | MaskArray]) -> RGBImage | MaskArray:
    h, w = frames[0].shape[:2]
    shape = (2 * h, 2 * w, 3) if frames[0].ndim == 3 else (2 * h, 2 * w)
    canvas = np.zeros(shape, dtype=np.uint8)
    for idx, tile in enumerate(frames):
        r, c = divmod(idx, 2)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    return canvas


def _save_edited_frame(edited_dir: Path, idx: int, comp: RGBImage, orig: RGBImage, mask: MaskArray) -> int:
    Image.fromarray(comp).save(edited_dir / f"{idx:06d}.png", format="PNG")
    return assert_unmasked_unchanged(orig, comp, mask)


def _propagate_sequential(
    indices: list[int],
    frame_paths: dict[int, Path],
    mask_paths: dict[int, Path],
    ref_edit: RGBImage,
    prompt: str,
    seed: int,
    edited_dir: Path,
    backend: InpaintBackend,
    feather_px: int,
) -> tuple[int, int]:
    edited_count, max_diff = 0, 0
    for idx in indices:
        orig = _load_rgb(frame_paths[idx])
        mask = _load_mask(mask_paths.get(idx, edited_dir / "_missing.png"), orig.shape[:2])
        if np.any(mask):
            comp = _run_padded_inpaint(orig, mask, prompt, seed, backend, feather_px, image_reference=ref_edit)
            edited_count += 1
        else:
            comp = orig
        max_diff = max(max_diff, _save_edited_frame(edited_dir, idx, comp, orig, mask))
    return edited_count, max_diff


def _propagate_grid_batch(
    batch: list[int],
    frame_paths: dict[int, Path],
    mask_paths: dict[int, Path],
    ref_edit: RGBImage,
    prompt: str,
    seed: int,
    edited_dir: Path,
    backend: InpaintBackend,
    feather_px: int,
) -> tuple[int, int]:
    origs = [_load_rgb(frame_paths[idx]) for idx in batch]
    h, w = origs[0].shape[:2]
    masks = [_load_mask(mask_paths.get(idx, edited_dir / "_missing.png"), (h, w)) for idx in batch]
    if not any(np.any(m) for m in masks):
        max_diff = max((_save_edited_frame(edited_dir, idx, o, o, m) for idx, o, m in zip(batch, origs, masks)), default=0)
        return 0, max_diff
    if len(batch) < 4 or (2 * h) * (2 * w) > MAX_GRID_PIXELS or any(o.shape[:2] != (h, w) for o in origs):
        return _propagate_sequential(batch, frame_paths, mask_paths, ref_edit, prompt, seed, edited_dir, backend, feather_px)
    grid_img = _tile_2x2(origs)
    grid_mask = _tile_2x2(masks)
    grid_ref = _tile_2x2([ref_edit, ref_edit, ref_edit, ref_edit])
    grid_out = _run_padded_inpaint(grid_img, grid_mask, prompt, seed, backend, feather_px, image_reference=grid_ref)
    edited_count, max_diff = 0, 0
    for slot, idx in enumerate(batch):
        r, c = divmod(slot, 2)
        tile_out = grid_out[r * h : (r + 1) * h, c * w : (c + 1) * w]
        comp = composite_strict_mask(origs[slot], tile_out, masks[slot], feather_px=feather_px)
        if np.any(masks[slot]):
            edited_count += 1
        max_diff = max(max_diff, _save_edited_frame(edited_dir, idx, comp, origs[slot], masks[slot]))
    return edited_count, max_diff


def propagate_masked_edits(
    frame_paths: dict[int, Path],
    mask_paths: dict[int, Path],
    ref_frame_idx: int,
    ref_edited_path: Path,
    prompt: str,
    seed: int,
    output_dir: Path,
    backend: InpaintBackend,
    skip_frames: set[int] | None = None,
    config: DiffusionConfig | None = None,
) -> dict[str, Any]:
    """Propagate an approved reference-frame edit across all active views with strict mask compositing."""
    cfg = config or DiffusionConfig()
    ref_edit = _load_rgb(ref_edited_path)
    edited_dir = output_dir / "edited_frames"
    edited_dir.mkdir(parents=True, exist_ok=True)
    skipped = skip_frames or set()
    active = [idx for idx in sorted(frame_paths) if idx not in skipped]
    edited_count, max_diff = 0, 0
    if cfg.grid_2x2:
        for i in range(0, len(active), 4):
            c, d = _propagate_grid_batch(active[i : i + 4], frame_paths, mask_paths, ref_edit, prompt, seed, edited_dir, backend, cfg.feather_px)
            edited_count, max_diff = edited_count + c, max(max_diff, d)
    else:
        edited_count, max_diff = _propagate_sequential(active, frame_paths, mask_paths, ref_edit, prompt, seed, edited_dir, backend, cfg.feather_px)
    metrics = {
        "ref_frame_idx": ref_frame_idx, "frames_edited": edited_count, "frames_skipped": len(skipped),
        "grid_2x2": cfg.grid_2x2, "max_unmasked_diff": max_diff, "seed": seed, "model_id": cfg.model_id,
    }
    (output_dir / "diffusion_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _resolve_gpu_ordinal(device: str, torch_mod: Any) -> tuple[bool, int]:
    if not device.startswith("cuda") or not torch_mod.cuda.is_available():
        return False, 0
    ordinal = int(device.split(":")[1]) if ":" in device else 0
    return (True, ordinal if ordinal < torch_mod.cuda.device_count() else 0)


def load_diffusion_backend(config: DiffusionConfig | None = None) -> InpaintBackend:
    """Load T4-safe float16/CPU-offload FLUX.2 Klein 4B or SDXL inpainting pipeline without bitsandbytes."""
    cfg = config or DiffusionConfig()
    try:
        torch = importlib.import_module("torch")
        diffusers = importlib.import_module("diffusers")
    except ImportError as exc:
        raise StageExecutionError("s06_diffusion requires torch and diffusers when loading live weights") from exc
    is_cuda, gpu_id = _resolve_gpu_ordinal(cfg.device, torch)
    cap = torch.cuda.get_device_capability(gpu_id) if is_cuda else None
    dtype = getattr(torch, resolve_t4_dtype(cap))
    flux_cls = getattr(diffusers, "Flux2KleinInpaintPipeline", None)
    sdxl_cls = getattr(diffusers, "StableDiffusionXLInpaintPipeline", None)
    hf_token = os.environ.get("HF_TOKEN") or None
    if flux_cls is not None:
        pipe = flux_cls.from_pretrained(cfg.model_id, torch_dtype=dtype, token=hf_token)
        supports_ref = True
    elif sdxl_cls is not None:
        pipe = sdxl_cls.from_pretrained(cfg.fallback_model_id, torch_dtype=dtype, token=hf_token)
        supports_ref = False
    else:
        raise StageExecutionError("diffusers must provide Flux2KleinInpaintPipeline or StableDiffusionXLInpaintPipeline")
    if is_cuda:
        if cfg.use_sequential_offload:
            pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
        else:
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    else:
        pipe.to("cpu")
    for fn_name in ("enable_vae_tiling", "enable_vae_slicing"):
        if hasattr(pipe, fn_name):
            getattr(pipe, fn_name)()
    exec_device = f"cuda:{gpu_id}" if is_cuda else "cpu"
    return _DiffusersBackend(pipe, torch, cfg, supports_ref=supports_ref, exec_device=exec_device)
