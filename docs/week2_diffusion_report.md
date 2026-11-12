# Week 2 Diffusion & T4 (`sm_75`) Hardware Report

## 1. Live Kaggle `2× Tesla T4` Hardware Baseline (`S0.2b-v1`, Verified)
- **Kernel:** `mohamedelbasyouni55/room3d-t4-preflight` (version `1`, `KernelWorkerStatus.COMPLETE`)
- **OS / Python:** `Linux-6.18.55+-x86_64-with-glibc2.39`, Kaggle container Python `3.13`
- **GPUs:** `2 × Tesla T4` (`15360 MiB` each), CUDA compute capability `(7, 5)` (`sm_75`, Turing)
- **Free Working Disk:** `19.5 GB` (`/kaggle/working`)
- **Free VRAM (idle GPU 0):** `14.41 GB`
- **CUDA `float16` Matmul + `torch.isfinite` Guard:** `true` (bf16 NOT required)
- **Pre-installed Kaggle Image Packages:**
  - `diffusers` (`Flux2KleinInpaintPipeline` available: `true`, `StableDiffusionXLInpaintPipeline` available: `true`)
  - `accelerate` (`available: true`)
  - Missing in base image (requires worker/notebook bootstrap): `pycolmap-cuda12`, `av`, `gsplat`, `spz`, `ppisp`, `bitsandbytes`

## 2. Live Bootstrap & API Verification (`S0.2b-v2` -> `S0.2b-v3`, Verified)
- **`S0.2b-v2` Trap Discovered (`bitsandbytes`):** Installing unpinned `bitsandbytes` on Kaggle `torch 2.11.0+cu130` crashes `diffusers.loaders.lora_pipeline` at import time (`libbitsandbytes_cuda130.so` missing).
- **`S0.2b-v3` (`KernelWorkerStatus.COMPLETE`):** Removing `bitsandbytes` restored clean imports for `Flux2KleinInpaintPipeline` (`float16`, `cpu_offload`) and `StableDiffusionXLInpaintPipeline` (`float16`).
- **Verified `Flux2KleinInpaintPipeline.__call__` parameters:** `attention_kwargs`, `callback_on_step_end`, `callback_on_step_end_tensor_inputs`, `generator`, `guidance_scale`, `height`, `image`, `image_reference`, `latents`, `mask_image`, `max_sequence_length`, `negative_prompt_embeds`, `num_images_per_prompt`, `num_inference_steps`, `output_type`, `padding_mask_crop`, `prompt`, `prompt_embeds`, `return_dict`, `sigmas`, `strength`, `text_encoder_out_layers`, `width`.
- **Verified `pycolmap 4.2.1` signature:** `ALIKED_N16ROT` + `ALIKED_LIGHTGLUE` (`onnx_support: true`, `global_mapping: true`). Requires 3 positional args `pycolmap.global_mapping(database_path, image_path, output_path)` with `database.db` stored **outside** `image_path`.
- **Verified packaging trap:** PyPI `spz` has a circular import error (`BoundingBox`) on Python 3.13; use `scaniverse/spz` or pure-Python protobuf gzip writer + `@playcanvas/splat-transform` post-processing.

## 3. Implemented T4-Safe Diffusion & Compositor Safeguards (`s06_diffusion.py`, S2.3)

| Safeguard | Implementation (`worker/room3d_worker/pipeline/s06_diffusion.py`) | Rationale / Verified Constraint |
| :--- | :--- | :--- |
| **Commercial License Gate** | `_assert_allowed_model_id` (`FORBIDDEN_MODEL_PATTERNS = ("9b", "qwen")`) | Permits `flux2_klein_4b` (`Apache-2.0`) and `sdxl_inpaint_t4` (`OpenRAIL++`); blocks `FLUX.2 [klein] 9B` and `Qwen-Image-2.1`. |
| **VRAM & Grid Pixel Guard** | `MAX_GRID_PIXELS = 1048576` (`1024 * 1024`), `enable_vae_tiling`, `enable_vae_slicing`, `resolve_t4_dtype` | Multi-frame 2x2 grid (`<= 4` views via `_tile_2x2` / `_propagate_grid_batch`) automatically falls back to sequential per-frame generation (`_propagate_sequential`) conditioned on `image_reference` whenever total grid pixels exceed `1024^2` or batch size `< 4`. Defaults: `num_inference_steps=8`, `guidance_scale=3.5`, `strength=0.85`, `feather_px=3`. |
| **Bit-Exact Background Preservation** | `composite_strict_mask` + `assert_unmasked_unchanged` | Enforces `max_abs_diff == 0` outside the binary mask (`worker/tests/test_diffusion.py`). |
| **Mod-16 & T4 Dtype Guard** | `pad_to_multiple(..., 16)` + `resolve_t4_dtype` | Pads non-mod-16 dimensions safely before inpainting and selects `torch.float16` on `sm_75` GPUs (CPU uses `float32`). |

---

## 4. Week 2 Localized 3DGS Rebuild (`s07_rebuild.py`, S2.4)

- **`masked_finetune` (`fine_tune_steps=5000`)**: Computes `compute_roi_gaussian_mask` over tracked masks (`roi_pad_px=8`), optimizes ONLY `roi_mask` parameters while keeping background Gaussians frozen in differentiable `gsplat.rasterization` (`_run_opt_loop`), and enforces `assert_frozen_gaussians_unchanged` post-rebuild / pre-export so any background mutation fails loudly (`worker/tests/test_rebuild.py`).
- **`from_scratch` (`from_scratch_steps=15000`)**: Full unmasked ablation mode comparing global re-optimization against localized Week 2 editing.
- **Test Status**: `worker/tests/test_diffusion.py` (`5/5`) and `worker/tests/test_rebuild.py` (`4/4`) pass on CPU with synthetic tensors.
