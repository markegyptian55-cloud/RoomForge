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

_Baseline verified. S06 parameter ablations, VRAM profiling, and artifact comparison will be appended in S2.3 / S3.1._
