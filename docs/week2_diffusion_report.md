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

## 2. Live Bootstrap & Quantization Benchmark (`S0.2b-v2`, Pending Run)
- Target tests: `bitsandbytes` NF4 load, `Flux2KleinInpaintPipeline` (`image=`, `mask_image=`, `image_reference=`) vs `StableDiffusionXLInpaintPipeline` peak VRAM, latency, and `max_abs_diff == 0` outside mask.
