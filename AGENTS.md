# Repository Engineering Standard — Room3D

You are writing production code for a commercial **Video to a Browsable 3D Room** platform.

## 1. Human-Crafted Code Discipline
- Write clear, minimal, idiomatic code as a senior systems/CV engineer would write by hand.
- Keep functions small (`<= 40` lines where practical), nesting shallow (`<= 3` levels), and prefer early returns.
- Never duplicate logic across modules or test helpers. Extract shared helpers only when two callers genuinely share behavior.
- Never write "AI-narrator" comments explaining obvious syntax (`# loop over frames`, `# initialize list`). Comment *why* an invariant, coordinate sign flip, or hardware workaround exists.
- Never swallow exceptions silently (`except Exception: pass`). Raise explicit typed domain errors with actionable context.
- Strict typing everywhere: Python 3.11+ type hints (`from __future__ import annotations`), Pydantic v2 schemas, TypeScript `strict: true` with zero `any`.

## 2. Architecture & Trust Boundaries
- Never import or copy code/training cells verbatim from `tasks/3d-con.ipynb`. Use our clean decoupled architecture.
- Control Plane (`backend/`) never loads PyTorch/CUDA models. It communicates with `worker/` exclusively over HTTP pull contracts (`claim`, `heartbeat`, `progress`, `complete`, `fail`) and `StorageDriver`.
- Storage & DB are environment-switchable: SQLite + `LocalObjectStore` in local dev (no Docker required); PostgreSQL (`SKIP LOCKED`) + S3/R2 in production.
- Never assume `ffmpeg.exe` on `PATH` on Windows: use `PyAV` (`av`) wheels for decode/trim/metadata-stripping.
- Heavy GPU stages run inside isolated subprocesses with explicit `GPU 0` (Diffusion) and `GPU 1` (SAM 2.1 + 3DGS) assignment. Interactive SAM segmentation uses a warm single-session loop on `GPU 1` (`<= 500ms` claim latency).

## 3. Computer Vision & 3D Invariants
- Canonical SfM format is undistorted OpenCV `PINHOLE` saved under `sparse/0/{cameras,images,points3D}.bin`.
- Undistorted image dimensions `(width, height)` must be divisible by `16` (`mod 16`) with exact synchronized `cx, cy, fx, fy`.
- Never mutate raw COLMAP poses to flip upright; store `T_viewer_from_colmap` (`4x4` row-major) inside `manifest.json`.
- Target GPU compute capability is Tesla T4 (`sm_75` Turing): never require hardware `bfloat16` or `fp8` Tensor Cores; use `fp16` / `NF4` quantization + CPU offload.
- Canonical viewer format is `.spz` (rendered by `@sparkjsdev/spark` WebGL2 with `Three.js` `Points` fallback).
- Week 2 masks are 1-channel `uint8` PNG (`0` or `255`) matching undistorted `W x H`. Mask compositing must preserve unmasked pixels identically (`max_abs_diff == 0`).

## 4. Commercial Licensing Guardrails
- Allowed: `Apache-2.0`, `MIT`, `BSD-3-Clause` (`COLMAP`, `VGGT-commercial/benchmark`, `MapAnything-Apache`, `gsplat`, `nv-tlabs/ppisp`, `spz`, `@playcanvas/splat-transform`, `@sparkjsdev/spark`, `SAM 2/2.1`, `FLUX.2 [klein] 4B / base-4B`, `SDXL`).
- Forbidden in runtime/dependencies: `GaussianSplats3D` (unmaintained), `SuperPoint`, `SuperGlue`, `DUSt3R`, `MASt3R`, `Pi3`, `2DGS`, `Mip-Splatting`, `PGSR`, `GOF`, `MILo`, `Qwen-Image-2.1`, `FLUX.2 [klein] 9B`.

## 5. Local Verification Standard
- Every slice must ship automated tests runnable locally on CPU + `GTX 1660 SUPER 6GB` without requiring Kaggle GPU or Docker (use synthetic `PyAV` MP4 fixtures, mocked CV tensors, and headless Playwright with SwiftShader WebGL flags).
