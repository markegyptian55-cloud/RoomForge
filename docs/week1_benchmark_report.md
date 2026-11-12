# Week 1 Reconstruction & Pose Backend Benchmark Report

## 1. Purpose & Evidence Classification

This report documents the hardware preflight measurements on Kaggle Dual Tesla T4 GPUs, the implemented Week 1 reconstruction pipeline (`s01_extract` through `s04_export`), and the verified local synthetic invariants. Every figure below is explicitly labeled by verification scope:

- **Verified (Kaggle `S0.2b`)**: Measured directly inside `mohamedelbasyouni55/room3d-t4-preflight` (version `1` and version `3`, `KernelWorkerStatus.COMPLETE`).
- **Verified (Local CPU + Playwright)**: Measured by automated unit and E2E integration suites in `worker/tests/` and `tests/e2e/test_s18_week1_vertical.py`.
- **Protocol / Design Target**: Runtime thresholds and Kaggle execution parameters configured in code, awaiting full multi-minute real-video benchmark capture on cloud T4 instances.

---

## 2. Verified Hardware Baseline (`Kaggle Dual Tesla T4`)

Measured by `worker/room3d_worker/kaggle/preflight_runner.py` in Kaggle runs `S0.2b-v1`..`v3`:

| Metric | Verified Value | Engineering Impact |
| :--- | :--- | :--- |
| GPU Hardware | `2 × NVIDIA Tesla T4` (`15360 MiB` each) | Dedicated `cuda:1` for SAM 2.1 + 3DGS, `cuda:0` for Diffusion (`WorkerClient.BATCH_GPU`) |
| Compute Capability | `(7, 5)` (`sm_75`, Turing) | Hardware `bfloat16` absent; pipeline enforces `float16` / `np.isfinite` checks and Kaggle `torch.isfinite` preflight probes |
| Idle GPU 0 VRAM | `14.41 GB` free | Requires CPU offload and bounded pixel/tile budgets for 4B diffusion models |
| `/kaggle/working` Disk | `19.5 GB` free | Requires artifact cleanup and compressed `.spz` output (~10× smaller than raw `.ply`) |
| OS / Runtime | `Linux-6.18.55+`, Python `3.13`, `torch 2.11.0+cu130` | Unpinned `bitsandbytes` fails `diffusers` import (`libbitsandbytes_cuda130.so`); excluded |
| `pycolmap` Version | `4.2.1` (`pycolmap-cuda12`) | Verified `ALIKED_N16ROT`, `ALIKED_LIGHTGLUE`, `onnx_support: true`, `global_mapping` |

---

## 3. Pose Backend Comparison & Selection Logic (`s02_poses.py`)

| Backend Identifier | Library / Method | License Status | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| `colmap_global` | `pycolmap` (`ALIKED_N16ROT` + `ALIKED_LIGHTGLUE` -> `global_mapping`) | `BSD-3-Clause` (Allowed) | Primary Week 1 backend (`PyColmapBackend("global")`) |
| `colmap_incremental` | `pycolmap.incremental_mapping` (`ba_use_gpu=True`) | `BSD-3-Clause` (Allowed) | Automatic fallback when registration ratio `< 0.85` |
| `vggt_chunked` | `VGGTChunkedBackend` (`chunk_size=32`, `overlap=6`, `fp16`, Kabsch `align_sim3`) | Benchmark/Commercial config | Feed-forward fallback for low-texture walls (`Sim(3)` overlap stitching) |
| `map_anything` | `MapAnythingBackend` (`fp16`) | `Apache-2.0` (Allowed) | Single-pass metric/pose feed-forward alternative |
| `DUSt3R` / `MASt3R` / `Pi3` | Excluded by policy | Non-commercial | Forbidden in runtime and dependencies |

### Key Verified Pose & Coordinate Safeguards
1. **Database Isolation (`s02_poses.py`)**: Enforces `database.db` outside `image_dir` before invoking `pycolmap.global_mapping(database_path, image_path, output_path)` (discovered during `S0.2b-v3` preflight).
2. **Mod-16 & Principal Point Sync (`coords.mod16_crop_or_pad_params`)**: Ensures undistorted `PINHOLE` frames are divisible by `16` and adjusts `cx`, `cy` by exact crop offsets `(x0, y0)` without altering `fx`, `fy`.
3. **Non-Destructive Gravity Alignment (`coords.align_gravity_up`)**: Computes floor normal via seeded RANSAC with SVD fallback, guarantees `det(R) == +1`, and writes `T_viewer_from_colmap` to `manifest.json` without mutating `sparse/0/`.

---

## 4. Extraction, 3DGS Training & Export Verification (`s01`, `s03`, `s04`)

| Stage | Module | Verified Local Test Checks |
| :--- | :--- | :--- |
| `01_extract` | `s01_extract.py` | Drops blurry/static frames (`baseline * 0.35`, `min_parallax_frac=0.04`), outputs `mod 16` JPEGs, verifies zero EXIF/GPS/audio metadata (`worker/tests/test_extract.py`). |
| `02_poses` | `s02_poses.py` | Tests `< 0.85` registration fallback from `colmap_global` to `colmap_incremental`, Sim(3) recovery (`atol=1e-4`), `sparse/0` layout, `mod 16` intrinsics (`worker/tests/test_poses.py`). |
| `03_train` | `s03_train.py` | Enforces `test_every=0` and `device="cuda:1"`, tests vectorized 2,000-point KNN scale init without `O(N^2)` pairwise `cdist`, retains dense wall geometry (`worker/tests/test_export.py`). |
| `04_export` | `s04_export.py` | Validates `.spz` + `sparse_points.json` + Pydantic `SceneManifest` (mirroring `contracts/manifest.schema.json`) conformance (`worker/tests/test_export.py`, `tests/e2e/test_s18_week1_vertical.py`). |
