# Room3D — Video to a Browsable 3D Room

Commercial-grade platform converting walkthrough videos into browsable 3D Gaussian Splatting (`.spz`) room scenes with interactive object selection, 5-tier multi-view tracking, T4-safe masked diffusion editing, and localized 3DGS rebuilding.

---

## Architecture Overview

```text
  Browser UI (Next.js 16 + React 19 + @sparkjsdev/spark / Three.js)
         |
         | REST + Multipart + Storage Streaming
         v
  Control Plane (backend/ - FastAPI + SQLAlchemy 2.x + Pydantic v2)
    - SQLite + LocalObjectStore (local dev, zero Docker required)
    - PostgreSQL (FOR UPDATE SKIP LOCKED) + S3 / Cloudflare R2 (production)
    - Pure-HTTP Worker Pull Queue (claim / heartbeat / progress / complete / fail)
         |
         | HTTP Pull Contract (never imports torch/CUDA inside backend/)
         v
  GPU Worker Plane (worker/room3d_worker/)
    - GPU 1 (cuda:1): Interactive Warm SAM 2.1 loop, 5-tier Tracking, 3DGS Training & Rebuild
    - GPU 0 (cuda:0): Diffusers Inpainting (FLUX.2 [klein] 4B / SDXL Inpaint, sm_75 FP16)
```

### Repository Layout

| Path | Purpose |
| :--- | :--- |
| `backend/app/` | FastAPI control plane (`auth`, `projects`, `scenes`, `tasks`, `edits`, queue & storage drivers) |
| `worker/room3d_worker/` | GPU pipeline (`s01_extract`..`s07_rebuild`), HTTP worker client, coordinate & click resolvers, Kaggle preflight |
| `frontend/src/` | Next.js 16 App Router workspace, WebGL2 `.spz` viewer (`Viewer3D`), Edit Studio & A/B version compare |
| `contracts/` | Shared canonical contracts (`enums.json`, `manifest.schema.json`) |
| `docs/` | Week 1 & Week 2 engineering reports, hardware baselines, and handoff specification |
| `tests/e2e/` | Full vertical Week 1 (`test_s18_week1_vertical.py`) and Week 2 (`test_s31_week2_vertical.py`) E2E suites |

---

## Key Technical Invariants

- **Canonical SfM Layout**: Undistorted OpenCV `PINHOLE` COLMAP reconstruction written strictly to `sparse/0/{cameras,images,points3D}.bin` with `mod 16` image dimensions and synchronized `cx, cy, fx, fy`.
- **Non-Destructive Gravity Alignment**: `T_viewer_from_colmap` (`4x4` row-major rigid matrix) aligns rooms to Three.js Y-up in `manifest.json` (`schema_version: "1.0.0"`) without mutating raw COLMAP coordinates.
- **Mask & Background Preservation**: Week 2 masks are 1-channel `uint8` (`{0, 255}`) PNGs matching undistorted `(height, width)`. Both pixel compositing (`composite_strict_mask`) and `masked_finetune` 3DGS rebuilding assert zero unmasked drift.
- **Tesla T4 (`sm_75`) Compatibility**: Runs in `float16` + CPU offload without requiring `bfloat16`, `fp8`, or `bitsandbytes`.
- **Commercial License Compliance**: Strictly uses `Apache-2.0`, `MIT`, and `BSD-3-Clause` runtime/build dependencies (`pycolmap`, `VGGT-commercial`, `MapAnything`, `gsplat`, `spz`, `@sparkjsdev/spark`, `SAM 2.1`, `FLUX.2 [klein] 4B`, `SDXL`). Excludes forbidden models (`DUSt3R`, `MASt3R`, `Pi3`, `2DGS`, `Mip-Splatting`, `PGSR`, `GOF`, `MILo`, `GaussianSplats3D`, `SuperPoint`, `SuperGlue`, `Qwen-Image-2.1`, `FLUX.2 [klein] 9B`).

---

## Local Development Quickstart (No Docker Required)

### 1. Control Plane (`backend/`)

Default `Settings` uses local SQLite (`./data/room3d.db`) and `./data/storage` (`AUTH_REQUIRED=false` in dev).

```powershell
# PowerShell (Windows) or Bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Production environment variables (`backend/app/config.py`):
- `DATABASE_URL` (`postgresql+psycopg://...`)
- `STORAGE_BACKEND=s3`, `S3_ENDPOINT_URL`, `S3_BUCKET`, `S3_REGION`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`
- `AUTH_REQUIRED=true`, `JWT_SECRET`, `WORKER_TOKEN`

### 2. Frontend (`frontend/`)

```powershell
cd frontend
npm install
npm run dev
```

### 3. Kaggle Dual-T4 Preflight CLI

```powershell
$env:PYTHONPATH = "worker"
python -m room3d_worker.kaggle.preflight_runner build --out-dir kaggle_bundle
```

---

## Verification & Automated Test Suite

```powershell
# Backend + Worker + Full Week 1 & Week 2 E2E suites (runs on CPU)
$env:PYTHONPATH = "backend;worker"
pytest backend/tests worker/tests tests/e2e/test_s18_week1_vertical.py tests/e2e/test_s31_week2_vertical.py -v

# Frontend TypeScript, ESLint, Vitest unit/component tests, Playwright E2E, and Next.js production build
cd frontend
npx tsc --noEmit
npm run lint
npx vitest run
npx playwright test
npm run build
```

Detailed stage evidence and Kaggle hardware baselines are documented in [`docs/week1_benchmark_report.md`](docs/week1_benchmark_report.md), [`docs/week1_to_week2_handoff.md`](docs/week1_to_week2_handoff.md), [`docs/week2_tracking_report.md`](docs/week2_tracking_report.md), and [`docs/week2_diffusion_report.md`](docs/week2_diffusion_report.md).
