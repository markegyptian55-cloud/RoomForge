<div align="center">

# 🏛️ RoomForge
### Video to a Browsable 3D Room with Generative Spatial Staging

[![Next.js](https://img.shields.io/badge/Next.js-16.4.6_App_Router-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x_CUDA-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL2_Splatting-000000?style=for-the-badge&logo=three.js)](https://threejs.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0_%2F_MIT-blue?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing_100%25-success?style=for-the-badge&logo=vitest)](tests)

<p align="center">
  <b>Turn casual smartphone walkthrough videos into photorealistic, browsable 3D Gaussian rooms.</b><br/>
  Select, restage, and redesign furniture with generative AI in real time directly inside the browser.
</p>

[Key Features](#-key-features) •
[Architecture](#-system-architecture) •
[Quickstart](#-quickstart-in-60-seconds) •
[Pipeline Stages](#-complete-pipeline) •
[Commercial Guardrails](#-commercial-licensing-guardrails)

</div>

---

## 🌟 Overview

**RoomForge** is an end-to-end commercial-grade computer vision platform that bridges the gap between raw video capture, 3D Gaussian Splatting (`3DGS`), and multimodal generative editing.

Instead of requiring expensive LiDAR rigs, specialized camera hardware, or static 3D meshes, **RoomForge** accepts a casual phone video of any room, reconstructs a photorealistic 3D Gaussian representation (`.spz`), and provides an interactive WebGL2 studio where users can freely walk around, click any piece of furniture, and replace or redesign it with text prompts.

```
       Casual Phone Walkthrough Video (.mp4)
                        │
                        ▼
    ┌────────────────────────────────────────┐
    │     1. Frame Extraction & Pose Sync    │  PyAV, Global Colmap, Pinhole
    └───────────────────┬────────────────────┘
                        │
                        ▼
    ┌────────────────────────────────────────┐
    │     2. 3D Gaussian Scene Splatting     │  gsplat, Continuous Gaussian Discs
    └───────────────────┬────────────────────┘
                        │
                        ▼
    ┌────────────────────────────────────────┐
    │     3. Real-Time WebGL2 3D Studio      │  Next.js 16, @sparkjsdev/spark
    └───────────────────┬────────────────────┘
                        │
                        ▼
    ┌────────────────────────────────────────┐
    │  4. Interactive 3D-to-2D Segmentation  │  Raycast Click, SAM 2.1 Tracking
    └───────────────────┬────────────────────┘
                        │
                        ▼
    ┌────────────────────────────────────────┐
    │     5. Diffusion Staging & Rebuild     │  FLUX.2 [klein] / SDXL, RoI Finetune
    └───────────────────┬────────────────────┘
                        │
                        ▼
     Browsing A/B Restaged 3D Scene in Browser (60 FPS)
```

---

## ✨ Key Features

### 🎥 Zero-Rig Casual Capture to 3D
- **No Hardware Dependencies**: Works with standard H.264/HEVC smartphone videos.
- **PyAV Frame Extraction**: Memory-efficient metadata stripping and keyframe selection without requiring system `ffmpeg.exe` binaries.
- **OpenCV PINHOLE Geometry**: Strict `mod 16` dimensions with synchronized principal point $(c_x, c_y)$ and focal length $(f_x, f_y)$.

### 🌐 High-Performance Browser Studio
- **Full-Viewport Immersive Experience**: Clean studio layout with a docked, independently scrolling controls sidebar and floating heads-up display (`HUD`).
- **Smooth 60 FPS WebGL2 Walkthrough**: Powered by `@sparkjsdev/spark` and Three.js with soft Gaussian alpha discs eliminating wireframe and square-billboard artifacts.
- **Dual Navigation Modes**:
  - **Orbit Mode**: Seamless mouse drag, pinch zoom, and inertia damping for overview inspection.
  - **Walk Mode**: First-person walkthrough with WASD and Arrow keys, pitch/yaw mouse-look, and proxy collision avoidance.
- **Adaptive Dark / Light Theme**: Built with React 19 `useSyncExternalStore` for zero layout flash.

### 🎯 3D-to-2D Interactive Object Selection
- **Direct 3D Raycasting**: Click any sofa, table, or chair directly inside the 3D room.
- **Intelligent View Synthesis**: Automatically projects the 3D hit point onto the most optimal 2D camera reference frame.
- **SAM 2.1 Mask Tracking**: Propagates segmented masks across all multi-view video frames with a 5-tier fallback cascade (Warm SAM 2.1 $\to$ Persistent Video State $\to$ Convex Hull Flow $\to$ Bounding Box $\to$ Manual 1-channel PNG Upload).

### 🎨 Generative Diffusion Restaging
- **Targeted Spatial Inpainting**: Uses `FLUX.2 [klein] 4B` (Apache-2.0) or `SDXL Inpaint` to generate photorealistic furniture variations from text prompts.
- **Strict Background Invariance**: Compositing asserts exact mathematical pixel equality on unmasked regions ($\max|\text{diff}| = 0$).

### 🔄 Localized Region-of-Interest 3DGS Rebuild
- **Fast 5k-Step Masked Fine-Tuning**: Freezes background Gaussians and optimizes only the target object volume in seconds, preventing room-wide catastrophic forgetting.
- **Instant A/B Version Switching**: Seamlessly toggle between design variations (e.g. `V1 Base (Navy Sofa)` vs `V2 Red Velvet Sofa`) without reloading assets.

---

## 🏛️ System Architecture

RoomForge follows a decoupled control plane and GPU worker architecture:

```
  ┌─────────────────────────────────────────────────────────────┐
  │                 Client Browser (Studio UI)                  │
  │     Next.js 16 (App Router) + React 19 + Three.js / Spark    │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ HTTP REST / Storage Stream
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │              Control Plane (backend/ - FastAPI)             │
  │  - Zero GPU/CUDA dependencies (lightweight & CPU-safe)      │
  │  - SQLite + LocalObjectStore (Dev) / Postgres + S3 (Prod)   │
  │  - Worker Pull Contract (claim, heartbeat, progress, done)  │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ HTTP Worker Pull Contract
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │            Heavy Compute Worker (worker/room3d_worker)      │
  │  - GPU 0: Diffusion Inpainting (FLUX.2 [klein] 4B, sm_75)   │
  │  - GPU 1: Warm SAM 2.1 Loop (<= 500ms) & 3DGS Rebuild Engine │
  └─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart in 60 Seconds

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** & `npm`

### One-Click Launch (Windows)
Double-click `start_room3d.bat` in the repository root or run:
```cmd
start_room3d.bat
```
This automatically boots the FastAPI control plane on port `8000`, the Next.js studio on port `3000`, and opens the live photorealistic demo room in your default browser.

### Manual Setup

#### 1. Setup Backend & Seed Demo Scene
```bash
# Set PYTHONPATH for decoupled modules
# Windows PowerShell:
$env:PYTHONPATH = "backend;worker"
# Linux / macOS:
export PYTHONPATH="backend:worker"

# Install backend dependencies
cd backend && pip install -r requirements.txt && cd ..

# Seed the high-fidelity photorealistic living room demo
python scripts/seed_demo.py

# Start FastAPI control plane
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

#### 2. Launch Next.js Web Studio
```bash
cd frontend
npm install
npm run dev -- -p 3000
```

Open your browser at:
👉 **[http://localhost:3000/projects/demo-room](http://localhost:3000/projects/demo-room)**

---

## 🔬 Complete Pipeline

| Stage | Module | Primary Tech | Function |
| :---: | :--- | :--- | :--- |
| **01** | `s01_extract` | `PyAV` | Decode video, sample synchronized keyframes, strip sensitive metadata. |
| **02** | `s02_poses` | `pycolmap` / `VGGT` | Pinhole camera calibration, feature matching, global pose estimation. |
| **03** | `s03_train` | `gsplat` | Train 3D Gaussian Splatting scene with Turing `sm_75` FP16 optimization. |
| **04** | `s04_export` | `spz` / `Three.js` | Export compressed `.spz` splat bundle, `.ply`, and `manifest.json`. |
| **05** | `s05_segment` | `SAM 2.1` | Interactive raycast click resolving, 2D reference projection, multi-view tracking. |
| **06** | `s06_diffuse` | `FLUX.2` / `SDXL` | Masked inpainting with strict background preservation. |
| **07** | `s07_rebuild` | `PyTorch` | Region-of-Interest masked 3D Gaussian fine-tuning with frozen background. |

---

## ⚖️ Commercial Licensing Guardrails

RoomForge is engineered from the ground up for commercial deployment. All core components use enterprise-friendly, non-copyleft open-source licenses:

| Component | Upstream / Library | License | Status |
| :--- | :--- | :--- | :---: |
| **Control Plane** | FastAPI / SQLAlchemy / Pydantic | `MIT` | Verified |
| **Frontend Studio** | Next.js / React 19 / Three.js | `MIT` | Verified |
| **Camera Tracking** | PyColmap / OpenGV | `BSD-3-Clause` | Verified |
| **3DGS Kernel** | gsplat (Nerfstudio) | `Apache-2.0` | Verified |
| **Object Tracking** | SAM 2.1 (Meta) | `Apache-2.0` | Verified |
| **Generative Inpainting**| FLUX.2 [klein] 4B (Black Forest Labs) | `Apache-2.0` | Verified |
| **Splat Format** | `.spz` (Scanline / Niantic) | `Apache-2.0` | Verified |

> [!IMPORTANT]
> **Strictly Blocked Non-Commercial Models**: The codebase explicitly enforces runtime and dependency gates preventing the inclusion of non-commercial or restrictive licenses (including `DUSt3R`, `MASt3R`, `Pi3`, `2DGS`, `Mip-Splatting`, `PGSR`, `GOF`, `MILo`, `SuperPoint`, `SuperGlue`, and `FLUX.2 [klein] 9B`).

---

## 🧪 Verification & Automated Tests

RoomForge maintains a strict test suite runnable locally on CPU:

```bash
# 1. Run all backend and worker unit/integration tests (54 tests)
$env:PYTHONPATH = "backend;worker"
pytest backend/tests worker/tests -v

# 2. Run frontend unit and component tests (Vitest)
cd frontend
npx vitest run

# 3. Check TypeScript types and ESLint
npx tsc --noEmit
npm run lint

# 4. Run end-to-end headless browser verification
npx playwright test
```

---

## 👤 Author & Maintainer

**Mohamed Mostafa Elbasyouni**  
GitHub: [@markegyptian55-cloud](https://github.com/markegyptian55-cloud)  
Email: [markegyptian55@gmail.com](mailto:markegyptian55@gmail.com)

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
