<div align="center">

# 🏛️ RoomForge
### Video to a Browsable 3D Room with Generative Spatial Staging

[![Next.js](https://img.shields.io/badge/Next.js-16.4.6_App_Router-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x_CUDA-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL2_Splatting-000000?style=for-the-badge&logo=three.js)](https://threejs.org/)
[![Vercel](https://img.shields.io/badge/Vercel-Live_Production-000000?style=for-the-badge&logo=vercel)](https://room-forge.vercel.app)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dual_Tesla_T4_GPU-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/code/mohamedelbasyouni55/roomforge-gpu-worker)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-67_Passing_100%25-success?style=for-the-badge&logo=pytest)](worker/tests)

<p align="center">
  <b>Turn casual smartphone walkthrough videos into photorealistic, browsable 3D Gaussian rooms.</b><br/>
  Select, restage, and redesign furniture with generative AI in real time directly inside the browser.
</p>

[Live Links](#-live-deployments--links) •
[Key Features](#-key-features) •
[Architecture](#-system-architecture) •
[Cloudflare Tunnel](#-cloudflare-tunnel-integration) •
[Kaggle GPU Worker](#-kaggle-gpu-worker-guide) •
[Repository Structure](#-repository-structure) •
[Quickstart](#-quickstart-in-60-seconds) •
[Commercial Guardrails](#-commercial-licensing-guardrails)

</div>

---

## 🌐 Live Deployments & Links

| Service | Destination | Purpose | Status |
| :--- | :--- | :--- | :---: |
| **Production Web Studio** | [room-forge.vercel.app](https://room-forge.vercel.app) | Interactive 3D WebGL2 browser studio | 🟢 Live |
| **GitHub Repository** | [github.com/markegyptian55-cloud/RoomForge](https://github.com/markegyptian55-cloud/RoomForge) | Source code & continuous integration | 🟢 Active |
| **Vercel Project Dashboard** | [vercel.com/markegyptian55-4853/roomforge](https://vercel.com/markegyptian55-4853/roomforge) | Deployment metrics & edge configuration | 🟢 Configured |
| **Kaggle GPU Worker** | [kaggle.com/code/mohamedelbasyouni55/roomforge-gpu-worker](https://www.kaggle.com/code/mohamedelbasyouni55/roomforge-gpu-worker) | Free Dual Tesla T4 (30 hrs/week) pull worker | 🟢 Ready |

---

## 🌟 Overview

**RoomForge** is an end-to-end commercial-grade computer vision platform that bridges raw smartphone video capture, 3D Gaussian Splatting (`3DGS`), and multimodal generative editing.

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
- **Zero Hardware Dependencies**: Works with standard H.264/HEVC smartphone videos.
- **PyAV Frame Extraction**: Memory-efficient metadata stripping and keyframe selection without requiring system `ffmpeg.exe` binaries on Windows/Linux.
- **Calibrated OpenCV PINHOLE Geometry**: Strict `mod 16` dimensions with synchronized principal point $(c_x, c_y)$, focal lengths $(f_x, f_y)$, and extrinsic $4 \times 4$ camera poses.

### 🌐 High-Performance Browser Studio
- **Full-Viewport Immersive Experience**: Clean studio layout with a docked, collapsible controls sidebar and floating heads-up display (`HUD`).
- **Smooth 60 FPS WebGL2 Walkthrough**: Powered by `@sparkjsdev/spark` and Three.js with soft Gaussian alpha discs eliminating wireframe and square-billboard artifacts.
- **Dual Navigation Modes**:
  - **Orbit Mode**: Seamless mouse drag, pinch zoom, and inertia damping for overview inspection.
  - **Walk Mode**: First-person walkthrough with WASD and Arrow keys, pitch/yaw mouse-look, and proxy collision avoidance.
- **Camera Boundary Clamping**: Prevents users from clipping outside walls or under floors.
- **Adaptive Dark / Light Theme**: Built with React 19 `useSyncExternalStore` for zero layout flash.

### 🎯 3D-to-2D Interactive Object Selection
- **Direct 3D Raycasting**: Click any sofa, table, or chair directly inside the 3D room.
- **Intelligent View Synthesis**: Automatically projects the 3D hit point onto the most optimal 2D camera reference frame.
- **SAM 2.1 Mask Tracking**: Propagates segmented masks across all multi-view video frames with a 5-tier fallback cascade (Warm SAM 2.1 $\to$ Persistent Video State $\to$ Convex Hull Flow $\to$ Bounding Box $\to$ Manual 1-channel PNG Upload).

### 🎨 Generative Diffusion Restaging
- **Targeted Spatial Inpainting**: Uses `FLUX.2 [klein] 4B` (Apache-2.0) or `SDXL Inpaint` to generate photorealistic furniture variations from text prompts.
- **Strict Background Invariance**: Compositing asserts exact mathematical pixel equality on unmasked regions ($\max|\text{diff}| = 0$).

### 🔄 Localized Region-of-Interest 3DGS Rebuild
- **Fast Masked Fine-Tuning**: Freezes background Gaussians and optimizes only the target object volume in seconds, preventing room-wide catastrophic forgetting.
- **Instant A/B Version Switching**: Seamlessly toggle between design variations (e.g. `V1 Base (Navy Sofa)` vs `V2 Red Velvet Sofa`) without reloading assets.

---

## 🏛️ System Architecture

RoomForge follows a decoupled control plane and GPU worker architecture:

```
  ┌─────────────────────────────────────────────────────────────┐
  │                 Client Browser (Studio UI)                  │
  │     Next.js 16 (App Router) + React 19 + Three.js / Spark   │
  │                   https://room-forge.vercel.app             │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ HTTPS REST / WebSocket
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │             Cloudflare Tunnel (Zero-Trust Ingress)          │
  │           https://<random-slug>.trycloudflare.com           │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Proxied HTTP (Port 8000)
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │              Control Plane (backend/ - FastAPI)             │
  │  - Zero GPU/CUDA dependencies (lightweight & CPU-safe)      │
  │  - SQLite + LocalObjectStore (Dev) / Postgres + S3 (Prod)   │
  │  - Worker Pull Contract (claim, heartbeat, progress, done)  │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ HTTP Worker Pull Loop
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │           Kaggle Heavy Compute Worker (Dual Tesla T4)       │
  │  - GPU 0: Diffusion Inpainting (FLUX.2 [klein] 4B, sm_75)   │
  │  - GPU 1: Warm SAM 2.1 Loop (<= 500ms) & 3DGS Rebuild Engine│
  └─────────────────────────────────────────────────────────────┘
```

---

## 🛡️ Cloudflare Tunnel Integration

**Cloudflare Tunnel** (`cloudflared`) creates a secure, encrypted outbound tunnel from your local machine (or private server) to the Cloudflare Global Anycast Edge.

### Why Use Cloudflare Tunnel for RoomForge?
1. **Zero Open Ports / No Static IP**: Securely exposes your FastAPI control plane without router port forwarding or exposing home IP addresses.
2. **Automatic SSL / TLS**: Cloudflare issues full trusted HTTPS certificates (`https://*.trycloudflare.com`), allowing secure requests from your Vercel web app.
3. **Bypass ISP Carrier-Grade NAT (CGNAT)**: Allows remote workers (such as Kaggle notebooks) to pull tasks and upload 3D models seamlessly.
4. **DDoS Protection & Edge Caching**: Filters malicious traffic and buffers large asset uploads before they reach your machine.

### Quick Launch Guide

1. **Download & Run Cloudflare Tunnel**:
   ```powershell
   # Windows (winget):
   winget install --id Cloudflare.cloudflared

   # Or download standalone binary:
   # https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
   ```

2. **Start the Zero-Configuration Tunnel**:
   ```powershell
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
   Cloudflare outputs a live HTTPS endpoint:
   ```
   +--------------------------------------------------------------------------------------------+
   |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
   |  https://roomforge-tunnel.trycloudflare.com                                                |
   +--------------------------------------------------------------------------------------------+
   ```

3. **Connect Vercel and Kaggle**:
   - In **Vercel** (`Settings` $\to$ `Environment Variables`), set:
     `NEXT_PUBLIC_API_URL=https://roomforge-tunnel.trycloudflare.com`
   - In **Kaggle**, pass this URL as the worker API target.

---

## ⚡ Kaggle GPU Worker Guide

Leverage Kaggle's free 30 hours/week of **Dual Tesla T4 GPUs** (`2 x 16GB VRAM` = `32GB VRAM` total):

1. Open the Kaggle notebook:  
   👉 **[mohamedelbasyouni55/roomforge-gpu-worker](https://www.kaggle.com/code/mohamedelbasyouni55/roomforge-gpu-worker)**

2. In Kaggle Notebook Settings:
   - **Accelerator**: `GPU T4 x 2`
   - **Internet**: `On`

3. In the first code cell, start the autonomous pull worker:
   ```bash
   !git clone https://github.com/markegyptian55-cloud/RoomForge.git
   %cd RoomForge
   !pip install -r worker/requirements.txt

   # Run the pull daemon targeting your Cloudflare Tunnel URL:
   !python -m room3d_worker.main \
     --api-base "https://your-tunnel-subdomain.trycloudflare.com" \
     --token "dev-worker-token" \
     --poll-interval 2.0
   ```
4. The worker automatically polls for queued reconstruction and editing tasks, runs 3DGS / SAM 2.1 / FLUX.2, and streams the finished 3D scene back to your live browser.

---

## 📁 Repository Structure

```
Video to a Browsable 3D Room/
├── backend/                        # FastAPI Control Plane (Decoupled, CPU-Safe)
│   ├── app/
│   │   ├── main.py                 # FastAPI application entry & CORS middleware
│   │   ├── config.py               # Pydantic BaseSettings & environment config
│   │   ├── db.py                   # SQLAlchemy engine & session factory
│   │   ├── models.py               # Project, SceneVersion, Task, EditSession ORM
│   │   ├── security.py             # JWT bearer tokens & worker auth
│   │   ├── routers/
│   │   │   ├── auth.py             # User signup & JWT login endpoints
│   │   │   ├── projects.py         # Project lifecycle & video upload handling
│   │   │   ├── scenes.py           # Scene versions & manifest.json streaming
│   │   │   ├── tasks.py            # Worker pull contract (claim, heartbeat, complete)
│   │   │   └── edits.py            # SAM 2.1 raycast click, tracking, & rebuild APIs
│   │   └── storage/                # Abstract Local / S3 / R2 storage drivers
│   └── tests/                      # 13 backend unit & integration tests
│
├── worker/                         # Heavy GPU Compute Engine (PyTorch / CUDA)
│   ├── room3d_worker/
│   │   ├── main.py                 # Autonomous pull worker daemon CLI
│   │   ├── client.py               # HTTP client for control plane pull contract
│   │   └── pipeline/
│   │       ├── s01_extract.py      # PyAV frame sampling & metadata stripping
│   │       ├── s02_poses.py        # PyColmap & VGGT PINHOLE pose estimation
│   │       ├── s03_train.py        # 3DGS iterative optimization (SelectiveAdam + MCMC)
│   │       ├── s04_export.py       # Full Gaussian .spz & .ply compression exporter
│   │       ├── s05_track.py        # Interactive 3D raycast & SAM 2.1 tracking
│   │       ├── s06_diffuse.py      # FLUX.2 [klein] 4B & SDXL spatial inpainting
│   │       └── s07_rebuild.py      # Masked RoI 3D fine-tuning with frozen background
│   └── tests/                      # 54 worker unit & integration tests
│
├── frontend/                       # Next.js 16 App Router Web Studio
│   ├── public/
│   │   ├── demo-assets/            # Bundled offline demo rooms (realistic & splat)
│   │   └── textures/               # PBR textures (parquet, walnut, canvas art)
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx          # Root theme provider & viewport metadata
│   │   │   ├── page.tsx            # Project dashboard & video upload hero
│   │   │   └── projects/[id]/      # Interactive 3D Studio page
│   │   ├── components/
│   │   │   ├── Viewer3D.tsx        # WebGL2 Three.js / Spark 3D viewer (60 FPS)
│   │   │   ├── EditStudio.tsx      # Multi-frame 2D keyframe selector & seed cards
│   │   │   ├── FloatingEditBar.tsx # Quick-action AI transformation floating bar
│   │   │   ├── ProgressPanel.tsx   # Real-time task telemetry & stage progress
│   │   │   └── AuthModal.tsx       # User authentication modal dialog
│   │   └── lib/
│   │       ├── api.ts              # Typed frontend API client & XMLHttpRequest upload
│   │       ├── splatLoader.ts      # SPZ / PLY parser & Three.js Gaussian renderer
│   │       └── cameraMath.ts       # Matrix transforms, raycasting, camera interpolation
│   └── tests/                      # Vitest unit & component test suite
│
├── data/                           # Local runtime database & storage
│   ├── room3d.db                   # SQLite database (auto-created)
│   └── storage/                    # Object storage root for scenes & masks
│
├── scripts/
│   ├── seed_demo.py                # Populates SQLite DB & storage with initial demo rooms
│   └── generate_pbr_textures.py    # Generates high-resolution PBR procedural textures
│
├── start_room3d.bat                # Windows one-click desktop launcher
├── AGENTS.md                       # Repository engineering standards & license guardrails
└── README.md                       # Project documentation & operational guide
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
👉 **[http://localhost:3000/projects/realistic-room](http://localhost:3000/projects/realistic-room)**

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

RoomForge maintains a strict test suite runnable locally on CPU (all 67 tests passing):

```bash
# 1. Run all worker and backend unit/integration tests (67 tests)
$env:PYTHONPATH = "backend;worker"
pytest worker/tests backend/tests -v

# 2. Run frontend TypeScript typecheck
cd frontend
npm run typecheck

# 3. Build optimized production frontend
npm run build
```

---

## 👤 Author & Maintainer

**Mohamed Mostafa Elbasyouni**  
GitHub: [@markegyptian55-cloud](https://github.com/markegyptian55-cloud)  
Email: [markegyptian55@gmail.com](mailto:markegyptian55@gmail.com)

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.
