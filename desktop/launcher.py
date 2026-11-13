from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def detect_gpu_hardware() -> str:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout.strip():
            name = res.stdout.strip().splitlines()[0]
            return f"NVIDIA GPU Detected: {name} (Full Local AI Acceleration)"
    except Exception:
        pass
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
            for line in lines:
                if "nvidia" in line.lower() or "geforce" in line.lower() or "rtx" in line.lower():
                    return f"NVIDIA GPU Detected: {line} (Full Local AI Acceleration)"
            if lines:
                return f"GPU Detected: {lines[0]} (High-Speed WebGL2 3D Viewer Mode)"
    except Exception:
        pass
    return "Standard Graphics (High-Speed WebGL2 3D Viewer Mode)"


def launch_roomforge() -> None:
    parser = argparse.ArgumentParser(description="RoomForge Desktop Launcher")
    parser.add_argument("--portable", action="store_true", help="Run in self-contained USB portable mode")
    parser.add_argument("--port-backend", type=int, default=8000, help="Backend API port")
    parser.add_argument("--port-frontend", type=int, default=3000, help="Frontend Studio port")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    backend_dir = repo_root / "backend"
    frontend_dir = repo_root / "frontend"

    # Resolve Data Directory
    if args.portable or (repo_root / "PORTABLE").exists():
        data_dir = repo_root / "data"
        mode_str = "USB Portable Mode (Relative Data Store)"
    else:
        appdata = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA") or str(repo_root)
        data_dir = Path(appdata) / "RoomForge" / "data"
        mode_str = "Installed Desktop Mode"

    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "room3d.db"
    storage_path = data_dir / "storage"
    storage_path.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("🏛️  ROOMFORGE DESKTOP STUDIO")
    print(f"📦 Runtime Mode: {mode_str}")
    print(f"🖥️  Graphics:    {detect_gpu_hardware()}")
    print(f"📂 Data Store:  {data_dir}")
    print("=================================================================")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{backend_dir}{os.pathsep}{repo_root / 'worker'}"
    env["ROOM3D_DATABASE_URL"] = f"sqlite:///{db_path}"
    env["ROOM3D_LOCAL_STORAGE_DIR"] = str(storage_path)
    env["ROOM3D_AUTH_REQUIRED"] = "false"

    backend_port = args.port_backend
    frontend_port = args.port_frontend

    # 1. Start FastAPI Control Plane
    backend_proc: subprocess.Popen[bytes] | None = None
    if not is_port_in_use(backend_port):
        print(f"🚀 Starting Control Plane on http://127.0.0.1:{backend_port}...")
        backend_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(backend_port),
            ],
            cwd=str(backend_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        print(f"ℹ️  Control Plane already active on port {backend_port}.")

    # 2. Start Next.js Studio
    frontend_proc: subprocess.Popen[bytes] | None = None
    if not is_port_in_use(frontend_port):
        print(f"🚀 Starting Web Studio on http://127.0.0.1:{frontend_port}...")
        frontend_proc = subprocess.Popen(
            ["npm.cmd" if os.name == "nt" else "npm", "run", "dev", "--", "-p", str(frontend_port)],
            cwd=str(frontend_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        print(f"ℹ️  Web Studio already active on port {frontend_port}.")

    studio_url = f"http://localhost:{frontend_port}/projects/demo-room"
    print("\n⏳ Initializing 3D Studio services...")
    time.sleep(4)

    if not args.no_browser:
        print(f"🌐 Opening {studio_url} in your default browser...")
        webbrowser.open(studio_url)

    print("\n✅ RoomForge is running smoothly.")
    print("Press Ctrl+C in this window to stop all services and exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down RoomForge background services...")
        if backend_proc:
            backend_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()
        print("Goodbye!")


if __name__ == "__main__":
    launch_roomforge()
