from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = REPO_ROOT / "dist"
PORTABLE_DIR = DIST_DIR / "RoomForge-Portable"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"⚙️  Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(cwd or REPO_ROOT), check=False)
    if res.returncode != 0:
        print(f"❌ Command failed with return code {res.returncode}")
        sys.exit(res.returncode)


def package_desktop() -> None:
    print("=================================================================")
    print("📦  ROOMFORGE DESKTOP PACKAGING PIPELINE")
    print("=================================================================\n")

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    if PORTABLE_DIR.exists():
        shutil.rmtree(PORTABLE_DIR)
    PORTABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Compile Native Launcher via PyInstaller
    print("🔨 Step 1: Compiling native launcher executable with custom icon...")
    icon_path = REPO_ROOT / "desktop" / "assets" / "icon.ico"
    launcher_src = REPO_ROOT / "desktop" / "launcher.py"

    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--onefile",
        f"--icon={icon_path}",
        "--name=RoomForge",
        f"--distpath={PORTABLE_DIR}",
        f"--workpath={DIST_DIR / 'build'}",
        f"--specpath={DIST_DIR}",
        str(launcher_src),
    ]
    run_cmd(pyinstaller_cmd)

    # Also keep a copy in dist/RoomForge/ for Inno Setup
    installer_stage = DIST_DIR / "RoomForge"
    installer_stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PORTABLE_DIR / "RoomForge.exe", installer_stage / "RoomForge.exe")

    # 2. Package USB Portable distribution
    print("\n📦 Step 2: Preparing self-contained USB portable distribution...")
    # Marker file
    (PORTABLE_DIR / "PORTABLE").touch()
    # Batch helper
    shutil.copy2(REPO_ROOT / "RoomForge.bat", PORTABLE_DIR / "RoomForge.bat")

    # Assets
    portable_assets = PORTABLE_DIR / "desktop" / "assets"
    portable_assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(icon_path, portable_assets / "icon.ico")
    shutil.copy2(REPO_ROOT / "desktop" / "assets" / "icon.png", portable_assets / "icon.png")

    # Copy project code folders needed for runtime
    for folder in ("backend", "worker", "frontend"):
        src = REPO_ROOT / folder
        dst = PORTABLE_DIR / folder
        if src.exists():
            print(f"   Copying {folder}/ ...")
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", ".pytest_cache", ".next", "node_modules", ".git"
                ),
            )

    # Copy pre-seeded offline demo data
    data_src = REPO_ROOT / "data"
    if data_src.exists():
        print("   Copying pre-seeded demo room data...")
        shutil.copytree(data_src, PORTABLE_DIR / "data", dirs_exist_ok=True)

    # 3. Check for Inno Setup Compiler
    print("\n📦 Step 3: Checking Inno Setup 6 installer compiler...")
    iscc_path = shutil.which("iscc") or shutil.which("ISCC.exe")
    standard_inno = Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe")
    if not iscc_path and standard_inno.exists():
        iscc_path = str(standard_inno)

    if iscc_path:
        print(f"   Found Inno Setup at: {iscc_path}")
        print("   Compiling RoomForge-Setup-v1.0.0.exe installer...")
        run_cmd([iscc_path, str(REPO_ROOT / "desktop" / "roomforge_setup.iss")])
    else:
        print("   ℹ️  Inno Setup 6 not found on PATH. Installer script ready at: desktop/roomforge_setup.iss")
        print("   Portable executable is ready to distribute!")

    print("\n=================================================================")
    print("✅  PACKAGING COMPLETED SUCCESSFULLY!")
    print(f"📁  Portable USB Bundle: {PORTABLE_DIR}")
    print(f"🚀  Standalone Binary:   {PORTABLE_DIR / 'RoomForge.exe'}")
    print("=================================================================")


if __name__ == "__main__":
    package_desktop()
