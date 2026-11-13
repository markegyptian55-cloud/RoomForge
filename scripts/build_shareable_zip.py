from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = REPO_ROOT / "dist"
PORTABLE_DIR = DIST_DIR / "RoomForge-Portable"
ZIP_OUTPUT = DIST_DIR / "RoomForge-Portable-v1.0.0.zip"


def create_shareable_zip() -> None:
    print("=================================================================")
    print("📦  CREATING SHAREABLE PORTABLE ZIP ARCHIVE")
    print("=================================================================\n")

    # Update data directory in portable bundle
    data_src = REPO_ROOT / "data"
    data_dst = PORTABLE_DIR / "data"
    if data_src.exists():
        print("📁 Synchronizing pre-seeded rooms (including realistic-room) into portable bundle...")
        shutil.copytree(data_src, data_dst, dirs_exist_ok=True)

    if ZIP_OUTPUT.exists():
        ZIP_OUTPUT.unlink()

    print(f"🗜️  Compressing {PORTABLE_DIR} -> {ZIP_OUTPUT.name}...")
    with zipfile.ZipFile(ZIP_OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in PORTABLE_DIR.rglob("*"):
            if file.is_file():
                rel = file.relative_to(PORTABLE_DIR)
                zf.write(file, arcname=str(rel))

    size_mb = ZIP_OUTPUT.stat().st_size / (1024 * 1024)
    print("\n=================================================================")
    print(f"✅  ZIP ARCHIVE READY TO SHARE: {ZIP_OUTPUT}")
    print(f"📊  Archive Size: {size_mb:.2f} MB")
    print("=================================================================")


if __name__ == "__main__":
    create_shareable_zip()
