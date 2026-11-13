from __future__ import annotations

import math
from pathlib import Path
from PIL import Image, ImageDraw

def generate_roomforge_icon() -> Path:
    assets_dir = Path(__file__).resolve().parents[1] / "desktop" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    ico_path = assets_dir / "icon.ico"
    png_path = assets_dir / "icon.png"

    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. Outer rounded container with deep space background
    margin = 12
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=48,
        fill=(15, 23, 42, 255),
        outline=(38, 51, 66, 255),
        width=3,
    )

    # 2. Isometric 3D Room Cube Geometry
    cx, cy = size // 2, size // 2 + 6
    r = 76

    # 3 isometric corner vectors
    top = (cx, cy - r)
    top_right = (cx + int(r * math.cos(math.radians(30))), cy - int(r * math.sin(math.radians(30))))
    bottom_right = (cx + int(r * math.cos(math.radians(30))), cy + int(r * math.sin(math.radians(30))))
    bottom = (cx, cy + r)
    bottom_left = (cx - int(r * math.cos(math.radians(30))), cy + int(r * math.sin(math.radians(30))))
    top_left = (cx - int(r * math.cos(math.radians(30))), cy - int(r * math.sin(math.radians(30))))
    center = (cx, cy)

    # Top ceiling face (Cyan/Teal tint)
    draw.polygon([center, top_left, top, top_right], fill=(16, 185, 129, 230), outline=(52, 211, 153, 255))

    # Left interior wall face (Deep emerald)
    draw.polygon([center, top_left, bottom_left, bottom], fill=(6, 78, 59, 240), outline=(16, 185, 129, 255))

    # Right interior wall face (Navy / Slate blue)
    draw.polygon([center, top_right, bottom_right, bottom], fill=(30, 41, 59, 240), outline=(56, 189, 248, 255))

    # 3. Floating 3D Splat Node in center (Glowing Cyan/Gold)
    node_r = 14
    for offset in range(node_r, 0, -2):
        alpha = int(255 * (1.0 - offset / (node_r + 4)))
        draw.ellipse(
            [cx - offset, cy - offset, cx + offset, cy + offset],
            fill=(245, 158, 11, alpha),
        )
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 255, 255, 255))

    # Save high-res PNG
    img.save(png_path, format="PNG")

    # Generate multi-size Windows .ICO
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(ico_path, format="ICO", sizes=sizes)
    print(f"Successfully generated RoomForge icons:\n - {ico_path}\n - {png_path}")
    return ico_path

if __name__ == "__main__":
    generate_roomforge_icon()
