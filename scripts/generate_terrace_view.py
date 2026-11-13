from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TEXTURES_DIR = Path("frontend/public/textures")
TEXTURES_DIR.mkdir(parents=True, exist_ok=True)


def create_outdoor_terrace_view(width: int = 1024, height: int = 512) -> None:
    """Generate high-quality sunny outdoor skyline and terrace view for the window."""
    img = Image.new("RGB", (width, height), (135, 206, 235))
    draw = ImageDraw.Draw(img)

    # 1. Sky gradient (Sunny azure blue to warm horizon)
    for y in range(int(height * 0.7)):
        t = y / (height * 0.7)
        r = int(100 * (1 - t) + 215 * t)
        g = int(170 * (1 - t) + 235 * t)
        b = int(245 * (1 - t) + 255 * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # 2. Distant soft clouds
    np.random.seed(42)
    for _ in range(12):
        cx = np.random.uniform(50, width - 50)
        cy = np.random.uniform(40, height * 0.35)
        rx = np.random.uniform(80, 220)
        ry = np.random.uniform(20, 50)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(255, 255, 255, 160))

    # 3. Distant City Skyline (Silhouettes of modern skyscrapers)
    buildings = [
        (40, 180, 120),
        (140, 240, 90),
        (220, 160, 140),
        (340, 290, 110),
        (430, 220, 150),
        (560, 310, 130),
        (670, 200, 100),
        (750, 270, 140),
        (870, 190, 120),
        (960, 250, 80),
    ]
    horizon_y = int(height * 0.65)
    for bx, bh, bw in buildings:
        by = horizon_y - bh
        # Building body with subtle blue haze
        draw.rectangle([bx, by, bx + bw, horizon_y], fill=(160, 185, 205), outline=(135, 160, 180))
        # Building glass grid
        for wy in range(by + 15, horizon_y - 20, 18):
            for wx in range(bx + 10, bx + bw - 10, 14):
                draw.rectangle([wx, wy, wx + 8, wy + 10], fill=(210, 230, 245))

    # 4. Sunny Terrace Floor (Warm stone tiles)
    for y in range(horizon_y, height):
        t = (y - horizon_y) / (height - horizon_y)
        r = int(210 * (1 - t) + 180 * t)
        g = int(205 * (1 - t) + 175 * t)
        b = int(195 * (1 - t) + 165 * t)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # 5. Modern Glass Balcony Railing
    rail_y = int(height * 0.62)
    draw.line([(0, rail_y), (width, rail_y)], fill=(80, 90, 100), width=6)
    for rx in range(0, width, 140):
        draw.line([(rx, rail_y), (rx, height)], fill=(90, 100, 110), width=4)

    img = img.filter(ImageFilter.GaussianBlur(0.8))
    out_path = TEXTURES_DIR / "window_view.jpg"
    img.save(out_path, quality=95)
    print(f"Generated clean outdoor terrace view: {out_path}")


if __name__ == "__main__":
    create_outdoor_terrace_view()
