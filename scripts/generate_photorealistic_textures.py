from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TEXTURES_DIR = Path("frontend/public/textures")
TEXTURES_DIR.mkdir(parents=True, exist_ok=True)


def create_herringbone_parquet(size: int = 1024) -> None:
    """Generate high-resolution herringbone oak parquet texture with wood grain."""
    img = Image.new("RGB", (size, size), (195, 155, 110))
    draw = ImageDraw.Draw(img)

    # Base warm wood gradient
    arr = np.zeros((size, size, 3), dtype=np.float32)
    base_color = np.array([195.0, 150.0, 105.0])

    # Add Perlin-like wood grain noise
    np.random.seed(42)
    x = np.linspace(0, 10, size)
    y = np.linspace(0, 80, size)
    xx, yy = np.meshgrid(x, y)
    grain = np.sin(xx + np.sin(yy * 0.2) * 2.0) * 12.0
    fine_grain = (np.random.rand(size, size) - 0.5) * 18.0

    for c in range(3):
        arr[:, :, c] = np.clip(base_color[c] + grain + fine_grain, 0, 255)

    base_img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(base_img)

    # Draw herringbone slats
    slat_w = 48
    slat_h = 192
    groove_color = (75, 52, 32)

    for iy in range(-slat_h, size + slat_h, slat_w):
        for ix in range(-slat_h, size + slat_h, slat_h):
            # Alternating plank shading
            tint = int(np.random.uniform(-15, 18))
            draw.rectangle([ix, iy, ix + slat_h, iy + slat_w], outline=groove_color, width=2)
            # Subtle plank tone shift
            if tint != 0:
                overlay = Image.new("RGBA", (slat_h, slat_w), (tint if tint > 0 else 0, tint if tint > 0 else 0, 0, abs(tint) * 2))
                base_img.paste(overlay, (ix, iy), overlay)

    base_img = base_img.filter(ImageFilter.SMOOTH_MORE)
    base_img.save(TEXTURES_DIR / "herringbone_parquet.jpg", quality=95)
    print("Generated herringbone_parquet.jpg")


def create_walnut_wood(size: int = 1024) -> None:
    """Generate rich walnut wood grain texture."""
    arr = np.zeros((size, size, 3), dtype=np.float32)
    walnut_base = np.array([92.0, 62.0, 38.0])

    np.random.seed(123)
    x = np.linspace(0, 4, size)
    y = np.linspace(0, 60, size)
    xx, yy = np.meshgrid(x, y)
    rings = np.sin(yy * 0.4 + np.sin(xx * 0.8) * 4.0) * 24.0
    fiber = (np.random.rand(size, size) - 0.5) * 15.0

    for c in range(3):
        mult = [1.0, 0.78, 0.55][c]
        arr[:, :, c] = np.clip((walnut_base[c] + rings * mult + fiber), 0, 255)

    img = Image.fromarray(arr.astype(np.uint8))
    img.save(TEXTURES_DIR / "walnut_wood.jpg", quality=95)
    print("Generated walnut_wood.jpg")


def create_carrara_marble(size: int = 1024) -> None:
    """Generate Carrara white marble with grey and soft gold veins."""
    arr = np.full((size, size, 3), 246.0, dtype=np.float32)

    np.random.seed(999)
    # Veining functions
    x = np.linspace(0, 8, size)
    y = np.linspace(0, 8, size)
    xx, yy = np.meshgrid(x, y)

    v1 = np.sin(xx * 1.2 + np.cos(yy * 0.8) * 3.5 + np.sin((xx + yy) * 0.5) * 2.0)
    v2 = np.cos(yy * 1.5 - np.sin(xx * 0.6) * 4.0)
    mask1 = np.exp(-((v1 - 0.2) ** 2) / 0.04) * 45.0
    mask2 = np.exp(-((v2 + 0.1) ** 2) / 0.03) * 35.0

    for c in range(3):
        grey_mod = [1.0, 1.0, 1.05][c]
        arr[:, :, c] = np.clip(arr[:, :, c] - (mask1 * grey_mod + mask2 * 0.9), 0, 255)

    img = Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    img.save(TEXTURES_DIR / "marble_white.jpg", quality=95)
    print("Generated marble_white.jpg")


def create_modern_art_canvas(style: str, filename: str, size: tuple[int, int] = (1024, 512)) -> None:
    """Generate high-resolution contemporary fine art for the wall canvas."""
    img = Image.new("RGB", size, (242, 238, 230))
    draw = ImageDraw.Draw(img)
    w, h = size

    if style == "default_abstract":
        # Luxury modern abstract: Ochre, Navy, Sage, and Gold leaf accents
        # Background gradient
        for y in range(h):
            t = y / h
            r = int(240 * (1 - t * 0.3) + 210 * (t * 0.3))
            g = int(235 * (1 - t * 0.3) + 205 * (t * 0.3))
            b = int(225 * (1 - t * 0.3) + 190 * (t * 0.3))
            draw.line([(0, y), (w, y)], fill=(r, g, b))

        # Big expressive color blocks
        draw.polygon([(w * 0.08, h * 0.15), (w * 0.45, h * 0.08), (w * 0.38, h * 0.88), (w * 0.12, h * 0.75)], fill=(32, 54, 88))
        draw.polygon([(w * 0.35, h * 0.25), (w * 0.78, h * 0.18), (w * 0.72, h * 0.82), (w * 0.42, h * 0.92)], fill=(215, 145, 55))
        draw.ellipse([w * 0.48, h * 0.18, w * 0.88, h * 0.78], fill=(135, 162, 142))
        draw.polygon([(w * 0.62, h * 0.35), (w * 0.95, h * 0.42), (w * 0.85, h * 0.95), (w * 0.58, h * 0.85)], fill=(45, 75, 65))
        # Gold leaf brushstroke accents
        for _ in range(25):
            gx = np.random.uniform(w * 0.2, w * 0.8)
            gy = np.random.uniform(h * 0.2, h * 0.8)
            gw = np.random.uniform(40, 140)
            draw.line([(gx, gy), (gx + gw, gy + np.random.uniform(-10, 10))], fill=(245, 205, 95), width=int(np.random.uniform(4, 12)))

    elif style == "green_emerald":
        # Emerald Botanical & Gold Luxury Art
        for y in range(h):
            t = y / h
            r = int(18 * (1 - t) + 32 * t)
            g = int(58 * (1 - t) + 95 * t)
            b = int(45 * (1 - t) + 72 * t)
            draw.line([(0, y), (w, y)], fill=(r, g, b))

        # Dynamic emerald swooshes and golden circles
        draw.ellipse([w * 0.15, h * 0.05, w * 0.75, h * 0.95], fill=(22, 101, 72))
        draw.polygon([(w * 0.45, h * 0.12), (w * 0.92, h * 0.22), (w * 0.82, h * 0.88), (w * 0.38, h * 0.78)], fill=(46, 139, 87))
        draw.ellipse([w * 0.65, h * 0.3, w * 0.9, h * 0.75], fill=(144, 238, 144))
        # Rich metallic gold streams
        for _ in range(35):
            gx = np.random.uniform(w * 0.1, w * 0.85)
            gy = np.random.uniform(h * 0.15, h * 0.85)
            gw = np.random.uniform(60, 200)
            draw.line([(gx, gy), (gx + gw, gy + np.random.uniform(-18, 18))], fill=(255, 215, 0), width=int(np.random.uniform(5, 14)))

    elif style == "sunflowers":
        # Impressionist Golden Sunflowers in warm terracotta & amber
        for y in range(h):
            t = y / h
            draw.line([(0, y), (w, y)], fill=(int(245 - t * 40), int(220 - t * 50), int(170 - t * 40)))
        # Sunflower petals and brown centers
        centers = [(w * 0.35, h * 0.45), (w * 0.65, h * 0.52), (w * 0.5, h * 0.72)]
        for cx, cy in centers:
            # Petals
            for a in range(0, 360, 18):
                rad = math.radians(a)
                px = cx + math.cos(rad) * 90
                py = cy + math.sin(rad) * 90
                draw.line([(cx, cy), (px, py)], fill=(255, 195, 15), width=18)
            # Dark seed disk
            draw.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], fill=(85, 45, 20))

    img = img.filter(ImageFilter.GaussianBlur(1.0))
    img.save(TEXTURES_DIR / filename, quality=95)
    print(f"Generated {filename}")


if __name__ == "__main__":
    create_herringbone_parquet()
    create_walnut_wood()
    create_carrara_marble()
    create_modern_art_canvas("default_abstract", "art_modern_abstract.jpg")
    create_modern_art_canvas("green_emerald", "art_green_emerald.jpg")
    create_modern_art_canvas("sunflowers", "art_sunflowers.jpg")
    print("✅ All high-resolution PBR textures created successfully!")
