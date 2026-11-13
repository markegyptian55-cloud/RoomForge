from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for subdir in ("backend", "worker"):
    path_str = str(REPO_ROOT / subdir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from app.config import get_settings  # noqa: E402
from app.db import Base, create_engine, sessionmaker  # noqa: E402
from app.models import EditSession, Project, SceneVersion, Task  # noqa: E402
from room3d_worker.pipeline.s04_export import ExportRequest, export_scene  # noqa: E402
from room3d_worker.pipeline.s07_rebuild import GaussianState, save_gaussian_state  # noqa: E402


def _sample_plane(
    axis: str,
    const: float,
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    count: int,
    color_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    u = np.random.uniform(u_range[0], u_range[1], count).astype(np.float32)
    v = np.random.uniform(v_range[0], v_range[1], count).astype(np.float32)
    pts = np.zeros((count, 3), dtype=np.float32)
    if axis == "y":
        pts[:, 0], pts[:, 1], pts[:, 2] = u, const, v
    elif axis == "z":
        pts[:, 0], pts[:, 1], pts[:, 2] = u, v, const
    elif axis == "x":
        pts[:, 0], pts[:, 1], pts[:, 2] = const, u, v
    colors = color_fn(u, v)
    return pts, colors


def _sample_box_surface(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    count: int,
    base_color: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    # Distribute points on the 6 outer faces of the box
    dx = x_range[1] - x_range[0]
    dy = y_range[1] - y_range[0]
    dz = z_range[1] - z_range[0]
    areas = np.array([dy * dz, dy * dz, dx * dz, dx * dz, dx * dy, dx * dy], dtype=np.float64)
    probs = areas / areas.sum()
    face_counts = np.random.multinomial(count, probs)

    all_pts: list[np.ndarray] = []
    # Face 0: -X face
    if face_counts[0] > 0:
        y = np.random.uniform(y_range[0], y_range[1], face_counts[0]).astype(np.float32)
        z = np.random.uniform(z_range[0], z_range[1], face_counts[0]).astype(np.float32)
        x = np.full(face_counts[0], x_range[0], dtype=np.float32)
        all_pts.append(np.column_stack([x, y, z]))
    # Face 1: +X face
    if face_counts[1] > 0:
        y = np.random.uniform(y_range[0], y_range[1], face_counts[1]).astype(np.float32)
        z = np.random.uniform(z_range[0], z_range[1], face_counts[1]).astype(np.float32)
        x = np.full(face_counts[1], x_range[1], dtype=np.float32)
        all_pts.append(np.column_stack([x, y, z]))
    # Face 2: -Y face
    if face_counts[2] > 0:
        x = np.random.uniform(x_range[0], x_range[1], face_counts[2]).astype(np.float32)
        z = np.random.uniform(z_range[0], z_range[1], face_counts[2]).astype(np.float32)
        y = np.full(face_counts[2], y_range[0], dtype=np.float32)
        all_pts.append(np.column_stack([x, y, z]))
    # Face 3: +Y face
    if face_counts[3] > 0:
        x = np.random.uniform(x_range[0], x_range[1], face_counts[3]).astype(np.float32)
        z = np.random.uniform(z_range[0], z_range[1], face_counts[3]).astype(np.float32)
        y = np.full(face_counts[3], y_range[1], dtype=np.float32)
        all_pts.append(np.column_stack([x, y, z]))
    # Face 4: -Z face
    if face_counts[4] > 0:
        x = np.random.uniform(x_range[0], x_range[1], face_counts[4]).astype(np.float32)
        y = np.random.uniform(y_range[0], y_range[1], face_counts[4]).astype(np.float32)
        z = np.full(face_counts[4], z_range[0], dtype=np.float32)
        all_pts.append(np.column_stack([x, y, z]))
    # Face 5: +Z face
    if face_counts[5] > 0:
        x = np.random.uniform(x_range[0], x_range[1], face_counts[5]).astype(np.float32)
        y = np.random.uniform(y_range[0], y_range[1], face_counts[5]).astype(np.float32)
        z = np.full(face_counts[5], z_range[1], dtype=np.float32)
        all_pts.append(np.column_stack([x, y, z]))

    pts = np.vstack(all_pts)
    base = np.array(base_color, dtype=np.float32)
    # Organic shading noise
    jitter = np.random.normal(0, 7.0, (len(pts), 3)).astype(np.float32)
    colors = np.clip(base + jitter, 0, 255).astype(np.uint8)
    return pts, colors


def _build_dense_room(sofa_color: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(1337)
    pts_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []
    is_sofa_list: list[np.ndarray] = []

    def add_geom(pts: np.ndarray, clrs: np.ndarray, sofa: bool = False) -> None:
        pts_list.append(pts)
        colors_list.append(clrs)
        is_sofa_list.append(np.full(len(pts), sofa, dtype=bool))

    # 1. Floor: Oak parquet plank pattern (18,000 points)
    def floor_shader(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        plank_id = (np.floor(u / 0.18) + np.floor(v / 0.9) * 5).astype(int)
        tones = np.array([
            [196, 150, 105],
            [182, 136, 92],
            [168, 124, 82],
            [155, 112, 72],
            [205, 160, 115],
        ], dtype=np.float32)
        base = tones[plank_id % len(tones)]
        grain = 6.0 * np.sin(u * 50.0 + v * 5.0)[:, None]
        # Seam darkening near plank boundaries
        du = np.abs(u - np.round(u / 0.18) * 0.18)
        dv = np.abs(v - np.round(v / 0.9) * 0.9)
        seam = ((du < 0.012) | (dv < 0.015))[:, None]
        base = np.where(seam, np.array([85, 60, 40], dtype=np.float32), base + grain)
        # Ambient occlusion near walls
        ao = np.clip(1.0 - 0.25 * ((np.abs(u) > 2.6) | (np.abs(v) > 2.6)), 0.6, 1.0)[:, None]
        return np.clip(base * ao, 0, 255).astype(np.uint8)

    p, c = _sample_plane("y", 0.0, (-3.0, 3.0), (-3.0, 3.0), 18000, floor_shader)
    add_geom(p, c)

    # 2. Ceiling: Warm off-white with recessed downlight halos (8,000 points)
    def ceiling_shader(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        base = np.full((len(u), 3), [238, 238, 240], dtype=np.float32)
        for lx, lz in [(-1.2, 0.6), (1.2, 0.6), (-1.2, 2.0), (1.2, 2.0)]:
            dist = np.hypot(u - lx, v - lz)
            halo = np.exp(-dist * 2.5)[:, None]
            base = base + halo * np.array([17, 12, -10], dtype=np.float32)
        return np.clip(base, 0, 255).astype(np.uint8)

    p, c = _sample_plane("y", 2.65, (-3.0, 3.0), (-3.0, 3.0), 8000, ceiling_shader)
    add_geom(p, c)

    # 3. Back Wall (Z = 3.0): Gallery plaster + dark baseboard (10,000 points)
    def back_wall_shader(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        base = np.full((len(u), 3), [232, 230, 226], dtype=np.float32)
        # Baseboard moulding at bottom
        baseboard = (v < 0.14)[:, None]
        base = np.where(baseboard, np.array([42, 48, 56], dtype=np.float32), base)
        noise = np.random.normal(0, 3.0, base.shape).astype(np.float32)
        return np.clip(base + noise, 0, 255).astype(np.uint8)

    p, c = _sample_plane("z", 3.0, (-3.0, 3.0), (0.0, 2.65), 10000, back_wall_shader)
    add_geom(p, c)

    # 4. Modern Canvas Abstract Art above Sofa (Z = 2.97, 4,500 points)
    def art_canvas_shader(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        # Abstract strokes: azure blue, terracotta, gold, ivory, charcoal
        s1 = np.sin(u * 3.5 + v * 4.0)
        s2 = np.cos(u * 5.0 - v * 2.5)
        clrs = np.zeros((len(u), 3), dtype=np.float32)
        for i in range(len(u)):
            val = s1[i] + s2[i]
            if val > 0.8:
                clrs[i] = [28, 115, 225]  # Cobalt Azure
            elif val > 0.2:
                clrs[i] = [224, 94, 46]   # Warm Terracotta
            elif val > -0.4:
                clrs[i] = [225, 178, 48]  # Rich Ochre Gold
            elif val > -1.0:
                clrs[i] = [248, 246, 240] # Ivory White
            else:
                clrs[i] = [32, 36, 44]    # Charcoal Accent
        # Black floating frame edge
        edge = (np.abs(u - (-0.5)) > 1.05) | (np.abs(v - 1.8) > 0.5)
        clrs[edge] = [18, 20, 24]
        return np.clip(clrs, 0, 255).astype(np.uint8)

    p, c = _sample_plane("z", 2.97, (-1.6, 0.6), (1.25, 2.35), 4500, art_canvas_shader)
    add_geom(p, c)

    # 5. Left Wall (X = -3.0): Panoramic glass window to terrace (10,000 points)
    def window_shader(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        # u is Y [0, 2.65], v is Z [-3, 3]
        clrs = np.zeros((len(u), 3), dtype=np.float32)
        # Outside scenery through glass
        sky_t = np.clip((u - 0.9) / 1.7, 0.0, 1.0)[:, None]
        sky = (1.0 - sky_t) * np.array([195, 220, 248]) + sky_t * np.array([125, 180, 235])
        balustrade = (u < 0.9) & (u > 0.85)
        terrace_floor = u <= 0.08
        clrs[:] = sky
        clrs[balustrade] = [220, 225, 230]
        clrs[terrace_floor] = [175, 180, 185]
        # Black aluminum window frames
        mullion_z = (np.abs(v + 1.6) < 0.05) | (np.abs(v) < 0.05) | (np.abs(v - 1.6) < 0.05)
        mullion_y = (np.abs(u - 1.3) < 0.04) | (u > 2.55) | (u < 0.04) | (np.abs(v) > 2.5)
        frame = mullion_z | mullion_y
        clrs[frame] = [28, 30, 35]
        return np.clip(clrs, 0, 255).astype(np.uint8)

    p, c = _sample_plane("x", -3.0, (0.0, 2.65), (-3.0, 3.0), 10000, window_shader)
    add_geom(p, c)

    # 6. Right Wall (X = 3.0): Sage accent wall with brass sconces (8,000 points)
    def right_wall_shader(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        base = np.full((len(u), 3), [192, 202, 194], dtype=np.float32)
        for sz in [-0.5, 1.5]:
            dist = np.hypot(u - 1.65, v - sz)
            sconce_light = np.exp(-dist * 2.8)[:, None]
            base = base + sconce_light * np.array([45, 30, -20], dtype=np.float32)
        baseboard = (u < 0.14)[:, None]
        base = np.where(baseboard, np.array([42, 48, 56], dtype=np.float32), base)
        return np.clip(base, 0, 255).astype(np.uint8)

    p, c = _sample_plane("x", 3.0, (0.0, 2.65), (-3.0, 3.0), 8000, right_wall_shader)
    add_geom(p, c)

    # 7. Front Wall (Z = -3.0): Soft neutral plaster (6,000 points)
    p, c = _sample_plane("z", -3.0, (-3.0, 3.0), (0.0, 2.65), 6000, lambda u, v: np.full((len(u), 3), [225, 222, 218], dtype=np.uint8))
    add_geom(p, c)

    # 8. Persian Area Rug under Seating (Y = 0.012, 8,000 points)
    def rug_shader(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        # Center medallion + ornate border
        dist_c = np.hypot(u - (-0.6), v - 1.25)
        border = (u < -2.0) | (u > 0.8) | (v < 0.18) | (v > 2.35)
        medallion = dist_c < 0.42
        clrs = np.full((len(u), 3), [238, 232, 220], dtype=np.float32)  # Cream ivory field
        clrs[border] = [178, 65, 48]  # Terracotta rust border
        clrs[medallion] = [165, 52, 38]
        # Navy geometric floral details
        detail = (np.sin(u * 25.0) * np.cos(v * 25.0) > 0.5) & (~border)
        clrs[detail] = [38, 55, 92]
        return np.clip(clrs, 0, 255).astype(np.uint8)

    p, c = _sample_plane("y", 0.012, (-2.15, 0.95), (0.08, 2.45), 8000, rug_shader)
    add_geom(p, c)

    # 9. Luxury Sectional Sofa (Target Object, ~20,000 points)
    # 9a. Main seat base
    p, c = _sample_box_surface((-1.8, 0.65), (0.0, 0.44), (1.15, 2.35), 7000, sofa_color)
    add_geom(p, c, sofa=True)
    # 9b. Backrest
    p, c = _sample_box_surface((-1.8, 0.65), (0.44, 0.86), (2.05, 2.38), 5000, sofa_color)
    add_geom(p, c, sofa=True)
    # 9c. Left armrest
    p, c = _sample_box_surface((-1.98, -1.75), (0.0, 0.64), (1.15, 2.38), 2500, sofa_color)
    add_geom(p, c, sofa=True)
    # 9d. Chaise lounge extension
    p, c = _sample_box_surface((0.15, 0.65), (0.0, 0.44), (0.45, 1.15), 3500, sofa_color)
    add_geom(p, c, sofa=True)
    # 9e. Throw accent pillows (Gold Velvet & Cream)
    p, c = _sample_box_surface((-1.55, -1.15), (0.44, 0.72), (1.9, 2.08), 1200, [217, 119, 6])
    add_geom(p, c, sofa=True)
    p, c = _sample_box_surface((0.0, 0.38), (0.44, 0.72), (1.9, 2.08), 1200, [217, 119, 6])
    add_geom(p, c, sofa=True)

    # 10. Live-Edge Walnut Coffee Table + Props (~5,000 points)
    # Table top
    p, c = _sample_box_surface((-1.05, 0.05), (0.34, 0.38), (0.35, 0.95), 3000, [112, 74, 42])
    add_geom(p, c)
    # 4 Steel Hairpin Legs
    for lx, lz in [(-1.0, 0.4), (-0.02, 0.4), (-1.0, 0.9), (-0.02, 0.9)]:
        p, c = _sample_box_surface((lx, lx + 0.04), (0.0, 0.34), (lz, lz + 0.04), 300, [30, 32, 36])
        add_geom(p, c)
    # Tabletop Book
    p, c = _sample_box_surface((-0.78, -0.45), (0.38, 0.41), (0.48, 0.78), 600, [245, 245, 240])
    add_geom(p, c)
    # Ceramic Coffee Mug
    p, c = _sample_box_surface((-0.22, -0.1), (0.38, 0.48), (0.55, 0.67), 500, [215, 110, 75])
    add_geom(p, c)

    # 11. Corner Fiddle Leaf Fig Plant (~4,000 points)
    # Pot
    pot_u = np.random.uniform(0.0, 0.52, 1200).astype(np.float32)
    pot_ang = np.random.uniform(0.0, 2.0 * math.pi, 1200).astype(np.float32)
    pot_r = 0.24
    pot_x = -2.4 + pot_r * np.cos(pot_ang)
    pot_z = 2.45 + pot_r * np.sin(pot_ang)
    pot_pts = np.column_stack([pot_x, pot_u, pot_z]).astype(np.float32)
    pot_clrs = np.full((len(pot_pts), 3), [195, 95, 62], dtype=np.uint8)
    add_geom(pot_pts, pot_clrs)
    # Foliage Canopy
    leaf_u = np.random.uniform(0.52, 1.85, 2800).astype(np.float32)
    leaf_ang = np.random.uniform(0.0, 2.0 * math.pi, 2800).astype(np.float32)
    leaf_r = np.random.uniform(0.05, 0.45, 2800).astype(np.float32)
    leaf_x = -2.4 + leaf_r * np.cos(leaf_ang)
    leaf_z = 2.45 + leaf_r * np.sin(leaf_ang)
    leaf_pts = np.column_stack([leaf_x, leaf_u, leaf_z]).astype(np.float32)
    green_shades = np.array([[22, 101, 52], [34, 197, 94], [21, 128, 61], [15, 80, 40]], dtype=np.uint8)
    leaf_clrs = green_shades[np.random.randint(0, len(green_shades), len(leaf_pts))]
    add_geom(leaf_pts, leaf_clrs)

    # 12. Designer Brass Arc Lamp (~1,800 points)
    lamp_t = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    lamp_x = 1.6 - 0.85 * lamp_t ** 1.8
    lamp_y = lamp_t * 2.15
    lamp_z = 2.1 - 0.75 * lamp_t
    lamp_pts = np.column_stack([lamp_x, lamp_y, lamp_z]).astype(np.float32)
    lamp_clrs = np.full((len(lamp_pts), 3), [215, 175, 95], dtype=np.uint8)
    add_geom(lamp_pts, lamp_clrs)
    # Lampshade
    p, c = _sample_box_surface((0.65, 0.85), (1.96, 2.14), (1.25, 1.45), 800, [255, 248, 225])
    add_geom(p, c)

    all_pts = np.vstack(pts_list)
    all_colors = np.vstack(colors_list)
    all_is_sofa = np.concatenate(is_sofa_list)
    return all_pts, all_colors, all_is_sofa


def _make_camera_poses(num_views: int = 8) -> list[dict[str, Any]]:
    frames = []
    target = np.array([0.0, 0.75, 0.5])
    radius = 2.7
    height = 1.35
    for idx in range(num_views):
        angle = 2.0 * math.pi * idx / num_views
        eye = np.array([radius * math.sin(angle), height, radius * math.cos(angle)])
        forward = (target - eye) / np.linalg.norm(target - eye)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = right
        c2w[:3, 1] = -true_up
        c2w[:3, 2] = forward
        c2w[:3, 3] = eye
        w2c = np.linalg.inv(c2w)
        frames.append({
            "frame_idx": idx,
            "orig_video_frame": idx * 10,
            "timestamp_sec": float(idx * 0.5),
            "image_relpath": f"images/frame_{idx:06d}.jpg",
            "colmap_image_id": idx + 1,
            "cam_from_world": w2c.tolist(),
        })
    return frames


def _render_synthetic_frames(
    version_dir: Path,
    frames: list[dict[str, Any]],
    pts: np.ndarray,
    colors: np.ndarray,
    is_sofa: np.ndarray,
    width: int = 640,
    height: int = 480,
    fx: float = 520.0,
    fy: float = 520.0,
) -> None:
    images_dir = version_dir / "images"
    masks_dir = version_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    cx = width / 2.0
    cy = height / 2.0
    pts_homo = np.column_stack([pts, np.ones(len(pts), dtype=np.float32)])

    for frame in frames:
        idx = int(frame["frame_idx"])
        w2c = np.asarray(frame["cam_from_world"], dtype=np.float32)
        p_cam = (w2c @ pts_homo.T).T[:, :3]
        z_c = p_cam[:, 2]

        valid = z_c > 0.25
        u = cx + fx * (p_cam[:, 0] / z_c)
        v = cy + fy * (p_cam[:, 1] / z_c)
        in_bounds = valid & (u >= 1) & (u < width - 2) & (v >= 1) & (v < height - 2)

        u_pix = u[in_bounds].astype(np.int32)
        v_pix = v[in_bounds].astype(np.int32)
        z_vals = z_c[in_bounds]
        c_vals = colors[in_bounds]
        sofa_vals = is_sofa[in_bounds]

        # Painter's depth sorting: render farthest points first
        order = np.argsort(-z_vals)
        u_sorted = u_pix[order]
        v_sorted = v_pix[order]
        c_sorted = c_vals[order]
        sofa_sorted = sofa_vals[order]

        # Ambient backdrop gradient
        img_buffer = np.full((height, width, 3), [16, 22, 34], dtype=np.uint8)
        mask_buffer = np.zeros((height, width), dtype=np.uint8)

        # Splat 2x2 footprint for seamless solid coverage
        for di in range(2):
            for dj in range(2):
                img_buffer[v_sorted + di, u_sorted + dj] = c_sorted
                sofa_mask = sofa_sorted.astype(np.uint8) * 255
                mask_buffer[v_sorted + di, u_sorted + dj] = sofa_mask

        Image.fromarray(img_buffer).save(images_dir / f"frame_{idx:06d}.jpg", quality=92)
        Image.fromarray(mask_buffer).save(masks_dir / f"mask_{idx:06d}.png")


def _seed_version(
    storage_dir: Path,
    project_id: str,
    version_id: str,
    parent_id: str | None,
    sofa_color: list[int],
    frames: list[dict[str, Any]],
) -> Path:
    version_dir = storage_dir / project_id / version_id
    (version_dir / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (version_dir / "sparse" / "0" / name).write_bytes(f"demo-{name}".encode("utf-8"))

    pts, colors, is_sofa = _build_dense_room(sofa_color)
    _render_synthetic_frames(version_dir, frames, pts, colors, is_sofa, width=640, height=480, fx=520.0, fy=520.0)

    opacity = np.full(len(pts), 0.95, dtype=np.float32)
    scales = np.full((len(pts), 3), 0.05, dtype=np.float32)
    export_scene(
        ExportRequest(
            output_dir=version_dir,
            project_id=project_id,
            version_id=version_id,
            parent_version_id=parent_id,
            frames=frames,
            width=640,
            height=480,
            intrinsics={"fx": 520.0, "fy": 520.0, "cx": 320.0, "cy": 240.0},
            T_viewer_from_colmap=np.eye(4).tolist(),
            pose_backend="colmap_global",
        ),
        pts,
        opacity,
        scales,
        colors=colors,
    )
    save_gaussian_state(version_dir, GaussianState(pts, colors, opacity, scales))
    return version_dir


def seed_demo_room(project_id: str = "demo-room") -> str:
    settings = get_settings()
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(engine, expire_on_commit=False)
    storage_dir = Path(settings.local_storage_dir).resolve()
    frames = _make_camera_poses(8)

    # V1: Deep Navy Blue Velvet Sectional [28, 45, 82]
    _seed_version(storage_dir, project_id, "v1", None, [28, 45, 82], frames)
    # V2: Sumptuous Wine / Crimson Red Velvet Sectional [159, 18, 57]
    _seed_version(storage_dir, project_id, "v2", "v1", [159, 18, 57], frames)

    with session_factory() as session:
        proj = session.get(Project, project_id) or Project(id=project_id)
        proj.name = "Modern Living Room (Demo)"
        proj.status = "completed"
        proj.active_version_id = "v2"

        v1 = session.get(SceneVersion, "v1") or SceneVersion(id="v1", project_id=project_id)
        v1.label = "V1 Base (Navy Sofa)"
        v1.manifest_relpath = f"{project_id}/v1/manifest.json"
        v1.spz_relpath = f"{project_id}/v1/scene.spz"

        v2 = session.get(SceneVersion, "v2") or SceneVersion(id="v2", project_id=project_id)
        v2.label = "V2 Red Velvet Sofa"
        v2.parent_version_id = "v1"
        v2.manifest_relpath = f"{project_id}/v2/manifest.json"
        v2.spz_relpath = f"{project_id}/v2/scene.spz"

        edit = session.get(EditSession, "edit-demo") or EditSession(id="edit-demo", project_id=project_id)
        edit.base_version_id = "v1"
        edit.target_version_id = "v2"
        edit.ref_frame_idx = 0
        edit.prompt = "red velvet sofa with soft tufted cushions"
        edit.status = "completed"
        masks_summary = {
            str(idx): {
                "status": "auto",
                "skip_frame": False,
                "mask_key": f"{project_id}/v1/masks/mask_{idx:06d}.png",
            }
            for idx in range(8)
        }
        edit.masks_summary_json = json.dumps(masks_summary)

        task = session.get(Task, "task-demo") or Task(id="task-demo", project_id=project_id)
        task.task_type = "rebuild_week2"
        task.status = "completed"
        task.progress_pct = 100.0
        task.current_stage = "07_rebuild"
        task.message = "Your photorealistic room is ready to explore."

        session.add_all([proj, v1, v2, edit, task])
        session.commit()

    print(f"Successfully seeded photorealistic demo room '{project_id}' into {settings.database_url}")
    return project_id


if __name__ == "__main__":
    seed_demo_room()
