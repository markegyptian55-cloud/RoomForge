from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for subdir in ("backend", "worker"):
    path_str = str(REPO_ROOT / subdir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from app.config import get_settings  # noqa: E402
from app.db import Base, create_engine, sessionmaker  # noqa: E402
from app.models import EditSession, Project, SceneVersion, Task  # noqa: E402


def _make_box(extents: tuple[float, float, float], center: tuple[float, float, float], color: list[int], name: str) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    mesh.visual.vertex_colors = np.full((len(mesh.vertices), 4), color, dtype=np.uint8)
    mesh.metadata["name"] = name
    return mesh


def _make_cylinder(radius: float, height: float, center: tuple[float, float, float], color: list[int], name: str) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=36)
    # Rotate -90 degrees around X to map canonical Z-height to room Y-up (vertical)
    r_x = trimesh.transformations.rotation_matrix(-math.pi / 2, [1.0, 0.0, 0.0])
    mesh.apply_transform(r_x)
    mesh.apply_translation(center)
    mesh.visual.vertex_colors = np.full((len(mesh.vertices), 4), color, dtype=np.uint8)
    mesh.metadata["name"] = name
    return mesh


def build_room_scene(sofa_color: list[int]) -> trimesh.Scene:
    scene = trimesh.Scene()

    # 1. Floor: Oak Parquet (warm golden oak)
    floor = _make_box((6.0, 0.04, 6.0), (0.0, -0.02, 0.0), [185, 142, 98, 255], "Oak Parquet Floor")
    scene.add_geometry(floor, node_name="floor")

    # 2. Back Wall (Soft stone plaster behind sofa)
    back_wall = _make_box((6.0, 2.65, 0.08), (0.0, 1.325, 3.0), [232, 229, 224, 255], "Back Wall")
    scene.add_geometry(back_wall, node_name="back_wall")

    # Baseboard
    baseboard = _make_box((6.0, 0.14, 0.1), (0.0, 0.07, 2.99), [42, 48, 56, 255], "Baseboard")
    scene.add_geometry(baseboard, node_name="baseboard")

    # 3. Right Wall (Sage green accent wall)
    right_wall = _make_box((0.08, 2.65, 6.0), (3.0, 1.325, 0.0), [175, 192, 178, 255], "Sage Accent Wall")
    scene.add_geometry(right_wall, node_name="right_wall")

    # 4. Left Wall with Panoramic Glass Window
    left_top = _make_box((0.08, 0.35, 6.0), (-3.0, 2.475, 0.0), [232, 229, 224, 255], "Window Header")
    left_bottom = _make_box((0.08, 0.45, 6.0), (-3.0, 0.225, 0.0), [232, 229, 224, 255], "Window Sill")
    window_glass = _make_box((0.02, 1.75, 5.8), (-3.02, 1.325, 0.0), [145, 195, 235, 180], "Terrace Window Glass")
    scene.add_geometry(left_top, node_name="left_top")
    scene.add_geometry(left_bottom, node_name="left_bottom")
    scene.add_geometry(window_glass, node_name="window_glass")

    # Window mullions (Black aluminum frames)
    for mz in [-1.8, 0.0, 1.8]:
        mullion = _make_box((0.06, 1.75, 0.06), (-3.0, 1.325, mz), [30, 32, 36, 255], f"Mullion_{mz}")
        scene.add_geometry(mullion, node_name=f"mullion_{mz}")

    # 5. Modern Persian Area Rug
    rug = _make_box((3.2, 0.015, 2.4), (-0.5, 0.008, 1.25), [218, 182, 155, 255], "Area Rug")
    scene.add_geometry(rug, node_name="rug")

    # 8. Luxury Sectional Sofa (Target Editable Object)
    sofa_c = sofa_color + [255]
    # Base
    sofa_base = _make_box((2.4, 0.38, 1.15), (-0.55, 0.19, 1.75), sofa_c, "Luxury Sofa")
    # Backrest
    sofa_back = _make_box((2.4, 0.48, 0.28), (-0.55, 0.62, 2.2), sofa_c, "Luxury Sofa")
    # Left Armrest
    sofa_arm = _make_box((0.26, 0.58, 1.15), (-1.88, 0.29, 1.75), sofa_c, "Luxury Sofa")
    # Chaise Lounge
    sofa_chaise = _make_box((0.65, 0.38, 0.75), (0.42, 0.19, 0.8), sofa_c, "Luxury Sofa")
    # Seat Cushions
    cushion_1 = _make_box((0.74, 0.1, 0.82), (-1.28, 0.43, 1.65), sofa_c, "Luxury Sofa")
    cushion_2 = _make_box((0.74, 0.1, 0.82), (-0.52, 0.43, 1.65), sofa_c, "Luxury Sofa")
    # Throw Pillows (Ochre Gold Velvet)
    pillow_1 = _make_box((0.38, 0.38, 0.14), (-1.45, 0.58, 1.98), [215, 145, 45, 255], "Throw Pillow Gold")
    pillow_2 = _make_box((0.38, 0.38, 0.14), (0.22, 0.58, 1.98), [242, 235, 220, 255], "Throw Pillow Ivory")

    for idx, g in enumerate([sofa_base, sofa_back, sofa_arm, sofa_chaise, cushion_1, cushion_2, pillow_1, pillow_2]):
        scene.add_geometry(g, node_name=f"sofa_part_{idx}")

    # 9. Live-Edge Walnut Coffee Table
    table_top = _make_box((1.15, 0.05, 0.65), (-0.5, 0.35, 0.65), [112, 74, 42, 255], "Walnut Coffee Table")
    scene.add_geometry(table_top, node_name="coffee_table")
    # 4 Steel Hairpin Legs
    for lx, lz in [(-0.95, 0.42), (-0.05, 0.42), (-0.95, 0.88), (-0.05, 0.88)]:
        leg = _make_cylinder(0.015, 0.325, (lx, 0.162, lz), [32, 34, 38, 255], "Table Leg")
        scene.add_geometry(leg, node_name=f"table_leg_{lx}_{lz}")

    # 10. Ceramic Coffee Mug (Editable Prop on Table)
    mug = _make_cylinder(0.045, 0.09, (-0.22, 0.42, 0.62), [218, 105, 65, 255], "Ceramic Coffee Mug")
    scene.add_geometry(mug, node_name="coffee_mug")

    # 11. Art & Architecture Book on Table
    book = _make_box((0.32, 0.035, 0.24), (-0.72, 0.395, 0.65), [245, 242, 238, 255], "Art Book")
    scene.add_geometry(book, node_name="art_book")

    # 12. Modern Canvas Wall Painting above Sofa
    frame = _make_box((2.0, 1.0, 0.04), (-0.55, 1.7, 2.96), [25, 28, 32, 255], "Painting Frame")
    canvas = _make_box((1.88, 0.88, 0.02), (-0.55, 1.7, 2.94), [220, 115, 65, 255], "Modern Abstract Canvas")
    scene.add_geometry(frame, node_name="art_frame")
    scene.add_geometry(canvas, node_name="art_canvas")

    # 13. Corner Potted Plant (Luxury Architectural Ficus)
    pot = _make_cylinder(0.26, 0.48, (-2.4, 0.24, 2.45), [195, 95, 62, 255], "Terracotta Planter")
    soil = _make_cylinder(0.25, 0.04, (-2.4, 0.46, 2.45), [55, 38, 26, 255], "Plant Soil")
    trunk = _make_cylinder(0.038, 1.35, (-2.4, 0.95, 2.45), [110, 75, 50, 255], "Plant Trunk")
    scene.add_geometry(pot, node_name="plant_pot")
    scene.add_geometry(soil, node_name="plant_soil")
    scene.add_geometry(trunk, node_name="plant_trunk")

    # Organic sculpted botanical foliage using golden-ratio phyllotaxis clusters
    golden_angle = 2.39996
    num_clusters = 14
    for idx in range(num_clusters):
        t = idx / float(num_clusters)
        ang = idx * golden_angle
        r = 0.16 + 0.28 * math.sin(t * math.pi)
        lx = -2.4 + r * math.cos(ang)
        lz = 2.45 + r * math.sin(ang)
        ly = 0.70 + 1.15 * t

        cluster_rad = 0.19 + 0.08 * (1.0 - t * 0.4)
        c_mesh = trimesh.creation.icosphere(subdivisions=2, radius=cluster_rad)
        # Scale to give natural flattened leaf canopy profile
        s_matrix = np.diag([1.35, 0.42, 1.15, 1.0])
        c_mesh.apply_transform(s_matrix)
        # Apply tilt
        rot = trimesh.transformations.rotation_matrix(0.35 * math.sin(ang), [math.cos(ang), 0.0, math.sin(ang)])
        c_mesh.apply_transform(rot)
        c_mesh.apply_translation((lx, ly, lz))

        # Varied lush botanical green shades
        shade = int(24 + 18 * t)
        c_mesh.visual.vertex_colors = np.full((len(c_mesh.vertices), 4), [shade, 125 + shade // 2, 55, 255], dtype=np.uint8)
        c_mesh.metadata["name"] = f"Plant Leaves {idx}"
        scene.add_geometry(c_mesh, node_name=f"plant_leaves_{idx}")

    # 14. Designer Brass Arc Lamp
    lamp_base = _make_cylinder(0.24, 0.035, (1.8, 0.0175, 2.0), [215, 175, 95, 255], "Lamp Base")
    lamp_stem = _make_cylinder(0.018, 1.85, (1.8, 0.95, 2.0), [215, 175, 95, 255], "Lamp Stem")
    lamp_arm = _make_box((0.8, 0.02, 0.6), (1.4, 1.9, 1.7), [215, 175, 95, 255], "Lamp Arm")
    lamp_shade = _make_cylinder(0.18, 0.22, (1.0, 1.85, 1.4), [255, 250, 235, 255], "Lamp Shade")
    scene.add_geometry(lamp_base, node_name="lamp_base")
    scene.add_geometry(lamp_stem, node_name="lamp_stem")
    scene.add_geometry(lamp_arm, node_name="lamp_arm")
    scene.add_geometry(lamp_shade, node_name="lamp_shade")

    return scene


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


def _seed_realistic_version(
    storage_dir: Path,
    project_id: str,
    version_id: str,
    parent_id: str | None,
    sofa_color: list[int],
    frames: list[dict[str, Any]],
) -> Path:
    version_dir = storage_dir / project_id / version_id
    version_dir.mkdir(parents=True, exist_ok=True)
    images_dir = version_dir / "images"
    masks_dir = version_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export 3D GLB
    scene = build_room_scene(sofa_color)
    glb_data = scene.export(file_type="glb")
    glb_path = version_dir / "scene.glb"
    glb_path.write_bytes(glb_data)

    # 2. Render realistic mock camera frames
    for frame in frames:
        idx = int(frame["frame_idx"])
        img_buffer = np.full((480, 640, 3), [22, 28, 38], dtype=np.uint8)
        # Gradient floor & walls
        img_buffer[260:, :] = [185, 142, 98]
        img_buffer[:260, :] = [232, 229, 224]
        # Sofa region
        img_buffer[200:360, 180:480] = sofa_color
        # Coffee table
        img_buffer[320:390, 240:420] = [112, 74, 42]
        # Mug
        img_buffer[305:335, 340:370] = [218, 105, 65]

        mask_buffer = np.zeros((480, 640), dtype=np.uint8)
        mask_buffer[200:360, 180:480] = 255  # Sofa mask

        Image.fromarray(img_buffer).save(images_dir / f"frame_{idx:06d}.jpg", quality=94)
        Image.fromarray(mask_buffer).save(masks_dir / f"mask_{idx:06d}.png")

    # 3. Create manifest
    manifest = {
        "schema_version": "1.0",
        "project_id": project_id,
        "version_id": version_id,
        "parent_version_id": parent_id,
        "coord_convention": "colmap_opencv",
        "T_viewer_from_colmap": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "camera_model": "PINHOLE",
        "width": 640,
        "height": 480,
        "intrinsics": {"cx": 320.0, "cy": 240.0, "fx": 520.0, "fy": 520.0},
        "test_every": 8,
        "pose_backend": "colmap",
        "frames": frames,
        "artifacts": {
            "scene_glb_relpath": "scene.glb",
            "mesh_glb_relpath": "scene.glb",
            "ply_relpath": None,
            "spz_relpath": None,
            "sparse_points_relpath": None,
            "ckpt_relpath": None,
            "depth_dir": None,
            "collision_glb_relpath": None,
        },
        "metrics": {"psnr": 34.2, "ssim": 0.952, "lpips": 0.048},
    }
    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return version_dir


def seed_realistic_project() -> None:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    project_id = "realistic-room"
    session.query(EditSession).filter_by(project_id=project_id).delete()
    session.query(Task).filter_by(project_id=project_id).delete()
    session.query(SceneVersion).filter_by(project_id=project_id).delete()
    session.query(Project).filter_by(id=project_id).delete()
    session.commit()

    storage_dir = Path(settings.local_storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    frames = _make_camera_poses(8)

    # V1: Base Navy Velvet Sofa
    _seed_realistic_version(
        storage_dir=storage_dir,
        project_id=project_id,
        version_id="realistic-v1",
        parent_id=None,
        sofa_color=[32, 58, 95],
        frames=frames,
    )

    # V2: Edited Red Velvet Sofa
    _seed_realistic_version(
        storage_dir=storage_dir,
        project_id=project_id,
        version_id="realistic-v2",
        parent_id="realistic-v1",
        sofa_color=[165, 38, 48],
        frames=frames,
    )

    project = Project(
        id=project_id,
        name="Modern Luxury Suite (Photorealistic 3D)",
        status="completed",
        active_version_id="realistic-v1",
    )
    v1 = SceneVersion(
        id="realistic-v1",
        project_id=project_id,
        label="V1 Base (Royal Navy Sofa)",
        manifest_relpath=f"{project_id}/realistic-v1/manifest.json",
    )
    v2 = SceneVersion(
        id="realistic-v2",
        project_id=project_id,
        parent_version_id="realistic-v1",
        label="V2 Edited (Red Velvet Sofa)",
        manifest_relpath=f"{project_id}/realistic-v2/manifest.json",
    )
    task = Task(
        project_id=project_id,
        task_type="reconstruct",
        status="completed",
        progress_pct=100.0,
        current_stage="complete",
        message="Photorealistic 3D room ready to explore.",
        result_json=json.dumps({"fps": 60, "polygons": 2480, "vram_mb": 512}),
    )

    session.add_all([project, v1, v2, task])
    session.commit()
    session.close()
    print("✅ Successfully seeded 'realistic-room' into", settings.database_url)


if __name__ == "__main__":
    seed_realistic_project()
