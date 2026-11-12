# Week 1 -> Week 2 Engineering Handoff Specification

## 1. Canonical Artifact Contract (`VersionDirectory`)

Week 1 (`reconstruct_week1`) publishes an immutable base scene version (`{project_id}_v0` or UUID) to `StorageDriver` under `{project_id}/{version_id}/`. Week 2 (`segment_init`, `track_propagate`, `diffusion_preview`, `diffusion_apply`, `rebuild_week2`) consumes these exact artifacts without mutating the base version:

```text
{project_id}/{version_id}/
  manifest.json                  # Validated against contracts/manifest.schema.json
  scene.spz                      # Canonical Spark WebGL2 viewer splat bundle
  scene.ply                      # Optional high-precision export
  sparse_points.json             # WebGL fallback sparse point cloud
  gs_state.npz / gs_ckpt.pt      # Gaussian parameters for localized Week 2 fine-tuning
  images/frame_000000.jpg ...    # Undistorted, metadata-stripped mod-16 RGB frames
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

---

## 2. Geometry & Manifest Invariants

| Field / Invariant | Requirement | Enforcement Point |
| :--- | :--- | :--- |
| `schema_version` | Exact `"1.0.0"` | `contracts/manifest.schema.json`, `common/manifest.py` |
| `coord_convention` | Exact `"opencv_colmap"` (`sparse/0` stays unflipped) | `common/manifest.py`, `pipeline/s02_poses.py` |
| `T_viewer_from_colmap` | Finite `4x4` row-major rigid matrix mapping COLMAP -> Three.js Y-up | `common/coords.py:align_gravity_up`, `frontend/src/lib/cameraMath.ts` |
| `camera_model` | Exact `"PINHOLE"` (undistorted) | `pipeline/s02_poses.py`, `manifest.schema.json` |
| `width`, `height` | Integers divisible by `16` (`>= 64`) | `pipeline/s01_extract.py`, `common/coords.py` |
| `test_every` | `0` in production (`build_train_config`) | `pipeline/s03_train.py`, `manifest.schema.json` |
| `parent_version_id` | `null` for Week 1 base, `base_version_id` for Week 2 rebuilt versions | `contracts/manifest.schema.json`, `pipeline/s07_rebuild.py` |

---

## 3. Control Plane & Worker Execution Handoff

### Unified Click Resolution (`S2.1`)
- Frontend `Viewer3D` converts a Three.js canvas click through `invertRigid4x4(T_viewer_from_colmap)` into COLMAP space (`worldHitColmap`, `rayOriginColmap`, `rayDirColmap`).
- `POST /api/projects/{id}/edits/resolve-click` invokes `worker/room3d_worker/common/click_resolve.py:resolve_viewer_click`.
- Projects `worldHitColmap` across active frames (`depth > 1e-4`, projection score `alignment * 3.0 - center_penalty * 2.0 - 0.1 * depth` where `center_penalty = math.hypot(u - cx, v - cy) / half_diag`) and falls back to ray snapping (`nearest_snap`, score `nearest_bonus + dir_align * 1.5 + origin_align - 0.1 * pos_dist`) when no frame captures the 3D hit.

### Dependency Barriers & GPU Partitioning (`S2.1`, `S2.4`)
- **Interactive (`segment_init`, `segment_refine`)**: Warm `cuda:1` worker loop (`interactive_only=True`, `0.5s` poll interval in `WorkerClient.run_interactive_loop`).
- **Diffusion (`diffusion_preview`, `diffusion_apply`)**: Isolated `cuda:0` batch subprocess (`WorkerClient.BATCH_GPU`).
- **3DGS & Tracking (`reconstruct_week1`, `track_propagate`, `rebuild_week2`)**: Isolated `cuda:1` batch subprocess (`WorkerClient.BATCH_GPU`).
- **DAG Barrier**: `POST /api/edits/{id}/apply` creates `diffusion_apply` and `rebuild_week2` with `depends_on_task_id = diffusion_task.id` recorded in `payload_json`. `backend/app/services/queue.py` refuses claiming `rebuild_week2` until `diffusion_apply` is `completed` and cascades `failed` status to waiting tasks and `EditSession` if upstream diffusion fails.
