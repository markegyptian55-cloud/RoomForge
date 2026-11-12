# API, Pull-Worker, and Viewer Integration Contracts

## 1. Authentication & Projects
- `POST /api/auth/register`, `POST /api/auth/login` (JWT bearer + optional guest dev mode via `AUTH_REQUIRED=false` setting).
- `GET /api/projects`, `POST /api/projects` -> `{id, name, status, active_version_id, created_at}`.
- `GET /api/projects/{project_id}` -> project + `versions[]` + latest `tasks[]` + `edit_sessions[]`.
- `POST /api/projects/{project_id}/upload` -> chunked or multipart streaming video upload (`>= 500MB` safe), validates container/MIME (`video/mp4`, `video/quicktime`, `video/webm`), stores input video via `StorageDriver`, creates `Task(type="reconstruct_week1")`.

## 2. Pull-Based Worker Endpoints (Worker Auth Header `X-Worker-Token`)
- `POST /api/tasks/claim` `{worker_id, capabilities, interactive_only}` -> `204` or `200 {task_id, attempt_id, project_id, task_type, payload}` using atomic lock + attempt fence.
- `POST /api/tasks/{task_id}/heartbeat` `{attempt_id, worker_id}`.
- `POST /api/tasks/{task_id}/progress` `{attempt_id, stage, progress_pct, message, telemetry?}`.
- `POST /api/tasks/{task_id}/complete` `{attempt_id, result_manifest_key?, result_payload}`. Rejects stale `attempt_id` (`409 Conflict`).
- `POST /api/tasks/{task_id}/fail` `{attempt_id, error_code, error_message, retryable}`.
- `POST /api/storage/{project_id}/{version_id}/upload` & `GET /api/storage/{project_id}/{version_id}/{path:path}` (streaming upload/download for frames, `sparse/0`, `gs_ckpt.pt`, `scene.spz`, masks).

## 3. Scene & Week 2 Editing Endpoints
- `GET /api/projects/{project_id}/versions` -> list `v0_base`, `edit_*` for Before/After & undo tree.
- `GET /api/projects/{project_id}/versions/{version_id}/manifest` -> validated `manifest.json`.
- `POST /api/projects/{project_id}/edits/resolve-click` -> resolves `{worldHitColmap?, rayOriginColmap?, rayDirColmap?, nearestFrameIdx?}` into best `ref_frame_idx` + `(u, v)` + creates `EditSession` + queues `segment_init`.
- `POST /api/edits/{edit_id}/refine` `{points: [{x, y, label: 1|0}]}` -> queues warm `segment_refine`.
- `POST /api/edits/{edit_id}/track` -> queues `track_propagate` across all frames.
- `GET /api/edits/{edit_id}/frames` -> paginated tracking thumbnails + mask summary for QC grid.
- `PUT /api/edits/{edit_id}/masks/{frame_idx}` -> upload manual 1-channel uint8 mask PNG or mark `{skip_frame: true|false}`.
- `POST /api/edits/{edit_id}/preview-seeds` `{prompt, seeds: number[], model_variant}` -> queues `diffusion_preview`.
- `POST /api/edits/{edit_id}/apply` `{chosen_seed_id, rebuild_mode: "masked_finetune"|"from_scratch"}` -> queues `diffusion_apply` + `rebuild_week2`.

## 4. Frontend Viewer3D Hook Contract (`Week 1 -> Week 2`)
```ts
export interface ViewerClickEvent {
  screenX: number;
  screenY: number;
  worldHitColmap: [number, number, number] | null;
  rayOriginColmap: [number, number, number];
  rayDirColmap: [number, number, number];
  nearestFrameIdx: number | null;
}

export interface ViewerCameraPose {
  positionColmap: [number, number, number];
  targetColmap: [number, number, number];
  upColmap: [number, number, number];
  fovDeg?: number;
}

export interface ViewerController {
  onUserClick: (handler: (ev: ViewerClickEvent) => void) => () => void;
  setCameraPose: (pose: ViewerCameraPose) => void;
  loadCameraPath: (poses: ViewerCameraPose[], durationMs?: number) => void;
  setActiveVersion: (versionId: string) => void;
}
```
