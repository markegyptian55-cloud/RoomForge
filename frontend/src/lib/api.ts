export type TaskStatus = "queued" | "claimed" | "running" | "completed" | "failed" | "cancelled";
export type StageId = "01_extract" | "02_poses" | "03_train" | "04_export" | "05_track" | "06_diffusion" | "07_rebuild";
export type RebuildMode = "masked_finetune" | "from_scratch";
export type DiffusionModelVariant = "flux2_klein_4b" | "sdxl_inpaint_t4";

export interface ResolveClickResponse { edit_id: string; ref_frame_idx: number; u: number; v: number; score: number; method: string; task_id: string; }
export interface EditFrameItem { frame_idx: number; image_relpath: string | null; mask_key: string | null; mask_status: string; skip_frame: boolean; }
export interface EditFramesResponse { edit_id: string; page: number; page_size: number; total: number; frames: EditFrameItem[]; }
export interface MaskUpdateResponse { edit_id: string; frame_idx: number; status: string; skip_frame: boolean; mask_key?: string | null; }
export interface EditSummaryItem { id: string; project_id: string; base_version_id: string | null; target_version_id: string | null; ref_frame_idx: number | null; prompt: string | null; chosen_seed_id: string | null; status: string; }

export interface Project { id: string; name: string; status: string; active_version_id: string | null; created_at: string; }
export interface TaskSummary { id: string; type: string; status: TaskStatus; progress_pct: number; stage?: StageId; message?: string | null; telemetry?: Telemetry | null; }
export interface Telemetry { vram_used_mb?: number; vram_total_mb?: number; runtime_seconds?: number; [key: string]: number | string | undefined; }
export interface SceneVersion { id: string; label: string; parent_version_id: string | null; created_at: string; }
export interface ProjectDetail extends Project { versions: Array<{ id: string; label: string }>; tasks: TaskSummary[]; edit_sessions: EditSession[]; }
export interface EditSession { id: string; status?: string; }
export interface Manifest { [key: string]: unknown; }
export interface ViewerClickEvent { screenX: number; screenY: number; worldHitColmap: [number, number, number] | null; rayOriginColmap: [number, number, number]; rayDirColmap: [number, number, number]; nearestFrameIdx: number | null; }
export interface ViewerCameraPose { positionColmap: [number, number, number]; targetColmap: [number, number, number]; upColmap: [number, number, number]; fovDeg?: number; }
export interface ViewerController { onUserClick: (handler: (ev: ViewerClickEvent) => void) => () => void; setCameraPose: (pose: ViewerCameraPose) => void; loadCameraPath: (poses: ViewerCameraPose[], durationMs?: number) => void; setActiveVersion: (versionId: string) => void; modifyObject?: (targetName: string, prompt: string, model: string) => Promise<boolean>; }

export class ApiError extends Error { constructor(public readonly status: number, message: string) { super(message); this.name = "ApiError"; } }

const jsonHeaders = { "Content-Type": "application/json" };
const asErrorMessage = (body: unknown): string => {
  if (typeof body === "object" && body !== null && "detail" in body && typeof body.detail === "string") return body.detail;
  return "The request could not be completed.";
};

async function request<T>(path: string, token: string | null, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (!(init?.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(path, { ...init, headers });
  const body: unknown = await response.json().catch(() => undefined);
  if (!response.ok) throw new ApiError(response.status, asErrorMessage(body));
  return body as T;
}

export function validateMedia(file: File, maxBytes = 500 * 1024 * 1024): string | null {
  const allowedVideos = new Set(["video/mp4", "video/quicktime", "video/webm"]);
  const allowedImages = new Set(["image/jpeg", "image/png", "image/webp"]);
  
  const isVideo = allowedVideos.has(file.type);
  const isImage = allowedImages.has(file.type) || /\.(jpe?g|png|webp)$/i.test(file.name);
  
  if (!isVideo && !isImage) {
    return "Please choose an MP4/WebM/MOV video or a JPEG/PNG/WebP photo.";
  }
  if (isImage && file.size > 50 * 1024 * 1024) {
    return "Image must be 50 MB or smaller.";
  }
  if (file.size > maxBytes) {
    return "File must be 500 MB or smaller.";
  }
  return null;
}

export function validateVideo(file: File, maxBytes = 500 * 1024 * 1024): string | null {
  return validateMedia(file, maxBytes);
}

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export class RoomApi {
  private token: string | null = null;
  constructor(private readonly baseUrl = DEFAULT_API_BASE) {}
  setToken(token: string | null): void { this.token = token; }
  private path(pathname: string): string { return `${this.baseUrl}${pathname}`; }
  register(email: string, password: string): Promise<{ access_token: string }> { return request(this.path("/api/auth/register"), this.token, { method: "POST", body: JSON.stringify({ email, password }) }); }
  login(email: string, password: string): Promise<{ access_token: string }> { return request(this.path("/api/auth/login"), this.token, { method: "POST", body: JSON.stringify({ email, password }) }); }
  listProjects(): Promise<Project[]> { return request(this.path("/api/projects"), this.token); }
  createProject(name: string): Promise<Project> { return request(this.path("/api/projects"), this.token, { method: "POST", body: JSON.stringify({ name }) }); }
  getProject(id: string): Promise<ProjectDetail> { return request(this.path(`/api/projects/${encodeURIComponent(id)}`), this.token); }
  uploadVideo(projectId: string, file: File, onProgress?: (percent: number) => void): Promise<{ project_id: string; task_id: string; bytes: number }> {
    const validationError = validateVideo(file);
    if (validationError) return Promise.reject(new TypeError(validationError));
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", this.path(`/api/projects/${encodeURIComponent(projectId)}/upload`));
      if (this.token) xhr.setRequestHeader("Authorization", `Bearer ${this.token}`);
      xhr.upload.onprogress = (event) => { if (event.lengthComputable) onProgress?.(Math.round((event.loaded / event.total) * 100)); };
      xhr.onerror = () => reject(new ApiError(0, "Upload failed. Check your connection and try again."));
      xhr.onload = () => { let body: unknown; try { body = JSON.parse(xhr.responseText) as unknown; } catch { body = undefined; } if (xhr.status >= 200 && xhr.status < 300) resolve(body as { project_id: string; task_id: string; bytes: number }); else reject(new ApiError(xhr.status, asErrorMessage(body))); };
      const form = new FormData(); form.append("file", file); xhr.send(form);
    });
  }
  async pollProject(id: string, onUpdate: (project: ProjectDetail) => void, intervalMs = 2000, signal?: AbortSignal): Promise<ProjectDetail> {
    while (!signal?.aborted) { const project = await this.getProject(id); onUpdate(project); if (project.tasks.some((task) => task.status === "completed" || task.status === "failed" || task.status === "cancelled")) return project; await new Promise<void>((resolve) => { const timer = window.setTimeout(resolve, intervalMs); signal?.addEventListener("abort", () => { window.clearTimeout(timer); resolve(); }, { once: true }); }); }
    throw new DOMException("Polling stopped", "AbortError");
  }
  versions(projectId: string): Promise<SceneVersion[]> { return request(this.path(`/api/projects/${encodeURIComponent(projectId)}/versions`), this.token); }
  manifest(projectId: string, versionId: string): Promise<Manifest> { return request(this.path(`/api/projects/${encodeURIComponent(projectId)}/versions/${encodeURIComponent(versionId)}/manifest`), this.token); }
  resolveClick(projectId: string, body: ViewerClickEvent): Promise<ResolveClickResponse> { return request(this.path(`/api/projects/${encodeURIComponent(projectId)}/edits/resolve-click`), this.token, { method: "POST", body: JSON.stringify(body) }); }
  refine(editId: string, points: Array<{ x: number; y: number; label: 1 | 0 }>): Promise<TaskSummary> { return request(this.path(`/api/edits/${encodeURIComponent(editId)}/refine`), this.token, { method: "POST", body: JSON.stringify({ points }) }); }
  track(editId: string): Promise<TaskSummary> { return request(this.path(`/api/edits/${encodeURIComponent(editId)}/track`), this.token, { method: "POST" }); }
  editFrames(editId: string, page = 1, pageSize = 24): Promise<EditFramesResponse> { return request(this.path(`/api/edits/${encodeURIComponent(editId)}/frames?page=${encodeURIComponent(page)}&page_size=${encodeURIComponent(pageSize)}`), this.token); }
  updateMask(editId: string, frameIdx: number, body: FormData | { skip_frame: boolean }): Promise<MaskUpdateResponse> { return request(this.path(`/api/edits/${encodeURIComponent(editId)}/masks/${encodeURIComponent(frameIdx)}`), this.token, { method: "PUT", headers: body instanceof FormData ? {} : jsonHeaders, body: body instanceof FormData ? body : JSON.stringify(body) }); }
  previewSeeds(editId: string, prompt: string, seeds: number[], model_variant: DiffusionModelVariant): Promise<TaskSummary> { return request(this.path(`/api/edits/${encodeURIComponent(editId)}/preview-seeds`), this.token, { method: "POST", body: JSON.stringify({ prompt, seeds, model_variant }) }); }
  applyEdit(editId: string, chosen_seed_id: string, rebuild_mode: RebuildMode, prompt?: string): Promise<TaskSummary & { edit_id: string; target_version_id: string; diffusion_task_id: string }> { return request(this.path(`/api/edits/${encodeURIComponent(editId)}/apply`), this.token, { method: "POST", body: JSON.stringify({ chosen_seed_id, rebuild_mode, prompt }) }); }
  edits(projectId: string): Promise<EditSummaryItem[]> { return request(this.path(`/api/projects/${encodeURIComponent(projectId)}/edits`), this.token); }
  quickEdit(projectId: string, targetObject: string, prompt: string, model: DiffusionModelVariant = "flux2_klein_4b", apiKey?: string): Promise<{ success: boolean; new_version_id: string; label: string; message: string }> {
    return request(this.path(`/api/projects/${encodeURIComponent(projectId)}/quick-edit`), this.token, {
      method: "POST",
      body: JSON.stringify({ target_object: targetObject, prompt, model, api_key: apiKey }),
    });
  }
}

export const roomApi = new RoomApi();
