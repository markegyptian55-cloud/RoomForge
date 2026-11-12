export type TaskStatus = "queued" | "claimed" | "running" | "completed" | "failed" | "cancelled";
export type StageId = "01_extract" | "02_poses" | "03_train" | "04_export" | "05_track" | "06_diffusion" | "07_rebuild";
export type RebuildMode = "masked_finetune" | "from_scratch";

export interface Project { id: string; name: string; status: string; active_version_id: string | null; created_at: string; }
export interface TaskSummary { id: string; type: string; status: TaskStatus; progress_pct: number; stage?: StageId; message?: string | null; telemetry?: Telemetry | null; }
export interface Telemetry { vram_used_mb?: number; vram_total_mb?: number; runtime_seconds?: number; [key: string]: number | string | undefined; }
export interface SceneVersion { id: string; label: string; parent_version_id: string | null; created_at: string; }
export interface ProjectDetail extends Project { versions: Array<{ id: string; label: string }>; tasks: TaskSummary[]; edit_sessions: EditSession[]; }
export interface EditSession { id: string; status?: string; }
export interface Manifest { [key: string]: unknown; }
export interface ViewerClickEvent { screenX: number; screenY: number; worldHitColmap: [number, number, number] | null; rayOriginColmap: [number, number, number]; rayDirColmap: [number, number, number]; nearestFrameIdx: number | null; }
export interface ViewerCameraPose { positionColmap: [number, number, number]; targetColmap: [number, number, number]; upColmap: [number, number, number]; fovDeg?: number; }

export class ApiError extends Error { constructor(public readonly status: number, message: string) { super(message); this.name = "ApiError"; } }

const jsonHeaders = { "Content-Type": "application/json" };
const asErrorMessage = (body: unknown): string => {
  if (typeof body === "object" && body !== null && "detail" in body && typeof body.detail === "string") return body.detail;
  return "The request could not be completed.";
};

async function request<T>(path: string, token: string | null, init?: RequestInit): Promise<T> {
  const auth = token ? { Authorization: `Bearer ${token}` } : {};
  const headers = init?.body instanceof FormData
    ? { ...auth, ...init.headers }
    : { ...jsonHeaders, ...auth, ...init?.headers };
  const response = await fetch(path, { ...init, headers });
  const body: unknown = await response.json().catch(() => undefined);
  if (!response.ok) throw new ApiError(response.status, asErrorMessage(body));
  return body as T;
}

export function validateVideo(file: File, maxBytes = 500 * 1024 * 1024): string | null {
  const allowed = new Set(["video/mp4", "video/quicktime", "video/webm"]);
  if (!allowed.has(file.type)) return "Choose an MP4, QuickTime, or WebM video.";
  if (file.size > maxBytes) return "Video must be 500 MB or smaller.";
  return null;
}

export class RoomApi {
  private token: string | null = null;
  constructor(private readonly baseUrl = "") {}
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
  versions(projectId: string): Promise<SceneVersion[]> { return request(this.path(`/api/projects/${projectId}/versions`), this.token); }
  manifest(projectId: string, versionId: string): Promise<Manifest> { return request(this.path(`/api/projects/${projectId}/versions/${versionId}/manifest`), this.token); }
  resolveClick(projectId: string, body: ViewerClickEvent): Promise<{ edit_id: string; ref_frame_idx: number; u: number; v: number }> { return request(this.path(`/api/projects/${projectId}/edits/resolve-click`), this.token, { method: "POST", body: JSON.stringify(body) }); }
  refine(editId: string, points: Array<{ x: number; y: number; label: 1 | 0 }>): Promise<TaskSummary> { return request(this.path(`/api/edits/${editId}/refine`), this.token, { method: "POST", body: JSON.stringify({ points }) }); }
  track(editId: string): Promise<TaskSummary> { return request(this.path(`/api/edits/${editId}/track`), this.token, { method: "POST" }); }
  editFrames(editId: string, page = 1): Promise<unknown> { return request(this.path(`/api/edits/${editId}/frames?page=${page}`), this.token); }
  updateMask(editId: string, frameIdx: number, body: FormData | { skip_frame: boolean }): Promise<unknown> { return request(this.path(`/api/edits/${editId}/masks/${frameIdx}`), this.token, { method: "PUT", headers: body instanceof FormData ? {} : jsonHeaders, body: body instanceof FormData ? body : JSON.stringify(body) }); }
  previewSeeds(editId: string, prompt: string, seeds: number[], model_variant: string): Promise<TaskSummary> { return request(this.path(`/api/edits/${editId}/preview-seeds`), this.token, { method: "POST", body: JSON.stringify({ prompt, seeds, model_variant }) }); }
  applyEdit(editId: string, chosen_seed_id: string, rebuild_mode: RebuildMode): Promise<TaskSummary> { return request(this.path(`/api/edits/${editId}/apply`), this.token, { method: "POST", body: JSON.stringify({ chosen_seed_id, rebuild_mode }) }); }
}

export const roomApi = new RoomApi();
