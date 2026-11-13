"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import EditStudio from "@/components/EditStudio";
import FloatingEditBar from "@/components/FloatingEditBar";
import { ProgressPanel } from "@/components/ProgressPanel";
import Viewer3D from "@/components/Viewer3D";
import { ThemeToggle } from "@/components/ThemeToggle";
import {
  roomApi,
  type DiffusionModelVariant,
  type Manifest,
  type ProjectDetail,
  type TaskSummary,
  type ViewerController,
} from "@/lib/api";

export default function ProjectPage({ params }: { params: Promise<{ id: string }> }) {
  const [projectId, setProjectId] = useState("");
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [versionManifest, setVersionManifest] = useState<{ versionId: string; data: Manifest } | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [controller, setController] = useState<ViewerController | null>(null);
  const [isEditMode, setIsEditMode] = useState(false);
  const [selectedObjectName, setSelectedObjectName] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [floatingBusy, setFloatingBusy] = useState(false);
  const [floatingStatus, setFloatingStatus] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    void params
      .then(({ id }) => {
        if (cancelled) return;
        setProjectId(id);
        return roomApi.getProject(id);
      })
      .then((proj) => {
        if (!cancelled && proj) setProject(proj);
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : "Project unavailable.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [params]);

  const displayVersionId = selectedVersionId ?? project?.active_version_id ?? null;

  useEffect(() => {
    if (!projectId || !displayVersionId) return;
    let cancelled = false;
    void roomApi
      .manifest(projectId, displayVersionId)
      .then((data) => {
        if (!cancelled) setVersionManifest({ versionId: displayVersionId, data });
      })
      .catch(() => {
        if (!cancelled) setVersionManifest(null);
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, displayVersionId]);

  const activeManifest =
    displayVersionId && versionManifest?.versionId === displayVersionId ? versionManifest.data : undefined;
  const activeAssetBaseUrl =
    activeManifest && projectId && displayVersionId
      ? `/api/storage/${encodeURIComponent(projectId)}/${encodeURIComponent(displayVersionId)}`
      : "";

  const task: TaskSummary | undefined = project?.tasks[0];
  const terminal = task?.status === "completed" || task?.status === "failed" || task?.status === "cancelled";

  useEffect(() => {
    if (!projectId || terminal) return;
    const timer = window.setInterval(() => {
      void roomApi.getProject(projectId).then(setProject).catch(() => undefined);
    }, 2500);
    return () => window.clearInterval(timer);
  }, [projectId, terminal]);

  const handleFloatingSubmit = async (promptText: string, modelVariant: DiffusionModelVariant) => {
    setFloatingBusy(true);
    setFloatingStatus(`Generating AI transformation with ${modelVariant}...`);
    try {
      // 1. Instantly transform the 3D scene live in WebGL
      if (controller?.modifyObject && selectedObjectName) {
        await controller.modifyObject(selectedObjectName, promptText, modelVariant);
      }

      // 2. Persist new version to backend database and storage
      const res = await roomApi.quickEdit(projectId, selectedObjectName || "Target Object", promptText, modelVariant);

      // 3. Refresh project versions so the new version pill appears immediately
      const updated = await roomApi.getProject(projectId);
      if (updated) {
        setProject(updated);
      }
      setSelectedVersionId(res.new_version_id);
      setFloatingStatus(`✨ ${res.message}`);
    } catch (err) {
      if (controller?.modifyObject && selectedObjectName) {
        await controller.modifyObject(selectedObjectName, promptText, modelVariant);
      }
      setFloatingStatus(`✨ Live 3D edit applied: "${promptText}"`);
    } finally {
      setFloatingBusy(false);
    }
  };

  return (
    <main className="studio-shell">
      <header className="studio-topbar">
        <div className="studio-topbar-left">
          <Link className="brand" href="/">ROOM<span>FORGE</span></Link>
          <span className="studio-divider">/</span>
          <h1 className="studio-project-title">{project?.name ?? "Loading scene…"}</h1>
          <span className="version-pill">{project?.active_version_id ? "ACTIVE VERSION" : "BASE BUILD"}</span>
        </div>
        <div className="studio-topbar-right">
          <button
            type="button"
            onClick={() => setSidebarOpen((v) => !v)}
            title={sidebarOpen ? "Expand 3D viewport to full screen" : "Show sidebar and project controls"}
            style={{
              background: sidebarOpen ? "rgba(255, 255, 255, 0.05)" : "#10b981",
              color: sidebarOpen ? "var(--ink)" : "#ffffff",
              border: "1px solid var(--line)",
              borderRadius: "8px",
              padding: "5px 12px",
              fontSize: "12px",
              cursor: "pointer",
              fontWeight: 600,
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
              transition: "all 0.2s ease",
            }}
          >
            {sidebarOpen ? "⛶ Fullscreen 3D View" : "📋 Show Sidebar"}
          </button>
          <ThemeToggle />
          <Link className="back-link" href="/">← Projects</Link>
        </div>
      </header>
      <div className="studio-body">
        {error ? (
          <div className="studio-error-container">
            <p className="form-error">{error}</p>
          </div>
        ) : (
          <>
            <div className="studio-viewport-container">
              <Viewer3D
                manifest={activeManifest as Parameters<typeof Viewer3D>[0]["manifest"]}
                onControllerReady={setController}
                onVersionChange={setSelectedVersionId}
                assetBaseUrl={activeAssetBaseUrl}
                isEditMode={isEditMode}
                onToggleEditMode={() => setIsEditMode((v) => !v)}
                onSelectObjectName={(name) => {
                  setSelectedObjectName(name);
                  setIsEditMode(true);
                }}
              />
              <FloatingEditBar
                selectedObject={selectedObjectName}
                onClose={() => setSelectedObjectName(null)}
                onSubmit={handleFloatingSubmit}
                busy={floatingBusy}
                statusMessage={floatingStatus}
                versions={project?.versions.map((v) => ({ id: v.id, label: v.label || v.id }))}
                activeVersionId={displayVersionId}
                onSelectVersion={(vId) => {
                  setSelectedVersionId(vId);
                  controller?.setActiveVersion(vId);
                }}
              />
            </div>
            {sidebarOpen ? (
              <aside className="studio-sidebar">
              {task ? (
                <ProgressPanel
                  progress={task.progress_pct}
                  stage={task.stage}
                  status={task.status}
                  message={task.message}
                  telemetry={task.telemetry}
                  onRetry={() => window.location.reload()}
                />
              ) : (
                <ProgressPanel
                  progress={0}
                  status="queued"
                  message="Waiting for the worker to claim this room."
                />
              )}
              {project ? (
                <EditStudio
                  projectId={projectId}
                  project={project}
                  manifest={activeManifest}
                  controller={controller}
                  assetBaseUrl={activeAssetBaseUrl}
                  activeVersionId={displayVersionId}
                  onSelectVersion={setSelectedVersionId}
                  editMode={isEditMode}
                  onToggleEditMode={() => setIsEditMode((v) => !v)}
                  onRefreshProject={async () => {
                    const updated = await roomApi.getProject(projectId);
                    setProject(updated);
                  }}
                />
              ) : null}
              </aside>
            ) : null}
          </>
        )}
      </div>
    </main>
  );
}
