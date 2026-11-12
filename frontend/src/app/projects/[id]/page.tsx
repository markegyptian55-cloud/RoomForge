"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import EditStudio from "@/components/EditStudio";
import { ProgressPanel } from "@/components/ProgressPanel";
import Viewer3D from "@/components/Viewer3D";
import { roomApi, type Manifest, type ProjectDetail, type TaskSummary, type ViewerController } from "@/lib/api";

export default function ProjectPage({ params }: { params: Promise<{ id: string }> }) {
  const [projectId, setProjectId] = useState("");
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [versionManifest, setVersionManifest] = useState<{ versionId: string; data: Manifest } | null>(null);
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(null);
  const [controller, setController] = useState<ViewerController | null>(null);
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

  return (
    <main className="shell">
      <header className="topbar">
        <Link className="brand" href="/">ROOM<span>3D</span></Link>
        <Link className="back-link" href="/">← Projects</Link>
      </header>
      <div className="scene-workspace">
        {error ? (
          <p className="form-error">{error}</p>
        ) : (
          <>
            <div className="scene-heading">
              <div>
                <p className="eyebrow">SCENE WORKSPACE</p>
                <h1>{project?.name ?? "Loading scene…"}</h1>
              </div>
              <span className="version-pill">{project?.active_version_id ? "ACTIVE VERSION" : "BASE BUILD"}</span>
            </div>
            <div className="scene-grid">
              <Viewer3D
                manifest={activeManifest as Parameters<typeof Viewer3D>[0]["manifest"]}
                onControllerReady={setController}
                onVersionChange={setSelectedVersionId}
                assetBaseUrl={activeAssetBaseUrl}
              />
              <div style={{ display: "grid", gap: "14px" }}>
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
                    onRefreshProject={async () => {
                      const updated = await roomApi.getProject(projectId);
                      setProject(updated);
                    }}
                  />
                ) : null}
              </div>
            </div>
          </>
        )}
      </div>
    </main>
  );
}
