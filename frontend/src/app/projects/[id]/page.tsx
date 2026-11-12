"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ProgressPanel } from "@/components/ProgressPanel";
import Viewer3D from "@/components/Viewer3D";
import { roomApi, type Manifest, type ProjectDetail, type TaskSummary } from "@/lib/api";

export default function ProjectPage({ params }: { params: Promise<{ id: string }> }) {
  const [projectId, setProjectId] = useState("");
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [error, setError] = useState("");
  const [manifest, setManifest] = useState<Manifest>();

  useEffect(() => {
    void params.then(({ id }) => {
      setProjectId(id);
      void roomApi.getProject(id).then(setProject).catch((reason: unknown) =>
        setError(reason instanceof Error ? reason.message : "Project unavailable."),
      );
    });
  }, [params]);

  useEffect(() => {
    if (!projectId || !project?.active_version_id) return;
    void roomApi.manifest(projectId, project.active_version_id).then(setManifest).catch(() => setManifest(undefined));
  }, [projectId, project?.active_version_id]);

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
                manifest={manifest as Parameters<typeof Viewer3D>[0]["manifest"]}
                assetBaseUrl={
                  projectId && project?.active_version_id
                    ? `/api/storage/${encodeURIComponent(projectId)}/${encodeURIComponent(project.active_version_id)}`
                    : ""
                }
              />
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
            </div>
          </>
        )}
      </div>
    </main>
  );
}
