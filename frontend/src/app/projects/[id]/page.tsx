"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ProgressPanel } from "@/components/ProgressPanel";
import { roomApi, type ProjectDetail, type TaskSummary } from "@/lib/api";

export default function ProjectPage({ params }: { params: Promise<{ id: string }> }) {
  const [projectId, setProjectId] = useState("");
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    void params.then(({ id }) => {
      setProjectId(id);
      void roomApi.getProject(id).then(setProject).catch((reason: unknown) =>
        setError(reason instanceof Error ? reason.message : "Project unavailable."),
      );
    });
  }, [params]);

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
              <div className="viewer-placeholder">
                <div className="viewer-grid" />
                <div>
                  <p className="eyebrow">VIEWER3D</p>
                  <h2>Your scene will appear here</h2>
                  <p>Interactive .spz rendering arrives in the next slice.</p>
                </div>
              </div>
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
