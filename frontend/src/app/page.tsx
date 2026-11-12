"use client";

import Link from "next/link";
import { ArrowUpRight, Plus, UploadCloud } from "lucide-react";
import { useEffect, useState } from "react";
import { roomApi, type Project } from "@/lib/api";

export default function HomePage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [name, setName] = useState("Living room");
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    void roomApi.listProjects().then(setProjects).catch(() => undefined);
  }, []);

  const create = async () => {
    if (!file) {
      setError("Choose a room video first.");
      return;
    }
    setError("");
    try {
      const project = await roomApi.createProject(name.trim() || "Untitled room");
      await roomApi.uploadVideo(project.id, file, (pct) => setUploadPct(pct));
      window.location.href = `/projects/${project.id}`;
    } catch (reason) {
      setUploadPct(null);
      setError(reason instanceof Error ? reason.message : "Could not start the room build.");
    }
  };

  return (
    <main className="shell">
      <header className="topbar">
        <Link className="brand" href="/">ROOM<span>3D</span></Link>
        <span className="header-note">PRIVATE WORKSPACE</span>
      </header>
      <div className="workspace">
        <div className="workspace-head">
          <div>
            <p className="eyebrow">YOUR LIBRARY</p>
            <h1>Projects</h1>
            <p className="muted">Build a room you can walk through.</p>
          </div>
          <button className="button primary" onClick={() => setOpen(true)}>
            <Plus size={17} /> New room video
          </button>
        </div>
        <div className="project-grid">
          {projects.map((project) => (
            <Link className="project-card" href={`/projects/${project.id}`} key={project.id}>
              <div className="card-mark">{project.name.slice(0, 1).toUpperCase()}</div>
              <div>
                <h2>{project.name}</h2>
                <p>{project.status}</p>
              </div>
              <ArrowUpRight size={18} />
            </Link>
          ))}
          {projects.length === 0 && (
            <button className="empty-card" onClick={() => setOpen(true)}>
              <UploadCloud size={24} />
              <strong>Start with a room video</strong>
              <span>Upload an MP4, MOV, or WebM to create your first scene.</span>
            </button>
          )}
        </div>
      </div>
      {open && (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <div className="modal">
            <button className="close" onClick={() => setOpen(false)} aria-label="Close">×</button>
            <p className="eyebrow">NEW PROJECT</p>
            <h2>Bring a room to life.</h2>
            <label>
              Project name
              <input value={name} onChange={(event) => setName(event.target.value)} />
            </label>
            <label className="dropzone">
              {file ? file.name : "Choose a video file"}
              <input
                type="file"
                accept="video/mp4,video/quicktime,video/webm"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
            </label>
            {uploadPct !== null && <p className="muted">Uploading {uploadPct}%…</p>}
            {error && <p className="form-error">{error}</p>}
            <button className="button primary full" disabled={uploadPct !== null} onClick={() => void create()}>
              Upload and build <ArrowUpRight size={16} />
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
