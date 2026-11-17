"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowUpRight, Camera, Film, Info, Plus, Sparkles, UploadCloud } from "lucide-react";
import { useEffect, useState } from "react";
import { roomApi, type Project } from "@/lib/api";
import { ThemeToggle } from "@/components/ThemeToggle";

const DEMO_PROJECTS: Project[] = [
  {
    id: "realistic-room",
    name: "Modern Luxury Suite",
    status: "completed",
    active_version_id: "realistic-v1",
    created_at: new Date().toISOString(),
  },
  {
    id: "demo-room",
    name: "Living Room (3D Gaussian Splats)",
    status: "completed",
    active_version_id: "v1",
    created_at: new Date().toISOString(),
  },
];

export default function HomePage() {
  const router = useRouter();
  const [projects, setProjects] = useState<Project[]>(DEMO_PROJECTS);
  const [name, setName] = useState("Modern Living Room");
  const [open, setOpen] = useState(false);
  const [uploadMode, setUploadMode] = useState<"video" | "photo">("video");
  const [file, setFile] = useState<File | null>(null);
  const [uploadPct, setUploadPct] = useState<number | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    roomApi
      .listProjects()
      .then((data) => {
        if (data && data.length > 0) {
          // Filter out synthetic / E2E automated test artifacts
          const realUserProjects = data.filter(
            (p) =>
              p.id !== "realistic-room" &&
              p.id !== "demo-room" &&
              !p.name.toLowerCase().includes("e2e") &&
              !p.name.toLowerCase().includes("synthetic") &&
              !p.name.toLowerCase().includes("test")
          );
          setProjects([...DEMO_PROJECTS, ...realUserProjects]);
        } else {
          setProjects(DEMO_PROJECTS);
        }
      })
      .catch(() => {
        setProjects(DEMO_PROJECTS);
      });
  }, []);

  const create = async () => {
    if (!file) {
      setError(uploadMode === "video" ? "Please choose a room walkthrough video first." : "Please choose a room photo first.");
      return;
    }
    setError("");
    try {
      const project = await roomApi.createProject(name.trim() || "Untitled Room");
      await roomApi.uploadVideo(project.id, file, (pct) => setUploadPct(pct));
      router.push(`/projects/${project.id}`);
    } catch (reason) {
      setUploadPct(null);
      setError(reason instanceof Error ? reason.message : "Could not initialize room build.");
    }
  };

  return (
    <main className="shell">
      <header className="topbar">
        <Link className="brand" href="/">
          ROOM<span>FORGE</span>
        </Link>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              padding: "4px 12px",
              background: "rgba(245, 158, 11, 0.1)",
              border: "1px solid rgba(245, 158, 11, 0.25)",
              borderRadius: "999px",
              fontSize: "12px",
              fontWeight: 600,
              color: "#fbbf24",
            }}
          >
            <Sparkles size={13} />
            <span>Kaggle Dual Tesla T4 Ready</span>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <div className="workspace">
        <div className="workspace-head">
          <div style={{ maxWidth: "700px" }}>
            <p className="eyebrow">Spatial Generative Studio</p>
            <h1>Craft &amp; Restage 3D Rooms</h1>
            <p className="muted">
              Turn casual smartphone videos or photos into photorealistic, browsable 3D Gaussian rooms.
              Click any furniture item to redesign and restage it with generative AI in real time.
            </p>
          </div>
          <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
            <button
              className="button primary"
              onClick={() => {
                setUploadMode("video");
                setOpen(true);
              }}
            >
              <Plus size={18} /> New Room Video
            </button>
            <button
              className="button secondary"
              onClick={() => {
                setUploadMode("photo");
                setOpen(true);
              }}
            >
              <Camera size={18} /> Single Photo 3D
            </button>
          </div>
        </div>

        <div className="project-grid">
          {projects
            .sort((a, b) => (a.id === "realistic-room" ? -1 : a.id === "demo-room" ? (b.id === "realistic-room" ? 1 : -1) : 1))
            .map((project) => {
              const isRealistic = project.id === "realistic-room";
              const isSplatDemo = project.id === "demo-room";
              return (
                <Link
                  className="project-card"
                  href={`/projects/${project.id}`}
                  key={project.id}
                  style={
                    isRealistic
                      ? {
                          borderColor: "rgba(245, 158, 11, 0.5)",
                          background: "linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(15, 23, 42, 0.8) 100%)",
                          boxShadow: "0 10px 30px rgba(245, 158, 11, 0.15)",
                        }
                      : isSplatDemo
                        ? {
                            borderColor: "rgba(45, 212, 191, 0.5)",
                            background: "linear-gradient(135deg, rgba(45, 212, 191, 0.08) 0%, rgba(15, 23, 42, 0.8) 100%)",
                            boxShadow: "0 10px 30px rgba(45, 212, 191, 0.15)",
                          }
                        : undefined
                  }
                >
                  <div className="card-mark" style={isRealistic ? { color: "#fbbf24" } : isSplatDemo ? { color: "#2dd4bf" } : undefined}>
                    {isRealistic ? <Sparkles size={24} /> : <Film size={24} />}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                      <h2 style={{ fontSize: "16px", fontWeight: 700 }}>{project.name}</h2>
                      {isRealistic ? (
                        <span
                          style={{
                            background: "rgba(245, 158, 11, 0.18)",
                            color: "#fbbf24",
                            fontSize: "10px",
                            fontWeight: 700,
                            padding: "2px 8px",
                            borderRadius: "999px",
                            border: "1px solid rgba(245, 158, 11, 0.4)",
                            letterSpacing: "0.05em",
                          }}
                        >
                          👑 PHOTOREALISTIC 3D MESH
                        </span>
                      ) : isSplatDemo ? (
                        <span
                          style={{
                            background: "rgba(45, 212, 191, 0.18)",
                            color: "#2dd4bf",
                            fontSize: "10px",
                            fontWeight: 700,
                            padding: "2px 8px",
                            borderRadius: "999px",
                            border: "1px solid rgba(45, 212, 191, 0.4)",
                            letterSpacing: "0.05em",
                          }}
                        >
                          🪐 3D GAUSSIAN SPLATS
                        </span>
                      ) : (
                        <span
                          style={{
                            background: "rgba(148, 163, 184, 0.12)",
                            color: "#cbd5e1",
                            fontSize: "10px",
                            fontWeight: 600,
                            padding: "2px 8px",
                            borderRadius: "999px",
                          }}
                        >
                          CUSTOM ROOM
                        </span>
                      )}
                    </div>
                    <p style={{ color: "var(--muted)", fontSize: "12px", marginTop: "6px", lineHeight: "1.5" }}>
                      {isRealistic
                        ? "Solid architectural 3D interior with PBR oak parquet, velvet sectional, walnut coffee table, ceramic mug, and live AI restaging."
                        : isSplatDemo
                          ? "Radiance Gaussian cloud with 269,900 colorful splats (حبيبات الرمل) with full 360° orbital walkthrough & furniture restaging."
                          : "User-uploaded spatial room walkthrough with interactive 3D navigation."}
                    </p>
                  </div>
                  <ArrowUpRight size={20} color={isRealistic ? "#fbbf24" : isSplatDemo ? "#2dd4bf" : "var(--primary)"} />
                </Link>
              );
            })}
        </div>
      </div>

      {open && (
        <div className="modal-backdrop" role="dialog" aria-modal="true">
          <div className="modal">
            <button className="close" onClick={() => setOpen(false)} aria-label="Close">
              ×
            </button>
            <p className="eyebrow">
              {uploadMode === "video" ? "360° Walkthrough Reconstruction" : "Instant Single-Photo 3D"}
            </p>
            <h2>{uploadMode === "video" ? "Upload Room Walkthrough" : "Upload Room Photo"}</h2>

            <div style={{ display: "flex", gap: "8px", marginBottom: "18px" }}>
              <button
                type="button"
                onClick={() => setUploadMode("video")}
                style={{
                  flex: 1,
                  padding: "9px 14px",
                  borderRadius: "8px",
                  fontSize: "12px",
                  fontWeight: 600,
                  background: uploadMode === "video" ? "var(--primary)" : "var(--card-hover)",
                  color: uploadMode === "video" ? "#07090e" : "var(--ink)",
                  border: "1px solid var(--line)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "6px",
                  transition: "all 0.2s",
                }}
              >
                <Film size={14} /> Full Video (360° Walkable)
              </button>
              <button
                type="button"
                onClick={() => setUploadMode("photo")}
                style={{
                  flex: 1,
                  padding: "9px 14px",
                  borderRadius: "8px",
                  fontSize: "12px",
                  fontWeight: 600,
                  background: uploadMode === "photo" ? "var(--primary)" : "var(--card-hover)",
                  color: uploadMode === "photo" ? "#07090e" : "var(--ink)",
                  border: "1px solid var(--line)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "6px",
                  transition: "all 0.2s",
                }}
              >
                <Camera size={14} /> Single Photo (Fast 3D)
              </button>
            </div>

            {/* Smart Video / Photo Shooting Guidelines */}
            <div className="guide-grid">
              <div className="guide-item">
                <strong>
                  <Info size={12} /> Optimal Duration
                </strong>
                <span>
                  {uploadMode === "video"
                    ? "15 – 30 seconds (30 – 60 keyframes sampled)"
                    : "High resolution landscape orientation (1080p+)"}
                </span>
              </div>
              <div className="guide-item">
                <strong>
                  <Info size={12} /> Motion &amp; Angles
                </strong>
                <span>
                  {uploadMode === "video"
                    ? "Smooth orbital arc or half-circle at chest height"
                    : "Wide field of view showing floor, walls & main furniture"}
                </span>
              </div>
              <div className="guide-item">
                <strong>
                  <Info size={12} /> Staging Conditions
                </strong>
                <span>
                  Works with <strong>furnished rooms</strong> (to replace/redesign) or <strong>empty rooms</strong> (to stage from scratch).
                </span>
              </div>
              <div className="guide-item">
                <strong>
                  <Info size={12} /> Lighting Standard
                </strong>
                <span>Bright, uniform indirect daylight. Avoid dark corners or direct camera flash.</span>
              </div>
            </div>

            <label style={{ margin: "14px 0 6px" }}>
              Room Project Name
              <input
                type="text"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="e.g. Minimalist Master Bedroom"
              />
            </label>

            <label className="dropzone">
              <UploadCloud size={32} color="var(--primary)" />
              {file ? (
                <div style={{ color: "var(--ink)", fontWeight: 600 }}>{file.name}</div>
              ) : (
                <>
                  <strong style={{ color: "var(--ink)", fontSize: "14px" }}>
                    {uploadMode === "video"
                      ? "Choose or drop a room video (MP4, MOV, WebM)"
                      : "Choose or drop a room photo (JPG, PNG, WebP)"}
                  </strong>
                  <span style={{ fontSize: "12px", color: "var(--muted)" }}>
                    Tap to upload or shoot directly with your smartphone camera (up to 500 MB)
                  </span>
                </>
              )}
              <input
                type="file"
                accept={
                  uploadMode === "video"
                    ? "video/mp4,video/quicktime,video/webm"
                    : "image/jpeg,image/png,image/webp,video/mp4,video/quicktime"
                }
                capture="environment"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
            </label>

            {/* Direct Mobile & Laptop Camera Capture Action */}
            <div style={{ marginTop: "10px" }}>
              <label
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: "8px",
                  padding: "10px 16px",
                  background: "rgba(245, 158, 11, 0.14)",
                  border: "1px solid rgba(245, 158, 11, 0.4)",
                  borderRadius: "8px",
                  color: "#fbbf24",
                  fontSize: "13px",
                  fontWeight: 700,
                  cursor: "pointer",
                  textAlign: "center",
                  transition: "all 0.2s",
                }}
              >
                <Camera size={16} />
                <span>{uploadMode === "video" ? "🎥 Record Room with Camera" : "📸 Open Camera & Snap Room"}</span>
                <input
                  type="file"
                  accept={uploadMode === "video" ? "video/*" : "image/*"}
                  capture="environment"
                  style={{ display: "none" }}
                  onChange={(event) => {
                    if (event.target.files?.[0]) setFile(event.target.files[0]);
                  }}
                />
              </label>
            </div>

            {uploadPct !== null && (
              <div style={{ margin: "12px 0" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", marginBottom: "4px" }}>
                  <span>Uploading to Kaggle compute queue…</span>
                  <span style={{ color: "var(--primary)", fontWeight: 700 }}>{uploadPct}%</span>
                </div>
                <div style={{ width: "100%", height: "6px", background: "rgba(255, 255, 255, 0.1)", borderRadius: "3px", overflow: "hidden" }}>
                  <div style={{ width: `${uploadPct}%`, height: "100%", background: "var(--primary)", transition: "width 0.2s" }} />
                </div>
              </div>
            )}

            {error && <p className="form-error">{error}</p>}

            <button
              className="button primary full"
              disabled={uploadPct !== null}
              onClick={() => void create()}
              style={{ marginTop: "16px" }}
            >
              Upload &amp; Build 3D Room <ArrowUpRight size={16} />
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
