"use client";

import React, { useCallback, useEffect, useState } from "react";
import ObjectSelector, { type PromptPoint } from "@/components/ObjectSelector";
import TrackingGrid from "@/components/TrackingGrid";
import {
  ApiError,
  roomApi,
  type DiffusionModelVariant,
  type EditFramesResponse,
  type Manifest,
  type ProjectDetail,
  type RebuildMode,
  type ViewerClickEvent,
  type ViewerController,
} from "@/lib/api";

export interface EditStudioProps {
  projectId: string;
  project: ProjectDetail;
  manifest?: Manifest;
  controller: ViewerController | null;
  assetBaseUrl: string;
  activeVersionId: string | null;
  onSelectVersion: (versionId: string) => void;
  onRefreshProject: () => Promise<void>;
  editMode?: boolean;
  onToggleEditMode?: () => void;
}

const DEFAULT_SEEDS = [101, 202, 303];

function manifestImageSize(manifest?: Manifest): { width: number; height: number } {
  const width = typeof manifest?.width === "number" ? manifest.width : 640;
  const height = typeof manifest?.height === "number" ? manifest.height : 480;
  return { width, height };
}

function frameRelpathFor(manifest: Manifest | undefined, frameIdx: number): string | null {
  const rawFrames = Array.isArray(manifest?.frames) ? manifest.frames : [];
  const hit = rawFrames.find(
    (item): item is { frame_idx: number; image_relpath?: string } =>
      typeof item === "object" && item !== null && "frame_idx" in item && item.frame_idx === frameIdx,
  );
  return typeof hit?.image_relpath === "string" ? hit.image_relpath : null;
}

export default function EditStudio({
  projectId,
  project,
  manifest,
  controller,
  assetBaseUrl,
  activeVersionId,
  onSelectVersion,
  onRefreshProject,
  editMode: propsEditMode,
  onToggleEditMode,
}: EditStudioProps) {
  const [internalEditMode, setInternalEditMode] = useState(false);
  const editMode = propsEditMode !== undefined ? propsEditMode : internalEditMode;
  const toggleEditMode = onToggleEditMode ?? (() => setInternalEditMode((v) => !v));
  const [editId, setEditId] = useState<string | null>(null);
  const [refFrameIdx, setRefFrameIdx] = useState<number>(0);
  const [points, setPoints] = useState<PromptPoint[]>([]);
  const [framesData, setFramesData] = useState<EditFramesResponse | null>(null);
  const [prompt, setPrompt] = useState("");
  const [modelVariant, setModelVariant] = useState<DiffusionModelVariant>("flux2_klein_4b");
  const [chosenSeed, setChosenSeed] = useState("seed_101");
  const [rebuildMode, setRebuildMode] = useState<RebuildMode>("masked_finetune");
  const [busy, setBusy] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [error, setError] = useState("");

  const loadFrames = useCallback(async (targetEditId: string, page = 1) => {
    const data = await roomApi.editFrames(targetEditId, page);
    setFramesData(data);
  }, []);

  const runAction = useCallback(async (action: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Operation failed.");
    } finally {
      setBusy(false);
    }
  }, []);

  const handleViewerClick = useCallback(
    async (click: ViewerClickEvent) => {
      await runAction(async () => {
        const res = await roomApi.resolveClick(projectId, click);
        setEditId(res.edit_id);
        setRefFrameIdx(res.ref_frame_idx);
        setPoints([{ x: Math.round(res.u), y: Math.round(res.v), label: 1 }]);
        setStatusMessage(`Selected reference frame #${res.ref_frame_idx}`);
        await loadFrames(res.edit_id, 1);
      });
    },
    [loadFrames, projectId, runAction],
  );

  useEffect(() => {
    if (!controller || !editMode) return;
    return controller.onUserClick((click) => {
      void handleViewerClick(click);
    });
  }, [controller, editMode, handleViewerClick]);

  const handleAddPoint = async (pt: PromptPoint) => {
    if (!editId) return;
    await runAction(async () => {
      await roomApi.refine(editId, [pt]);
      setPoints((prev) => [...prev, pt]);
      setStatusMessage("Refined segmentation mask.");
    });
  };

  const handleTrack = async () => {
    if (!editId) return;
    await runAction(async () => {
      await roomApi.track(editId);
      setStatusMessage("Queued cross-frame tracking.");
      await loadFrames(editId, 1);
    });
  };

  const handleToggleSkip = async (frameIdx: number, skip: boolean) => {
    if (!editId) return;
    await runAction(async () => {
      await roomApi.updateMask(editId, frameIdx, { skip_frame: skip });
      await loadFrames(editId, framesData?.page ?? 1);
    });
  };

  const handleUploadMask = async (frameIdx: number, file: File) => {
    if (!editId) return;
    const isPng = file.name.toLowerCase().endsWith(".png") && (!file.type || file.type === "image/png");
    if (!isPng) {
      setError("Custom mask must be a 1-channel uint8 PNG file (.png).");
      return;
    }
    if (file.size === 0) {
      setError("Custom mask file cannot be empty.");
      return;
    }
    if (file.size > 16 * 1024 * 1024) {
      setError("Custom mask exceeds maximum allowed size of 16 MB.");
      return;
    }
    const form = new FormData();
    form.append("file", file);
    await runAction(async () => {
      await roomApi.updateMask(editId, frameIdx, form);
      await loadFrames(editId, framesData?.page ?? 1);
    });
  };

  const handlePreviewSeeds = async () => {
    if (!editId || !prompt.trim()) {
      setError("Enter an edit prompt before generating previews.");
      return;
    }
    await runAction(async () => {
      await roomApi.previewSeeds(editId, prompt.trim(), DEFAULT_SEEDS, modelVariant);
      setStatusMessage("Seed preview generation queued.");
    });
  };

  const handleApplyRebuild = async () => {
    if (!editId || !prompt.trim()) {
      setError("Enter a prompt and choose a seed before applying rebuild.");
      return;
    }
    await runAction(async () => {
      const res = await roomApi.applyEdit(editId, chosenSeed, rebuildMode, prompt.trim());
      setStatusMessage(`Rebuild queued for version ${res.target_version_id}`);
      await onRefreshProject();
    });
  };

  const { width, height } = manifestImageSize(manifest);
  const relpath = frameRelpathFor(manifest, refFrameIdx);
  const refImageUrl = relpath && assetBaseUrl ? `${assetBaseUrl}/${relpath}` : null;
  const rawFrames = Array.isArray(manifest?.frames) ? manifest.frames : [];
  const activeFrameMaskKey = framesData?.frames.find((f) => f.frame_idx === refFrameIdx)?.mask_key;
  const activeMaskUrl = activeFrameMaskKey && assetBaseUrl ? `${assetBaseUrl}/${activeFrameMaskKey}` : null;

  return (
    <section className="glass-card" aria-label="Week 2 Edit Studio" style={{ padding: "16px", display: "grid", gap: "14px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
        <div>
          <p className="eyebrow">WEEK 2 EDIT STUDIO</p>
          <strong>Interactive Object Editing</strong>
        </div>
        <button
          type="button"
          aria-pressed={editMode}
          onClick={toggleEditMode}
          style={{
            borderColor: editMode ? "#2dd4bf" : "#3b82f6",
            background: editMode ? "#064e3b" : "#1e293b",
            color: editMode ? "#6ee7b7" : "#f8fafc",
            fontWeight: 700,
            fontSize: "12px",
            padding: "8px 14px",
            borderRadius: "8px",
            borderWidth: "2px",
            boxShadow: editMode ? "0 0 12px rgba(45, 212, 191, 0.4)" : "none",
            display: "inline-flex",
            alignItems: "center",
            gap: "6px",
          }}
        >
          {editMode ? "🎯 Edit Object Mode: ON" : "✏️ Edit Object Mode: OFF (Press E)"}
        </button>
      </div>

      {/* 2D Keyframe Sample Carousel (1-4 fps) */}
      {rawFrames.length > 0 ? (
        <div style={{ display: "grid", gap: "6px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--muted, #94a3b8)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              📸 Sampled Keyframes ({rawFrames.length} views)
            </span>
            <span style={{ fontSize: "11px", color: "#38bdf8" }}>
              Active Frame #{refFrameIdx}
            </span>
          </div>
          <div
            style={{
              display: "flex",
              gap: "8px",
              overflowX: "auto",
              paddingBottom: "6px",
              scrollbarWidth: "thin",
            }}
          >
            {rawFrames.map((f, idx) => {
              const frameIdx = typeof f === "object" && f !== null && "frame_idx" in f ? (f.frame_idx as number) : idx;
              const rel = typeof f === "object" && f !== null && "image_relpath" in f ? (f.image_relpath as string) : "";
              const src = rel && assetBaseUrl ? `${assetBaseUrl}/${rel}` : null;
              const isSelected = frameIdx === refFrameIdx;
              return (
                <button
                  key={frameIdx}
                  type="button"
                  onClick={() => {
                    setRefFrameIdx(frameIdx);
                    if (!editId) {
                      setEditId(`session-${frameIdx}`);
                      setPoints([{ x: Math.round(width / 2), y: Math.round(height / 2), label: 1 }]);
                    }
                  }}
                  style={{
                    flex: "0 0 76px",
                    display: "grid",
                    gap: "2px",
                    padding: "3px",
                    background: isSelected ? "rgba(45, 212, 191, 0.2)" : "#0f172a",
                    border: isSelected ? "2px solid #2dd4bf" : "1px solid #334155",
                    borderRadius: "6px",
                    cursor: "pointer",
                    textAlign: "center",
                  }}
                >
                  <div style={{ width: "100%", height: "48px", overflow: "hidden", borderRadius: "3px", background: "#1e293b" }}>
                    {src ? (
                      <img src={src} alt={`Keyframe ${frameIdx}`} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                    ) : (
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", fontSize: "10px", color: "#64748b" }}>
                        #{frameIdx}
                      </div>
                    )}
                  </div>
                  <span style={{ fontSize: "9px", color: isSelected ? "#2dd4bf" : "#94a3b8", fontWeight: 600 }}>
                    #{frameIdx}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      ) : null}

      {editMode && !editId ? (
        <div style={{ background: "#064e3b33", border: "1px solid #10b981", borderRadius: "8px", padding: "10px", color: "#6ee7b7", fontSize: "13px" }}>
          👉 <strong>Object Selection is Active:</strong> Click anywhere on the sofa or table in the 3D room to select it and start editing!
        </div>
      ) : null}
      {!editMode && !editId ? (
        <p style={{ color: "var(--muted)", fontSize: "12px", margin: 0 }}>
          💡 To select an object for editing: Click <strong>&quot;Edit Object Mode: OFF&quot;</strong> above to turn it ON, then click any object in the 3D room.
        </p>
      ) : null}

      {project.versions.length > 0 ? (
        <div role="group" aria-label="Scene Version Switcher" style={{ display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" }}>
          <span>A/B Versions:</span>
          {project.versions.map((ver) => {
            const active = ver.id === activeVersionId;
            return (
              <button
                key={ver.id}
                type="button"
                aria-pressed={active}
                onClick={() => {
                  onSelectVersion(ver.id);
                  controller?.setActiveVersion(ver.id);
                }}
                style={{ borderColor: active ? "#2dd4bf" : undefined }}
              >
                {ver.label || ver.id}
              </button>
            );
          })}
        </div>
      ) : null}

      {editMode && !editId ? <p className="hint-text">Click any object inside the 3D scene to start masked segmentation.</p> : null}
      {statusMessage ? <p className="hint-text">{statusMessage}</p> : null}
      {error ? <p role="alert" className="error-text">{error}</p> : null}

      {editId ? (
        <div style={{ display: "grid", gap: "14px" }}>
          {/* Primary Generative Controls */}
          <div style={{ display: "grid", gap: "10px", background: "rgba(15, 23, 42, 0.6)", padding: "12px", borderRadius: "10px", border: "1px solid #1e293b" }}>
            <label style={{ display: "grid", gap: "4px" }}>
              <span style={{ fontWeight: 600, fontSize: "13px" }}>Edit Instruction Prompt</span>
              <input
                type="text"
                aria-label="Edit instruction prompt"
                maxLength={1000}
                value={prompt}
                placeholder="e.g. modern emerald velvet armchair"
                onChange={(e) => setPrompt(e.target.value)}
              />
            </label>

            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" }}>
              <label style={{ display: "grid", gap: "2px", fontSize: "12px", flex: 1, minWidth: "160px" }}>
                <span>AI Model:</span>
                <select
                  aria-label="Diffusion Model Variant"
                  value={modelVariant}
                  onChange={(e) => setModelVariant(e.target.value as DiffusionModelVariant)}
                >
                  <option value="flux2_klein_4b">FLUX.2 [klein] 4B (Apache-2.0)</option>
                  <option value="sdxl_inpaint_t4">SDXL Inpaint (T4 Fallback)</option>
                </select>
              </label>

              <label style={{ display: "grid", gap: "2px", fontSize: "12px" }}>
                <span>Seed:</span>
                <select aria-label="Chosen Diffusion Seed" value={chosenSeed} onChange={(e) => setChosenSeed(e.target.value)}>
                  {DEFAULT_SEEDS.map((s) => (
                    <option key={s} value={`seed_${s}`}>{`seed_${s}`}</option>
                  ))}
                </select>
              </label>
            </div>

            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", marginTop: "4px" }}>
              <button
                type="button"
                className="button primary"
                aria-label="Apply & Rebuild 3D Room"
                disabled={busy}
                onClick={() => void handleApplyRebuild()}
                style={{ flex: 1, justifyContent: "center" }}
              >
                ✨ Apply &amp; Rebuild 3D Room
              </button>
              <button
                type="button"
                aria-label="Generate Seed Previews (Preview Seeds)"
                disabled={busy}
                onClick={() => void handlePreviewSeeds()}
                style={{ fontSize: "12px", padding: "6px 10px" }}
              >
                Generate Seed Previews (Preview Seeds)
              </button>
            </div>

            {/* First-Frame Seed Preview Gallery */}
            <div style={{ display: "grid", gap: "6px", background: "rgba(15, 23, 42, 0.7)", border: "1px solid #1e293b", borderRadius: "8px", padding: "10px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "12px", fontWeight: 600 }}>🌱 Diffusion Seed Candidates</span>
                <span style={{ fontSize: "11px", color: "#2dd4bf" }}>Selected: {chosenSeed}</span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "8px" }}>
                {DEFAULT_SEEDS.map((s) => {
                  const seedKey = `seed_${s}`;
                  const isSelected = chosenSeed === seedKey;
                  return (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setChosenSeed(seedKey)}
                      style={{
                        border: isSelected ? "2px solid #2dd4bf" : "1px solid #334155",
                        background: isSelected ? "rgba(45, 212, 191, 0.15)" : "#0b1220",
                        borderRadius: "6px",
                        padding: "6px",
                        display: "grid",
                        gap: "4px",
                        cursor: "pointer",
                        textAlign: "center",
                      }}
                    >
                      <div style={{ height: "46px", background: "#1e293b", borderRadius: "4px", display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden" }}>
                        <span style={{ fontSize: "11px", fontWeight: 700, color: isSelected ? "#2dd4bf" : "#94a3b8" }}>
                          Seed {s}
                        </span>
                      </div>
                      <span style={{ fontSize: "10px", color: isSelected ? "#2dd4bf" : "#cbd5e1" }}>
                        {isSelected ? "✓ Active" : "Select"}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Advanced Frame & Mask Inspection Collapsible Accordion */}
          <details
            open
            style={{
              background: "#090d16",
              border: "1px solid #1e293b",
              borderRadius: "8px",
              padding: "10px",
            }}
          >
            <summary style={{ cursor: "pointer", fontWeight: 600, fontSize: "12px", color: "#94a3b8" }}>
              🛠️ Advanced Frame Tracking &amp; Segmentation Masks ({framesData?.frames.length ?? 0} frames)
            </summary>
            <div style={{ maxHeight: "360px", overflowY: "auto", marginTop: "10px", display: "grid", gap: "12px" }}>
              <ObjectSelector
                editId={editId}
                refFrameIdx={refFrameIdx}
                points={points}
                imageUrl={refImageUrl}
                maskUrl={activeMaskUrl}
                imageWidth={width}
                imageHeight={height}
                busy={busy}
                onAddPoint={handleAddPoint}
                onConfirmTrack={handleTrack}
              />

              {framesData ? (
                <TrackingGrid
                  assetBaseUrl={assetBaseUrl}
                  framesData={framesData}
                  busy={busy}
                  onPageChange={(page) => runAction(() => loadFrames(editId, page))}
                  onToggleSkip={handleToggleSkip}
                  onUploadMask={handleUploadMask}
                />
              ) : null}

              <div style={{ borderTop: "1px solid #1e293b", paddingTop: "8px" }}>
                <label style={{ display: "grid", gap: "4px", fontSize: "12px" }}>
                  <span>Rebuild Mode (Ablation):</span>
                  <select
                    aria-label="Rebuild Mode"
                    value={rebuildMode}
                    onChange={(e) => setRebuildMode(e.target.value as RebuildMode)}
                  >
                    <option value="masked_finetune">Masked Fine-Tune (5k steps)</option>
                    <option value="from_scratch">From Scratch Ablation (15k steps)</option>
                  </select>
                </label>
              </div>
            </div>
          </details>
        </div>
      ) : null}
    </section>
  );
}
