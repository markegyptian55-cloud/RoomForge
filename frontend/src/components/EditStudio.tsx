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
}: EditStudioProps) {
  const [editMode, setEditMode] = useState(false);
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
          onClick={() => setEditMode((v) => !v)}
          style={{ borderColor: editMode ? "#2dd4bf" : undefined }}
        >
          {editMode ? "Edit Object Mode: ON" : "Edit Object Mode: OFF"}
        </button>
      </div>

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
          <ObjectSelector
            editId={editId}
            refFrameIdx={refFrameIdx}
            points={points}
            imageUrl={refImageUrl}
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

          <div style={{ display: "grid", gap: "10px", borderTop: "1px solid #334155", paddingTop: "10px" }}>
            <label style={{ display: "grid", gap: "4px" }}>
              <span>Edit Prompt</span>
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
              <label>
                Model:
                <select
                  aria-label="Diffusion Model Variant"
                  value={modelVariant}
                  onChange={(e) => setModelVariant(e.target.value as DiffusionModelVariant)}
                >
                  <option value="flux2_klein_4b">FLUX.2 [klein] 4B (Apache-2.0)</option>
                  <option value="sdxl_inpaint_t4">SDXL Inpaint (T4 Fallback)</option>
                </select>
              </label>

              <label>
                Seed:
                <select aria-label="Chosen Diffusion Seed" value={chosenSeed} onChange={(e) => setChosenSeed(e.target.value)}>
                  {DEFAULT_SEEDS.map((s) => (
                    <option key={s} value={`seed_${s}`}>{`seed_${s}`}</option>
                  ))}
                </select>
              </label>

              <label>
                Rebuild Mode:
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

            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
              <button type="button" disabled={busy} onClick={() => void handlePreviewSeeds()}>
                Generate Seed Previews
              </button>
              <button type="button" disabled={busy} onClick={() => void handleApplyRebuild()}>
                Apply &amp; Rebuild 3D Room
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
