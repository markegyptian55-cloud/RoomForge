"use client";

import React, { useState } from "react";
import type { DiffusionModelVariant } from "@/lib/api";

export interface FloatingEditBarProps {
  selectedObject: string | null;
  onClose: () => void;
  onSubmit: (prompt: string, model: DiffusionModelVariant) => Promise<void>;
  busy: boolean;
  statusMessage?: string;
  versions?: Array<{ id: string; label: string }>;
  activeVersionId?: string | null;
  onSelectVersion?: (id: string) => void;
}

const MODEL_INFO: Record<DiffusionModelVariant, { label: string; desc: string }> = {
  flux2_klein_4b: {
    label: "FLUX.2 [klein] 4B (Next-Gen Apache-2.0)",
    desc: "Ultra-fast next-gen commercial diffusion. Excels at modern architectural fabrics, leather, and wood textures.",
  },
  sdxl_inpaint_t4: {
    label: "SDXL Inpainting (Turing T4 Optimized)",
    desc: "Classic high-detail photorealistic generative inpainting with deep specular highlights.",
  },
};

export default function FloatingEditBar({
  selectedObject,
  onClose,
  onSubmit,
  busy,
  statusMessage,
  versions = [],
  activeVersionId,
  onSelectVersion,
}: FloatingEditBarProps) {
  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState<DiffusionModelVariant>("flux2_klein_4b");
  const [showAdvanced, setShowAdvanced] = useState(false);

  if (!selectedObject) return null;

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey && prompt.trim() && !busy) {
      e.preventDefault();
      void onSubmit(prompt.trim(), model);
    }
    if (e.key === "Escape") {
      onClose();
    }
  };

  return (
    <div
      role="dialog"
      aria-label="AI Generative 3D Editor"
      style={{
        position: "absolute",
        bottom: "76px",
        left: "50%",
        transform: "translateX(-50%)",
        width: "92%",
        maxWidth: "580px",
        background: "rgba(15, 23, 42, 0.94)",
        backdropFilter: "blur(16px)",
        border: "1px solid rgba(45, 212, 191, 0.5)",
        borderRadius: "16px",
        padding: "16px 20px",
        boxShadow: "0 20px 40px rgba(0, 0, 0, 0.6), 0 0 20px rgba(45, 212, 191, 0.15)",
        color: "#f8fafc",
        zIndex: 30,
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        animation: "fadeInUp 0.25s ease-out",
      }}
    >
      {/* Header with Selected Object & Close */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span
            style={{
              background: "rgba(45, 212, 191, 0.2)",
              color: "#2dd4bf",
              padding: "3px 10px",
              borderRadius: "999px",
              fontSize: "11px",
              fontWeight: 700,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
            }}
          >
            Target Object
          </span>
          <strong style={{ fontSize: "14px", color: "#f1f5f9" }}>{selectedObject}</strong>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          style={{
            background: "transparent",
            border: 0,
            color: "#94a3b8",
            fontSize: "18px",
            lineHeight: 1,
            padding: "4px 8px",
            cursor: "pointer",
            borderRadius: "6px",
          }}
        >
          ✕
        </button>
      </div>

      {/* Main Prompt Input Bar */}
      <div style={{ display: "flex", gap: "8px" }}>
        <input
          type="text"
          autoFocus
          value={prompt}
          placeholder={`Replace ${selectedObject.toLowerCase()} with... (e.g. Modern camel leather armchair)`}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            e.stopPropagation();
            handleKeyDown(e);
          }}
          onKeyUp={(e) => e.stopPropagation()}
          onKeyPress={(e) => e.stopPropagation()}
          maxLength={1000}
          disabled={busy}
          style={{
            flex: 1,
            background: "#090d16",
            border: "1px solid #334155",
            borderRadius: "10px",
            padding: "10px 14px",
            color: "#f8fafc",
            fontSize: "13px",
            outline: "none",
            boxShadow: "inset 0 1px 3px rgba(0,0,0,0.5)",
          }}
        />
        <button
          type="button"
          disabled={busy || !prompt.trim()}
          onClick={() => void onSubmit(prompt.trim(), model)}
          style={{
            background: prompt.trim() && !busy ? "#10b981" : "#1e293b",
            color: prompt.trim() && !busy ? "#ffffff" : "#64748b",
            border: 0,
            borderRadius: "10px",
            padding: "10px 18px",
            fontWeight: 700,
            fontSize: "13px",
            cursor: prompt.trim() && !busy ? "pointer" : "not-allowed",
            display: "inline-flex",
            alignItems: "center",
            gap: "6px",
            transition: "all 0.2s",
            boxShadow: prompt.trim() && !busy ? "0 0 15px rgba(16, 185, 129, 0.4)" : "none",
          }}
        >
          {busy ? "Generating…" : "✨ Generate"}
        </button>
      </div>

      {/* AI Model Selector */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "4px",
          background: "rgba(15, 23, 42, 0.6)",
          padding: "8px 12px",
          borderRadius: "8px",
          border: "1px solid #1e293b",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <label style={{ fontSize: "11px", fontWeight: 700, color: "#94a3b8", display: "flex", alignItems: "center", gap: "6px" }}>
            <span>🤖 AI Diffusion Engine:</span>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value as DiffusionModelVariant)}
              disabled={busy}
              style={{
                background: "#0c111d",
                color: "#2dd4bf",
                border: "1px solid #334155",
                borderRadius: "6px",
                padding: "3px 8px",
                fontSize: "11px",
                fontWeight: 600,
                outline: "none",
                cursor: "pointer",
              }}
            >
              <option value="flux2_klein_4b">FLUX.2 [klein] 4B (Fast, Apache-2.0)</option>
              <option value="sdxl_inpaint_t4">SDXL Inpainting (High Detail, T4)</option>
            </select>
          </label>
          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            style={{
              background: "transparent",
              border: 0,
              color: "#64748b",
              fontSize: "10px",
              cursor: "pointer",
              textDecoration: "underline",
            }}
          >
            {showAdvanced ? "Hide Info" : "Model Info"}
          </button>
        </div>
        {showAdvanced ? (
          <p style={{ margin: "4px 0 0", fontSize: "11px", color: "#94a3b8", lineHeight: 1.4 }}>
            💡 {MODEL_INFO[model].desc}
          </p>
        ) : null}
      </div>

      {/* Status feedback & A/B version switches */}
      {statusMessage ? (
        <p style={{ margin: 0, fontSize: "12px", color: "#6ee7b7", textAlign: "center" }}>
          {statusMessage}
        </p>
      ) : null}

      {versions.length > 0 ? (
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap", borderTop: "1px solid #1e293b", paddingTop: "8px" }}>
          <span style={{ fontSize: "11px", color: "#64748b" }}>Quick Version Switch:</span>
          {versions.map((v) => {
            const active = v.id === activeVersionId;
            return (
              <button
                key={v.id}
                type="button"
                onClick={() => onSelectVersion?.(v.id)}
                style={{
                  background: active ? "#064e3b" : "#1e293b",
                  borderColor: active ? "#2dd4bf" : "#334155",
                  borderWidth: "1px",
                  borderStyle: "solid",
                  color: active ? "#6ee7b7" : "#cbd5e1",
                  fontSize: "11px",
                  padding: "3px 10px",
                  borderRadius: "6px",
                  cursor: "pointer",
                  fontWeight: active ? 700 : 500,
                }}
              >
                {v.label}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
