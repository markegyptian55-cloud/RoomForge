"use client";

import React, { useMemo, useState } from "react";
import { Sparkles, Wand2, X } from "lucide-react";
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
    desc: "Ultra-fast commercial generative diffusion on Kaggle Dual Tesla T4. Excels at architectural textiles, velvet, leather, and wood grains.",
  },
  sdxl_inpaint_t4: {
    label: "SDXL Inpainting (Turing T4 Optimized)",
    desc: "High-detail photorealistic masked inpainting with fine specular reflections.",
  },
};

interface StyleSuggestion {
  label: string;
  prompt: string;
  icon: string;
}

function getCuratedSuggestions(objectName: string): StyleSuggestion[] {
  const name = objectName.toLowerCase();

  // 1. Small tabletop & decor items (highest specificity)
  if (name.includes("mug") || name.includes("cup") || name.includes("vase") || name.includes("pot") || name.includes("bowl")) {
    return [
      { icon: "🏺", label: "Terracotta Vase", prompt: "Organic handcrafted raw terracotta ceramic vase" },
      { icon: "✨", label: "Amber Fluted Glass", prompt: "Hand-blown fluted amber glass with golden reflections" },
      { icon: "☕", label: "Sage Ceramic Mug", prompt: "Artisanal speckled matte sage green stoneware mug" },
      { icon: "🖤", label: "Obsidian Stoneware", prompt: "Architectural matte black volcanic stone vessel" },
    ];
  }

  // 2. Wall-mounted Art & Canvas (specific over raw walls)
  if (name.includes("canvas") || name.includes("art") || name.includes("frame") || name.includes("painting")) {
    return [
      { icon: "🎨", label: "Abstract Oil", prompt: "Large vibrant abstract expressionist canvas with textured impasto" },
      { icon: "🌻", label: "Golden Sunflowers", prompt: "Post-impressionist warm golden sunflowers in heavy oil paint" },
      { icon: "📐", label: "Bauhaus Shapes", prompt: "Mid-century Bauhaus geometric color-block canvas" },
      { icon: "🖤", label: "Monochrome Ink", prompt: "Minimalist Japanese sumi-e monochrome line drawing" },
    ];
  }

  // 3. Sofas & Seating
  if (name.includes("sofa") || name.includes("cushion") || name.includes("chair") || name.includes("couch") || name.includes("pillow")) {
    return [
      { icon: "🛋️", label: "Emerald Velvet", prompt: "Luxurious deep emerald green velvet with brass piping" },
      { icon: "🤎", label: "Cognac Leather", prompt: "Vintage hand-distressed Italian cognac leather" },
      { icon: "☁️", label: "Nordic Bouclé", prompt: "Minimalist Scandinavian cream bouclé fabric" },
      { icon: "🖤", label: "Matte Charcoal", prompt: "Modern architectural matte charcoal textured weave" },
    ];
  }

  // 4. Tables & Desks
  if (name.includes("table") || name.includes("desk") || name.includes("surface") || name.includes("stand")) {
    return [
      { icon: "🪵", label: "Live-Edge Walnut", prompt: "Solid organic live-edge American walnut wood" },
      { icon: "✨", label: "Calacatta Marble", prompt: "Polished Italian Calacatta gold marble with amber veining" },
      { icon: "🪨", label: "Brutalist Concrete", prompt: "Smooth hand-cast brutalist architectural concrete" },
      { icon: "🪞", label: "Smoked Glass", prompt: "Tempered smoked bronze glass with satin brass frame" },
    ];
  }

  // 5. Walls & Architectural Backdrops
  if (name.includes("wall") || name.includes("backdrop") || name.includes("partition")) {
    return [
      { icon: "🌿", label: "Sage Wood Slats", prompt: "Deep sage green accent wall with vertical oak acoustic slats" },
      { icon: "🏛️", label: "Venetian Plaster", prompt: "Textured warm beige Venetian limewash plaster" },
      { icon: "🧱", label: "Exposed Brick", prompt: "Rustic industrial exposed loft brick wall with warm mortar" },
      { icon: "🌌", label: "Midnight Blue", prompt: "Dramatic matte midnight blue accent wall with brass moldings" },
    ];
  }

  return [
    { icon: "✨", label: "Nordic Minimalist", prompt: "Light Scandinavian minimalist design with natural oak" },
    { icon: "🌿", label: "Japandi Organic", prompt: "Japandi aesthetic with wabi-sabi organic textures" },
    { icon: "🏛️", label: "Italian Luxury", prompt: "Contemporary Milanese luxury with brass accents and velvet" },
    { icon: "🖤", label: "Industrial Loft", prompt: "Urban industrial loft style with dark metals and leather" },
  ];
}

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

  const suggestions = useMemo(() => {
    return selectedObject ? getCuratedSuggestions(selectedObject) : [];
  }, [selectedObject]);

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

  const handleSelectSuggestion = (s: StyleSuggestion) => {
    setPrompt(s.prompt);
    void onSubmit(s.prompt, model);
  };

  return (
    <>
      {/* Tap anywhere outside to dismiss */}
      <div className="popover-backdrop" onClick={onClose} aria-hidden="true" />

      <div className="water-glass-popover" role="dialog" aria-label="AI Generative 3D Editor">
        {/* Mobile Drag / Pull Indicator */}
        <div className="sheet-drag-handle" onClick={onClose} title="Tap to close" />

        {/* Popover Header with prominent, easy-to-tap close button */}
        <div className="popover-header">
          <div className="popover-title">
            <Sparkles size={16} />
            <span>Restage:</span>
            <span
              style={{
                color: "#fff",
                background: "rgba(245, 158, 11, 0.18)",
                border: "1px solid rgba(245, 158, 11, 0.35)",
                padding: "2px 8px",
                borderRadius: "6px",
                fontSize: "12px",
                fontWeight: 700,
              }}
            >
              {selectedObject}
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close editor"
            className="popover-close-btn"
            style={{
              color: "#fbbf24",
              background: "rgba(245, 158, 11, 0.12)",
              border: "1px solid rgba(245, 158, 11, 0.3)",
              padding: "4px 10px",
              display: "flex",
              alignItems: "center",
              gap: "4px",
              borderRadius: "8px",
              fontSize: "12px",
              fontWeight: 700,
              cursor: "pointer",
            }}
          >
            <X size={16} />
            <span>Close</span>
          </button>
        </div>

      {/* 4 Smart Style Suggestion Pills (Frosted Glass / Watermark Effect) */}
      <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
        <span style={{ fontSize: "11px", color: "var(--muted)", fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase" }}>
          Instant AI Curated Transformations:
        </span>
        <div className="suggestion-pills-grid">
          {suggestions.map((s, idx) => (
            <button
              key={idx}
              type="button"
              className="suggestion-pill"
              onClick={() => handleSelectSuggestion(s)}
              disabled={busy}
              title={`Apply: "${s.prompt}"`}
            >
              <span style={{ fontSize: "16px" }}>{s.icon}</span>
              <div style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
                <strong style={{ fontSize: "12px", color: "#f8fafc" }}>{s.label}</strong>
                <span style={{ fontSize: "10px", color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {s.prompt}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Custom Prompt Input Bar */}
      <div style={{ display: "flex", gap: "8px", marginTop: "2px" }}>
        <input
          type="text"
          autoFocus
          value={prompt}
          placeholder={`Or type custom prompt for ${selectedObject.toLowerCase()} (e.g. Modern terracotta linen)...`}
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
            background: "rgba(6, 9, 14, 0.75)",
            border: "1px solid rgba(255, 255, 255, 0.12)",
            borderRadius: "10px",
            padding: "10px 14px",
            color: "#f8fafc",
            fontSize: "13px",
            outline: "none",
            transition: "border-color 0.2s",
          }}
        />
        <button
          type="button"
          disabled={busy || !prompt.trim()}
          onClick={() => void onSubmit(prompt.trim(), model)}
          style={{
            background: prompt.trim() && !busy ? "var(--primary)" : "rgba(255, 255, 255, 0.08)",
            color: prompt.trim() && !busy ? "#07090e" : "var(--muted)",
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
            boxShadow: prompt.trim() && !busy ? "0 0 20px var(--primary-glow)" : "none",
            flexShrink: 0,
          }}
        >
          <Wand2 size={15} />
          {busy ? "Kaggle GPU Working…" : "Transform"}
        </button>
      </div>

      {/* Model Selection & Advanced Options */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          background: "rgba(255, 255, 255, 0.02)",
          padding: "6px 10px",
          borderRadius: "8px",
          border: "1px solid rgba(255, 255, 255, 0.05)",
          fontSize: "11px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ color: "var(--muted)", fontWeight: 600 }}>GPU Engine:</span>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value as DiffusionModelVariant)}
            disabled={busy}
            style={{
              background: "#080b10",
              color: "var(--primary)",
              border: "1px solid rgba(255, 255, 255, 0.12)",
              borderRadius: "6px",
              padding: "2px 8px",
              fontSize: "11px",
              fontWeight: 600,
              outline: "none",
              cursor: "pointer",
            }}
          >
            <option value="flux2_klein_4b">FLUX.2 [klein] 4B (Next-Gen Apache-2.0)</option>
            <option value="sdxl_inpaint_t4">SDXL Inpaint (High-Detail T4)</option>
          </select>
        </div>
        <button
          type="button"
          onClick={() => setShowAdvanced((v) => !v)}
          style={{
            color: "var(--muted)",
            fontSize: "10px",
            textDecoration: "underline",
            cursor: "pointer",
          }}
        >
          {showAdvanced ? "Hide specs" : "Hardware specs"}
        </button>
      </div>

      {showAdvanced ? (
        <p style={{ margin: 0, fontSize: "11px", color: "var(--muted)", lineHeight: 1.4 }}>
          💡 {MODEL_INFO[model].desc}
        </p>
      ) : null}

      {/* Status feedback & A/B version switches */}
      {statusMessage ? (
        <div
          style={{
            padding: "8px 12px",
            background: "rgba(16, 185, 129, 0.12)",
            border: "1px solid rgba(16, 185, 129, 0.25)",
            borderRadius: "8px",
            fontSize: "12px",
            color: "#6ee7b7",
            textAlign: "center",
            fontWeight: 500,
          }}
        >
          {statusMessage}
        </div>
      ) : null}

      {versions.length > 0 ? (
        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap", borderTop: "1px solid var(--line)", paddingTop: "8px" }}>
          <span style={{ fontSize: "11px", color: "var(--muted)", fontWeight: 600 }}>Switch 3D Scene Version:</span>
          {versions.map((v) => {
            const active = v.id === activeVersionId;
            return (
              <button
                key={v.id}
                type="button"
                onClick={() => onSelectVersion?.(v.id)}
                style={{
                  background: active ? "rgba(245, 158, 11, 0.2)" : "rgba(255, 255, 255, 0.05)",
                  borderColor: active ? "var(--primary)" : "var(--line)",
                  borderWidth: "1px",
                  borderStyle: "solid",
                  color: active ? "#fbbf24" : "var(--muted)",
                  fontSize: "11px",
                  padding: "4px 10px",
                  borderRadius: "6px",
                  cursor: "pointer",
                  fontWeight: active ? 700 : 500,
                  transition: "all 0.2s",
                }}
              >
                {v.label}
              </button>
            );
          })}
        </div>
      ) : null}
      </div>
    </>
  );
}
