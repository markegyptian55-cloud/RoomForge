"use client";

import React from "react";
import type { EditFramesResponse } from "@/lib/api";

export interface TrackingGridProps {
  assetBaseUrl: string;
  framesData: EditFramesResponse;
  busy: boolean;
  onPageChange: (page: number) => Promise<void>;
  onToggleSkip: (frameIdx: number, skip: boolean) => Promise<void>;
  onUploadMask: (frameIdx: number, file: File) => Promise<void>;
}

export default function TrackingGrid({
  assetBaseUrl,
  framesData,
  busy,
  onPageChange,
  onToggleSkip,
  onUploadMask,
}: TrackingGridProps) {
  const totalPages = Math.max(1, Math.ceil(framesData.total / Math.max(1, framesData.page_size)));

  return (
    <section aria-label="Tracked Frames Review Grid" style={{ display: "grid", gap: "10px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <strong>Tracked Frames ({framesData.total})</strong>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <button type="button" disabled={busy || framesData.page <= 1} onClick={() => void onPageChange(framesData.page - 1)}>
            Prev
          </button>
          <span>Page {framesData.page} / {totalPages}</span>
          <button type="button" disabled={busy || framesData.page >= totalPages} onClick={() => void onPageChange(framesData.page + 1)}>
            Next
          </button>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))",
          gap: "8px",
          maxHeight: "250px",
          overflowY: "auto",
          padding: "4px",
          border: "1px solid var(--line, #334155)",
          borderRadius: "8px",
        }}
      >
        {framesData.frames.map((item) => {
          const thumbSrc = item.image_relpath ? `${assetBaseUrl}/${item.image_relpath}` : null;
          return (
            <article
              key={item.frame_idx}
              style={{
                border: "1px solid var(--line, #334155)",
                borderRadius: "8px",
                padding: "8px",
                display: "grid",
                gap: "6px",
                background: "var(--card, #0f172a)",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", fontWeight: 600 }}>
                <span>Frame #{item.frame_idx}</span>
                <span style={{ color: item.skip_frame ? "var(--red, #f87171)" : "var(--green, #4ade80)" }}>
                  {item.skip_frame ? "Skipped" : item.mask_status}
                </span>
              </div>
              <div style={{ position: "relative", width: "100%", height: "64px", borderRadius: "4px", overflow: "hidden", background: "#0b1220" }}>
                {thumbSrc ? (
                  <img
                    src={thumbSrc}
                    alt={`Frame ${item.frame_idx}`}
                    style={{ width: "100%", height: "100%", objectFit: "cover" }}
                  />
                ) : null}
                {item.mask_key ? (
                  <img
                    src={`${assetBaseUrl}/${item.mask_key}`}
                    alt={`Mask ${item.frame_idx}`}
                    style={{
                      position: "absolute",
                      inset: 0,
                      width: "100%",
                      height: "100%",
                      objectFit: "cover",
                      mixBlendMode: "screen",
                      opacity: 0.65,
                      filter: "sepia(1) saturate(6) hue-rotate(95deg)",
                    }}
                  />
                ) : null}
              </div>
              <button
                type="button"
                disabled={busy}
                onClick={() => void onToggleSkip(item.frame_idx, !item.skip_frame)}
                style={{ fontSize: "11px", padding: "4px 8px" }}
              >
                {item.skip_frame ? "Include Frame" : "Skip Frame"}
              </button>
              <label style={{ fontSize: "10px", cursor: "pointer", textAlign: "center" }}>
                Upload PNG Mask
                <input
                  type="file"
                  accept="image/png"
                  aria-label={`Upload PNG mask for frame ${item.frame_idx}`}
                  disabled={busy}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) void onUploadMask(item.frame_idx, file);
                    e.target.value = "";
                  }}
                />
              </label>
            </article>
          );
        })}
      </div>
    </section>
  );
}
