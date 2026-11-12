"use client";

import { AlertTriangle, CheckCircle2, Clock3, RotateCcw } from "lucide-react";
import type { StageId, TaskStatus, Telemetry } from "@/lib/api";

const WEEK1_STAGES: Array<{ id: StageId; label: string; threshold: number }> = [
  { id: "01_extract", label: "Extract keyframes", threshold: 15 },
  { id: "02_poses", label: "Align camera poses", threshold: 45 },
  { id: "03_train", label: "Train Gaussian scene", threshold: 85 },
  { id: "04_export", label: "Export viewer bundle", threshold: 100 },
];

const WEEK2_STAGES: Array<{ id: StageId; label: string; threshold: number }> = [
  { id: "05_track", label: "Track object masks", threshold: 25 },
  { id: "06_diffusion", label: "Apply object edit", threshold: 60 },
  { id: "07_rebuild", label: "Rebuild edited scene", threshold: 100 },
];

export interface ProgressPanelProps {
  progress: number;
  stage?: StageId;
  status: TaskStatus | string;
  message?: string | null;
  telemetry?: Telemetry | null;
  onRetry?: () => void;
}

const formatRuntime = (seconds?: number): string =>
  seconds === undefined ? "—" : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;

export function ProgressPanel({ progress, stage, status, message, telemetry, onRetry }: ProgressPanelProps) {
  const failed = status === "failed";
  const done = status === "completed";
  const isWeek2 = stage?.startsWith("05_") || stage?.startsWith("06_") || stage?.startsWith("07_");
  const activeStages = isWeek2 ? WEEK2_STAGES : WEEK1_STAGES;

  return (
    <section className="progress-panel" aria-label="Scene processing progress">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">{isWeek2 ? "OBJECT EDIT" : "ROOM BUILD"}</p>
          <h2>{failed ? "Build needs attention" : done ? "Room ready" : "Building your room"}</h2>
        </div>
        <strong>{Math.round(progress)}%</strong>
      </div>
      <div className="progress-track">
        <span style={{ width: `${Math.min(100, Math.max(0, progress))}%` }} />
      </div>
      <div className="stage-list">
        {activeStages.map((item, index) => {
          const active = item.id === stage;
          const complete = progress >= item.threshold || (done && index <= activeStages.length - 1);
          return (
            <div className={`stage ${active ? "active" : ""} ${complete ? "complete" : ""}`} key={item.id}>
              <span className="stage-number">
                {complete ? <CheckCircle2 size={15} /> : String(index + 1).padStart(2, "0")}
              </span>
              <span>{item.label}</span>
              {active && <span className="stage-live">LIVE</span>}
            </div>
          );
        })}
      </div>
      <div className={`status-message ${failed ? "error" : ""}`}>
        {failed ? <AlertTriangle size={17} /> : <Clock3 size={17} />}
        <span>{message || (done ? "Your scene is ready to explore." : "Processing is running in the background.")}</span>
      </div>
      <div className="telemetry">
        <h3>Runtime telemetry</h3>
        <table>
          <tbody>
            <tr>
              <th>VRAM</th>
              <td>
                {telemetry?.vram_used_mb !== undefined
                  ? `${telemetry.vram_used_mb} / ${telemetry.vram_total_mb ?? "—"} MB`
                  : "—"}
              </td>
            </tr>
            <tr>
              <th>Elapsed</th>
              <td>{formatRuntime(telemetry?.runtime_seconds)}</td>
            </tr>
            <tr>
              <th>Worker status</th>
              <td>{status}</td>
            </tr>
          </tbody>
        </table>
      </div>
      {failed && onRetry && (
        <button className="button secondary retry" onClick={onRetry}>
          <RotateCcw size={16} /> Try again
        </button>
      )}
    </section>
  );
}
