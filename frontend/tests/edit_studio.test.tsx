import React from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import EditStudio from "@/components/EditStudio";
import { roomApi, type ProjectDetail, type ViewerClickEvent, type ViewerController } from "@/lib/api";

describe("EditStudio interactive Week 2 workflow", () => {
  let clickCallback: ((ev: ViewerClickEvent) => void) | null = null;

  const project: ProjectDetail = {
    id: "proj-w2",
    name: "Living Room",
    status: "completed",
    active_version_id: "v1",
    created_at: "2026-11-12T00:00:00Z",
    versions: [
      { id: "v1", label: "V1 Base" },
      { id: "v2", label: "V2 Velvet Chair" },
    ],
    tasks: [],
    edit_sessions: [],
  };

  const controller: ViewerController = {
    onUserClick: (handler) => {
      clickCallback = handler;
      return () => {
        clickCallback = null;
      };
    },
    setCameraPose: vi.fn(),
    loadCameraPath: vi.fn(),
    setActiveVersion: vi.fn(),
  };

  beforeEach(() => {
    clickCallback = null;
    vi.restoreAllMocks();
  });

  it("drives click resolve -> refine -> track -> skip/mask upload -> seed preview -> apply -> version switch", async () => {
    vi.spyOn(roomApi, "resolveClick").mockResolvedValue({
      edit_id: "edit-1234",
      ref_frame_idx: 2,
      u: 120,
      v: 90,
      score: 2.5,
      method: "projection",
      task_id: "task-init",
    });
    const refineSpy = vi.spyOn(roomApi, "refine").mockResolvedValue({ id: "t-ref", type: "segment_refine", status: "queued", progress_pct: 0 });
    const trackSpy = vi.spyOn(roomApi, "track").mockResolvedValue({ id: "t-trk", type: "track_propagate", status: "queued", progress_pct: 0 });
    vi.spyOn(roomApi, "editFrames").mockResolvedValue({
      edit_id: "edit-1234",
      page: 1,
      page_size: 24,
      total: 2,
      frames: [
        { frame_idx: 0, image_relpath: "images/frame_000000.jpg", mask_key: null, mask_status: "auto", skip_frame: false },
        { frame_idx: 2, image_relpath: "images/frame_000002.jpg", mask_key: null, mask_status: "auto", skip_frame: false },
      ],
    });
    const updateMaskSpy = vi.spyOn(roomApi, "updateMask").mockResolvedValue({
      edit_id: "edit-1234",
      frame_idx: 0,
      status: "skipped",
      skip_frame: true,
    });
    const previewSpy = vi.spyOn(roomApi, "previewSeeds").mockResolvedValue({ id: "t-prev", type: "diffusion_preview", status: "queued", progress_pct: 0 });
    const applySpy = vi.spyOn(roomApi, "applyEdit").mockResolvedValue({
      id: "t-reb",
      type: "rebuild_week2",
      status: "queued",
      progress_pct: 0,
      edit_id: "edit-1234",
      target_version_id: "v2",
      diffusion_task_id: "t-diff",
    });

    const onSelectVersion = vi.fn();
    const onRefreshProject = vi.fn().mockResolvedValue(undefined);

    render(
      <EditStudio
        projectId="proj-w2"
        project={project}
        manifest={{ width: 640, height: 480, frames: [{ frame_idx: 2, image_relpath: "images/frame_000002.jpg" }] }}
        controller={controller}
        assetBaseUrl="/api/storage/proj-w2/v1"
        activeVersionId="v1"
        onSelectVersion={onSelectVersion}
        onRefreshProject={onRefreshProject}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Edit Object Mode: OFF/i }));
    expect(clickCallback).not.toBeNull();
    await act(async () => {
      clickCallback?.({
        screenX: 320,
        screenY: 240,
        worldHitColmap: [0, 0, 3],
        rayOriginColmap: [0, 0, 0],
        rayDirColmap: [0, 0, 1],
        nearestFrameIdx: 2,
      });
    });

    await waitFor(() => expect(screen.getByText(/Ref Frame #2/i)).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "- Background" }));
    fireEvent.keyDown(screen.getByTestId("ref-frame-canvas"), { key: "Enter" });
    await waitFor(() => expect(refineSpy).toHaveBeenCalledWith("edit-1234", [{ x: 320, y: 240, label: 0 }]));

    fireEvent.click(screen.getByRole("button", { name: /Confirm Mask & Track Across Frames/i }));
    await waitFor(() => expect(trackSpy).toHaveBeenCalledWith("edit-1234"));

    const skipButtons = screen.getAllByRole("button", { name: "Skip Frame" });
    expect(skipButtons[0]).toBeDefined();
    fireEvent.click(skipButtons[0] as HTMLElement);
    await waitFor(() => expect(updateMaskSpy).toHaveBeenCalledWith("edit-1234", 0, { skip_frame: true }));

    const fileInput = screen.getByLabelText("Upload PNG mask for frame 2");
    const pngFile = new File([new Uint8Array([137, 80, 78, 71])], "mask.png", { type: "image/png" });
    fireEvent.change(fileInput, { target: { files: [pngFile] } });
    await waitFor(() => expect(updateMaskSpy).toHaveBeenCalledTimes(2));

    fireEvent.change(screen.getByLabelText("Edit instruction prompt"), { target: { value: "emerald armchair" } });
    fireEvent.click(screen.getByRole("button", { name: "Generate Seed Previews" }));
    await waitFor(() => expect(previewSpy).toHaveBeenCalledWith("edit-1234", "emerald armchair", [101, 202, 303], "flux2_klein_4b"));

    fireEvent.click(screen.getByRole("button", { name: "Apply & Rebuild 3D Room" }));
    await waitFor(() => expect(applySpy).toHaveBeenCalledWith("edit-1234", "seed_101", "masked_finetune", "emerald armchair"));
    await waitFor(() => expect(screen.getByText(/Rebuild queued for version v2/i)).toBeInTheDocument());
    expect(onRefreshProject).toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "V2 Velvet Chair" }));
    expect(onSelectVersion).toHaveBeenCalledWith("v2");
    expect(controller.setActiveVersion).toHaveBeenCalledWith("v2");
  });
});
