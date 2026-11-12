import { expect, test } from "@playwright/test";

test("Week 2 interactive Edit Studio flow in Chromium", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message));

  await page.route("**/api/projects/demo-w2", async (route) => {
    await route.fulfill({
      json: {
        id: "demo-w2",
        name: "Studio Loft",
        status: "completed",
        active_version_id: "v1",
        created_at: "2026-11-12T00:00:00Z",
        versions: [
          { id: "v1", label: "V1 Original" },
          { id: "v2", label: "V2 Edited" },
        ],
        tasks: [{ id: "t1", type: "reconstruct_week1", status: "completed", progress_pct: 100, stage: "04_export" }],
        edit_sessions: [],
      },
    });
  });

  await page.route("**/api/projects/demo-w2/versions/*/manifest", async (route) => {
    await route.fulfill({
      json: {
        schema_version: "1.0.0",
        project_id: "demo-w2",
        version_id: "v1",
        coord_convention: "opencv_colmap",
        T_viewer_from_colmap: [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        camera_model: "PINHOLE",
        width: 320,
        height: 240,
        intrinsics: { fx: 250, fy: 250, cx: 160, cy: 120 },
        test_every: 0,
        frames: [{ frame_idx: 0, image_relpath: "images/frame_000000.jpg", cam_from_world: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] }],
        artifacts: { sparse_dir: "sparse/0", spz_relpath: "scene.spz", sparse_points_relpath: "sparse_points.json" },
      },
    });
  });

  await page.route("**/api/storage/**", async (route) => {
    if (route.request().url().endsWith(".json")) {
      await route.fulfill({ json: [{ position: [0, 0, 2], color: [200, 180, 160] }] });
      return;
    }
    await route.fulfill({ status: 404, body: "missing" });
  });

  await page.route("**/api/projects/demo-w2/edits/resolve-click", async (route) => {
    await route.fulfill({ status: 201, json: { edit_id: "edit-pw", ref_frame_idx: 0, u: 160, v: 120, score: 3.0, method: "projection", task_id: "t-init" } });
  });
  await page.route("**/api/edits/edit-pw/frames*", async (route) => {
    await route.fulfill({
      json: { edit_id: "edit-pw", page: 1, page_size: 24, total: 1, frames: [{ frame_idx: 0, image_relpath: null, mask_key: null, mask_status: "auto", skip_frame: false }] },
    });
  });
  let maskUpdates = 0;
  await page.route("**/api/edits/edit-pw/masks/0", async (route) => {
    maskUpdates += 1;
    await route.fulfill({
      json: { edit_id: "edit-pw", frame_idx: 0, status: maskUpdates === 1 ? "skipped" : "manual", skip_frame: maskUpdates === 1 },
    });
  });
  await page.route("**/api/edits/edit-pw/preview-seeds", async (route) => {
    await route.fulfill({ json: { id: "t-prev", type: "diffusion_preview", status: "queued", progress_pct: 0 } });
  });
  await page.route("**/api/edits/edit-pw/apply", async (route) => {
    await route.fulfill({ json: { id: "t-reb", type: "rebuild_week2", status: "queued", progress_pct: 0, edit_id: "edit-pw", target_version_id: "v2", diffusion_task_id: "t-diff" } });
  });

  await page.goto("/projects/demo-w2");
  await expect(page.getByText("Studio Loft")).toBeVisible();
  await page.getByRole("button", { name: /Edit Object Mode: OFF/i }).click();
  await page.locator("canvas").first().click({ position: { x: 100, y: 100 } });
  await expect(page.getByText(/Ref Frame #0/i)).toBeVisible();

  await page.getByRole("button", { name: "Skip Frame" }).click();
  await expect.poll(() => maskUpdates).toBe(1);

  await page.getByLabel("Upload PNG mask for frame 0").setInputFiles({
    name: "mask.png",
    mimeType: "image/png",
    buffer: Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
  });
  await expect.poll(() => maskUpdates).toBe(2);

  await page.getByLabel("Edit instruction prompt").fill("red leather armchair");
  await page.getByRole("button", { name: "Generate Seed Previews" }).click();
  await expect(page.getByText(/Seed preview generation queued/i)).toBeVisible();

  await page.getByRole("button", { name: "Apply & Rebuild 3D Room" }).click();
  await expect(page.getByText(/Rebuild queued for version v2/i)).toBeVisible();

  await page.getByRole("button", { name: "V2 Edited" }).click();
  expect(errors).toEqual([]);
});
