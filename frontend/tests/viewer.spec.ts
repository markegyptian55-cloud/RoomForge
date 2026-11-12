import { test, expect } from "@playwright/test";

test("viewer mounts with WebGL and exposes mode controls", async ({ page }) => {
  await page.route("**/api/projects/demo", async (route) =>
    route.fulfill({
      json: {
        id: "demo",
        name: "Demo room",
        status: "completed",
        active_version_id: "v0",
        created_at: "2026-01-01",
        versions: [],
        tasks: [],
        edit_sessions: [],
      },
    }),
  );
  await page.route("**/api/projects/demo/versions/v0/manifest", async (route) =>
    route.fulfill({
      json: {
        T_viewer_from_colmap: [
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1],
        ],
        frames: [],
        artifacts: { sparse_points_relpath: "sparse_points.json" },
      },
    }),
  );
  await page.route("**/sparse_points.json", async (route) =>
    route.fulfill({
      json: [
        { x: 0, y: 0, z: 0, color: [0.4, 0.8, 0.6] },
        { x: 1, y: 0, z: 0 },
      ],
    }),
  );

  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto("/projects/demo");
  await expect(page.getByRole("heading", { name: "Demo room" })).toBeVisible({ timeout: 15000 });
  await expect(page.locator(".viewer3d canvas")).toBeVisible({ timeout: 15000 });
  await expect(page.getByRole("button", { name: "Walk" })).toBeVisible();

  await page.getByRole("button", { name: "Walk" }).click();
  await expect(page.getByRole("button", { name: "Walk" })).toHaveClass(/active/);
  await page.getByRole("button", { name: "Orbit" }).click();
  await expect(page.getByRole("button", { name: "Orbit" })).toHaveClass(/active/);

  expect(errors).toEqual([]);
});
