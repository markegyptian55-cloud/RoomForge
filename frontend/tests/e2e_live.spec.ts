import { test, expect } from "@playwright/test";

const liveProjectPath = process.env.ROOM3D_LIVE_PROJECT_PATH;

test.describe("Live Week 1 Backend <-> Frontend E2E", () => {
  test.skip(!liveProjectPath, "Requires ROOM3D_LIVE_PROJECT_PATH from S1.8 live runner.");

  test("renders real worker-exported room over Next -> FastAPI proxy without mocks", async ({ page }) => {
    const pageErrors: string[] = [];
    const httpFailures: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    page.on("response", (response) => {
      if (response.status() >= 400 && !response.url().includes("favicon")) {
        httpFailures.push(`${response.status()} ${response.url()}`);
      }
    });

    await page.goto(liveProjectPath!);
    await expect(page.getByRole("heading", { name: "E2E Live Room" })).toBeVisible({ timeout: 20000 });
    await expect(page.getByText("100%")).toBeVisible({ timeout: 15000 });
    await expect(page.locator(".viewer3d canvas")).toBeVisible({ timeout: 15000 });

    const hud = page.locator(".viewer-hud");
    await expect(hud).toContainText(/[1-9]\d*\s+(points|splats)/, { timeout: 15000 });

    await page.getByRole("button", { name: "Walk" }).click();
    await expect(page.getByRole("button", { name: "Walk" })).toHaveClass(/active/);
    await page.getByRole("button", { name: "Orbit" }).click();
    await page.locator(".viewer3d canvas").click({ position: { x: 180, y: 180 } });
    await page.getByRole("button", { name: "Reset Camera" }).click();

    expect(httpFailures).toEqual([]);
    expect(pageErrors).toEqual([]);
  });
});
