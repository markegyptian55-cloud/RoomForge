import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { defineConfig, devices } from "@playwright/test";

function resolveCachedChromium(): string | undefined {
  const localAppData = process.env.LOCALAPPDATA;
  if (!localAppData) return undefined;
  const cacheRoot = join(localAppData, "ms-playwright");
  if (!existsSync(cacheRoot)) return undefined;
  const chromiumDirs = readdirSync(cacheRoot)
    .filter((entry) => entry.startsWith("chromium-"))
    .sort()
    .reverse();
  for (const dir of chromiumDirs) {
    const candidate = join(cacheRoot, dir, "chrome-win64", "chrome.exe");
    if (existsSync(candidate)) return candidate;
  }
  return undefined;
}

const executablePath = resolveCachedChromium();

export default defineConfig({
  testDir: "./tests",
  testMatch: "**/*.spec.ts",
  webServer: {
    command: "npx next dev -H 127.0.0.1 -p 3100",
    url: "http://127.0.0.1:3100",
    reuseExistingServer: true,
    timeout: 120000,
  },
  use: {
    baseURL: "http://127.0.0.1:3100",
    ...devices["Desktop Chrome"],
    headless: true,
    launchOptions: {
      ...(executablePath ? { executablePath } : {}),
      args: [
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-webgl",
        "--enable-unsafe-swiftshader",
        "--ignore-gpu-blocklist",
      ],
    },
  },
});
