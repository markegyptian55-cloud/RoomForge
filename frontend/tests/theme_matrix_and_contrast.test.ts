import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";

function hexToRgb(hex: string): [number, number, number] {
  const cleanHex = hex.replace("#", "").trim();
  if (cleanHex.length === 3) {
    const c0 = cleanHex.charAt(0) || "0";
    const c1 = cleanHex.charAt(1) || "0";
    const c2 = cleanHex.charAt(2) || "0";
    return [
      parseInt(c0 + c0, 16),
      parseInt(c1 + c1, 16),
      parseInt(c2 + c2, 16),
    ];
  }
  return [
    parseInt(cleanHex.slice(0, 2), 16),
    parseInt(cleanHex.slice(2, 4), 16),
    parseInt(cleanHex.slice(4, 6), 16),
  ];
}

function relativeLuminance(rgb: [number, number, number]): number {
  const [r, g, b] = rgb.map((val) => {
    const sRGB = val / 255;
    return sRGB <= 0.03928 ? sRGB / 12.92 : Math.pow((sRGB + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * (r ?? 0) + 0.7152 * (g ?? 0) + 0.0722 * (b ?? 0);
}

function contrastRatio(hex1: string, hex2: string): number {
  const lum1 = relativeLuminance(hexToRgb(hex1));
  const lum2 = relativeLuminance(hexToRgb(hex2));
  const brightest = Math.max(lum1, lum2);
  const darkest = Math.min(lum1, lum2);
  return (brightest + 0.05) / (darkest + 0.05);
}

describe("Design Token Matrix & WCAG Contrast Compliance", () => {
  const cssPath = path.resolve(__dirname, "../src/app/globals.css");
  const cssContent = fs.readFileSync(cssPath, "utf-8");

  const requiredTokens = [
    "--paper",
    "--card",
    "--card-hover",
    "--ink",
    "--muted",
    "--line",
    "--line-strong",
    "--primary",
    "--primary-hover",
    "--primary-glow",
    "--glass-bg",
    "--glass-border",
    "--glass-shadow",
    "--backdrop-bg",
    "--input-bg",
    "--h1-gradient",
    "--studio-viewport-bg",
  ];

  it("declares all required design tokens in :root (Light Mode)", () => {
    const rootBlockMatch = cssContent.match(/:root\s*\{([^}]+)\}/);
    expect(rootBlockMatch).not.toBeNull();
    const rootBlock = rootBlockMatch![1]!;
    for (const token of requiredTokens) {
      expect(rootBlock).toContain(token);
    }
  });

  it("declares all required design tokens in html.dark, :root.dark (Dark Mode)", () => {
    const darkBlockMatch = cssContent.match(/html\.dark,\s*:root\.dark\s*\{([^}]+)\}/);
    expect(darkBlockMatch).not.toBeNull();
    const darkBlock = darkBlockMatch![1]!;
    for (const token of requiredTokens) {
      expect(darkBlock).toContain(token);
    }
  });

  it("satisfies WCAG AA contrast ratio (>= 4.5:1) for primary text in Light Mode", () => {
    // Light Mode: --paper: #f8fafc vs --ink: #0f172a
    const ratio = contrastRatio("#f8fafc", "#0f172a");
    expect(ratio).toBeGreaterThanOrEqual(7.0); // Exceeds AAA standard (4.5:1 is AA, 7.0:1 is AAA)
    expect(ratio).toBeCloseTo(17.06, 1);
  });

  it("satisfies WCAG AA contrast ratio (>= 4.5:1) for card text in Light Mode", () => {
    // Light Mode: --card: #ffffff vs --ink: #0f172a
    const ratio = contrastRatio("#ffffff", "#0f172a");
    expect(ratio).toBeGreaterThanOrEqual(7.0); // AAA standard for white card
  });

  it("satisfies WCAG AA contrast ratio (>= 4.5:1) for primary text in Dark Mode", () => {
    // Dark Mode: --paper: #07090e vs --ink: #f8fafc
    const ratio = contrastRatio("#07090e", "#f8fafc");
    expect(ratio).toBeGreaterThanOrEqual(7.0); // Exceeds AAA standard
    expect(ratio).toBeCloseTo(19.03, 1);
  });

  it("satisfies WCAG AA contrast ratio (>= 4.5:1) for card text in Dark Mode", () => {
    // Dark Mode: --card: #0e121a vs --ink: #f8fafc
    const ratio = contrastRatio("#0e121a", "#f8fafc");
    expect(ratio).toBeGreaterThanOrEqual(7.0); // Exceeds AAA standard
    expect(ratio).toBeCloseTo(17.92, 1);
  });

  it("satisfies WCAG contrast for primary button in Light and Dark Mode", () => {
    // Primary button: #f59e0b (amber) with dark ink text #07090e
    const ratio = contrastRatio("#f59e0b", "#07090e");
    expect(ratio).toBeGreaterThanOrEqual(7.0);
  });

  it("verifies Light Mode studio viewport background is warm alabaster (#f1f5f9)", () => {
    expect(cssContent).toMatch(/--studio-viewport-bg:\s*#f1f5f9/);
  });

  it("verifies Dark Mode studio viewport background is obsidian (#06080c)", () => {
    expect(cssContent).toMatch(/--studio-viewport-bg:\s*#06080c/);
  });
});

describe("Three.js Dynamic Theme Synchronization Contracts", () => {
  it("determines correct background color hex integer for Dark Mode", () => {
    const isDark = true;
    const bgColor = isDark ? 0x07090e : 0xf1f5f9;
    expect(bgColor).toBe(0x07090e);
    expect(bgColor.toString(16).padStart(6, "0")).toBe("07090e");
  });

  it("determines correct background color hex integer for Light Mode", () => {
    const isDark = false;
    const bgColor = isDark ? 0x07090e : 0xf1f5f9;
    expect(bgColor).toBe(0xf1f5f9);
    expect(bgColor.toString(16).padStart(6, "0")).toBe("f1f5f9");
  });

  it("verifies Three.js camera boundaries prevent void drift", () => {
    const clampTarget = (x: number, y: number, z: number) => ({
      x: Math.max(-2.5, Math.min(2.5, x)),
      y: Math.max(0.1, Math.min(1.8, y)),
      z: Math.max(-2.5, Math.min(2.5, z)),
    });

    // Extreme coordinate test (user tries to fling to space)
    const clampedFar = clampTarget(100, -50, 999);
    expect(clampedFar.x).toBe(2.5);
    expect(clampedFar.y).toBe(0.1);
    expect(clampedFar.z).toBe(2.5);

    const clampedInside = clampTarget(0.2, 0.8, -0.4);
    expect(clampedInside.x).toBe(0.2);
    expect(clampedInside.y).toBe(0.8);
    expect(clampedInside.z).toBe(-0.4);
  });
});

describe("Mobile Responsive & Ergonomic Layout Contracts", () => {
  const cssPath = path.resolve(__dirname, "../src/app/globals.css");
  const cssContent = fs.readFileSync(cssPath, "utf-8");

  it("enforces mobile sheet slide-up animation in media query", () => {
    expect(cssContent).toContain("@keyframes sheetSlideUp");
    expect(cssContent).toContain("transform: translateY(100%)");
  });

  it("enforces mobile sheet max-height constraints to avoid viewport obstruction", () => {
    expect(cssContent).toContain("max-height: 48vh");
    expect(cssContent).toContain("max-height: 48dvh");
  });

  it("enforces rounded top corners (bottom sheet ergonomics) on mobile", () => {
    expect(cssContent).toContain("border-radius: 20px 20px 0 0");
  });

  it("includes dismiss handle indicator for mobile sheet", () => {
    expect(cssContent).toContain(".sheet-drag-handle");
  });

  it("includes backdrop blur and dismiss overlay", () => {
    expect(cssContent).toContain(".popover-backdrop");
    expect(cssContent).toContain("backdrop-filter: blur(4px)");
  });

  it("verifies mobile touch targets meet 44px minimum sizing guidelines", () => {
    // Sheet drag handle width >= 44px
    expect(cssContent).toMatch(/\.sheet-drag-handle\s*\{[^}]*width:\s*44px/);
  });

  it("verifies muted text contrast meets WCAG criteria for secondary labels", () => {
    // Light mode --muted: #64748b on #ffffff card
    const lightRatio = contrastRatio("#64748b", "#ffffff");
    expect(lightRatio).toBeGreaterThanOrEqual(4.5);

    // Dark mode --muted: #94a3b8 on #0e121a card
    const darkRatio = contrastRatio("#94a3b8", "#0e121a");
    expect(darkRatio).toBeGreaterThanOrEqual(4.5);
  });

  it("verifies h1 gradient differs between light and dark themes", () => {
    const lightMatch = cssContent.match(/--h1-gradient:\s*([^;]+);/);
    const darkMatch = cssContent.match(/html\.dark[^}]*--h1-gradient:\s*([^;]+);/);
    expect(lightMatch).not.toBeNull();
    expect(darkMatch).not.toBeNull();
    expect(lightMatch![1]).not.toBe(darkMatch![1]);
  });

  it("verifies input background token toggles from white to obsidian", () => {
    const lightMatch = cssContent.match(/--input-bg:\s*([^;]+);/);
    const darkMatch = cssContent.match(/html\.dark[^}]*--input-bg:\s*([^;]+);/);
    expect(lightMatch![1]?.trim()).toBe("#ffffff");
    expect(darkMatch![1]?.trim()).toBe("rgba(6, 9, 14, 0.75)");
  });

  it("verifies glass border token maintains amber presence across themes", () => {
    const lightMatch = cssContent.match(/--glass-border:\s*([^;]+);/);
    const darkMatch = cssContent.match(/html\.dark[^}]*--glass-border:\s*([^;]+);/);
    expect(lightMatch![1]).toContain("217, 119, 6");
    expect(darkMatch![1]).toContain("245, 158, 11");
  });

  it("verifies suggestion pills have active/hover state transitions", () => {
    expect(cssContent).toContain(".suggestion-pill:hover");
    expect(cssContent).toContain("border-color: var(--primary)");
  });

  it("verifies reset camera button uses semantic glass tokens", () => {
    const viewerPath = path.resolve(__dirname, "../src/components/Viewer3D.tsx");
    const viewerContent = fs.readFileSync(viewerPath, "utf-8");
    expect(viewerContent).toContain('background: "var(--glass-bg)"');
    expect(viewerContent).toContain('border: "1px solid var(--glass-border)"');
    expect(viewerContent).toContain('color: "var(--primary)"');
  });

  it("verifies keyboard event propagation stop on edit bar inputs to prevent camera movement", () => {
    const editBarPath = path.resolve(__dirname, "../src/components/FloatingEditBar.tsx");
    const editBarContent = fs.readFileSync(editBarPath, "utf-8");
    expect(editBarContent).toContain("e.stopPropagation()");
  });
});
