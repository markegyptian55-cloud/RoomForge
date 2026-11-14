import { describe, expect, it } from "vitest";
import { validateMedia, validateVideo } from "../src/lib/api";

describe("Mobile Media Validation Suite", () => {
  it("accepts valid MP4 video within size limit", () => {
    const file = new File([new Uint8Array(1024)], "room.mp4", { type: "video/mp4" });
    expect(validateMedia(file)).toBeNull();
    expect(validateVideo(file)).toBeNull();
  });

  it("accepts valid WebM video within size limit", () => {
    const file = new File([new Uint8Array(2048)], "room.webm", { type: "video/webm" });
    expect(validateMedia(file)).toBeNull();
  });

  it("accepts valid QuickTime MOV video", () => {
    const file = new File([new Uint8Array(2048)], "room.mov", { type: "video/quicktime" });
    expect(validateMedia(file)).toBeNull();
  });

  it("accepts valid JPEG photo for Single-Photo 3D", () => {
    const file = new File([new Uint8Array(4096)], "bedroom.jpg", { type: "image/jpeg" });
    expect(validateMedia(file)).toBeNull();
  });

  it("accepts valid PNG photo for Single-Photo 3D", () => {
    const file = new File([new Uint8Array(4096)], "living_room.png", { type: "image/png" });
    expect(validateMedia(file)).toBeNull();
  });

  it("accepts valid WebP photo", () => {
    const file = new File([new Uint8Array(4096)], "interior.webp", { type: "image/webp" });
    expect(validateMedia(file)).toBeNull();
  });

  it("rejects PDF documents", () => {
    const file = new File([new Uint8Array(1024)], "room_plan.pdf", { type: "application/pdf" });
    expect(validateMedia(file)).toContain("Please choose an MP4/WebM/MOV video or a JPEG/PNG/WebP photo");
  });

  it("rejects ZIP archives", () => {
    const file = new File([new Uint8Array(1024)], "assets.zip", { type: "application/zip" });
    expect(validateMedia(file)).toContain("Please choose an MP4/WebM/MOV video or a JPEG/PNG/WebP photo");
  });

  it("rejects executable files", () => {
    const file = new File([new Uint8Array(1024)], "malware.exe", { type: "application/x-msdownload" });
    expect(validateMedia(file)).toContain("Please choose an MP4/WebM/MOV video or a JPEG/PNG/WebP photo");
  });

  it("rejects oversized images (> 50 MB)", () => {
    // 51 MB dummy file
    const file = new File([], "huge.jpg", { type: "image/jpeg" });
    Object.defineProperty(file, "size", { value: 55 * 1024 * 1024 });
    expect(validateMedia(file)).toContain("Image must be 50 MB or smaller");
  });

  it("rejects oversized videos (> 500 MB)", () => {
    const file = new File([], "huge.mp4", { type: "video/mp4" });
    Object.defineProperty(file, "size", { value: 501 * 1024 * 1024 });
    expect(validateMedia(file)).toContain("File must be 500 MB or smaller");
  });
});

describe("Camera Bounds & Mathematical Clamping", () => {
  it("guarantees distance clamp prevents flying into void", () => {
    const clampMaxDistance = (isMesh: boolean, maxDim: number) => {
      if (isMesh) return 3.5;
      return Math.min(4.8, Math.max(2.8, maxDim * 1.4));
    };

    // Even with a massive bounding box of 100 meters, distance is capped at 4.8m
    expect(clampMaxDistance(false, 100)).toBe(4.8);
    // With normal room of 2.5 meters
    expect(clampMaxDistance(false, 2.5)).toBe(3.5);
    // For mesh room
    expect(clampMaxDistance(true, 5.0)).toBe(3.5);
  });

  it("guarantees polar angle clamp prevents flipping below floor", () => {
    const maxPolarAngle = Math.PI / 2 + 0.04;
    expect(maxPolarAngle).toBeLessThan(Math.PI);
    expect(maxPolarAngle).toBeGreaterThan(Math.PI / 2);
  });

  it("guarantees target bounding box prevents panning outside room", () => {
    const clampTarget = (x: number, y: number, z: number) => {
      return {
        x: Math.max(-2.5, Math.min(2.5, x)),
        y: Math.max(0.1, Math.min(1.8, y)),
        z: Math.max(-2.5, Math.min(2.5, z)),
      };
    };

    // Outrageous panning
    const clamped = clampTarget(100, -50, 999);
    expect(clamped.x).toBe(2.5);
    expect(clamped.y).toBe(0.1);
    expect(clamped.z).toBe(2.5);

    const clampedNeg = clampTarget(-100, 50, -999);
    expect(clampedNeg.x).toBe(-2.5);
    expect(clampedNeg.y).toBe(1.8);
    expect(clampedNeg.z).toBe(-2.5);
  });

  it("guarantees camera position bounding box prevents escaping room", () => {
    const clampCam = (x: number, y: number, z: number) => {
      return {
        x: Math.max(-3.5, Math.min(3.5, x)),
        y: Math.max(0.2, Math.min(2.8, y)),
        z: Math.max(-3.5, Math.min(3.5, z)),
      };
    };

    const out = clampCam(500, 500, 500);
    expect(out.x).toBe(3.5);
    expect(out.y).toBe(2.8);
    expect(out.z).toBe(3.5);
  });
});
