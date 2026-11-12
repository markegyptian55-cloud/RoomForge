import { describe, expect, it } from "vitest";
import {
  applyDirection4RowMajor,
  applyMatrix4RowMajor,
  findNearestTrainingFrame,
  interpolateCameraPath,
  invertRigid4x4,
} from "@/lib/cameraMath";

describe("cameraMath", () => {
  it("round trips points between colmap and viewer coordinates", () => {
    const transform = [0, -1, 0, 2, 1, 0, 0, 3, 0, 0, 1, 4, 0, 0, 0, 1];
    const point = [1, 2, 3];
    expect(applyMatrix4RowMajor(invertRigid4x4(transform), applyMatrix4RowMajor(transform, point))).toEqual(point);
  });

  it("rotates direction vectors without adding translation", () => {
    const transform = [0, -1, 0, 25, 1, 0, 0, -40, 0, 0, 1, 12, 0, 0, 0, 1];
    expect(applyDirection4RowMajor(transform, [0, 1, 0])).toEqual([-1, 0, 0]);
  });

  it("ranks the training frame aligned with the click ray", () => {
    const frames = [
      { frame_idx: 2, cam_from_world: [[1, 0, 0, -5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] },
      { frame_idx: 8, cam_from_world: [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]] },
    ];
    expect(findNearestTrainingFrame([1, 0, 0], [0, 0, 0], [1, 0, 0], frames)).toBe(8);
  });

  it("interpolates camera position and orientation", () => {
    const result = interpolateCameraPath(
      [
        { positionColmap: [0, 0, 0], targetColmap: [0, 0, -1], upColmap: [0, 1, 0] },
        { positionColmap: [10, 0, 0], targetColmap: [0, 0, -2], upColmap: [0, 1, 0], fovDeg: 80 },
      ],
      0.5,
    );
    expect(result.positionColmap[0]).toBe(5);
    expect(result.targetColmap[2]).toBe(-1.5);
    expect(result.fovDeg).toBe(70);
  });
});
