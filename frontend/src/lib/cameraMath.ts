import type { ViewerCameraPose } from "./api";

export type Vec3 = [number, number, number];
export type Matrix4RowMajor = [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
];

export interface TrainingFrame {
  frame_idx: number;
  cam_from_world: number[][];
}

export type CameraPathPose = ViewerCameraPose;

export function applyMatrix4RowMajor(matrix: readonly number[], point: readonly number[]): Vec3 {
  if (matrix.length !== 16 || point.length !== 3) {
    throw new RangeError("Expected a 4x4 matrix and 3D point.");
  }
  const w = matrix[12]! * point[0]! + matrix[13]! * point[1]! + matrix[14]! * point[2]! + matrix[15]!;
  const divisor = Math.abs(w) > Number.EPSILON ? w : 1;
  return [
    (matrix[0]! * point[0]! + matrix[1]! * point[1]! + matrix[2]! * point[2]! + matrix[3]!) / divisor,
    (matrix[4]! * point[0]! + matrix[5]! * point[1]! + matrix[6]! * point[2]! + matrix[7]!) / divisor,
    (matrix[8]! * point[0]! + matrix[9]! * point[1]! + matrix[10]! * point[2]! + matrix[11]!) / divisor,
  ];
}

export function applyDirection4RowMajor(matrix: readonly number[], vector: readonly number[]): Vec3 {
  if (matrix.length !== 16 || vector.length !== 3) {
    throw new RangeError("Expected a 4x4 matrix and 3D direction.");
  }
  return [
    matrix[0]! * vector[0]! + matrix[1]! * vector[1]! + matrix[2]! * vector[2]!,
    matrix[4]! * vector[0]! + matrix[5]! * vector[1]! + matrix[6]! * vector[2]!,
    matrix[8]! * vector[0]! + matrix[9]! * vector[1]! + matrix[10]! * vector[2]!,
  ];
}

export function invertRigid4x4(matrix: readonly number[]): Matrix4RowMajor {
  if (matrix.length !== 16) {
    throw new RangeError("Expected a 4x4 matrix.");
  }
  const result = Array.from({ length: 16 }, () => 0);
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      result[row * 4 + col] = matrix[col * 4 + row]!;
    }
  }
  for (let row = 0; row < 3; row += 1) {
    result[row * 4 + 3] = -(
      result[row * 4]! * matrix[3]!
      + result[row * 4 + 1]! * matrix[7]!
      + result[row * 4 + 2]! * matrix[11]!
    );
  }
  result[15] = 1;
  return result as Matrix4RowMajor;
}

function frameCenter(frame: TrainingFrame): Vec3 {
  const pose = frame.cam_from_world;
  const flat: number[] = [
    pose[0]?.[0] ?? 1, pose[0]?.[1] ?? 0, pose[0]?.[2] ?? 0, pose[0]?.[3] ?? 0,
    pose[1]?.[0] ?? 0, pose[1]?.[1] ?? 1, pose[1]?.[2] ?? 0, pose[1]?.[3] ?? 0,
    pose[2]?.[0] ?? 0, pose[2]?.[1] ?? 0, pose[2]?.[2] ?? 1, pose[2]?.[3] ?? 0,
    0, 0, 0, 1,
  ];
  return applyMatrix4RowMajor(invertRigid4x4(flat), [0, 0, 0]);
}

function dot(a: readonly number[], b: readonly number[]): number {
  return a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
}

function normalize(vector: readonly number[]): Vec3 {
  const length = Math.hypot(vector[0] ?? 0, vector[1] ?? 0, vector[2] ?? 0);
  if (length <= Number.EPSILON) {
    return [0, 0, -1];
  }
  return [vector[0]! / length, vector[1]! / length, vector[2]! / length];
}

export function findNearestTrainingFrame(
  clickPointColmap: Vec3,
  rayOriginColmap: Vec3,
  rayDirColmap: Vec3,
  frames: readonly TrainingFrame[],
): number | null {
  if (!frames.length) return null;
  const direction = normalize(rayDirColmap);
  let bestFrame = frames[0]!.frame_idx;
  let bestScore = -Infinity;

  for (const frame of frames) {
    const center = frameCenter(frame);
    const toCenter: Vec3 = [
      center[0] - rayOriginColmap[0],
      center[1] - rayOriginColmap[1],
      center[2] - rayOriginColmap[2],
    ];
    const distance = Math.hypot(
      clickPointColmap[0] - center[0],
      clickPointColmap[1] - center[1],
      clickPointColmap[2] - center[2],
    );
    const alignment = Math.max(0, dot(normalize(toCenter), direction));
    const score = alignment * 4 - distance;
    if (score > bestScore) {
      bestScore = score;
      bestFrame = frame.frame_idx;
    }
  }
  return bestFrame;
}

function lerp(a: Vec3, b: Vec3, t: number): Vec3 {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function slerp(a: Vec3, b: Vec3, t: number): Vec3 {
  const first = normalize(a);
  const second = normalize(b);
  const cosine = Math.max(-1, Math.min(1, dot(first, second)));
  if (Math.abs(cosine) > 0.9995) {
    return normalize(lerp(first, second, t));
  }
  const angle = Math.acos(cosine);
  const sinAngle = Math.sin(angle);
  const w0 = Math.sin((1 - t) * angle) / sinAngle;
  const w1 = Math.sin(t * angle) / sinAngle;
  return normalize([
    w0 * first[0] + w1 * second[0],
    w0 * first[1] + w1 * second[1],
    w0 * first[2] + w1 * second[2],
  ]);
}

export function interpolateCameraPath(poses: readonly CameraPathPose[], t: number): ViewerCameraPose {
  if (!poses.length) {
    throw new RangeError("Camera path requires at least one pose.");
  }
  if (poses.length === 1) {
    return { ...poses[0]! };
  }
  const scaled = Math.max(0, Math.min(1, t)) * (poses.length - 1);
  const index = Math.min(poses.length - 2, Math.floor(scaled));
  const local = scaled - index;
  const current = poses[index]!;
  const next = poses[index + 1]!;
  const currentFov = current.fovDeg ?? 60;
  const nextFov = next.fovDeg ?? 60;
  return {
    positionColmap: lerp(current.positionColmap, next.positionColmap, local),
    targetColmap: lerp(current.targetColmap, next.targetColmap, local),
    upColmap: slerp(current.upColmap, next.upColmap, local),
    fovDeg: currentFov + (nextFov - currentFov) * local,
  };
}
