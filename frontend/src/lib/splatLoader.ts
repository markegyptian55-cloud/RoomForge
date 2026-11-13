import * as THREE from "three";

export interface SplatManifestArtifacts {
  spz_relpath?: string;
  sog_relpath?: string | null;
  sparse_points_relpath?: string;
  collision_glb_relpath?: string | null;
}

export interface SplatManifest {
  artifacts?: SplatManifestArtifacts;
}

export interface SplatLoadOptions {
  baseUrl?: string;
  showCollision?: boolean;
}

export interface SplatLoadResult {
  mode: "spark" | "points" | "empty";
  object: THREE.Object3D | null;
  sparkRenderer: THREE.Object3D | null;
  count: number;
  collision: THREE.Object3D | null;
}

type PointRecord = {
  x?: number;
  y?: number;
  z?: number;
  position?: readonly number[];
  color?: [number, number, number];
};

const joinUrl = (base: string, path: string): string => {
  if (/^https?:\/\//i.test(path) || path.startsWith("/") || !base) {
    return path;
  }
  return `${base.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
};

const supportsSpark = (renderer: THREE.WebGLRenderer): boolean =>
  Boolean(renderer.capabilities.isWebGL2 && renderer.extensions.has("EXT_color_buffer_float"));

async function loadPoints(url: string, group: THREE.Group): Promise<number> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Sparse point cloud request failed (${response.status}).`);
  }
  const data = (await response.json()) as PointRecord[] | { points: PointRecord[] };
  const points = Array.isArray(data) ? data : data.points;
  if (!Array.isArray(points)) {
    throw new TypeError("Sparse point cloud must be an array or { points }.");
  }

  const positions = new Float32Array(points.length * 3);
  const colors = new Float32Array(points.length * 3);
  points.forEach((point, index) => {
    const pos = point.position;
    const px = pos ? pos[0]! : point.x ?? 0;
    const py = pos ? pos[1]! : point.y ?? 0;
    const pz = pos ? pos[2]! : point.z ?? 0;
    positions.set([px, py, pz], index * 3);
    const rawColor = point.color ?? [0.58, 0.82, 0.67];
    colors.set(
      rawColor.map((value) => (value > 1 ? value / 255 : value)),
      index * 3,
    );
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({ size: 0.055, vertexColors: true, sizeAttenuation: true });
  group.add(new THREE.Points(geometry, material));
  return points.length;
}

async function loadCollision(
  url: string,
  scene: THREE.Scene,
  visible: boolean,
): Promise<THREE.Object3D | null> {
  try {
    const { GLTFLoader } = await import("three/examples/jsm/loaders/GLTFLoader.js");
    const result = await new GLTFLoader().loadAsync(url);
    result.scene.visible = visible;
    result.scene.userData.collisionProxy = true;
    scene.add(result.scene);
    return result.scene;
  } catch (error) {
    console.warn("Collision proxy unavailable.", error);
    return null;
  }
}

export async function loadSplats(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  manifest: SplatManifest,
  options: SplatLoadOptions = {},
): Promise<SplatLoadResult> {
  const artifacts = manifest.artifacts ?? {};
  const base = options.baseUrl ?? "";
  let collision: THREE.Object3D | null = null;
  if (artifacts.collision_glb_relpath) {
    collision = await loadCollision(
      joinUrl(base, artifacts.collision_glb_relpath),
      scene,
      options.showCollision ?? false,
    );
  }

  const url = artifacts.spz_relpath
    ? joinUrl(base, artifacts.spz_relpath)
    : artifacts.sog_relpath
      ? joinUrl(base, artifacts.sog_relpath)
      : "";

  if (url && supportsSpark(renderer)) {
    let sparkRenderer: THREE.Object3D | null = null;
    let mesh: (THREE.Object3D & { initialized: Promise<unknown>; splats?: { getNumSplats: () => number }; packedSplats?: { numSplats?: number }; dispose?: () => void }) | null = null;
    try {
      const { SparkRenderer, SplatMesh } = await import("@sparkjsdev/spark");
      sparkRenderer = new SparkRenderer({ renderer });
      mesh = new SplatMesh({ url, raycastable: true });
      await mesh.initialized;
      const splatCount = mesh.splats?.getNumSplats() ?? mesh.packedSplats?.numSplats ?? 0;
      scene.add(sparkRenderer, mesh);
      return { mode: "spark", object: mesh, sparkRenderer, count: splatCount, collision };
    } catch (error) {
      mesh?.dispose?.();
      if (sparkRenderer && "dispose" in sparkRenderer && typeof sparkRenderer.dispose === "function") {
        (sparkRenderer as { dispose: () => void }).dispose();
      }
      console.warn("Spark splat loading unavailable; using sparse points.", error);
    }
  }

  if (!artifacts.sparse_points_relpath) {
    return { mode: "empty", object: null, sparkRenderer: null, count: 0, collision };
  }
  const group = new THREE.Group();
  const count = await loadPoints(joinUrl(base, artifacts.sparse_points_relpath), group);
  scene.add(group);
  return { mode: "points", object: group, sparkRenderer: null, count, collision };
}

export function disposeLoadedSplats(scene: THREE.Scene, asset: SplatLoadResult | null): void {
  if (!asset) return;
  for (const root of [asset.object, asset.sparkRenderer, asset.collision]) {
    if (!root) continue;
    root.traverse((node) => {
      const renderable = node as THREE.Object3D & {
        geometry?: THREE.BufferGeometry;
        material?: THREE.Material | THREE.Material[];
        dispose?: () => void;
      };
      renderable.geometry?.dispose();
      renderable.dispose?.();
      if (Array.isArray(renderable.material)) {
        renderable.material.forEach((mat) => mat.dispose());
      } else {
        renderable.material?.dispose();
      }
    });
    scene.remove(root);
  }
}
