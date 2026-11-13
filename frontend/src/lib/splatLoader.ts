import * as THREE from "three";

export interface SplatManifestArtifacts {
  spz_relpath?: string;
  sog_relpath?: string | null;
  mesh_glb_relpath?: string | null;
  scene_glb_relpath?: string | null;
  sparse_points_relpath?: string;
  collision_glb_relpath?: string | null;
}

export interface SplatManifest {
  artifacts?: SplatManifestArtifacts;
  T_viewer_from_colmap?: number[][];
}

export interface SplatLoadOptions {
  baseUrl?: string;
  showCollision?: boolean;
}

export interface SplatLoadResult {
  mode: "spark" | "points" | "mesh" | "empty";
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

function createGaussianCircleTexture(): THREE.Texture | null {
  if (typeof document === "undefined") return null;
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const imgData = ctx.createImageData(size, size);
  const data = imgData.data;
  const center = (size - 1) / 2;
  const radius = size / 2;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (x - center) / radius;
      const dy = (y - center) / radius;
      const d2 = dx * dx + dy * dy;
      const idx = (y * size + x) * 4;
      // All RGB channels are strictly 255 to eliminate any dark fringe on bilinear sampling
      data[idx] = 255;
      data[idx + 1] = 255;
      data[idx + 2] = 255;
      if (d2 <= 1.0) {
        // Gaussian falloff exp(-d2 * 3.0)
        const alpha = Math.exp(-d2 * 3.0);
        data[idx + 3] = Math.round(alpha * 255);
      } else {
        data[idx + 3] = 0;
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

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
  const gaussianMap = createGaussianCircleTexture();
  const material = new THREE.PointsMaterial({
    size: 0.048,
    vertexColors: true,
    sizeAttenuation: true,
    ...(gaussianMap ? { map: gaussianMap, transparent: true, alphaTest: 0.015, depthWrite: true } : {}),
  });
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

function applyViewerTransform(object: THREE.Object3D, transform?: number[][]): void {
  if (!transform || !Array.isArray(transform) || transform.length !== 4) return;
  const flat = transform.flat();
  if (flat.length !== 16) return;
  const mat = new THREE.Matrix4().set(
    flat[0]!, flat[1]!, flat[2]!, flat[3]!,
    flat[4]!, flat[5]!, flat[6]!, flat[7]!,
    flat[8]!, flat[9]!, flat[10]!, flat[11]!,
    flat[12]!, flat[13]!, flat[14]!, flat[15]!,
  );
  object.applyMatrix4(mat);
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

  const glbPath = artifacts.scene_glb_relpath ?? artifacts.mesh_glb_relpath;
  if (glbPath) {
    try {
      const { GLTFLoader } = await import("three/examples/jsm/loaders/GLTFLoader.js");
      const gltf = await new GLTFLoader().loadAsync(joinUrl(base, glbPath));
      scene.add(gltf.scene);
      let triCount = 0;
      gltf.scene.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const m = child as THREE.Mesh;
          m.castShadow = true;
          m.receiveShadow = true;
          if (m.geometry) {
            const index = m.geometry.index;
            triCount += index ? index.count / 3 : m.geometry.attributes.position?.count ? m.geometry.attributes.position.count / 3 : 0;
          }
        }
      });
      return { mode: "mesh", object: gltf.scene, sparkRenderer: null, count: Math.round(triCount), collision };
    } catch (error) {
      console.warn("Photorealistic GLB room loading failed, falling back to splat/points.", error);
    }
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
      applyViewerTransform(mesh, manifest.T_viewer_from_colmap);
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
  applyViewerTransform(group, manifest.T_viewer_from_colmap);
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
