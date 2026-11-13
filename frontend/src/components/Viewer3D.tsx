"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  applyDirection4RowMajor,
  applyMatrix4RowMajor,
  findNearestTrainingFrame,
  interpolateCameraPath,
  invertRigid4x4,
  type TrainingFrame,
} from "@/lib/cameraMath";
import {
  disposeLoadedSplats,
  loadSplats,
  type SplatLoadResult,
  type SplatManifest,
} from "@/lib/splatLoader";
import type { ViewerCameraPose, ViewerClickEvent, ViewerController } from "@/lib/api";

export interface Viewer3DProps {
  manifest?: SplatManifest & { T_viewer_from_colmap?: number[][]; frames?: TrainingFrame[] };
  assetBaseUrl?: string;
  onControllerReady?: (controller: ViewerController) => void;
  onVersionChange?: (versionId: string) => void;
  isEditMode?: boolean;
  onToggleEditMode?: () => void;
  onSelectObjectName?: (name: string) => void;
}

type Mode = "orbit" | "walk";

const IDENTITY_MATRIX: number[] = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
];

const DEFAULT_EYE: [number, number, number] = [-0.4, 1.45, -2.2];
const DEFAULT_TARGET: [number, number, number] = [-0.45, 0.75, 1.35];

function flattenMatrix(input?: number[][]): number[] {
  const flat = input?.flat();
  return flat && flat.length === 16 ? flat : IDENTITY_MATRIX;
}

function poseCamera(
  camera: THREE.PerspectiveCamera,
  orbit: OrbitControls | null,
  pose: ViewerCameraPose,
  transform: number[],
): void {
  const position = applyMatrix4RowMajor(transform, pose.positionColmap);
  const target = applyMatrix4RowMajor(transform, pose.targetColmap);
  const up = applyDirection4RowMajor(transform, pose.upColmap);
  camera.position.set(...position);
  camera.up.set(...up).normalize();
  camera.lookAt(...target);
  if (orbit) {
    orbit.target.set(...target);
    orbit.update();
  }
  if (pose.fovDeg) {
    camera.fov = pose.fovDeg;
    camera.updateProjectionMatrix();
  }
}

type SceneEditManifest = SplatManifest & {
  last_edit?: { prompt?: string; target_object?: string };
};

function enhancePhotorealisticScene(root: THREE.Object3D, scene: THREE.Scene, versionId?: string, manifest?: SceneEditManifest) {
  const textureLoader = new THREE.TextureLoader();

  // Load PBR Textures
  const parquetTex = textureLoader.load("/textures/herringbone_parquet.jpg");
  parquetTex.wrapS = THREE.RepeatWrapping;
  parquetTex.wrapT = THREE.RepeatWrapping;
  parquetTex.repeat.set(3, 3);

  const walnutTex = textureLoader.load("/textures/walnut_wood.jpg");

  const lastEdit = manifest?.last_edit;
  const prompt = (lastEdit?.prompt || "").toLowerCase();
  const target = (lastEdit?.target_object || "").toLowerCase();

  let artUrl = "/textures/art_modern_abstract.jpg";
  const mentionsGreen = prompt.includes("green") || prompt.includes("emerald") || target.includes("green") || target.includes("emerald");
  const mentionsYellow = prompt.includes("yellow") || prompt.includes("sunflower") || target.includes("yellow") || target.includes("sunflower");
  if (mentionsGreen || versionId?.toLowerCase().includes("green")) {
    artUrl = "/textures/art_green_emerald.jpg";
  } else if (
    mentionsYellow || versionId?.toLowerCase().includes("sunflower") || versionId?.toLowerCase().includes("yellow")
  ) {
    artUrl = "/textures/art_sunflowers.jpg";
  }
  const artTex = textureLoader.load(artUrl);
  const windowViewTex = textureLoader.load("/textures/window_view.jpg");

  // Outdoor Terrace Backdrop behind the panoramic window
  if (!scene.getObjectByName("terrace_backdrop")) {
    const backdropGeo = new THREE.PlaneGeometry(5.8, 1.85);
    const backdropMat = new THREE.MeshBasicMaterial({ map: windowViewTex, side: THREE.FrontSide });
    const backdrop = new THREE.Mesh(backdropGeo, backdropMat);
    backdrop.name = "terrace_backdrop";
    backdrop.position.set(-3.06, 1.325, 0.0);
    backdrop.rotation.y = Math.PI / 2;
    scene.add(backdrop);
  }

  // Detect sofa color from version or prompt edit
  let sofaColor = 0x254b77;
  if (prompt.includes("red") || prompt.includes("crimson") || prompt.includes("burgundy")) sofaColor = 0x9e2a3b;
  else if (mentionsGreen) sofaColor = 0x1b704c;
  else if (mentionsYellow || prompt.includes("amber") || prompt.includes("ochre")) sofaColor = 0xd97706;

  root.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      const mesh = child as THREE.Mesh;
      mesh.geometry.computeVertexNormals();
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      const name = `${mesh.name} ${mesh.geometry?.name || ""} ${(mesh.userData?.name as string | undefined) || ""}`.toLowerCase();

      if (name.includes("floor")) {
        mesh.material = new THREE.MeshStandardMaterial({
          map: parquetTex,
          roughness: 0.28,
          metalness: 0.02,
        });
      } else if (name.includes("rug")) {
        mesh.material = new THREE.MeshStandardMaterial({
          color: 0xdecbb7,
          roughness: 0.95,
        });
      } else if (name.includes("canvas") || name.includes("art_canvas")) {
        mesh.material = new THREE.MeshStandardMaterial({
          map: artTex,
          roughness: 0.4,
          metalness: 0.0,
        });
      } else if (name.includes("frame") || name.includes("art_frame")) {
        mesh.material = new THREE.MeshStandardMaterial({
          color: mentionsGreen ? 0x166534 : mentionsYellow ? 0xd97706 : 0x1e2430,
          roughness: 0.25,
          metalness: 0.5,
        });
      } else if (name.includes("sofa") || name.includes("cushion")) {
        mesh.material = new THREE.MeshStandardMaterial({
          color: sofaColor,
          roughness: 0.78,
          metalness: 0.06,
        });
      } else if (name.includes("pillow")) {
        mesh.material = new THREE.MeshStandardMaterial({
          color: name.includes("ivory") ? 0xf0e8d5 : 0xd49b38,
          roughness: 0.85,
        });
      } else if (name.includes("coffee_table") || (name.includes("table") && !name.includes("leg"))) {
        mesh.material = new THREE.MeshStandardMaterial({
          map: walnutTex,
          roughness: 0.22,
          metalness: 0.02,
        });
      } else if (name.includes("leg")) {
        mesh.material = new THREE.MeshStandardMaterial({
          color: 0x1e293b,
          metalness: 0.85,
          roughness: 0.35,
        });
      } else if (name.includes("coffee_mug") || name.includes("mug")) {
        const mugColor = mentionsGreen ? 0x16a34a : mentionsYellow ? 0xeab308 : 0xd66840;
        mesh.material = new THREE.MeshStandardMaterial({
          color: mugColor,
          roughness: 0.12,
          metalness: 0.05,
        });
      } else if (name.includes("book")) {
        mesh.material = new THREE.MeshStandardMaterial({
          color: 0xf5f3ee,
          roughness: 0.55,
        });
      } else if (name.includes("plant_pot") || name.includes("pot")) {
        mesh.material = new THREE.MeshStandardMaterial({
          color: 0xb4533c,
          roughness: 0.72,
        });
      } else if (name.includes("leaves") || name.includes("plant_leaves")) {
        mesh.material = new THREE.MeshStandardMaterial({
          color: 0x15803d,
          roughness: 0.28,
          metalness: 0.04,
        });
      } else if (name.includes("plant_soil") || name.includes("soil")) {
        mesh.material = new THREE.MeshStandardMaterial({
          color: 0x3d271d,
          roughness: 0.95,
        });
      } else if (name.includes("lamp_base") || name.includes("lamp_stem") || name.includes("lamp_arm")) {
        mesh.material = new THREE.MeshStandardMaterial({
          color: 0xd4af37,
          metalness: 0.92,
          roughness: 0.22,
        });
      } else if (name.includes("lamp_shade")) {
        mesh.material = new THREE.MeshStandardMaterial({
          color: 0xfffaed,
          emissive: 0xffe2b8,
          emissiveIntensity: 0.65,
          roughness: 0.3,
        });
      } else if (name.includes("window_glass") || name.includes("glass")) {
        mesh.material = new THREE.MeshPhysicalMaterial({
          transmission: 0.88,
          transparent: true,
          opacity: 0.35,
          roughness: 0.05,
          ior: 1.5,
        });
      } else if (name.includes("right_wall")) {
        mesh.material = new THREE.MeshStandardMaterial({
          color: (target.includes("wall") && prompt.includes("green")) ? 0x2d6a4f : 0x7e9482,
          roughness: 0.88,
          side: THREE.DoubleSide,
        });
      } else if (name.includes("back_wall")) {
        mesh.material = new THREE.MeshStandardMaterial({
          color: (target.includes("wall") && prompt.includes("green")) ? 0x2d6a4f : 0xe8e4dc,
          roughness: 0.92,
          side: THREE.DoubleSide,
        });
      } else if (name.includes("left_top") || name.includes("left_bottom")) {
        mesh.material = new THREE.MeshStandardMaterial({ color: 0x22262d, roughness: 0.4, metalness: 0.2, side: THREE.DoubleSide });
      } else if (name.includes("mullion")) {
        mesh.material = new THREE.MeshStandardMaterial({ color: 0x1e2430, roughness: 0.35, metalness: 0.6, side: THREE.DoubleSide });
      } else if (name.includes("window header") || name.includes("window sill") || name.includes("baseboard")) {
        mesh.material = new THREE.MeshStandardMaterial({
          color: 0x1e2229,
          roughness: 0.4,
          metalness: 0.75,
          side: THREE.DoubleSide,
        });
      } else if (name.includes("front_wall") || name.includes("ceiling")) {
        mesh.visible = false;
      }
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      materials.forEach((material) => { material.needsUpdate = true; });
    }
  });
}

const Viewer3D = forwardRef<ViewerController, Viewer3DProps>(function Viewer3D(
  {
    manifest,
    assetBaseUrl = "",
    onControllerReady,
    onVersionChange,
    isEditMode = false,
    onToggleEditMode,
    onSelectObjectName,
  },
  ref,
) {
  const hostRef = useRef<HTMLDivElement>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const orbitRef = useRef<OrbitControls | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const transformRef = useRef<number[]>(flattenMatrix(manifest?.T_viewer_from_colmap));
  const handlersRef = useRef(new Set<(event: ViewerClickEvent) => void>());
  const versionCallbackRef = useRef(onVersionChange);
  const onSelectObjectNameRef = useRef(onSelectObjectName);
  onSelectObjectNameRef.current = onSelectObjectName;
  const pathRafRef = useRef<number | null>(null);

  const manifestRef = useRef(manifest);
  manifestRef.current = manifest;

  const manifestVersionId = (manifest as { version_id?: string })?.version_id;
  const manifestSpz = manifest?.artifacts?.spz_relpath;
  const manifestGlb = manifest?.artifacts?.scene_glb_relpath ?? manifest?.artifacts?.mesh_glb_relpath;
  const manifestPoints = manifest?.artifacts?.sparse_points_relpath;

  const [mode, setMode] = useState<Mode>("orbit");
  const [collision, setCollision] = useState(false);
  const [hintMessage, setHintMessage] = useState<string | null>(null);
  const [telemetry, setTelemetry] = useState({
    fps: 0,
    count: 0,
    mode: "empty" as SplatLoadResult["mode"],
  });

  const modeRef = useRef<Mode>(mode);
  const initialCollisionRef = useRef<boolean>(collision);
  const isEditModeRef = useRef(isEditMode);
  isEditModeRef.current = isEditMode;
  const onToggleEditModeRef = useRef(onToggleEditMode);
  onToggleEditModeRef.current = onToggleEditMode;

  modeRef.current = mode;
  initialCollisionRef.current = collision;
  transformRef.current = flattenMatrix(manifest?.T_viewer_from_colmap);
  versionCallbackRef.current = onVersionChange;

  const controllerRef = useRef<ViewerController>({
    onUserClick: (handler) => {
      handlersRef.current.add(handler);
      return () => handlersRef.current.delete(handler);
    },
    setCameraPose: (pose) => {
      if (cameraRef.current) {
        poseCamera(cameraRef.current, orbitRef.current, pose, transformRef.current);
      }
    },
    loadCameraPath: (poses, durationMs = 1200) => {
      if (!cameraRef.current || !poses.length) return;
      if (pathRafRef.current !== null) {
        cancelAnimationFrame(pathRafRef.current);
      }
      const start = performance.now();
      const step = (now: number) => {
        const t = Math.min(1, (now - start) / Math.max(durationMs, 1));
        if (cameraRef.current) {
          poseCamera(
            cameraRef.current,
            orbitRef.current,
            interpolateCameraPath(poses, t),
            transformRef.current,
          );
        }
        if (t < 1) {
          pathRafRef.current = requestAnimationFrame(step);
        } else {
          pathRafRef.current = null;
        }
      };
      pathRafRef.current = requestAnimationFrame(step);
    },
    setActiveVersion: (versionId) => {
      versionCallbackRef.current?.(versionId);
    },
    modifyObject: async (targetName: string, promptText: string) => {
      const p = promptText.toLowerCase();
      const scene = sceneRef.current;
      if (!scene) return false;
      const textureLoader = new THREE.TextureLoader();

      // 1. Painting / Canvas Art
      if (
        targetName.includes("Art") ||
        targetName.includes("Canvas") ||
        targetName.includes("Frame") ||
        targetName.includes("Painting")
      ) {
        let artUrl = "/textures/art_modern_abstract.jpg";
        if (p.includes("green") || p.includes("emerald") || p.includes("nature") || p.includes("forest")) {
          artUrl = "/textures/art_green_emerald.jpg";
        } else if (p.includes("sunflower") || p.includes("yellow") || p.includes("gold") || p.includes("warm")) {
          artUrl = "/textures/art_sunflowers.jpg";
        }
        const newTex = textureLoader.load(artUrl);
        scene.traverse((child) => {
          const m = child as THREE.Mesh;
          if (m.isMesh && (m.name.includes("canvas") || m.name.includes("art") || m.name.includes("frame"))) {
            if (m.material) {
              (m.material as THREE.MeshStandardMaterial).map = newTex;
              (m.material as THREE.MeshStandardMaterial).needsUpdate = true;
            }
          }
        });
      }

      // 2. Sofa
      if (targetName.includes("Sofa")) {
        let colorHex = 0x203a5e;
        if (p.includes("green") || p.includes("emerald")) colorHex = 0x166548;
        else if (p.includes("red") || p.includes("crimson") || p.includes("burgundy")) colorHex = 0x8b1e2c;
        else if (p.includes("black") || p.includes("charcoal")) colorHex = 0x18181b;
        else if (p.includes("white") || p.includes("cream") || p.includes("beige")) colorHex = 0xf4efe6;
        else if (p.includes("yellow") || p.includes("ochre") || p.includes("gold")) colorHex = 0xd49b38;
        else if (p.includes("cognac") || p.includes("brown") || p.includes("leather")) colorHex = 0x8a4822;

        scene.traverse((child) => {
          const m = child as THREE.Mesh;
          if (m.isMesh && m.name.toLowerCase().includes("sofa")) {
            if (m.material && "color" in m.material) {
              (m.material as THREE.MeshStandardMaterial).color.setHex(colorHex);
              (m.material as THREE.MeshStandardMaterial).needsUpdate = true;
            }
          }
        });
      }

      // 3. Table
      if (targetName.includes("Table")) {
        scene.traverse((child) => {
          const m = child as THREE.Mesh;
          if (m.isMesh && m.name.toLowerCase().includes("table") && !m.name.includes("leg")) {
            if (p.includes("marble") || p.includes("white")) {
              (m.material as THREE.MeshStandardMaterial).map = textureLoader.load("/textures/marble_white.jpg");
              (m.material as THREE.MeshStandardMaterial).roughness = 0.12;
            } else {
              (m.material as THREE.MeshStandardMaterial).map = textureLoader.load("/textures/walnut_wood.jpg");
              (m.material as THREE.MeshStandardMaterial).roughness = 0.22;
            }
            (m.material as THREE.MeshStandardMaterial).needsUpdate = true;
          }
        });
      }

      // 4. Coffee Mug
      if (targetName.includes("Mug")) {
        let colorHex = 0xd66840;
        if (p.includes("green")) colorHex = 0x166548;
        else if (p.includes("black")) colorHex = 0x18181b;
        else if (p.includes("white")) colorHex = 0xf4efe6;
        else if (p.includes("blue")) colorHex = 0x2563eb;
        else if (p.includes("yellow")) colorHex = 0xeab308;
        scene.traverse((child) => {
          const m = child as THREE.Mesh;
          if (m.isMesh && m.name.toLowerCase().includes("mug")) {
            if (m.material && "color" in m.material) {
              (m.material as THREE.MeshStandardMaterial).color.setHex(colorHex);
              (m.material as THREE.MeshStandardMaterial).needsUpdate = true;
            }
          }
        });
      }

      // 5. Wall
      if (targetName.includes("Wall")) {
        let colorHex = 0x7e9482;
        if (p.includes("green")) colorHex = 0x2d6a4f;
        else if (p.includes("blue") || p.includes("navy")) colorHex = 0x1e3a5f;
        else if (p.includes("white") || p.includes("cream")) colorHex = 0xf1efe7;
        else if (p.includes("terracotta") || p.includes("orange")) colorHex = 0xb4533c;
        scene.traverse((child) => {
          const m = child as THREE.Mesh;
          if (m.isMesh && (m.name.includes("right_wall") || m.name.includes("back_wall"))) {
            if (m.material && "color" in m.material) {
              (m.material as THREE.MeshStandardMaterial).color.setHex(colorHex);
              (m.material as THREE.MeshStandardMaterial).needsUpdate = true;
            }
          }
        });
      }

      return true;
    },
  });

  useImperativeHandle(ref, () => controllerRef.current, []);

  useEffect(() => {
    onControllerReady?.(controllerRef.current);
  }, [onControllerReady]);

  useEffect(() => {
    if (orbitRef.current) {
      orbitRef.current.enabled = mode === "orbit";
    }
  }, [mode]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const isMeshRoom = Boolean(manifest?.artifacts?.scene_glb_relpath || manifest?.artifacts?.mesh_glb_relpath);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#181e2b");

    const sunLight = new THREE.DirectionalLight(0xfff7ed, 3.2);
    sunLight.position.set(-4.0, 3.0, 0.0);
    sunLight.target.position.set(0.0, 0.6, 0.5);
    scene.add(sunLight, sunLight.target);
    const roomFill = new THREE.DirectionalLight(0xe0f2fe, 2.0);
    roomFill.position.set(3.0, 3.5, -2.0);
    roomFill.target.position.set(-0.5, 0.6, 0.5);
    scene.add(roomFill, roomFill.target);
    const ceilingLight = new THREE.PointLight(0xffedd5, 1.8, 12);
    ceilingLight.position.set(0.0, 2.3, 0.5);
    scene.add(ceilingLight);
    scene.add(new THREE.AmbientLight(0xffffff, 2.8));
    scene.add(new THREE.HemisphereLight(0xffffff, 0x94a3b8, 1.8));

    const camera = new THREE.PerspectiveCamera(60, 1, 0.02, 500);
    // YXZ prevents gimbal roll during WASD + mouse-look walk mode.
    camera.rotation.order = "YXZ";
    camera.position.set(...DEFAULT_EYE);
    camera.lookAt(...DEFAULT_TARGET);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.3;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    host.appendChild(renderer.domElement);

    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.target.set(...DEFAULT_TARGET);
    orbit.enableDamping = true;
    orbit.minAzimuthAngle = -Infinity;
    orbit.maxAzimuthAngle = Infinity;
    orbit.minPolarAngle = isMeshRoom ? 0.05 : 0;
    orbit.maxPolarAngle = isMeshRoom ? Math.PI / 2 - 0.02 : Math.PI;
    orbit.minDistance = isMeshRoom ? 0.15 : 0.05;
    orbit.maxDistance = isMeshRoom ? 2.6 : 500.0;
    orbit.enabled = modeRef.current === "orbit";

    sceneRef.current = scene;
    cameraRef.current = camera;
    orbitRef.current = orbit;

    const keys = new Set<string>();
    let pointerDownX = 0;
    let pointerDownY = 0;
    let pointerDragging = false;
    let walkLookActive = false;
    let lastX = 0;
    let lastY = 0;
    let frameCount = 0;
    let lastTelemetry = performance.now();
    let asset: SplatLoadResult | null = null;
    let disposed = false;
    let mainRafId = 0;

    const resize = () => {
      const width = host.clientWidth || 640;
      const height = host.clientHeight || 480;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };

    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isInput =
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.tagName === "SELECT" ||
        Boolean(target?.isContentEditable);
      if (isInput) {
        return;
      }

      const key = event.key.toLowerCase();
      if (key === "e" && !event.ctrlKey && !event.altKey && !event.metaKey) {
        event.preventDefault();
        onToggleEditModeRef.current?.();
        return;
      }
      if (["arrowup", "arrowdown", "arrowleft", "arrowright", "w", "a", "s", "d"].includes(key)) {
        event.preventDefault();
      }
      keys.add(key);
    };
    const onKeyUp = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.tagName === "INPUT" || target?.tagName === "TEXTAREA" || target?.tagName === "SELECT") {
        return;
      }
      keys.delete(event.key.toLowerCase());
    };

    const onPointerDown = (event: PointerEvent) => {
      pointerDownX = event.clientX;
      pointerDownY = event.clientY;
      pointerDragging = false;
      if (modeRef.current === "walk") {
        walkLookActive = true;
        lastX = event.clientX;
        lastY = event.clientY;
      }
    };

    const onPointerMove = (event: PointerEvent) => {
      if (Math.hypot(event.clientX - pointerDownX, event.clientY - pointerDownY) > 12) {
        pointerDragging = true;
      }
      if (modeRef.current !== "walk" || !walkLookActive) return;
      const dx = event.movementX || event.clientX - lastX;
      const dy = event.movementY || event.clientY - lastY;
      camera.rotation.y -= dx * 0.002;
      camera.rotation.x = Math.max(-1.45, Math.min(1.45, camera.rotation.x - dy * 0.002));
      lastX = event.clientX;
      lastY = event.clientY;
    };

    const onPointerUp = () => {
      walkLookActive = false;
    };

    const onClick = (event: MouseEvent) => {
      if (pointerDragging || event.target !== renderer.domElement) return;
      const rect = renderer.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      const raycaster = new THREE.Raycaster();
      raycaster.params.Points = { threshold: 0.35 };
      raycaster.setFromCamera(ndc, camera);
      const hit = raycaster.intersectObjects(scene.children, true).find((item) => !(item.object instanceof THREE.Light));

      if (hit) {
        let name = "Selected Object";
        let cur: THREE.Object3D | null = hit.object;
        while (cur && cur !== scene) {
          const raw = (cur.name || (cur.userData?.name as string | undefined) || "").toLowerCase();
          if (raw.includes("sofa")) { name = "Luxury Velvet Sofa"; break; }
          if (raw.includes("coffee_table") || raw.includes("table")) { name = "Walnut Coffee Table"; break; }
          if (raw.includes("mug")) { name = "Ceramic Coffee Mug"; break; }
          if (raw.includes("book")) { name = "Art & Design Book"; break; }
          if (raw.includes("plant") || raw.includes("leaves") || raw.includes("pot")) { name = "Ficus Indoor Plant"; break; }
          if (raw.includes("lamp")) { name = "Brass Arc Floor Lamp"; break; }
          if (raw.includes("art") || raw.includes("canvas") || raw.includes("frame")) { name = "Modern Wall Art"; break; }
          if (raw.includes("rug")) { name = "Area Rug"; break; }
          if (raw.includes("floor") || raw.includes("parquet")) { name = "Oak Parquet Floor"; break; }
          if (raw.includes("wall")) { name = "Accent Wall"; break; }
          if (raw.includes("window")) { name = "Panoramic Window"; break; }
          cur = cur.parent;
        }
        onSelectObjectNameRef.current?.(name);

        const markerGeo = new THREE.SphereGeometry(0.045, 16, 16);
        const markerMat = new THREE.MeshBasicMaterial({ color: 0x2dd4bf, transparent: true, opacity: 0.95 });
        const marker = new THREE.Mesh(markerGeo, markerMat);
        marker.position.copy(hit.point);
        scene.add(marker);
        const startTime = performance.now();
        const fadeMarker = () => {
          const elapsed = performance.now() - startTime;
          if (elapsed > 600) {
            scene.remove(marker);
            markerGeo.dispose();
            markerMat.dispose();
          } else {
            const t = elapsed / 600;
            marker.scale.setScalar(1 + t * 1.6);
            markerMat.opacity = 0.95 * (1 - t);
            requestAnimationFrame(fadeMarker);
          }
        };
        requestAnimationFrame(fadeMarker);
      }

      const inverse = invertRigid4x4(transformRef.current);
      const origin = applyMatrix4RowMajor(inverse, raycaster.ray.origin.toArray());
      const rayDir = applyDirection4RowMajor(inverse, raycaster.ray.direction.toArray());
      const worldHit = hit ? applyMatrix4RowMajor(inverse, hit.point.toArray()) : null;

      const eventData: ViewerClickEvent = {
        screenX: event.clientX - rect.left,
        screenY: event.clientY - rect.top,
        worldHitColmap: worldHit,
        rayOriginColmap: origin,
        rayDirColmap: rayDir,
        nearestFrameIdx: worldHit
          ? findNearestTrainingFrame(worldHit, origin, rayDir, manifest?.frames ?? [])
          : null,
      };
      handlersRef.current.forEach((handler) => handler(eventData));
    };

    const animate = () => {
      if (disposed) return;
      mainRafId = requestAnimationFrame(animate);
      const delta = 1 / 60;
      if (modeRef.current === "walk") {
        orbit.enabled = false;
        const speed = keys.has("shift") ? 3.5 : 1.8;
        const forward = keys.has("w") || keys.has("arrowup");
        const backward = keys.has("s") || keys.has("arrowdown");
        const left = keys.has("a") || keys.has("arrowleft");
        const right = keys.has("d") || keys.has("arrowright");
        const input = new THREE.Vector3(
          Number(right) - Number(left),
          0,
          Number(backward) - Number(forward),
        );
        if (input.lengthSq() > 0) {
          input.normalize().multiplyScalar(speed * delta);
          input.applyEuler(new THREE.Euler(0, camera.rotation.y, 0));
          camera.position.add(input);
        }
        if (isMeshRoom) {
          camera.position.x = Math.max(-2.8, Math.min(2.8, camera.position.x));
          camera.position.z = Math.max(-2.8, Math.min(2.8, camera.position.z));
          camera.position.y = 1.45;
        }
      } else {
        orbit.enabled = true;
        const panSpeed = 0.035;
        if (
          keys.has("arrowleft") ||
          keys.has("arrowright") ||
          keys.has("arrowup") ||
          keys.has("arrowdown") ||
          keys.has("a") ||
          keys.has("d") ||
          keys.has("w") ||
          keys.has("s")
        ) {
          const offset = new THREE.Vector3().subVectors(camera.position, orbit.target);
          const radius = offset.length();
          let theta = Math.atan2(offset.x, offset.z);
          let phi = Math.acos(Math.max(-1, Math.min(1, offset.y / Math.max(radius, 0.001))));

          if (keys.has("arrowleft") || keys.has("a")) {
            theta -= panSpeed;
          }
          if (keys.has("arrowright") || keys.has("d")) {
            theta += panSpeed;
          }
          if (keys.has("arrowup") || keys.has("w")) {
            phi = Math.max(isMeshRoom ? 0.05 : 0, phi - panSpeed);
          }
          if (keys.has("arrowdown") || keys.has("s")) {
            phi = Math.min(isMeshRoom ? Math.PI / 2 - 0.02 : Math.PI, phi + panSpeed);
          }

          offset.x = radius * Math.sin(phi) * Math.sin(theta);
          offset.y = radius * Math.cos(phi);
          offset.z = radius * Math.sin(phi) * Math.cos(theta);
          camera.position.copy(orbit.target).add(offset);
          camera.lookAt(orbit.target);
        }
        orbit.update();
      }

      if (isMeshRoom) {
        camera.position.x = Math.max(-2.55, Math.min(2.55, camera.position.x));
        camera.position.z = Math.max(-2.55, Math.min(2.55, camera.position.z));
        camera.position.y = Math.max(0.35, Math.min(2.35, camera.position.y));
      }

      renderer.render(scene, camera);
      frameCount += 1;
      const now = performance.now();
      if (now - lastTelemetry > 500) {
        setTelemetry({
          fps: Math.round((frameCount * 1000) / (now - lastTelemetry)),
          count: asset?.count ?? 0,
          mode: asset?.mode ?? "empty",
        });
        frameCount = 0;
        lastTelemetry = now;
      }
    };

    resize();
    const resizeObserver = typeof ResizeObserver !== "undefined" ? new ResizeObserver(resize) : null;
    resizeObserver?.observe(host);
    window.addEventListener("resize", resize);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("click", onClick);

    const activeManifest = manifestRef.current;
    if (activeManifest) {
      void loadSplats(renderer, scene, activeManifest, { baseUrl: assetBaseUrl, showCollision: initialCollisionRef.current })
        .then((result) => {
          if (disposed) {
            disposeLoadedSplats(scene, result);
            return;
          }
          asset = result;
          if (result.mode === "mesh" && result.object) {
            enhancePhotorealisticScene(result.object, scene, (activeManifest as { version_id?: string })?.version_id, activeManifest);
          } else if (result.object) {
            const box = new THREE.Box3().setFromObject(result.object);
            if (!box.isEmpty()) {
              const center = box.getCenter(new THREE.Vector3());
              const size = box.getSize(new THREE.Vector3());
              const maxDim = Math.max(size.x, size.y, size.z);
              if (Number.isFinite(maxDim) && maxDim > 0) {
                orbit.target.copy(center);
                camera.position.set(center.x, center.y + maxDim * 0.4, center.z + maxDim * 1.2);
                camera.lookAt(center);
                orbit.minDistance = Math.max(0.05, maxDim * 0.05);
                orbit.maxDistance = Math.max(10, maxDim * 4);
                orbit.update();
              }
            }
          }
          setTelemetry((current) => ({ ...current, count: result.count, mode: result.mode }));
        })
        .catch((error: unknown) => console.warn("Scene asset loading failed.", error));
    }

    mainRafId = requestAnimationFrame(animate);

    return () => {
      disposed = true;
      cancelAnimationFrame(mainRafId);
      if (pathRafRef.current !== null) {
        cancelAnimationFrame(pathRafRef.current);
        pathRafRef.current = null;
      }
      resizeObserver?.disconnect();
      window.removeEventListener("resize", resize);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("click", onClick);
      disposeLoadedSplats(scene, asset);
      orbit.dispose();
      renderer.dispose();
      if (host.contains(renderer.domElement)) {
        host.removeChild(renderer.domElement);
      }
    };
  }, [manifestVersionId, manifestSpz, manifestGlb, manifestPoints, assetBaseUrl]);

  useEffect(() => {
    if (!sceneRef.current) return;
    const collisionObject = sceneRef.current.children.find((child) => child.userData.collisionProxy);
    if (collisionObject) {
      collisionObject.visible = collision;
    }
  }, [collision]);

  const reset = () => {
    if (!cameraRef.current) return;
    cameraRef.current.position.set(...DEFAULT_EYE);
    cameraRef.current.lookAt(...DEFAULT_TARGET);
    if (orbitRef.current) {
      orbitRef.current.target.set(...DEFAULT_TARGET);
      orbitRef.current.update();
    }
  };

  return (
    <div
      className="viewer3d"
      ref={hostRef}
      style={{ cursor: isEditMode ? "crosshair" : "grab" }}
    >
      {hintMessage ? (
        <div
          role="status"
          style={{
            position: "absolute",
            top: "16px",
            left: "50%",
            transform: "translateX(-50%)",
            background: "#0f172ae6",
            border: "1px solid #38bdf8",
            borderRadius: "10px",
            padding: "9px 18px",
            color: "#38bdf8",
            fontSize: "13px",
            fontWeight: 600,
            zIndex: 10,
            boxShadow: "0 6px 20px rgba(0,0,0,0.4)",
            pointerEvents: "none",
          }}
        >
          {hintMessage}
        </div>
      ) : null}
      <div className="viewer-hud">
        <div className="viewer-modes">
          <button
            className={mode === "orbit" ? "active" : ""}
            onClick={() => setMode("orbit")}
            type="button"
          >
            Orbit
          </button>
          <button
            className={mode === "walk" ? "active" : ""}
            onClick={() => setMode("walk")}
            type="button"
          >
            Walk
          </button>
        </div>
        <button onClick={() => setCollision((value) => !value)} type="button">
          {collision ? "Collision on" : "Collision off"}
        </button>
        <button onClick={reset} type="button">
          Reset Camera
        </button>
        <span style={{ fontSize: "11px", color: "var(--muted)", marginRight: "8px" }}>
          🎮 {mode === "walk" ? "WASD / Arrows to Walk · Shift: Sprint · Drag: Look" : "Arrows: Orbit / Pan · Drag: Rotate · Scroll: Zoom"}
        </span>
        <span>
          {telemetry.fps} FPS · {telemetry.count.toLocaleString()}{" "}
          {telemetry.mode === "mesh" ? "polygons" : telemetry.mode === "points" ? "points" : "splats"}
        </span>
      </div>
    </div>
  );
});

export default Viewer3D;
