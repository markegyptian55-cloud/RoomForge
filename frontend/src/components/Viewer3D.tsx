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
}

type Mode = "orbit" | "walk";

const IDENTITY_MATRIX: number[] = [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
];

const DEFAULT_EYE: [number, number, number] = [0.2, 1.4, -2.0];
const DEFAULT_TARGET: [number, number, number] = [-0.25, 0.75, 1.6];

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

const Viewer3D = forwardRef<ViewerController, Viewer3DProps>(function Viewer3D(
  { manifest, assetBaseUrl = "", onControllerReady, onVersionChange, isEditMode = false, onToggleEditMode },
  ref,
) {
  const hostRef = useRef<HTMLDivElement>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const orbitRef = useRef<OrbitControls | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const transformRef = useRef<number[]>(flattenMatrix(manifest?.T_viewer_from_colmap));
  const handlersRef = useRef(new Set<(event: ViewerClickEvent) => void>());
  const versionCallbackRef = useRef(onVersionChange);
  const pathRafRef = useRef<number | null>(null);

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

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#0c111d");
    const light = new THREE.HemisphereLight(0xffffff, 0x55705f, 2);
    scene.add(light);

    const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 500);
    // YXZ prevents gimbal roll during WASD + mouse-look walk mode.
    camera.rotation.order = "YXZ";
    camera.position.set(...DEFAULT_EYE);
    camera.lookAt(...DEFAULT_TARGET);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(renderer.domElement);

    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.target.set(...DEFAULT_TARGET);
    orbit.enableDamping = true;
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
      const key = event.key.toLowerCase();
      if (key === "e" && !event.ctrlKey && !event.altKey && !event.metaKey) {
        const target = event.target as HTMLElement | null;
        if (target?.tagName !== "INPUT" && target?.tagName !== "TEXTAREA" && target?.tagName !== "SELECT") {
          event.preventDefault();
          onToggleEditModeRef.current?.();
          return;
        }
      }
      if (["arrowup", "arrowdown", "arrowleft", "arrowright", "w", "a", "s", "d"].includes(key)) {
        event.preventDefault();
      }
      keys.add(key);
    };
    const onKeyUp = (event: KeyboardEvent) => keys.delete(event.key.toLowerCase());

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
      const hit = raycaster.intersectObjects(scene.children, true).find((item) => item.object !== light);

      if (hit) {
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

      if (!isEditModeRef.current) {
        setHintMessage("💡 Object Selection: Click 'Edit Object Mode: OFF' in the sidebar (or press E) to select objects for editing.");
        setTimeout(() => setHintMessage(null), 4500);
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
            phi = Math.max(0.08, phi - panSpeed);
          }
          if (keys.has("arrowdown") || keys.has("s")) {
            phi = Math.min(Math.PI - 0.08, phi + panSpeed);
          }

          offset.x = radius * Math.sin(phi) * Math.sin(theta);
          offset.y = radius * Math.cos(phi);
          offset.z = radius * Math.sin(phi) * Math.cos(theta);
          camera.position.copy(orbit.target).add(offset);
          camera.lookAt(orbit.target);
        }
        orbit.update();
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

    if (manifest) {
      void loadSplats(renderer, scene, manifest, { baseUrl: assetBaseUrl, showCollision: initialCollisionRef.current })
        .then((result) => {
          if (disposed) {
            disposeLoadedSplats(scene, result);
            return;
          }
          asset = result;
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
  }, [manifest, assetBaseUrl]);

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
          {telemetry.mode === "points" ? "points" : "splats"}
        </span>
      </div>
    </div>
  );
});

export default Viewer3D;
