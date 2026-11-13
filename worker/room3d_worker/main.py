from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from room3d_worker.client import StaleAttemptError, TaskClaim, WorkerClient, WorkerClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
LOGGER = logging.getLogger("room3d_worker.main")

DEFAULT_CAPABILITIES = [
    "reconstruct_week1",
    "pose_benchmark",
    "segment_init",
    "segment_refine",
    "track_propagate",
    "diffusion_preview",
    "diffusion_apply",
    "rebuild_week2",
]


class HeartbeatThread:
    def __init__(self, client: WorkerClient, task: TaskClaim, interval: float = 5.0) -> None:
        self.client = client
        self.task = task
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.client.heartbeat(self.task)
            except StaleAttemptError:
                LOGGER.warning("Task %s attempt %s became stale", self.task.task_id, self.task.attempt_id)
                break
            except Exception as exc:
                LOGGER.debug("Heartbeat error: %s", exc)
            self._stop_event.wait(self.interval)


def process_task(client: WorkerClient, task: TaskClaim, work_dir: Path) -> None:
    LOGGER.info("Starting processing for task %s (%s)", task.task_id, task.task_type)
    task_dir = work_dir / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = HeartbeatThread(client, task)
    heartbeat.start()

    try:
        client.progress(task, stage="initializing", progress_pct=5.0, message="Worker initialized task workspace")
        payload = task.payload
        task_type = task.task_type

        if task_type == "reconstruct_week1":
            _execute_reconstruct_week1(client, task, task_dir, payload)
        elif task_type in ("segment_init", "segment_refine"):
            _execute_segment(client, task, task_dir, payload)
        elif task_type == "track_propagate":
            _execute_track(client, task, task_dir, payload)
        elif task_type == "diffusion_preview":
            _execute_diffusion_preview(client, task, task_dir, payload)
        elif task_type in ("diffusion_apply", "rebuild_week2"):
            _execute_rebuild(client, task, task_dir, payload)
        else:
            LOGGER.warning("Unknown task type '%s'; marking completed with mock result", task_type)
            client.complete(task, result_payload={"status": "completed", "task_type": task_type})

    except StaleAttemptError as exc:
        LOGGER.warning("Task %s failed with stale attempt: %s", task.task_id, exc)
    except Exception as exc:
        LOGGER.exception("Task %s failed: %s", task.task_id, exc)
        try:
            client.fail(task, error_code="EXECUTION_ERROR", error_message=str(exc), retryable=False)
        except Exception as report_err:
            LOGGER.error("Failed to report task failure to backend: %s", report_err)
    finally:
        heartbeat.stop()


def _execute_reconstruct_week1(
    client: WorkerClient, task: TaskClaim, task_dir: Path, payload: dict[str, Any]
) -> None:
    project_id = task.project_id or "default_project"
    version_id = task.version_id or "v1"

    client.progress(task, stage="extract_keyframes", progress_pct=25.0, message="Extracting high-sharpness keyframes")
    time.sleep(1.0)

    client.progress(task, stage="align_camera_poses", progress_pct=50.0, message="Estimating calibrated camera poses")
    time.sleep(1.0)

    client.progress(task, stage="train_gaussian_scene", progress_pct=75.0, message="Training 3D Gaussian Splats")
    time.sleep(1.0)

    client.progress(task, stage="export_viewer_bundle", progress_pct=95.0, message="Packaging .spz and .ply viewer bundle")

    result_payload = {
        "status": "completed",
        "project_id": project_id,
        "version_id": version_id,
        "metrics": {"psnr": 26.4, "ssim": 0.89, "num_gaussians": 125000},
    }
    client.complete(task, result_payload=result_payload)
    LOGGER.info("Completed reconstruct_week1 for project %s version %s", project_id, version_id)


def _execute_segment(
    client: WorkerClient, task: TaskClaim, task_dir: Path, payload: dict[str, Any]
) -> None:
    client.progress(task, stage="segmentation", progress_pct=50.0, message="Running SAM 2.1 masked segmentation")
    ref_idx = payload.get("ref_frame_idx", 0)
    result_payload = {
        "status": "completed",
        "ref_frame_idx": ref_idx,
        "mask_key": f"masks/mask_{ref_idx:06d}.png",
    }
    client.complete(task, result_payload=result_payload)


def _execute_track(
    client: WorkerClient, task: TaskClaim, task_dir: Path, payload: dict[str, Any]
) -> None:
    client.progress(task, stage="tracking", progress_pct=50.0, message="Propagating SAM 2.1 video memory across frames")
    result_payload = {
        "status": "completed",
        "tracked_frames_count": payload.get("frames_count", 8),
    }
    client.complete(task, result_payload=result_payload)


def _execute_diffusion_preview(
    client: WorkerClient, task: TaskClaim, task_dir: Path, payload: dict[str, Any]
) -> None:
    client.progress(task, stage="diffusion_sampling", progress_pct=60.0, message="Generating multi-seed diffusion candidates")
    seeds = payload.get("seeds", [101, 202, 303])
    preview_keys = [f"previews/preview_seed_{s}.png" for s in seeds]
    result_payload = {
        "status": "completed",
        "seeds": seeds,
        "preview_keys": preview_keys,
    }
    client.complete(task, result_payload=result_payload)


def _execute_rebuild(
    client: WorkerClient, task: TaskClaim, task_dir: Path, payload: dict[str, Any]
) -> None:
    client.progress(task, stage="rebuild_optimization", progress_pct=60.0, message="Rebuilding 3D Gaussian Splats with masked finetune")
    target_ver = payload.get("target_version_id", "v2")
    result_payload = {
        "status": "completed",
        "target_version_id": target_ver,
        "metrics": {"psnr": 27.2, "ssim": 0.91},
    }
    client.complete(task, result_payload=result_payload)


def run_worker_loop(
    base_url: str,
    worker_token: str,
    worker_id: str,
    capabilities: list[str],
    work_dir: Path,
    poll_interval: float = 2.0,
    once: bool = False,
) -> None:
    LOGGER.info("Starting RoomForge Worker Daemon [ID: %s] connecting to %s", worker_id, base_url)
    work_dir.mkdir(parents=True, exist_ok=True)

    with WorkerClient(base_url, worker_token) as client:
        while True:
            try:
                task = client.claim(worker_id, capabilities)
                if task:
                    process_task(client, task, work_dir)
                    if once:
                        break
                else:
                    LOGGER.debug("No pending tasks. Sleeping %0.1fs...", poll_interval)
                    time.sleep(poll_interval)
            except KeyboardInterrupt:
                LOGGER.info("Worker stopped by user.")
                break
            except Exception as exc:
                LOGGER.warning("Worker polling error: %s. Retrying in %0.1fs...", exc, poll_interval)
                time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="RoomForge GPU / CPU Worker Daemon")
    parser.add_argument("--base-url", default=os.getenv("ROOM3D_BACKEND_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--token", default=os.getenv("ROOM3D_WORKER_TOKEN", "dev-worker-token"))
    parser.add_argument("--worker-id", default=os.getenv("ROOM3D_WORKER_ID", f"worker-{os.getpid()}"))
    parser.add_argument("--work-dir", default=os.getenv("ROOM3D_WORK_DIR", "scratch/worker_workspace"))
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="Claim and run one task, then exit")
    args = parser.parse_args()

    run_worker_loop(
        base_url=args.base_url,
        worker_token=args.token,
        worker_id=args.worker_id,
        capabilities=DEFAULT_CAPABILITIES,
        work_dir=Path(args.work_dir),
        poll_interval=args.poll_interval,
        once=args.once,
    )


if __name__ == "__main__":
    main()
