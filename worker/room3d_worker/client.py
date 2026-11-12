from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

import httpx

from room3d_worker.common.subprocess_runner import StageResult, run_isolated_stage


class WorkerClientError(RuntimeError):
    """The backend rejected or could not complete a worker operation."""


class StaleAttemptError(WorkerClientError):
    """The task attempt fence no longer belongs to this worker."""


@dataclass(frozen=True)
class TaskClaim:
    task_id: str
    attempt_id: str
    project_id: str | None
    version_id: str | None
    task_type: str
    payload: dict[str, Any]
    worker_id: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TaskClaim:
        return cls(
            payload["task_id"], payload["attempt_id"], payload.get("project_id"),
            payload.get("version_id"), payload["task_type"], payload.get("payload", {}),
        )


class WorkerClient:
    BATCH_GPU = {
        "diffusion_preview": "cuda:0", "diffusion_apply": "cuda:0",
        "reconstruct_week1": "cuda:1", "pose_benchmark": "cuda:1",
        "track_propagate": "cuda:1", "rebuild_week2": "cuda:1",
    }
    INTERACTIVE_TASKS = {"segment_init", "segment_refine"}

    def __init__(
        self, base_url: str, worker_token: str, *,
        transport: httpx.BaseTransport | None = None, timeout: float = 30.0,
    ) -> None:
        headers = {"X-Worker-Token": worker_token, "Accept": "application/json"}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> WorkerClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._http.request(method, path, **kwargs)
        if response.status_code == 409:
            raise StaleAttemptError(f"Backend rejected stale attempt for {path}")
        if response.status_code >= 400:
            raise WorkerClientError(
                f"Backend {method} {path} failed with {response.status_code}: {response.text}"
            )
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    def claim(
        self, worker_id: str, capabilities: list[str], *, interactive_only: bool = False,
    ) -> TaskClaim | None:
        response = self._http.post(
            "/api/tasks/claim",
            json={
                "worker_id": worker_id, "capabilities": capabilities,
                "interactive_only": interactive_only,
            },
        )
        if response.status_code == 204:
            return None
        if response.status_code >= 400:
            raise WorkerClientError(
                f"Task claim failed with {response.status_code}: {response.text}"
            )
        task = TaskClaim.from_payload(response.json())
        return TaskClaim(
            task.task_id, task.attempt_id, task.project_id, task.version_id,
            task.task_type, task.payload, worker_id,
        )

    def heartbeat(self, task: TaskClaim, worker_id: str | None = None) -> dict[str, Any]:
        payload = {"attempt_id": task.attempt_id, "worker_id": worker_id or task.worker_id or ""}
        return self._request("POST", f"/api/tasks/{task.task_id}/heartbeat", json=payload)

    def progress(
        self, task: TaskClaim, stage: str, progress_pct: float, message: str,
        telemetry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "attempt_id": task.attempt_id, "stage": stage,
            "progress_pct": progress_pct, "message": message,
        }
        if telemetry is not None:
            payload["telemetry"] = telemetry
        return self._request("POST", f"/api/tasks/{task.task_id}/progress", json=payload)

    def complete(
        self, task: TaskClaim, result_payload: dict[str, Any],
        result_manifest_key: str | None = None,
    ) -> dict[str, Any]:
        payload = {"attempt_id": task.attempt_id, "result_payload": result_payload}
        if result_manifest_key is not None:
            payload["result_manifest_key"] = result_manifest_key
        return self._request("POST", f"/api/tasks/{task.task_id}/complete", json=payload)

    def fail(
        self, task: TaskClaim, error_code: str, error_message: str, retryable: bool,
    ) -> dict[str, Any]:
        payload = {
            "attempt_id": task.attempt_id, "error_code": error_code,
            "error_message": error_message, "retryable": retryable,
        }
        return self._request("POST", f"/api/tasks/{task.task_id}/fail", json=payload)

    def upload(
        self, project_id: str, version_id: str, path: str, source: BinaryIO | Path,
    ) -> dict[str, Any]:
        handle = source.open("rb") if isinstance(source, Path) else source
        try:
            return self._request(
                "POST", f"/api/storage/{project_id}/{version_id}/upload",
                files={"file": (path, handle, "application/octet-stream")},
            )
        finally:
            if isinstance(source, Path):
                handle.close()

    def download(self, project_id: str, version_id: str, path: str, destination: Path) -> None:
        with self._http.stream("GET", f"/api/storage/{project_id}/{version_id}/{path}") as response:
            if response.status_code >= 400:
                raise WorkerClientError(f"Storage download failed with {response.status_code}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)

    @staticmethod
    def _chunks(handle: BinaryIO, chunk_size: int) -> Iterator[bytes]:
        while chunk := handle.read(chunk_size):
            yield chunk

    def dispatch_batch(
        self, task_type: str, command: list[str], *, timeout_sec: int,
        env_extra: dict[str, str] | None = None,
    ) -> StageResult:
        try:
            device = self.BATCH_GPU[task_type]
        except KeyError as exc:
            raise WorkerClientError(f"Unsupported batch task type: {task_type}") from exc
        return run_isolated_stage(command, device, env_extra, timeout_sec)

    def run_interactive_loop(
        self, worker_id: str, handler: Callable[[TaskClaim], dict[str, Any]],
        stop_event: threading.Event | None = None,
    ) -> None:
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            task = self.claim(worker_id, ["segment_init", "segment_refine"], interactive_only=True)
            if task is None:
                stop_event.wait(0.5)
                continue
            try:
                self.complete(task, handler(task))
            except StaleAttemptError:
                continue
            except Exception as exc:
                self.fail(task, type(exc).__name__, str(exc), retryable=True)
