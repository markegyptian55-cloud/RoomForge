from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ..config import Settings


@dataclass(frozen=True)
class DispatchReceipt:
    backend: str
    dispatched: bool
    task_id: str
    task_type: str
    external_id: str | None = None


@lru_cache(maxsize=4)
def _cached_celery_app(redis_url: str) -> Any:
    if not redis_url.startswith(("redis://", "rediss://")):
        raise ValueError("redis_url must start with redis:// or rediss://")
    try:
        celery_mod = importlib.import_module("celery")
    except ImportError as exc:
        raise RuntimeError("celery package is required when queue_backend='celery_redis'") from exc
    return celery_mod.Celery("room3d", broker=redis_url, backend=redis_url)


def dispatch_task_if_configured(
    task_id: str,
    task_type: str,
    settings: Settings,
    payload: dict[str, Any] | None = None,
) -> DispatchReceipt:
    """Optional Celery/Redis fan-out; local dev uses pull polling over HTTP without Redis."""
    if settings.queue_backend == "db_pull":
        return DispatchReceipt(backend="db_pull", dispatched=False, task_id=task_id, task_type=task_type)
    if settings.queue_backend != "celery_redis":
        raise ValueError(f"Unsupported queue_backend: {settings.queue_backend}")
    if not settings.redis_url:
        raise RuntimeError("redis_url must be set when queue_backend='celery_redis'")
    app = _cached_celery_app(settings.redis_url)
    async_result = app.send_task(
        "room3d.worker.execute",
        kwargs={"task_id": task_id, "task_type": task_type, "payload": payload or {}},
    )
    return DispatchReceipt(
        backend="celery_redis",
        dispatched=True,
        task_id=task_id,
        task_type=task_type,
        external_id=str(getattr(async_result, "id", "") or ""),
    )
