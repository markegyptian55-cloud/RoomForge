from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import Settings  # noqa: E402
from app.services.celery_bridge import dispatch_task_if_configured  # noqa: E402


def test_celery_bridge_respects_db_pull_and_validates_redis_config() -> None:
    receipt = dispatch_task_if_configured("task-1", "rebuild_week2", Settings(queue_backend="db_pull"))
    assert receipt.dispatched is False
    assert receipt.backend == "db_pull"

    with pytest.raises(RuntimeError, match="redis_url"):
        dispatch_task_if_configured("task-1", "rebuild_week2", Settings(queue_backend="celery_redis", redis_url=None))

    with pytest.raises(ValueError, match="redis://"):
        dispatch_task_if_configured("task-1", "rebuild_week2", Settings(queue_backend="celery_redis", redis_url="http://bad"))
