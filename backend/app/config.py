from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_DB = (_REPO_ROOT / "data" / "room3d.db").resolve()
_CANONICAL_STORAGE = (_REPO_ROOT / "data" / "storage").resolve()


class Settings(BaseSettings):
    database_url: str = f"sqlite:///{_CANONICAL_DB.as_posix()}"
    storage_backend: str = "local"
    local_storage_dir: str = str(_CANONICAL_STORAGE)
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_region: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    worker_token: str = "dev-worker-token"
    jwt_secret: str = Field(default="dev-jwt-secret-change-me", min_length=16)
    auth_required: bool = False
    max_upload_bytes: int = 500 * 1024 * 1024
    queue_backend: Literal["db_pull", "celery_redis"] = "db_pull"
    redis_url: str | None = None
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
