from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./data/room3d.db"
    storage_backend: str = "local"
    local_storage_dir: str = "./data/storage"
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_region: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    worker_token: str = "dev-worker-token"
    jwt_secret: str = Field(default="dev-jwt-secret-change-me", min_length=16)
    auth_required: bool = False
    max_upload_bytes: int = 500 * 1024 * 1024

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
