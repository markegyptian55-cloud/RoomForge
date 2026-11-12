from __future__ import annotations

from typing import TYPE_CHECKING

from .base import StorageDriver
from .local import LocalObjectStore, LocalStorage
from .s3 import S3Storage

if TYPE_CHECKING:
    from ..config import Settings


def get_storage(settings: Settings) -> StorageDriver:
    if settings.storage_backend == "s3":
        if not settings.s3_bucket:
            raise ValueError("S3_BUCKET is required when storage_backend='s3'")
        kwargs: dict[str, object] = {}
        if settings.s3_endpoint_url:
            kwargs["endpoint_url"] = settings.s3_endpoint_url
        if settings.s3_region:
            kwargs["region_name"] = settings.s3_region
        if settings.s3_access_key:
            kwargs["aws_access_key_id"] = settings.s3_access_key
        if settings.s3_secret_key:
            kwargs["aws_secret_access_key"] = settings.s3_secret_key
        return S3Storage(settings.s3_bucket, **kwargs)
    return LocalStorage(settings.local_storage_dir)


__all__ = ["LocalObjectStore", "LocalStorage", "S3Storage", "StorageDriver", "get_storage"]
