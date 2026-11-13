from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO


class StorageDriver(ABC):
    @abstractmethod
    def put_stream(self, key: str, stream: BinaryIO, chunk_size: int = 1024 * 1024) -> int: ...

    @abstractmethod
    def open_read(self, key: str) -> BinaryIO: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int: ...

    @abstractmethod
    def safe_resolve(self, key: str) -> Path: ...


class BoundedStream:
    def __init__(self, source: BinaryIO, limit: int) -> None:
        self.source, self.limit, self.total = source, limit, 0

    def read(self, size: int = -1) -> bytes:
        remaining = self.limit + 1 - self.total
        chunk = self.source.read(min(size, remaining) if size >= 0 else remaining)
        self.total += len(chunk)
        if self.total > self.limit:
            raise ValueError("upload exceeds MAX_UPLOAD_BYTES")
        return chunk


def validate_key(key: str) -> str:
    from urllib.parse import unquote

    decoded = key
    for _ in range(10):
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    if not decoded or "\x00" in decoded or decoded.startswith(("/", "\\")):
        raise ValueError("storage key must be relative")
    if len(decoded) >= 2 and decoded[1] == ":":
        raise ValueError("storage key cannot contain a drive prefix")
    parts = decoded.replace("\\", "/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("storage key contains an unsafe path component")
    return "/".join(parts)
