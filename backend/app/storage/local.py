from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO

from .base import StorageDriver, validate_key


class LocalStorage(StorageDriver):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def safe_resolve(self, key: str) -> Path:
        candidate = (self.root / validate_key(key)).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("storage path escapes storage root")
        return candidate

    def put_stream(self, key: str, stream: BinaryIO, chunk_size: int = 1024 * 1024) -> int:
        target = self.safe_resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with target.open("wb") as output:
            while chunk := stream.read(chunk_size):
                output.write(chunk)
                total += len(chunk)
        return total

    def open_read(self, key: str) -> BinaryIO:
        return self.safe_resolve(key).open("rb")

    def exists(self, key: str) -> bool:
        return self.safe_resolve(key).is_file()

    def delete_prefix(self, prefix: str) -> int:
        target = self.safe_resolve(prefix)
        if not target.exists():
            return 0
        if target.is_file():
            target.unlink()
            return 1
        count = sum(1 for path in target.rglob("*") if path.is_file())
        shutil.rmtree(target)
        return count


LocalObjectStore = LocalStorage
