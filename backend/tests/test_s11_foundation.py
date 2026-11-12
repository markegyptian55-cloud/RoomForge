from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
import sys

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import Base
from app.models import Project, User
from app.security import create_access_token, hash_password, require_worker_token, verify_access_token, verify_password
from app.storage import LocalStorage, S3Storage, get_storage


def test_storage_rejects_traversal_and_streams(tmp_path) -> None:
    storage = LocalStorage(tmp_path / "objects")
    for key in ("../secret", "%2e%2e/secret", "C:\\secret", "/absolute", "a/..\\secret"):
        with pytest.raises(ValueError):
            storage.safe_resolve(key)
    payload = b"abc" * 10000
    assert storage.put_stream("project/video.bin", BytesIO(payload), chunk_size=7) == len(payload)
    with storage.open_read("project/video.bin") as stream:
        assert stream.read() == payload


def test_password_and_jwt_tamper_detection() -> None:
    encoded = hash_password("correct horse")
    assert verify_password("correct horse", encoded)
    assert not verify_password("wrong horse", encoded)
    token = create_access_token("user-1")
    assert verify_access_token(token)["sub"] == "user-1"
    with pytest.raises(Exception):
        verify_access_token(token[:-1] + ("a" if token[-1] != "a" else "b"))


def test_sqlite_crud_and_schema(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'room3d.db'}")
    Base.metadata.create_all(engine)
    assert {"users", "projects", "tasks", "task_attempts"}.issubset(inspect(engine).get_table_names())
    with Session(engine) as session:
        user = User(email="a@example.test", password_hash="hash")
        session.add(user)
        session.flush()
        session.add(Project(owner_id=user.id, name="Room", status="queued"))
        session.commit()
        assert session.query(Project).one().owner_id == user.id


def test_storage_factory_and_s3_counting(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    local = get_storage(Settings(storage_backend="local", local_storage_dir=str(tmp_path / "store")))
    assert isinstance(local, LocalStorage)
    uploaded: dict[str, object] = {}

    class FakeClient:
        def upload_fileobj(self, fileobj, bucket: str, key: str) -> None:
            uploaded["data"] = fileobj.read()
            uploaded["bucket"] = bucket
            uploaded["key"] = key

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *_a, **_kw: FakeClient()))
    s3 = get_storage(Settings(storage_backend="s3", s3_bucket="room3d-test"))
    assert isinstance(s3, S3Storage)
    written = s3.put_stream("p1/raw.mp4", BytesIO(b"123456789"))
    assert written == 9
    assert uploaded == {"data": b"123456789", "bucket": "room3d-test", "key": "p1/raw.mp4"}


def test_worker_token_and_nullable_owner_fk(tmp_path) -> None:
    require_worker_token("dev-worker-token")
    with pytest.raises(HTTPException):
        require_worker_token("wrong")
    engine = create_engine(f"sqlite:///{tmp_path / 'fk.db'}")

    @event.listens_for(engine, "connect")
    def _fk(conn: object, _: object) -> None:
        conn.cursor().execute("PRAGMA foreign_keys=ON")  # type: ignore[attr-defined]

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Project(owner_id=None, name="DevAnonRoom", status="queued"))
        session.commit()
        session.add(Project(owner_id="missing-user", name="BadFK", status="queued"))
        with pytest.raises(IntegrityError):
            session.commit()
