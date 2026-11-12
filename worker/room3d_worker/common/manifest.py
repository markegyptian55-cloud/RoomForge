from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from room3d_worker.common.errors import ManifestValidationError

Matrix4 = Annotated[list[list[float]], Field(min_length=4, max_length=4)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Intrinsics(StrictModel):
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float = Field(ge=0)
    cy: float = Field(ge=0)


class FrameRecord(StrictModel):
    frame_idx: int = Field(ge=0)
    orig_video_frame: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)
    image_relpath: str = Field(min_length=1)
    colmap_image_id: int = Field(ge=1)
    cam_from_world: Matrix4
    excluded: bool = False

    @field_validator("cam_from_world")
    @classmethod
    def validate_pose(cls, matrix: list[list[float]]) -> list[list[float]]:
        return _validate_matrix(matrix)


class ArtifactBundle(StrictModel):
    sparse_dir: str
    ply_relpath: str | None = None
    spz_relpath: str = Field(min_length=1)
    sog_relpath: str | None = None
    sparse_points_relpath: str = Field(min_length=1)
    ckpt_relpath: str | None = None
    depth_dir: str | None = None
    collision_glb_relpath: str | None = None

    @field_validator("sparse_dir")
    @classmethod
    def validate_sparse_dir(cls, value: str) -> str:
        if value != "sparse/0":
            raise ValueError("sparse_dir must be sparse/0")
        return value


class SceneManifest(StrictModel):
    schema_version: str
    project_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    parent_version_id: str | None = None
    coord_convention: str
    T_viewer_from_colmap: Matrix4
    camera_model: str
    width: int = Field(ge=64, multiple_of=16)
    height: int = Field(ge=64, multiple_of=16)
    intrinsics: Intrinsics
    test_every: int = Field(ge=0)
    pose_backend: str | None = None
    frames: list[FrameRecord] = Field(min_length=1)
    artifacts: ArtifactBundle
    metrics: dict[str, Any] | None = None

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != "1.0.0":
            raise ValueError("schema_version must be 1.0.0")
        return value

    @field_validator("coord_convention")
    @classmethod
    def validate_convention(cls, value: str) -> str:
        if value != "opencv_colmap":
            raise ValueError("coord_convention must be opencv_colmap")
        return value

    @field_validator("camera_model")
    @classmethod
    def validate_camera_model(cls, value: str) -> str:
        if value != "PINHOLE":
            raise ValueError("camera_model must be PINHOLE")
        return value

    @field_validator("pose_backend")
    @classmethod
    def validate_backend(cls, value: str | None) -> str | None:
        allowed = {"colmap_global", "colmap_incremental", "vggt_chunked", "map_anything"}
        if value is not None and value not in allowed:
            raise ValueError(f"pose_backend must be one of {sorted(allowed)}")
        return value

    @field_validator("T_viewer_from_colmap")
    @classmethod
    def validate_transform(cls, matrix: list[list[float]]) -> list[list[float]]:
        return _validate_matrix(matrix)


def _validate_matrix(matrix: list[list[float]]) -> list[list[float]]:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("transform must be a 4x4 matrix")
    return matrix


def load_manifest(path: str | Path) -> SceneManifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return SceneManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ManifestValidationError(f"Could not load valid manifest at {path}: {exc}") from exc


def save_manifest(path: str | Path, manifest: SceneManifest) -> None:
    target = Path(path)
    temp_path: str | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            temp_path = handle.name
            handle.write(manifest.model_dump_json(indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    except (OSError, ValidationError) as exc:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
        raise ManifestValidationError(f"Could not atomically save manifest to {target}: {exc}") from exc
