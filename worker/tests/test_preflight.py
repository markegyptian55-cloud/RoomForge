from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch
import pytest

from room3d_worker.kaggle.preflight_runner import (
    build_preflight_notebook,
    main,
    validate_notebook_syntax,
    write_kaggle_bundle,
)


def test_preflight_notebook_has_valid_schema_and_python_cells() -> None:
    notebook = build_preflight_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert len(notebook["cells"]) == 5
    assert all(cell["cell_type"] == "code" for cell in notebook["cells"])
    assert validate_notebook_syntax(notebook) == []


def test_write_kaggle_bundle_creates_notebook_and_metadata(tmp_path: Path) -> None:
    notebook_path, metadata_path = write_kaggle_bundle(
        tmp_path, "owner/room3d-preflight", ["owner/sample-dataset"]
    )

    assert notebook_path.name == "room3d_t4_preflight.ipynb"
    assert json.loads(notebook_path.read_text(encoding="utf-8"))["nbformat"] == 4
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["id"] == "owner/room3d-preflight"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["dataset_sources"] == ["owner/sample-dataset"]
    assert set(metadata) == {
        "id", "title", "code_file", "language", "kernel_type", "is_private",
        "enable_gpu", "enable_internet", "dataset_sources", "competition_sources", "kernel_sources",
    }


def test_slug_requires_owner_and_cell_probes_are_real() -> None:
    with pytest.raises(ValueError, match="kernel_slug"):
        write_kaggle_bundle(Path("."), "room3d-preflight")
    source = "\n".join("".join(cell["source"]) for cell in build_preflight_notebook()["cells"])
    assert "pycolmap.extract_features" in source
    assert "pycolmap.match_sequential" in source
    assert "pycolmap.global_mapping" in source
    assert "gsplat.rasterization" in source
    assert "render_mode=\"ED\"" in source
    assert "device=\"cuda:0\"" in source
    assert "torch.float16" in source


def test_cli_push_reports_kaggle_failure(tmp_path: Path, capsys: object) -> None:
    completed = subprocess.CompletedProcess(
        args=["kaggle"], returncode=2, stdout="", stderr="permission denied"
    )
    with patch("room3d_worker.kaggle.preflight_runner.subprocess.run", return_value=completed):
        result = main(["push", "--out-dir", str(tmp_path)])

    assert result == 2
    assert "permission denied" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_cli_push_uses_t4_accelerator(tmp_path: Path) -> None:
    completed = subprocess.CompletedProcess(args=["kaggle"], returncode=0, stdout="", stderr="")
    with patch("room3d_worker.kaggle.preflight_runner.subprocess.run", return_value=completed) as run:
        assert main(["push", "--out-dir", str(tmp_path)]) == 0
    assert run.call_args.args[0][-2:] == ["--accelerator", "NvidiaTeslaT4"]


def test_cli_status_reports_missing_kaggle(tmp_path: Path, capsys: object) -> None:
    with patch(
        "room3d_worker.kaggle.preflight_runner.subprocess.run",
        side_effect=FileNotFoundError("kaggle not found"),
    ):
        result = main(["status", "--out-dir", str(tmp_path)])

    assert result == 127
    assert "kaggle not found" in capsys.readouterr().err  # type: ignore[attr-defined]
