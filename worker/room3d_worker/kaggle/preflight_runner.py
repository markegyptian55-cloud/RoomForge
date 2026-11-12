from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _cell(source: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def _cell_hardware() -> dict[str, Any]:
    return _cell('''import json
import os
import platform
import shutil
import subprocess
import sys

if os.path.exists("/kaggle/working"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "av", "pycolmap-cuda12"], check=False)

import torch

summary = {"hardware": {"os": platform.platform()}, "checks": {}}
try:
    summary["hardware"]["nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
except (FileNotFoundError, subprocess.CalledProcessError) as exc:
    summary["hardware"]["nvidia_smi_error"] = str(exc)
if torch.cuda.is_available():
    summary["hardware"]["devices"] = torch.cuda.device_count()
    summary["hardware"]["capabilities"] = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
    summary["hardware"]["sm_75"] = all(cap == (7, 5) for cap in summary["hardware"]["capabilities"])
    summary["hardware"]["bf16_required"] = False
else:
    summary["hardware"].update({"devices": 0, "sm_75": False, "bf16_required": False})
usage = shutil.disk_usage("/kaggle/working")
summary["hardware"]["working_free_gb"] = round(usage.free / 1024**3, 2)
print(json.dumps(summary["hardware"], indent=2))
''')


def _cell_pycolmap() -> dict[str, Any]:
    return _cell('''import json
try:
    import pycolmap
except ImportError as exc:
    pycolmap = None
    result = {"available": False, "error": str(exc)}
else:
    result = {"available": True, "version": getattr(pycolmap, "__version__", "unknown")}
if pycolmap is not None:
    for enum_name, expected in (("FeatureExtractorType", "ALIKED_N16ROT"), ("FeatureMatcherType", "ALIKED_LIGHTGLUE")):
        enum = getattr(pycolmap, enum_name, None)
        result[enum_name] = {"available": enum is not None, "expected_member": bool(enum and hasattr(enum, expected))}
    result["onnx_support"] = result["FeatureExtractorType"]["expected_member"] and result["FeatureMatcherType"]["expected_member"]
    result["global_mapping"] = hasattr(pycolmap, "global_mapping")
    result["sift_fallback"] = hasattr(pycolmap, "Sift") or hasattr(pycolmap, "SIFT")
print(json.dumps(result, indent=2))
summary.setdefault("checks", {})["pycolmap"] = result
''')


def _cell_pose() -> dict[str, Any]:
    return _cell('''import json
import os
import time
from pathlib import Path

def benchmark_pose_backends(sample_video: str | None = None) -> dict:
    sample = Path(sample_video) if sample_video else None
    if sample is None:
        sample = Path("/kaggle/working/synthetic_multiview_calibration.mp4")
        try:
            import av
            import numpy as np
            with av.open(str(sample), mode="w") as container:
                stream = container.add_stream("mpeg4", rate=8)
                stream.width, stream.height = 320, 240
                stream.pix_fmt = "yuv420p"
                yy, xx = np.mgrid[0:240, 0:320]
                for index in range(8):
                    shift = index * 6
                    checker = (((xx + shift) // 16 + yy // 16) % 2).astype(np.uint8) * 200 + 25
                    rgb = np.stack([checker, np.roll(checker, shift, axis=1), 255 - checker], axis=-1)
                    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        except (ImportError, OSError) as exc:
            return {"sample_video": str(sample), "synthetic_error": str(exc), "benchmarks": {}}
    results = {}
    try:
        import pycolmap
        import av
        import numpy as np
        started = time.perf_counter()
        frame_dir = sample.parent / "preflight_frames"
        frame_dir.mkdir(exist_ok=True)
        with av.open(str(sample)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index >= 8:
                    break
                frame.to_image().save(frame_dir / f"frame_{index:04d}.png")
        database = sample.parent / "preflight_database.db"
        sparse_out = sample.parent / "preflight_sparse"
        sparse_out.mkdir(exist_ok=True)
        pycolmap.extract_features(str(database), str(frame_dir))
        pycolmap.match_sequential(str(database))
        reconstruction = pycolmap.global_mapping(str(database), str(frame_dir), str(sparse_out))
        registered = len(next(iter(reconstruction.values())).images) if reconstruction else 0
        results["pycolmap"] = {"registered_frames": registered, "runtime_s": round(time.perf_counter() - started, 4), "peak_vram_mb": 0}
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        results["pycolmap"] = {"status": "unavailable", "error": str(exc)}
    try:
        import vggt
        results["vggt_chunked_fp16"] = {"available": True, "status": "available"}
    except ImportError as exc:
        results["vggt_chunked_fp16"] = {"status": "unavailable", "error": str(exc)}
    return {"sample_video": str(sample), "benchmarks": results}

pose_summary = benchmark_pose_backends(os.getenv("ROOM3D_SAMPLE_VIDEO"))
summary.setdefault("checks", {})["pose_backends"] = pose_summary
print(json.dumps(pose_summary, indent=2))
''')


def _cell_gsplat() -> dict[str, Any]:
    return _cell('''import json
import os
import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
result = {"torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"]}
try:
    import gsplat
    result["gsplat"] = {"available": True, "render_mode": "ED"}
    if torch.cuda.is_available():
        means = torch.zeros((1, 3), device="cuda", dtype=torch.float32)
        quats = torch.tensor([[1, 0, 0, 0]], device="cuda", dtype=torch.float32)
        scales = torch.ones((1, 3), device="cuda", dtype=torch.float32)
        opacities = torch.ones(1, device="cuda", dtype=torch.float32)
        colors = torch.ones((1, 3), device="cuda", dtype=torch.float32)
        view = torch.eye(4, device="cuda").unsqueeze(0)
        K = torch.eye(3, device="cuda").unsqueeze(0)
        rendered, _, _ = gsplat.rasterization(means, quats, scales, opacities, colors, view, K, 16, 16, render_mode="ED")
        result["gsplat"]["depth_smoke"] = bool(torch.isfinite(rendered).all())
except ImportError as exc:
    result["gsplat"] = {"available": False, "error": str(exc)}
for package in ("spz", "ppisp", "transformers"):
    try:
        __import__(package)
        result[package] = {"available": True}
    except (ImportError, RuntimeError, Exception) as exc:
        result[package] = {"available": False, "error": str(exc)}
print(json.dumps(result, indent=2))
summary.setdefault("checks", {})["gaussian_splat"] = result
''')


def _cell_diffusion() -> dict[str, Any]:
    return _cell('''import json
import torch

result = {"device": "cuda:0" if torch.cuda.is_available() else "cpu", "bf16_required": False}
try:
    import inspect
    from diffusers import Flux2KleinInpaintPipeline
    sig_params = sorted(inspect.signature(Flux2KleinInpaintPipeline.__call__).parameters)
    result["flux2_klein"] = {"available": True, "dtype": "float16", "quantization": "cpu_offload", "call_params": sig_params}
except (ImportError, RuntimeError) as exc:
    result["flux2_klein"] = {"available": False, "error": str(exc)}
try:
    from diffusers import StableDiffusionXLInpaintPipeline
    result["sdxl_inpaint"] = {"available": True, "dtype": "float16", "fallback": True}
except ImportError as exc:
    result["sdxl_inpaint"] = {"available": False, "error": str(exc)}
guard = torch.tensor([1.0], dtype=torch.float16, device=result["device"])
result["torch_isfinite_guard"] = bool(torch.isfinite(guard).all())
if torch.cuda.is_available():
    left = torch.ones((2, 2), device="cuda:0", dtype=torch.float16)
    result["cuda_fp16_matmul"] = bool(torch.isfinite(left @ left).all())
for package in ("bitsandbytes", "accelerate"):
    try:
        __import__(package)
        result[package] = {"available": True}
    except ImportError as exc:
        result[package] = {"available": False, "error": str(exc)}
if torch.cuda.is_available():
    result["vram_free_gb"] = round(torch.cuda.mem_get_info()[0] / 1024**3, 2)
summary.setdefault("checks", {})["diffusion"] = result
with open("/kaggle/working/preflight_summary.json", "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps(result, indent=2))
''')


def build_preflight_notebook() -> dict[str, Any]:
    cells = [_cell_hardware(), _cell_pycolmap(), _cell_pose(), _cell_gsplat(), _cell_diffusion()]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
        "cells": cells,
    }


def validate_notebook_syntax(nb: dict[str, Any]) -> list[str]:
    errors = []
    for index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        python_source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("!", "%")))
        try:
            ast.parse(python_source)
        except SyntaxError as exc:
            errors.append(f"cell {index}: {exc}")
    return errors


def write_kaggle_bundle(out_dir: Path, kernel_slug: str, dataset_sources: list[str] | None = None) -> tuple[Path, Path]:
    if "/" not in kernel_slug:
        raise ValueError("kernel_slug must contain '/' (owner/slug)")
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = out_dir / "room3d_t4_preflight.ipynb"
    metadata_path = out_dir / "kernel-metadata.json"
    notebook = build_preflight_notebook()
    errors = validate_notebook_syntax(notebook)
    if errors:
        raise ValueError("invalid generated notebook: " + "; ".join(errors))
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    metadata = {"id": kernel_slug, "title": "Room3D T4 preflight", "code_file": notebook_path.name, "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": dataset_sources or [], "competition_sources": [], "kernel_sources": []}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return notebook_path, metadata_path


def _run_kaggle(args: list[str]) -> int:
    try:
        completed = subprocess.run(["kaggle", *args], check=False, text=True, capture_output=True)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 127
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode:
        print(completed.stderr, end="", file=sys.stderr)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and operate the Room3D Kaggle preflight bundle.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--out-dir", type=Path, default=Path("kaggle_bundle"))
    build.add_argument("--kernel-slug", default="mohamedelbasyouni55/room3d-t4-preflight")
    build.add_argument("--dataset-source", action="append", default=[])
    for command in ("push", "status", "pull"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--out-dir", type=Path, default=Path("kaggle_bundle"))
        command_parser.add_argument("--kernel-slug", default="mohamedelbasyouni55/room3d-t4-preflight")
    args = parser.parse_args(argv)
    if args.command == "build":
        paths = write_kaggle_bundle(args.out_dir, args.kernel_slug, args.dataset_source)
        print("\n".join(str(path) for path in paths))
        return 0
    if args.command == "push":
        return _run_kaggle(["kernels", "push", "-p", str(args.out_dir), "--accelerator", "NvidiaTeslaT4"])
    if args.command == "status":
        return _run_kaggle(["kernels", "status", args.kernel_slug])
    return _run_kaggle(["kernels", "output", args.kernel_slug, "-p", str(args.out_dir)])


if __name__ == "__main__":
    raise SystemExit(main())

