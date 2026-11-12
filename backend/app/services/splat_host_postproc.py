from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def find_splat_transform() -> str | None:
    for name in ("splat-transform", "splat-transform.cmd"):
        found = shutil.which(name)
        if found:
            return found
    return None


def postprocess_spz(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    executable = find_splat_transform()
    if not executable:
        return {"available": False, "reason": "splat-transform executable not found"}
    source = Path(input_path)
    target = Path(output_path) if output_path else source
    command = [executable, str(source), str(target)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, shell=False)
    if completed.returncode:
        return {"available": True, "ok": False, "returncode": completed.returncode,
                "reason": completed.stderr.strip() or "splat-transform failed"}
    return {"available": True, "ok": True, "output": str(target)}
