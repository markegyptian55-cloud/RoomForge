from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import av
import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


@dataclass(frozen=True)
class ExtractionResult:
    output_dir: Path
    index_path: Path
    index: dict[str, object]


def _frame_timestamp(frame: av.VideoFrame, index: int, rate: float) -> float:
    if frame.pts is not None and frame.time_base is not None:
        return float(frame.pts * frame.time_base)
    return index / rate if rate > 0 else 0.0


def _sharpness(gray: Image) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _parallax(previous: Image, current: Image) -> float:
    points = cv2.goodFeaturesToTrack(previous, maxCorners=160, qualityLevel=0.01, minDistance=8)
    if points is None or len(points) < 4:
        return 0.0
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if next_points is None or status is None:
        return 0.0
    valid = status.reshape(-1).astype(bool)
    if valid.sum() < 4:
        return 0.0
    displacement = np.linalg.norm(next_points[valid] - points[valid], axis=1)
    return float(np.median(displacement) / min(previous.shape[:2]))


def _prepare(image: Image, max_long_side: int) -> tuple[Image, int, int]:
    height, width = image.shape[:2]
    scale = min(1.0, max_long_side / max(width, height))
    if scale < 1.0:
        width, height = round(width * scale), round(height * scale)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    width -= width % 16
    height -= height % 16
    if width < 16 or height < 16:
        raise ValueError("Video frames must be at least 16 pixels in both dimensions")
    x0, y0 = (image.shape[1] - width) // 2, (image.shape[0] - height) // 2
    return image[y0:y0 + height, x0:x0 + width], width, height


def _write_jpeg(path: Path, image: Image) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise OSError(f"Could not encode JPEG {path}")
    path.write_bytes(encoded.tobytes())


def extract_keyframes(
    video: str | os.PathLike[str] | BinaryIO,
    output_dir: str | os.PathLike[str],
    *,
    max_long_side: int = 960,
    min_parallax_frac: float = 0.04,
    sharpness_window: int = 7,
) -> ExtractionResult:
    if not 640 <= max_long_side <= 960:
        raise ValueError("max_long_side must be between 640 and 960")
    if min_parallax_frac < 0 or sharpness_window < 1:
        raise ValueError("min_parallax_frac and sharpness_window must be non-negative/positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    sharpnesses: list[float] = []
    previous_gray: Image | None = None
    canonical_w: int = 0
    canonical_h: int = 0
    max_decode_frames = 10_000
    with av.open(video) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or 0.0)
        for original_index, frame in enumerate(container.decode(video=0)):
            if original_index >= max_decode_frames:
                break
            image = frame.to_ndarray(format="bgr24")
            prepared, width, height = _prepare(image, max_long_side)
            gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
            score = _sharpness(gray)
            sharpnesses.append(score)
            baseline = float(np.median(sharpnesses[-sharpness_window:]))
            if score < max(12.0, baseline * 0.35):
                continue
            motion = 1.0 if previous_gray is None else _parallax(previous_gray, gray)
            if previous_gray is not None and motion < min_parallax_frac:
                continue
            if not records:
                canonical_w, canonical_h = width, height
            frame_index = len(records)
            relative = Path("images") / f"frame_{frame_index:06d}.jpg"
            target = destination / relative
            target.parent.mkdir(exist_ok=True)
            _write_jpeg(target, prepared)
            records.append({
                "frame_idx": frame_index,
                "orig_video_frame": original_index,
                "timestamp_sec": _frame_timestamp(frame, original_index, rate),
                "image_relpath": relative.as_posix(),
            })
            previous_gray = gray
    if not records:
        raise ValueError("No sharp, sufficiently parallax-separated keyframes were found")
    index: dict[str, object] = {"frames": records, "width": canonical_w, "height": canonical_h}
    fd, temporary = tempfile.mkstemp(prefix="extract_index.", suffix=".json", dir=destination)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        index_path = destination / "extract_index.json"
        os.replace(temporary, index_path)
    except OSError:
        Path(temporary).unlink(missing_ok=True)
        raise
    return ExtractionResult(destination, index_path, index)
