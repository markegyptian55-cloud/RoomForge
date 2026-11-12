from __future__ import annotations

import json
from pathlib import Path

import av
import cv2
import numpy as np
import pytest
from PIL import Image

from room3d_worker.client import StaleAttemptError, WorkerClient
from room3d_worker.pipeline.s01_extract import extract_keyframes


def _video(path: Path) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 322, 242, "yuv420p"
        for index in range(12):
            if index < 3:
                image = np.full((242, 322, 3), 80, np.uint8)
            elif index < 6:
                image = np.full((242, 322, 3), 80 + index, np.uint8)
            else:
                image = np.zeros((242, 322, 3), np.uint8)
                shift = (index - 6) * 24
                cv2.rectangle(image, (30 + shift, 30), (150 + shift, 180), (255, 120, 20), -1)
                cv2.line(image, (0, 240 - index * 8), (321, index * 8), (20, 240, 255), 4)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_extract_keyframes_drops_blurry_duplicates_and_writes_clean_index(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "frames"
    _video(source)

    result = extract_keyframes(source, output, max_long_side=640, min_parallax_frac=0.04)

    index = json.loads(result.index_path.read_text(encoding="utf-8"))
    assert index == result.index
    assert len(index["frames"]) >= 2
    assert all(item["orig_video_frame"] >= 6 for item in index["frames"])
    assert [item["frame_idx"] for item in index["frames"]] == list(range(len(index["frames"])))
    for item in index["frames"]:
        image_path = output / item["image_relpath"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        assert image.shape[1] % 16 == 0 and image.shape[0] % 16 == 0
        with av.open(str(image_path)) as container:
            stream = container.streams.video[0]
            assert not container.streams.audio
            assert not stream.metadata.get("exif")
            assert not stream.metadata.get("gps")
        assert not Image.open(image_path).getexif()


def test_worker_client_claim_heartbeat_and_stale_completion() -> None:
    responses = {
        "/api/tasks/claim": (200, {
            "task_id": "t1", "attempt_id": "a1", "task_type": "segment_refine", "payload": {},
        }),
        "/api/tasks/t1/heartbeat": (200, {"ok": True}),
        "/api/tasks/t1/complete": (409, {"detail": "stale attempt"}),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        status, payload = responses[request.url.path]
        return httpx.Response(status, json=payload)

    import httpx

    client = WorkerClient("https://backend.test", "secret", transport=httpx.MockTransport(handler))
    task = client.claim("worker-1", ["segment_refine"], interactive_only=True)
    assert task.task_id == "t1"
    assert client.heartbeat(task) == {"ok": True}
    with pytest.raises(StaleAttemptError):
        client.complete(task, result_payload={})

