"""Fixture camera backend for Reachy 1.2 simulator.

Pure Python — no gRPC, no ROS, no OpenCV.  Importable on any host for testing.

Generates deterministic left/right JPEG test patterns that embed camera id,
sequence number, and timestamp.  Runs an internal loop at a configurable rate
and stores the latest frame for each camera in an atomic reference.

`frame_file_writer()` writes JPEG frames to files via atomic rename so that
external readers (ROS node, web server) never see a partial frame.

Frame content (deliberately simple for fixture use):
  LEFT  — blue-tinted 640×480 with horizontal gradient stripe
  RIGHT — green-tinted 640×480 with horizontal gradient stripe
  Both  — include a 4-pixel-wide sequence counter bar (darkens as seq grows)
         so tests can verify different frames.
"""

from __future__ import annotations

import io
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from PIL import Image as _PILImage

_STEP_HZ_DEFAULT: float = 15.0
_WIDTH_DEFAULT: int = 640
_HEIGHT_DEFAULT: int = 480
_JPEG_QUALITY: int = 85

LEFT: int = 0
RIGHT: int = 1

_LEFT_FILE: str = "/tmp/reachy_left.jpg"
_RIGHT_FILE: str = "/tmp/reachy_right.jpg"


@dataclass(frozen=True)
class CameraFrame:
    camera_id: int             # LEFT (0) or RIGHT (1)
    sequence: int              # monotonically increasing per camera
    wall_time_ns: int          # time.time_ns() at capture
    width: int
    height: int
    jpeg_bytes: bytes          # JPEG-encoded RGB frame


def _encode_jpeg(rgb_array: np.ndarray, quality: int = _JPEG_QUALITY) -> bytes:
    img = _PILImage.fromarray(rgb_array, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _render_frame(
    camera_id: int,
    sequence: int,
    wall_time_ns: int,
    width: int,
    height: int,
) -> CameraFrame:
    """Render one deterministic fixture frame (RGB numpy array → JPEG)."""
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    # Base tint per camera: LEFT=blue, RIGHT=green
    if camera_id == LEFT:
        rgb[:, :] = [60, 80, 200]   # R, G, B in RGB
    else:
        rgb[:, :] = [60, 190, 80]

    # Horizontal gradient stripe in the middle third of the frame
    stripe_top = height // 3
    stripe_bot = 2 * height // 3
    gradient = np.linspace(30, 230, width, dtype=np.uint8)
    rgb[stripe_top:stripe_bot, :, 0] = gradient          # R channel
    rgb[stripe_top:stripe_bot, :, 1] = 255 - gradient    # G channel
    rgb[stripe_top:stripe_bot, :, 2] = 128               # B channel

    # Sequence counter bar at top-left: 4px tall, width proportional to seq mod width
    bar_width = (sequence % width) + 1
    rgb[0:4, 0:bar_width, :] = [255, 255, 0]  # yellow

    # Camera-id indicator: a small red (LEFT) or cyan (RIGHT) rectangle top-right
    indicator_w = 40
    if camera_id == LEFT:
        rgb[0:20, width - indicator_w:, :] = [220, 40, 40]
    else:
        rgb[0:20, width - indicator_w:, :] = [40, 220, 220]

    jpeg = _encode_jpeg(rgb)
    return CameraFrame(
        camera_id=camera_id,
        sequence=sequence,
        wall_time_ns=wall_time_ns,
        width=width,
        height=height,
        jpeg_bytes=jpeg,
    )


class CameraFixture:
    """Deterministic JPEG frame generator for left and right cameras.

    Usage::

        fixture = CameraFixture(fps=15.0, width=640, height=480)
        fixture.start()
        frame = fixture.latest_frame(LEFT)
        # frame is a CameraFrame with .jpeg_bytes, .sequence, etc.
    """

    def __init__(
        self,
        fps: float = _STEP_HZ_DEFAULT,
        width: int = _WIDTH_DEFAULT,
        height: int = _HEIGHT_DEFAULT,
    ) -> None:
        self._fps = float(fps)
        self._width = int(width)
        self._height = int(height)

        self._sequences: Dict[int, int] = {LEFT: 0, RIGHT: 0}
        self._frames: Dict[int, Optional[CameraFrame]] = {LEFT: None, RIGHT: None}
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> threading.Thread:
        """Start the frame-generation loop in a daemon thread."""
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True, name="camera-fixture")
        t.start()
        return t

    def stop(self) -> None:
        self._running = False

    def latest_frame(self, camera_id: int) -> Optional[CameraFrame]:
        """Return the most recently generated frame for *camera_id*, or None."""
        with self._lock:
            return self._frames[camera_id]

    def render_single_frame(self, camera_id: int) -> CameraFrame:
        """Render one frame on demand (blocking, synchronous).  Advances sequence."""
        with self._lock:
            seq = self._sequences[camera_id] + 1
            self._sequences[camera_id] = seq
        frame = _render_frame(camera_id, seq, time.time_ns(), self._width, self._height)
        with self._lock:
            self._frames[camera_id] = frame
        return frame

    def _loop(self) -> None:
        dt = 1.0 / self._fps
        while self._running:
            t0 = time.monotonic()
            self._step()
            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, dt - elapsed)
            if sleep_for > 0.0:
                time.sleep(sleep_for)

    def _step(self) -> None:
        wall_ns = time.time_ns()
        new_frames: Dict[int, CameraFrame] = {}
        seqs: Dict[int, int] = {}
        with self._lock:
            for cid in (LEFT, RIGHT):
                seqs[cid] = self._sequences[cid] + 1
                self._sequences[cid] = seqs[cid]

        for cid in (LEFT, RIGHT):
            new_frames[cid] = _render_frame(cid, seqs[cid], wall_ns, self._width, self._height)

        with self._lock:
            self._frames.update(new_frames)


def frame_file_writer(
    fixture: CameraFixture,
    left_path: Optional[str] = None,
    right_path: Optional[str] = None,
) -> None:
    """Write left/right JPEG frames to files via atomic rename at ~15 Hz.

    Runs forever (designed to be a daemon thread).  Each write is:
        open(tmp) → write → os.replace(tmp, dest)
    which is POSIX-atomic and prevents external readers from seeing partial data.
    """
    left_dest = left_path if left_path is not None else _LEFT_FILE
    right_dest = right_path if right_path is not None else _RIGHT_FILE
    left_tmp = left_dest + ".tmp"
    right_tmp = right_dest + ".tmp"

    dt = 1.0 / 15.0
    last_seqs = {LEFT: -1, RIGHT: -1}

    while True:
        for cam_id, dest, tmp in (
            (LEFT, left_dest, left_tmp),
            (RIGHT, right_dest, right_tmp),
        ):
            frame = fixture.latest_frame(cam_id)
            if frame is not None and frame.sequence != last_seqs[cam_id]:
                try:
                    with open(tmp, "wb") as f:
                        f.write(frame.jpeg_bytes)
                    os.replace(tmp, dest)
                except Exception:
                    pass  # transient filesystem error; next tick will retry
                last_seqs[cam_id] = frame.sequence

        time.sleep(dt)
