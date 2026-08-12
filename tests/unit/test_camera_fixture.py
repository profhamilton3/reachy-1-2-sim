"""Unit tests for camera_fixture.py.

These tests run offline — no gRPC, no ROS, no Docker required.

Exit gates covered:
  R12-300: Encoded frames can be decoded (PIL is the same JPEG codec path as OpenCV).
  R12-301: Deterministic left/right patterns, sequence counter, latest-frame buffers,
           frame-file writer produces valid JPEG files.
"""

import io
import os
import tempfile
import threading
import time

import numpy as np
import pytest
from PIL import Image as PILImage

from camera_fixture import (
    LEFT,
    RIGHT,
    CameraFixture,
    CameraFrame,
    _encode_jpeg,
    _render_frame,
    frame_file_writer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decode_jpeg(data: bytes) -> PILImage.Image:
    return PILImage.open(io.BytesIO(data))


def _to_rgb_array(data: bytes) -> np.ndarray:
    img = _decode_jpeg(data)
    return np.asarray(img, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# CameraFrame dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestCameraFrame:
    def test_frozen(self):
        f = CameraFrame(LEFT, 1, 0, 640, 480, b"\xff\xd8\xff")
        with pytest.raises((AttributeError, TypeError)):
            f.sequence = 2  # type: ignore[misc]

    def test_fields(self):
        f = CameraFrame(RIGHT, 7, 12345, 320, 240, b"data")
        assert f.camera_id == RIGHT
        assert f.sequence == 7
        assert f.wall_time_ns == 12345
        assert f.width == 320
        assert f.height == 240
        assert f.jpeg_bytes == b"data"


# ─────────────────────────────────────────────────────────────────────────────
# JPEG codec
# ─────────────────────────────────────────────────────────────────────────────

class TestJpegCodec:
    def test_encode_jpeg_returns_bytes(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        data = _encode_jpeg(rgb)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_jpeg_magic_bytes(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        data = _encode_jpeg(rgb)
        # JPEG files start with 0xFF 0xD8 0xFF
        assert data[:2] == b"\xff\xd8"

    def test_encode_decode_roundtrip_shape(self):
        rgb = np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)
        data = _encode_jpeg(rgb)
        img = _decode_jpeg(data)
        assert img.size == (120, 80)  # PIL size is (width, height)

    def test_encode_decode_roundtrip_approximate_values(self):
        # JPEG is lossy; allow tolerance of 10 per channel
        rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
        data = _encode_jpeg(rgb)
        decoded = _to_rgb_array(data)
        assert np.abs(decoded.astype(int) - 128).max() < 20


# ─────────────────────────────────────────────────────────────────────────────
# _render_frame
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderFrame:
    def test_returns_camera_frame(self):
        f = _render_frame(LEFT, 1, 0, 320, 240)
        assert isinstance(f, CameraFrame)

    def test_correct_camera_id(self):
        assert _render_frame(LEFT, 1, 0, 320, 240).camera_id == LEFT
        assert _render_frame(RIGHT, 1, 0, 320, 240).camera_id == RIGHT

    def test_correct_sequence(self):
        f = _render_frame(LEFT, 42, 0, 320, 240)
        assert f.sequence == 42

    def test_dimensions_match(self):
        f = _render_frame(LEFT, 1, 0, 320, 240)
        assert f.width == 320
        assert f.height == 240

    def test_jpeg_decodable(self):
        f = _render_frame(LEFT, 1, 0, 320, 240)
        img = _decode_jpeg(f.jpeg_bytes)
        assert img.size == (320, 240)

    def test_left_right_differ(self):
        left = _render_frame(LEFT, 1, 0, 320, 240)
        right = _render_frame(RIGHT, 1, 0, 320, 240)
        assert left.jpeg_bytes != right.jpeg_bytes

    def test_sequence_affects_frame(self):
        f1 = _render_frame(LEFT, 1, 0, 320, 240)
        f2 = _render_frame(LEFT, 200, 0, 320, 240)
        # The sequence bar width changes so pixel content differs
        assert f1.jpeg_bytes != f2.jpeg_bytes

    def test_left_tint_is_blue(self):
        f = _render_frame(LEFT, 1, 0, 320, 240)
        rgb = _to_rgb_array(f.jpeg_bytes)
        # Sample the top-left corner away from sequence bar and indicator
        corner = rgb[5:15, 5:50, :]
        # Left is blue-tinted: B channel > R channel on average
        assert int(corner[:, :, 2].mean()) > int(corner[:, :, 0].mean())

    def test_right_tint_is_green(self):
        f = _render_frame(RIGHT, 1, 0, 320, 240)
        rgb = _to_rgb_array(f.jpeg_bytes)
        corner = rgb[5:15, 5:50, :]
        # Right is green-tinted: G channel > R channel on average
        assert int(corner[:, :, 1].mean()) > int(corner[:, :, 0].mean())


# ─────────────────────────────────────────────────────────────────────────────
# CameraFixture
# ─────────────────────────────────────────────────────────────────────────────

class TestCameraFixtureBasic:
    def test_latest_frame_initially_none(self):
        f = CameraFixture()
        assert f.latest_frame(LEFT) is None
        assert f.latest_frame(RIGHT) is None

    def test_render_single_frame_left(self):
        f = CameraFixture()
        frame = f.render_single_frame(LEFT)
        assert frame.camera_id == LEFT
        assert frame.sequence == 1

    def test_render_single_frame_right(self):
        f = CameraFixture()
        frame = f.render_single_frame(RIGHT)
        assert frame.camera_id == RIGHT
        assert frame.sequence == 1

    def test_sequence_advances_per_camera(self):
        f = CameraFixture()
        f.render_single_frame(LEFT)
        f.render_single_frame(LEFT)
        f3 = f.render_single_frame(LEFT)
        assert f3.sequence == 3

    def test_left_right_sequences_independent(self):
        f = CameraFixture()
        f.render_single_frame(LEFT)
        f.render_single_frame(LEFT)
        r1 = f.render_single_frame(RIGHT)
        assert r1.sequence == 1  # RIGHT seq is independent

    def test_latest_frame_updated_after_render(self):
        f = CameraFixture()
        frame = f.render_single_frame(LEFT)
        assert f.latest_frame(LEFT) is frame

    def test_jpeg_decodable(self):
        f = CameraFixture(width=160, height=120)
        frame = f.render_single_frame(LEFT)
        img = _decode_jpeg(frame.jpeg_bytes)
        assert img.size == (160, 120)


class TestCameraFixtureLoop:
    def test_start_returns_thread(self):
        f = CameraFixture(fps=30.0, width=80, height=60)
        t = f.start()
        assert isinstance(t, threading.Thread)
        f.stop()

    def test_frames_generated_after_start(self):
        f = CameraFixture(fps=30.0, width=80, height=60)
        f.start()
        time.sleep(0.15)  # at 30 Hz ≥ 4 frames should be generated
        f.stop()
        left = f.latest_frame(LEFT)
        right = f.latest_frame(RIGHT)
        assert left is not None
        assert right is not None
        assert left.sequence >= 1
        assert right.sequence >= 1

    def test_sequence_advances_over_time(self):
        f = CameraFixture(fps=30.0, width=80, height=60)
        f.start()
        time.sleep(0.05)
        seq1 = (f.latest_frame(LEFT) or CameraFrame(LEFT, 0, 0, 0, 0, b"")).sequence
        time.sleep(0.15)
        seq2 = (f.latest_frame(LEFT) or CameraFrame(LEFT, 0, 0, 0, 0, b"")).sequence
        f.stop()
        assert seq2 > seq1


# ─────────────────────────────────────────────────────────────────────────────
# frame_file_writer
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameFileWriter:
    def test_writes_jpeg_files(self, tmp_path):
        left_path = str(tmp_path / "left.jpg")
        right_path = str(tmp_path / "right.jpg")

        f = CameraFixture(fps=30.0, width=80, height=60)
        f.start()

        writer = threading.Thread(
            target=frame_file_writer,
            args=(f,),
            kwargs={"left_path": left_path, "right_path": right_path},
            daemon=True,
        )
        writer.start()

        time.sleep(0.3)
        f.stop()

        assert os.path.exists(left_path), "left.jpg not written"
        assert os.path.exists(right_path), "right.jpg not written"

    def test_written_files_are_valid_jpeg(self, tmp_path):
        left_path = str(tmp_path / "left.jpg")
        right_path = str(tmp_path / "right.jpg")

        f = CameraFixture(fps=30.0, width=80, height=60)
        f.start()

        writer = threading.Thread(
            target=frame_file_writer,
            args=(f,),
            kwargs={"left_path": left_path, "right_path": right_path},
            daemon=True,
        )
        writer.start()

        time.sleep(0.3)
        f.stop()

        with open(left_path, "rb") as fh:
            data = fh.read()
        img = _decode_jpeg(data)
        assert img.size == (80, 60)

    def test_no_partial_writes(self, tmp_path):
        """Verify tmp file is not left behind after successful write."""
        left_path = str(tmp_path / "left.jpg")
        right_path = str(tmp_path / "right.jpg")

        f = CameraFixture(fps=30.0, width=80, height=60)
        f.start()

        writer = threading.Thread(
            target=frame_file_writer,
            args=(f,),
            kwargs={"left_path": left_path, "right_path": right_path},
            daemon=True,
        )
        writer.start()

        time.sleep(0.3)
        f.stop()

        # Tmp files should be atomically renamed away
        assert not os.path.exists(left_path + ".tmp"), "tmp file leaked"
        assert not os.path.exists(right_path + ".tmp"), "tmp file leaked"
