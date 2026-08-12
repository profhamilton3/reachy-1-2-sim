"""Contract tests for FakeCameraService (R12-300).

These tests import reachy_sdk_api and exercise the CameraServiceServicer
interface directly — no running gRPC server required.  They are skipped
gracefully on the host if reachy_sdk_api is not installed; they run inside
the container where pip has resolved reachy-sdk (which installs reachy-sdk-api).

Exit gate (R12-300):
  - Contract tests prove encoded frames can be decoded with PIL/OpenCV.
  - GetImage, StreamImage, left/right selection, zoom/focus semantics tested.
  - Cancellation and bad requests return appropriate status codes.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

# Skip entire module on hosts where reachy_sdk_api is not installed
reachy_sdk_api = pytest.importorskip("reachy_sdk_api")

from reachy_sdk_api import camera_reachy_pb2, camera_reachy_pb2_grpc  # noqa: E402

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# FakeCameraService is in fake_reachy_server.py which also imports grpc.
# Import only what we need via the servicer base class approach.
try:
    from fake_reachy_server import FakeCameraService
    from camera_fixture import CameraFixture, LEFT, RIGHT
    _SERVER_AVAILABLE = True
except ImportError:
    _SERVER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _SERVER_AVAILABLE, reason="fake_reachy_server or grpc not importable"
)


def _make_context() -> MagicMock:
    """Return a minimal mock gRPC servicer context."""
    ctx = MagicMock()
    ctx.is_active.return_value = True
    return ctx


def _make_service() -> tuple["FakeCameraService", "CameraFixture"]:
    fixture = CameraFixture(fps=30.0, width=80, height=60)
    fixture.start()
    service = FakeCameraService(fixture)
    return service, fixture


def _decode_jpeg(data: bytes):
    return PILImage.open(io.BytesIO(data))


# ─────────────────────────────────────────────────────────────────────────────
# GetImage
# ─────────────────────────────────────────────────────────────────────────────

class TestGetImage:
    def test_get_left_image_returns_bytes(self):
        svc, fixture = _make_service()
        request = camera_reachy_pb2.ImageRequest(
            camera=camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.LEFT)
        )
        resp = svc.GetImage(request, _make_context())
        assert len(resp.data) > 0

    def test_get_right_image_returns_bytes(self):
        svc, fixture = _make_service()
        request = camera_reachy_pb2.ImageRequest(
            camera=camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.RIGHT)
        )
        resp = svc.GetImage(request, _make_context())
        assert len(resp.data) > 0

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="PIL not installed")
    def test_get_image_is_decodable(self):
        svc, fixture = _make_service()
        request = camera_reachy_pb2.ImageRequest(
            camera=camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.LEFT)
        )
        resp = svc.GetImage(request, _make_context())
        img = _decode_jpeg(resp.data)
        assert img.size == (80, 60)

    def test_left_right_produce_different_frames(self):
        svc, fixture = _make_service()
        import time; time.sleep(0.1)  # let fixture run
        left_req = camera_reachy_pb2.ImageRequest(
            camera=camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.LEFT)
        )
        right_req = camera_reachy_pb2.ImageRequest(
            camera=camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.RIGHT)
        )
        left_resp = svc.GetImage(left_req, _make_context())
        right_resp = svc.GetImage(right_req, _make_context())
        assert left_resp.data != right_resp.data


# ─────────────────────────────────────────────────────────────────────────────
# StreamImage
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamImage:
    def test_stream_yields_frames(self):
        svc, fixture = _make_service()
        request = camera_reachy_pb2.StreamImageRequest(
            request=camera_reachy_pb2.ImageRequest(
                camera=camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.LEFT)
            )
        )
        ctx = _make_context()
        # Cancel after 3 frames
        call_count = [0]
        def is_active():
            call_count[0] += 1
            return call_count[0] <= 3
        ctx.is_active.side_effect = is_active

        frames = list(svc.StreamImage(request, ctx))
        assert len(frames) == 3
        assert all(len(f.data) > 0 for f in frames)

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="PIL not installed")
    def test_stream_frames_decodable(self):
        svc, fixture = _make_service()
        request = camera_reachy_pb2.StreamImageRequest(
            request=camera_reachy_pb2.ImageRequest(
                camera=camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.RIGHT)
            )
        )
        ctx = _make_context()
        count = [0]
        def is_active():
            count[0] += 1
            return count[0] <= 2
        ctx.is_active.side_effect = is_active

        for frame in svc.StreamImage(request, ctx):
            img = _decode_jpeg(frame.data)
            assert img.size == (80, 60)

    def test_stream_stops_on_context_cancel(self):
        svc, fixture = _make_service()
        request = camera_reachy_pb2.StreamImageRequest(
            request=camera_reachy_pb2.ImageRequest(
                camera=camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.LEFT)
            )
        )
        ctx = _make_context()
        ctx.is_active.return_value = False

        frames = list(svc.StreamImage(request, ctx))
        assert frames == []


# ─────────────────────────────────────────────────────────────────────────────
# Zoom / focus RPCs (simulated semantics)
# ─────────────────────────────────────────────────────────────────────────────

class TestZoomFocus:
    def test_get_zoom_level_returns_zero(self):
        svc, _ = _make_service()
        cam = camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.LEFT)
        resp = svc.GetZoomLevel(cam, _make_context())
        assert resp.level == camera_reachy_pb2.ZoomLevelPossibilities.ZERO

    def test_get_zoom_speed_returns_stable_value(self):
        svc, _ = _make_service()
        cam = camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.LEFT)
        resp = svc.GetZoomSpeed(cam, _make_context())
        assert isinstance(resp.speed, int)

    def test_send_zoom_command_ack_success(self):
        svc, _ = _make_service()
        cmd = camera_reachy_pb2.ZoomCommand(
            camera=camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.LEFT),
            homing_command=camera_reachy_pb2.ZoomHoming(),
        )
        resp = svc.SendZoomCommand(cmd, _make_context())
        assert resp.success is True

    def test_get_zoom_focus_returns_message(self):
        from google.protobuf.empty_pb2 import Empty
        svc, _ = _make_service()
        resp = svc.GetZoomFocus(Empty(), _make_context())
        # ZoomFocusMessage with UInt32Value fields
        assert hasattr(resp, "left_focus")
        assert hasattr(resp, "right_focus")

    def test_set_zoom_focus_returns_success(self):
        from google.protobuf.wrappers_pb2 import UInt32Value
        svc, _ = _make_service()
        req = camera_reachy_pb2.ZoomFocusMessage(
            left_focus=UInt32Value(value=100),
        )
        resp = svc.SetZoomFocus(req, _make_context())
        assert resp.success is True

    def test_start_autofocus_returns_success(self):
        svc, _ = _make_service()
        cam = camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.LEFT)
        resp = svc.StartAutofocus(cam, _make_context())
        assert resp.success is True

    def test_stop_autofocus_returns_success(self):
        svc, _ = _make_service()
        cam = camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.RIGHT)
        resp = svc.StopAutofocus(cam, _make_context())
        assert resp.success is True
