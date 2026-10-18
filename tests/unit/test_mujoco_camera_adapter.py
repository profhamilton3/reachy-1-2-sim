"""mujoco-remote camera delivery: one gRPC frame per rendered image, and room
on the gRPC server for more than two reachy-sdk clients.

Skipped where reachy_sdk_api / grpc are not installed.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

pytest.importorskip("reachy_sdk_api")
grpc = pytest.importorskip("grpc")

from reachy_sdk_api import camera_reachy_pb2, camera_reachy_pb2_grpc  # noqa: E402

import fake_reachy_server  # noqa: E402
from camera_fixture import LEFT, RIGHT, CameraFixture  # noqa: E402
from fake_reachy_server import FakeCameraService, _MujocoRemoteCameraAdapter  # noqa: E402
from mujoco_remote_backend import _atomic_write  # noqa: E402

# reachy-sdk 0.7.0 holds StreamJointsState, StreamSensorStates,
# StreamJointsCommands and one StreamImage per camera for its whole connection.
STREAMS_PER_SDK_CLIENT = 5


@pytest.fixture
def frame_files(tmp_path, monkeypatch):
    left, right = tmp_path / "reachy_left.jpg", tmp_path / "reachy_right.jpg"
    monkeypatch.setattr(fake_reachy_server, "_MUJOCO_LEFT_JPG", left)
    monkeypatch.setattr(fake_reachy_server, "_MUJOCO_RIGHT_JPG", right)
    return left, right


def test_an_unchanged_file_is_the_same_frame(frame_files):
    left, _ = frame_files
    adapter = _MujocoRemoteCameraAdapter()
    assert adapter.latest_frame(LEFT) is None  # nothing rendered yet
    _atomic_write(left, b"render-1")
    first = adapter.latest_frame(LEFT)
    assert [adapter.latest_frame(LEFT) for _ in range(5)] == [first] * 5
    assert first.sequence == 1 and first.jpeg_bytes == b"render-1"


def test_each_bridge_write_is_a_new_frame_even_with_identical_bytes(frame_files):
    left, right = frame_files
    adapter = _MujocoRemoteCameraAdapter()
    _atomic_write(left, b"same")
    _atomic_write(right, b"other-eye")
    assert adapter.latest_frame(LEFT).sequence == 1
    _atomic_write(left, b"same")  # a static scene renders identical JPEGs
    assert adapter.latest_frame(LEFT).sequence == 2
    assert adapter.latest_frame(RIGHT).sequence == 1  # per-camera sequences


def test_a_missing_file_keeps_the_last_frame(frame_files):
    left, _ = frame_files
    adapter = _MujocoRemoteCameraAdapter()
    _atomic_write(left, b"render-1")
    first = adapter.latest_frame(LEFT)
    left.unlink()
    assert adapter.latest_frame(LEFT) is first


def test_stream_image_sends_each_render_once(frame_files):
    left, _ = frame_files
    _atomic_write(left, b"render-1")
    service = FakeCameraService(_MujocoRemoteCameraAdapter())
    request = camera_reachy_pb2.StreamImageRequest(
        request=camera_reachy_pb2.ImageRequest(
            camera=camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.LEFT)))
    ticks = [0]

    def is_active():
        ticks[0] += 1
        if ticks[0] == 4:
            _atomic_write(left, b"render-2")  # one new render during six ticks
        return ticks[0] <= 6

    context = MagicMock()
    context.is_active.side_effect = is_active
    sent = [m.data for m in service.StreamImage(request, context)]
    assert sent == [b"render-1", b"render-2"]


def test_three_sdk_clients_streaming_still_leave_room_for_a_unary_call():
    fixture = CameraFixture(fps=30.0, width=80, height=60)
    fixture.start()
    server = grpc.server(ThreadPoolExecutor(max_workers=fake_reachy_server._GRPC_MAX_WORKERS))
    camera_reachy_pb2_grpc.add_CameraServiceServicer_to_server(FakeCameraService(fixture), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = camera_reachy_pb2_grpc.CameraServiceStub(channel)
    request = camera_reachy_pb2.ImageRequest(
        camera=camera_reachy_pb2.Camera(id=camera_reachy_pb2.CameraId.RIGHT))
    streams, started = [], []

    def first_frame(stream):
        try:
            next(stream)  # its handler now holds a server thread
            started.append(stream)
        except grpc.RpcError:
            pass

    try:
        for _ in range(3 * STREAMS_PER_SDK_CLIENT):
            # The deadline only bounds the test; streams must all be served at once.
            stream = stub.StreamImage(camera_reachy_pb2.StreamImageRequest(request=request), timeout=30)
            streams.append(stream)
            threading.Thread(target=first_frame, args=(stream,), daemon=True).start()
        deadline = time.monotonic() + 5
        while len(started) < len(streams) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(started) == len(streams), f"only {len(started)} of {len(streams)} streams were served"
        assert stub.GetImage(request, timeout=3).data
    finally:
        for stream in streams:
            stream.cancel()
        channel.close()
        server.stop(0)
