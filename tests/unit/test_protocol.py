"""Unit tests for native_mujoco/protocol.py (R12-400)."""

import json
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../native_mujoco"))

from protocol import (
    PROTOCOL_VERSION,
    CameraFrame,
    Disconnect,
    Error,
    Hello,
    HelloAck,
    Heartbeat,
    HeartbeatAck,
    JointCommand,
    Pause,
    Reset,
    SceneAck,
    SceneLoad,
    Shutdown,
    State,
    decode,
    encode,
    message_type,
    validate_image_payload,
    validate_joint_command,
)


class TestEncodeDecodeRoundtrip:
    def test_hello(self):
        h = Hello(client_id="pytest")
        d = decode(h.encode())
        assert d["type"] == "hello"
        assert d["protocol_version"] == PROTOCOL_VERSION
        assert d["client_id"] == "pytest"

    def test_hello_ack(self):
        a = HelloAck(sim_fps=500.0, camera_fps=15.0, num_joints=21)
        d = decode(a.encode())
        assert d["type"] == "hello_ack"
        assert d["num_joints"] == 21
        assert d["sim_fps"] == 500.0

    def test_joint_command(self):
        tgt = [0.1 * i for i in range(21)]
        jc = JointCommand(seq=42, target_rad=tgt)
        d = decode(jc.encode())
        assert d["type"] == "joint_command"
        assert d["seq"] == 42
        assert len(d["target_rad"]) == 21
        assert d["target_rad"][5] == pytest.approx(0.5)

    def test_pause(self):
        p = Pause(paused=True)
        d = decode(p.encode())
        assert d["type"] == "pause"
        assert d["paused"] is True

    def test_reset(self):
        r = Reset(seed=42, request_id="r1")
        d = decode(r.encode())
        assert d["type"] == "reset"
        assert d["seed"] == 42
        assert d["request_id"] == "r1"

    def test_heartbeat(self):
        h = Heartbeat()
        d = decode(h.encode())
        assert d["type"] == "heartbeat"
        assert isinstance(d["sent_ns"], int)
        assert d["sent_ns"] > 0

    def test_heartbeat_ack(self):
        ha = HeartbeatAck(echo_ns=12345)
        d = decode(ha.encode())
        assert d["type"] == "heartbeat_ack"
        assert d["echo_ns"] == 12345

    def test_state(self):
        s = State(seq=7, sim_step=350, sim_time_s=0.7, paused=False)
        d = decode(s.encode())
        assert d["type"] == "state"
        assert d["seq"] == 7
        assert d["sim_time_s"] == pytest.approx(0.7)

    def test_camera_frame(self):
        cf = CameraFrame(camera="left_camera", seq=3, width=640, height=480,
                         jpeg_b64="abc123", render_us=1500)
        d = decode(cf.encode())
        assert d["type"] == "camera_frame"
        assert d["camera"] == "left_camera"
        assert d["jpeg_b64"] == "abc123"
        assert d["render_us"] == 1500

    def test_scene_ack(self):
        sa = SceneAck(request_id="r1", accepted=True, scene_revision="abc12",
                      warnings=["w1"])
        d = decode(sa.encode())
        assert d["type"] == "scene_ack"
        assert d["accepted"] is True
        assert d["warnings"] == ["w1"]

    def test_error(self):
        e = Error(code="bad_cmd", message="oops")
        d = decode(e.encode())
        assert d["type"] == "error"
        assert d["code"] == "bad_cmd"

    def test_shutdown(self):
        s = Shutdown(reason="test_exit")
        d = decode(s.encode())
        assert d["type"] == "shutdown"
        assert d["reason"] == "test_exit"

    def test_disconnect(self):
        dc = Disconnect(reason="done")
        d = decode(dc.encode())
        assert d["type"] == "disconnect"


class TestMessageType:
    def test_hello(self):
        assert message_type(Hello().encode()) == "hello"

    def test_state(self):
        assert message_type(State().encode()) == "state"

    def test_unknown(self):
        assert message_type('{"type":"custom"}') == "custom"

    def test_missing_type(self):
        assert message_type('{}') == ""


class TestValidateJointCommand:
    def test_valid(self):
        msg = {"type": "joint_command", "seq": 1, "target_rad": [0.0] * 21}
        validate_joint_command(msg, 21)  # should not raise

    def test_wrong_length(self):
        msg = {"type": "joint_command", "seq": 1, "target_rad": [0.0] * 10}
        with pytest.raises(ValueError, match="length"):
            validate_joint_command(msg, 21)

    def test_not_a_list(self):
        msg = {"type": "joint_command", "seq": 1, "target_rad": 0.0}
        with pytest.raises(ValueError, match="list"):
            validate_joint_command(msg, 21)

    def test_mask_wrong_length(self):
        msg = {
            "type": "joint_command", "seq": 1,
            "target_rad": [0.0] * 21,
            "mask": [True] * 10,
        }
        with pytest.raises(ValueError, match="mask"):
            validate_joint_command(msg, 21)

    def test_mask_none_ok(self):
        msg = {
            "type": "joint_command", "seq": 1,
            "target_rad": [0.0] * 21,
            "mask": None,
        }
        validate_joint_command(msg, 21)  # should not raise


class TestValidateImagePayload:
    def test_small_payload_ok(self):
        import base64
        data = base64.b64encode(b"x" * 1000).decode("ascii")
        validate_image_payload(data)  # should not raise

    def test_oversized_payload_raises(self):
        import base64
        # ~6 MB of base64 → exceeds 4 MB limit
        data = base64.b64encode(b"x" * (5 * 1024 * 1024)).decode("ascii")
        with pytest.raises(ValueError, match="limit"):
            validate_image_payload(data)


class TestSceneLoadSizeLimit:
    def test_small_ok(self):
        sl = SceneLoad(scene_document={"a": 1}, request_id="r1")
        sl.encode()  # should not raise

    def test_oversized_raises(self):
        import string
        sl = SceneLoad(
            scene_document={"data": "x" * 300_000},
            request_id="r1"
        )
        with pytest.raises(ValueError, match="limit"):
            sl.encode()
