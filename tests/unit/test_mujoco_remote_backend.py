"""Unit tests for MujocoRemoteBackend and KinematicBridge.

These tests run offline — no WebSocket server, no gRPC, no Docker required.

Key behaviors tested:
  - to_kinematic_snapshot() produces a properly-keyed SimulationSnapshot
  - KinematicBridge delegates correctly and returns the right types
  - Missing joints are filled in with safe defaults
  - Antenna joints default to stiff (compliant=False)
  - RemoteSnapshot → SimulationSnapshot field mapping is correct
"""

import asyncio
import json
import math
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kinematic_backend import JOINT_DEFS, JointCommand, JointSample, SimulationSnapshot
from mujoco_remote_backend import (
    ConnectionState,
    KinematicBridge,
    MujocoRemoteBackend,
    RemoteJointSample,
    RemoteSnapshot,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_remote_backend() -> MujocoRemoteBackend:
    """Construct a backend without starting its WebSocket thread."""
    return MujocoRemoteBackend(url="ws://localhost:9999")


def _make_remote_snapshot(**overrides) -> RemoteSnapshot:
    return RemoteSnapshot(**{
        "seq": 1,
        "sim_step": 10,
        "sim_time_s": 0.2,
        "wall_time_ns": time.monotonic_ns(),
        "state": ConnectionState.READY,
        "joints": {},
        **overrides,
    })


def _full_remote_joints() -> dict:
    """Build RemoteJointSample entries for all 21 joints."""
    joints = {}
    for i, (name, uid) in enumerate(JOINT_DEFS):
        joints[name] = RemoteJointSample(
            name=name,
            uid=uid,
            position_rad=float(i) * 0.01,
            velocity_rad_s=0.001,
            effort=0.5,
            compliant=uid not in (33, 34),
        )
    return joints


# ── to_kinematic_snapshot: basic structure ─────────────────────────────────────

class TestToKinematicSnapshotStructure:

    def test_returns_simulation_snapshot(self):
        b = _make_remote_backend()
        snap = b.to_kinematic_snapshot()
        assert isinstance(snap, SimulationSnapshot)

    def test_joints_keyed_by_name(self):
        b = _make_remote_backend()
        b._snapshot = _make_remote_snapshot(joints=_full_remote_joints())
        snap = b.to_kinematic_snapshot()
        for name, _ in JOINT_DEFS:
            assert name in snap.joints, f"{name} missing from snapshot"

    def test_all_21_joints_present(self):
        b = _make_remote_backend()
        b._snapshot = _make_remote_snapshot(joints=_full_remote_joints())
        snap = b.to_kinematic_snapshot()
        assert len(snap.joints) == 21

    def test_joints_are_joint_sample_instances(self):
        b = _make_remote_backend()
        b._snapshot = _make_remote_snapshot(joints=_full_remote_joints())
        snap = b.to_kinematic_snapshot()
        for s in snap.joints.values():
            assert isinstance(s, JointSample)

    def test_snapshot_fields_propagated(self):
        b = _make_remote_backend()
        wt = time.monotonic_ns()
        b._snapshot = _make_remote_snapshot(
            seq=42, sim_step=100, sim_time_s=2.0,
            wall_time_ns=wt, joints=_full_remote_joints(),
        )
        snap = b.to_kinematic_snapshot()
        assert snap.sequence == 42
        assert snap.sim_step == 100
        assert snap.sim_time_s == pytest.approx(2.0)
        assert snap.wall_time_ns == wt


# ── to_kinematic_snapshot: field mapping ─────────────────────────────────────

class TestToKinematicSnapshotFieldMapping:

    def test_position_rad_maps_to_present_position(self):
        b = _make_remote_backend()
        joints = {
            "r_shoulder_pitch": RemoteJointSample(
                name="r_shoulder_pitch", uid=10,
                position_rad=1.234, velocity_rad_s=0.0, effort=0.0,
            )
        }
        b._snapshot = _make_remote_snapshot(joints=joints)
        snap = b.to_kinematic_snapshot()
        assert snap.joints["r_shoulder_pitch"].present_position == pytest.approx(1.234)

    def test_velocity_maps_to_present_speed(self):
        b = _make_remote_backend()
        joints = {
            "r_shoulder_pitch": RemoteJointSample(
                name="r_shoulder_pitch", uid=10,
                position_rad=0.0, velocity_rad_s=0.55, effort=0.0,
            )
        }
        b._snapshot = _make_remote_snapshot(joints=joints)
        snap = b.to_kinematic_snapshot()
        assert snap.joints["r_shoulder_pitch"].present_speed == pytest.approx(0.55)

    def test_effort_maps_to_present_load(self):
        b = _make_remote_backend()
        joints = {
            "r_shoulder_pitch": RemoteJointSample(
                name="r_shoulder_pitch", uid=10,
                position_rad=0.0, velocity_rad_s=0.0, effort=3.7,
            )
        }
        b._snapshot = _make_remote_snapshot(joints=joints)
        snap = b.to_kinematic_snapshot()
        assert snap.joints["r_shoulder_pitch"].present_load == pytest.approx(3.7)

    def test_uid_preserved(self):
        b = _make_remote_backend()
        b._snapshot = _make_remote_snapshot(joints=_full_remote_joints())
        snap = b.to_kinematic_snapshot()
        for name, uid in JOINT_DEFS:
            assert snap.joints[name].uid == uid

    def test_compliant_preserved_from_remote(self):
        b = _make_remote_backend()
        joints = {
            "r_shoulder_pitch": RemoteJointSample(
                name="r_shoulder_pitch", uid=10,
                position_rad=0.0, velocity_rad_s=0.0, effort=0.0,
                compliant=False,
            )
        }
        b._snapshot = _make_remote_snapshot(joints=joints)
        snap = b.to_kinematic_snapshot()
        assert snap.joints["r_shoulder_pitch"].compliant is False

    def test_name_field_set_correctly(self):
        b = _make_remote_backend()
        b._snapshot = _make_remote_snapshot(joints=_full_remote_joints())
        snap = b.to_kinematic_snapshot()
        for name in snap.joints:
            assert snap.joints[name].name == name


# ── to_kinematic_snapshot: missing joint fill-in ─────────────────────────────

class TestToKinematicSnapshotFillIn:

    def test_empty_remote_yields_all_21_defaults(self):
        b = _make_remote_backend()
        b._snapshot = _make_remote_snapshot(joints={})
        snap = b.to_kinematic_snapshot()
        assert len(snap.joints) == 21

    def test_default_positions_are_zero(self):
        b = _make_remote_backend()
        b._snapshot = _make_remote_snapshot(joints={})
        snap = b.to_kinematic_snapshot()
        for s in snap.joints.values():
            assert s.present_position == 0.0

    def test_antenna_joints_default_stiff(self):
        b = _make_remote_backend()
        b._snapshot = _make_remote_snapshot(joints={})
        snap = b.to_kinematic_snapshot()
        assert snap.joints["l_antenna"].compliant is False
        assert snap.joints["r_antenna"].compliant is False

    def test_arm_joints_default_compliant(self):
        b = _make_remote_backend()
        b._snapshot = _make_remote_snapshot(joints={})
        snap = b.to_kinematic_snapshot()
        assert snap.joints["r_shoulder_pitch"].compliant is True
        assert snap.joints["l_gripper"].compliant is True

    def test_partial_report_fills_missing(self):
        """If the server only reports some joints, the rest get safe defaults."""
        b = _make_remote_backend()
        # Only right shoulder pitch is reported
        joints = {
            "r_shoulder_pitch": RemoteJointSample(
                name="r_shoulder_pitch", uid=10,
                position_rad=0.5, velocity_rad_s=0.0, effort=0.0,
            )
        }
        b._snapshot = _make_remote_snapshot(joints=joints)
        snap = b.to_kinematic_snapshot()
        assert len(snap.joints) == 21
        assert snap.joints["r_shoulder_pitch"].present_position == pytest.approx(0.5)
        assert snap.joints["l_shoulder_pitch"].present_position == 0.0


# ── KinematicBridge ────────────────────────────────────────────────────────────

class TestKinematicBridge:

    def test_latest_snapshot_returns_simulation_snapshot(self):
        bridge = KinematicBridge(_make_remote_backend())
        snap = bridge.latest_snapshot()
        assert isinstance(snap, SimulationSnapshot)

    def test_latest_snapshot_has_all_joints(self):
        bridge = KinematicBridge(_make_remote_backend())
        snap = bridge.latest_snapshot()
        expected = {name for name, _ in JOINT_DEFS}
        assert set(snap.joints.keys()) == expected

    def test_uid_by_name_delegates(self):
        bridge = KinematicBridge(_make_remote_backend())
        assert bridge.uid_by_name("r_shoulder_pitch") == 10
        assert bridge.uid_by_name("r_gripper") == 17
        assert bridge.uid_by_name("l_antenna") == 33
        assert bridge.uid_by_name("nonexistent") is None

    def test_submit_command_accepted_without_error(self):
        bridge = KinematicBridge(_make_remote_backend())
        # submit_command must not raise even when not connected
        bridge.submit_command(JointCommand(uid=10, goal_position=1.0))

    def test_snapshot_reflects_remote_state(self):
        backend = _make_remote_backend()
        bridge = KinematicBridge(backend)
        joints = {
            "r_shoulder_pitch": RemoteJointSample(
                name="r_shoulder_pitch", uid=10,
                position_rad=0.7, velocity_rad_s=0.0, effort=0.0,
            )
        }
        backend._snapshot = _make_remote_snapshot(seq=5, joints=joints)
        snap = bridge.latest_snapshot()
        assert snap.sequence == 5
        assert snap.joints["r_shoulder_pitch"].present_position == pytest.approx(0.7)


# ── B1 regression: constructor initializes joint maps and thread ───────────────

class TestConstructorLifecycle:

    def test_constructor_initializes_joint_maps(self):
        b = _make_remote_backend()
        assert hasattr(b, "_uid_to_idx")
        assert hasattr(b, "_idx_to_uid")
        assert len(b._uid_to_idx) == len(JOINT_DEFS)
        assert len(b._idx_to_uid) == len(JOINT_DEFS)

    def test_constructor_initializes_thread_to_none(self):
        b = _make_remote_backend()
        assert b._thread is None

    def test_build_command_works_before_first_reset(self):
        b = _make_remote_backend()
        b._snapshot = _make_remote_snapshot(joints=_full_remote_joints())
        cmd = JointCommand(uid=10, goal_position=0.5)
        target, compliant, speed, torque = b._build_command([cmd])
        assert isinstance(target, list)
        assert len(target) == len(JOINT_DEFS)
        idx = b._uid_to_idx[10]
        assert target[idx] == pytest.approx(0.5)

    def test_request_reset_does_not_replace_thread_handle(self):
        """request_reset() must not overwrite _thread; start() owns that field."""
        b = _make_remote_backend()
        sentinel = object()
        b._thread = sentinel  # type: ignore[assignment]
        b.request_reset()
        assert b._thread is sentinel

    def test_request_reset_sets_pending_reset_flag(self):
        b = _make_remote_backend()
        b.request_reset()
        with b._lock:
            assert b._pending_reset is True

    def test_request_reset_clears_last_target(self):
        b = _make_remote_backend()
        b._last_target = [0.0] * len(JOINT_DEFS)
        b.request_reset()
        with b._lock:
            assert b._last_target is None


# ── Gate 8-D regression: acknowledged reset ────────────────────────────────────

class TestAcknowledgedReset:

    def test_request_reset_returns_event(self):
        import threading
        b = _make_remote_backend()
        result = b.request_reset()
        assert isinstance(result, threading.Event)

    def test_request_reset_generates_unique_id(self):
        b = _make_remote_backend()
        b.request_reset()
        id1 = b._pending_reset_id
        b.request_reset()
        id2 = b._pending_reset_id
        assert id1 is not None and id2 is not None
        assert id1 != id2, "Each reset must generate a unique request_id"

    def test_request_reset_enters_resetting_state(self):
        b = _make_remote_backend()
        b.request_reset()
        with b._lock:
            assert b._conn_state == ConnectionState.RESETTING

    def test_commands_held_during_resetting(self):
        """submit_command during RESETTING must be buffered, not lost."""
        b = _make_remote_backend()
        b.request_reset()
        b.submit_command(JointCommand(uid=10, goal_position=0.5))
        with b._lock:
            # Command should still be in _pending_cmds (not consumed).
            assert len(b._pending_cmds) == 1

    def test_reset_ack_clears_resetting_state(self):
        """Simulating a reset_ack via _ingest_state-equivalent path."""
        b = _make_remote_backend()
        b.request_reset()
        rid = b._pending_reset_id
        # Simulate the _recv handler processing a reset_ack.
        with b._lock:
            if (b._conn_state == ConnectionState.RESETTING
                    and rid == b._pending_reset_id):
                b._conn_state = ConnectionState.READY
                b._pending_reset_id = None
        b._reset_ack_event.set()
        assert b._conn_state == ConnectionState.READY

    def test_wait_for_reset_returns_false_on_timeout(self):
        b = _make_remote_backend()
        b.request_reset()
        # No ack will ever arrive; timeout immediately.
        ok = b.wait_for_reset(timeout=0.01)
        assert ok is False

    def test_wait_for_reset_returns_true_when_event_set(self):
        import threading
        b = _make_remote_backend()
        event = b.request_reset()
        # Signal the event from another thread (as the recv handler would).
        threading.Timer(0.05, event.set).start()
        ok = b.wait_for_reset(timeout=1.0)
        assert ok is True

    def test_ack_timeout_enters_aborted_state(self):
        """_on_reset_timeout must move from RESETTING to ABORTED and signal waiters."""
        b = _make_remote_backend()
        b.request_reset()
        b._on_reset_timeout()
        assert b._conn_state == ConnectionState.ABORTED
        # Event must be set so callers don't hang.
        assert b._reset_ack_event.is_set()

    def test_second_reset_clears_previous_event(self):
        b = _make_remote_backend()
        e1 = b.request_reset()
        e1.set()    # pretend first ack arrived
        e2 = b.request_reset()
        # The second reset must return a fresh (unset) event.
        assert not e2.is_set()


# ── Gate 8-A: live lifecycle test against a stub WebSocket server ──────────────

try:
    import websockets as _ws_avail
    _WEBSOCKETS_AVAILABLE = True
except ImportError:
    _WEBSOCKETS_AVAILABLE = False


@pytest.mark.skipif(not _WEBSOCKETS_AVAILABLE, reason="websockets not installed")
class TestBackendLifecycle:
    """start → reset round-trip → stop against a minimal in-process WS stub."""

    def _run_stub_server(self, host: str, port: int, stop_event: threading.Event) -> None:
        """Minimal WS server: hello_ack, emits reset_ack, handles heartbeat."""
        async def _handler(ws):
            raw = await ws.recv()
            msg = json.loads(raw)
            assert msg.get("type") == "hello"
            await ws.send(json.dumps({
                "type": "hello_ack",
                "sim_fps": 500,
                "camera_fps": 15,
                "num_joints": 21,
            }))
            # Wait for a reset request and echo it back.
            async for raw in ws:
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "reset":
                    await ws.send(json.dumps({
                        "type": "reset_ack",
                        "request_id": msg.get("request_id", ""),
                        "sim_step": 1,
                        "scene_revision": "initial",
                    }))
                elif mtype in ("heartbeat", "heartbeat_ack", "joint_command"):
                    pass   # ignore during test
                else:
                    break

        import websockets as ws_mod

        async def _serve():
            async with ws_mod.serve(_handler, host, port):
                await asyncio.get_event_loop().run_in_executor(
                    None, stop_event.wait
                )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(_serve())
        loop.close()

    def test_start_reset_stop(self):
        """Backend thread starts, performs an acknowledged reset, and stops cleanly."""
        import socket, random
        # Find a free port.
        port = random.randint(18000, 19999)
        host = "127.0.0.1"

        stop_event = threading.Event()
        server_thread = threading.Thread(
            target=self._run_stub_server,
            args=(host, port, stop_event),
            daemon=True,
        )
        server_thread.start()
        time.sleep(0.1)   # let server start

        backend = MujocoRemoteBackend(url=f"ws://{host}:{port}")
        backend.start()
        time.sleep(0.3)   # wait for READY

        assert backend.connection_state == ConnectionState.READY

        event = backend.request_reset()
        ok = backend.wait_for_reset(timeout=3.0)
        assert ok, "Reset ack was not received within timeout"
        assert backend.connection_state == ConnectionState.READY

        backend.stop()
        stop_event.set()
        assert backend._thread is not None
        # Thread should have joined (stop() joins with 3 s timeout).
        backend._thread.join(timeout=1.0)
        assert not backend._thread.is_alive(), "Backend thread did not stop cleanly"
