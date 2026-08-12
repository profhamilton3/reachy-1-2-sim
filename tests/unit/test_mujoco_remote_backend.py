"""Unit tests for MujocoRemoteBackend and KinematicBridge.

These tests run offline — no WebSocket server, no gRPC, no Docker required.

Key behaviors tested:
  - to_kinematic_snapshot() produces a properly-keyed SimulationSnapshot
  - KinematicBridge delegates correctly and returns the right types
  - Missing joints are filled in with safe defaults
  - Antenna joints default to stiff (compliant=False)
  - RemoteSnapshot → SimulationSnapshot field mapping is correct
"""

import math
import os
import sys
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
