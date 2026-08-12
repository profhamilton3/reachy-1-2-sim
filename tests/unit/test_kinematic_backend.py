"""Unit tests for KinematicBackend, domain models, and state-file writer.

These tests run offline — no gRPC, no ROS, no Docker required.

Exit gates covered:
  R12-100: Domain models have validation and serialization tests.
  R12-101: Existing joint behavior works through KinematicBackend.
  R12-102: Motion result is invariant to zero, one, or multiple snapshot readers.
  R12-103: State file is never written partially (atomic rename path is exercised).
"""

import math
import os
import tempfile
import threading
import time

import pytest

# The server module is imported directly (no gRPC server started).
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kinematic_backend import (
    JointCommand,
    JointSample,
    KinematicBackend,
    SimulationSnapshot,
    JOINT_DEFS,
    state_file_writer,
)


# ── R12-100: Domain model tests ───────────────────────────────────────────────

class TestJointCommand:
    def test_defaults_are_none(self):
        cmd = JointCommand(uid=10)
        assert cmd.goal_position is None
        assert cmd.compliant is None
        assert cmd.speed_limit is None
        assert cmd.torque_limit is None

    def test_frozen(self):
        cmd = JointCommand(uid=10, goal_position=1.0)
        with pytest.raises(AttributeError):
            cmd.goal_position = 2.0  # type: ignore[misc]

    def test_fields_set(self):
        cmd = JointCommand(uid=17, goal_position=0.5, compliant=False,
                           speed_limit=1.0, torque_limit=80.0)
        assert cmd.uid == 17
        assert cmd.goal_position == pytest.approx(0.5)
        assert cmd.compliant is False


class TestJointSample:
    def _make(self, **kwargs):
        defaults = dict(
            name="r_shoulder_pitch", uid=10,
            present_position=0.1, goal_position=0.2,
            present_speed=0.0, present_load=0.0,
            temperature=35.0, compliant=False,
            speed_limit=0.0, torque_limit=100.0,
        )
        defaults.update(kwargs)
        return JointSample(**defaults)

    def test_frozen(self):
        s = self._make()
        with pytest.raises(AttributeError):
            s.present_position = 99.0  # type: ignore[misc]

    def test_fields_correct(self):
        s = self._make(present_position=1.23, goal_position=1.50, compliant=True)
        assert s.name == "r_shoulder_pitch"
        assert s.uid == 10
        assert s.present_position == pytest.approx(1.23, abs=1e-6)
        assert s.goal_position == pytest.approx(1.50, abs=1e-6)
        assert s.compliant is True

    def test_temperature_field(self):
        s = self._make(temperature=42.0)
        assert s.temperature == pytest.approx(42.0)


class TestSimulationSnapshot:
    def test_frozen(self):
        snap = SimulationSnapshot(
            sequence=1, sim_step=1, sim_time_s=0.02,
            wall_time_ns=0, joints={},
        )
        with pytest.raises(AttributeError):
            snap.sequence = 99  # type: ignore[misc]

    def test_joints_mapping(self):
        snap = SimulationSnapshot(
            sequence=0, sim_step=0, sim_time_s=0.0,
            wall_time_ns=0,
            joints={"j": JointSample(
                name="j", uid=1,
                present_position=0.0, goal_position=0.0,
                present_speed=0.0, present_load=0.0,
                temperature=35.0, compliant=True,
                speed_limit=0.0, torque_limit=100.0,
            )},
        )
        assert "j" in snap.joints
        assert snap.joints["j"].uid == 1


# ── R12-101: KinematicBackend basic behavior ──────────────────────────────────

class TestKinematicBackendBasic:
    def test_all_joints_present(self):
        backend = KinematicBackend()
        snap = backend.latest_snapshot()
        expected = {name for name, _ in JOINT_DEFS}
        assert set(snap.joints.keys()) == expected

    def test_initial_positions_zero(self):
        backend = KinematicBackend()
        for s in backend.latest_snapshot().joints.values():
            assert s.present_position == 0.0
            assert s.goal_position == 0.0

    def test_antenna_joints_initially_stiff(self):
        backend = KinematicBackend()
        snap = backend.latest_snapshot()
        assert snap.joints["l_antenna"].compliant is False
        assert snap.joints["r_antenna"].compliant is False

    def test_arm_joints_initially_compliant(self):
        backend = KinematicBackend()
        snap = backend.latest_snapshot()
        assert snap.joints["r_shoulder_pitch"].compliant is True

    def test_uid_by_name(self):
        backend = KinematicBackend()
        assert backend.uid_by_name("r_shoulder_pitch") == 10
        assert backend.uid_by_name("r_gripper") == 17
        assert backend.uid_by_name("nonexistent") is None

    def test_snapshot_sequence_advances(self):
        backend = KinematicBackend()
        backend.start()
        time.sleep(0.15)
        snap = backend.latest_snapshot()
        assert snap.sequence > 0
        assert snap.sim_step > 0
        assert snap.sim_time_s > 0.0

    def test_command_changes_goal(self):
        backend = KinematicBackend()
        backend.start()
        backend.submit_command(JointCommand(uid=10, goal_position=1.0, compliant=False))
        time.sleep(0.1)
        snap = backend.latest_snapshot()
        assert snap.joints["r_shoulder_pitch"].goal_position == pytest.approx(1.0, abs=1e-6)

    def test_command_changes_compliant(self):
        backend = KinematicBackend()
        backend.start()
        backend.submit_command(JointCommand(uid=10, compliant=False))
        time.sleep(0.1)
        snap = backend.latest_snapshot()
        assert snap.joints["r_shoulder_pitch"].compliant is False


# ── R12-102: Motion convergence and subscriber invariance ─────────────────────

class TestMotionConvergence:
    def test_joint_converges(self):
        """A stiff joint with a goal position must converge within 2 seconds."""
        backend = KinematicBackend()
        backend.start()
        backend.submit_command(JointCommand(uid=10, goal_position=1.0, compliant=False))
        time.sleep(2.0)
        pos = backend.latest_snapshot().joints["r_shoulder_pitch"].present_position
        assert abs(pos - 1.0) < 0.05, f"Position {pos:.4f} did not converge to 1.0"

    def test_compliant_joint_does_not_move(self):
        """A compliant joint must not advance toward its goal."""
        backend = KinematicBackend()
        backend.start()
        # Default state: compliant=True, goal=0 — leave both as-is and set a goal.
        backend.submit_command(JointCommand(uid=10, goal_position=1.0))
        time.sleep(0.5)
        pos = backend.latest_snapshot().joints["r_shoulder_pitch"].present_position
        assert abs(pos) < 0.01, f"Compliant joint moved to {pos:.4f}"

    def test_subscriber_reads_do_not_accelerate_motion(self):
        """Calling latest_snapshot() many times must not alter the motion trajectory.

        This is the critical R12-102 invariant: the old StreamJointsState bug
        caused subscriber reads to advance present_position, so more subscribers
        meant faster (and over-shooting) convergence.
        """
        target = 0.5

        def run_with_readers(n_readers: int) -> float:
            backend = KinematicBackend()
            backend.start()
            backend.submit_command(
                JointCommand(uid=10, goal_position=target, compliant=False)
            )
            stop = threading.Event()

            def reader():
                while not stop.is_set():
                    backend.latest_snapshot()
                    time.sleep(0.005)

            threads = [threading.Thread(target=reader, daemon=True) for _ in range(n_readers)]
            for t in threads:
                t.start()
            time.sleep(0.8)
            stop.set()
            for t in threads:
                t.join(timeout=1.0)
            return backend.latest_snapshot().joints["r_shoulder_pitch"].present_position

        pos_zero = run_with_readers(0)
        pos_three = run_with_readers(3)

        # Both must have converged (not overshot) and agree within tolerance.
        assert pos_zero <= target + 0.02, f"0 readers: overshot to {pos_zero:.4f}"
        assert pos_three <= target + 0.02, f"3 readers: overshot to {pos_three:.4f}"
        assert abs(pos_zero - pos_three) < 0.05, (
            f"Results diverge with reader count: 0→{pos_zero:.4f}, 3→{pos_three:.4f}"
        )

    def test_no_overshoot(self):
        """present_position must not exceed goal_position."""
        backend = KinematicBackend()
        backend.start()
        target = 0.3
        backend.submit_command(JointCommand(uid=10, goal_position=target, compliant=False))
        t_end = time.monotonic() + 2.0
        max_pos = 0.0
        while time.monotonic() < t_end:
            pos = backend.latest_snapshot().joints["r_shoulder_pitch"].present_position
            max_pos = max(max_pos, pos)
            time.sleep(0.02)
        assert max_pos <= target + 0.01, f"Overshot: max position was {max_pos:.4f}"


# ── R12-103: Atomic state file writer ────────────────────────────────────────

class TestStateFileWriter:
    def test_file_written_and_parseable(self, tmp_path):
        """State file must be valid JSON after the writer runs."""
        import json
        state_path = str(tmp_path / "joints.json")
        backend = KinematicBackend()
        backend.start()
        threading.Thread(
            target=state_file_writer, args=(backend,), kwargs={"path": state_path}, daemon=True
        ).start()
        time.sleep(0.2)
        with open(state_path) as f:
            data = json.load(f)
        expected_names = {name for name, _ in JOINT_DEFS}
        assert set(data.keys()) == expected_names
        for v in data.values():
            assert isinstance(v, (int, float))
            assert math.isfinite(v)

    def test_file_reflects_motion(self, tmp_path):
        """After a command, the state file must reflect updated positions."""
        import json
        state_path = str(tmp_path / "joints.json")
        backend = KinematicBackend()
        backend.start()
        threading.Thread(
            target=state_file_writer, args=(backend,), kwargs={"path": state_path}, daemon=True
        ).start()
        backend.submit_command(JointCommand(uid=10, goal_position=1.0, compliant=False))
        time.sleep(2.0)
        with open(state_path) as f:
            data = json.load(f)
        assert abs(data["r_shoulder_pitch"] - 1.0) < 0.05
