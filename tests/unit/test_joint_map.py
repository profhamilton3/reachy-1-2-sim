"""Unit tests for native_mujoco/joint_map.py (R12-401)."""

import math
import sys
import os

import pytest

# native_mujoco is not a package; add it to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../native_mujoco"))

from joint_map import (
    JOINT_TABLE,
    NUM_JOINTS,
    all_entries,
    build_qpos_from_sdk_degrees,
    by_mjcf_index,
    by_name,
    by_uid,
    qpos_to_sdk_degrees,
)


class TestJointTableCompleteness:
    def test_num_joints(self):
        assert NUM_JOINTS == 21

    def test_all_entries_length(self):
        assert len(all_entries()) == 21

    def test_mjcf_indices_contiguous(self):
        indices = sorted(e.mjcf_index for e in JOINT_TABLE)
        assert indices == list(range(21))

    def test_uids_unique(self):
        uids = [e.uid for e in JOINT_TABLE]
        assert len(set(uids)) == len(uids)

    def test_sdk_names_unique(self):
        names = [e.sdk_name for e in JOINT_TABLE]
        assert len(set(names)) == len(names)

    def test_right_arm_uids(self):
        right_arm = [e for e in JOINT_TABLE if e.sdk_name.startswith("r_") and "antenna" not in e.sdk_name]
        uids = sorted(e.uid for e in right_arm)
        assert uids == [10, 11, 12, 13, 14, 15, 16, 17]

    def test_left_arm_uids(self):
        left_arm = [e for e in JOINT_TABLE if e.sdk_name.startswith("l_") and "antenna" not in e.sdk_name]
        uids = sorted(e.uid for e in left_arm)
        assert uids == [20, 21, 22, 23, 24, 25, 26, 27]

    def test_head_uids(self):
        head = [e for e in JOINT_TABLE if e.sdk_name.startswith("neck_") or "antenna" in e.sdk_name]
        uids = sorted(e.uid for e in head)
        assert uids == [30, 31, 32, 33, 34]


class TestLookupHelpers:
    def test_by_name_found(self):
        e = by_name("r_shoulder_pitch")
        assert e is not None
        assert e.uid == 10
        assert e.mjcf_index == 0

    def test_by_name_missing(self):
        assert by_name("nonexistent_joint") is None

    def test_by_uid_found(self):
        e = by_uid(20)
        assert e is not None
        assert e.sdk_name == "l_shoulder_pitch"

    def test_by_uid_missing(self):
        assert by_uid(999) is None

    def test_by_mjcf_index_found(self):
        e = by_mjcf_index(16)
        assert e is not None
        assert e.sdk_name == "neck_roll"

    def test_by_mjcf_index_missing(self):
        assert by_mjcf_index(99) is None


class TestConversions:
    def test_limits_deg_approx(self):
        e = by_name("r_shoulder_pitch")
        lo, hi = e.limits_deg
        assert abs(lo - (-150.0)) < 1.0
        assert abs(hi - 90.0) < 1.0

    def test_sdk_deg_to_mjcf_rad_zero(self):
        e = by_name("neck_yaw")
        assert e.sdk_deg_to_mjcf_rad(0.0) == pytest.approx(0.0)

    def test_sdk_deg_to_mjcf_rad_90(self):
        e = by_name("neck_roll")
        assert e.sdk_deg_to_mjcf_rad(90.0) == pytest.approx(math.pi / 2)

    def test_mjcf_rad_to_sdk_deg_roundtrip(self):
        e = by_name("r_elbow_pitch")
        original = -45.0
        rad = e.sdk_deg_to_mjcf_rad(original)
        back = e.mjcf_rad_to_sdk_deg(rad)
        assert back == pytest.approx(original, abs=1e-9)


class TestBuildQpos:
    def test_empty_dict_all_zeros(self):
        qpos = build_qpos_from_sdk_degrees({})
        assert qpos == [0.0] * 21

    def test_single_joint_placed_correctly(self):
        qpos = build_qpos_from_sdk_degrees({"r_shoulder_pitch": 90.0})
        e = by_name("r_shoulder_pitch")
        assert qpos[e.mjcf_index] == pytest.approx(math.pi / 2)
        assert all(v == 0.0 for i, v in enumerate(qpos) if i != e.mjcf_index)

    def test_unknown_name_ignored(self):
        # Should not raise; unknown joints are silently skipped.
        qpos = build_qpos_from_sdk_degrees({"bad_joint": 45.0})
        assert qpos == [0.0] * 21

    def test_neck_joints(self):
        qpos = build_qpos_from_sdk_degrees({
            "neck_roll": 10.0,
            "neck_pitch": -20.0,
            "neck_yaw": 30.0,
        })
        assert qpos[16] == pytest.approx(math.radians(10.0))
        assert qpos[17] == pytest.approx(math.radians(-20.0))
        assert qpos[18] == pytest.approx(math.radians(30.0))


class TestQposToSdkDegrees:
    def test_zeros_all_zero(self):
        result = qpos_to_sdk_degrees([0.0] * 21)
        assert all(v == pytest.approx(0.0) for v in result.values())
        assert set(result.keys()) == {e.sdk_name for e in JOINT_TABLE}

    def test_roundtrip_full_pose(self):
        input_deg = {e.sdk_name: float(i * 3 - 30) for i, e in enumerate(JOINT_TABLE)}
        qpos = build_qpos_from_sdk_degrees(input_deg)
        result = qpos_to_sdk_degrees(qpos)
        for name, deg in input_deg.items():
            assert result[name] == pytest.approx(deg, abs=1e-9)
