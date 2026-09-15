"""Pilot item (2026-09-15 matrix readiness): scripts/e1_tail_check.py,
moved unchanged from the E1 pilot evidence directory into scripts/ so it
has a real unit test and cannot silently drift from `rig_routes`.

This is the same 11-case table `e1_tail_check.selftest()` has always run,
now as individually-visible pytest cases (each failure names ITS case,
rather than a single 11/11 pass/fail line) plus the module's own
invariant assertions (REST vs REST_SHUT differing only in `r_gripper`).
Offline throughout: no samples come from a live server or SDK.
"""

import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../scripts"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))

import e1_tail_check as etc  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402

_CASES = [
    ("parked at REST, judged REST", etc._synthetic(R.REST), "REST", True),
    ("parked at REST_SHUT, judged REST", etc._synthetic(R.REST_SHUT), "REST", False),
    ("parked at REST, judged REST_SHUT", etc._synthetic(R.REST), "REST_SHUT", False),
    ("parked at REST_SHUT, judged REST_SHUT", etc._synthetic(R.REST_SHUT), "REST_SHUT", True),
    ("parked at PRESENT, judged PRESENT", etc._synthetic(R.PRESENT), "PRESENT", True),
    ("parked at PRESENT, judged REST", etc._synthetic(R.PRESENT), "REST", False),
    ("parked at HOME, judged PRESENT", etc._synthetic(R.HOME), "PRESENT", False),
    ("moving at the end, judged REST",
     etc._synthetic(R.REST, moving_from_s=17.0), "REST", False),
    ("REST arm joints, gripper half-open (-20)",
     etc._synthetic(R.REST, gripper=-20.0), "REST", False),
    ("REST, gripper -47 (within 3)", etc._synthetic(R.REST, gripper=-47.0), "REST", True),
    ("REST, wrist_pitch drooped 18 deg (info only)",
     [dict(s, joints=dict(s["joints"], r_wrist_pitch=R.REST["r_wrist_pitch"] - 18.0))
      for s in etc._synthetic(R.REST)], "REST", True),
]


@pytest.mark.parametrize("label,samples,target,expect", _CASES, ids=[c[0] for c in _CASES])
def test_selftest_case(label, samples, target, expect):
    assert etc.check(samples, target)["ok"] == expect, label


def test_rest_and_rest_shut_differ_only_in_gripper():
    """The check's whole design rests on this: if REST and REST_SHUT ever
    diverge on an arm joint too, the gripper criterion stops being the
    only thing separating them and `--selftest`'s premise breaks."""
    diff = {j for j in R.R_JOINTS if R.REST[j] != R.REST_SHUT[j]}
    assert diff == {"r_gripper"}
    assert R.REST["r_gripper"] == R.OPEN == -45.0
    assert R.REST_SHUT["r_gripper"] == R.SHUT == 20.0


def test_gross_joint_posture_alone_does_not_separate_rest_from_rest_shut():
    assert R.at_pose(R.REST_SHUT, R.REST, tol=8.0, joints=list(R.GROSS_JOINTS))


def test_selftest_entrypoint_still_passes():
    assert etc.selftest() is True
