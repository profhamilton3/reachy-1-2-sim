"""Issue #56/#74 (review 2026-09-12, Part 3): validate `link_capsules`'s new
"shells" hand mode against the compiled MJCF -- the model file is the
reference, not a re-implementation of the chain.

Two things are checked, and they are NOT the same claim:

  * TestShellsMatchMJCF -- "shells" reproduces the frames MuJoCo places
    r_gripper_thumb / r_gripper_finger / r_wrist_ball at, to 2 mm.  This
    passes at every named waypoint and every FOOTPRINT_LEGS sample, over the
    full gripper range.

  * TestTubeVsShellsAxisDistance -- a DISCREPANCY from the 2026-09-12
    assignment, kept here (not silently dropped) because it is a real,
    verified geometric fact and the assignment says to report a contradicted
    inequality rather than alter it to pass.  The cause is a lever-arm error
    in `hand_radius()`, not (mainly) wrist roll -- see its class docstring.

Offline throughout: MuJoCo is used only to `mj_forward` a compiled model and
read back geom frames, never `mj_step`, and no server is started.
"""

import math
import os
import sys

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../src"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

from objects import build_scene_model_xml  # noqa: E402
from scene_io import load_scene  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.kinematics import (  # noqa: E402
    hand_radius,
    joint_path,
    link_capsules,
    link_frames,
)

_SCENE_PATH = os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml")

#: The full commanded range: the MJCF's own joint limits (-68.8 open, +20.05
#: shut -- see kinematics._GRIPPER_OPEN_LIMIT_DEG) plus two interior points.
GRIPPER_SAMPLES = (-68.8, -45.0, 0.0, 20.0)


@pytest.fixture(scope="module")
def compiled():
    doc = load_scene(_SCENE_PATH)
    xml = build_scene_model_xml(doc)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for jid in range(m.njnt):
        if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            adr = m.jnt_qposadr[jid]
            d.qpos[adr:adr + 7] = m.qpos0[adr:adr + 7]
    mujoco.mj_forward(m, d)
    return m, d


def _joint_qpos_addrs(m):
    return {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])
            for n in R.R_JOINTS}


def _set_arm(m, d, jq, q7, gripper_deg):
    for name, val in zip(R.ARM7, q7):
        d.qpos[jq[name]] = math.radians(val)
    d.qpos[jq["r_gripper"]] = math.radians(gripper_deg)
    mujoco.mj_forward(m, d)


def _box_endpoints(m, d, gname):
    """World endpoints of the capsule that bounds box geom `gname`, along its
    own local Z through its centre -- the same convention `link_capsules`
    uses for "shells", and the one `outputs/probes-2026-09-12/
    probe_hand_candidates.py` validated this construction against."""
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname)
    h = m.geom_size[g]
    Rm = d.geom_xmat[g].reshape(3, 3)
    c = d.geom_xpos[g]
    axis = Rm @ np.array([0.0, 0.0, 1.0])
    return c - axis * h[2], c + axis * h[2]


def _named_poses():
    """Every named waypoint pose in rig_routes (postures + route waypoints),
    plus the wave's two poses."""
    poses = {}
    for name, p in R.POSTURES.items():
        poses[name] = p
    for route_name in ("PLACE_ROUTE", "STOW_ROUTE", "LIFT_TO_PRESENT",
                       "LOWER_TO_REST", "RAISE_TO_SIDE", "STOW_FROM_SIDE"):
        for wp in R.route_named(route_name):
            poses.setdefault(wp.name, wp.pose)
    poses["WAVE_A"] = R.WAVE_A
    poses["WAVE_B"] = R.WAVE_B
    return poses


def _footprint_leg_samples():
    """13 samples of every FOOTPRINT_LEGS leg, as (label, q7) pairs."""
    out = []
    for route_name, waypoints in R.FOOTPRINT_LEGS.items():
        for i, (a, b) in enumerate(zip(waypoints, waypoints[1:])):
            qa = [a[j] for j in R.ARM7]
            qb = [b[j] for j in R.ARM7]
            for k, q7 in enumerate(joint_path(qa, qb, steps=13)):
                out.append((f"{route_name}[{i}]#{k}", q7))
    return out


def _assert_shells_match_mjcf(m, d, jq, q7, gripper_deg, label):
    _set_arm(m, d, jq, q7, gripper_deg)
    caps = {c[0]: c for c in link_capsules(q7, "right", gripper_deg, hand="shells")}

    thumb_p0, thumb_p1 = _box_endpoints(m, d, "r_thumb_body")
    finger_p0, finger_p1 = _box_endpoints(m, d, "r_finger_body")
    wrist_ball = d.geom_xpos[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "r_wrist_ball")]

    checks = (
        ("thumb", thumb_p0, thumb_p1),
        ("finger", finger_p0, finger_p1),
        ("wrist_ball", wrist_ball, wrist_ball),
    )
    for name, mjcf_p0, mjcf_p1 in checks:
        _n, mine_p0, mine_p1, _r = caps[name]
        for mine, mjcf in ((mine_p0, mjcf_p0), (mine_p1, mjcf_p1)):
            err = max(abs(a - b) for a, b in zip(mine, mjcf))
            assert err < 0.002, (
                f"{label} gripper={gripper_deg}: {name} endpoint off by "
                f"{err * 1000:.2f} mm")


class TestShellsMatchMJCF:
    """`link_capsules(hand="shells")` must reproduce the frames MuJoCo itself
    places r_gripper_thumb / r_gripper_finger / r_wrist_ball at -- not
    approximate them. `upper_arm`/`forearm` are unchanged by `hand="shells"`
    and are already covered by test_arm_clearance.py."""

    def test_named_waypoints(self, compiled):
        m, d = compiled
        jq = _joint_qpos_addrs(m)
        for name, pose in _named_poses().items():
            q7 = [pose[j] for j in R.ARM7]
            for gripper_deg in GRIPPER_SAMPLES:
                _assert_shells_match_mjcf(m, d, jq, q7, gripper_deg, name)

    def test_footprint_leg_samples(self, compiled):
        m, d = compiled
        jq = _joint_qpos_addrs(m)
        for label, q7 in _footprint_leg_samples():
            for gripper_deg in GRIPPER_SAMPLES:
                _assert_shells_match_mjcf(m, d, jq, q7, gripper_deg, label)


def _worst_axis_margin(posture_name, gripper_deg):
    """hand_radius(gripper_deg) minus the farthest any "shells" capsule
    surface point reaches from the wrist axis, at `posture_name`. Negative
    means the tube does NOT bound the shells there."""
    pose = getattr(R, posture_name)
    q7 = [pose[j] for j in R.ARM7]
    _shoulder, _elbow, wrist, Rm = link_frames(q7, "right")
    axis_dir = Rm @ np.array([0.0, 0.0, -1.0])
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    r_tube = hand_radius(gripper_deg)
    worst = None
    for name, p0, p1, r in link_capsules(q7, "right", gripper_deg, hand="shells"):
        if name in ("upper_arm", "forearm"):
            continue
        for p in (np.array(p0), np.array(p1)):
            rel = p - wrist
            perp = rel - np.dot(rel, axis_dir) * axis_dir
            dist = float(np.linalg.norm(perp)) + r
            margin = r_tube - dist
            if worst is None or margin < worst:
                worst = margin
    return worst


class TestTubeVsShellsAxisDistance:
    """DISCREPANCY from the 2026-09-12 assignment, reported for Opus rather
    than forced to pass (per its own instruction: stop and report a
    contradicted inequality rather than alter it to fit).

    The assignment asked this file to assert "the tube's radius >= every
    shells capsule's distance from the wrist axis (the tube bounds the
    shells)". That does not hold whenever the gripper is open, at real,
    already-guarded poses, verified two ways: through
    `link_capsules(hand="shells")` (below) and independently straight off
    MuJoCo's own `r_finger_body`/`r_thumb_body` frames (bypassing this
    file's own construction entirely, to rule out a bug in it rather than a
    fact about the geometry).

    CAUSE, in order of size:

    1. `hand_radius()` swings the wrong lever arm. It rotates the finger
       box's CENTRE (`_FINGER_ARM = 0.038`) by the gripper angle, then adds
       the box's unrotated Y half-extent (0.010). But `r_finger_body`
       extends 0.076 below the hinge, and at -68.8 deg that far end swings
       out 0.076 * sin(68.8) = 7.1 cm, not 3.5. So at HOVER/PRESENT/HOME
       (`r_wrist_roll = 0`), gripper open, the finger shell's far end sits
       10.8 cm (centreline) / 12.4 cm (surface) from the wrist axis, 4.0 cm
       past the tube's 8.3 cm radius. This is the whole of the HOVER rows
       below and 4.0 of REST's 5.6 cm.
    2. Wrist roll adds 0.0325 * sin(roll). The tube is anchored at the wrist
       frame, but the hand pivots at r_gripper_thumb, 3.25 cm below it in
       the pitch-only frame, so the hand's centreline is offset from the
       tube axis by that much -- 1.6 cm at REST/REST_SHUT's
       `r_wrist_roll = 30`, used throughout FOOTPRINT_LEGS. That is the
       difference between the REST and HOVER rows at every aperture.
    3. The ~0.3 cm rows at gripper >= 0 are NOT geometry: they are the
       `thumb` capsule's own slack over the thumb box (radius = the box's XY
       half-diagonal, around a centreline 1.8 cm off axis). `hand_radius`'s
       `_THUMB_RADIUS = hypot(0.025, 0.046)` is the exact corner distance,
       so the tube already contains the box itself.

    The MJCF COLLISION pad `r_finger_col` is outside the tube too (2.2 cm at
    HOVER open, 1.8 cm at -45, 3.8 cm at REST open), so this is a real
    under-conservatism in the model every consumer flies today, in the
    finger's swing direction -- the review's 224-position sweep shows it as
    tube reading more clearance than shells at 3/224 positions
    (test_footprint_boards.py::TestSweepOrdering). Not evidence the guard is
    unsafe against any board it has been flown over (this slice changes no
    default and no margin), but the `hand_radius` lever arm is a defect and
    is the first open item in docs/adr/0003; correcting it changes which
    boards the footprint guard refuses, which is why it is not done here.
    The "shells" construction itself is verified to machine precision
    against MuJoCo in TestShellsMatchMJCF above.
    """

    @pytest.mark.parametrize("posture,gripper_deg,expected_margin_cm", [
        ("REST", -68.8, -5.64),
        ("REST", -45.0, -4.78),
        ("REST", 0.0, -1.94),
        ("REST", 20.0, -1.94),
        ("REST_SHUT", -68.8, -5.64),
        ("HOVER", -68.8, -4.02),
        ("HOVER", -45.0, -3.15),
        ("HOVER", 0.0, -0.32),
        ("PRESENT", -68.8, -4.02),
        ("HOME", -68.8, -4.02),
    ])
    def test_actual_margin_at_named_poses(self, posture, gripper_deg,
                                          expected_margin_cm):
        margin = _worst_axis_margin(posture, gripper_deg)
        assert margin == pytest.approx(expected_margin_cm / 100.0, abs=0.003)
        assert margin < 0, (
            "the discrepancy documented above no longer reproduces -- if "
            "this pose/gripper pair now clears, re-check whether the "
            "assignment's inequality holds generally before relying on it")
