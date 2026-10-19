"""#187: the SDK's FK/IK frame is the gripper tip, and the pads did not move.

The simulator's `forward_kinematics()` / `inverse_kinematics()` now report the
tip (URDF right_tip), as the real robot's do.  The planner still plans from
the wrist, so it converts with the joints (`kinematics.fk_wrist`,
`kinematics.ik_wrist`).  These tests hold the change to:

  * the tip frame: the local chain's `tip_pose` reproduces the real robot's
    FK for B2_SAFE_JOINTS, and its offsets are the URDF's;
  * PAD INVARIANCE: for fixed joint sets -- PRESENT, B2_SAFE with its wrist
    roll swept, and every rung of the crane's r3c2 -> r3c3 placement plan
    (#195/#197) -- the pad is where it was before #187, within 0.1 mm
    (fixture recorded at a12c11a, the commit before);
  * the PLAN: re-planning r3c2 -> r3c3 gives the same joints;
  * IK ROUND TRIPS through the host mirror of the service, tip and wrist.

The SDK itself (gRPC, the real `reachy_sdk` arm) is covered by
tests/contract/test_arm_kinematics_tip.py.
"""

import json
import math
import os
import re
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "."):
    _abs = os.path.join(_HERE, _p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

from reachy_ai.motion import kinematics as K  # noqa: E402

_FIXTURES = os.path.join(_HERE, "..", "fixtures", "kinematics")
_URDF = os.path.join(_HERE, "..", "..", "native_mujoco", "model", "reachy.urdf")

#: Position (m) and attitude (deg) agreement the issue asks for.
_POS_TOL = 1e-4
_ROT_TOL_DEG = 0.01


def _load(name):
    with open(os.path.join(_FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


_B2 = _load("b2_safe_real_fk.json")
_BEFORE = _load("pad_invariance_187.json")


def _rot_err_deg(a, b):
    c = (np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class _TipArm:
    """The SDK's FK, from the local chain; no IK."""

    def forward_kinematics(self, q):
        return K.tip_pose(q)


# ── The tip frame ─────────────────────────────────────────────────────────────

def test_tip_pose_reproduces_the_real_robots_fk_for_b2_safe_joints():
    T = K.tip_pose(_B2["joints_deg"])
    ref = np.asarray(_B2["right_tip_pose"])
    assert np.linalg.norm(T[:3, 3] - ref[:3, 3]) <= _POS_TOL
    assert _rot_err_deg(T[:3, :3], ref[:3, :3]) <= _ROT_TOL_DEG


def _urdf_origin(joint):
    with open(_URDF, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r'<joint name="%s"[^>]*>.*?<origin xyz="([^"]+)"' % joint, text, re.S)
    return np.array([float(v) for v in m.group(1).split()])


def test_the_tip_offsets_are_the_urdfs_on_both_arms():
    # l_tip is checked on its own, not assumed to be r_tip's mirror.
    assert np.allclose(K._TIP_OFFSET["right"], _urdf_origin("r_tip"))
    assert np.allclose(K._TIP_OFFSET["left"], _urdf_origin("l_tip"))
    assert np.allclose(K._THUMB_ORIGIN_OFFSET, _urdf_origin("r_wrist_roll"))
    assert np.allclose(K._THUMB_ORIGIN_OFFSET, _urdf_origin("l_wrist_roll"))


def test_the_tip_is_a_roll_dependent_offset_from_the_wrist():
    """Not a fixed tool length: ~2.5 mm along the axis and 12 mm sideways at
    roll 0, turned by the roll; the rotation is the wrist's."""
    q = list(_B2["joints_deg"])
    for roll in (-45.0, 0.0, 45.0):
        q[6] = roll
        _s, _e, wrist, R = K.link_frames(q)
        T = K.tip_pose(q)
        assert np.allclose(T[:3, :3], R)
        local = R.T @ (T[:3, 3] + K._BASE - wrist)
        r = math.radians(roll)
        assert local == pytest.approx([0.0, -0.012 - 0.0325 * math.sin(r),
                                       -0.085 - 0.0325 * math.cos(r)], abs=1e-12)


# ── Pad invariance ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(_BEFORE["poses"]))
def test_the_pad_is_where_it_was_before_187(name):
    pose = _BEFORE["poses"][name]
    before = np.asarray(pose["pad_world_m"])
    q = pose["joints_deg"]
    planner = K.CartesianPlanner(_TipArm())
    assert np.linalg.norm(np.asarray(planner.fk_world(q)) - before) <= _POS_TOL
    assert np.linalg.norm(K.pad_point(q) - before) <= _POS_TOL


def test_the_fixture_covers_the_crane_plan_and_the_wrist_roll():
    names = set(_BEFORE["poses"])
    assert _BEFORE["provenance"]["commit"] == "a12c11a"
    assert sum(n.startswith("crane_pick_rung_") for n in names) >= 10
    assert sum(n.startswith("crane_dest_rung_") for n in names) >= 10
    rolls = {round(p["joints_deg"][6]) for p in _BEFORE["poses"].values()}
    assert {-45, 45} <= rolls


# ── The plan, through the host mirror of the service's IK ─────────────────────

def _mirror():
    pytest.importorskip("scipy")
    from host_ik_arm import HostIKArm
    return HostIKArm()


def test_the_crane_r3c2_to_r3c3_plan_is_unchanged():
    arm = _mirror()
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks import crane_pick_live as CP
    from reachy_ai.tasks import crane_place_live as PL

    scene = SceneModel.from_yaml(os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml"))
    x, y, z = scene.cell_center("cell_r3c2")
    pos = (x, y, z + 0.03)
    scene.update_poses({"red_cube": pos})
    pp = PL.plan_crane_place(K.CartesianPlanner(arm, scene=scene), scene, "red_cube",
                             "r3c3", CP.Observation(pos, (1.0, 0.0, 0.0, 0.0), 0.0, False))
    ladders = [("crane_pick_rung_%02d", pp.pick.rungs), ("crane_dest_rung_%02d", pp.dest.rungs)]
    for fmt, rungs in ladders:
        for k, q in enumerate(rungs):
            before = _BEFORE["poses"][fmt % k]
            # Joints to 0.05 deg (measured: <= 0.03 over a flight's 265 IK
            # calls), pads to 0.1 mm.
            assert np.abs(np.asarray(q) - before["joints_deg"]).max() <= 0.05, fmt % k
            assert np.linalg.norm(K.pad_point(q) - before["pad_world_m"]) <= _POS_TOL


# ── IK round trips ────────────────────────────────────────────────────────────

_ROUND_TRIP = sorted(n for n in _BEFORE["poses"] if not n.startswith("zero"))

#: Also round-tripped with no seed.  The crane's rungs are not: from the zero
#: pose the solver stalls short of most of them, as the pre-#187 wrist IK did
#: (by 0.1-0.2 m); five it solved unseeded now stall too (#187 PR, "Known
#: limitations").  Every planner call is seeded.
_UNSEEDED = sorted(n for n in _ROUND_TRIP if not n.startswith("crane_"))


@pytest.mark.parametrize("name", _ROUND_TRIP)
def test_tip_ik_round_trips_through_the_service_mirror(name):
    arm = _mirror()
    q = _BEFORE["poses"][name]["joints_deg"]
    T = arm.forward_kinematics(q)
    for q0 in ((q, None) if name in _UNSEEDED else (q,)):
        back = arm.forward_kinematics(arm.inverse_kinematics(T, q0=q0))
        assert np.linalg.norm(back[:3, 3] - T[:3, 3]) <= _POS_TOL, q0
        assert _rot_err_deg(back[:3, :3], T[:3, :3]) <= _ROT_TOL_DEG, q0


@pytest.mark.parametrize("name", _ROUND_TRIP)
def test_ik_wrist_round_trips_the_wrist(name):
    arm = _mirror()
    q = _BEFORE["poses"][name]["joints_deg"]
    M = K.fk_wrist(arm, q)
    back = K.fk_wrist(arm, K.ik_wrist(arm, M, q))
    assert np.linalg.norm(back[:3, 3] - M[:3, 3]) <= _POS_TOL
    assert _rot_err_deg(back[:3, :3], M[:3, :3]) <= _ROT_TOL_DEG


def test_ik_wrist_leaves_an_answer_that_misses_the_tip_alone():
    """An arm that ignores the target (or cannot reach it) is not 're-solved'
    into a different pose: its answer comes back as it was."""

    class _Fixed(_TipArm):
        def inverse_kinematics(self, M, q0=None):
            return list(_B2["joints_deg"])

    M = np.eye(4)
    M[:3, 3] = (0.3, -0.2, -0.3)
    assert K.ik_wrist(_Fixed(), M, [0.0] * 7) == list(_B2["joints_deg"])


def test_ik_wrists_bounds_and_threshold_are_the_services():
    """`ik_wrist`'s re-solve keeps to the service's bounds (not the MJCF
    travel) and lands at its threshold; read from the server's source, which
    cannot be imported on the host."""
    import ast
    path = os.path.join(_HERE, "..", "..", "fake_reachy_server.py")
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    values = {t.id: ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) for t in node.targets
              if isinstance(t, ast.Name) and t.id in
              {"_R_ARM_LIMITS", "_L_ARM_LIMITS", "_IK_LANDED", "_R_TIP", "_L_TIP"}}
    assert [tuple(b) for b in values["_R_ARM_LIMITS"]] == list(K._IK_LIMITS_RAD["right"])
    assert [tuple(b) for b in values["_L_ARM_LIMITS"]] == list(K._IK_LIMITS_RAD["left"])
    assert values["_IK_LANDED"] == K._IK_LANDED
    assert np.allclose(values["_R_TIP"], K._TIP_OFFSET["right"])
    assert np.allclose(values["_L_TIP"], K._TIP_OFFSET["left"])
