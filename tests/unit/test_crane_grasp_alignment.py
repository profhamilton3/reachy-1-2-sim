"""Issue #55: crane-pick alignment from the MEASURED hand.

Focused checks only:

  * the pad boxes ``grasp_alignment.hand_frames`` builds are the compiled
    MJCF's ``r_thumb_col`` / ``r_finger_col``, and its palm is ``r_palm_col``
    (#195) (so a clearance in mm means something);
  * an obstructed or misaligned insertion cannot reach the close: a pad
    force reading, or a hand that drifts out of line below the insertion
    checkpoint, halts the descent and the gripper is never closed.

The replays of the two recorded 2026-10-06 misses went with the August mount
(#192): their joint angles put the hand somewhere else now.  The executor
checks below fly a plan made at the October mount instead, red_cube on
cell_r3c2 (the cell the crane plans from there; the crane is re-tuned for the
October mount in #195).

Offline: no simulator, no server; MuJoCo (when installed) only compiles the
model and runs mj_forward.
"""

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../src"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))

from reachy_ai.motion import grasp_alignment as G  # noqa: E402
from reachy_ai.motion import primitives as P  # noqa: E402
from reachy_ai.motion.kinematics import CartesianPlanner, R_ARM_JOINTS, within_limits  # noqa: E402
from reachy_ai.scene.awareness import SceneModel  # noqa: E402
from reachy_ai.tasks import crane_pick_live as C  # noqa: E402

_SCENE = os.path.join(_HERE, "../../scenes/FWDCenterLabSivaPool.yaml")


# ── the pads are the MJCF's pads ──────────────────────────────────────────────

class TestPadBoxesMatchMJCF:
    mujoco = pytest.importorskip("mujoco")

    @pytest.fixture(scope="class")
    def model(self):
        from objects import build_scene_model_xml
        from scene_io import load_scene
        doc = load_scene(_SCENE)
        m = self.mujoco.MjModel.from_xml_string(build_scene_model_xml(
            doc, os.path.join(_HERE, "../../native_mujoco/model/reachy_1_2.xml")))
        return m, self.mujoco.MjData(m)

    @pytest.mark.parametrize("q, grip", [
        # three crane hovers (2026-10-06; hand-frame geometry, any mount)
        ([-83.21, -18.49, 90.17, -83.91, -48.05, 45.04, 39.18], -64.96),
        ([-82.61, -24.14, 90.25, -76.11, 28.93, -31.1, 45.04], -65.0),
        ([-70.09, -16.21, 90.09, -76.51, -37.05, 40.86, 36.79], -65.0),
        ([-40.0, -10.0, 20.0, -70.0, 10.0, 20.0, -10.0], -20.0),
        ([-70.0, -25.0, 0.0, -80.0, 0.0, 0.0, 0.0], 15.0),
    ])
    def test_pad_box_frames(self, model, q, grip):
        mj = self.mujoco
        m, d = model
        mj.mj_resetData(m, d)
        for n, v in zip(R_ARM_JOINTS, q):
            d.qpos[m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)]] = math.radians(v)
        d.qpos[m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "r_gripper")]] = math.radians(grip)
        mj.mj_forward(m, d)
        hand = G.hand_frames(q, grip)
        for box, geom in ((hand.thumb, "r_thumb_col"), (hand.finger, "r_finger_col"),
                          (hand.palm, "r_palm_col")):
            g = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, geom)
            assert np.allclose(box.center, d.geom_xpos[g], atol=1e-4), geom
            assert np.allclose(box.axes, d.geom_xmat[g].reshape(3, 3), atol=1e-4), geom
            assert np.allclose(box.half, m.geom_size[g], atol=1e-9), geom


# ── an obstructed insertion cannot proceed to closure ─────────────────────────

class _Joint:
    def __init__(self, v):
        self.goal_position = float(v)
        self.offset = 0.0

    @property
    def present_position(self):
        return self.goal_position + self.offset


class _Arm:
    def __init__(self, q, grip):
        for n, v in zip(R_ARM_JOINTS, q):
            setattr(self, n, _Joint(v))
        self.r_gripper = _Joint(grip)
        self.gripper_goals = []

    def q(self):
        return [getattr(self, n).goal_position for n in R_ARM_JOINTS]

    def inverse_kinematics(self, M, q0=None):
        raise ValueError("this fake arm has no IK service")   # as reachy_sdk raises


class _Robot:
    def __init__(self, arm):
        self.r_arm = arm


def _october_plan():
    """The crane's own plan for red_cube standing on cell_r3c2, made with the
    simulator's IK (host mirror) and checked, as the panel's preflight does."""
    pytest.importorskip("scipy")
    scene = SceneModel.from_yaml(_SCENE)
    x, y, z = scene.cell_center("cell_r3c2")
    cube = (x, y, z + scene.get("red_cube").size[2] / 2.0)
    scene.update_poses({"red_cube": cube})
    obs = C.Observation(cube, (1.0, 0.0, 0.0, 0.0), 0.0, False, 0.0)
    plan = C.plan_crane_pick(_ik_planner(scene), scene, "red_cube", obs,
                             spec=C.CraneSpec(pause_s=0.2, hold_s=0.2))
    return scene, plan


@pytest.fixture
def rig(monkeypatch):
    """A fake arm that tracks perfectly unless told otherwise; no sleeping."""
    monkeypatch.setattr(P.time, "sleep", lambda *_a: None)
    monkeypatch.setattr(P, "look_at", lambda *a, **k: None)
    monkeypatch.setattr(P, "stow_from_side", lambda *a, **k: None)
    scene, plan = _october_plan()
    arm = _Arm(C.present_joints(), -45.0)
    orig = P.smooth_move

    def smooth(a, pose, duration=2.0):
        if "r_gripper" in pose:
            a.gripper_goals.append(pose["r_gripper"])
        return orig(a, pose, duration)
    monkeypatch.setattr(P, "smooth_move", smooth)
    planner = CartesianPlanner(arm, scene=scene)
    C.check_crane_plan(planner, scene, plan)          # the plan is clean as planned
    return scene, arm, planner, plan


def _run(rig, observe):
    scene, arm, planner, plan = rig
    events = []
    out = C.execute_crane_pick(_Robot(arm), planner, scene, plan, observe,
                               # the parameter is named as the panel's is (`kind`), so an
                               # event field of that name fails here as it did live
                               on_event=lambda kind, **kw: events.append((kind, kw)))
    return out, events, arm


def _observer(arm, plan, force_when=None):
    centre = tuple(plan.obj.center)

    def observe():
        hand = G.hand_frames(arm.q(), arm.r_gripper.present_position)
        f = 0.4 if force_when is not None and force_when(hand) else 0.0
        return C.Observation(centre, (1.0, 0.0, 0.0, 0.0), f, False, 0.0)
    return observe


def test_pad_force_during_insertion_halts_before_closure(rig):
    """A pad reading force on the way down (an obstruction on the top edge)
    stops the descent; the gripper is never commanded closed."""
    _scene_, arm, _planner, plan = rig
    top = plan.obj.highest_z
    obs = _observer(arm, plan,
                    force_when=lambda h: min(h.thumb.lowest_z, h.finger.lowest_z) < top - 0.005)
    out, events, arm = _run(rig, obs)
    assert out["aligned_at_hover"] is True
    assert out["closed"] is False
    assert out["straddled_before_close"] is None
    assert "grip force" in out["halt"]
    assert plan.spec.close_deg not in arm.gripper_goals
    assert any(k == "HALT" for k, _ in events)
    assert out["final"] == "REST"                     # retreated by the checked route


def test_hand_drifting_out_of_line_below_insertion_halts_before_closure(rig, monkeypatch):
    """A tracking error that appears inside the insertion zone (below the last
    rung where a correction is allowed) is caught by the per-rung re-check and
    halts; no correction is attempted there and nothing closes."""
    _scene_, arm, _planner, plan = rig
    k_bad = plan.insertion_rung - 2
    z_bad = G.hand_frames(plan.rungs[k_bad], -65.0).gap_mid[2]
    real_execute = P.execute_trajectory

    def execute(a, traj, names, rate_hz=25, on_step=None):
        real_execute(a, traj, names, rate_hz, on_step)
        # From rung k_bad down, the arm yaw settles 1 deg off its command --
        # a sideways error of the size the recorded attempts carried (arm
        # yaw read +0.65 deg there), appearing mid-insertion.
        low = G.hand_frames(a.q(), -65.0).gap_mid[2] <= z_bad + 1e-6
        a.r_arm_yaw.offset = -1.0 if low else 0.0
    monkeypatch.setattr(P, "execute_trajectory", execute)
    out, events, arm = _run(rig, _observer(arm, plan))
    assert out["closed"] is False
    assert out["halt"] is not None and f"rung +{k_bad}" in out["halt"]
    # Both are per-rung re-checks of the predicted hand: with the palm in the
    # model (#195) the same 1 deg sideways drift can trip the pad-route
    # clearance or the straddle check first, depending on which margin it
    # crosses at that rung.  Either way it halts with no correction.
    assert "pad_route" in out["halt"] or "straddle" in out["halt"]
    assert not any(k == "CORRECTION" for k, _ in events)
    assert plan.spec.close_deg not in arm.gripper_goals


def test_aligned_clean_insertion_does_close(rig):
    """Control: with nothing in the way and the hand where it was commanded,
    the same code reaches the straddle check, passes it, and closes."""
    _scene_, arm, _planner, plan = rig
    out, events, arm = _run(rig, _observer(arm, plan))
    assert out["straddled_before_close"] is True
    assert out["closed"] is True
    assert plan.spec.close_deg in arm.gripper_goals


# ── hold, replacement and withdrawal (after the 2026-10-06 attempt) ───────────

class _Cube:
    """The cube as the observer reports it: still until the gripper closes,
    then riding the hand (optionally slipping ``slip_m`` down in it), never
    below the table, and left where the hand put it -- plus ``shift_after``
    -- once the gripper opens again."""

    def __init__(self, arm, plan, *, slip_m=0.0, shift_after=(0.0, 0.0, 0.0),
                 force_when=None):
        self.arm, self.plan = arm, plan
        self.c0 = np.asarray(plan.obj.center, dtype=float)
        self.rest_z = float(self.c0[2])
        self.slip, self.shift = slip_m, np.asarray(shift_after, dtype=float)
        self.force_when = force_when
        self.state, self.ref_gap, self.pos = "free", None, self.c0.copy()

    def _gap(self):
        return G.hand_frames(self.arm.q(), -65.0).gap_mid

    def __call__(self):
        g = self.arm.r_gripper.goal_position
        close = self.plan.spec.close_deg
        if self.state == "free" and g == close:
            self.state, self.ref_gap = "held", self._gap()
        elif self.state == "held" and g != close:
            self.state = "released"
            self.pos = np.array([self.pos[0], self.pos[1], self.rest_z]) + self.shift
        if self.state == "held":
            d = self._gap() - self.ref_gap
            rise = max(0.0, d[2] - (self.slip if d[2] > self.slip else d[2]))
            self.pos = np.array([self.c0[0] + d[0], self.c0[1] + d[1],
                                 max(self.rest_z, self.rest_z + rise)])
        f = 0.0
        if self.state == "held":
            f = 200.0
        elif self.force_when is not None and self.force_when(self):
            f = 0.3
        return C.Observation(tuple(self.pos), (1.0, 0.0, 0.0, 0.0), f,
                             self.state == "held", 0.0)


def _goal_log(monkeypatch):
    goals = []
    real = P.execute_trajectory

    def execute(a, traj, names, rate_hz=25, on_step=None):
        real(a, traj, names, rate_hz, on_step)
        goals.append(list(a.q()))
    monkeypatch.setattr(P, "execute_trajectory", execute)
    return goals


def test_replacement_stops_at_support_and_withdraws_cleanly(rig, monkeypatch):
    """The cube slipped 7 mm down in the hand while held: lowering stops when
    its bottom reaches the table -- not at the grasp rung, which would push
    the hand 7 mm further down -- then opens there and climbs out clean."""
    _scene_, arm, _planner, plan = rig
    goals = _goal_log(monkeypatch)
    cube = _Cube(arm, plan, slip_m=0.007)
    out, events, arm = _run(rig, cube)
    kinds = [k for k, _ in events]
    sup = dict(events[kinds.index("SUPPORTED")][1])
    assert sup["reached"] is True
    # the lowest commanded jaw-gap height after the lift is ~7 mm above the
    # grasp rung's, not the grasp rung itself
    z0 = G.hand_frames(plan.rungs[0], -65.0).gap_mid[2]
    i_rel = kinds.index("RELEASED")
    lowest = min(G.hand_frames(g, -65.0).gap_mid[2] for g in goals[-400:])
    assert out["replaced"]["supported_before_open"] is True
    assert lowest - z0 > 0.005
    assert out["withdrawal"]["clean"] is True
    assert out["criteria"]["clean_placement"] and out["criteria"]["clean_withdrawal"]
    assert out["criteria"]["returned_to_present"]
    assert i_rel < kinds.index("WITHDRAWN")


def test_moves_start_from_the_commanded_pose_not_the_measured_one(rig, monkeypatch):
    """With the shoulder settling 1.1 deg short (as recorded), no setpoint is
    ever the previous setpoint plus that error: a line started from the
    MEASURED pose is how the thumb pad was dropped onto the table."""
    _scene_, arm, _planner, plan = rig
    goals = _goal_log(monkeypatch)
    arm.r_shoulder_pitch.offset = 1.1
    top = plan.obj.highest_z
    cube = _Cube(arm, plan)
    # a pad force near the bottom halts the descent, so the retreat runs too
    cube.force_when = lambda c: (c.state == "free" and
                                 min(G.hand_frames(c.arm.q(), -65.0).thumb.lowest_z,
                                     G.hand_frames(c.arm.q(), -65.0).finger.lowest_z)
                                 < top - 0.04)
    out, events, arm = _run(rig, cube)
    # A line started from the measured pose shows up as a setpoint where the
    # shoulder pitch alone jumps by about the offset; every planned stream
    # moves it in steps of <= 0.05 deg, or moves several joints at once.
    assert len(goals) > 100
    for a, b in zip(goals, goals[1:]):
        d = np.abs(np.asarray(b) - np.asarray(a))
        assert not (d[0] > 0.5 and int(np.sum(d[1:] > 0.06)) == 0), (a, b)


def test_obstructed_withdrawal_is_refused_before_it_moves(rig, monkeypatch):
    """After release the cube sits 8 mm toward the thumb: the pads' way out is
    blocked, the sideways shift that would clear it exceeds the 6 mm cap, and
    the thumb starts too deep in the cube for a separating retreat.

    #173: that is now found BEFORE the hand opens.  With the grip let go, the
    way out is judged for both openings and neither is clear, so the hand is
    not opened further, the withdrawal is refused, and nothing climbs."""
    _scene_, arm, _planner, plan = rig
    goals = _goal_log(monkeypatch)
    pc = G.pad_clearance(G.hand_frames(plan.rungs[0], -65.0), plan.obj)
    toward_thumb = -0.008 * np.asarray(pc.face_normal)
    cube = _Cube(arm, plan, shift_after=toward_thumb)
    out, events, arm = _run(rig, cube)
    kinds = [k for k, _ in events]
    checks = [dict(kw) for k, kw in events if k == "WITHDRAWAL_PRECHECK"]
    # both openings (wider first), without and then with a separating retreat
    wide, usual = plan.spec.wide_opening_deg, plan.spec.opening_deg
    assert [c["opening_deg"] for c in checks] == [wide, usual, wide, usual]
    assert [len(c["allowed"]) for c in checks] == [2, 2, 3, 3]
    assert not any(c["ok"] for c in checks)
    # never opened into the refusal: after the close, no command to either
    # full opening (the -65 before it is the approach's)
    after_close = arm.gripper_goals[arm.gripper_goals.index(plan.spec.close_deg) + 1:]
    assert plan.spec.opening_deg not in after_close
    assert plan.spec.wide_opening_deg not in after_close
    assert out["withdrawal"]["clean"] is False
    assert "withdrawal refused before opening" in out["halt"]
    assert "WITHDRAWN" not in kinds
    # nothing climbed: the last commanded pose is still down beside the cube
    assert out["criteria"]["clean_withdrawal"] is False
    assert G.hand_frames(goals[-1], -65.0).gap_mid[2] < \
        G.hand_frames(plan.rungs[2], -65.0).gap_mid[2]


def test_pad_force_during_withdrawal_stops_the_climb(rig, monkeypatch):
    """The monitor stays on through the withdrawal: a pad reading force on the
    way out stops it there (the live feed cannot say WHAT was touched; the
    stop does not depend on knowing)."""
    _scene_, arm, _planner, plan = rig
    top = plan.obj.highest_z
    cube = _Cube(arm, plan, force_when=lambda c: (
        c.state == "released" and
        G.hand_frames(c.arm.q(), -65.0).thumb.lowest_z > top - 0.03))
    out, events, arm = _run(rig, cube)
    kinds = [k for k, _ in events]
    assert "WITHDRAWAL_JUDGED" in kinds and "WITHDRAWN" not in kinds
    assert out["withdrawal"]["clean"] is False
    assert "grip force" in out["halt"]


# ── #173: let go, check before opening, and a way out from a resting pad ──────

def test_the_release_lets_go_checks_and_then_opens(rig, monkeypatch):
    """The gripper opens only until the grip force reads zero; the way out is
    judged for the opened hand (wider first) BEFORE it opens further; then it
    opens to the angle that was judged clear, and RELEASED says all three."""
    _scene_, arm, _planner, plan = rig
    cube = _Cube(arm, plan)
    out, events, arm = _run(rig, cube)
    kinds = [k for k, _ in events]
    checks = [dict(kw) for k, kw in events if k == "WITHDRAWAL_PRECHECK"]
    assert checks and checks[0]["opening_deg"] == plan.spec.wide_opening_deg
    assert checks[-1]["ok"] is True
    assert kinds.index("WITHDRAWAL_PRECHECK") < kinds.index("RELEASED")
    rel = out["replaced"]
    assert rel["opening_deg"] == checks[-1]["opening_deg"]
    assert rel["let_go_deg"] > rel["opening_deg"]           # let go before fully open
    assert "release_drag_mm" in rel
    after_close = arm.gripper_goals[arm.gripper_goals.index(plan.spec.close_deg) + 1:]
    assert after_close[0] == rel["opening_deg"]             # one opening, after the check
    assert out["withdrawal"]["clean"] is True


def test_a_pad_left_resting_on_the_cube_now_withdraws(rig, monkeypatch):
    """A1/B4 live: the release left the cube ~3 mm toward the finger, on it.
    Before #173 that refused (the first sample is the pose the hand is in);
    now the hand opens wider and, if needed, separates before it climbs."""
    _scene_, arm, _planner, plan = rig
    pc = G.pad_clearance(G.hand_frames(plan.rungs[0], -65.0), plan.obj)
    toward_finger = (pc.finger_side_m + 0.0001) * np.asarray(pc.face_normal)
    cube = _Cube(arm, plan, shift_after=toward_finger)
    out, events, arm = _run(rig, cube)
    assert out["withdrawal"]["clean"] is True, out.get("halt")
    assert out["criteria"]["returned_to_present"]
    judged = [dict(kw) for k, kw in events if k == "WITHDRAWAL_JUDGED"][0]
    assert judged["ok"] is True


def _ik_planner(scene):
    """A planner whose arm has the simulator's IK (host mirror): the
    sideways and separating ways out are solved, not looked up."""
    pytest.importorskip("scipy")
    sys.path.insert(0, _HERE)
    from host_ik_arm import HostIKArm
    return CartesianPlanner(HostIKArm(), scene=scene)


def _box_at(plan, shift):
    return plan.obj.moved(np.eye(3), np.asarray(shift, dtype=float))


def test_a_separating_retreat_may_not_start_deep_in_the_object(rig):
    """O1 accepts a start under the allowance only down to the floor: a pad
    1 mm into the object is a collision, not a resting contact."""
    _scene_, arm, _planner, plan = rig
    planner = _ik_planner(_scene_)
    pc = G.pad_clearance(G.hand_frames(plan.rungs[0], -65.0), plan.obj)
    n = np.asarray(pc.face_normal)
    cmd, off = list(plan.rungs[0]), np.zeros(7)
    for depth, ok in ((-0.0002, True), (-0.0010, False)):
        live = _box_at(plan, (pc.finger_side_m - depth) * n)
        found, reports = C.plan_way_out(planner, _scene_, plan, cmd, 0, -65.0, live, off)
        assert (found is not None and found.kind == "separating") is ok, reports
        if ok:
            assert found.unmonitored == 1 and reports[-1]["start_mm"][1] < 0


def test_the_way_out_kinds_are_tried_in_order(rig):
    """Straight up first; the pre-#173 sideways correction second; the
    separating retreat only when neither is clear."""
    _scene_, arm, _planner, plan = rig
    planner = _ik_planner(_scene_)
    cmd, off = list(plan.rungs[0]), np.zeros(7)
    found, reports = C.plan_way_out(planner, _scene_, plan, cmd, 0, -65.0, plan.obj, off)
    assert found.kind == "direct" and len(reports) == 1
    pc = G.pad_clearance(G.hand_frames(plan.rungs[0], -65.0), plan.obj)
    on_finger = _box_at(plan, (pc.finger_side_m + 0.0001) * np.asarray(pc.face_normal))
    found, reports = C.plan_way_out(planner, _scene_, plan, cmd, 0, -65.0, on_finger, off,
                                    kinds=("direct", "sideways"))
    assert found is None or found.kind == "sideways"
    found, _ = C.plan_way_out(planner, _scene_, plan, cmd, 0, -65.0, on_finger, off)
    assert found is not None and found.kind in ("sideways", "separating")


def test_the_wider_opening_is_inside_the_gripper_travel():
    from joint_map import by_name
    lo, _hi = by_name("r_gripper").limits_deg
    spec = C.CraneSpec()
    assert lo < spec.wide_opening_deg < spec.opening_deg
    assert spec.wide_opening_deg - lo >= 0.5
    assert C.release_openings(spec) == (spec.wide_opening_deg, spec.opening_deg)


def test_release_grip_stops_where_the_grip_lets_go(monkeypatch):
    monkeypatch.setattr(P.time, "sleep", lambda *_a: None)
    arm = _Arm(C.present_joints(), -25.0)
    goal = P.release_grip(arm, -65.0, lambda: arm.r_gripper.goal_position <= -40.0, 0.8)
    assert -42.0 <= goal <= -40.0
    assert arm.r_gripper.goal_position == goal
    full = P.release_grip(_Arm(C.present_joints(), -25.0), -65.0, lambda: False, 0.8)
    assert full == pytest.approx(-65.0)
