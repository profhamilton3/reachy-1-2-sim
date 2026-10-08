"""Issue #55: crane-pick alignment from the MEASURED hand.

Focused checks only:

  * the pad boxes ``grasp_alignment.hand_frames`` builds are the compiled
    MJCF's ``r_thumb_col`` / ``r_finger_col`` (so a clearance in mm means
    something);
  * the misalignment behind both recorded misses (2026-10-06) is DETECTED
    from the measured hover alone: the plan had ~2.3 mm per side, the
    measured hand predicts the pad that actually landed on the top edge;
  * an obstructed or misaligned insertion cannot reach the close: a pad
    force reading, or a hand that drifts out of line below the insertion
    checkpoint, halts the descent and the gripper is never closed.

Offline: no simulator, no server; MuJoCo (when installed) only compiles the
model and runs mj_forward.
"""

import json
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
with open(os.path.join(_HERE, "../fixtures/crane/recorded_crane_attempts_2026_10_06.json")) as f:
    REC = json.load(f)


def _scene(cube):
    s = SceneModel.from_yaml(_SCENE)
    s.update_poses({"red_cube": cube})
    return s


def _cube_box(cube=None):
    s = _scene(REC["cube"]["center"] if cube is None else cube)
    c = REC["cube"]["center"] if cube is None else cube
    return G.object_box(s.get("red_cube"), c, REC["cube"]["quat_wxyz"])


def _rungs(run):
    r = REC[run]["commanded_rungs"]
    return [r[str(k)] for k in range(9, -1, -1)]          # hover -> grasp


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
        (REC["x_face"]["hover_measured_q"], REC["x_face"]["hover_measured_gripper"]),
        (REC["y_face"]["hover_measured_q"], -65.0),
        (REC["offline_plan"]["rungs"][0], -65.0),
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
        for box, geom in ((hand.thumb, "r_thumb_col"), (hand.finger, "r_finger_col")):
            g = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, geom)
            assert np.allclose(box.center, d.geom_xpos[g], atol=1e-4), geom
            assert np.allclose(box.axes, d.geom_xmat[g].reshape(3, 3), atol=1e-4), geom
            assert np.allclose(box.half, m.geom_size[g], atol=1e-9), geom


# ── the recorded misalignment is detected ─────────────────────────────────────

class TestRecordedMisalignmentDetected:
    """The two crane attempts of 2026-10-06, re-read from the hover gate."""

    @pytest.mark.parametrize("run", ["x_face", "y_face"])
    def test_measured_hover_predicts_the_pad_that_hit(self, run):
        box = _cube_box()
        rec = REC[run]
        rungs = _rungs(run)
        off = G.joint_offset(rec["hover_measured_q"], rungs[0])
        planned = G.route_clearance(rungs, -65.0, box)
        predicted = G.route_clearance(rungs, rec["hover_measured_gripper"], box, offset=off)
        tol = C.prediction_tolerance(0.09)
        # As commanded, the hit pad cleared the cube on its way down ...
        hit_planned = planned.thumb_m if rec["hit"] == "thumb" else planned.finger_m
        assert hit_planned > 0.0
        # ... as measured, the same pad's route falls below what the
        # prediction is good for, and it is the worse of the two.
        hit_pred = predicted.thumb_m if rec["hit"] == "thumb" else predicted.finger_m
        other_pred = predicted.finger_m if rec["hit"] == "thumb" else predicted.thumb_m
        assert hit_pred < tol
        assert hit_pred < other_pred

    def test_x_face_plan_had_the_reported_slack_and_lost_it(self):
        box = _cube_box()
        rungs = _rungs("x_face")
        off = G.joint_offset(REC["x_face"]["hover_measured_q"], rungs[0])
        planned = G.route_clearance(rungs, -65.0, box)
        predicted = G.route_clearance(rungs, REC["x_face"]["hover_measured_gripper"], box,
                                      offset=off)
        assert 0.002 < planned.thumb_m < 0.003 and 0.002 < planned.finger_m < 0.003
        assert predicted.finger_m < -0.003            # 3.5 mm into the top +x edge

    def test_first_contact_pose_is_not_a_straddle(self):
        """The measured x-face hand at grasp depth (finger on the top edge)
        must never be judged ready to close."""
        box = _cube_box()
        hand = G.hand_frames(REC["x_face"]["first_contact_grasp_rung_measured_q"],
                             REC["x_face"]["hover_measured_gripper"])
        st = G.straddle_check(hand, box, 0.74, 0.0, C.jaw_face_limit_deg(0.0023))
        assert not st.ok
        assert "finger_not_outside_face" in st.failed


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


def _recorded_offline_plan(scene):
    """The recorded offline plan exactly as recorded (2026-10-06).  It predates
    the exact joint travel: its grasp rung commands r_arm_yaw past the stop."""
    rungs = REC["offline_plan"]["rungs"]
    spec = C.CraneSpec(pause_s=0.2, hold_s=0.2)
    obj = G.object_box(scene.get("red_cube"), REC["offline_plan"]["cube"], (1, 0, 0, 0))
    targets = [G.hand_frames(q, spec.opening_deg).gap_mid for q in rungs]
    plan = C.CranePlan("red_cube", spec, C.CraneLimits(), obj,
                       G.hand_rotation(rungs[0]), targets, rungs, C.present_joints())
    plan.insertion_rung = C._insertion_rung(plan)
    return plan


def _offline_plan(scene):
    """The recorded plan with each rung re-solved INSIDE the joint travel to
    the same jaw-gap target, by the planner's own re-solve with the attitude
    soft (as the ladder solves it); check_crane_plan then judges the result.  The recorded data are unchanged; only rungs past a stop
    move, by as much as the stops require."""
    rec = _recorded_offline_plan(scene)
    spec, tol = rec.spec, rec.limits.ik_position_tol_m
    rungs = []
    for q, tgt in zip(rec.rungs, rec.gap_targets):
        if not within_limits(q):
            q, err = C._gap_resolve(G.hand_rotation(q), tgt, spec.opening_deg, q, tol,
                                    C._RESOLVE_ROT_WEIGHT_RELAX_M)   # as the ladder does
            assert err < tol and within_limits(q)
        rungs.append(list(q))
    plan = C.CranePlan("red_cube", spec, rec.limits, rec.obj, G.hand_rotation(rungs[0]),
                       rec.gap_targets, rungs, C.present_joints())
    plan.insertion_rung = C._insertion_rung(plan)
    return plan


def test_the_recorded_plan_past_a_stop_is_refused(monkeypatch):
    """#55: the recorded rungs command r_arm_yaw past its 89.954 deg stop.  The
    plan check refuses that instead of letting the simulator clip it."""
    scene = _scene(REC["offline_plan"]["cube"])
    planner = CartesianPlanner(_Arm(C.present_joints(), -45.0), scene=scene)
    with pytest.raises(C.CraneRefused) as info:
        C.check_crane_plan(planner, scene, _recorded_offline_plan(scene))
    assert info.value.stage == "joint_limits"


@pytest.fixture
def rig(monkeypatch):
    """A fake arm that tracks perfectly unless told otherwise; no sleeping."""
    monkeypatch.setattr(P.time, "sleep", lambda *_a: None)
    monkeypatch.setattr(P, "look_at", lambda *a, **k: None)
    monkeypatch.setattr(P, "stow_from_side", lambda *a, **k: None)
    scene = _scene(REC["offline_plan"]["cube"])
    arm = _Arm(C.present_joints(), -45.0)
    orig = P.smooth_move

    def smooth(a, pose, duration=2.0):
        if "r_gripper" in pose:
            a.gripper_goals.append(pose["r_gripper"])
        return orig(a, pose, duration)
    monkeypatch.setattr(P, "smooth_move", smooth)
    planner = CartesianPlanner(arm, scene=scene)
    plan = _offline_plan(scene)
    C.check_crane_plan(planner, scene, plan)          # the plan is clean as planned
    return scene, arm, planner, plan


def _run(rig, observe):
    scene, arm, planner, plan = rig
    events = []
    out = C.execute_crane_pick(_Robot(arm), planner, scene, plan, observe,
                               on_event=lambda k, **kw: events.append((k, kw)))
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
    assert out["final"] == "HOME"                     # retreated by the checked route


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
    assert "pad_route" in out["halt"]
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
    blocked and the sideways shift that would clear it exceeds the 6 mm cap.
    The withdrawal is refused -- the hand stays where it opened -- and the
    attempt reports it rather than climbing through the cube."""
    _scene_, arm, _planner, plan = rig
    goals = _goal_log(monkeypatch)
    pc = G.pad_clearance(G.hand_frames(plan.rungs[0], -65.0), plan.obj)
    toward_thumb = -0.008 * np.asarray(pc.face_normal)
    cube = _Cube(arm, plan, shift_after=toward_thumb)
    out, events, arm = _run(rig, cube)
    kinds = [k for k, _ in events]
    assert "WITHDRAWAL_JUDGED" in kinds
    judged = dict(events[kinds.index("WITHDRAWAL_JUDGED")][1])
    assert judged["ok"] is False
    assert out["withdrawal"]["clean"] is False
    assert "withdrawal refused" in out["halt"]
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
