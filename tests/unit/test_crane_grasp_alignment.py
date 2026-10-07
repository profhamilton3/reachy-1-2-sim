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
from reachy_ai.motion.kinematics import CartesianPlanner, R_ARM_JOINTS  # noqa: E402
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


class _Robot:
    def __init__(self, arm):
        self.r_arm = arm


def _offline_plan(scene):
    rungs = REC["offline_plan"]["rungs"]
    spec = C.CraneSpec(pause_s=0.2, hold_s=0.2)
    obj = G.object_box(scene.get("red_cube"), REC["offline_plan"]["cube"], (1, 0, 0, 0))
    targets = [G.hand_frames(q, spec.opening_deg).gap_mid for q in rungs]
    plan = C.CranePlan("red_cube", spec, C.CraneLimits(), obj,
                       G.hand_rotation(rungs[0]), targets, rungs, C.present_joints())
    plan.insertion_rung = C._insertion_rung(plan)
    return plan


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
