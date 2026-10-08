"""
R12-502: Gripper and contact model tests.

Requires `mujoco` (native side); skipped in the Python-3.8 container / CI.

The headline test is the exit-gate scenario: a cube is contacted, lifted,
released, and detected — using real MuJoCo contact physics (no weld/attach
shortcut).  The single-finger gripper forms a friction pinch; the cube rests
on a support pillar (dedicated collision channel bit-3 so the robot ignores it)
until grasped, then is lifted clear and dropped on release.
"""

import os
import re

import pytest

mujoco = pytest.importorskip("mujoco")
import numpy as np  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../native_mujoco"))
from actuator import ActuatorController  # noqa: E402
from gripper import GripperModel, GRIPPER_DEFS  # noqa: E402

_MODEL = os.path.join(
    os.path.dirname(__file__), "../../native_mujoco/model/reachy_1_2.xml"
)
_CUBE_HALF = 0.010  # 20 mm cube fits the ~28 mm open pad gap


_ORIGINAL_CUBE_FRICTION = "2.5 0.1 0.01"


def _grasp_scene(cube_friction=_ORIGINAL_CUBE_FRICTION, noslip_iterations=None):
    """Model with the table removed (clear workspace) + support pillar + cube.

    Support: contype=8 conaffinity=4  -> objects rest on it, robot ignores it.
    Cube:    contype=4 conaffinity=15 -> collides with world/robot/object/support.
    ``noslip_iterations=None`` keeps the model file's contact model (ADR-0005).
    """
    xml = open(_MODEL).read()
    xml = re.sub(r'<body name="table".*?</body>\s*', "", xml, flags=re.DOTALL)
    extra = f"""
    <body name="support" pos="0 -0.20 0.1847">
      <geom name="support" type="box" size="0.06 0.06 0.15"
            contype="8" conaffinity="4" rgba="0.4 0.4 0.9 0.35"/>
    </body>
    <body name="cube" pos="0 0 0">
      <freejoint name="cube_free"/>
      <geom name="cube" type="box" size="{_CUBE_HALF} {_CUBE_HALF} {_CUBE_HALF}"
            contype="4" conaffinity="15" rgba="0.9 0.2 0.2 1" mass="0.02"
            friction="{cube_friction}"/>
    </body>
  </worldbody>"""
    m = mujoco.MjModel.from_xml_string(xml.replace("  </worldbody>", extra, 1))
    if noslip_iterations is not None:
        m.opt.noslip_iterations = noslip_iterations
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    return m, d


def _roll(m, d, ctl, n):
    dt = m.opt.timestep
    for _ in range(n):
        ctl.apply(d, dt)
        mujoco.mj_step(m, d)


class TestGripperModelBasics:
    def test_construction_resolves_pads(self):
        m = mujoco.MjModel.from_xml_path(_MODEL)
        g = GripperModel(m)
        states = g.update(mujoco.MjData(m))
        assert set(states) == {"right", "left"}

    def test_open_gripper_no_force_no_grasp(self):
        m = mujoco.MjModel.from_xml_path(_MODEL)
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        g = GripperModel(m)
        st = g.update(d)["right"]
        assert st.grip_force_n == pytest.approx(0.0)
        assert st.grasping is False

    def test_force_by_sensor_uid_keys(self):
        m = mujoco.MjModel.from_xml_path(_MODEL)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        forces = GripperModel(m).force_by_sensor_uid(d)
        assert set(forces) == {1, 2}   # r_force_gripper, l_force_gripper


# The exit gate runs under BOTH contact models (ADR-0005, #55), with the same
# assertions.  It verifies the gripper/contact model: a cube is contacted,
# lifted, released and detected through real contact physics.  The cube's
# friction is a property of the fixture, not the thing under test.
#
# - noslip_off_prior_fixture: the original fixture unchanged (cube mu=2.5)
#   under the prior contact model (noslip_iterations=0).
# - noslip_on_default: the model default (no-slip on) with cube mu=1.0.  The
#   pads carry MuJoCo's default sliding friction 1.0 (condim 3), and MuJoCo
#   takes the larger of two geoms' frictions, so this gives the same
#   effective pad-cube friction as the live scene's red_cube (0.8 -> 1.0).
#   mu=2.5 matches no contact in the project's scenes; it was tuned for the
#   soft model.  Neither value is a claim about physical friction.
#
# The original mu=2.5 fixture does NOT grasp with no-slip on: the closing
# finger tips the cube on the pillar and wedges it, and the thumb never
# touches it.  TestOriginalFixtureUnderNoslip keeps that visible.  (At
# mu=1.0 with no-slip off, the cube flips over instead.)  Measured
# 2026-10-07.
_GRASP_CONFIGS = {
    "noslip_off_prior_fixture": dict(cube_friction=_ORIGINAL_CUBE_FRICTION,
                                     noslip_iterations=0),
    "noslip_on_default": dict(cube_friction="1.0 0.1 0.01"),
}


def _run_grasp_scenario(**scene_kwargs):
    """Place the cube in the open gap, close, lift, release; measure each."""
    m, d = _grasp_scene(**scene_kwargs)
    ctl = ActuatorController(m)
    ctl.sync_targets_to_current(d)
    grip = GripperModel(m)
    fg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "r_finger_col")
    tg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "r_thumb_col")
    sg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "support")
    cadr = m.jnt_qposadr[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    ]

    # Right arm stiff & straight; gripper open.
    for i in range(7):
        ctl.set_compliant(i, False)
        ctl.set_goal_position(i, 0.0)
    ctl.set_compliant("act_r_gripper", False)
    ctl.set_goal_position("act_r_gripper", -0.6)
    _roll(m, d, ctl, 200)

    # Place the cube centred in the open pad gap, resting on the support.
    fp, tp = d.geom_xpos[fg], d.geom_xpos[tg]
    cube_y = ((fp[1] + 0.008) + (tp[1] - 0.008)) / 2
    cube_z = d.geom_xpos[sg][2] + 0.15 + _CUBE_HALF
    d.qpos[cadr:cadr + 7] = [tp[0], cube_y, cube_z, 1, 0, 0, 0]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    z_start = float(d.qpos[cadr + 2])
    contacts_at_placement = int(d.ncon)

    # CLOSE
    ctl.set_goal_position("act_r_gripper", 0.35)
    _roll(m, d, ctl, 700)
    close = grip.update(d)["right"]
    cube_body = m.geom_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube")]
    thumb_on_cube = any(
        tg in (c.geom1, c.geom2)
        and cube_body in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
        for c in d.contact[:d.ncon])

    # LIFT (shoulder pitch forward to -0.7, within the reliable envelope)
    ctl.set_goal_position(0, -0.7)
    ctl.set_speed_limit(0, 0.6)
    _roll(m, d, ctl, 2200)
    lift = grip.update(d)["right"]
    z_lift = float(d.qpos[cadr + 2])

    # RELEASE
    ctl.set_goal_position("act_r_gripper", -0.6)
    _roll(m, d, ctl, 1500)
    rel = grip.update(d)["right"]
    z_rel = float(d.qpos[cadr + 2])

    return dict(noslip=int(m.opt.noslip_iterations), thumb_on_cube=thumb_on_cube,
                z_start=z_start, contacts_at_placement=contacts_at_placement,
                close=close, lift=lift, rel=rel, z_lift=z_lift, z_rel=z_rel)


class TestGraspScenario:
    """Exit gate: contacted, lifted, released, detected — deterministic."""

    @pytest.fixture(scope="class", params=sorted(_GRASP_CONFIGS))
    def result(self, request):
        r = _run_grasp_scenario(**_GRASP_CONFIGS[request.param])
        # Each configuration really runs the contact model it is named for.
        assert (r["noslip"] == 0) == request.param.startswith("noslip_off")
        return r

    def test_clean_placement_no_penetration(self, result):
        assert result["contacts_at_placement"] == 0

    def test_contacted_and_detected_on_close(self, result):
        assert result["close"].grasping is True
        assert result["close"].grip_force_n > 0.1
        assert "cube" in result["close"].grasped_geoms

    def test_lifted_while_grasped(self, result):
        assert result["lift"].grasping is True
        # Cube rises with the gripper by a clear margin (~13 cm at -0.7).
        assert result["z_lift"] > result["z_start"] + 0.08

    def test_released_and_dropped(self, result):
        assert result["rel"].grasping is False
        assert result["z_rel"] < result["z_lift"] - 0.05

    def test_force_sensor_positive_during_grasp(self, result):
        # r_force_gripper is uid 1.
        assert result["close"].sensor_uid == 1
        assert result["lift"].grip_force_n > 0.0


class TestOriginalFixtureUnderNoslip:
    """The original mu=2.5 fixture under the default (no-slip on): no grasp.

    Recorded, not hidden (ADR-0005).  The closing finger tips the cube on the
    pillar and wedges it, and the thumb never touches it, so the detector
    correctly reports no grasp.  If this starts grasping, the contact model
    changed: revisit the configurations above.
    """

    def test_original_fixture_does_not_grasp_with_noslip_on(self):
        r = _run_grasp_scenario(cube_friction=_ORIGINAL_CUBE_FRICTION)
        assert r["noslip"] > 0
        assert r["close"].grasping is False
        assert r["thumb_on_cube"] is False
