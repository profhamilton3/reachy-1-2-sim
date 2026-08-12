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


def _grasp_scene():
    """Model with the table removed (clear workspace) + support pillar + cube.

    Support: contype=8 conaffinity=4  -> objects rest on it, robot ignores it.
    Cube:    contype=4 conaffinity=15 -> collides with world/robot/object/support.
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
            friction="2.5 0.1 0.01"/>
    </body>
  </worldbody>"""
    m = mujoco.MjModel.from_xml_string(xml.replace("  </worldbody>", extra, 1))
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


class TestGraspScenario:
    """Exit gate: contacted, lifted, released, detected — deterministic."""

    @pytest.fixture(scope="class")
    def result(self):
        m, d = _grasp_scene()
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

        return dict(z_start=z_start, contacts_at_placement=contacts_at_placement,
                    close=close, lift=lift, rel=rel, z_lift=z_lift, z_rel=z_rel)

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
